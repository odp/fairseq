import numpy as np
import os

from .iterators import EpochBatchIterator, ShardedIterator, BufferedIterator, CountingIterator
import torch
from torch.utils.data import DataLoader
from fairseq.data import data_utils
from contextlib import contextmanager
import time
from adaptdl.torch.data import AdaptiveDataLoaderHelper, current_dataloader, _AdaptiveDataLoaderState

from icecream import ic
import pdb


from adaptdl.torch.data import AdaptiveDataLoaderMixin
from adaptdl.torch._metrics import (
    profile_step_start, profile_step_commit,
    set_batch_size, get_goodput_fn, get_progress, _metrics_state)

import logging
logger = logging.getLogger(__name__)

def gen():
    i = 0
    while True:
        yield max(i, 0)
        i += 1
G = gen()

def get_progress_test():
    return next(G)

_ELASTIC_STATE = None

def elastic_state():
    global _ELASTIC_STATE
    if _ELASTIC_STATE is None:
        _ELASTIC_STATE = _AdaptiveDataLoaderState()
    return _ELASTIC_STATE
        
class AdaptiveIterator(CountingIterator):
    def __init__(self, iterable, epoch=0, num_replicas=1, start=0, max_tokens=0):
        self.iterable = iterable
        self.num_replicas = num_replicas
        self.start = start
        self.epoch = epoch
        self.max_tokens = max_tokens
        self.elastic = AdaptiveDataLoaderHelper(self.max_tokens)
        assert self.elastic._accum_count == 0
        # ALL dataloaders MUST share same self.elastic._state
        self.elastic._state = elastic_state()
        super().__init__(self.iterable, 0, len(self.iterable) * self.num_replicas)
        AdaptiveDataLoaderHelper._current = self.elastic
        
    @property
    def done(self):
        if self.elastic.max_batch_size is None:
            return True
        if get_progress() >= (len(self.iterable.batch_sampler) * self.epoch * self.num_replicas):
            self.n = self.total
            return True
        else:
            return False

    def __next__(self):
        if self.elastic.training and self.n == 0: # before first batch
            batch_size = self.elastic._sync_local_bsz()
        if not self.has_next():
            raise StopIteration  # Mostly for Valid iterators

        if self.elastic.training and self.n >= self.num_replicas:
            profile_step_commit(self.elastic.is_accum_step())

        if self.elastic.training:
            self.elastic._accum_count = (0 if self.elastic.is_optim_step()
                                         else self.elastic._accum_count + 1)
        x = next(self._itr)
        self.n += self.num_replicas
        if not self.done and not self.has_next(): # iter exhausted
            # Replay the dataloader
            logger.info(f"Replaying DataLoader at progress: {get_progress()}")
            super().__init__(self.iterable, 0, len(self.iterable) * self.num_replicas)
            AdaptiveDataLoaderHelper._current = self.elastic
            AdaptiveDataLoaderHelper._training = self.elastic
        profile_step_start(self.max_tokens)
        return x

    def __len__(self):
        return len(self.iterable.batch_sampler)

    def has_next(self):
        """Whether the iterator has been exhausted."""
        if self.n < self.total:
            return True
        else:
            self.elastic._state.current_index = 0
            self.elastic._state.end_index = 0
            AdaptiveDataLoaderHelper._current = None
            AdaptiveDataLoaderHelper._training = None
            return False

class ElasticEpochBatchIterator(EpochBatchIterator):
    def __init__(
        self,
        dataset,
        max_tokens,
        collate_fn,
        batch_sampler,
        seed=1,
        num_shards=1,
        shard_id=0,
        num_workers=0,
        epoch=1,
        buffer_size=0,
        timeout=0,
        disable_shuffling=False,
        skip_remainder_batch=False,
        grouped_shuffling=False,
    ):
        self.max_tokens = max_tokens
        super().__init__(dataset, collate_fn, batch_sampler, seed, num_shards,
                shard_id, num_workers, epoch, buffer_size, timeout,
                disable_shuffling, skip_remainder_batch, grouped_shuffling)


    def _get_iterator_for_epoch(
        self, epoch, shuffle, fix_batches_to_gpus=False, offset=0
    ):
        def shuffle_batches(batches, seed):
            with data_utils.numpy_seed(seed):

                if self.grouped_shuffling:
                    grouped_batches = [
                        batches[(i * self.num_shards) : ((i + 1) * self.num_shards)]
                        for i in range((len(batches) // self.num_shards))
                    ]
                    np.random.shuffle(grouped_batches)
                    batches = list(itertools.chain(*grouped_batches))
                else:
                    np.random.shuffle(batches)

            return batches

        if self._supports_prefetch:
            batches = self.frozen_batches

            if shuffle and not fix_batches_to_gpus:
                batches = shuffle_batches(list(batches), self.seed + epoch)

            batches = list(
                ShardedIterator(batches[offset:], self.num_shards, self.shard_id, fill_value=[])
            )
            self.dataset.prefetch([i for s in batches for i in s])

            if shuffle and fix_batches_to_gpus:
                batches = shuffle_batches(batches, self.seed + epoch + self.shard_id)
        else:
            if shuffle:
                batches = shuffle_batches(list(self.frozen_batches), self.seed + epoch)
            else:
                batches = self.frozen_batches

            batches = list(ShardedIterator(batches[offset:], self.num_shards, self.shard_id, fill_value=[]))

        if offset > 0 and offset >= len(batches):
            return None

        if self.num_workers > 0:
            os.environ["PYTHONWARNINGS"] = "ignore:semaphore_tracker:UserWarning"

        # Create data loader
        itr = torch.utils.data.DataLoader(
            self.dataset,
            collate_fn=self.collate_fn,
            batch_sampler=batches,
            num_workers=self.num_workers,
            timeout=self.timeout,
            pin_memory=True,
            prefetch_factor=self.buffer_size,
            persistent_workers=True
        )
        
        itr = AdaptiveIterator(itr, epoch, self.num_shards, offset, self.max_tokens)

        if self.skip_remainder_batch:
            # TODO: Below is a lazy implementation which discard the final batch regardless
            # of whether it is a full batch or not.
            total_num_itrs = len(batches) - 1
            itr.take(total_num_itrs)
            logger.info(f"skip final residual batch, total_num_itrs = {total_num_itrs}")

        return itr
    
    def __len__(self):
        return int(math.ceil(len(self.frozen_batches) / float(self.num_shards)))

    def end_of_epoch(self) -> bool:
        """Returns whether the most recent epoch iterator has been exhausted"""
        return not self._cur_epoch_itr.has_next()

    def state_dict(self):
        """Returns a dictionary containing a whole state of the iterator."""
        if self.end_of_epoch():
            epoch = self.epoch + 1
            iter_in_epoch = 0
        else:
            epoch = self.epoch
            iter_in_epoch = self.iterations_in_epoch
        return {
            "version": 2,
            "epoch": epoch,
            "iterations_in_epoch": iter_in_epoch,
            "shuffle": self.shuffle,
        }

    def load_state_dict(self, state_dict):
        """Copies the state of the iterator from the given *state_dict*."""
        self.epoch = state_dict["epoch"]
        itr_pos = state_dict.get("iterations_in_epoch", 0)
        version = state_dict.get("version", 1)
        if itr_pos > 0:
            # fast-forward epoch iterator
            self._next_epoch_itr = self._get_iterator_for_epoch(
                self.epoch,
                shuffle=state_dict.get("shuffle", True),
                offset=itr_pos,
            )
            if self._next_epoch_itr is None:
                if version == 1:
                    # legacy behavior: we finished the epoch, increment epoch counter
                    self.epoch += 1
                else:
                    raise RuntimeError(
                        "Cannot resume training due to dataloader mismatch, please "
                        "report this to the fairseq developers. You can relaunch "
                        "training with `--reset-dataloader` and it should work."
                    )
        else:
            self._next_epoch_itr = None
