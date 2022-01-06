import numpy as np
import os

from .iterators import EpochBatchIterator, ShardedIterator, BufferedIterator, CountingIterator
import torch
from torch.utils.data import DataLoader
from fairseq.data import data_utils
from contextlib import contextmanager
import time
import pdb


from adaptdl.torch.data import AdaptiveDataLoaderMixin
from adaptdl.torch._metrics import (
    profile_step_start, profile_step_commit,
    set_batch_size, get_goodput_fn, get_progress, _metrics_state)

class ProfilingIterator(CountingIterator):
    def __init__(self, iterable, max_tokens, start=None, total=None):
        assert max_tokens is not None
        self.start = start
        self.max_tokens = max_tokens
        super().__init__(iterable, start, total)

    def __next__(self):
        if self.n > self.start:
            profile_step_commit(False)
        batch = super().__next__()
        profile_step_start(self.max_tokens)
        return batch    

class GlobalCountingIterator(CountingIterator):
    def __init__(self, iterable, num_replicas=1, start=None, total=None):
        self.num_replicas = num_replicas
        super().__init__(iterable, start, len(iterable) * self.num_replicas)

    def __next__(self):
        if not self.has_next():
            raise StopIteration
        try:
            x = next(self._itr)
        except StopIteration:
            raise IndexError(
                f"Iterator expected to have length {self.total}, "
                "but exhausted at position {self.n}."
            )
        self.n += self.num_replicas
        return x

class AdaptiveDataLoader(DataLoader, AdaptiveDataLoaderMixin):
    def __init__(self, dataset, max_tokens, shuffle=False, **kwargs):
        super().__init__(dataset, **kwargs)
        AdaptiveDataLoaderMixin.__init__(self, batch_size=max_tokens)
    
    def __iter__(self):
        epoch = 0
        num_replicas = 1
        with self._elastic.context():
            done = False
            while not done:
                max_tokens = self._elastic._sync_local_bsz()
                print(max_tokens)
                for idx, batch in enumerate(super().__iter__()):
                    with self._elastic.profile(self.training and idx >= 1):
                        yield batch
                        if self._elastic.max_batch_size is not None and \
                                get_progress() >= len(self.dataset) * \
                                (epoch + 1) / self.batch_size:
                            done = True
                            break
                if self._elastic.max_batch_size is None:
                    done = True
                self._elastic.current_index -= \
                    self._elastic.current_index % -len(self.dataset)


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
        itr = AdaptiveDataLoader(
            self.dataset,
            self.max_tokens * self.num_workers,  # batch_size proxy
            collate_fn=self.collate_fn,
            batch_sampler=batches,
            num_workers=self.num_workers,
            timeout=self.timeout,
            pin_memory=True,
        )

        # Wrap with a BufferedIterator if needed
        if self.buffer_size > 0:
            itr = BufferedIterator(self.buffer_size, itr)

        itr = GlobalCountingIterator(itr, self.num_shards, offset)

        if self.skip_remainder_batch:
            # TODO: Below is a lazy implementation which discard the final batch regardless
            # of whether it is a full batch or not.
            total_num_itrs = len(batches) - 1
            itr.take(total_num_itrs)
            logger.info(f"skip final residual batch, total_num_itrs = {total_num_itrs}")

        return itr
    
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
