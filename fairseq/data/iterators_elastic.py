import numpy as np
import os

from .iterators import EpochBatchIterator, ShardedIterator, BufferedIterator, CountingIterator
import torch
from torch.utils.data import DataLoader
from fairseq.data import data_utils
from contextlib import contextmanager
import time
import pdb


from adaptdl.torch._metrics import (
    profile_step_start, profile_step_commit,
    set_batch_size, get_goodput_fn, get_progress, _metrics_state)


class ProfilingIterator(CountingIterator):
    def __init__(self, iterable, start=None, total=None):
        self.start = start
        super().__init__(iterable, start, total)

    def __next__(self):
        if self.n > self.start:
            profile_step_commit(False)
        batch = super().__next__()
        profile_step_start(32)
        return batch    


class ElasticEpochBatchIterator(EpochBatchIterator):
    def __init__(
        self,
        dataset,
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
                ShardedIterator(batches, self.num_shards, self.shard_id, fill_value=[])
            )
            self.dataset.prefetch([i for s in batches for i in s])

            if shuffle and fix_batches_to_gpus:
                batches = shuffle_batches(batches, self.seed + epoch + self.shard_id)
        else:
            if shuffle:
                batches = shuffle_batches(list(self.frozen_batches), self.seed + epoch)
            else:
                batches = self.frozen_batches
            batches = list(
                ShardedIterator(batches, self.num_shards, self.shard_id, fill_value=[])
            )

        if offset > 0 and offset >= len(batches):
            return None

        if self.num_workers > 0:
            os.environ["PYTHONWARNINGS"] = "ignore:semaphore_tracker:UserWarning"

        # Create data loader
        itr = torch.utils.data.DataLoader(
            self.dataset,
            collate_fn=self.collate_fn,
            batch_sampler=batches[offset:],
            num_workers=self.num_workers,
            timeout=self.timeout,
            pin_memory=True,
        )

        # Wrap with a BufferedIterator if needed
        if self.buffer_size > 0:
            itr = BufferedIterator(self.buffer_size, itr)

        # Wrap with CountingIterator
        itr = CountingIterator(itr, start=offset)

        if self.skip_remainder_batch:
            # TODO: Below is a lazy implementation which discard the final batch regardless
            # of whether it is a full batch or not.
            total_num_itrs = len(batches) - 1
            itr.take(total_num_itrs)
            logger.info(f"skip final residual batch, total_num_itrs = {total_num_itrs}")

        itr = ProfilingIterator(itr, offset)

        return itr
