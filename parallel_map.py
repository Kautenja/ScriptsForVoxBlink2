"""A parallel context manager for updating a tqdm progress bar."""
from typing import Callable
import logging
import multiprocessing as mp
import contextlib
import joblib
from tqdm import tqdm


logger = logging.getLogger('sensory.tqdm_extensions.tqdm_parallel')


@contextlib.contextmanager
def tqdm_parallel(progress_bar: tqdm):
    """
    A parallel context manager for updating a tqdm progress bar.

    Args:
        progress_bar: the tqdm progress bar to provide updates to

    Returns:
        a context manager / generator for managing the progress bar

    Examples:
        The intended usage of this module context manager is:

        ```
        from joblib import Parallel, delayed
        import multiprocessing as mp
        with tqdm_parallel(tqdm(total=max(1, mp.cpu_count()))):
            x = Parallel(n_jobs=NUM_JOBS)(list(map(delayed(FUNCTION), DATA)))
        ```

    """
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        """A parallel callback for updating a tqdm progress bar."""

        def __call__(self, *args, **kwargs):
            """Call the callback function and update the global progress bar."""
            progress_bar.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    # cache the old callback function and replace it with the tqdm update callback
    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:  # yield the tqdm object to the generator
        yield progress_bar
    finally:  # upon completion of the context, replace the old callback function
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        progress_bar.close()


def parallel_map(mapper: Callable, data: list,
    total: int=None,
    n_jobs: int=mp.cpu_count(),
    **kwargs,
) -> list:
    """
    Map a function over a dataset using parallel processing.

    Args:
        mapper: The mapping function to apply to the data.
        data: The data to iteratively apply the mapping function to.
        total: The optional override for the total number of items in data. This
            should be used in cases where data is a generator with no __len__
            function definition.
        n_jobs: The number of parallel processes to spawn. Can also be the
            string "cpu_count" to use `max(1, NUM_CORES - 1)` cores.
        kwargs: Any additional key-word arguments to provide to the tqdm
            constructor for the primary workload execution.

    Returns:
        The output of mapping the function over the data. The original order of
        the data will be preserved.

    """
    logger.info('Running parallel map with %d workers', n_jobs)
    # Count the items to process.
    if total is None and hasattr(data, '__len__'):
        total = len(data)
    logger.info('Creating delayed work for %s items', total)
    data = tqdm(data, total=total, desc='Generating delayed work')
    work = list(map(joblib.delayed(mapper), data))
    # Start the parallel workload.
    logger.info('Dispatching parallel workload')
    with tqdm_parallel(tqdm(total=total, **kwargs)):
        return joblib.Parallel(n_jobs=n_jobs)(work)


# Explicitly define the outward facing API of this module
__all_ = [tqdm_parallel.__name__, parallel_map.__name__]
