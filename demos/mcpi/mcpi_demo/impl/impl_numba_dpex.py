import numba
from numba_dpex import dpjit


@dpjit(parallel=True)
def monte_carlo_pi_batch(x, y, batch_size):
    acc = 0.0
    for i in numba.prange(batch_size):
        if x[i] * x[i] + y[i] * y[i] <= 1.0:
            acc += 1.0
    return 4.0 * acc / batch_size
