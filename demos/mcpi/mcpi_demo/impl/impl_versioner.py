import dpnp
import numpy as np
from mcpi_demo.impl.arg_parser import parse_args

RUN_VERSION = parse_args().variant

if RUN_VERSION == "Numba".casefold():
    from mcpi_demo.impl.impl_numba import monte_carlo_pi_batch
elif RUN_VERSION == "NumPy".casefold():
    from mcpi_demo.impl.impl_numpy import monte_carlo_pi_batch
elif RUN_VERSION == "DPNP".casefold():
    from mcpi_demo.impl.impl_dpnp import monte_carlo_pi_batch
elif RUN_VERSION == "Numba-DPEX".casefold():
    from mcpi_demo.impl.impl_numba_dpex import monte_carlo_pi_batch


def monte_carlo_pi(batch_size, n_batches):
    s = dpnp.empty(n_batches)
    if RUN_VERSION == "Numba-DPEX".casefold():
        for i in range(n_batches):
            print(f"Batch #{i}")
            a = dpnp.random.random(size=batch_size)
            b = dpnp.random.random(size=batch_size)
            s[i] += monte_carlo_pi_batch(a, b, batch_size)
    else:
        s = np.empty(n_batches)
        for i in range(n_batches):
            print(f"Batch #{i}")
            s[i] = monte_carlo_pi_batch(batch_size)
    return s.mean(), s.std()
