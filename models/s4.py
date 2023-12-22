import mlx.core as mx
import mlx.nn as nn
import numpy as np

# contract = torch.einsum
# _conj = lambda x: torch.cat([x, x.conj()], dim=-1)
# _c2r = torch.view_as_real
# _r2c = torch.view_as_complex

"""Helper functions"""


# Returns the complex conjugate, element-wise
def conj(x):
    return mx.concat(x, np.conj(x), dim=-1)


"""Structured matrix kernels"""


# Cauchy Kernel
def cauchy(v, z, w):
    v = mx.concat(v, mx.conj(v), dim=-1)
    return 1 / (v - z * w)


# Log-Vandermonde Kernel
def log_vandermonde(v, x, L):
    pass


if __name__ == "__main__":
    pass
