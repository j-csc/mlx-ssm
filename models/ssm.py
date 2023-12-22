# Intuition for SSMs - annotated-s4
from functools import partial
import mlx.core as mx
import mlx.nn as nn
import numpy as np


""" Background Section """


def random_ssm(rng, N):
    a_r, b_r, c_r = mx.random.split(rng, 3)
    A = mx.random.uniform(a_r, (N, N))
    B = mx.random.uniform(b_r, (N, 1))
    C = mx.random.uniform(c_r, (1, N))
    return A, B, C


def discretize(A, B, C, step):
    I = mx.eye(A.shape[0])
    BL = inv(I - (step / 2.0) * A)
    Ab = BL @ (I + (step / 2.0) * A)
    Bb = (BL * step) @ B
    return Ab, Bb, C


if __name__ == "__main__":
    pass
