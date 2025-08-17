import numpy as np
from ppcg.ppcg import ppcg
import pytest


def test_simple():
    np.random.seed(2354)
    n = 100
    A = np.random.randn(n, n) * 0.05 + np.diag(np.linspace(2, 20, n))
    A += A.T
    k = 20
    vals, vecs = np.linalg.eigh(A)
    valst, X = ppcg(A, X=np.random.randn(n, k), blocksize=10, T=None, tol=1e-10, rr_interval=5)
    assert vals[:18] == pytest.approx(valst[:18])