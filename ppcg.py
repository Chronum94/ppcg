# Implement a blocked-simplified version of Algo 2 from
# https://arxiv.org/pdf/1407.7506
# Want to implement algo 3 but head empty
import numpy as np
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt

from line_profiler import profile

@profile
def ppcg(A, k=10, T=None, X=None, blocksize=60, rr_interval=5, qr_interval=5, tol=1e-5):
    """
    """
    n = len(A)
    if X is None:
        X, _ = np.linalg.qr(np.random.rand(len(A), k))

    # Strictly speaking, for the first iteration, this shouldn't exist...
    P = np.random.rand(n, k) * 1e-4

    traceold = np.inf # np.sum(np.diag(X.T @ A @ X))
    convergence_marker = 1.0
    vals = np.zeros(k)
    for iconvergence in range(500):
        AX = A @ X

        W = (AX - X @ (X.T @ AX))
        if T is not None:
            W = T @ W
        W -= X @ (X.T @ W)
        P -= X @ (X.T @ P) if P is not None else P
        j = 0
        while j < k:
            block_slice = slice(j, min(j + blocksize, k))
            # This keeps the block size constant except for the last block
            blocksize = block_slice.stop - block_slice.start
            C_X = np.zeros((blocksize, blocksize)) # The alphas
            C_W = np.zeros_like(C_X) # The betas
            C_P = np.zeros_like(C_X) # The gammas

            S = np.column_stack([X[:, block_slice], W[:, block_slice], P[:, block_slice]])
            # We only need the smallest algebraic eigenvector for this
            # But also this is the tiniest ass eigenvalue problem of 3 blocksize x 3 blocksize...
            # This is an EXTREMELY ill-conditioned eigenproblem unless the converged vectors are locked out.
            STS = S.T @ S
            _, cmin = eigh(S.T @ A @ S, STS)
            cmin = cmin[:, :blocksize]
            # Ye olde updates
            C_X[:] = cmin[:blocksize, :blocksize]
            C_W[:] = cmin[blocksize:2 * blocksize, :blocksize]
            C_P[:] = cmin[2 * blocksize:3 * blocksize, :blocksize] if iconvergence != 0 else 0.0
            P[:, block_slice] = W[:, block_slice] @ C_W + P[:, block_slice] @ C_P
            X[:, block_slice] = X[:, block_slice] @ C_X + P[:, block_slice]
            j += blocksize
        # RR step?
        if iconvergence % qr_interval == 0:
            X, _ = np.linalg.qr(X)
        if iconvergence % rr_interval == 0:
            valsnew, vecs = eigh(X.T @ A @ X)
            X = X @ vecs

            tracenew = np.sum(np.diag(X.T @ A @ X))
            convergence_marker = np.abs(tracenew - traceold) / np.abs(traceold)
            print(iconvergence, convergence_marker, np.linalg.norm((valsnew - vals)) / k)
            vals = valsnew
            if convergence_marker < 1e-5:
                break
            traceold = tracenew

    vals, vecs = eigh(X.T @ A @ X)
    X = X @ vecs
    return vals, X

np.random.seed(2354)
n = 1000
A = np.random.rand(n, n) * 0.3 + np.diag(np.linspace(2, 20, n))
A += A.T
k = 300
# print(np.linalg.cond(np.diag(1 / np.diag(A)) @ A))
vals = np.linalg.eigvalsh(A)
# vals, X = ppcg(A, k=k, blocksize=80,)# T=np.diag(1 / vals))
valst, X = ppcg(A, k=k, blocksize=50, T=None)
# plt.semilogy()
# plt.show()
