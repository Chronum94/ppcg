# Implementing algorithm 3 from below as closely as I can
# https://arxiv.org/pdf/1407.7506
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
    else:
        X, _ = np.linalg.qr(X)

    # Strictly speaking, for the first iteration, this shouldn't exist...
    P = np.random.rand(n, k) * 1e-4

    # Also we don't add buffer vectors since I can't get myself to care yet

    traceold = np.inf # np.sum(np.diag(X.T @ A @ X))
    convergence_marker = 1.0
    vals = np.zeros(k)
    lock_idx = []
    active_idx = set(np.arange(k))
    Xlocked = np.array([])
    k_active = len(active_idx)

    # We also don't define splittings, but that may change depending on MPI requirements
    for iconvergence in range(500):
        AX = A @ X
        
        # Even though this is in line 4 of algo 3, having it in here seems to help with convergence a lot.
        W = (AX - X @ (X.T @ AX))

        if T is not None:
            W = T @ W

        # Lines 9 and 10
        W -= X @ (X.T @ W)
        P -= X @ (X.T @ P) if P is not None else P
        if Xlocked.shape[-1] > 0:
            W -= Xlocked @ (Xlocked.T @ W)
            P -= Xlocked @ (Xlocked.T @ P)

        j = 0
        # print("W, P, k_active:", W.shape, P.shape, k_active)
        # Kinda-sorta lines 11 onwards
        while j < k_active:
            # block_slice = slice(j, min(j + blocksize, k))
            block_slice = slice(j, min(j + blocksize, k_active))
            # block_slice = active_block_idx
            # This keeps the block size constant except for the last block
            blocksize = block_slice.stop - block_slice.start

            # Do I even need to allocate these here? Either way, in prep of line 13
            C_X = np.zeros((blocksize, blocksize)) # The alphas
            C_W = np.zeros_like(C_X) # The betas
            C_P = np.zeros_like(C_X) # The gammas

            # In this algorithm listing, it says k smallest eigenvalues, but shouldn't it be blocksize smallest eigenvalues?
            # Since solving the k-large eigenproblem is exactly what we're trying to avoid.
            S = np.column_stack([X[:, block_slice], W[:, block_slice], P[:, block_slice]])

            # We only need the smallest algebraic eigenvector for this
            # But also this is the tiniest ass eigenvalue problem of 3 blocksize x 3 blocksize...
            # This is an EXTREMELY ill-conditioned eigenproblem unless the converged vectors are locked out.
            STS = S.T @ S
            # print(np.linalg.cond(STS))
            _, cmin = eigh(S.T @ A @ S, STS)
            cmin = cmin[:, :blocksize]
            # Ye olde updates
            C_X[:] = cmin[:blocksize, :blocksize]
            C_W[:] = cmin[blocksize:2 * blocksize, :blocksize]
            C_P[:] = cmin[2 * blocksize:3 * blocksize, :blocksize] if iconvergence != 0 else 0.0
            # Line 14-15
            P[:, block_slice] = W[:, block_slice] @ C_W + P[:, block_slice] @ C_P
            X[:, block_slice] = X[:, block_slice] @ C_X + P[:, block_slice]

            j += blocksize
        # Lines 17-19
        if iconvergence % qr_interval == 0:
            X, _ = np.linalg.qr(X)
            # W = (AX - X @ (X.T @ AX))
        # Lines 21 onwards
        if iconvergence % rr_interval == 0:
            # print("Inside RR")
            # Line 22
            if Xlocked.shape[-1] > 0:
                X = np.column_stack([X, Xlocked])
                # print("X shapes", X.shape, Xlocked.shape if Xlocked is not None else None)
            # Line 23-25
            vals, vecs = eigh(X.T @ A @ X)
            # print("RR vals", vals)
            X = X @ vecs
            W = A @ X - X @ np.diag(vals)

            tracenew = np.sum(np.diag(X.T @ A @ X))
            convergence_marker = np.abs(tracenew - traceold) / np.abs(traceold)
            print(iconvergence, convergence_marker)

            # Linw 26-29
            lock_idx = np.linalg.norm(W, axis=0) < 1e-5
            active_idx = np.logical_not(lock_idx)
            lock_idx = np.where(lock_idx)[0]
            active_idx = np.where(active_idx)[0]
            k_active = len(active_idx)
            # print("Idx", lock_idx, active_idx)

            # Convergence checks etc
            if convergence_marker < tol:
                break
            traceold = tracenew
            # Update locked and active arrays
            Xlocked = X[:, lock_idx]
            X = X[:, active_idx]

    vals, vecs = eigh(X.T @ A @ X)
    X = X @ vecs
    return vals, X

np.random.seed(2354)
n = 2000
A = np.random.rand(n, n) * 0.3 + np.diag(np.linspace(2, 20, n))
A += A.T
k = 400
vals, vecs = np.linalg.eigh(A)
# vals, X = ppcg(A, k=k, blocksize=80,)# T=np.diag(1 / vals))
valst, X = ppcg(A, k=k, blocksize=50, T=None, tol=1e-7)
print(np.abs(vals[:k] - valst) < 1e-6)
