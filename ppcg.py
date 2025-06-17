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
        X, _ = np.linalg.qr(np.random.rand(n, k))
    else:
        X, _ = np.linalg.qr(X)

    W = (A @ X - X @ (X.T @ A @ X))
    # Strictly speaking, for the first iteration, this shouldn't exist...
    P = np.random.rand(n, k) * 1e-8

    # Also we don't add buffer vectors since I can't get myself to care yet

    traceold = np.inf # np.sum(np.diag(X.T @ A @ X))
    convergence_marker = 1.0
    vals = np.zeros(k)
    active_idx = np.arange(k)
    # This serves as a buffer workspace for locked/unlocked updates
    work_XWP = np.empty((n, 3 * k))
    # Plocked = np.array([])

    # We also don't define splittings, but that may change depending on MPI requirements
    for iconvergence in range(500):
        # Dims: Nkactive = NN @ Nkactive
        AX = A @ X
        
        # Even though this is in line 4 of algo 3, having it in here seems to help with convergence a lot.
        # X.T @ AX dims = kactiveN @ Nkactive
        # W dims = Nkactive - Nkactive @ kactivekactive = Nkactive
        W[:] = (AX - X @ (X.T @ AX))

        if T is not None:
            W[:] = T @ W

        # Lines 9 and 10
        # Nk @ kN @ Nk = Nk
        W -= X @ (X.T @ W)
        P -= X @ (X.T @ P) if P is not None else P

        n_active_vectors = len(active_idx)
        # print("Active vectors:", n_active_vectors)

        # print("W, P, k_active:", W.shape, P.shape, k_active)
        # Kinda-sorta lines 11 onwards
        for j in range(0, n_active_vectors, blocksize):
            block_slice = slice(j, min(j + blocksize, n_active_vectors))
            # This keeps the block size constant except for the last block
            blocksize = block_slice.stop - block_slice.start
            block_slice = active_idx[block_slice]

            # Do I even need to allocate these here? Either way, in prep of line 13
            C_X = np.zeros((blocksize, blocksize)) # The alphas
            C_W = np.zeros_like(C_X) # The betas
            C_P = np.zeros_like(C_X) # The gammas

            work_XWP[:, :blocksize] = X[:, block_slice]
            work_XWP[:, blocksize:2 * blocksize] = W[:, block_slice]
            if j > 0:
                nblocksizes = 3 * blocksize
                work_XWP[:, 2*blocksize:3 * blocksize] = P[:, block_slice]
            else:
                nblocksizes = 2 * blocksize

            # In this algorithm listing, it says k smallest eigenvalues, but shouldn't it be blocksize smallest eigenvalues?
            # Since solving the k-large eigenproblem is exactly what we're trying to avoid.
            S = work_XWP[:, :nblocksizes]
            # print("S shape", S.shape)

            # We only need the smallest algebraic eigenvector for this
            # But also this is the tiniest ass eigenvalue problem of 3 blocksize x 3 blocksize...
            # This is an EXTREMELY ill-conditioned eigenproblem unless the converged vectors are locked out.
            STS = S.T @ S
            # print(np.linalg.cond(STS))
            try:
                _, cmin = eigh(S.T @ A @ S, STS)
            except Exception as e:
                j += blocksize
                continue
            cmin = cmin[:, :blocksize]
            # Ye olde updates
            C_X[:] = cmin[:blocksize, :blocksize]
            C_W[:] = cmin[blocksize:2 * blocksize, :blocksize]
            if nblocksizes == 3 * blocksize:
                C_P[:] = cmin[2 * blocksize:3 * blocksize, :blocksize] if iconvergence != 0 else 0.0
                # Line 14-15
                # P = W C_W + P C_P
                work_XWP[:, 2 * blocksize:3 * blocksize] = work_XWP[:, blocksize:2 * blocksize] @ C_W + work_XWP[:, 2 * blocksize:3 * blocksize] @ C_P
                # X = X C_X + P
                work_XWP[:, :blocksize] = work_XWP[:, :blocksize] @ C_X + work_XWP[:, 2 * blocksize:3 * blocksize]

            if nblocksizes == 2 * blocksize:
                # X = X C_X + W C_W
                work_XWP[:, :blocksize] = work_XWP[:, :blocksize] @ C_X + work_XWP[:, blocksize:2 * blocksize] @ C_W

            X[:, block_slice] = work_XWP[:, :blocksize]
            W[:, block_slice] = work_XWP[:, blocksize:2 * blocksize]
            if nblocksizes == 3 * blocksize:
                P[:, block_slice] = work_XWP[:, 2 * blocksize:3 * blocksize]

            j += blocksize
        # Lines 17-19
        # Lines 21 onwards
        if iconvergence % rr_interval == 0:
            X, _ = np.linalg.qr(X)
            # Line 23-25
            # plt.matshow(1 + X.T @ X, norm='log')
            # plt.colorbar()
            # plt.show()
            vals, vecs = eigh(X.T @ A @ X)
            # print(vecs)
            # print("RR vals", vals)
            X = X @ vecs
            W = A @ X - X * vals[np.newaxis, :]

            tracenew = np.sum(np.diag(X.T @ A @ X))
            convergence_marker = np.abs(tracenew - traceold) / np.abs(traceold)
            print(iconvergence, convergence_marker)

            # Linw 26-29
            active_idx = np.where(np.linalg.norm(W, axis=0) > tol * np.sqrt(n))[0]
            n_active_vectors = len(active_idx)

            # Convergence checks etc
            if convergence_marker < tol:
                break
            traceold = tracenew

    return vals, X

np.random.seed(2354)
n = 1000
A = np.random.randn(n, n) * 0.05 + np.diag(np.linspace(2, 20, n))
A += A.T
k = 300
vals, vecs = np.linalg.eigh(A)
valst, X = ppcg(A, k=k, blocksize=50, T=None, tol=1e-14, rr_interval=5)
print(np.max(np.abs(valst - vals[:k])))
