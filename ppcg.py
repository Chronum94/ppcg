import numpy as np
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh

from line_profiler import profile

@profile
def ppcg(A, k=6, T=None, X=None, blocksize=60, rr_interval=5, qr_interval=5):
    
    if T is None:
        # vals, vecs = np.linalg.eigh(A)
        # T = np.linalg.inv(A)
        # T = np.eye(len(A))#np.diag(np.diag(A))
        T = np.diag(1 / np.diag(A))
    n = len(A)
    if X is None:
        X, _ = np.linalg.qr(np.random.rand(len(A), k))

    # Strictly speaking, for the first iteration, this shouldn't exist...
    P = np.random.rand(n, k) * 1e-8

    C_X = np.zeros((k, k)) # The alphas
    C_W = np.zeros_like(C_X) # The betas
    C_P = np.zeros_like(C_X) # The gammas
    traceold = np.sum(np.diag(X.T @ A @ X))
    for iconvergence in range(500):
        AX = A @ X
        XXT = X @ X.T
        W = T @ (AX - X @ (X.T @ AX))
        W -= X @ (X.T @ W)
        P -= X @ (X.T @ P) if P is not None else P
        j = 0
        while j < k:
            block_slice = slice(j, min(j + blocksize, k))
            # This keeps the block size constant except for the last block
            blocksize = block_slice.stop - block_slice.start

            S = np.column_stack([X[:, block_slice], W[:, block_slice], P[:, block_slice]])
            # We only need the smallest algebraic eigenvector for this
            # But also this is the tiniest ass eigenvalue problem of 3 blocksize x 3 blocksize...
            thetamin, cmin = eigh(S.T @ A @ S, S.T @ S)
            thetamin = thetamin[0]
            cmin = cmin[:, :blocksize]
            # Ye olde updates
            C_X[block_slice, block_slice] = cmin[:blocksize, :blocksize]
            C_W[block_slice, block_slice] = cmin[blocksize:2 * blocksize, :blocksize]
            C_P[block_slice, block_slice] = cmin[2 * blocksize:3 * blocksize, :blocksize] if iconvergence != 0 else 0.0
            P[:, block_slice] = W[:, block_slice] @ C_W[block_slice, block_slice]+ P[:, block_slice] @ C_P[block_slice, block_slice]
            X[:, block_slice] = X[:, block_slice] @ C_X[block_slice, block_slice]+ P[:, block_slice]
            j += blocksize
        # RR step?
        if iconvergence % qr_interval == 0:
            X, _ = np.linalg.qr(X)
        if iconvergence % rr_interval == 0:
            vals, vecs = eigh(X.T @ A @ X)
            X = X @ vecs

            tracenew = np.sum(np.diag(X.T @ A @ X))
            convergence_marker = np.abs(tracenew - traceold) / traceold
            print(iconvergence, convergence_marker)
            if convergence_marker < 1e-8:
                break
            traceold = tracenew

    return vals, X

np.random.seed(2354)
n = 3000
A = np.random.rand(n, n) * 0.1 + np.diag(np.linspace(2, 20, n))
A += A.T
k = 400
print(np.linalg.cond(np.diag(1 / np.diag(A)) @ A))
vals, X = ppcg(A, k=k, blocksize=80)
# valsref, vecsref = np.linalg.eigh(A)
# Almost entire 0s except last few
# print(vals - valsref[:k])
# Almost entirely identity except last few
# print(X.T @ vecsref[:, :k])
