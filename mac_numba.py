"""
Numba-jitted (@njit nopython=True) versions of all non-neural MI estimators,
suffixed _numba. Mathematically identical to the originals; designed for parallel
per-sample work via prange. Grid construction and channel sampling stay in numpy
(cheap, exact); the heavy O(T x Q^K) inner loops run in fused, allocation-free
kernels. Complex distances are computed via the real/imag components directly
(no complex temporaries inside the kernel).

Closed-form Gaussian quantities (joint, per-user ceiling, linear) are not jitted:
they are dominated by LAPACK calls (eigvalsh, inv) already vectorised through
Accelerate BLAS, and numba JIT overhead would be a net loss. They are re-exported
here for a uniform interface.
"""

import numpy as np
from numba import njit, prange
import mac

LOG2E = float(np.log2(np.e))


# --------------------------------------------------------------------------- #
# Joint mutual information I(x;y) (discrete)
# --------------------------------------------------------------------------- #
@njit(cache=True, parallel=True, fastmath=True)
def _joint_kernel(yr, yi, Hr, Hi, Hg_sq, ysq):
    T = yr.shape[0]; M = Hr.shape[0]; N = yr.shape[1]
    lse = np.empty(T)
    for t in prange(T):
        # pass 1: max over m of d_m = -|y_t|^2 + 2 Re(y_t . conj(Hg_m)) - |Hg_m|^2
        maxd = -1.0e300
        for m in range(M):
            dot = 0.0
            for n in range(N):
                dot += yr[t, n] * Hr[m, n] + yi[t, n] * Hi[m, n]
            d = -ysq[t] + 2.0 * dot - Hg_sq[m]
            if d > maxd:
                maxd = d
        # pass 2: sum exp(d - maxd)
        s = 0.0
        for m in range(M):
            dot = 0.0
            for n in range(N):
                dot += yr[t, n] * Hr[m, n] + yi[t, n] * Hi[m, n]
            d = -ysq[t] + 2.0 * dot - Hg_sq[m]
            s += np.exp(d - maxd)
        lse[t] = maxd + np.log(s)
    return lse


def joint_MI_numba(H, n_sym, rng, const=mac.QPSK):
    K = H.shape[1]
    grid, _ = mac._cached_grid(K, const)
    Hg = grid @ H.T
    Hg_sq = np.sum(np.abs(Hg) ** 2, axis=1)
    s, y = mac._sample_channel(H, n_sym, rng, const)
    d_true = -np.sum(np.abs(y - s @ H.T) ** 2, axis=1)
    ysq = np.sum(np.abs(y) ** 2, axis=1)
    lse = _joint_kernel(
        np.ascontiguousarray(y.real), np.ascontiguousarray(y.imag),
        np.ascontiguousarray(Hg.real), np.ascontiguousarray(Hg.imag),
        Hg_sq, ysq,
    )
    out = (d_true - lse + np.log(grid.shape[0])) * LOG2E
    return float(np.mean(out))


# --------------------------------------------------------------------------- #
# Per-user ceiling sum_k I(x_k;y) (discrete)
# --------------------------------------------------------------------------- #
@njit(cache=True, parallel=True, fastmath=True)
def _ceiling_kernel(yr, yi, Hr, Hi, Hg_sq, ysq, K, Q):
    T = yr.shape[0]; M = Hr.shape[0]; N = yr.shape[1]
    Hcond = np.zeros((T, K))
    Q_pow = np.empty(K, dtype=np.int64)
    Q_pow[0] = 1
    for k in range(1, K):
        Q_pow[k] = Q_pow[k - 1] * Q
    for t in prange(T):
        # pass 1: max
        maxd = -1.0e300
        for m in range(M):
            dot = 0.0
            for n in range(N):
                dot += yr[t, n] * Hr[m, n] + yi[t, n] * Hi[m, n]
            d = -ysq[t] + 2.0 * dot - Hg_sq[m]
            if d > maxd:
                maxd = d
        # pass 2: accumulate per-user per-symbol sums
        sum_ka = np.zeros((K, Q))
        for m in range(M):
            dot = 0.0
            for n in range(N):
                dot += yr[t, n] * Hr[m, n] + yi[t, n] * Hi[m, n]
            d = -ysq[t] + 2.0 * dot - Hg_sq[m]
            ed = np.exp(d - maxd)
            for k in range(K):
                digit = (m // Q_pow[k]) % Q
                sum_ka[k, digit] += ed
        # total (same for every k, take k=0)
        total = 0.0
        for a in range(Q):
            total += sum_ka[0, a]
        # per-user conditional entropy
        for k in range(K):
            ht = 0.0
            for a in range(Q):
                P = sum_ka[k, a] / total
                if P > 1.0e-300:
                    ht -= P * np.log2(P)
            Hcond[t, k] = ht
    return Hcond


def peruser_ceiling_MI_numba(H, n_sym, rng, const=mac.QPSK):
    K = H.shape[1]; Q = len(const)
    grid, _ = mac._cached_grid(K, const)
    Hg = grid @ H.T
    Hg_sq = np.sum(np.abs(Hg) ** 2, axis=1)
    s, y = mac._sample_channel(H, n_sym, rng, const)
    ysq = np.sum(np.abs(y) ** 2, axis=1)
    Hc = _ceiling_kernel(
        np.ascontiguousarray(y.real), np.ascontiguousarray(y.imag),
        np.ascontiguousarray(Hg.real), np.ascontiguousarray(Hg.imag),
        Hg_sq, ysq, K, Q,
    )
    return float(np.sum(np.log2(Q) - Hc.mean(axis=0)))


# --------------------------------------------------------------------------- #
# Linear per-user rate sum_k I(x_k;(Wy)_k) (discrete)
# Implemented for a single user at a time; the wrapper loops K users.
# --------------------------------------------------------------------------- #
@njit(cache=True, parallel=True, fastmath=True)
def _linear_oneuser_kernel(rr, ri, mr, mi, vk, sidx_k, K, Q, k0):
    """Returns mean over T of (log_cond_true - log_marg) * LOG2E for user index k0."""
    T = rr.shape[0]; M = mr.shape[0]
    Q_pow_k0 = 1
    for _ in range(k0):
        Q_pow_k0 *= Q
    log_M = np.log(M); log_M_Q = np.log(M // Q)
    acc = np.zeros(T)
    for t in prange(T):
        # pass 1: max
        maxd = -1.0e300
        for m in range(M):
            dr = rr[t] - mr[m]; di = ri[t] - mi[m]
            d = -(dr * dr + di * di) / vk
            if d > maxd:
                maxd = d
        # pass 2: per-a sums
        sum_a = np.zeros(Q)
        for m in range(M):
            dr = rr[t] - mr[m]; di = ri[t] - mi[m]
            d = -(dr * dr + di * di) / vk
            ed = np.exp(d - maxd)
            digit = (m // Q_pow_k0) % Q
            sum_a[digit] += ed
        total = 0.0
        for a in range(Q):
            total += sum_a[a]
        log_marg = maxd + np.log(total) - log_M
        log_cond = maxd + np.log(sum_a[sidx_k[t]]) - log_M_Q
        acc[t] = (log_cond - log_marg) * LOG2E
    return acc


def linear_peruser_MI_numba(H, W, n_sym, rng, const=mac.QPSK):
    K = H.shape[1]; Q = len(const)
    _, _ = mac._cached_grid(K, const)
    grid, _ = mac._cached_grid(K, const)
    means_grid = grid @ (W @ H).T
    v = np.real(np.einsum('ij,ij->i', W, W.conj()))
    s, y = mac._sample_channel(H, n_sym, rng, const)
    r = y @ W.T
    sidx = mac._to_index(s, const)
    total = 0.0
    for k in range(K):
        acc = _linear_oneuser_kernel(
            np.ascontiguousarray(r[:, k].real), np.ascontiguousarray(r[:, k].imag),
            np.ascontiguousarray(means_grid[:, k].real), np.ascontiguousarray(means_grid[:, k].imag),
            float(v[k]), np.ascontiguousarray(sidx[:, k]), K, Q, k,
        )
        total += float(np.mean(acc))
    return total


def linear_oneuser_MI_numba(H, W, k0, n_sym, rng, const=mac.QPSK):
    K = H.shape[1]; Q = len(const)
    grid, _ = mac._cached_grid(K, const)
    means_grid = grid @ (W @ H).T
    vk = float(np.real(np.dot(W[k0], W[k0].conj())))
    s, y = mac._sample_channel(H, n_sym, rng, const)
    r = y @ W.T
    sidx = mac._to_index(s, const)
    acc = _linear_oneuser_kernel(
        np.ascontiguousarray(r[:, k0].real), np.ascontiguousarray(r[:, k0].imag),
        np.ascontiguousarray(means_grid[:, k0].real), np.ascontiguousarray(means_grid[:, k0].imag),
        vk, np.ascontiguousarray(sidx[:, k0]), K, Q, k0,
    )
    return float(np.mean(acc))


def lmmse_sic_MI_numba(H, n_sym, rng, const=mac.QPSK, order=None):
    K = H.shape[1]
    if order is None:
        order = list(range(K))
    total = 0.0
    for idx in range(K):
        rem = order[idx:]
        Hr = H[:, rem]
        total += linear_oneuser_MI_numba(Hr, mac.lmmse(Hr), 0, n_sym, rng, const)
    return total


# --------------------------------------------------------------------------- #
# Gaussian closed-forms: re-export the numpy versions (LAPACK / Accelerate already
# optimal; JIT would add overhead for sub-ms calls).
# --------------------------------------------------------------------------- #
joint_gaussian_MI_numba = mac.joint_gaussian_MI
peruser_ceiling_gaussian_MI_numba = mac.peruser_ceiling_gaussian_MI
linear_gaussian_MI_numba = mac.linear_gaussian_MI
