"""
Multiple-access channel: y = sqrt(Es) * S @ s + n

  s : K x 1 unit-power discrete symbols (QPSK), E[|s_k|^2] = 1
  S : N x K spreading matrix (columns ~ unit norm)
  n : N x 1 complex AWGN, E[|n_i|^2] = 1  (so Cov(n) = I_N)
  Es: per-user symbol energy = Var(x_k), folded into the channel as H = sqrt(Es) S

Fixed-Eb framework
------------------
beta = K / N (load).  Spectral efficiency  R = I(x;y) / N   [bits / complex chip].
Energy-per-bit relation (user's convention):   Es = R * Eb / beta.
Equivalently Eb = K*Es / I(x;y) = (total energy per use) / (total bits per use).
For each scheme R(Es) differs, so each sits at its own operating Es for a common Eb.

Rates (all in bits, summed over users unless noted)
---------------------------------------------------
  I_opt(Es)        = I(x;y)              exact joint MI (Monte-Carlo over noise)
  I_peruser(Es)    = sum_k I(x_k;y)      independent-decoding CEILING (exact marginal APP)
  I_linear(Es,W)   = sum_k I(x_k;(Wy)_k) linear detector + independent decoding (exact, discrete)

Win A claim:  I_linear  <  I_peruser  <=  I_opt   (the first strict whenever interference remains).
A neural detector approximating the per-user posterior P(x_k|y) recovers I_peruser at scale.
"""

import numpy as np

LOG2E = np.log2(np.e)

# unit-power constellations, E[|a|^2] = 1
QPSK = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j], dtype=np.complex128) / np.sqrt(2.0)

_lv = np.array([-3, -1, 1, 3], dtype=np.float64)
QAM16 = np.array([re + 1j * im for re in _lv for im in _lv], dtype=np.complex128) / np.sqrt(10.0)

CONSTELLATIONS = {"QPSK": QPSK, "QAM16": QAM16}


# --------------------------------------------------------------------------- #
# Spreading matrices
# --------------------------------------------------------------------------- #
def random_iid_S(N, K, rng):
    """i.i.d. complex Gaussian columns, normalized to unit column norm."""
    S = (rng.standard_normal((N, K)) + 1j * rng.standard_normal((N, K))) / np.sqrt(2.0)
    S /= np.linalg.norm(S, axis=0, keepdims=True)
    return S


def correlated_S(N, K, rho, rng):
    """
    Correlated columns: each signature is a shared common component (weight rho)
    plus an independent part. Larger rho -> harder for linear detectors.
    """
    common = (rng.standard_normal((N, 1)) + 1j * rng.standard_normal((N, 1))) / np.sqrt(2.0)
    indep = (rng.standard_normal((N, K)) + 1j * rng.standard_normal((N, K))) / np.sqrt(2.0)
    S = np.sqrt(rho) * common + np.sqrt(1.0 - rho) * indep
    S /= np.linalg.norm(S, axis=0, keepdims=True)
    return S


# --------------------------------------------------------------------------- #
# Alphabet enumeration  (all 4^K symbol vectors)  -- only for exact references
# --------------------------------------------------------------------------- #
def all_symbol_vectors(K, const=QPSK):
    """Return (M, K) complex array of all M = Q^K constellation vectors (Q=|const|)."""
    Q = len(const)
    M = Q ** K
    idx = np.arange(M)
    digits = np.zeros((M, K), dtype=np.int64)
    for k in range(K):
        digits[:, k] = (idx // (Q ** k)) % Q
    return const[digits]  # (M, K)


def _to_index(s, const):
    """Map symbols (T,K) to class indices 0..Q-1 by nearest constellation point."""
    d = np.abs(s[..., None] - const[None, None, :])     # (T,K,Q)
    return np.argmin(d, axis=-1)


# Grid cache: all_symbol_vectors / grid_idx depend only on (K, constellation), not on
# Es or H, so caching across the ~10 fixed-point evaluations is exact (accuracy-neutral).
_GRID_CACHE = {}


def _cached_grid(K, const):
    key = (K, id(const))
    if key not in _GRID_CACHE:
        grid = all_symbol_vectors(K, const)
        grid_idx = _to_index(grid[None], const)[0]
        _GRID_CACHE[key] = (grid, grid_idx)
    return _GRID_CACHE[key]


# --------------------------------------------------------------------------- #
# Exact mutual-information estimators (Monte-Carlo over data + noise)
# --------------------------------------------------------------------------- #
def _neg_dist2(y, Hg, Hg_sqnorm):
    """-||y - Hg||^2 as (T,M) via matmul, no 3-D broadcast.
       -||y-Hg||^2 = -||y||^2 + 2 Re(y Hg^H) - ||Hg||^2."""
    ysq = np.sum(np.abs(y) ** 2, axis=1, keepdims=True)          # (T,1)
    cross = 2.0 * np.real(y @ Hg.conj().T)                       # (T,M)
    return -ysq + cross - Hg_sqnorm[None, :]


def _sample_channel(H, n_sym, rng, const=QPSK):
    """Draw n_sym transmissions. Returns s (T,K) symbols, y (T,N) received."""
    K = H.shape[1]
    T = n_sym
    sidx = rng.integers(0, len(const), size=(T, K))
    s = const[sidx]                    # (T,K)
    x = s @ H.T                         # (T,N)  == (H @ s_t) stacked
    noise = (rng.standard_normal((T, H.shape[0])) + 1j * rng.standard_normal((T, H.shape[0]))) / np.sqrt(2.0)
    y = x + noise
    return s, y


def joint_MI(H, n_sym, rng, const=QPSK, chunk=1000):
    """
    I(x;y) in bits per channel use:  I = E[ log2( p(y|x) / mean_{x'} p(y|x') ) ].
    p(y|x) ∝ exp(-||y - H x||^2)  (complex, unit noise var per dim).
    """
    K = H.shape[1]
    grid = all_symbol_vectors(K, const)  # (M,K)
    Hg = grid @ H.T                     # (M,N)
    Hg_sq = np.sum(np.abs(Hg) ** 2, axis=1)
    M = grid.shape[0]
    s, y = _sample_channel(H, n_sym, rng, const)
    Hx_true = s @ H.T
    d_true = -np.sum(np.abs(y - Hx_true) ** 2, axis=1)                     # (T,)

    out = np.empty(y.shape[0])
    for i in range(0, y.shape[0], chunk):
        d_all = _neg_dist2(y[i:i + chunk], Hg, Hg_sq)                      # (t,M)
        lse = _logsumexp(d_all, axis=1)
        out[i:i + chunk] = (d_true[i:i + chunk] - lse + np.log(M)) * LOG2E
    return float(np.mean(out))          # bits per channel use (== sum over users)


def peruser_ceiling_MI(H, n_sym, rng, const=QPSK):
    """
    sum_k I(x_k; y) via the exact marginal posterior P(x_k=a|y).
    This is the independent-decoding ceiling: no per-user detector (linear or NN)
    can exceed it.  I(x_k;y) = log2|A| - E[ H(P(x_k|y)) ].
    """
    K = H.shape[1]
    Q = len(const)
    grid = all_symbol_vectors(K, const)  # (M,K)
    grid_idx = _to_index(grid[None], const)[0]   # (M,K)
    Hg = grid @ H.T                     # (M,N)
    Hg_sq = np.sum(np.abs(Hg) ** 2, axis=1)
    s, y = _sample_channel(H, n_sym, rng, const)

    masks = [[grid_idx[:, k] == a for a in range(Q)] for k in range(K)]
    Hcond = np.zeros(K)
    nseen = 0
    chunk = 1000
    for i in range(0, y.shape[0], chunk):
        d_all = _neg_dist2(y[i:i + chunk], Hg, Hg_sq)
        logpost = d_all - _logsumexp(d_all, axis=1, keepdims=True)
        post = np.exp(logpost)                                            # (t,M)
        t = post.shape[0]
        for k in range(K):
            Pk = np.stack([post[:, masks[k][a]].sum(axis=1) for a in range(Q)], axis=1)
            Pk = np.clip(Pk, 1e-300, 1.0)
            Hcond[k] += -np.sum(Pk * np.log2(Pk))
        nseen += t
    return float(np.sum(np.log2(Q) - Hcond / nseen))   # sum_k [log2 Q - E H(P(x_k|y))]


def linear_peruser_MI(H, W, n_sym, rng, const=QPSK):
    """
    sum_k I(x_k; (W y)_k) for a linear detector W (K x N), discrete inputs, EXACT (MC).
    Per user the scalar channel is  r_k = (W H s)_k + (W n)_k,  noise var (W W^H)_kk.
    I(x_k;r_k) = E[ log2( p(r_k|x_k) / p(r_k) ) ], with
        p(r_k|x_k=a) = mean over x_{-k} of CN(r_k; (W H x)_k, v_k),
        p(r_k)       = mean over all x of CN(r_k; (W H x)_k, v_k).
    """
    K = H.shape[1]
    Q = len(const)
    grid = all_symbol_vectors(K, const)  # (M,K)
    grid_idx = _to_index(grid[None], const)[0]   # (M,K)
    WH = W @ H                          # (K,K)
    means_grid = grid @ WH.T            # (M,K)  noiseless detector output per grid vector
    v = np.real(np.einsum('ij,ij->i', W, W.conj()))   # (K,) noise var per user = (W W^H)_kk

    s, y = _sample_channel(H, n_sym, rng, const)
    r = y @ W.T                         # (T,K) detector outputs
    sidx = _to_index(s, const)          # (T,K)

    M = grid.shape[0]
    sel = [[grid_idx[:, k] == a for a in range(Q)] for k in range(K)]
    acc = np.zeros(K)
    nseen = 0
    chunk = 1000
    for i in range(0, r.shape[0], chunk):
        rc, sc = r[i:i + chunk], sidx[i:i + chunk]
        t = rc.shape[0]
        for k in range(K):
            vk = v[k]
            ll = -np.abs(rc[:, k][:, None] - means_grid[:, k][None, :]) ** 2 / vk   # (t,M)
            log_marg = _logsumexp(ll, axis=1) - np.log(M)
            log_cond = np.empty(t)
            for a in range(Q):
                rows = np.where(sc[:, k] == a)[0]
                if rows.size:
                    log_cond[rows] = _logsumexp(ll[np.ix_(rows, sel[k][a])], axis=1) - np.log(sel[k][a].sum())
            acc[k] += float(np.sum((log_cond - log_marg) * LOG2E))
        nseen += t
    return float(np.sum(acc / nseen))


# --------------------------------------------------------------------------- #
# _fast exact MC estimators: identical math (float64), but cache the grid across
# calls and replace boolean-mask marginalization with a structured reshape-sum.
# Bit-for-bit equivalent to the originals up to fp summation order.
# --------------------------------------------------------------------------- #
def joint_MI_fast(H, n_sym, rng, const=QPSK, chunk=1000):
    K = H.shape[1]
    grid, _ = _cached_grid(K, const)
    Hg = grid @ H.T
    Hg_sq = np.sum(np.abs(Hg) ** 2, axis=1)
    M = grid.shape[0]
    s, y = _sample_channel(H, n_sym, rng, const)
    d_true = -np.sum(np.abs(y - s @ H.T) ** 2, axis=1)
    out = np.empty(y.shape[0])
    for i in range(0, y.shape[0], chunk):
        d_all = _neg_dist2(y[i:i + chunk], Hg, Hg_sq)
        out[i:i + chunk] = (d_true[i:i + chunk] - _logsumexp(d_all, axis=1) + np.log(M)) * LOG2E
    return float(np.mean(out))


def peruser_ceiling_MI_fast(H, n_sym, rng, const=QPSK, chunk=1000):
    """sum_k I(x_k;y); marginalization via reshape-sum (M = Q^{K-1-k} x Q x Q^k)."""
    K = H.shape[1]
    Q = len(const)
    grid, _ = _cached_grid(K, const)
    Hg = grid @ H.T
    Hg_sq = np.sum(np.abs(Hg) ** 2, axis=1)
    s, y = _sample_channel(H, n_sym, rng, const)
    Hcond = np.zeros(K)
    nseen = 0
    for i in range(0, y.shape[0], chunk):
        d_all = _neg_dist2(y[i:i + chunk], Hg, Hg_sq)
        post = np.exp(d_all - _logsumexp(d_all, axis=1, keepdims=True))     # (t,M)
        t = post.shape[0]
        for k in range(K):
            # digit_k is the middle axis when M is viewed as (Q^{K-1-k}, Q, Q^k)
            Pk = post.reshape(t, Q ** (K - 1 - k), Q, Q ** k).sum(axis=(1, 3))   # (t,Q)
            Pk = np.clip(Pk, 1e-300, 1.0)
            Hcond[k] += -np.sum(Pk * np.log2(Pk))
        nseen += t
    return float(np.sum(np.log2(Q) - Hcond / nseen))


def linear_peruser_MI_fast(H, W, n_sym, rng, const=QPSK, chunk=1000):
    """
    sum_k I(x_k;(Wy)_k), discrete inputs, exact. Identical math to the normal version
    but with the grid cached across Es-grid calls. Cost is dominated by building the
    (T, M=Q^K) likelihood tensor and is therefore intrinsic to discrete Q^K marginalisation;
    a row-subset per-a inner loop minimises the exp work to O(T*M/Q).
    """
    K = H.shape[1]
    Q = len(const)
    grid, grid_idx = _cached_grid(K, const)
    means_grid = grid @ (W @ H).T
    v = np.real(np.einsum('ij,ij->i', W, W.conj()))
    s, y = _sample_channel(H, n_sym, rng, const)
    r = y @ W.T
    sidx = _to_index(s, const)
    M = grid.shape[0]
    sel = [[grid_idx[:, k] == a for a in range(Q)] for k in range(K)]
    acc = np.zeros(K)
    nseen = 0
    for i in range(0, r.shape[0], chunk):
        rc, sc = r[i:i + chunk], sidx[i:i + chunk]
        t = rc.shape[0]
        for k in range(K):
            ll = -np.abs(rc[:, k][:, None] - means_grid[:, k][None, :]) ** 2 / v[k]
            log_marg = _logsumexp(ll, axis=1) - np.log(M)
            log_cond = np.empty(t)
            for a in range(Q):
                rows = np.where(sc[:, k] == a)[0]
                if rows.size:
                    log_cond[rows] = _logsumexp(ll[np.ix_(rows, sel[k][a])], axis=1) - np.log(sel[k][a].sum())
            acc[k] += float(np.sum((log_cond - log_marg) * LOG2E))
        nseen += t
    return float(np.sum(acc / nseen))


def linear_oneuser_MI_fast(H, W, k0, n_sym, rng, const=QPSK):
    """I(x_{k0}; (Wy)_{k0}) for one user, with cached grid."""
    K = H.shape[1]
    Q = len(const)
    grid, grid_idx = _cached_grid(K, const)
    means_grid = grid @ (W @ H).T
    vk = float(np.real(np.dot(W[k0], W[k0].conj())))
    s, y = _sample_channel(H, n_sym, rng, const)
    r = y @ W.T
    sidx = _to_index(s, const)
    M = grid.shape[0]
    sel = [grid_idx[:, k0] == a for a in range(Q)]
    ll = -np.abs(r[:, k0][:, None] - means_grid[:, k0][None, :]) ** 2 / vk
    log_marg = _logsumexp(ll, axis=1) - np.log(M)
    log_cond = np.empty(r.shape[0])
    for a in range(Q):
        rows = np.where(sidx[:, k0] == a)[0]
        if rows.size:
            log_cond[rows] = _logsumexp(ll[np.ix_(rows, sel[a])], axis=1) - np.log(sel[a].sum())
    return float(np.mean((log_cond - log_marg) * LOG2E))


def lmmse_sic_MI_fast(H, n_sym, rng, const=QPSK, order=None):
    K = H.shape[1]
    if order is None:
        order = list(range(K))
    total = 0.0
    for idx in range(K):
        rem = order[idx:]
        Hr = H[:, rem]
        total += linear_oneuser_MI_fast(Hr, lmmse(Hr), 0, n_sym, rng, const)
    return total


# --------------------------------------------------------------------------- #
# Successive interference cancellation (genie-aided) achievable rates
# --------------------------------------------------------------------------- #
def linear_oneuser_MI(H, W, k0, n_sym, rng, const=QPSK):
    """I(x_{k0}; (W y)_{k0}) for one user, discrete inputs, exact MC (helper for SIC)."""
    K = H.shape[1]
    Q = len(const)
    grid = all_symbol_vectors(K, const)
    grid_idx = _to_index(grid[None], const)[0]
    means_grid = grid @ (W @ H).T
    vk = float(np.real(np.dot(W[k0], W[k0].conj())))
    s, y = _sample_channel(H, n_sym, rng, const)
    r = y @ W.T
    sidx = _to_index(s, const)
    M = grid.shape[0]
    sel = [grid_idx[:, k0] == a for a in range(Q)]
    ll = -np.abs(r[:, k0][:, None] - means_grid[:, k0][None, :]) ** 2 / vk     # (T,M)
    log_marg = _logsumexp(ll, axis=1) - np.log(M)
    log_cond = np.empty(r.shape[0])
    for a in range(Q):
        rows = np.where(sidx[:, k0] == a)[0]
        if rows.size:
            log_cond[rows] = _logsumexp(ll[np.ix_(rows, sel[a])], axis=1) - np.log(sel[a].sum())
    return float(np.mean((log_cond - log_marg) * LOG2E))


def lmmse_sic_MI(H, n_sym, rng, const=QPSK, order=None):
    """
    Genie-aided LMMSE-SIC achievable rate, sum_k I(x_k; (W_k y_k)_k), discrete inputs.
    Stage k: users decoded earlier are perfectly cancelled (true symbols), an LMMSE
    filter for the remaining users is applied, and user k's scalar output is scored.
    Mirrors the teacher-forced neural-joint decoder, so the comparison is apples-to-apples.
    For Gaussian inputs this would equal the sum capacity I(x;y).
    """
    K = H.shape[1]
    if order is None:
        order = list(range(K))
    total = 0.0
    for idx in range(K):
        rem = order[idx:]                 # user order[idx] (=rem[0]) plus undecoded
        Hr = H[:, rem]                    # decoded users removed (genie)
        Wr = lmmse(Hr)
        total += linear_oneuser_MI(Hr, Wr, 0, n_sym, rng, const)
    return total


# --------------------------------------------------------------------------- #
# Gaussian-input rates (closed form)
# --------------------------------------------------------------------------- #
def joint_gaussian_MI(H):
    """I(x;y) = log2 det(I + H H^H), Gaussian inputs. bits per channel use."""
    eig = np.linalg.eigvalsh(H @ H.conj().T)
    return float(np.sum(np.log2(1.0 + np.clip(eig, 0, None))))


def peruser_ceiling_gaussian_MI(H):
    """
    Gaussian per-user ceiling  sum_k I(x_k;y) = -sum_k log2(mmse_k),
    mmse_k = [(I + H^H H)^-1]_kk.  Equals sum_k I(x_k; Wy) for ANY full-rank linear
    front-end W (y->Wy injective); for the LMMSE filter it also equals the scalar
    sum_k I(x_k;(Wy)_k).  Gaussian analog of the (discrete) per-user ceiling.
    """
    K = H.shape[1]
    mmse = np.real(np.diag(np.linalg.inv(np.eye(K) + H.conj().T @ H)))
    return float(np.sum(-np.log2(mmse)))


def linear_gaussian_MI(H, W):
    """
    sum_k log2(1+SINR_k) for linear detector W with Gaussian unit-power inputs,
    treating residual interference as Gaussian. r = W H x + W n.
    """
    WH = W @ H
    G = np.abs(WH) ** 2
    sig = np.diag(G).real
    intf = G.sum(axis=1) - sig
    noise = np.real(np.einsum('ij,ij->i', W, W.conj()))     # (W W^H)_kk
    sinr = sig / (intf + noise)
    return float(np.sum(np.log2(1.0 + sinr)))


# --------------------------------------------------------------------------- #
# Linear detectors
# --------------------------------------------------------------------------- #
def matched_filter(H):
    return H.conj().T


def decorrelator(H):
    return np.linalg.pinv(H)


def lmmse(H):
    """LMMSE for unit-power symbols and unit noise var: W = H^H (H H^H + I)^-1."""
    N = H.shape[0]
    return H.conj().T @ np.linalg.inv(H @ H.conj().T + np.eye(N))


def _logsumexp(a, axis=None, keepdims=False):
    m = np.max(a, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))
    if not keepdims and axis is not None:
        out = np.squeeze(out, axis=axis)
    return out
