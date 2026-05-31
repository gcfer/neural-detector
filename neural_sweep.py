"""
Neural INDEPENDENT (per-user) detector: achievable spectral efficiency C vs load beta,
at fixed Eb/N0 = 10 dB, for QPSK and 16-QAM. Produces neural_peruser_sweep_final.npz,
which classical_curves.py overlays on the large-system figure.

The detector g_theta(y) outputs a per-user posterior over the constellation and is trained
by per-user cross-entropy; the achievable rate is the GMI  R_k = log2 Q - CE_k  (a provable
lower bound on the independent-decoding rate I(x_k;y), tight when well trained). It needs no
Q^K enumeration and runs on Apple MLX (GPU). Implementation: nn_mlx.train_nn_rate_mlx_fast.

Each (config, constellation) is solved at its own fixed-Eb operating point via a damped
fixed-point iteration on  Es = C*Eb/beta  (lands ON the operating Es, no grid interpolation),
then evaluated at high budget averaged over seeds.

Requires Apple Silicon (MLX). Run: python neural_sweep.py
"""
import numpy as np
import mac
import nn_mlx as nm

Eb = 10.0 ** (10.0 / 10.0)
CONFIGS = [(4, 2), (3, 3), (3, 4), (3, 5), (2, 4), (2, 5)]   # (N, K) -> beta = K/N

SEARCH_ITERS = 6
SEARCH_STEPS = 6000
DAMP = 0.6
FINAL_STEPS = 14000
FINAL_EVAL = 80000
FINAL_SEEDS = 2


def C_of(Hof, const, Es, steps, eval_sym, N, seed=0):
    return nm.train_nn_rate_mlx_fast(Hof(Es), const, steps=steps, eval_sym=eval_sym, seed=seed)[0] / N


def operating_point(Hof, const, beta, N):
    Es = max(2.0, np.log2(len(const)) * Eb / max(beta, 0.5) * 0.5)
    for _ in range(SEARCH_ITERS):
        C = C_of(Hof, const, Es, SEARCH_STEPS, 20000, N)
        Es = (1 - DAMP) * Es + DAMP * (C * Eb / beta)
    Cs = [C_of(Hof, const, Es, FINAL_STEPS, FINAL_EVAL, N, seed=s) for s in range(FINAL_SEEDS)]
    return float(np.mean(Cs)), float(np.std(Cs))


def main():
    rows = {}
    print(f"{'N':>2}{'K':>3}{'beta':>7}{'QPSK':>9}{'16-QAM':>9}", flush=True)
    for N, K in CONFIGS:
        beta = K / N
        S = mac.random_iid_S(N, K, np.random.default_rng(0))
        Hof = lambda Es: np.sqrt(Es) * S
        cq, sq = operating_point(Hof, mac.QPSK, beta, N)
        cm, sm = operating_point(Hof, mac.QAM16, beta, N)
        rows[f"N{N}_K{K}"] = (beta, cq, cm, sq, sm)
        print(f"{N:>2}{K:>3}{beta:>7.3f}{cq:>9.3f}{cm:>9.3f}", flush=True)
        np.savez("neural_peruser_sweep_final.npz",
                 config=np.array(list(rows.keys())),
                 beta=np.array([rows[k][0] for k in rows]),
                 qpsk=np.array([rows[k][1] for k in rows]),
                 qam16=np.array([rows[k][2] for k in rows]),
                 qpsk_std=np.array([rows[k][3] for k in rows]),
                 qam16_std=np.array([rows[k][4] for k in rows]))
    print("done -> neural_peruser_sweep_final.npz", flush=True)


if __name__ == "__main__":
    main()
