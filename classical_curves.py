"""
Large-system spectral-efficiency curves vs load beta at fixed Eb/N0, and a TikZ/pgfplots
figure that overlays the simulated NEURAL INDEPENDENT detector points (QPSK and 16-QAM).

Classical curves (closed form + light Monte-Carlo), each at its own fixed-Eb operating point
[snr = gamma*C/beta, solved self-consistently]:
    orthogonal            C = min{1,beta} * r*,  r* solves r = log2(1+gamma r)
    optimal dense         Verdu-Shamai log-det (Marchenko-Pastur)
    optimal sparse Ns=1   E_{d~Poisson(beta)}[log2(1+snr d)]
    optimal sparse Ns=2   Monte-Carlo (1/N) E[log det(I + snr S S^H)], 2 nonzeros/col
    LMMSE dense (Gauss)   Tse-Hanly SINR fixed point
    SUMF sparse Ns=1      beta * E_{m~Poisson(beta)}[log2(1+ snr/(1+m snr))]
    SUMF dense            beta * log2(1 + snr/(1+beta snr))

Neural points are read from neural_peruser_sweep_final.npz (produced by neural_sweep.py).

Usage:
    python classical_curves.py [Eb_dB]      # default 10 dB
-> writes classical_curves{_suffix}.npz and classical_curves{_suffix}.tex
Compile the figure with:  pdflatex classical_curves.tex
"""

import sys
import os
import numpy as np
from math import lgamma

EBDB = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0   # Eb/N0 in dB
GAMMA = 10.0 ** (EBDB / 10.0)
SUFFIX = "" if EBDB == 10.0 else f"_{EBDB:g}dB"
NEURAL_NPZ = "neural_peruser_sweep_final.npz"


# ---------- per-scheme spectral efficiency C(snr, beta) ---------- #
def _poisson_pmf(beta, dmax):
    d = np.arange(0, dmax + 1)
    logp = -beta + d * np.log(beta + 1e-300) - np.array([lgamma(k + 1) for k in d])
    return d, np.exp(logp)


def opt_dense(snr, beta):
    F = (np.sqrt(snr * (1 + np.sqrt(beta)) ** 2 + 1) -
         np.sqrt(snr * (1 - np.sqrt(beta)) ** 2 + 1)) ** 2
    return (beta * np.log2(1 + snr - 0.25 * F)
            + np.log2(1 + snr * beta - 0.25 * F)
            - (np.log2(np.e) / (4 * snr)) * F)


def opt_sparse1(snr, beta):
    dmax = int(beta + 12 * np.sqrt(beta + 1) + 30)
    d, p = _poisson_pmf(beta, dmax)
    return float(np.sum(p * np.log2(1 + snr * d)))


def sparse2_eigs(beta, N=800, reps=4, rng=None):
    rng = rng or np.random.default_rng(0)
    K = int(round(beta * N))
    if K == 0:
        return np.zeros(N)
    eigs = []
    for _ in range(reps):
        S = np.zeros((N, K))
        for k in range(K):
            rows = rng.choice(N, size=2, replace=False)
            S[rows, k] = 1.0 / np.sqrt(2.0)
        eigs.append(np.clip(np.linalg.eigvalsh(S @ S.T), 0, None))
    return np.concatenate(eigs)


def opt_sparse2_from_eigs(snr, eigs, N):
    reps = len(eigs) // N
    return float(np.sum(np.log2(1 + snr * eigs)) / (N * reps))


def lmmse_sinr(snr, beta):
    g = snr
    for _ in range(500):
        g_new = snr / (1.0 + beta * snr / (1.0 + g))
        if abs(g_new - g) < 1e-12:
            g = g_new
            break
        g = g_new
    return g


def lmmse_dense(snr, beta):
    return beta * np.log2(1 + lmmse_sinr(snr, beta))


def sumf_dense(snr, beta):
    return beta * np.log2(1 + snr / (1 + beta * snr))


def sumf_sparse1(snr, beta):
    dmax = int(beta + 12 * np.sqrt(beta + 1) + 30)
    m, p = _poisson_pmf(beta, dmax)
    return float(beta * np.sum(p * np.log2(1 + snr / (1 + m * snr))))


def solve_fixedEb(f, beta, snr_grid):
    Cg = np.array([f(s, beta) for s in snr_grid])
    line = (beta / GAMMA) * snr_grid
    diff = Cg - line
    sc = np.where(np.diff(np.sign(diff)) < 0)[0]
    if sc.size == 0:
        return 0.0
    i = sc[-1]
    s = snr_grid[i] - diff[i] * (snr_grid[i + 1] - snr_grid[i]) / (diff[i + 1] - diff[i])
    return float(np.interp(s, snr_grid, Cg))


def compute_curves():
    betas = np.round(np.arange(0.05, 5.0001, 0.05), 4)
    betas_mc = np.round(np.arange(0.1, 5.0001, 0.1), 4)
    snr_grid = np.geomspace(1e-3, 1e4, 4000)
    r = 1.0
    for _ in range(200):
        r = np.log2(1 + GAMMA * r)
    curves = {
        "opt_dense":    np.array([solve_fixedEb(opt_dense, b, snr_grid) for b in betas]),
        "opt_sparse1":  np.array([solve_fixedEb(opt_sparse1, b, snr_grid) for b in betas]),
        "lmmse_dense":  np.array([solve_fixedEb(lmmse_dense, b, snr_grid) for b in betas]),
        "sumf_dense":   np.array([solve_fixedEb(sumf_dense, b, snr_grid) for b in betas]),
        "sumf_sparse1": np.array([solve_fixedEb(sumf_sparse1, b, snr_grid) for b in betas]),
        "orthogonal":   np.minimum(1.0, betas) * r,
    }
    rng = np.random.default_rng(0)
    sp2 = []
    for b in betas_mc:
        eigs = sparse2_eigs(b, N=800, rng=rng)
        sp2.append(solve_fixedEb(lambda s, bb, e=eigs: opt_sparse2_from_eigs(s, e, 800), b, snr_grid))
    return betas, curves, betas_mc, {"opt_sparse2": np.array(sp2)}


# ---------- TikZ generation ---------- #
def _coords(x, y, origin=False):
    s = "(0.0000,0.0000) " if origin else ""
    return s + " ".join(f"({xi:.4f},{yi:.4f})" for xi, yi in zip(x, y))


def write_tex(betas, curves, betas_mc, curves_mc):
    neural = None
    if os.path.exists(NEURAL_NPZ):
        d = np.load(NEURAL_NPZ, allow_pickle=True)
        neural = (d["beta"], d["qpsk"], d["qam16"])

    L = []
    A = L.append
    A(r"\documentclass[tikz,border=3pt]{standalone}")
    A(r"\usepackage{pgfplots}")
    A(r"\pgfplotsset{compat=1.18}")
    A(r"% Toggle: \showsimtrue overlays the simulated neural detector points; \showsimfalse hides them.")
    A(r"\newif\ifshowsim")
    A(r"\showsimtrue")
    A(r"\begin{document}")
    A(r"\begin{tikzpicture}")
    A(r"\begin{axis}[")
    A(r"  width=16cm, height=11cm,")
    A(r"  xlabel={load $\beta = K/N$}, ylabel={spectral efficiency $C$ [bits/channel use]},")
    A(r"  xmin=0, xmax=5, ymin=0, xtick={0,1,2,3,4,5},")
    A(r"  grid=both, grid style={gray!20},")
    A(rf"  title={{Large-system spectral efficiency, $E_b/N_0 = {EBDB:g}$ dB}},")
    A(r"  legend cell align=left, legend style={font=\small, at={(1.02,0.5)}, anchor=west},")
    A(r"]")

    def plot(style, x, y, label):
        A(rf"\addplot[{style}] coordinates {{{_coords(x, y, origin=True)}}};")
        A(rf"\addlegendentry{{{label}}}")

    plot("black, very thick", betas, curves["opt_dense"], r"optimal, dense")
    plot("blue, thick", betas, curves["opt_sparse1"], r"optimal, sparse $N_s{=}1$")
    plot("blue, thick, dashed", betas_mc, curves_mc["opt_sparse2"], r"optimal, sparse $N_s{=}2$")
    plot("red, thick", betas, curves["lmmse_dense"], r"LMMSE, dense (Gaussian)")
    plot("teal, thick", betas, curves["sumf_sparse1"], r"SUMF, sparse $N_s{=}1$")
    plot("orange, thick", betas, curves["sumf_dense"], r"SUMF, dense")
    plot("black, dotted, thick", betas, curves["orthogonal"], r"orthogonal")

    if neural is not None:
        nb, nq, nm = neural
        A(r"\ifshowsim")
        A(rf"  \addplot[mark=*, mark size=2pt, green!55!black, thick] coordinates {{{_coords(nb, nq, origin=True)}}};")
        A(r"  \addlegendentry{neural independent (QPSK)}")
        A(rf"  \addplot[mark=triangle*, mark size=2.8pt, purple, thick] coordinates {{{_coords(nb, nm, origin=True)}}};")
        A(r"  \addlegendentry{neural independent (16-QAM)}")
        A(r"\fi")

    A(r"\end{axis}")
    A(r"\end{tikzpicture}")
    A(r"\end{document}")
    fname = f"classical_curves{SUFFIX}.tex"
    with open(fname, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"wrote {fname}")


def main():
    betas, curves, betas_mc, curves_mc = compute_curves()
    np.savez(f"classical_curves{SUFFIX}.npz", beta=betas, beta_mc=betas_mc, gamma_dB=EBDB,
             **curves, **curves_mc)
    write_tex(betas, curves, betas_mc, curves_mc)


if __name__ == "__main__":
    main()
