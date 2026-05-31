# neural-detector

Large-system **spectral efficiency vs load** for a Gaussian NOMA channel,
comparing classical multiuser receivers with a **neural independent (per-user) detector**, at a
fixed energy-per-bit `Eb/N0`.

![figure](classical_curves.pdf)

The figure (`classical_curves.pdf`) overlays:

**Classical large-system curves** (closed-form + light Monte-Carlo), each solved at its own
fixed-`Eb` operating point:
- optimal, dense (log-det, i.i.d. spreading)
- optimal, sparse `Ns=1` and `Ns=2` (each column has `Ns` nonzero entries)
- LMMSE, dense (fixed point)
- SUMF (single-user matched filter), dense and sparse `Ns=1`
- orthogonal (`min{1,β}` usable dimensions)

**Simulated neural detector points** (markers):
- neural independent (QPSK)
- neural independent (16-QAM)

These are the achievable spectral efficiencies of a learned **per-user** detector `g_θ(y)` that
outputs a posterior over the constellation for each user and is trained by per-user
cross-entropy. Its rate is the generalized mutual information `R_k = log2 Q − CE_k` — a provable
lower bound on the independent-decoding rate `Σ_k I(x_k;y)`, tight when well trained. It needs
**no `Q^K` enumeration**, so it scales to any load.

## Channel model

```
y = sqrt(Es) · S · x + n
```
`S` is the `N×K` spreading matrix (unit-norm columns), `x` the `K` user symbols from a unit-power
constellation, `n ~ CN(0, I)`. Load `β = K/N`. Spectral efficiency `C = (sum-rate)/N`
[bits/channel use]. Fixed-`Eb`: `Es = C·Eb/β`, solved self-consistently per scheme.

## Files

| file | purpose |
|---|---|
| `mac.py`            | channel model, constellations (QPSK / 16-QAM), spreading matrices (numpy) |
| `nn_mlx.py`         | neural detectors in Apple MLX (**fastest neural path**); `train_nn_rate_mlx_fast` is the per-user (independent) detector used here |
| `mac_numba.py`      | **fastest exact** discrete-input rate estimators (Numba JIT): joint MI, per-user ceiling, linear per-user, LMMSE-SIC. Not needed for the figure (its curves are analytic + neural), but available to compute exact discrete-input references / detector points |
| `neural_sweep.py`   | computes neural-independent `C(β)` for QPSK & 16-QAM → `neural_peruser_sweep_final.npz` |
| `classical_curves.py` | computes the classical curves and writes the TikZ/pgfplots figure (overlays the neural npz) |
| `classical_curves.npz` | cached classical-curve data |
| `neural_peruser_sweep_final.npz` | cached neural-detector points (β, QPSK, 16-QAM) |
| `classical_curves.tex` / `.pdf` | the figure (rendered with all detectors) |

## Reproduce

Dependencies: `pip install -r requirements.txt` (numpy; **MLX requires Apple Silicon** for the
neural part), plus a LaTeX install with `pgfplots` for the figure.

**Redraw the figure from cached data** (numpy + LaTeX only, no GPU):
```bash
python classical_curves.py        # recomputes classical curves, writes classical_curves.tex
pdflatex classical_curves.tex     # -> classical_curves.pdf
```

**Recompute the neural detector points** (Apple Silicon + MLX):
```bash
python neural_sweep.py            # -> neural_peruser_sweep_final.npz  (~30-40 min)
python classical_curves.py        # regenerate the .tex overlay
pdflatex classical_curves.tex
```

For a different operating point pass `Eb/N0` in dB, e.g. `python classical_curves.py 3`.

## Figure toggle

`classical_curves.tex` defines a boolean:
```latex
\newif\ifshowsim
\showsimtrue     % \showsimfalse hides the neural points (classical baseline only)
```

## Fast implementations

Two GPU/JIT-accelerated, accuracy-neutral paths are included:
- **Neural (MLX):** `nn_mlx.py` runs on the Apple-Silicon GPU with on-device sampling,
  real-augmented channel arithmetic, and `mx.compile` (the `*_mlx_fast` functions). This is
  what `neural_sweep.py` uses.
- **Exact discrete (Numba):** `mac_numba.py` JIT-compiles the exact `Q^K` rate estimators with
  a fused, parallel two-pass log-sum-exp (`@njit(parallel=True, fastmath=True)`), bit-identical
  to a plain-numpy reference but ~1–2 orders of magnitude faster (most for 16-QAM). Use these if
  you want to add exact discrete-input detector points (e.g. discrete joint/ceiling/LMMSE-SIC).

## Method notes

- Neural rate = GMI from cross-entropy training (mismatched-decoding achievable rate).
- Operating point: damped fixed-point iteration on `Es = C·Eb/β` (no grid-interpolation error),
  then high-budget evaluation averaged over seeds.
- Key effect visible in the figure: under heavy load the **independent** per-user detector cannot
  resolve dense (16-QAM) interference, so 16-QAM peaks early and collapses while QPSK degrades
  gracefully — a crossover near `β ≈ 1.5`.
