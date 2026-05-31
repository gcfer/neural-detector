"""
MLX (Apple-native) reimplementations of the neural trainers, suffixed _mlx.

Same architectures / steps / batch / lr / seed as the torch versions, so achieved
rates match within training noise. Complex channel + feature math stays in numpy
(cheap, exact); only the NN forward/backward/optimizer run on the MLX GPU. On Apple
Silicon's unified memory the numpy->mx handoff is essentially free.

Achievable rate read-out is identical:  R_k = log2 Q - CE_k.
"""

import numpy as np
import mlx.core as mx
import mlx.nn as mnn
import mlx.optimizers as optim
import mac


class _MLP(mnn.Module):
    def __init__(self, d_in, d_out, width, depth):
        super().__init__()
        self.layers = []
        d = d_in
        for _ in range(depth):
            self.layers.append(mnn.Linear(d, width)); d = width
        self.layers.append(mnn.Linear(d, d_out))

    def __call__(self, x):
        for lin in self.layers[:-1]:
            x = mnn.relu(lin(x))
        return self.layers[-1](x)


def _ce_bits_per_user(logits, tgt, K, Q):
    """logits (B,K,Q) mx, tgt (B,K) mx int -> per-user CE in bits (numpy (K,))."""
    lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)            # log-softmax
    ce = -mx.take_along_axis(lp, tgt[..., None], axis=-1).squeeze(-1)     # (B,K) nats
    return np.array(mx.mean(ce, axis=0)) / np.log(2)


# --------------------------------------------------------------------------- #
# per-user MLP / MLP-S
# --------------------------------------------------------------------------- #
def train_nn_rate_mlx(H, const=mac.QPSK, steps=4000, batch=4096, width=256, depth=4, lr=1e-3,
                      eval_sym=40000, seed=0, use_mf=False):
    N, K = H.shape
    Q = len(const)
    rng = np.random.default_rng(seed)
    d_in = 2 * K if use_mf else 2 * N
    model = _MLP(d_in, K * Q, width, depth)
    opt = optim.Adam(learning_rate=lr)

    def feats(y):
        z = y @ H.conj() if use_mf else y
        return mx.array(np.concatenate([z.real, z.imag], axis=1).astype(np.float32))

    def loss_fn(model, x, tgt):
        logits = model(x).reshape(-1, K, Q)
        lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        ce = -mx.take_along_axis(lp, tgt[..., None], axis=-1).squeeze(-1)
        return mx.mean(ce)

    lg = mnn.value_and_grad(model, loss_fn)
    for _ in range(steps):
        s, y = mac._sample_channel(H, batch, rng, const)
        x = feats(y); tgt = mx.array(mac._to_index(s, const).astype(np.int32))
        _, grads = lg(model, x, tgt)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state)

    s, y = mac._sample_channel(H, eval_sym, rng, const)
    logits = model(feats(y)).reshape(-1, K, Q)
    ce_bits = _ce_bits_per_user(logits, mx.array(mac._to_index(s, const).astype(np.int32)), K, Q)
    R_k = np.log2(Q) - ce_bits
    return float(R_k.sum()), R_k


def train_mlpS_rate_mlx(H, const=mac.QPSK, **kw):
    return train_nn_rate_mlx(H, const, use_mf=True, **kw)


# --------------------------------------------------------------------------- #
# joint decoder (weight-shared neural SIC), vectorized over stages
# --------------------------------------------------------------------------- #
class _JointMLX(mnn.Module):
    def __init__(self, K, Q, width, depth, anchored, mmse0=None):
        super().__init__()
        self.mlp = _MLP(4 * K, Q, width, depth)
        self.anchored = anchored
        if anchored:
            self.gate = mx.zeros((K,))
            self.logs2 = mx.log(mx.array(mmse0.astype(np.float32)))

    def __call__(self, feat, dist2=None):
        B, K, _ = feat.shape
        logits = self.mlp(feat.reshape(B * K, -1)).reshape(B, K, -1)
        if self.anchored:
            logits = -dist2 / mx.exp(self.logs2)[None, :, None] + self.gate[None, :, None] * logits
        return logits


def _joint_mlx(H, const, steps, batch, width, depth, lr, eval_sym, seed, order, anchored):
    N, K = H.shape
    Q = len(const)
    if order is None:
        order = list(range(K))
    rng = np.random.default_rng(seed)
    G = H.conj().T @ H
    pts = const.astype(np.complex128)
    onehot = np.zeros((K, K), np.float32); decmask = np.zeros((K, K), np.float32)
    for idx, k in enumerate(order):
        onehot[idx, k] = 1.0; decmask[idx, order[:idx]] = 1.0
    order_arr = np.array(order)

    Wr0 = mmse0 = None
    if anchored:
        Wr0 = np.zeros((K, N), np.complex128); mmse0 = np.zeros(K)
        for idx in range(K):
            rem = order[idx:]; Hr = H[:, rem]
            Wr0[idx] = mac.lmmse(Hr)[0]
            mmse0[idx] = np.real(np.linalg.inv(np.eye(len(rem)) + Hr.conj().T @ Hr)[0, 0])

    model = _JointMLX(K, Q, width, depth, anchored, mmse0)
    opt = optim.Adam(learning_rate=lr)

    def build(batch_n):
        s, y = mac._sample_channel(H, batch_n, rng, const)
        sidx = mac._to_index(s, const)
        x_dec = s[:, None, :] * decmask[None, :, :]                 # (B,K,K) complex
        r = y @ H.conj()                                            # matched filter (B,K)
        r = r[:, None, :] - x_dec @ G.T                            # residual (B,K,K)
        feat = np.concatenate([r.real, r.imag,
                               np.broadcast_to(onehot[None], (batch_n, K, K)),
                               np.broadcast_to(decmask[None], (batch_n, K, K))], axis=2).astype(np.float32)
        tgt = sidx[:, order_arr].astype(np.int32)                  # (B,K) stage targets
        dist2 = None
        if anchored:
            y_res = y[:, None, :] - x_dec @ H.T                    # (B,K,N)
            xhat = (y_res * Wr0[None, :, :]).sum(axis=2)           # (B,K)
            dist2 = (np.abs(xhat[:, :, None] - pts[None, None, :]) ** 2).astype(np.float32)
        return mx.array(feat), mx.array(tgt), (mx.array(dist2) if anchored else None)

    def loss_fn(model, feat, tgt, dist2):
        logits = model(feat, dist2)
        lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        ce = -mx.take_along_axis(lp, tgt[..., None], axis=-1).squeeze(-1)
        return mx.mean(ce)

    lg = mnn.value_and_grad(model, loss_fn)
    for _ in range(steps):
        feat, tgt, dist2 = build(batch)
        _, grads = lg(model, feat, tgt, dist2)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state)

    feat, tgt, dist2 = build(eval_sym)
    ce_bits = _ce_bits_per_user(model(feat, dist2), tgt, K, Q)
    R_k = np.log2(Q) - ce_bits
    return float(R_k.sum()), R_k


def train_neural_joint_rate_mlx(H, const=mac.QPSK, steps=4000, batch=4096, width=64, depth=2,
                                lr=1e-3, eval_sym=40000, seed=0, order=None):
    return _joint_mlx(H, const, steps, batch, width, depth, lr, eval_sym, seed, order, False)


def train_neural_joint_lmmse_rate_mlx(H, const=mac.QPSK, steps=4000, batch=4096, width=64, depth=2,
                                      lr=1e-3, eval_sym=40000, seed=0, order=None):
    return _joint_mlx(H, const, steps, batch, width, depth, lr, eval_sym, seed, order, True)


# --------------------------------------------------------------------------- #
# _mlx_fast variants: on-device sampling + real-augmented channel arithmetic +
# mx.compile. The step loop never touches numpy. Same architecture / steps /
# batch / lr / seed as the _mlx versions.
# --------------------------------------------------------------------------- #
def _real_aug(H):
    """(N,K) complex -> (2N, 2K) real-augmented, acts as y_aug = x_aug @ H_aug.T + n_aug."""
    return np.block([[H.real, -H.imag], [H.imag, H.real]]).astype(np.float32)


def train_nn_rate_mlx_fast(H, const=mac.QPSK, steps=4000, batch=4096, width=256, depth=4,
                           lr=1e-3, eval_sym=40000, seed=0, use_mf=False):
    """Per-user MLP / MLP-S, on-device sampling + real-augmented channel + mx.compile."""
    N, K = H.shape; Q = len(const)
    H_aug = mx.array(_real_aug(H))
    pts_re = mx.array(const.real.astype(np.float32))
    pts_im = mx.array(const.imag.astype(np.float32))
    inv_sqrt2 = mx.array(np.float32(1.0 / np.sqrt(2)))
    mx.random.seed(seed)
    model = _MLP(2 * K if use_mf else 2 * N, K * Q, width, depth)
    opt = optim.Adam(learning_rate=lr)

    def sample(B):
        idx = mx.random.randint(0, Q, (B, K))
        x_aug = mx.concatenate([pts_re[idx], pts_im[idx]], axis=1)        # (B, 2K)
        y_aug = x_aug @ H_aug.T + mx.random.normal((B, 2 * N)) * inv_sqrt2
        feat = (y_aug @ H_aug) if use_mf else y_aug                       # MLP-S: H^H y in real-aug
        return feat, idx

    def loss_fn(model, feat, tgt):
        logits = model(feat).reshape(-1, K, Q)
        lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        ce = -mx.take_along_axis(lp, tgt[..., None], axis=-1).squeeze(-1)
        return mx.mean(ce)
    lg = mnn.value_and_grad(model, loss_fn)
    state = [model.state, opt.state]
    def step(feat, tgt):
        loss, grads = lg(model, feat, tgt)
        opt.update(model, grads)
        return loss
    step_c = mx.compile(step, inputs=state, outputs=state)

    for _ in range(steps):
        feat, idx = sample(batch)
        step_c(feat, idx)
        mx.eval(model.state, opt.state)

    feat, idx = sample(eval_sym)
    ce_bits = _ce_bits_per_user(model(feat).reshape(-1, K, Q), idx, K, Q)
    R_k = np.log2(Q) - ce_bits
    return float(R_k.sum()), R_k


def train_mlpS_rate_mlx_fast(H, const=mac.QPSK, **kw):
    return train_nn_rate_mlx_fast(H, const, use_mf=True, **kw)


def _joint_mlx_fast(H, const, steps, batch, width, depth, lr, eval_sym, seed, order, anchored):
    N, K = H.shape; Q = len(const)
    if order is None:
        order = list(range(K))
    H_aug_np = _real_aug(H)
    H_aug = mx.array(H_aug_np)
    G_aug = mx.array(H_aug_np.T @ H_aug_np)
    pts_re = mx.array(const.real.astype(np.float32))
    pts_im = mx.array(const.imag.astype(np.float32))
    inv_sqrt2 = mx.array(np.float32(1.0 / np.sqrt(2)))
    onehot_np = np.zeros((K, K), np.float32); decmask_np = np.zeros((K, K), np.float32)
    for idx, k in enumerate(order):
        onehot_np[idx, k] = 1.0; decmask_np[idx, order[:idx]] = 1.0
    onehot = mx.array(onehot_np); decmask = mx.array(decmask_np)
    decmask_tiled = mx.array(np.concatenate([decmask_np, decmask_np], axis=1))   # (K, 2K)
    order_mx = mx.array(np.array(order, np.int32))
    mx.random.seed(seed)

    mmse0 = Wr0_aug = pts_aug = None
    if anchored:
        Wr0_np = np.zeros((K, N), np.complex128); mmse0 = np.zeros(K)
        for idx in range(K):
            rem = order[idx:]; Hr = H[:, rem]
            Wr0_np[idx] = mac.lmmse(Hr)[0]
            mmse0[idx] = np.real(np.linalg.inv(np.eye(len(rem)) + Hr.conj().T @ Hr)[0, 0])
        # per-stage real-augmented 2x2N row block
        WA = np.zeros((K, 2, 2 * N), np.float32)
        for k in range(K):
            wr, wi = Wr0_np[k].real, Wr0_np[k].imag
            WA[k, 0, :N] = wr; WA[k, 0, N:] = -wi
            WA[k, 1, :N] = wi; WA[k, 1, N:] = wr
        Wr0_aug = mx.array(WA)
        pts_aug = mx.array(np.stack([const.real, const.imag], axis=1).astype(np.float32))

    class _JointModel(mnn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = _MLP(4 * K, Q, width, depth)
            if anchored:
                self.gate = mx.zeros((K,))
                self.logs2 = mx.log(mx.array(mmse0.astype(np.float32)))

    jm = _JointModel()
    opt = optim.Adam(learning_rate=lr)

    def sample_feat(B):
        idx = mx.random.randint(0, Q, (B, K))
        x_aug = mx.concatenate([pts_re[idx], pts_im[idx]], axis=1)                   # (B, 2K)
        y_aug = x_aug @ H_aug.T + mx.random.normal((B, 2 * N)) * inv_sqrt2          # (B, 2N)
        z_aug = y_aug @ H_aug                                                        # (B, 2K) matched filter
        x_dec_aug = x_aug[:, None, :] * decmask_tiled[None, :, :]                    # (B, K, 2K)
        r_aug = z_aug[:, None, :] - x_dec_aug @ G_aug.T                              # (B, K, 2K)
        oh = mx.broadcast_to(onehot[None], (B, K, K))
        dm = mx.broadcast_to(decmask[None], (B, K, K))
        feat = mx.concatenate([r_aug, oh, dm], axis=2)                               # (B, K, 4K)
        tgt = idx[:, order_mx]
        dist2 = None
        if anchored:
            y_res_aug = y_aug[:, None, :] - x_dec_aug @ H_aug.T                      # (B, K, 2N)
            xhat = (y_res_aug.reshape(B, K, 1, 2 * N) @
                    Wr0_aug.transpose(0, 2, 1)[None]).reshape(B, K, 2)               # (B, K, 2)
            diff = xhat[:, :, None, :] - pts_aug[None, None, :, :]                   # (B, K, Q, 2)
            dist2 = (diff * diff).sum(axis=-1)                                       # (B, K, Q)
        return feat, tgt, dist2

    def loss_fn(jm, feat, tgt, dist2):
        B, Kd, _ = feat.shape
        logits = jm.mlp(feat.reshape(B * Kd, -1)).reshape(B, Kd, Q)
        if anchored:
            logits = -dist2 / mx.exp(jm.logs2)[None, :, None] + jm.gate[None, :, None] * logits
        lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        ce = -mx.take_along_axis(lp, tgt[..., None], axis=-1).squeeze(-1)
        return mx.mean(ce)
    lg = mnn.value_and_grad(jm, loss_fn)
    state = [jm.state, opt.state]
    def step(feat, tgt, dist2):
        loss, grads = lg(jm, feat, tgt, dist2)
        opt.update(jm, grads)
        return loss
    step_c = mx.compile(step, inputs=state, outputs=state)

    for _ in range(steps):
        feat, tgt, dist2 = sample_feat(batch)
        step_c(feat, tgt, dist2)
        mx.eval(jm.state, opt.state)

    feat, tgt, dist2 = sample_feat(eval_sym)
    B, Kd, _ = feat.shape
    logits = jm.mlp(feat.reshape(B * Kd, -1)).reshape(B, Kd, Q)
    if anchored:
        logits = -dist2 / mx.exp(jm.logs2)[None, :, None] + jm.gate[None, :, None] * logits
    ce_bits = _ce_bits_per_user(logits, tgt, Kd, Q)
    R_k = np.log2(Q) - ce_bits
    return float(R_k.sum()), R_k


def train_neural_joint_rate_mlx_fast(H, const=mac.QPSK, steps=4000, batch=4096, width=64, depth=2,
                                     lr=1e-3, eval_sym=40000, seed=0, order=None):
    return _joint_mlx_fast(H, const, steps, batch, width, depth, lr, eval_sym, seed, order, False)


def train_neural_joint_lmmse_rate_mlx_fast(H, const=mac.QPSK, steps=4000, batch=4096, width=64, depth=2,
                                           lr=1e-3, eval_sym=40000, seed=0, order=None):
    return _joint_mlx_fast(H, const, steps, batch, width, depth, lr, eval_sym, seed, order, True)
