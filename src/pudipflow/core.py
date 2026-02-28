"""
Author: Yuyang Ren
Email: renyy2022@shanghaitech.edu.cn

"""
from typing import Sequence, Optional
import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.animation as animation

from .unet_pudipflow import UNet3D_PUDIPFlow


def _wrap(x: torch.Tensor) -> torch.Tensor:
    return x - torch.div(x + torch.pi, 2 * torch.pi, rounding_mode="floor") * 2 * torch.pi



def _finite_diff_3d(
    x: torch.Tensor,
    device: str,
    weights4: Optional[Sequence[float]] = None,  # (wT, wFE, wPE, wSPE)
) -> torch.Tensor:
    B, T, FE, PE, SPE = x.shape

    if weights4 is None:
        wT, wFE, wPE, wSPE = 1.0, 1.0, 1.0, 1.0
    else:
        if len(weights4) != 4:
            raise ValueError(f"weights4 must have length 4 (T,FE,PE,SPE), got {len(weights4)}")
        wT, wFE, wPE, wSPE = (float(weights4[0]), float(weights4[1]), float(weights4[2]), float(weights4[3]))

    dc = torch.cat(
        [x[:, 1:] - x[:, :-1], torch.zeros(B, 1, FE, PE, SPE, device=device, dtype=x.dtype)],
        dim=1,
    )

    dfe = torch.cat(
        [x[:, :, 1:] - x[:, :, :-1], torch.zeros(B, T, 1, PE, SPE, device=device, dtype=x.dtype)],
        dim=2,
    )
    dpe = torch.cat(
        [x[:, :, :, 1:] - x[:, :, :, :-1], torch.zeros(B, T, FE, 1, SPE, device=device, dtype=x.dtype)],
        dim=3,
    )
    dspe = torch.cat(
        [x[:, :, :, :, 1:] - x[:, :, :, :, :-1], torch.zeros(B, T, FE, PE, 1, device=device, dtype=x.dtype)],
        dim=4,
    )

    return torch.cat(
        [
            dc.unsqueeze(1) * wT,
            dfe.unsqueeze(1) * wFE,
            dpe.unsqueeze(1) * wPE,
            dspe.unsqueeze(1) * wSPE,
        ],
        dim=1,
    )


def _nrmse(pred: torch.Tensor, ref: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask != 0
    return torch.sqrt(torch.mean((pred[m] - ref[m]) ** 2)) / (ref[m].max() - ref[m].min())


def _make_thumb(img2d, thumb_size: int = 64, cmap: str = "gray", origin: str = "lower") -> np.ndarray:
    a = np.asarray(img2d, dtype=np.float32)

    if origin == "lower":
        a = np.flipud(a)

    p1, p99 = np.nanpercentile(a, [1, 99])
    if not np.isfinite(p1) or not np.isfinite(p99) or (p99 - p1) < 1e-8:
        p1, p99 = np.nanmin(a), np.nanmax(a)
        if not np.isfinite(p1) or not np.isfinite(p99) or (p99 - p1) < 1e-8:
            p1, p99 = 0.0, 1.0

    a = np.clip((a - p1) / (p99 - p1 + 1e-8), 0.0, 1.0)
    cm = plt.get_cmap(cmap)
    rgba = (cm(a) * 255).astype(np.uint8)

    h, w = rgba.shape[:2]
    if h > thumb_size or w > thumb_size:
        sy = max(1, h // thumb_size)
        sx = max(1, w // thumb_size)
        rgba = rgba[::sy, ::sx]

    return rgba


def _plot(history, snapshots=None, fs: int = 12) -> None:
    has_nrmse = ("nrmse" in history) and (len(history["nrmse"]) > 0)
    ncols = 2 if has_nrmse else 1

    fig, axes = plt.subplots(1, ncols, figsize=(fs, fs * 0.35))
    if ncols == 1:
        axes = [axes]

    ax0 = axes[0]
    loss_arr = np.asarray(history["loss"], dtype=np.float64)
    x = np.arange(1, len(loss_arr) + 1)

    ax0.plot(x, loss_arr, linewidth=0.9)
    ax0.set(xlabel="Epoch", ylabel="Loss", title="Training Loss")
    ax0.set_yscale("log")
    ax0.grid(True, alpha=0.3)

    if snapshots:
        for s in snapshots:
            thumb = s.get("thumb_rgba", None)
            if thumb is None:
                continue
            e = int(s["epoch"])
            y = float(s["loss"]) * 1.3

            im = OffsetImage(thumb, zoom=0.6)
            ab = AnnotationBbox(
                im,
                (e, y),
                frameon=True,
                pad=0.2,
                bboxprops=dict(edgecolor="white", linewidth=0.8, alpha=0.9),
            )
            ax0.add_artist(ab)
        ax0.margins(x=0.05, y=0.25)

    if has_nrmse:
        ax1 = axes[1]
        nrmse_arr = np.asarray(history["nrmse"], dtype=np.float64)
        xe = np.arange(1, len(nrmse_arr) + 1)

        ax1.plot(xe, nrmse_arr, linewidth=0.9, color="tab:orange")
        best_idx = int(nrmse_arr.argmin()) + 1
        ax1.axvline(
            best_idx,
            color="red",
            linestyle="--",
            alpha=0.6,
            label=f"Best {nrmse_arr[best_idx-1]:.4f} @ {best_idx}",
        )

        last_idx = int(len(nrmse_arr))
        if last_idx != best_idx:
            ax1.axvline(
                last_idx,
                color="tab:green",
                linestyle="--",
                alpha=0.6,
                label=f"Last {nrmse_arr[last_idx-1]:.4f} @ {last_idx}",
            )
        else:
            ax1.plot(
                [],
                [],
                color="tab:green",
                linestyle="--",
                alpha=0.6,
                label=f"Last {nrmse_arr[last_idx-1]:.4f} @ {last_idx}",
            )
        ax1.set(xlabel="Epoch", ylabel="NRMSE", title="NRMSE vs. Epoch")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

    plt.tight_layout()
    plt.show()


class _TrainingVideoWriter:
    def __init__(
        self,
        out_path: str,
        fps: int = 12,
        dpi: int = 120,
        cmap: str = "gray",
        vmin: float = -150.0,
        vmax: float = 150.0,
    ):
        self.out_path = out_path
        self.fps = int(fps)
        self.dpi = int(dpi)
        self.cmap = cmap
        self.vmin = float(vmin)
        self.vmax = float(vmax)

        self.fig, (self.ax_loss, self.ax_img) = plt.subplots(1, 2, figsize=(10, 4), dpi=self.dpi)

        self.ax_loss.set_title("Training Loss")
        self.ax_loss.set_xlabel("Epoch")
        self.ax_loss.set_ylabel("Loss")
        self.ax_loss.grid(True, alpha=0.3)
        self.ax_loss.set_yscale("log")
        (self.loss_line,) = self.ax_loss.plot([], [], lw=1.2, color="tab:blue")
        (self.loss_dot,) = self.ax_loss.plot([], [], marker="o", ms=3, color="tab:red", lw=0)

        self.ax_img.set_title("Reconstruction Slice")
        self.ax_img.set_axis_off()
        self.im = self.ax_img.imshow(
            np.zeros((8, 8), dtype=np.float32),
            cmap=self.cmap,
            origin="lower",
            vmin=self.vmin,
            vmax=self.vmax,
            animated=False,
        )
        self.cbar = self.fig.colorbar(self.im, ax=self.ax_img, fraction=0.046, pad=0.04)
        self.cbar.set_label("Velocity")

        self.loss_x = []
        self.loss_y = []

        try:
            self.writer = animation.FFMpegWriter(fps=self.fps)
            self._ctx = self.writer.saving(self.fig, self.out_path, dpi=self.dpi)
            self._ctx.__enter__()
        except Exception:
            if not self.out_path.lower().endswith(".gif"):
                self.out_path = self.out_path.rsplit(".", 1)[0] + ".gif"
            self.writer = animation.PillowWriter(fps=self.fps)
            self._ctx = self.writer.saving(self.fig, self.out_path, dpi=self.dpi)
            self._ctx.__enter__()

    def add_frame(self, epoch: int, loss_val: float, img2d) -> None:
        epoch = int(epoch)
        loss_val = float(loss_val)
        img2d = np.asarray(img2d)

        self.loss_x.append(epoch)
        self.loss_y.append(loss_val)

        self.loss_line.set_data(self.loss_x, self.loss_y)
        self.loss_dot.set_data([epoch], [loss_val])

        self.ax_loss.set_xlim(1, max(10, epoch))
        y = np.asarray(self.loss_y, dtype=np.float64)
        y = y[np.isfinite(y) & (y > 0)]
        if y.size >= 2:
            lo = max(np.min(y) * 0.7, np.percentile(y, 5) * 0.7)
            hi = max(np.max(y) * 1.3, np.percentile(y, 95) * 1.3)
            if hi > lo:
                self.ax_loss.set_ylim(lo, hi)

        self.im.set_data(img2d)

        self.fig.tight_layout()
        self.writer.grab_frame()

    def close(self) -> None:
        if getattr(self, "_ctx", None) is not None:
            self._ctx.__exit__(None, None, None)
            self._ctx = None
        plt.close(self.fig)


def _divergence_from_B(v: torch.Tensor, spacing=(1.0, 1.0, 1.0)) -> torch.Tensor:
    """
    v shape = (B, T, FE, PE, SPE)
    spacing = (dFE, dPE, dSPE)  i.e., (dx, dy, dz) in physical units
    returns: (1, T, FE, PE, SPE)
    """
    Bc, T, FE, PE, SPE = v.shape
    dFE, dPE, dSPE = (float(spacing[0]), float(spacing[1]), float(spacing[2]))

    def ddx(a: torch.Tensor) -> torch.Tensor:
        d = torch.roll(a, shifts=-1, dims=2) - a
        d[:, :, -1, :, :] = 0
        return d / max(dFE, 1e-12)

    def ddy(a: torch.Tensor) -> torch.Tensor:
        d = torch.roll(a, shifts=-1, dims=3) - a
        d[:, :, :, -1, :] = 0
        return d / max(dPE, 1e-12)

    def ddz(a: torch.Tensor) -> torch.Tensor:
        d = torch.roll(a, shifts=-1, dims=4) - a
        d[:, :, :, :, -1] = 0
        return d / max(dSPE, 1e-12)

    div = torch.zeros((1, T, FE, PE, SPE), device=v.device, dtype=v.dtype)
    if Bc >= 1: div += ddx(v[0:1])
    if Bc >= 2: div += ddy(v[1:2])
    if Bc >= 3: div += ddz(v[2:3])
    return div


class PUDIPFlow:
    """
    Phase Unwrapping via Deep Image Prior for 4D Flow MRI.
    """

    def __init__(
        self,
        venc: float = 150.0,
        level: int = 4,
        features: int = 64,
        input_depth: int = 32,
        lr: float = 1e-3,
        num_iter: int = 1000,
        tv_weights=(1.0, 1.0, 1.0, 1.0),
        loss_type: str = "l1",
        device: str = "cuda",
        lr_scheduler: str = "cosine",
        div_weight: float = 0.0,
        spacing=(1.0, 1.0, 1.0),
        reshape_mode: str = "bt_as_channel",  # "bt_as_channel" or "bt_as_batch"
    ):
        if loss_type not in ("l1", "l2"):
            raise ValueError(f"loss_type must be 'l1' or 'l2', got '{loss_type}'")
        if reshape_mode not in ("bt_as_channel", "bt_as_batch"):
            raise ValueError("reshape_mode must be 'bt_as_channel' or 'bt_as_batch'")
        self.spacing = tuple(float(s) for s in spacing)
        self.div_weight = float(div_weight)
        self.venc = venc
        self.level = int(level)
        self.features = int(features)
        self.input_depth = int(input_depth)
        self.lr = float(lr)
        self.num_iter = int(num_iter)
        self.tv_weights = tuple(float(w) for w in tv_weights)
        self.loss_type = loss_type
        self.device = device
        self.lr_scheduler_type = lr_scheduler
        self.reshape_mode = reshape_mode

    @staticmethod
    def _broadcast_to_wrapped(arr, wrapped_shape, name: str = "array"):
        if arr is None:
            return None

        a = np.asarray(arr)
        if a.ndim > 5:
            raise ValueError(f"{name} must have <= 5 dims, got shape {a.shape}")

        while a.ndim < 5:
            a = a[None, ...]

        Nv, Nt, FE, PE, SPE = wrapped_shape
        if tuple(a.shape[-3:]) != (FE, PE, SPE):
            raise ValueError(
                f"{name} spatial dims must be (FE,PE,SPE)=({FE},{PE},{SPE}), got {a.shape[-3:]}"
            )

        inNv, inNt = a.shape[0], a.shape[1]
        if inNv not in (1, Nv):
            raise ValueError(f"{name} Nv dim must be 1 or {Nv}, got {inNv} (shape {a.shape})")
        if inNt not in (1, Nt):
            raise ValueError(f"{name} Nt dim must be 1 or {Nt}, got {inNt} (shape {a.shape})")

        a = np.broadcast_to(a, (Nv, Nt, FE, PE, SPE)).copy()
        return a
    def _as_venc_tensor(self, venc, B: int, device: str, dtype=torch.float32) -> torch.Tensor:
        """
        Return venc as a torch.Tensor broadcastable to (B, T, FE, PE, SPE) via (B,1,1,1,1).
        Accepts: float/int, np scalar, (B,), or already-broadcastable shapes.
        """
        if isinstance(venc, (float, int, np.floating, np.integer)):
            v = torch.tensor(float(venc), device=device, dtype=dtype).view(1, 1, 1, 1, 1)
            v = v.expand(B, 1, 1, 1, 1)  # per-B, same value
            return v

        if isinstance(venc, np.ndarray):
            v = torch.as_tensor(venc, device=device, dtype=dtype)
        elif torch.is_tensor(venc):
            v = venc.to(device=device, dtype=dtype)
        else:
            # generic sequence
            v = torch.as_tensor(np.asarray(venc), device=device, dtype=dtype)

        # now v is tensor
        if v.ndim == 1:
            if v.shape[0] != B:
                raise ValueError(f"venc vector must have shape (B,), got {tuple(v.shape)} with B={B}")
            v = v.view(B, 1, 1, 1, 1)
            return v

        # allow already broadcastable, e.g. (B,1,1,1,1) or (1,1,1,1,1)
        # normalize to 5D
        while v.ndim < 5:
            v = v.unsqueeze(-1)
        if v.ndim != 5:
            raise ValueError(f"venc must be scalar, (B,), or <=5D broadcastable to (B,1,1,1,1); got {tuple(v.shape)}")

        # check B dim
        if v.shape[0] not in (1, B):
            raise ValueError(f"venc first dim must be 1 or B={B}, got {v.shape[0]} (shape {tuple(v.shape)})")
        # check other dims are broadcastable (should be 1)
        for d in v.shape[1:]:
            if d not in (1,):
                raise ValueError(f"venc must be broadcastable as (B,1,1,1,1); got shape {tuple(v.shape)}")

        if v.shape[0] == 1:
            v = v.expand(B, 1, 1, 1, 1)
        return v


    def _venc_float_for_showv(self, venc_t_B1111: torch.Tensor, showv: int) -> float:
        """For plotting/video limits: pick venc of the displayed velocity component."""
        showv = int(showv)
        showv = max(0, min(showv, venc_t_B1111.shape[0] - 1))
        return float(venc_t_B1111[showv, 0, 0, 0, 0].detach().cpu().item())
    def _crop_to_mask(self, wrapped, weight, segmask, gt, margin=2):
        idx = np.where(weight[0, 0] == 1)
        if len(idx[0]) == 0:
            FE, PE, SPE = wrapped.shape[2:]
            return wrapped, weight, segmask, gt, (0, FE, 0, PE, 0, SPE)

        if isinstance(margin, (int, np.integer)):
            margin = (int(margin),) * 3
        else:
            margin = tuple(int(x) for x in margin)
            assert len(margin) == 3

        FE, PE, SPE = wrapped.shape[2:]
        dims = (FE, PE, SPE)

        bounds = []
        for i in range(3):
            lo = int(idx[i].min())
            hi = int(idx[i].max()) + 1 
            m = margin[i]

            lo = max(0, lo - m)
            hi = min(dims[i], hi + m)
            bounds.append((lo, hi))

        bbox = tuple(v for b in bounds for v in b)
        s = (..., slice(*bounds[0]), slice(*bounds[1]), slice(*bounds[2]))

        out_seg = segmask[s] if segmask is not None else None
        out_gt  = gt[s] if gt is not None else None
        return wrapped[s], weight[s], out_seg, out_gt, bbox

    def _pad_spatial(self, *arrays):
        ref = arrays[0]
        FE, PE, SPE = ref.shape[2], ref.shape[3], ref.shape[4]

        pads = [(0, 0), (0, 0)]
        for size in (FE, PE, SPE):
            target = ((size + 2**self.level - 1) // 2**self.level) * 2**self.level
            if size < 2**self.level:
                pl = (target - size) // 2
                pr = target - size - pl
                pads.append((pl, pr))
            else:
                pads.append((0, 0))

        results = tuple(
            np.pad(a, pads, mode="constant", constant_values=0) if a is not None else None
            for a in arrays
        )
        return (*results, pads)

    @staticmethod
    def _unpad(tensor: torch.Tensor, pads, padded_shape):
        _, _, FE, PE, SPE = padded_shape
        sl = []
        for dim_size, (pl, pr) in zip((FE, PE, SPE), pads[2:]):
            end = dim_size - pr if pr > 0 else dim_size
            sl.append(slice(pl, end))
        return tensor[:, :, sl[0], sl[1], sl[2]]

    def _to_net_layout(self, x_BT: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """
        x_BT: (B, T, FE, PE, SPE)
        returns:
          x_net: tensor to UNet
          meta: dict for restoring to (B,T,...)
        """
        B, T, FE, PE, SPE = x_BT.shape
        meta = {"B": B, "T": T}

        if self.reshape_mode == "bt_as_channel":
            x_net = x_BT.reshape(B * T, FE, PE, SPE).unsqueeze(0)  # (1, B*T, FE,PE,SPE)
        else:  # bt_as_batch
            x_net = x_BT  # (B, T, FE,PE,SPE)

        return x_net, meta

    def _from_net_layout(self, x_net: torch.Tensor, meta: dict) -> torch.Tensor:
        """
        x_net -> (B, T, FE, PE, SPE)
        """
        B, T = int(meta["B"]), int(meta["T"])

        if self.reshape_mode == "bt_as_channel":
            if x_net.shape[0] != 1:
                raise ValueError(f"Expected batch=1 in bt_as_channel mode, got {x_net.shape}")
            _, C, FE, PE, SPE = x_net.shape
            if C != B * T:
                raise ValueError(f"Expected C=B*T={B*T}, got C={C}")
            x_BT = x_net.squeeze(0).reshape(B, T, FE, PE, SPE)
        else:  # bt_as_batch
            x_BT = x_net
            if x_BT.shape[0] != B or x_BT.shape[1] != T:
                raise ValueError(f"Expected (B,T)=({B},{T}), got {tuple(x_BT.shape[:2])}")

        return x_BT

    def run(
        self,
        wrapped_data,
        weightmask,
        segmask=None,
        gt=None,
        plot: bool = True,
        snapshot_every: int = 200,
        max_snapshots: int = 16,
        showv: int = 0,
        showt: int = 3,
        showz: int = 10,
        snapshot_cmap: str = "gray",
        plot_figsize: int = 12,
        save_video: bool = True,
        video_path: str = "pudipflow_training.mp4",
        video_every: int = 20,
        video_fps: int = 12,
        video_dpi: int = 120,
        video_cmap: str = "gray",
    ):
        device = self.device
        original_shape = wrapped_data.shape
        if len(original_shape) != 5:
            raise ValueError(f"wrapped_data must be (Nv, Nt, FE, PE, SPE), got {original_shape}")

        weightmask = self._broadcast_to_wrapped(weightmask, original_shape, name="weightmask")
        segmask = self._broadcast_to_wrapped(segmask, original_shape, name="segmask")
        gt = self._broadcast_to_wrapped(gt, original_shape, name="gt")

        eval_mode = (gt is not None) and (segmask is not None)
        if (gt is None) != (segmask is None):
            raise ValueError("gt and segmask must be provided together (or both omitted).")

        wrapped_data, weightmask, segmask, gt, bbox = self._crop_to_mask(
            wrapped_data, weightmask, segmask, gt
        )
        minx, maxx, miny, maxy, minz, maxz = bbox

        wrapped_data, weightmask, segmask, gt, pads = self._pad_spatial(
            wrapped_data, weightmask, segmask, gt
        )

        # BT layout (B=Nv, T=Nt) for all physics/loss computations
        B, T, FE, PE, SPE = wrapped_data.shape
        venc_t_B1111 = self._as_venc_tensor(self.venc, B=B, device=device, dtype=torch.float32)
        weight_t_BT = torch.as_tensor(weightmask, dtype=torch.float32, device=device)
        weight_t_BT = weight_t_BT / (weight_t_BT.max() + 1e-12)
        wrapped_t_BT = torch.as_tensor(wrapped_data, dtype=torch.float32, device=device)

        if eval_mode:
            gt_t_BT = torch.as_tensor(gt, dtype=torch.float32, device=device)
            seg_t_BT = torch.as_tensor(segmask, dtype=torch.float32, device=device)

        # Build target finite-differences in BT space (no BT-merge artifacts)
        fd_target = _wrap(_finite_diff_3d(wrapped_t_BT, device, self.tv_weights))

        # Net layout selection
        wrapped_net, meta = self._to_net_layout(wrapped_t_BT)
        out_channels = int(wrapped_net.shape[1])  # either T or (B*T)

        model = UNet3D_PUDIPFlow(self.input_depth, out_channels, self.features).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        if self.lr_scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.num_iter, eta_min=self.lr * 0.01
            )
        elif self.lr_scheduler_type == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=max(1, self.num_iter // 4), gamma=0.5
            )
        else:
            scheduler = None

        scaler = GradScaler()

        # z batch must match net batch
        z = torch.randn(wrapped_net.shape[0], self.input_depth, FE, PE, SPE, device=device) * 0.1

        history = {"loss": []}
        if eval_mode:
            history["nrmse"] = []
            best_nrmse, best_epoch = float("inf"), 0

        snapshots = []
        snapshot_every = int(snapshot_every) if snapshot_every else 0
        max_snapshots = int(max_snapshots) if max_snapshots else 0

        video_writer = None
        video_every = int(video_every) if video_every else 0
        if save_video:
            venc_show = self._venc_float_for_showv(venc_t_B1111, showv=showv)
            video_writer = _TrainingVideoWriter(
                out_path=video_path,
                fps=video_fps,
                dpi=video_dpi,
                cmap=video_cmap,
                vmin=-venc_show,
                vmax=venc_show,
            )

        model.train()
        pbar = tqdm(range(1, self.num_iter + 1), total=self.num_iter)

        try:
            for epoch in pbar:
                optimizer.zero_grad(set_to_none=True)

                with autocast():
                    output_net = model(z)  # either (1,B*T,...) or (B,T,...)
                    output_BT = self._from_net_layout(output_net, meta)  # always (B,T,...)

                    fd_pred = _finite_diff_3d(output_BT, device, self.tv_weights)

                    residual = _wrap(fd_pred - fd_target) * weight_t_BT[:, None]

                    if self.loss_type == "l1":
                        loss_data = torch.sum(torch.abs(residual))
                    else:
                        loss_data = torch.sqrt(torch.sum(residual ** 2))

                    if self.div_weight > 0:
                        rec_BT = (output_BT + _wrap(wrapped_t_BT - output_BT)) / torch.pi * venc_t_B1111
                        div = _divergence_from_B(rec_BT, spacing=self.spacing)
                        div_w = weight_t_BT[0:1] if weight_t_BT.shape[0] >= 1 else 1.0

                        if self.loss_type == "l1":
                            loss_div = torch.sum(torch.abs(div) * div_w)
                        else:
                            loss_div = torch.sqrt(torch.sum((div * div_w) ** 2))
                        loss = loss_data + self.div_weight * loss_div
                    else:
                        loss = loss_data

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None:
                    scheduler.step()

                loss_val = float(loss.item())
                history["loss"].append(loss_val)

                with torch.no_grad():
                    rec_BT = (output_BT + _wrap(wrapped_t_BT - output_BT)) / torch.pi * venc_t_B1111

                    if eval_mode:
                        nrmse_val = float(_nrmse(rec_BT, gt_t_BT, seg_t_BT).item())
                        history["nrmse"].append(nrmse_val)
                        if nrmse_val < best_nrmse:
                            best_nrmse, best_epoch = nrmse_val, epoch
                        pbar.set_description(
                            f"Epoch {epoch} | Loss {loss_val:.2e} | NRMSE {nrmse_val:.4f}"
                        )
                    else:
                        pbar.set_description(f"Epoch {epoch} | Loss {loss_val:.2e}")

                    need_snapshot = plot and (snapshot_every > 0) and (
                        (epoch % snapshot_every == 0) or (epoch == 1) or (epoch == self.num_iter)
                    )
                    need_video = (video_writer is not None) and (video_every > 0) and (
                        (epoch % video_every == 0) or (epoch == 1) or (epoch == self.num_iter)
                    )

                    if need_snapshot or need_video:
                        rec_unpad = self._unpad(rec_BT, pads, (B, T, FE, PE, SPE))

                        full = torch.zeros(original_shape, dtype=rec_unpad.dtype, device=device)
                        full[..., minx:maxx, miny:maxy, minz:maxz] = rec_unpad

                        frame_np = full[showv, showt, :, :, showz].detach().cpu().numpy()

                        if need_snapshot:
                            thumb = _make_thumb(frame_np, thumb_size=72, cmap=snapshot_cmap)
                            snapshots.append({"epoch": epoch, "loss": loss_val, "thumb_rgba": thumb})

                            if max_snapshots and len(snapshots) > max_snapshots:
                                keep_n = int(max_snapshots)
                                idx = np.linspace(0, len(snapshots) - 1, keep_n).round().astype(int)
                                snapshots = [snapshots[i] for i in idx]

                        if need_video:
                            video_writer.add_frame(epoch, loss_val, frame_np)

        finally:
            if video_writer is not None:
                video_writer.close()
                print(f"[PUDIPFlow] Saved training video to: {video_writer.out_path}")

        with torch.no_grad():
            output_net = model(z)
            output_BT = self._from_net_layout(output_net, meta)
            rec_BT = (output_BT + _wrap(wrapped_t_BT - output_BT)) / torch.pi * venc_t_B1111
            rec_BT = self._unpad(rec_BT, pads, (B, T, FE, PE, SPE))

        result = torch.zeros(original_shape, dtype=rec_BT.dtype, device=device)
        result[..., minx:maxx, miny:maxy, minz:maxz] = rec_BT
        result_np = result.detach().cpu().numpy()

        if eval_mode:
            print(f"[PUDIPFlow] Best NRMSE: {best_nrmse:.4f} @ epoch {best_epoch}")

        if plot:
            _plot(history, snapshots=snapshots, fs=plot_figsize)

        return result_np, history