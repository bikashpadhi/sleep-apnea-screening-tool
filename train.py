"""
train.py
========
Full training pipeline for the Sleep Apnea detection model.

Features
--------
- WeightedRandomSampler to handle class imbalance
- FocalSeparationLoss (focal + class-separation margin)
- OneCycleLR learning-rate schedule
- Per-epoch Gaussian temporal smoothing on validation probabilities
- Threshold sweep on smoothed probs (optimises F1 subject to spec ≥ 0.55)
- EMA-tracked best model with patience-based early stopping
- Automatic training-curve plots saved to models/

Usage
-----
    python src/train.py

Key environment variables (all have sensible defaults):
    AUGMENTED_DIR   path to augmented training data
    PROCESSED_DIR   path to processed val/test data
    MODELS_DIR      where checkpoints + plots are saved
"""

import os
import numpy as np
import torch
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import (
    f1_score, recall_score, precision_score,
    confusion_matrix, classification_report,
)
from scipy.ndimage import gaussian_filter1d

from model import ApneaModel, FocalSeparationLoss
from preprocessing import UCDApneaDataset, preprocess_ucd
from augmentation import normalize, augment_batch

# ─────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────
AUGMENTED_DIR = os.getenv("AUGMENTED_DIR", "data/augmented")
PROCESSED_DIR = os.getenv("PROCESSED_DIR", "data/processed")
MODELS_DIR    = os.getenv("MODELS_DIR",    "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# ─────────────────────────────────────────
# HYPER-PARAMETERS
# ─────────────────────────────────────────
SEED          = 42
BATCH_SIZE    = 32
EPOCHS        = 70
LR_MAX        = 1e-4
WEIGHT_DECAY  = 3e-3
PATIENCE      = 15
SPEC_FLOOR    = 0.55   # minimum acceptable specificity during threshold sweep
SMOOTH_SIGMA  = 1.5    # Gaussian σ in windows (= ±90s context)
SMOOTH_TRUNC  = 3.0    # kernel truncation multiplier

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# ─────────────────────────────────────────
# TEMPORAL SMOOTHING
# ─────────────────────────────────────────

def smooth_probs(
    probs:   np.ndarray,
    rec_ids: np.ndarray | None = None,
    sigma:   float = SMOOTH_SIGMA,
    truncate: float = SMOOTH_TRUNC,
) -> np.ndarray:
    """
    Apply Gaussian temporal smoothing per recording.

    Smoothing is applied independently within each recording to prevent
    probability leakage across patient boundaries. When rec_ids is None,
    the entire sequence is treated as a single recording.

    Parameters
    ----------
    probs   : (N,)  raw sigmoid probabilities in temporal order
    rec_ids : (N,)  integer recording IDs, or None
    sigma   : Gaussian σ in windows
    truncate: kernel truncation in units of σ

    Returns
    -------
    smoothed : (N,)  clipped to [0, 1]
    """
    if rec_ids is None:
        return np.clip(
            gaussian_filter1d(probs, sigma=sigma, truncate=truncate), 0.0, 1.0
        )

    smoothed = np.zeros_like(probs)
    for rid in np.unique(rec_ids):
        mask = rec_ids == rid
        seg  = probs[mask]
        smoothed[mask] = (
            seg
            if len(seg) < 3
            else np.clip(
                gaussian_filter1d(seg, sigma=sigma, truncate=truncate), 0.0, 1.0
            )
        )
    return smoothed


# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────

def sweep_thresholds(
    probs:  np.ndarray,
    labels: np.ndarray,
    spec_floor: float = SPEC_FLOOR,
) -> dict:
    """
    Find threshold that maximises F1 subject to specificity ≥ spec_floor.

    Returns dict with keys: f1, thresh, prec, rec, spec
    """
    best = {"f1": 0.0, "thresh": 0.50, "prec": 0.0, "rec": 0.0, "spec": 0.0}
    for t in np.linspace(0.20, 0.80, 61):
        preds = (probs >= t).astype(int)
        cm    = confusion_matrix(labels, preds, labels=[0, 1])
        if cm.shape != (2, 2):
            continue
        tn, fp, fn, tp = cm.ravel()
        spec = tn / (tn + fp + 1e-9)
        if spec < spec_floor:
            continue
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best["f1"]:
            best.update({
                "f1":    f1,
                "thresh": t,
                "prec":  precision_score(labels, preds, zero_division=0),
                "rec":   recall_score(labels, preds, zero_division=0),
                "spec":  spec,
            })
    return best


class EMA:
    """Exponential Moving Average tracker."""

    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self.value = None

    def update(self, x: float) -> float:
        self.value = x if self.value is None else (
            self.alpha * x + (1 - self.alpha) * self.value
        )
        return self.value


# ─────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────

def run_validation(model, loader, device):
    """
    Collect raw probabilities and labels from val_loader.
    Loader MUST have shuffle=False to preserve temporal order.

    Returns
    -------
    probs   : (N,) np.ndarray of sigmoid probabilities
    labels  : (N,) np.ndarray of ground-truth labels
    rec_ids : (N,) np.ndarray of recording IDs, or None
    """
    model.eval()
    probs_list, labels_list, rec_id_list = [], [], []

    with torch.no_grad():
        for batch_idx, (ecg, spo2, lbl) in enumerate(loader):
            ecg  = normalize(ecg.to(device))
            spo2 = normalize(spo2.to(device))
            p    = torch.sigmoid(model(ecg, spo2)).cpu().numpy()
            probs_list.extend(p)
            labels_list.extend(lbl.numpy())

            if loader.dataset.rec_ids is not None:
                start = batch_idx * loader.batch_size
                end   = min(start + loader.batch_size, len(loader.dataset))
                rec_id_list.extend(loader.dataset.rec_ids[start:end])

    probs   = np.array(probs_list)
    labels  = np.array(labels_list)
    rec_ids = np.array(rec_id_list) if rec_id_list else None
    return probs, labels, rec_ids


# ─────────────────────────────────────────
# MAIN TRAINING LOOP
# ─────────────────────────────────────────

def train():
    # ── Load data ──────────────────────────────────────
    print("Loading data…")
    X_tr_ecg   = np.load(os.path.join(AUGMENTED_DIR, "X_train_ecg.npy"))
    X_tr_spo2  = np.load(os.path.join(AUGMENTED_DIR, "X_train_spo2.npy"))
    y_tr       = np.load(os.path.join(AUGMENTED_DIR, "y_train.npy"))

    X_vl_ecg   = np.load(os.path.join(PROCESSED_DIR,  "X_val_ecg.npy"))
    X_vl_spo2  = np.load(os.path.join(PROCESSED_DIR,  "X_val_spo2.npy"))
    y_vl       = np.load(os.path.join(PROCESSED_DIR,   "y_val.npy"))

    # Optional: per-recording IDs for val set (enables per-patient smoothing)
    rec_ids_path = os.path.join(PROCESSED_DIR, "rec_ids_val.npy")
    val_rec_ids  = (np.load(rec_ids_path)
                    if os.path.exists(rec_ids_path) else None)
    if val_rec_ids is not None:
        print(f"  val rec_ids: {len(np.unique(val_rec_ids))} unique patients")
    else:
        print("  rec_ids_val.npy not found — smoothing over full val sequence")

    # ── Preprocessing ──────────────────────────────────
    X_tr_ecg,  X_tr_spo2  = preprocess_ucd(X_tr_ecg,  X_tr_spo2)
    X_vl_ecg,  X_vl_spo2  = preprocess_ucd(X_vl_ecg,  X_vl_spo2)

    # ── Class balance ──────────────────────────────────
    n_pos  = int(y_tr.sum())
    n_neg  = int((1 - y_tr).sum())
    ratio  = n_neg / max(n_pos, 1)
    pw     = float(min(ratio, 3.0))
    print(f"Train  pos={n_pos}  neg={n_neg}  ratio={ratio:.2f}  pos_weight={pw:.3f}")

    # ── WeightedRandomSampler ──────────────────────────
    cw      = np.where(y_tr == 1, ratio, 1.0)
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(cw), num_samples=len(y_tr), replacement=True
    )

    # ── DataLoaders ────────────────────────────────────
    train_loader = DataLoader(
        UCDApneaDataset(X_tr_ecg, X_tr_spo2, y_tr),
        batch_size=BATCH_SIZE, sampler=sampler, drop_last=True,
    )
    val_loader = DataLoader(
        UCDApneaDataset(X_vl_ecg, X_vl_spo2, y_vl, rec_ids=val_rec_ids),
        batch_size=BATCH_SIZE, shuffle=False,   # ← MUST stay False
    )

    # ── Model, loss, optimiser ─────────────────────────
    model     = ApneaModel(hidden_dim=128, gru_layers=2).to(device)
    n_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    criterion = FocalSeparationLoss(pos_weight=pw, gamma=2.0,
                                    margin=0.35, sep_weight=0.4)
    optimizer = optim.AdamW(model.parameters(), lr=LR_MAX,
                            weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR_MAX, epochs=EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.20, anneal_strategy="cos",
        div_factor=10.0, final_div_factor=1000.0,
    )

    # ── Training state ─────────────────────────────────
    best_f1_global  = 0.0
    pat_counter     = 0
    saved_thresh    = 0.50
    model_save_path = os.path.join(MODELS_DIR, "best_model.pth")
    f1_ema          = EMA(alpha=0.3)

    history = {
        "loss": [], "f1_smooth": [], "f1_raw": [],
        "f1_ema": [], "spec": [], "prec": [], "rec": [],
    }

    print(f"\nTraining  (Gaussian σ={SMOOTH_SIGMA} | pos_weight={pw:.2f} | "
          f"focal_γ=2.0 | margin=0.35 | sep_w=0.4)\n")
    print(f"{'Ep':>4} {'Loss':>8} {'F1_sm':>7} {'F1_raw':>7} {'EMA':>7} "
          f"{'Prec':>7} {'Rec':>7} {'Spec':>7} {'Thr':>5}")
    print("─" * 72)

    for epoch in range(EPOCHS):
        model.train()
        ep_loss = 0.0

        for ecg, spo2, labels in train_loader:
            ecg    = normalize(ecg.to(device))
            spo2   = normalize(spo2.to(device))
            labels = labels.to(device)
            ecg, spo2 = augment_batch(ecg, spo2)

            optimizer.zero_grad()
            out  = model(ecg, spo2)
            loss = criterion(out, labels)

            if not torch.isnan(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                ep_loss += loss.item()

        # ── Validation ─────────────────────────────────
        probs_raw, labels_vl, rec_ids_vl = run_validation(model, val_loader, device)

        best_raw = sweep_thresholds(probs_raw, labels_vl)
        probs_sm = smooth_probs(probs_raw, rec_ids=rec_ids_vl)
        best_sm  = sweep_thresholds(probs_sm, labels_vl)

        ema_f1 = f1_ema.update(best_sm["f1"])

        history["loss"].append(ep_loss / len(train_loader))
        history["f1_smooth"].append(best_sm["f1"])
        history["f1_raw"].append(best_raw["f1"])
        history["f1_ema"].append(ema_f1)
        history["spec"].append(best_sm["spec"])
        history["prec"].append(best_sm["prec"])
        history["rec"].append(best_sm["rec"])

        print(f"{epoch+1:>4} {ep_loss/len(train_loader):>8.4f} "
              f"{best_sm['f1']:>7.4f} {best_raw['f1']:>7.4f} {ema_f1:>7.4f} "
              f"{best_sm['prec']:>7.4f} {best_sm['rec']:>7.4f} "
              f"{best_sm['spec']:>7.4f} {best_sm['thresh']:>5.2f}")

        if ema_f1 > best_f1_global:
            best_f1_global = ema_f1
            saved_thresh   = best_sm["thresh"]
            pat_counter    = 0
            torch.save(
                {
                    "epoch":        epoch + 1,
                    "state_dict":   model.state_dict(),
                    "optimizer":    optimizer.state_dict(),
                    "best_f1":      best_sm["f1"],
                    "best_f1_raw":  best_raw["f1"],
                    "ema_f1":       ema_f1,
                    "threshold":    saved_thresh,
                    "smooth_sigma": SMOOTH_SIGMA,
                },
                model_save_path,
            )
            print(f"     Saved  F1_sm={best_sm['f1']:.4f}  "
                  f"EMA={ema_f1:.4f}  thresh={saved_thresh:.2f}")
        else:
            pat_counter += 1
            if pat_counter >= PATIENCE:
                print(f"\nEarly stop at epoch {epoch + 1}")
                break

    print(f"\nBest EMA val F1 (smoothed): {best_f1_global:.4f}")

    # ── Training curves ────────────────────────────────
    ep = range(1, len(history["loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].plot(ep, history["loss"], "b-")
    axes[0].set_title("Train Loss"); axes[0].set_xlabel("Epoch")

    axes[1].plot(ep, history["f1_raw"],    "g--", alpha=0.6, label="Raw F1")
    axes[1].plot(ep, history["f1_smooth"], "g-",  alpha=0.7, label="Smoothed F1")
    axes[1].plot(ep, history["f1_ema"],    "g-",  lw=2.5,    label="EMA(Smooth F1)")
    axes[1].fill_between(ep, history["f1_raw"], history["f1_smooth"],
                         alpha=0.15, color="green", label="Smoothing gain")
    axes[1].legend(fontsize=8); axes[1].set_title("Val F1"); axes[1].set_xlabel("Epoch")

    axes[2].plot(ep, history["spec"], "r-", label="Specificity")
    axes[2].plot(ep, history["rec"],  "b-", label="Recall")
    axes[2].plot(ep, history["prec"], "m-", label="Precision")
    axes[2].legend(fontsize=8)
    axes[2].set_title("Val Metrics (Smoothed)"); axes[2].set_xlabel("Epoch")

    plt.tight_layout()
    plot_path = os.path.join(MODELS_DIR, "training_curves.png")
    plt.savefig(plot_path, dpi=120)
    print(f"Curves saved → {plot_path}")


if __name__ == "__main__":
    train()
