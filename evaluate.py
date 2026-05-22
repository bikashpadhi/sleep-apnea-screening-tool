"""
evaluate.py
===========
Evaluation script for the trained Sleep Apnea detection model.

Runs evaluation on the held-out test set (or validation set) and
produces:
  - Classification report (precision / recall / F1 per class)
  - Confusion matrix (numeric + saved PNG)
  - Side-by-side comparison: with vs without Gaussian temporal smoothing

Usage
-----
    python src/evaluate.py                  # evaluates on test set
    python src/evaluate.py --split val      # evaluates on val set

Environment variables
---------------------
    PROCESSED_DIR  path to processed data (X_test_ecg.npy, etc.)
    MODELS_DIR     directory containing best_model.pth
    RESULTS_DIR    where plots and metrics are saved
"""

import os
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
    roc_auc_score, average_precision_score,
)
from scipy.ndimage import gaussian_filter1d

from model import ApneaModel
from preprocessing import UCDApneaDataset, preprocess_ucd
from augmentation import normalize
from torch.utils.data import DataLoader

# ─────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────
PROCESSED_DIR = os.getenv("PROCESSED_DIR", "data/processed")
MODELS_DIR    = os.getenv("MODELS_DIR",    "models")
RESULTS_DIR   = os.getenv("RESULTS_DIR",   "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

BATCH_SIZE = 64
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────
# SMOOTHING
# ─────────────────────────────────────────

def smooth_probs(
    probs:    np.ndarray,
    rec_ids:  np.ndarray | None,
    sigma:    float,
    truncate: float = 3.0,
) -> np.ndarray:
    """Apply Gaussian smoothing per recording. See train.py for rationale."""
    if rec_ids is None:
        return np.clip(gaussian_filter1d(probs, sigma=sigma, truncate=truncate), 0.0, 1.0)
    smoothed = np.zeros_like(probs)
    for rid in np.unique(rec_ids):
        mask = rec_ids == rid
        seg  = probs[mask]
        smoothed[mask] = (
            seg if len(seg) < 3
            else np.clip(gaussian_filter1d(seg, sigma=sigma, truncate=truncate), 0.0, 1.0)
        )
    return smoothed


# ─────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────

def get_probabilities(model, loader) -> tuple:
    """
    Run model on loader and collect probabilities.

    Returns
    -------
    probs   : (N,)
    labels  : (N,)
    rec_ids : (N,) or None
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

    return (np.array(probs_list),
            np.array(labels_list),
            np.array(rec_id_list) if rec_id_list else None)


# ─────────────────────────────────────────
# METRICS DISPLAY
# ─────────────────────────────────────────

def print_metrics(probs: np.ndarray, labels: np.ndarray, thresh: float, title: str):
    """Print a formatted metrics block for a given threshold."""
    preds = (probs >= thresh).astype(int)
    cm    = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    f1   = f1_score(labels,         preds, zero_division=0)
    prec = precision_score(labels,  preds, zero_division=0)
    rec  = recall_score(labels,     preds, zero_division=0)
    spec = tn / (tn + fp + 1e-9)

    try:
        auroc = roc_auc_score(labels, probs)
        auprc = average_precision_score(labels, probs)
    except ValueError:
        auroc = auprc = float("nan")

    print(f"\n{'─'*50}")
    print(f"  {title}")
    print(f"{'─'*50}")
    print(f"  F1 Score     : {f1:.4f}")
    print(f"  Precision    : {prec:.4f}")
    print(f"  Recall       : {rec:.4f}")
    print(f"  Specificity  : {spec:.4f}")
    print(f"  AUROC        : {auroc:.4f}")
    print(f"  AUPRC        : {auprc:.4f}")
    print(f"  TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    print(f"\n{classification_report(labels, preds, zero_division=0)}")
    return cm, f1


def save_confusion_matrix(cm: np.ndarray, save_path: str, title: str = "Confusion Matrix"):
    """Save a styled confusion matrix as a PNG."""
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    classes = ["Normal (0)", "Apnea (1)"]
    ax.set(
        xticks=np.arange(2), yticks=np.arange(2),
        xticklabels=classes, yticklabels=classes,
        ylabel="True Label", xlabel="Predicted Label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"Confusion matrix saved → {save_path}")


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

def evaluate(split: str = "test"):
    assert split in ("val", "test"), "split must be 'val' or 'test'"

    # ── Load model ─────────────────────────────────────
    ckpt_path = os.path.join(MODELS_DIR, "best_model.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found at {ckpt_path}. Run train.py first."
        )

    ckpt  = torch.load(ckpt_path, map_location=device)
    model = ApneaModel(hidden_dim=128, gru_layers=2).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    thresh       = ckpt["threshold"]
    smooth_sigma = ckpt.get("smooth_sigma", 1.5)
    print(f"Loaded epoch {ckpt['epoch']}  |  "
          f"val F1={ckpt['best_f1']:.4f}  |  "
          f"threshold={thresh:.2f}  |  σ={smooth_sigma}")

    # ── Load data ──────────────────────────────────────
    X_ecg  = np.load(os.path.join(PROCESSED_DIR, f"X_{split}_ecg.npy"))
    X_spo2 = np.load(os.path.join(PROCESSED_DIR, f"X_{split}_spo2.npy"))
    y      = np.load(os.path.join(PROCESSED_DIR, f"y_{split}.npy"))

    rec_ids_path = os.path.join(PROCESSED_DIR, f"rec_ids_{split}.npy")
    rec_ids      = np.load(rec_ids_path) if os.path.exists(rec_ids_path) else None

    X_ecg, X_spo2 = preprocess_ucd(X_ecg, X_spo2)

    loader = DataLoader(
        UCDApneaDataset(X_ecg, X_spo2, y, rec_ids=rec_ids),
        batch_size=BATCH_SIZE, shuffle=False,
    )

    # ── Inference ──────────────────────────────────────
    probs_raw, labels, rec_ids_arr = get_probabilities(model, loader)
    probs_sm = smooth_probs(probs_raw, rec_ids_arr, sigma=smooth_sigma)

    # ── Results ────────────────────────────────────────
    print(f"\n{'═'*50}")
    print(f"  EVALUATION ON {split.upper()} SET")
    print(f"{'═'*50}")

    cm_sm,  f1_sm  = print_metrics(probs_sm,  labels, thresh,
                                    f"WITH Gaussian smoothing (σ={smooth_sigma})")
    cm_raw, f1_raw = print_metrics(probs_raw, labels, thresh,
                                    "WITHOUT smoothing (baseline)")

    print(f"\nSmoothing net gain: {f1_sm - f1_raw:+.4f} F1 points")

    save_confusion_matrix(
        cm_sm,
        os.path.join(RESULTS_DIR, f"confusion_matrix_{split}_smoothed.png"),
        title=f"Confusion Matrix — {split} (smoothed)",
    )
    save_confusion_matrix(
        cm_raw,
        os.path.join(RESULTS_DIR, f"confusion_matrix_{split}_raw.png"),
        title=f"Confusion Matrix — {split} (raw)",
    )

    # ── Save numeric results ───────────────────────────
    results_txt = os.path.join(RESULTS_DIR, f"metrics_{split}.txt")
    preds_sm    = (probs_sm >= thresh).astype(int)
    with open(results_txt, "w") as f:
        f.write(f"Split         : {split}\n")
        f.write(f"Epoch         : {ckpt['epoch']}\n")
        f.write(f"Threshold     : {thresh:.4f}\n")
        f.write(f"Smooth sigma  : {smooth_sigma}\n\n")
        f.write(classification_report(labels, preds_sm, zero_division=0))
    print(f"Metrics saved → {results_txt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["val", "test"])
    args = parser.parse_args()
    evaluate(args.split)
