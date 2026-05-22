"""
augmentation.py
===============
Data augmentation for the UCD Sleep Apnea training set.

Three strategies are applied to the minority (apnea) and majority
(normal) classes alike to improve generalisation:

  1. Standard augmentation — Gaussian noise, amplitude scaling, time shift
  2. Mixup synthesis      — convex combination of two random samples
  3. Online augmentation  — applied per-batch during training (in train.py)

Usage
-----
Run as a script to pre-generate and save augmented training arrays:

    python src/augmentation.py

Or import individual functions to apply augmentation on-the-fly.
"""

import os
import numpy as np
import torch

PROCESSED_DIR = os.getenv("PROCESSED_DIR", "data/processed")
AUGMENTED_DIR = os.getenv("AUGMENTED_DIR", "data/augmented")

# Augmentation hyper-parameters
AUG_PER_SAMPLE = 2    # number of augmented copies per original sample
MIXUP_SAMPLES  = None  # if None, defaults to len(X_ecg) at runtime
MIXUP_ALPHA    = 0.4   # Beta distribution concentration parameter


# ─────────────────────────────────────────
# OFFLINE AUGMENTATION FUNCTIONS
# ─────────────────────────────────────────

def augment_signal(ecg: np.ndarray, spo2: np.ndarray):
    """
    Apply random combination of: Gaussian noise, amplitude scaling, time shift.

    Parameters
    ----------
    ecg  : (L_ecg,)  single ECG epoch
    spo2 : (L_spo2,) single SpO2 epoch

    Returns
    -------
    ecg_aug, spo2_aug : augmented copies (same shapes)
    """
    # Gaussian additive noise
    if np.random.rand() < 0.5:
        ecg  = ecg  + np.random.normal(0, 0.01,  size=ecg.shape)
        spo2 = spo2 + np.random.normal(0, 0.003, size=spo2.shape)

    # Amplitude scaling (ECG only — SpO2 is a percentage, scaling is misleading)
    if np.random.rand() < 0.4:
        scale = np.random.uniform(0.9, 1.1)
        ecg   = ecg * scale

    # Time shift (circular roll preserves signal length)
    if np.random.rand() < 0.4:
        shift = np.random.randint(-20, 20)
        ecg   = np.roll(ecg,  shift)
        spo2  = np.roll(spo2, shift)

    return ecg, spo2


def mixup(
    ecg1:  np.ndarray,
    spo21: np.ndarray,
    y1:    float,
    ecg2:  np.ndarray,
    spo22: np.ndarray,
    y2:    float,
    alpha: float = MIXUP_ALPHA,
):
    """
    Mixup data augmentation: interpolate two samples at ratio λ ~ Beta(α, α).

    Parameters
    ----------
    ecg1, spo21, y1 : first sample (signal arrays + scalar label)
    ecg2, spo22, y2 : second sample
    alpha           : Beta distribution parameter

    Returns
    -------
    ecg_mix, spo2_mix, y_mix : mixed sample with soft label
    """
    lam   = np.random.beta(alpha, alpha)
    ecg   = lam * ecg1  + (1 - lam) * ecg2
    spo2  = lam * spo21 + (1 - lam) * spo22
    y_mix = lam * y1    + (1 - lam) * y2
    return ecg, spo2, y_mix


# ─────────────────────────────────────────
# ONLINE (IN-TRAINING) AUGMENTATION
# ─────────────────────────────────────────

def normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Z-score normalise a batch of signals along the time axis.

    Parameters
    ----------
    x   : (B, 1, L) tensor
    eps : small constant for numerical stability

    Returns
    -------
    x_norm : (B, 1, L) normalised tensor
    """
    mean = x.mean(dim=-1, keepdim=True)
    std  = x.std(dim=-1,  keepdim=True).clamp(min=eps)
    return (x - mean) / std


def augment_batch(
    ecg:  torch.Tensor,
    spo2: torch.Tensor,
    noise_std_ecg:  float = 0.005,
    noise_std_spo2: float = 0.002,
) -> tuple:
    """
    Lightweight online augmentation applied per-batch during training.

    Only Gaussian noise is added — enough to regularise without
    distorting the signal beyond clinical plausibility.

    Parameters
    ----------
    ecg  : (B, 1, L_ecg)  batch of ECG epochs on device
    spo2 : (B, 1, L_spo2) batch of SpO2 epochs on device

    Returns
    -------
    ecg_aug, spo2_aug : same shapes, on same device
    """
    ecg  = ecg  + torch.randn_like(ecg)  * noise_std_ecg
    spo2 = spo2 + torch.randn_like(spo2) * noise_std_spo2
    return ecg, spo2


# ─────────────────────────────────────────
# OFFLINE PRE-GENERATION (CLI)
# ─────────────────────────────────────────

def generate_and_save(
    processed_dir: str = PROCESSED_DIR,
    output_dir:    str = AUGMENTED_DIR,
    aug_per_sample: int = AUG_PER_SAMPLE,
    mixup_n: int | None = None,
):
    """
    Load training arrays, generate augmented copies, concatenate, and save.

    Output files mirror the input structure:
        data/augmented/X_train_ecg.npy
        data/augmented/X_train_spo2.npy
        data/augmented/y_train.npy

    Validation / test sets are NOT augmented.
    """
    os.makedirs(output_dir, exist_ok=True)

    X_ecg  = np.load(os.path.join(processed_dir, "X_train_ecg.npy"))
    X_spo2 = np.load(os.path.join(processed_dir, "X_train_spo2.npy"))
    y      = np.load(os.path.join(processed_dir, "y_train.npy"))

    N = len(X_ecg)
    n_mixup = mixup_n if mixup_n is not None else N

    print(f"Original training set: {X_ecg.shape}")

    ecg_aug, spo2_aug, y_aug = [], [], []

    # Standard augmentation
    for i in range(N):
        for _ in range(aug_per_sample):
            e, s = augment_signal(X_ecg[i], X_spo2[i])
            ecg_aug.append(e)
            spo2_aug.append(s)
            y_aug.append(y[i])

    # Mixup synthesis
    for _ in range(n_mixup):
        i1, i2 = np.random.randint(0, N, size=2)
        e, s, yl = mixup(X_ecg[i1], X_spo2[i1], y[i1],
                         X_ecg[i2], X_spo2[i2], y[i2])
        ecg_aug.append(e)
        spo2_aug.append(s)
        y_aug.append(yl)

    ecg_aug  = np.array(ecg_aug,  dtype=np.float32)
    spo2_aug = np.array(spo2_aug, dtype=np.float32)
    y_aug    = np.array(y_aug,    dtype=np.float32)

    X_ecg_final  = np.concatenate([X_ecg,  ecg_aug],  axis=0)
    X_spo2_final = np.concatenate([X_spo2, spo2_aug], axis=0)
    y_final      = np.concatenate([y,      y_aug],    axis=0)

    np.save(os.path.join(output_dir, "X_train_ecg.npy"),  X_ecg_final)
    np.save(os.path.join(output_dir, "X_train_spo2.npy"), X_spo2_final)
    np.save(os.path.join(output_dir, "y_train.npy"),       y_final)

    print(f"Augmented training set: {X_ecg_final.shape}")
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    generate_and_save()
