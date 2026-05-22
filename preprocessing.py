"""
preprocessing.py
================
Handles all data loading and preprocessing for the UCD Sleep Apnea dataset.

Pipeline:
    1. Rename .rec files to .edf
    2. Extract ECG + SpO2 signals from EDF files using MNE
    3. Parse respiratory-event annotation (.txt) files
    4. Create per-minute (60s) binary-labeled epochs
    5. Split data patient-wise (17 train / 4 val / 4 test)
    6. Save processed arrays to disk as .npy files
"""

import os
import glob
import numpy as np
import mne
from datetime import datetime, timedelta
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
from torch.utils.data import Dataset
import torch

mne.set_log_level("ERROR")

# ─────────────────────────────────────────
# PATHS  (override via CLI or environment)
# ─────────────────────────────────────────
RAW_DIR       = os.getenv("RAW_DIR",       "data/raw")
PROCESSED_DIR = os.getenv("PROCESSED_DIR", "data/processed")
AUGMENTED_DIR = os.getenv("AUGMENTED_DIR", "data/augmented")

TARGET_ECG  = "ECG"
TARGET_SPO2 = "SpO2"
EPOCH_SEC   = 60          # window length in seconds
ECG_FS      = 128         # ECG sampling frequency (Hz)
SPO2_FS     = 8           # SpO2 sampling frequency (Hz)
RANDOM_SEED = 42


# ─────────────────────────────────────────
# STEP 0 — Rename .rec → .edf
# ─────────────────────────────────────────
def rename_rec_to_edf(raw_dir: str = RAW_DIR) -> int:
    """
    Rename every *.rec file in raw_dir to *.edf in-place.

    Returns
    -------
    int : number of files renamed
    """
    rec_files = glob.glob(os.path.join(raw_dir, "*.rec"))
    count = 0
    for old in rec_files:
        new = old[:-4] + ".edf"
        os.rename(old, new)
        count += 1
    print(f"Renamed {count} .rec → .edf files.")
    return count


# ─────────────────────────────────────────
# STEP 1 — Single-patient extraction
# ─────────────────────────────────────────
def process_single_patient(edf_path: str, txt_path: str):
    """
    Load one patient's EDF + annotation file and return per-minute epochs.

    Parameters
    ----------
    edf_path : str  Path to the .edf polysomnography file.
    txt_path : str  Path to the matching _respevt.txt annotation file.

    Returns
    -------
    ecg_epochs  : np.ndarray  shape (n_epochs, ECG_FS * EPOCH_SEC)
    spo2_epochs : np.ndarray  shape (n_epochs, SPO2_FS * EPOCH_SEC)
    labels      : np.ndarray  shape (n_epochs,)  binary (0=normal, 1=apnea)
    """
    raw      = mne.io.read_raw_edf(edf_path, preload=True)
    start_dt = raw.info["meas_date"]
    total_sec = int(raw.times[-1])

    ecg_full  = raw.copy().pick(TARGET_ECG).get_data()[0]
    spo2_full = raw.copy().pick(TARGET_SPO2).get_data()[0]

    mask = np.zeros(total_sec)

    with open(txt_path, "r") as f:
        lines = f.readlines()

    for line in lines[3:]:          # first 3 lines are header
        tokens = line.strip().split()
        if not tokens or ":" not in tokens[0]:
            continue

        time_str     = tokens[0]
        duration_sec = 0
        for token in tokens[1:]:
            if token.isdigit():
                duration_sec = int(token)
                break

        event_time = datetime.strptime(time_str, "%H:%M:%S").time()
        event_dt   = datetime.combine(start_dt.date(), event_time).replace(
                        tzinfo=start_dt.tzinfo)

        if event_dt < start_dt:     # midnight rollover
            event_dt += timedelta(days=1)

        rel_start = int((event_dt - start_dt).total_seconds())
        safe_start = max(0, rel_start)
        safe_end   = min(total_sec, rel_start + duration_sec)
        mask[safe_start:safe_end] = 1

    n_epochs = total_sec // EPOCH_SEC

    ecg_epochs  = ecg_full[:n_epochs * ECG_FS  * EPOCH_SEC].reshape(n_epochs, ECG_FS  * EPOCH_SEC)
    spo2_epochs = spo2_full[:n_epochs * SPO2_FS * EPOCH_SEC].reshape(n_epochs, SPO2_FS * EPOCH_SEC)

    mask_2d = mask[:n_epochs * EPOCH_SEC].reshape(n_epochs, EPOCH_SEC)
    labels  = (mask_2d.sum(axis=1) > 0).astype(int)

    return ecg_epochs, spo2_epochs, labels


# ─────────────────────────────────────────
# STEP 2 — Bulk extraction
# ─────────────────────────────────────────
def extract_all_patients(raw_dir: str = RAW_DIR):
    """
    Iterate over all (edf, txt) pairs in raw_dir and extract epochs.

    Returns
    -------
    patient_ecg    : list of per-patient ECG arrays
    patient_spo2   : list of per-patient SpO2 arrays
    patient_labels : list of per-patient label arrays
    """
    edf_files = sorted(glob.glob(os.path.join(raw_dir, "*.edf")))
    patient_ecg, patient_spo2, patient_labels = [], [], []

    for edf_file in edf_files:
        base     = os.path.basename(edf_file).replace(".edf", "")
        txt_file = os.path.join(raw_dir, f"{base}_respevt.txt")

        if not os.path.exists(txt_file) or "lifecard" in base:
            continue

        try:
            print(f"  Extracting {base}…")
            ecg, spo2, labels = process_single_patient(edf_file, txt_file)
            patient_ecg.append(ecg)
            patient_spo2.append(spo2)
            patient_labels.append(labels)
        except Exception as exc:
            print(f"  [SKIP] {base} — {exc}")

    print(f"\nExtracted {len(patient_ecg)} patients successfully.")
    return patient_ecg, patient_spo2, patient_labels


# ─────────────────────────────────────────
# STEP 3 — Patient-wise split & save
# ─────────────────────────────────────────
def split_and_save(
    patient_ecg,
    patient_spo2,
    patient_labels,
    processed_dir: str = PROCESSED_DIR,
    train_n: int = 17,
    val_n:   int = 4,
):
    """
    Randomly split patients into train / val / test sets and save .npy files.

    Split (default, 25 patients total): 17 train | 4 val | 4 test
    """
    os.makedirs(processed_dir, exist_ok=True)
    n = len(patient_ecg)
    np.random.seed(RANDOM_SEED)
    idx = np.random.permutation(n)

    train_idx = idx[:train_n]
    val_idx   = idx[train_n : train_n + val_n]
    test_idx  = idx[train_n + val_n:]

    def aggregate(id_list):
        ecg_a    = np.vstack([patient_ecg[i]    for i in id_list])
        spo2_a   = np.vstack([patient_spo2[i]   for i in id_list])
        labels_a = np.concatenate([patient_labels[i] for i in id_list])
        return ecg_a, spo2_a, labels_a

    splits = {
        "train": aggregate(train_idx),
        "val":   aggregate(val_idx),
        "test":  aggregate(test_idx),
    }

    for split_name, (ecg, spo2, labels) in splits.items():
        np.save(os.path.join(processed_dir, f"X_{split_name}_ecg.npy"),  ecg)
        np.save(os.path.join(processed_dir, f"X_{split_name}_spo2.npy"), spo2)
        np.save(os.path.join(processed_dir, f"y_{split_name}.npy"),      labels)
        print(f"  {split_name:5s} — {len(labels):5d} epochs  "
              f"(apnea: {int(labels.sum())})")

    print(f"\nAll arrays saved to: {processed_dir}")
    return splits


# ─────────────────────────────────────────
# SIGNAL PREPROCESSING (used before model)
# ─────────────────────────────────────────
def _bandpass(signal: np.ndarray, low: float, high: float, fs: int, order: int = 4):
    """Zero-phase Butterworth bandpass filter."""
    nyq = fs / 2
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal, axis=-1)


def _resample_spo2(spo2: np.ndarray, target_len: int) -> np.ndarray:
    """Upsample SpO2 from SPO2_FS*60 → target_len using linear interpolation."""
    src_len = spo2.shape[-1]
    if src_len == target_len:
        return spo2
    x_old = np.linspace(0, 1, src_len)
    x_new = np.linspace(0, 1, target_len)
    fn    = interp1d(x_old, spo2, kind="linear", axis=-1)
    return fn(x_new)


def preprocess_ucd(ecg: np.ndarray, spo2: np.ndarray):
    """
    Apply bandpass filtering to ECG (0.5–40 Hz) and SpO2 (0.01–0.5 Hz).
    Also resamples SpO2 to match ECG temporal length.

    Parameters
    ----------
    ecg  : (N, ECG_FS*EPOCH_SEC)  raw ECG epochs
    spo2 : (N, SPO2_FS*EPOCH_SEC) raw SpO2 epochs

    Returns
    -------
    ecg_out, spo2_out : filtered arrays of same shape as input
    """
    ecg_out  = _bandpass(ecg,  low=0.5,  high=40.0, fs=ECG_FS)
    spo2_out = _bandpass(spo2, low=0.01, high=0.5,  fs=SPO2_FS)
    return ecg_out.astype(np.float32), spo2_out.astype(np.float32)


# ─────────────────────────────────────────
# PYTORCH DATASET
# ─────────────────────────────────────────
class UCDApneaDataset(Dataset):
    """
    PyTorch Dataset for the UCD Sleep Apnea corpus.

    Each sample is a (ecg_epoch, spo2_epoch, label) triple.

    Parameters
    ----------
    ecg_data  : (N, L_ecg)
    spo2_data : (N, L_spo2)
    labels    : (N,)  float binary labels
    rec_ids   : (N,) optional int array — recording ID per epoch,
                used by temporal smoothing at inference time.
    """

    def __init__(self, ecg_data, spo2_data, labels, rec_ids=None):
        self.ecg    = torch.tensor(np.expand_dims(ecg_data,  axis=1), dtype=torch.float32)
        self.spo2   = torch.tensor(np.expand_dims(spo2_data, axis=1), dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.rec_ids = rec_ids  # numpy array or None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.ecg[idx], self.spo2[idx], self.labels[idx]


# ─────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("=== Sleep Apnea Preprocessing Pipeline ===\n")

    rename_rec_to_edf(RAW_DIR)
    patient_ecg, patient_spo2, patient_labels = extract_all_patients(RAW_DIR)

    if patient_ecg:
        split_and_save(patient_ecg, patient_spo2, patient_labels, PROCESSED_DIR)
    else:
        print("No patients found. Check RAW_DIR path.")
