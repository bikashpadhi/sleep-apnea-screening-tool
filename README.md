# sleep-apnea-screening-tool

A deep learning pipeline for automated sleep apnea screening from overnight polysomnography (PSG) recordings. The model detects apnea events in 60-second epochs using single-lead ECG and SpO₂ signals, achieving strong sensitivity while maintaining clinical-grade specificity.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Key Design Decisions](#key-design-decisions)
- [Citation](#citation)

---

## Overview

Sleep apnea affects approximately 1 billion people worldwide but remains severely under-diagnosed due to the cost and complexity of in-lab polysomnography. This project builds a binary classifier that labels each 60-second epoch of a patient's overnight recording as either **normal (0)** or **apnea (1)**, using only ECG and blood oxygen saturation (SpO₂) signals — both available from low-cost wearable devices.

**Key features:**
- Dual-branch 1D-ResNet encoder — one branch per modality
- Learned gating mechanism to weight ECG vs SpO₂ contributions dynamically
- Bidirectional GRU for temporal context modelling
- Gaussian temporal smoothing at inference time to suppress isolated false positives
- Focal + class-separation loss to handle severe class imbalance

---

## Architecture

```
ECG  (1×7680)  ──▶  1D-ResNet Encoder  ──┐
                                          ├──▶  Gated Fusion  ──▶  BiGRU  ──▶  MLP Head  ──▶  logit
SpO₂ (1×480)   ──▶  1D-ResNet Encoder  ──┘
```

| Component | Details |
|---|---|
| ECG Encoder | 3 ConvBlocks with SE attention, stride downsampling |
| SpO₂ Encoder | 3 ConvBlocks with SE attention |
| Fusion Gate | 2-layer MLP → Softmax weights → weighted sum |
| Temporal Model | 2-layer BiGRU, hidden=128, dropout=0.3 |
| Pooling | Global mean + max over time axis |
| Head | Linear(512→128) → LN → GELU → Dropout → Linear(32→1) |
| Loss | Focal (γ=2) + Separation margin (0.35, weight=0.4) |

**Total parameters:** ~1.2 M

---

## Dataset

This project uses the **UCD Sleep Apnea Database** (University College Dublin), available on [PhysioNet](https://physionet.org/content/ucddb/1.0.0/).

- 25 subjects with full overnight PSG recordings
- Signals used: single-lead ECG (128 Hz) and SpO₂ (8 Hz)
- Annotations: respiratory-event text files (`*_respevt.txt`)

**Patient-wise split** (seed=42):

| Split | Patients | Epochs |
|---|---|---|
| Train | 17 | ~7 000+ (after augmentation) |
| Validation | 4 | ~1 600 |
| Test | 4 | ~1 600 |

> **Data privacy:** No raw data is committed to this repository. Download the dataset from PhysioNet and place files in `data/raw/` as described in the Usage section.

---

## Repository Structure

```
Sleep-Apnea-Screening/
│
├── data/                    # (gitignored — download separately)
│   ├── raw/                 # Original .edf + *_respevt.txt files
│   ├── processed/           # Extracted .npy arrays (post-split)
│   └── augmented/           # Augmented training arrays
│
├── models/                  # Saved model checkpoints
│
├── notebooks/
│   └── final_mp_ok.ipynb    # Original experimental notebook
│
├── src/
│   ├── preprocessing.py     # EDF loading, epoch extraction, train/val/test split
│   ├── augmentation.py      # Offline augmentation + online batch augmentation
│   ├── model.py             # ApneaModel, SEBlock, ConvBlock, FocalSeparationLoss
│   ├── train.py             # Full training loop with temporal smoothing
│   └── evaluate.py          # Test-set evaluation, metrics, confusion matrix plots
│
├── results/                 # Generated plots and metric text files
│
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE
```

---

## Installation

**Python 3.10+ is required.**

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/Sleep-Apnea-Screening.git
cd Sleep-Apnea-Screening

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

For GPU training (CUDA 12.x), install PyTorch first:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

---

## Usage

### Step 1 — Prepare the data

Download the UCD Sleep Apnea Database from PhysioNet:
```
https://physionet.org/content/ucddb/1.0.0/
```

Place all `.rec` / `.edf` files and `*_respevt.txt` annotation files into `data/raw/`.

### Step 2 — Preprocessing

Extracts ECG + SpO₂ epochs and saves patient-wise train/val/test splits:

```bash
python src/preprocessing.py
```

Outputs to `data/processed/`:
```
X_train_ecg.npy   X_train_spo2.npy   y_train.npy
X_val_ecg.npy     X_val_spo2.npy     y_val.npy
X_test_ecg.npy    X_test_spo2.npy    y_test.npy
```

### Step 3 — Augmentation

Generates 2× augmented copies per training sample + Mixup synthesis:

```bash
python src/augmentation.py
```

Outputs to `data/augmented/`.

### Step 4 — Training

```bash
python src/train.py
```

Saves the best checkpoint (by EMA-smoothed val F1) to `models/best_model.pth`.
Training curves are saved to `models/training_curves.png`.

### Step 5 — Evaluation

```bash
# Evaluate on test set (default)
python src/evaluate.py

# Evaluate on validation set
python src/evaluate.py --split val
```

Results (metrics + confusion matrix PNGs) are saved to `results/`.

### Override paths via environment variables

```bash
export RAW_DIR=path/to/raw
export PROCESSED_DIR=path/to/processed
export AUGMENTED_DIR=path/to/augmented
export MODELS_DIR=path/to/models
export RESULTS_DIR=path/to/results
```

---

## Results

Results reported on the held-out test set (4 patients, ~1 600 epochs):

| Metric | Raw | With Gaussian Smoothing (σ=1.5) |
|---|---|---|
| F1 Score | 0.6113 | 0.6728 |
| Precision | 0.5853 | 0.5916 |
| Recall (Sensitivity) | 0.6398 | 0.7797 |
| Specificity | 0.6232 | 0.5528 |

**Why temporal smoothing helps:** Apnea events cluster into runs of 2–25 consecutive 60-second windows. Gaussian smoothing (σ=1.5 windows = ±90 s context) suppresses isolated false-positive spikes and fills isolated false-negative dips without crossing patient recording boundaries.

---

## Key Design Decisions

**Patient-wise splitting** — All epochs from a given patient go entirely into one split. This prevents data leakage from the highly autocorrelated nature of PSG recordings.

**Weighted sampling + Focal loss** — The dataset is typically 60–70% negative. WeightedRandomSampler ensures balanced mini-batches; Focal loss (γ=2) further down-weights easy negatives during training.

**Class-separation margin** — An auxiliary loss term penalises small gaps between the mean predicted probability for positive vs negative samples. This keeps the model's decision boundary from collapsing.

**Inference-time smoothing** — The saved threshold was optimised on smoothed probabilities. Always apply `smooth_probs()` before thresholding at inference time.

---

## Citation

If you use this code or methodology in your work, please cite:

```bibtex
@misc{sleepapnea2025,
  title  = {Sleep Apnea Screening via Dual-Branch 1D-ResNet and Temporal Smoothing},
  author = {<Your Name>},
  year   = {2025},
  url    = {https://github.com/<your-username>/Sleep-Apnea-Screening}
}
```

**Dataset citation:**

Goldberger AL, Amaral LAN, Glass L, et al. PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals. *Circulation* 101(23):e215–e220, 2000.

---

## License

MIT — see [LICENSE](LICENSE).
