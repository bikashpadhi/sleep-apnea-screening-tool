"""
model.py
========
1D-ResNet + Bidirectional GRU architecture for sleep apnea detection.

Architecture overview
---------------------
Two parallel convolutional encoders (one per modality) extract
temporal features from ECG and SpO2 signals independently. A learned
gating mechanism fuses the two representations before a bidirectional
GRU aggregates sequential context. A small MLP head produces the
binary logit.

Components
----------
SEBlock       — Squeeze-and-Excitation channel attention
ConvBlock     — Residual conv block with SE and GELU activations
ApneaModel    — Full two-branch network (ECG + SpO2 → logit)
FocalSeparationLoss — Focal loss + class-separation margin term
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────
# BUILDING BLOCKS
# ─────────────────────────────────────────

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block for 1-D signals.

    Recalibrates channel-wise feature responses by modelling
    inter-channel dependencies.

    Parameters
    ----------
    channels  : int  number of input channels
    reduction : int  bottleneck reduction ratio (default 8)
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        scale = self.fc(x.mean(dim=2))          # (B, C)
        return x * scale.unsqueeze(-1)


class ConvBlock(nn.Module):
    """
    Residual 1-D conv block with optional strided downsampling.

    Structure: Conv → BN → GELU → Conv → BN → GELU → SE → residual add

    Parameters
    ----------
    in_c   : int  input channels
    out_c  : int  output channels
    k      : int  kernel size (default 7)
    stride : int  stride for downsampling (default 1)
    """

    def __init__(self, in_c: int, out_c: int, k: int = 7, stride: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_c,  out_c, k, stride=stride, padding=k // 2),
            nn.BatchNorm1d(out_c),
            nn.GELU(),
            nn.Conv1d(out_c, out_c, k, padding=k // 2),
            nn.BatchNorm1d(out_c),
            nn.GELU(),
        )
        # Projection shortcut when shape changes
        self.skip = (
            nn.Sequential(
                nn.Conv1d(in_c, out_c, 1, stride=stride),
                nn.BatchNorm1d(out_c),
            )
            if (stride != 1 or in_c != out_c)
            else nn.Identity()
        )
        self.se = SEBlock(out_c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.se(self.net(x) + self.skip(x))


# ─────────────────────────────────────────
# MAIN MODEL
# ─────────────────────────────────────────

class ApneaModel(nn.Module):
    """
    Dual-branch 1D-ResNet + Bidirectional GRU for sleep apnea detection.

    Input
    -----
    ecg  : (B, 1, L_ecg)   — single-lead ECG epoch (128 Hz × 60 s = 7 680 samples)
    spo2 : (B, 1, L_spo2)  — SpO2 epoch (8 Hz × 60 s = 480 samples)

    Output
    ------
    logit : (B,)  — raw (pre-sigmoid) binary classification score

    Parameters
    ----------
    hidden_dim : int  GRU hidden size (default 128)
    gru_layers : int  number of stacked GRU layers (default 2)
    """

    def __init__(self, hidden_dim: int = 128, gru_layers: int = 2):
        super().__init__()

        # ECG encoder — aggressive downsampling to match SpO2 temporal length
        self.ecg_enc = nn.Sequential(
            ConvBlock(1,   32, k=15, stride=4),
            nn.MaxPool1d(4),
            ConvBlock(32,  64, k=7,  stride=2),
            ConvBlock(64, 128, k=7,  stride=2),
        )

        # SpO2 encoder
        self.spo2_enc = nn.Sequential(
            ConvBlock(1,   32, k=7, stride=2),
            ConvBlock(32,  64, k=5, stride=2),
            ConvBlock(64, 128, k=3, stride=1),
        )

        # Gating: learns how much to trust each modality
        self.gate = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=1),
        )

        # Temporal context modelling
        self.gru = nn.GRU(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )

        # Classification head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 4, 128),   # *4 = bidirectional × {mean, max} pool
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(128, 32),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
        )

    def forward(self, ecg: torch.Tensor, spo2: torch.Tensor) -> torch.Tensor:
        e = self.ecg_enc(ecg)    # (B, 128, T_e)
        s = self.spo2_enc(spo2)  # (B, 128, T_s)

        # Compute per-modality weights
        w = self.gate(torch.cat([e.mean(dim=2), s.mean(dim=2)], dim=1))  # (B, 2)

        # Fuse at the shorter temporal dimension to avoid shape mismatch
        t      = min(e.size(2), s.size(2))
        fused  = (
            w[:, 0:1].unsqueeze(-1) * e[:, :, :t]
            + w[:, 1:2].unsqueeze(-1) * s[:, :, :t]
        )  # (B, 128, t)

        gru_out, _ = self.gru(fused.permute(0, 2, 1))  # (B, t, hidden*2)

        # Global mean + max pooling over time
        pooled = torch.cat(
            [gru_out.mean(dim=1), gru_out.max(dim=1).values], dim=1
        )  # (B, hidden*4)

        return self.head(pooled).squeeze(1)  # (B,)


# ─────────────────────────────────────────
# LOSS FUNCTION
# ─────────────────────────────────────────

class FocalSeparationLoss(nn.Module):
    """
    Focal Loss + class-separation margin term.

    Focal loss addresses class imbalance by down-weighting easy
    negatives. The separation term penalises the model when the
    mean predicted probability for positive examples is not
    sufficiently higher than for negatives.

    Parameters
    ----------
    pos_weight  : float  BCE weight for positive class (counters imbalance)
    gamma       : float  focal modulation exponent (default 2.0)
    margin      : float  minimum required gap between pos/neg means (0.35)
    sep_weight  : float  weight of separation term in total loss (0.4)
    """

    def __init__(
        self,
        pos_weight:  float = 1.0,
        gamma:       float = 2.0,
        margin:      float = 0.35,
        sep_weight:  float = 0.4,
    ):
        super().__init__()
        self.pos_weight = pos_weight
        self.gamma      = gamma
        self.margin     = margin
        self.sep_weight = sep_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_raw = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
            pos_weight=torch.tensor(self.pos_weight, device=logits.device),
        )
        probs        = torch.sigmoid(logits)
        p_t          = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss   = (focal_weight * bce_raw).mean()

        pos_mask = targets == 1
        neg_mask = targets == 0
        sep      = torch.tensor(0.0, device=logits.device)
        if pos_mask.sum() > 0 and neg_mask.sum() > 0:
            sep = F.relu(
                self.margin
                - (probs[pos_mask].mean() - probs[neg_mask].mean())
            )

        return focal_loss + self.sep_weight * sep


# ─────────────────────────────────────────
# QUICK SANITY CHECK
# ─────────────────────────────────────────
if __name__ == "__main__":
    model = ApneaModel(hidden_dim=128, gru_layers=2)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"ApneaModel — trainable parameters: {n_params:,}")

    ecg_dummy  = torch.randn(4, 1, 7680)   # batch=4, 128 Hz × 60 s
    spo2_dummy = torch.randn(4, 1, 480)    # batch=4,   8 Hz × 60 s
    out        = model(ecg_dummy, spo2_dummy)
    print(f"Output shape: {out.shape}")    # expected: torch.Size([4])
