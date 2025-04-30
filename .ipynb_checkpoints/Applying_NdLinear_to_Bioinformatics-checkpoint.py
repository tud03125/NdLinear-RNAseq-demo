#!/usr/bin/env python3
# ndlinear_rnaseq_generalized.py

import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split, Subset
from sklearn.model_selection import KFold
from ndlinear import NdLinear

# ─────────────────────────────────────────────────────────────────────────────
# 1) Paths & hyperparameters
# ─────────────────────────────────────────────────────────────────────────────
COUNTS_PATH   = "salmon.merged.gene_counts.tsv"
SAMPLE_SHEET  = "human_liver_GSE126848_rnaseq.csv"
BATCH_SIZE    = 16
HIDDEN_UNITS  = 128
LR            = 1e-3
WEIGHT_DECAY  = 1e-5
MAX_EPOCHS    = 50
PATIENCE      = 5     # for early stopping
KFOLDS        = 5
SEED          = 42

# Save dirs
CKPT_DIR      = "models/checkpoints"
FINAL_DIR     = "models"
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(FINAL_DIR, exist_ok=True)

# Reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# 2) Load & preprocess data
# ─────────────────────────────────────────────────────────────────────────────
counts_df = pd.read_csv(COUNTS_PATH, sep="\t", index_col=0)
print(f"> Raw counts: {counts_df.shape}")

# Align samples to sheet order
ss = pd.read_csv(SAMPLE_SHEET)
sample_ids = ss["sample"].tolist()
counts_df = counts_df.loc[:, sample_ids]

# Extract groups and binary labels
def extract_group(name):
    parts = name.split("_")
    return "_".join(parts[:-1])

groups = [extract_group(s) for s in sample_ids]
label_map = {"Normal_weight": 0, "Obese": 0, "NAFL":1, "NASH":1}
y = np.array([label_map[g] for g in groups])

print(f"> Label counts: {{0: {np.sum(y==0)}, 1: {np.sum(y==1)}}}")

# Log2-transform
counts_log2 = np.log2(counts_df + 1)

# Build PyTorch dataset (samples × features)
X = torch.tensor(counts_log2.values.T, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)
full_ds = TensorDataset(X, y)
n_total  = len(full_ds)

# ─────────────────────────────────────────────────────────────────────────────
# 3) Create held-out test split (10%)
# ─────────────────────────────────────────────────────────────────────────────
n_test     = int(0.10 * n_total)
n_trainval = n_total - n_test
trainval_ds, test_ds = random_split(
    full_ds,
    [n_trainval, n_test],
    generator=torch.Generator().manual_seed(SEED)
)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
print(f"> Split off test set of size {n_test}")

# ─────────────────────────────────────────────────────────────────────────────
# 4) Define model with dropout
# ─────────────────────────────────────────────────────────────────────────────
class NdLinearClassifier(nn.Module):
    def __init__(self, in_feats, hidden, out_classes):
        super().__init__()
        self.fc1     = NdLinear(input_dims=(in_feats,), hidden_size=(hidden,))
        self.dropout = nn.Dropout(p=0.5)
        self.act     = nn.ReLU()
        self.fc2     = NdLinear(input_dims=(hidden,), hidden_size=(out_classes,))

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────────────────────
# 5) k-Fold Cross-Validation on train+val set
# ─────────────────────────────────────────────────────────────────────────────
kf = KFold(n_splits=KFOLDS, shuffle=True, random_state=SEED)
cv_metrics = []

for fold, (train_idx, val_idx) in enumerate(kf.split(range(n_trainval)), start=1):
    print(f"\n=== Fold {fold}/{KFOLDS} ===")
    train_ds = Subset(trainval_ds, train_idx)
    val_ds   = Subset(trainval_ds, val_idx)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    # Instantiate fresh model & optimizer per fold
    model = NdLinearClassifier(
        in_feats=X.shape[1],
        hidden=HIDDEN_UNITS,
        out_classes=2
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )
    criterion = nn.CrossEntropyLoss()

    best_val_loss     = float("inf")
    epochs_no_improve = 0
    best_ckpt_path    = None

    for epoch in range(1, MAX_EPOCHS + 1):
        # ---- Train ----
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss  = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_ds)

        # ---- Validate ----
        model.eval()
        val_loss = 0.0
        correct  = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                val_loss += criterion(preds, yb).item() * xb.size(0)
                correct  += (preds.argmax(dim=1) == yb).sum().item()
        val_loss /= len(val_ds)
        val_acc  = correct / len(val_ds)

        print(
            f"Epoch {epoch:02d} – "
            f"Train Loss: {train_loss:.4f}  "
            f"Val Loss: {val_loss:.4f}  "
            f"Val Acc: {val_acc:.4f}"
        )

        # ---- Early stopping & checkpointing ----
        if val_loss < best_val_loss:
            best_val_loss     = val_loss
            epochs_no_improve = 0
            ckpt = {
                "fold": fold,
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc
            }
            best_ckpt_path = os.path.join(
                CKPT_DIR, f"fold{fold}_best.pth"
            )
            torch.save(ckpt, best_ckpt_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    # Load best model for this fold
    checkpoint = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    cv_metrics.append({
        "fold": fold,
        "best_epoch": checkpoint["epoch"],
        "val_loss":  checkpoint["val_loss"],
        "val_acc":   checkpoint["val_acc"]
    })
    print(f"> Fold {fold} best: epoch {checkpoint['epoch']}, "
          f"Val Loss={checkpoint['val_loss']:.4f}, "
          f"Val Acc={checkpoint['val_acc']:.4f}")

# Summary of cross-validation performance
avg_val_acc = np.mean([m["val_acc"] for m in cv_metrics])
print(f"\n=== CV Summary ===")
for m in cv_metrics:
    print(f"Fold {m['fold']}: loss={m['val_loss']:.4f}, acc={m['val_acc']:.4f}")
print(f"Average CV Acc: {avg_val_acc:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 6) Final evaluation on held-out test set
# ─────────────────────────────────────────────────────────────────────────────
# Reload the best overall CV model (lowest val_loss across folds)
best_fold = min(cv_metrics, key=lambda m: m["val_loss"])["fold"]
best_ckpt = torch.load(
    os.path.join(CKPT_DIR, f"fold{best_fold}_best.pth"),
    map_location=device
)
model = NdLinearClassifier(
    in_feats=X.shape[1],
    hidden=HIDDEN_UNITS,
    out_classes=2
).to(device)
model.load_state_dict(best_ckpt["model_state"])
model.eval()

test_loss = 0.0
correct   = 0
criterion = nn.CrossEntropyLoss()
with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        preds = model(xb)
        test_loss += criterion(preds, yb).item() * xb.size(0)
        correct   += (preds.argmax(dim=1) == yb).sum().item()
test_loss /= len(test_ds)
test_acc  = correct / len(test_ds)

print(
    f"\n=== Test Set Results ===\n"
    f"Loss: {test_loss:.4f}   "
    f"Accuracy: {test_acc:.4f}"
)

# ─────────────────────────────────────────────────────────────────────────────
# 7) Save final model weights
# ─────────────────────────────────────────────────────────────────────────────
final_model_path = os.path.join(FINAL_DIR, "ndlinear_human_liver_final.pth")
torch.save(model.state_dict(), final_model_path)
print(f"> Final model saved to {final_model_path}")
