#!/usr/bin/env python3
# Applying_NdLinear_to_Bioinformatics.py

import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils import resample
from ndlinear import NdLinear

# ───────────────────────────── Hyperparameters ─────────────────────────────
COUNTS_PATH       = "salmon.merged.gene_counts.tsv"
SAMPLE_SHEET      = "human_liver_GSE126848_rnaseq.csv"
FEATURE_TOP_K     = 100       # top genes by variance
PCA_COMPONENTS    = 2         # aggressive PCA reduction
OUTER_FOLDS       = 5         # outer CV splits
INNER_FOLDS       = 3         # ElasticNet CV
BATCH_SIZE        = 16
LR                = 1e-3
WEIGHT_DECAYS     = [1e-4, 1e-5, 1e-6]
EPOCHS            = 20
TEST_SPLIT_FRAC   = 0.10
BOOTSTRAP_ITERS   = 1000
PERMUTATION_ITERS = 1000
SEED              = 42

# create output dir and set seeds
os.makedirs("models", exist_ok=True)
np.random.seed(SEED)
torch.manual_seed(SEED)

def load_data():
    # 1) Load counts and sample sheet
    counts = pd.read_csv(COUNTS_PATH, sep="\t", index_col=0)
    ss     = pd.read_csv(SAMPLE_SHEET)
    samples = ss["sample"].tolist()
    counts = counts.loc[:, samples]

    # 2) Map to binary labels
    label_map = {"Normal_weight":0, "Obese":0, "NAFL":1, "NASH":1}
    y = np.array([ label_map["_".join(s.split("_")[:-1])] for s in samples ])
    print(f"> Loaded {len(y)} samples: classes {{0:{np.sum(y==0)},1:{np.sum(y==1)}}}")

    # 3) Log2-transform and variance filter
    data = np.log2(counts + 1).T.values  # shape=(n_samples, n_genes)
    vars_ = data.var(axis=0)
    top_idx = np.argsort(vars_)[-FEATURE_TOP_K:]
    data_var = data[:, top_idx]
    print(f"> Variance filter: {data_var.shape[1]} genes selected")

    # 4) ElasticNet feature selection
    scaler  = StandardScaler()
    data_scl = scaler.fit_transform(data_var)
    en      = ElasticNetCV(l1_ratio=[0.1,0.5,0.9], cv=INNER_FOLDS, random_state=SEED)
    en.fit(data_scl, y)
    mask    = np.abs(en.coef_) > 1e-5
    data_sel = data_var[:, mask] if mask.sum()>0 else data_var
    print(f"> ElasticNet: retained {data_sel.shape[1]} genes")

    # 5) PCA reduction
    data_sel_s = scaler.fit_transform(data_sel)
    n_pca = min(PCA_COMPONENTS, data_sel_s.shape[1])
    pca = PCA(n_components=n_pca, random_state=SEED)
    X_all = pca.fit_transform(data_sel_s)
    print(f"> PCA: reduced to {n_pca} components capturing {pca.explained_variance_ratio_.sum():.2%} variance")

    return X_all, y

# reset torch device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────── Model Definition ───────────────────
class NdLinearClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.fc1 = NdLinear(input_dims=(input_dim,), hidden_size=(hidden_dim,))
        self.act = nn.ReLU()
        self.fc2 = NdLinear(input_dims=(hidden_dim,), hidden_size=(num_classes,))

    def forward(self, x):
        x = self.act(self.fc1(x))
        return self.fc2(x)

# ─────────────────── Training Utilities ───────────────────
criterion = nn.CrossEntropyLoss()

def train_model(model, loader, optimizer):
    model.train()
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()

@torch.no_grad()
def evaluate_model(model, loader):
    model.eval()
    correct = 0
    total   = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        preds = model(xb).argmax(dim=1)
        correct += (preds == yb).sum().item()
        total   += yb.size(0)
    return correct / total

# ───────────────────── Main Pipeline ─────────────────────
if __name__ == "__main__":
    # Load and preprocess data
    X_all, y = load_data()

    # Train/Test split
    X_tv, X_test, y_tv, y_test = train_test_split(
        X_all, y, test_size=TEST_SPLIT_FRAC, stratify=y, random_state=SEED
    )
    print(f"> Split Train+Val={len(y_tv)}, Test={len(y_test)} samples")

    # Nested CV to select weight_decay
    outer_cv = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=SEED)
    outer_accs, best_wds = [], []

    for fold, (tr_idx, va_idx) in enumerate(outer_cv.split(X_tv, y_tv), start=1):
        X_tr, X_val = X_tv[tr_idx], X_tv[va_idx]
        y_tr, y_val = y_tv[tr_idx], y_tv[va_idx]

        best_acc, best_wd = 0, None
        for wd in WEIGHT_DECAYS:
            inner_accs = []
            inner_cv = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=SEED)
            for itr, iva in inner_cv.split(X_tr, y_tr):
                ds_tr = TensorDataset(
                    torch.tensor(X_tr[itr], dtype=torch.float32),
                    torch.tensor(y_tr[itr], dtype=torch.long)
                )
                ds_va = TensorDataset(
                    torch.tensor(X_tr[iva], dtype=torch.float32),
                    torch.tensor(y_tr[iva], dtype=torch.long)
                )
                loader_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True)
                loader_va = DataLoader(ds_va, batch_size=BATCH_SIZE)

                model = NdLinearClassifier(input_dim=X_tr.shape[1], hidden_dim=64, num_classes=2).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=wd)
                # Train for fixed epochs
                for epoch in range(EPOCHS):
                    train_model(model, loader_tr, optimizer)

                acc = evaluate_model(model, loader_va)
                inner_accs.append(acc)

            mean_acc = np.mean(inner_accs)
            if mean_acc > best_acc:
                best_acc, best_wd = mean_acc, wd

        outer_accs.append(best_acc)
        best_wds.append(best_wd)
        print(f"Fold {fold} best wd={best_wd}, val acc={best_acc:.4f}")

    print(f"> Average outer CV acc: {np.mean(outer_accs):.4f}")

    # Final training on full Train+Val with best WD median
    top_wd = sorted(best_wds)[len(best_wds)//2]
    ds_tv = TensorDataset(
        torch.tensor(X_tv, dtype=torch.float32),
        torch.tensor(y_tv, dtype=torch.long)
    )
    loader_tv = DataLoader(ds_tv, batch_size=BATCH_SIZE, shuffle=True)

    model = NdLinearClassifier(input_dim=X_tv.shape[1], hidden_dim=64, num_classes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=top_wd)
    for epoch in range(EPOCHS):
        train_model(model, loader_tv, optimizer)

    # Evaluate on test set
    ds_test    = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    loader_test = DataLoader(ds_test, batch_size=BATCH_SIZE)
    test_acc    = evaluate_model(model, loader_test)
    print(f"> Test accuracy: {test_acc:.4f}")

    # Bootstrap CI
    boot_accs = []
    for _ in range(BOOTSTRAP_ITERS):
        idxs = resample(np.arange(len(y_test)), replace=True)
        ds_bs = TensorDataset(torch.tensor(X_test[idxs], dtype=torch.float32), torch.tensor(y_test[idxs], dtype=torch.long))
        acc = evaluate_model(model, DataLoader(ds_bs, batch_size=BATCH_SIZE))
        boot_accs.append(acc)
    ci_low, ci_up = np.percentile(boot_accs, [2.5, 97.5])
    print(f"> Bootstrap 95% CI: [{ci_low:.4f}, {ci_up:.4f}]")

    # Permutation test
    perm_accs = []
    for _ in range(PERMUTATION_ITERS):
        perm_y = np.random.RandomState().permutation(y_test)
        ds_py = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(perm_y, dtype=torch.long))
        perm_accs.append(evaluate_model(model, DataLoader(ds_py, batch_size=BATCH_SIZE)))
    pvalue = np.mean(np.array(perm_accs) >= test_acc)
    print(f"> Permutation test p-value: {pvalue:.4f}")

    # Save model weights
    torch.save(model.state_dict(), "models/ndlinear_rnaseq_final.pth")
    print("> Saved NdLinear model state to models/ndlinear_rnaseq_final.pth")
