#!/usr/bin/env python3
# Applying_NdLinear_to_Bioinformatics.py

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNetCV
from sklearn.neighbors import NearestCentroid
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils import resample
import joblib

# ───────────────────────────── Hyperparameters & Setup ─────────────────────────────
COUNTS_PATH       = "salmon.merged.gene_counts.tsv"
SAMPLE_SHEET      = "human_liver_GSE126848_rnaseq.csv"
FEATURE_TOP_K     = 100       # top genes by variance
PCA_COMPONENTS    = 2         # more aggressive reduction
OUTER_FOLDS       = 5         # outer CV folds
INNER_FOLDS       = 3         # ElasticNet CV
TEST_SPLIT_FRAC   = 0.10
BOOTSTRAP_ITERS   = 1000
PERMUTATION_ITERS = 1000
SEED              = 42

os.makedirs("models", exist_ok=True)
np.random.seed(SEED)

# ────────────────────────── 1) Load Data & Labels ────────────────────────────────
counts_df = pd.read_csv(COUNTS_PATH, sep="\t", index_col=0)
ss        = pd.read_csv(SAMPLE_SHEET)
samples   = ss["sample"].tolist()
counts_df = counts_df.loc[:, samples]

# map sample names to binary labels
label_map = {"Normal_weight": 0, "Obese": 0, "NAFL": 1, "NASH": 1}
y = np.array([label_map["_".join(s.split("_")[:-1])] for s in samples])
print(f"> Loaded {len(y)} samples: class distribution {{0:{np.sum(y==0)},1:{np.sum(y==1)}}}")

# ────────────────────────── 2) Log2-transform & to NumPy ───────────────────────────
data = np.log2(counts_df + 1).T.values  # shape=(n_samples, n_genes)

# ──────────────────────────── 3) Variance Filter ────────────────────────────────
vars_    = data.var(axis=0)
top_idx  = np.argsort(vars_)[-FEATURE_TOP_K:]
data_var = data[:, top_idx]
print(f"> Variance filter: selected {data_var.shape[1]} genes")

# ─────────────────────── 4) ElasticNet Feature Selection ─────────────────────────
scaler    = StandardScaler()
data_scld = scaler.fit_transform(data_var)
en        = ElasticNetCV(l1_ratio=[0.1,0.5,0.9], cv=INNER_FOLDS, random_state=SEED)
en.fit(data_scld, y)
mask      = np.abs(en.coef_) > 1e-5
data_sel  = data_var[:, mask] if mask.sum()>0 else data_var
print(f"> ElasticNet: retained {data_sel.shape[1]} genes")

# ───────────────────────── 5) PCA Dimensionality Reduction ────────────────────────
data_sel_s = scaler.fit_transform(data_sel)
n_pca      = min(PCA_COMPONENTS, data_sel_s.shape[1])
pca        = PCA(n_components=n_pca, random_state=SEED)
X_all      = pca.fit_transform(data_sel_s)
print(f"> PCA: reduced to {n_pca} components capture {pca.explained_variance_ratio_.sum():.2%} variance")

# ────────────────────────── 6) Train/Test Split ─────────────────────────────────
X_tv, X_test, y_tv, y_test = train_test_split(
    X_all, y, test_size=TEST_SPLIT_FRAC, stratify=y, random_state=SEED
)
print(f"> Split TrainVal={len(y_tv)}, Test={len(y_test)} samples")

# ─────────────────────────── 7) Nested CV with Nearest Centroid ──────────────────
outer_cv   = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=SEED)
outer_accs = []
for fold, (tr_idx, va_idx) in enumerate(outer_cv.split(X_tv, y_tv), 1):
    X_tr, X_val = X_tv[tr_idx], X_tv[va_idx]
    y_tr, y_val = y_tv[tr_idx], y_tv[va_idx]
    # extremely simple classifier
    clf = NearestCentroid()
    clf.fit(X_tr, y_tr)
    val_acc = clf.score(X_val, y_val)
    outer_accs.append(val_acc)
    print(f"Fold {fold}: val acc={val_acc:.4f}")
print(f"> Average outer CV acc: {np.mean(outer_accs):.4f}")

# ───────────────────────────── 8) Final Fit & Test ────────────────────────────────
clf = NearestCentroid()
clf.fit(X_tv, y_tv)
test_acc = clf.score(X_test, y_test)
print(f"> Test accuracy: {test_acc:.4f}")

# bootstrap 95% CI on test
boot_accs = []
for _ in range(BOOTSTRAP_ITERS):
    idxs = resample(np.arange(len(y_test)), replace=True)
    boot_accs.append(clf.score(X_test[idxs], y_test[idxs]))
ci_low, ci_up = np.percentile(boot_accs, [2.5, 97.5])
print(f"> Bootstrap 95% CI: [{ci_low:.4f}, {ci_up:.4f}]")

# ────────────────────────── 9) Permutation Testing ────────────────────────────────
perm_accs = []
for _ in range(PERMUTATION_ITERS):
    perm_y = np.random.RandomState().permutation(y_test)
    perm_accs.append(np.mean(clf.predict(X_test)==perm_y))
pvalue = np.mean(np.array(perm_accs) >= test_acc)
print(f"> Permutation test p-value: {pvalue:.4f}")

# ───────────────────────────── 10) Save Model ─────────────────────────────────────
joblib.dump(clf, "models/nearest_centroid_rnaseq.pkl")
print("> Saved model: models/nearest_centroid_rnaseq.pkl")

