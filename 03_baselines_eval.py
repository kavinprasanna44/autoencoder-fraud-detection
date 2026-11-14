from google.colab import drive
drive.mount('/content/drive')

import numpy as np
import matplotlib.pyplot as plt

PATH = "/content/drive/MyDrive/Auto Encoder Senior Thesis/"

d = np.load(PATH + "splits.npz")
X_train = d["X_train"]
X_test  = d["X_test"]
y_test  = d["y_test"]

ae_scores = np.load(PATH + "test_scores.npy")

print("X_train:", X_train.shape)
print("X_test: ", X_test.shape)
print("frauds in test:", int(y_test.sum()))
print("ae_scores:", ae_scores.shape)


from sklearn.ensemble import IsolationForest

iso = IsolationForest(n_estimators=100, contamination="auto", random_state=42, n_jobs=-1)
iso.fit(X_train)

iso_scores = -iso.score_samples(X_test)

print("Isolation Forest trained on", X_train.shape[0], "normal transactions")
print("iso_scores:", iso_scores.shape)


from sklearn.svm import OneClassSVM

rng = np.random.RandomState(42)
sub = rng.choice(len(X_train), 10000, replace=False)

ocsvm = OneClassSVM(kernel="rbf", nu=0.01, gamma="scale")
ocsvm.fit(X_train[sub])

ocsvm_scores = -ocsvm.score_samples(X_test)

print("One-Class SVM trained on", len(sub), "normal transactions (subsample)")
print("ocsvm_scores:", ocsvm_scores.shape)


import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

models = {"Autoencoder": ae_scores,
          "Isolation Forest": iso_scores,
          "One-Class SVM": ocsvm_scores}

def prec_rec_at_budget(scores, y, frac):
    k = int(round(frac * len(scores)))
    top_k = np.argsort(scores)[::-1][:k]
    caught = int(y[top_k].sum())
    return caught / k, caught / int(y.sum()), caught, k

rows = []
for name, s in models.items():
    p1, r1, _, _ = prec_rec_at_budget(s, y_test, 0.01)
    rows.append({"Model": name,
                 "Average Precision": round(average_precision_score(y_test, s), 3),
                 "ROC-AUC": round(roc_auc_score(y_test, s), 3),
                 "Precision @1%": round(p1, 3),
                 "Recall @1%": round(r1, 3)})

metrics_df = pd.DataFrame(rows).sort_values("Average Precision", ascending=False)
metrics_df


from sklearn.metrics import precision_recall_curve

plt.figure(figsize=(7, 5))
for name, s in models.items():
    prec, rec, _ = precision_recall_curve(y_test, s)
    ap = average_precision_score(y_test, s)
    plt.plot(rec, prec, label=f"{name} (AP = {ap:.3f})", linewidth=2)

baseline = y_test.mean()
plt.axhline(baseline, ls="--", c="gray", label=f"Random guessing (AP = {baseline:.3f})")

plt.xlabel("Recall (fraction of all fraud caught)")
plt.ylabel("Precision (fraction of flags that are fraud)")
plt.title("Precision-Recall Curves: Fraud Detection on Test Set")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(PATH + "fig3_pr_curves.png", dpi=150)
plt.show()


budget_rows = []
for frac in [0.001, 0.005, 0.01, 0.02]:
    for name, s in models.items():
        p, r, caught, k = prec_rec_at_budget(s, y_test, frac)
        budget_rows.append({"Budget": f"{frac*100:g}%", "Reviewed": k, "Model": name,
                            "Frauds caught": caught, "Precision": round(p, 3), "Recall": round(r, 3)})

budget_df = pd.DataFrame(budget_rows)
budget_df


np.save(PATH + "iso_scores.npy", iso_scores)
np.save(PATH + "ocsvm_scores.npy", ocsvm_scores)
metrics_df.to_csv(PATH + "table1_metrics.csv", index=False)
budget_df.to_csv(PATH + "table2_review_budget.csv", index=False)
print("Saved iso_scores, ocsvm_scores, table1_metrics.csv, table2_review_budget.csv, fig3_pr_curves.png")


import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

X_val = d["X_val"]
Xtr_t = torch.tensor(X_train, dtype=torch.float32)
Xva_t = torch.tensor(X_val,   dtype=torch.float32)
Xte_t = torch.tensor(X_test,  dtype=torch.float32)

class Autoencoder(nn.Module):
    def __init__(self, input_dim=31, bottleneck_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 20), nn.ReLU(),
            nn.Linear(20, 14), nn.ReLU(),
            nn.Linear(14, bottleneck_dim))
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 14), nn.ReLU(),
            nn.Linear(14, 20), nn.ReLU(),
            nn.Linear(20, input_dim))
    def forward(self, x):
        return self.decoder(self.encoder(x))

def train_ae(seed, bottleneck_dim=8, max_epochs=100, patience=5):
    torch.manual_seed(seed); np.random.seed(seed)
    model = Autoencoder(31, bottleneck_dim)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    loader = DataLoader(TensorDataset(Xtr_t), batch_size=256, shuffle=True)
    best_val, best_state, wait = float("inf"), None, 0
    for epoch in range(max_epochs):
        model.train()
        for (xb,) in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), xb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = loss_fn(model(Xva_t), Xva_t).item()
        if vl < best_val - 1e-6:
            best_val, wait = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        scores = torch.mean((Xte_t - model(Xte_t)) ** 2, dim=1).numpy()
    return scores, epoch + 1

def evaluate(scores):
    p1, r1, _, _ = prec_rec_at_budget(scores, y_test, 0.01)
    return {"AP": average_precision_score(y_test, scores),
            "ROC-AUC": roc_auc_score(y_test, scores),
            "P@1%": p1, "R@1%": r1}

seeds = [0, 1, 2, 3, 4]
results = []
for seed in seeds:
    ae_s, n_ep = train_ae(seed)
    results.append({"Model": "Autoencoder", "seed": seed, **evaluate(ae_s)})

    iso_s = -IsolationForest(n_estimators=100, random_state=seed, n_jobs=-1).fit(X_train).score_samples(X_test)
    results.append({"Model": "Isolation Forest", "seed": seed, **evaluate(iso_s)})

    sub_s = np.random.RandomState(seed).choice(len(X_train), 10000, replace=False)
    oc_s = -OneClassSVM(kernel="rbf", nu=0.01, gamma="scale").fit(X_train[sub_s]).score_samples(X_test)
    results.append({"Model": "One-Class SVM", "seed": seed, **evaluate(oc_s)})

    print(f"seed {seed} done (AE stopped at epoch {n_ep}) | "
          f"AE AP={results[-3]['AP']:.3f}  IF AP={results[-2]['AP']:.3f}  OC-SVM AP={results[-1]['AP']:.3f}")

seed_df = pd.DataFrame(results)
seed_df.to_csv(PATH + "table1_multiseed_raw.csv", index=False)

summary = seed_df.groupby("Model")[["AP", "ROC-AUC", "P@1%", "R@1%"]].agg(["mean", "std"]).round(3)
summary.to_csv(PATH + "table1_multiseed_summary.csv")
summary
