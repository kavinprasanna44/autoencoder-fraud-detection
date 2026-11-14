from google.colab import drive
drive.mount('/content/drive')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score

PATH = "/content/drive/MyDrive/Auto Encoder Senior Thesis/"

d = np.load(PATH + "splits.npz")
X_train, X_val, X_test = d["X_train"], d["X_val"], d["X_test"]
y_test      = d["y_test"]
amount_test = d["amount_test"]
hour_test   = d["hour_test"]

scores = np.load(PATH + "test_scores.npy")
latent = np.load(PATH + "latent_test.npy")

fraud = y_test == 1
night = hour_test <= 5

print("scores:", scores.shape, "| latent:", latent.shape, "| frauds:", int(fraud.sum()))


k = int(round(0.01 * len(scores)))
threshold = np.sort(scores)[::-1][k - 1]
flagged = scores >= threshold

TP = int(np.sum(flagged & fraud))
FN = int(np.sum(~flagged & fraud))
FP = int(np.sum(flagged & ~fraud))
TN = int(np.sum(~flagged & ~fraud))

print(f"Review budget: top {k} transactions (1%), error threshold = {threshold:.3f}")
print(f"Caught fraud (TP): {TP}   Missed fraud (FN): {FN}")
print(f"False alarms (FP): {FP}   Correctly ignored (TN): {TN}")
print(f"Recall = {TP/(TP+FN):.3f}   Precision = {TP/(TP+FP):.3f}")


groups = {
    "Caught fraud (TP)":  flagged & fraud,
    "Missed fraud (FN)":  ~flagged & fraud,
    "False alarms (FP)":  flagged & ~fraud,
    "All normal":         ~fraud,
}

rows = []
for name, m in groups.items():
    rows.append({"Group": name,
                 "Count": int(m.sum()),
                 "Median amount (€)": round(float(np.median(amount_test[m])), 2),
                 "Mean amount (€)": round(float(np.mean(amount_test[m])), 2),
                 "% at night (0-5am)": round(100 * float(night[m].mean()), 1),
                 "Median recon. error": round(float(np.median(scores[m])), 3)})

failure_df = pd.DataFrame(rows)
failure_df


from sklearn.decomposition import PCA

pca = PCA(n_components=2, random_state=42)
lat2d = pca.fit_transform(latent)
print(f"These 2 directions capture {100*pca.explained_variance_ratio_.sum():.1f}% of the variation in the 8-D bottleneck")

plt.figure(figsize=(7, 6))
plt.scatter(lat2d[~fraud, 0], lat2d[~fraud, 1], s=2, alpha=0.15, c="steelblue", label="normal")
plt.scatter(lat2d[fraud & flagged, 0], lat2d[fraud & flagged, 1], s=18, c="crimson", label="fraud — caught")
plt.scatter(lat2d[fraud & ~flagged, 0], lat2d[fraud & ~flagged, 1], s=24, c="orange", marker="x", label="fraud — missed")
plt.xlabel("Latent PC1"); plt.ylabel("Latent PC2")
plt.title("Autoencoder bottleneck (8-D) projected to 2-D")
plt.legend(); plt.tight_layout()
plt.savefig(PATH + "fig4_latent_pca.png", dpi=150)
plt.show()


def recall_in(mask):
    sub = fraud & mask
    n = int(sub.sum())
    return n, (round(float(flagged[sub].mean()), 3) if n > 0 else np.nan)

slices = {
    "≤ €1":        amount_test <= 1,
    "€1–10":       (amount_test > 1) & (amount_test <= 10),
    "€10–100":     (amount_test > 10) & (amount_test <= 100),
    "> €100":      amount_test > 100,
    "Night 0–5am": night,
    "Day 6am–11pm": ~night,
}

slice_rows = []
for name, m in slices.items():
    n, rec = recall_in(m)
    slice_rows.append({"Slice": name, "Frauds in slice": n, "Recall @1%": rec})
slice_df = pd.DataFrame(slice_rows)
display(slice_df)

colors = ["#4c72b0"] * 4 + ["#dd8452"] * 2
plt.figure(figsize=(9, 4.5))
bars = plt.bar(slice_df["Slice"], slice_df["Recall @1%"], color=colors)
for b, (_, r) in zip(bars, slice_df.iterrows()):
    plt.text(b.get_x() + b.get_width()/2, b.get_height() + 0.02,
             f"{r['Recall @1%']:.2f}\n(n={r['Frauds in slice']})", ha="center", fontsize=9)
plt.ylim(0, 1.15); plt.ylabel("Fraction of fraud caught (recall @1%)")
plt.title("Autoencoder detection rate by transaction amount and time of day")
plt.tight_layout()
plt.savefig(PATH + "fig5_detection_by_slice.png", dpi=150)
plt.show()


import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

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
        s = torch.mean((Xte_t - model(Xte_t)) ** 2, dim=1).numpy()
    return s, best_val, epoch + 1

sizes = [4, 8, 16, 32]
seeds = [0, 1, 2]
sweep_rows = []
for b in sizes:
    for seed in seeds:
        s, vmse, n_ep = train_ae(seed, bottleneck_dim=b)
        ap = average_precision_score(y_test, s)
        sweep_rows.append({"Bottleneck": b, "seed": seed, "AP": ap, "Val MSE": vmse})
        print(f"bottleneck={b:2d} seed={seed} | AP={ap:.3f} | val MSE={vmse:.4f} | epochs={n_ep}")

sweep_df = pd.DataFrame(sweep_rows)
sweep_summary = sweep_df.groupby("Bottleneck")[["AP", "Val MSE"]].agg(["mean", "std"]).round(4)
sweep_summary


m = sweep_df.groupby("Bottleneck").agg(["mean", "std"])

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
axes[0].errorbar(sizes, m[("AP", "mean")], yerr=m[("AP", "std")], marker="o", capsize=4)
axes[0].set_title("Fraud detection (higher = better)")
axes[0].set_ylabel("Average Precision")

axes[1].errorbar(sizes, m[("Val MSE", "mean")], yerr=m[("Val MSE", "std")], marker="o", capsize=4, color="#dd8452")
axes[1].set_title("Reconstruction of normal data (lower = better)")
axes[1].set_ylabel("Validation MSE")

for ax in axes:
    ax.set_xscale("log", base=2); ax.set_xticks(sizes); ax.set_xticklabels(sizes)
    ax.set_xlabel("Bottleneck dimension"); ax.grid(alpha=0.3)
plt.suptitle("Effect of bottleneck size")
plt.tight_layout()
plt.savefig(PATH + "fig6_bottleneck_sensitivity.png", dpi=150)
plt.show()


failure_df.to_csv(PATH + "table3_failure_profile.csv", index=False)
slice_df.to_csv(PATH + "table4_detection_by_slice.csv", index=False)
sweep_df.to_csv(PATH + "table5_bottleneck_sweep_raw.csv", index=False)
sweep_summary.to_csv(PATH + "table5_bottleneck_sweep_summary.csv")
print("Saved Tables 3-5 and Figures 4-6 to Drive")


xlim, ylim = (-60, 15), (-25, 25)
outside = fraud & ((lat2d[:, 0] < xlim[0]) | (lat2d[:, 0] > xlim[1]) | (lat2d[:, 1] < ylim[0]) | (lat2d[:, 1] > ylim[1]))

plt.figure(figsize=(7, 6))
plt.scatter(lat2d[~fraud, 0], lat2d[~fraud, 1], s=2, alpha=0.15, c="steelblue", label="normal")
plt.scatter(lat2d[fraud & flagged, 0], lat2d[fraud & flagged, 1], s=18, c="crimson", label="fraud — caught")
plt.scatter(lat2d[fraud & ~flagged, 0], lat2d[fraud & ~flagged, 1], s=24, c="orange", marker="x", label="fraud — missed")
plt.xlim(*xlim); plt.ylim(*ylim)
plt.xlabel("Latent PC1"); plt.ylabel("Latent PC2")
plt.title(f"Autoencoder bottleneck projected to 2-D (zoomed)\n{int(outside.sum())} of {int(fraud.sum())} fraud points fall outside this view")
plt.legend(); plt.tight_layout()
plt.savefig(PATH + "fig4_latent_pca_zoom.png", dpi=150)
plt.show()
print("fraud outside view:", int(outside.sum()), "| of which caught:", int((outside & flagged).sum()), "| missed:", int((outside & ~flagged).sum()))
