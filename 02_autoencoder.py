import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from google.colab import drive
drive.mount('/content/drive')

d = np.load("/content/drive/MyDrive/Auto Encoder Senior Thesis/splits.npz")
X_train = d["X_train"]
X_val   = d["X_val"]
X_test  = d["X_test"]
y_test  = d["y_test"]

print("X_train:", X_train.shape)
print("X_val:  ", X_val.shape)
print("X_test: ", X_test.shape)
print("Frauds in test:", y_test.sum())


X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_val_t   = torch.tensor(X_val,   dtype=torch.float32)
X_test_t  = torch.tensor(X_test,  dtype=torch.float32)

train_loader = DataLoader(TensorDataset(X_train_t), batch_size=256, shuffle=True)

print("Tensors created")
print("Number of training batches:", len(train_loader))


class Autoencoder(nn.Module):
    def __init__(self, input_dim=31, bottleneck_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 20),
            nn.ReLU(),
            nn.Linear(20, 14),
            nn.ReLU(),
            nn.Linear(14, bottleneck_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 14),
            nn.ReLU(),
            nn.Linear(14, 20),
            nn.ReLU(),
            nn.Linear(20, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat
import random
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
model = Autoencoder()
print(model)
print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")


optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

train_losses = []
val_losses = []
best_val_loss = float("inf")
best_state = None
patience = 5
no_improve = 0

for epoch in range(100):
    model.train()
    epoch_loss = 0.0
    for (batch,) in train_loader:
        optimizer.zero_grad()
        output = model(batch)
        loss = loss_fn(output, batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * batch.size(0)
    epoch_loss /= len(train_loader.dataset)
    train_losses.append(epoch_loss)

    model.eval()
    with torch.no_grad():
        val_output = model(X_val_t)
        val_loss = loss_fn(val_output, X_val_t).item()
    val_losses.append(val_loss)

    if val_loss < best_val_loss - 1e-6:
        best_val_loss = val_loss
        no_improve = 0
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    if epoch % 5 == 0:
        print(f"Epoch {epoch:3d}: train={epoch_loss:.5f}  val={val_loss:.5f}")

model.load_state_dict(best_state)
print(f"\nBest val loss: {best_val_loss:.5f}")


plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="train loss")
plt.plot(val_losses, label="val loss")
plt.xlabel("epoch")
plt.ylabel("MSE reconstruction loss")
plt.title("Autoencoder training curves")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("/content/drive/MyDrive/Auto Encoder Senior Thesis/fig1_loss_curves.png", dpi=150)
plt.show()


def reconstruction_error(model, X_tensor):
    model.eval()
    with torch.no_grad():
        X_hat = model(X_tensor)
        errors = torch.mean((X_tensor - X_hat) ** 2, dim=1)
    return errors.numpy()

test_scores = reconstruction_error(model, X_test_t)

print(f"Median reconstruction error:")
print(f"  normal: {np.median(test_scores[y_test == 0]):.4f}")
print(f"  fraud:  {np.median(test_scores[y_test == 1]):.4f}")
print(f"  ratio:  {np.median(test_scores[y_test == 1]) / np.median(test_scores[y_test == 0]):.1f}x")

plt.figure(figsize=(8, 5))
upper = np.percentile(test_scores, 99.5)
bins = np.linspace(0, upper, 60)
plt.hist(test_scores[y_test == 0], bins=bins, alpha=0.6, density=True, label="normal")
plt.hist(test_scores[y_test == 1], bins=bins, alpha=0.6, density=True, label="fraud")
plt.xlabel("reconstruction error (per-transaction MSE)")
plt.ylabel("density")
plt.title("Reconstruction error: normal vs. fraud")
plt.legend()
plt.tight_layout()
plt.savefig("/content/drive/MyDrive/Auto Encoder Senior Thesis/fig2_error_histograms.png", dpi=150)
plt.show()


np.save("/content/drive/MyDrive/Auto Encoder Senior Thesis/test_scores.npy", test_scores)
torch.save(model.state_dict(), "/content/drive/MyDrive/Auto Encoder Senior Thesis/best_autoencoder.pt")

model.eval()
with torch.no_grad():
    latent_test = model.encoder(X_test_t).numpy()
np.save("/content/drive/MyDrive/Auto Encoder Senior Thesis/latent_test.npy", latent_test)

print("Saved:")
print("  test_scores.npy      - autoencoder anomaly scores for test set")
print("  best_autoencoder.pt  - trained model weights")
print("  latent_test.npy      - 8-D bottleneck representations for test set")
print(f"  latent_test shape: {latent_test.shape}")
