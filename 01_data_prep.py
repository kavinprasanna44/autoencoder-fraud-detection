import pandas as pd
import numpy as np

from google.colab import drive
drive.mount('/content/drive')

import os
print(os.path.exists("/content/drive/MyDrive/Auto Encoder Senior Thesis/creditcard.csv"))

df = pd.read_csv("/content/drive/MyDrive/Auto Encoder Senior Thesis/creditcard.csv")

print("Shape:", df.shape)
print("\nColumns:", list(df.columns))
print("\nClass balance:")
print(df["Class"].value_counts())
print(df["Class"].value_counts(normalize=True))
print("\nMissing values:", df.isnull().sum().sum())
print("\nAmount summary:")
print(df["Amount"].describe())
print("\nTime span (hours):", round(df["Time"].max() / 3600, 1))
df.head()


import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(df[df["Class"]==0]["Amount"], bins=100, alpha=0.6, label="normal", density=True)
axes[0].hist(df[df["Class"]==1]["Amount"], bins=100, alpha=0.6, label="fraud", density=True)
axes[0].set_title("Raw Amount (heavily skewed)")
axes[0].set_xlabel("Amount"); axes[0].set_ylabel("density"); axes[0].legend()

axes[1].hist(np.log1p(df[df["Class"]==0]["Amount"]), bins=100, alpha=0.6, label="normal", density=True)
axes[1].hist(np.log1p(df[df["Class"]==1]["Amount"]), bins=100, alpha=0.6, label="fraud", density=True)
axes[1].set_title("log(1 + Amount)")
axes[1].set_xlabel("log Amount"); axes[1].set_ylabel("density"); axes[1].legend()

plt.tight_layout()
plt.show()

print("Median amount — normal:", df[df.Class==0]["Amount"].median(),
      "| fraud:", df[df.Class==1]["Amount"].median())


df["Hour"] = (df["Time"] // 3600) % 24

hourly = df.groupby("Hour")["Class"].agg(["mean", "count"])

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].bar(hourly.index, hourly["count"])
axes[0].set_title("Transaction volume by hour")
axes[0].set_xlabel("hour of day"); axes[0].set_ylabel("num transactions")

axes[1].bar(hourly.index, hourly["mean"] * 100)
axes[1].set_title("Fraud rate by hour")
axes[1].set_xlabel("hour of day"); axes[1].set_ylabel("% fraud")

plt.tight_layout()
plt.show()

print(hourly)


df["LogAmount"] = np.log1p(df["Amount"])

df["Hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
df["Hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)

feature_cols = [f"V{i}" for i in range(1, 29)] + ["LogAmount", "Hour_sin", "Hour_cos"]
print("Number of features:", len(feature_cols))
print("Features:", feature_cols)


from sklearn.model_selection import train_test_split

normal = df[df["Class"] == 0]
fraud  = df[df["Class"] == 1]

normal_train, normal_temp = train_test_split(normal, test_size=0.30, random_state=42)

normal_val, normal_test = train_test_split(normal_temp, test_size=0.50, random_state=42)

test_df = pd.concat([normal_test, fraud]).sample(frac=1.0, random_state=42)

print("Train (normal only):", len(normal_train))
print("Val   (normal only):", len(normal_val))
print("Test  (normal+fraud):", len(test_df))
print("Frauds in test:", test_df["Class"].sum(), "— expected 492")


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(normal_train[feature_cols].values)
X_val   = scaler.transform(normal_val[feature_cols].values)
X_test  = scaler.transform(test_df[feature_cols].values)

y_test       = test_df["Class"].values
amount_test  = test_df["Amount"].values
hour_test    = test_df["Hour"].values

print("X_train:", X_train.shape)
print("X_val:  ", X_val.shape)
print("X_test: ", X_test.shape)
print("Frauds in y_test:", y_test.sum())


import numpy as np

np.savez_compressed(
    "/content/drive/MyDrive/Auto Encoder Senior Thesis/splits.npz",
    X_train=X_train,
    X_val=X_val,
    X_test=X_test,
    y_test=y_test,
    amount_test=amount_test,
    hour_test=hour_test
)

import joblib
joblib.dump(scaler, "/content/drive/MyDrive/Auto Encoder Senior Thesis/scaler.joblib")

print("Saved splits.npz and scaler.joblib to Drive")
