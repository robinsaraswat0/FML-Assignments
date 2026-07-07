# Q2_solution.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# === USER-ADJUSTABLE PATHS ===
TRAIN_CSV = "A2Q2Data_train - A2Q2Data_train.csv"  # adjust if needed
TEST_CSV  = "A2Q2Data_test.csv"                  # adjust if needed

np.random.seed(0)

# -------- Load data --------
df = pd.read_csv(TRAIN_CSV, header=None)
X = df.iloc[:, :-1].values.astype(float)
y = df.iloc[:, -1].values.astype(float)
n, d = X.shape
print(f"Loaded train: n={n}, d={d}")

# -------- Center data (features & response) --------
X_mean = X.mean(axis=0)
Xc = X - X_mean
y_mean = y.mean()
yc = y - y_mean

# -------- (i) Analytical least squares (w_ML) --------
# Use lstsq for numerical stability (handles singular cases)
wML, *_ = np.linalg.lstsq(Xc, yc, rcond=None)
bML = y_mean - X_mean.dot(wML)
print("Computed w_ML (analytical). ||w_ML||_2 =", np.linalg.norm(wML))

# -------- (ii) Gradient Descent --------
# Precompute XtX for efficient gradients
XtX = Xc.T.dot(Xc)
# estimate Lipschitz L = (1/n) * lambda_max(XtX)
eigvals = np.linalg.eigvalsh(XtX)
L = eigvals.max() / n
eta = 0.9 / L  # safe choice; reduce if unstable

max_iters = 2000
w = np.zeros(d)
norm_diff_gd = np.zeros(max_iters + 1)
norm_diff_gd[0] = np.linalg.norm(w - wML)

for t in range(1, max_iters + 1):
    grad = (1.0 / n) * (Xc.T.dot(Xc.dot(w) - yc))
    w = w - eta * grad
    norm_diff_gd[t] = np.linalg.norm(w - wML)
    # optional early stop if small
    if norm_diff_gd[t] < 1e-10:
        norm_diff_gd = norm_diff_gd[:t+1]
        break

# Plot GD convergence
plt.figure(figsize=(7,4))
plt.plot(np.arange(norm_diff_gd.shape[0]), norm_diff_gd)
plt.xlabel("Iteration t")
plt.ylabel("||w_t - w_ML||_2")
plt.title("Gradient Descent: ||w_t - w_ML||_2 vs iterations")
plt.grid(True)
plt.tight_layout()
plt.savefig("gd_convergence.png")
plt.show()

# -------- (iii) SGD (mini-batch of 100) --------
batch_size = 100
updates_per_epoch = n // batch_size
epochs = 50
total_updates = updates_per_epoch * epochs

w_sgd = np.zeros(d)
norm_diff_sgd = np.zeros(total_updates + 1)
norm_diff_sgd[0] = np.linalg.norm(w_sgd - wML)
eta_sgd = 0.03  # smaller than GD step

u = 0
for epoch in range(epochs):
    perm = np.random.permutation(Xc.shape[0])
    Xs = Xc[perm]
    ys = yc[perm]
    for i in range(updates_per_epoch):
        start = i * batch_size
        end = start + batch_size
        Xbatch = Xs[start:end]
        ybatch = ys[start:end]
        grad_batch = (1.0 / batch_size) * (Xbatch.T.dot(Xbatch.dot(w_sgd) - ybatch))
        w_sgd = w_sgd - eta_sgd * grad_batch
        u += 1
        norm_diff_sgd[u] = np.linalg.norm(w_sgd - wML)

# Plot SGD convergence
plt.figure(figsize=(7,4))
plt.plot(np.arange(total_updates + 1), norm_diff_sgd)
plt.xlabel("Update number (t)")
plt.ylabel("||w_t - w_ML||_2")
plt.title("SGD (batch=100): ||w_t - w_ML||_2 vs updates")
plt.grid(True)
plt.tight_layout()
plt.savefig("sgd_convergence.png")
plt.show()

# -------- (iv) Ridge regression (CV for lambda) --------
# split train-> train_cv (80%) and val (20%)
train_data = pd.read_csv("A2Q2Data_train - A2Q2Data_train.csv",header=None)
test_data = pd.read_csv("A2Q2Data_test - A2Q2Data_test.csv",header=None)

X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

lambdas = np.logspace(-4, 2, 20)
epochs = 1280
lr = 0.001
k = 5  # number of folds
X_train = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)
X_test = (X_test - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)
trainX_ = np.hstack([np.ones((X_train.shape[0], 1)), X_train])

trainy = y_train
d = trainX_.shape[1]    
n = trainX_.shape[0]
indices = np.arange(n)

np.random.seed(42)
np.random.shuffle(indices)
fold_size = n // k
folds = [indices[i*fold_size:(i+1)*fold_size] for i in range(k)]

val_errors, train_errors, all_ws = [], [], []

for lam in lambdas:
    print(f"\nTraining with lambda: {lam:.6f}")
    fold_train_mse, fold_val_mse = [], []

    # Iterate through folds
    for fold in range(k):
        val_idx = folds[fold]
        train_idx = np.hstack([folds[i] for i in range(k) if i != fold])

        Xtrain, ytrain = trainX_[train_idx], trainy[train_idx]
        valX, valy = trainX_[val_idx], trainy[val_idx]

        np.random.seed(fold)
        w_ridge = np.random.randn(d)
        for j in range(epochs):
            n= Xtrain.shape[0]
            y_pred = Xtrain @ w_ridge
            diff = y_pred - ytrain
            grad = (2 / n) * (Xtrain.T @ diff)
            grad[1:] += 2 * lam * w_ridge[1:]
            err = (diff.T @ diff) / n + lam * np.sum(w_ridge[1:] ** 2)

            w_ridge -= lr * grad

        # Compute train and val MSE
        train_mse = np.mean((Xtrain @ w_ridge - ytrain) ** 2)
        val_mse = np.mean((valX @ w_ridge - valy) ** 2)

        fold_train_mse.append(train_mse)
        fold_val_mse.append(val_mse)

    # Average across folds
    mean_train_mse = np.mean(fold_train_mse)
    mean_val_mse = np.mean(fold_val_mse)

    train_errors.append(mean_train_mse)
    val_errors.append(mean_val_mse)
    all_ws.append(w_ridge.copy())

    print(f"Mean Validation MSE: {mean_val_mse:.6f}")

color = "#D55E00"
plt.figure(figsize=(10, 6))
plt.plot(lambdas, val_errors, marker='o', label="Validation error", color=color, linewidth=2)
plt.plot(lambdas, train_errors, marker='o', label="Training error", color='skyblue', linewidth=2)

plt.xlabel('Lambda (λ)', fontsize=12)
plt.ylabel('Mean Squared Error', fontsize=12)
plt.title('Ridge Regression - Train & Val Error vs λ', fontsize=14, fontweight="bold")

plt.grid(True, linestyle="--", alpha=0.6)
plt.yticks(fontsize=10)
plt.legend(frameon=False, fontsize=11)
plt.tight_layout()

# plt.savefig(f"Plots/train_val_error_vs_lambdas.png", dpi=300, bbox_inches="tight")
plt.show()
best_idx = np.argmin(val_errors)
best_lambda = lambdas[best_idx]
wr = all_ws[best_idx]
testX_ = np.hstack([np.ones((X_test.shape[0], 1)), X_test])
testy = y_test
test_mse_ridge = np.mean((testX_ @ wr - testy) ** 2)
train_mse_ridge = np.mean((trainX_ @ wr - trainy) ** 2)
print(f"\nBest λ: {best_lambda:.6f}")
print(f"Train MSE (Ridge): {train_mse_ridge:.6f}")
print(f"Test MSE (Ridge): {test_mse_ridge:.6f}")