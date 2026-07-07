import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("dataset1.csv", header=None)

n_samples, n_features = df.shape

def compute_sigma(data):
    dists = []
    n = len(data)
    for i in range(n):
        for j in range(i+1, n):
            dx = data[i][0] - data[j][0]
            dy = data[i][1] - data[j][1]
            d2 = dx*dx + dy*dy
            dists.append(d2)
    dists.sort()
    median_d2 = dists[len(dists)//2]
    sigma = (median_d2 / 2.0) ** 0.5
    return sigma

data = df.to_numpy()
sigma = compute_sigma(data)
print("Recommended sigma:", sigma)

def meanFinder(nums):
    total = 0
    for num in nums:
        total += num
    return total / len(nums)

def Covariance(a):
    return (a.T @ a) / a.shape[0]

def run_standard_pca(data):
    print("--- Running Standard PCA ---")
    mean_vec = np.mean(data, axis=0)
    centered_data = data - mean_vec
    cov_matrix = (centered_data.T @ centered_data) / centered_data.shape[0]
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    projected_data = df @ eigenvectors
    total_variance = meanFinder(eigenvalues) * len(eigenvalues)
    explained_variance_ratio = eigenvalues / total_variance
    print(f"Explained variance by PC1: {explained_variance_ratio[0]:.4f}")
    print(f"Explained variance by PC2: {explained_variance_ratio[1]:.4f}\n")
    return projected_data, explained_variance_ratio

def rbf_kernel(x1, x2, sigma=1.0):
    distance_sq = np.sum((x1 - x2) ** 2)
    return np.exp(-distance_sq / (2 * sigma ** 2))

def polynomial_kernel(x1, x2, degree=3, coef0=1):
    return (x1.T @ x2 + coef0) ** degree

def run_kernel_pca(data, kernel_func, **args):
    n_samples = data.shape[0]
    K = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            row_i = data.iloc[i].values
            row_j = data.iloc[j].values
            K[i, j] = kernel_func(row_i, row_j, **args)
    one_n = np.ones((n_samples, n_samples)) / n_samples
    K_centered = K - one_n @ K - K @ one_n + one_n @ K @ one_n
    eigenvalues, eigenvectors = np.linalg.eig(K_centered)
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    total = eigenvalues.sum()
    explained_ratio = eigenvalues / total if total > 0 else np.zeros_like(eigenvalues)
    print("Top 6 eigenvalues:", eigenvalues[:6])
    print(f"Explained variance by PC1: {explained_ratio[0]:.6f}")
    if len(eigenvalues) >= 2:
        print(f"Explained variance by PC1+PC2: {(explained_ratio[0] + explained_ratio[1]):.6f}")
    projection = eigenvectors[:, :2]
    return projection


colors = ['red' if i < n_samples // 2 else 'blue' for i in range(n_samples)]

if __name__ == "__main__":
    pca_projected_data, explained_variance = run_standard_pca(df)

    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    plt.title('Original Data')
    plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=colors, alpha=0.6)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True)
    plt.axis('equal')

    plt.subplot(1, 2, 2)
    plt.title('Standard PCA Projection')
    plt.scatter(pca_projected_data.iloc[:, 0], pca_projected_data.iloc[:, 1],
                c=colors, alpha=0.6)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    plt.axis('equal')
    plt.show()

    print("--- Running Kernel PCA ---")
    kpca_poly = run_kernel_pca(df, polynomial_kernel)
    print("hello")

    sigmas = [0.1, 0.7, 0.98, 1.3239006441423413]
    kpca_rbfs = [run_kernel_pca(df, rbf_kernel, sigma=s) for s in sigmas]
    print("RBF")

    plt.figure(figsize=(8, 6))
    plt.title('Kernel PCA: Polynomial Kernel (degree=2)')
    plt.scatter(kpca_poly[:, 0], kpca_poly[:, 1], c=colors, alpha=0.6)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    plt.axis('equal')
    plt.show()
    print("done")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Kernel PCA: RBF Kernel with Different Sigma Values', fontsize=16)
    axes = axes.ravel()
    for i, s in enumerate(sigmas):
        ax = axes[i]
        projection = kpca_rbfs[i]
        ax.scatter(projection[:, 0], projection[:, 1], c=colors, alpha=0.6)
        ax.set_title(f'Sigma (σ) = {s}')
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.grid(True)
        ax.axis('equal')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
