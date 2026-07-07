import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_distance_sq(point1, point2):
    return np.sum((point1 - point2)**2)

def kmeans_from_scratch(data, k=4, max_iterations=100, initial_centroids=None):
    n_samples, n_features = data.shape
    
    if initial_centroids is None:
        random_indices = np.random.choice(n_samples, k, replace=False)
        centroids = data[random_indices, :]
    else:
        centroids = initial_centroids
    
    error_history = []

    for iteration in range(max_iterations):
        labels = np.zeros(n_samples)
        for i, point in enumerate(data):
            distances = [calculate_distance_sq(point, centroid) for centroid in centroids]
            labels[i] = np.argmin(distances)
        
        current_error = 0
        for i, point in enumerate(data):
            assigned_centroid = centroids[int(labels[i])]
            current_error += calculate_distance_sq(point, assigned_centroid)
        error_history.append(current_error)

        new_centroids = np.zeros((k, n_features))
        for j in range(k):
            points_in_cluster = data[labels == j]
            if len(points_in_cluster) > 0:
                new_centroids[j] = np.mean(points_in_cluster, axis=0)
            else:
                new_centroids[j] = data[np.random.choice(n_samples)]
        
        if np.all(centroids == new_centroids):
            break
            
        centroids = new_centroids
        
    return centroids, labels, error_history

def run_and_plot_kmeans_experiment(data, k, num_runs):

    print(f"--- Running K-means with k={k} for {num_runs} different initializations ---")
    
    for i in range(num_runs):
        print(f"Running initialization {i+1}/{num_runs}...")
        
        np.random.seed(i * 42) 
        final_centroids, final_labels, error_history = kmeans_from_scratch(data, k=k)
        
        fig, (ax_cluster, ax_error) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f'K-means Initialization {i+1} (k={k}): Clusters vs Error Function', fontsize=14, fontweight='bold')
        
        unique_labels = np.unique(final_labels)
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
        
        for j, label in enumerate(unique_labels):
            mask = final_labels == label
            ax_cluster.scatter(data[mask, 0], data[mask, 1], c=[colors[j]], 
                             alpha=0.7, s=40, label=f'Cluster {int(label)}', edgecolors='black', linewidth=0.5)
        
        ax_cluster.scatter(final_centroids[:, 0], final_centroids[:, 1], c='red', marker='X', s=300, 
                          edgecolors='black', linewidth=2, label='Centroids', zorder=10)
        
        ax_cluster.set_title(f'Final Clusters (Found {len(unique_labels)} clusters)')
        ax_cluster.set_xlabel('Feature 1')
        ax_cluster.set_ylabel('Feature 2')
        ax_cluster.grid(True, alpha=0.3)
        ax_cluster.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax_cluster.set_aspect('equal', adjustable='box')
        
        ax_error.plot(range(len(error_history)), error_history, marker='o', linestyle='-', 
                     color='red', linewidth=2, markersize=6, markerfacecolor='darkred', markeredgecolor='black')
        ax_error.set_title(f'Error Function Convergence')
        ax_error.set_xlabel('Iteration')
        ax_error.set_ylabel('Sum of Squared Distances (Error)')
        ax_error.grid(True, alpha=0.3)
        
        if len(error_history) > 1:
            ax_error.set_xlim(-0.5, len(error_history)-0.5)
            error_range = max(error_history) - min(error_history)
            if error_range > 0:
                ax_error.set_ylim(min(error_history) - error_range*0.05, 
                                 max(error_history) + error_range*0.05)
        
        final_error = error_history[-1] if error_history else 0
        iterations_to_converge = len(error_history)
        improvement = ((error_history[0] - error_history[-1]) / error_history[0] * 100) if len(error_history) > 1 and error_history[0] > 0 else 0
        
        info_text = f'Final Error: {final_error:.1f}\nIterations: {iterations_to_converge}\nImprovement: {improvement:.1f}%'
        ax_error.text(0.02, 0.98, info_text, transform=ax_error.transAxes, verticalalignment='top', 
                     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        cluster_sizes = [np.sum(final_labels == j) for j in unique_labels]
        stats_text = f'Cluster sizes: {cluster_sizes}\nTotal points: {len(data)}'
        ax_cluster.text(0.02, 0.02, stats_text, transform=ax_cluster.transAxes, verticalalignment='bottom',
                       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        
        print(f"  -> Found {len(unique_labels)} clusters with sizes: {cluster_sizes}")
        print(f"  -> Converged in {iterations_to_converge} iterations with final error: {final_error:.2f}")
        print()

def run_and_plot_voronoi_experiment(data, k_values):

    print("\n--- Running K-means with Fixed Initialization and Plotting Voronoi Regions ---")
    np.random.seed(42) 
    max_k = max(k_values)
    random_indices = np.random.choice(data.shape[0], max_k, replace=False)
    fixed_initial_centroids_full = data[random_indices, :]
    fig, axes = plt.subplots(1, len(k_values), figsize=(20, 5), sharex=True, sharey=True)
    fig.suptitle('K-means Voronoi Regions for a Fixed Initialization', fontsize=16)
    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    for i, k in enumerate(k_values):
        print(f"Running for K = {k}...")
        ax = axes[i]
        initial_centroids = fixed_initial_centroids_full[:k, :]
        final_centroids, final_labels, _ = kmeans_from_scratch(data, k=k, initial_centroids=initial_centroids)
        grid_labels = np.zeros(len(grid_points))
        for j, point in enumerate(grid_points):
            distances = [calculate_distance_sq(point, centroid) for centroid in final_centroids]
            grid_labels[j] = np.argmin(distances)
        grid_labels = grid_labels.reshape(xx.shape)
        ax.contourf(xx, yy, grid_labels, cmap=plt.cm.viridis, alpha=0.3)
        ax.scatter(data[:, 0], data[:, 1], c=final_labels, cmap='viridis', edgecolor='k', s=20)
        ax.scatter(final_centroids[:, 0], final_centroids[:, 1], c='red', marker='X', s=200, label='Centroids')
        ax.set_title(f'K = {k}')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.legend()
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

# --- NEW FUNCTIONS FOR QUESTION 2(c) ---

def rbf_kernel(x1, x2, sigma=1.0):
    distance_sq = np.sum((x1 - x2)**2)
    return np.exp(-distance_sq / (2 * sigma**2))

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

def kmeans(data, k, max_iterations=100, random_state=42):
    np.random.seed(random_state)
    
    random_indices = np.random.choice(data.shape[0], size=k, replace=False)
    centroids = data[random_indices, :]

    for _ in range(max_iterations):
        clusters = np.zeros(data.shape[0], dtype=int)
        for i, point in enumerate(data):
            distances = [euclidean_distance(point, centroid) for centroid in centroids]
            cluster_index = np.argmin(distances)
            clusters[i] = cluster_index

        new_centroids = np.zeros((k, data.shape[1]))
        for i in range(k):
            points_in_cluster = data[clusters == i]
            if len(points_in_cluster) == 0:
                new_centroids[i] = data[np.random.choice(data.shape[0])]
            else:
                new_centroids[i] = np.mean(points_in_cluster, axis=0)
        
        if np.allclose(centroids, new_centroids):
            break
        
        centroids = new_centroids
    return clusters

def compute_rbf_affinity(data, sigma):
    sq_dists = -2 * np.dot(data, data.T) + np.sum(data**2, axis=1) + np.sum(data**2, axis=1)[:, np.newaxis]
    
    two_sigma_sq = 2 * (sigma ** 2)
    
    affinity_matrix = np.exp(-sq_dists / two_sigma_sq)
    np.fill_diagonal(affinity_matrix, 0)
    
    return affinity_matrix
def spectral_clustering(data, n_clusters, sigma):
    affinity_matrix = compute_rbf_affinity(data, sigma)

    degree_matrix = np.diag(affinity_matrix.sum(axis=1))

    epsilon = 1e-8
    degree_inv_sqrt = np.linalg.inv(np.sqrt(degree_matrix + epsilon))
    laplacian_matrix = np.eye(data.shape[0]) - degree_inv_sqrt @ affinity_matrix @ degree_inv_sqrt

    eigvals, eigvecs = np.linalg.eigh(laplacian_matrix)
    
    indices = np.argsort(eigvals)[:n_clusters]
    embedding = eigvecs[:, indices]
    norm = np.linalg.norm(embedding, axis=1, keepdims=True)
    embedding_normalized = embedding / (norm + epsilon)
    
    labels = kmeans(embedding_normalized, k=n_clusters, random_state=42)
    
    return labels

def plotting_spectral_clustering_data(data, all_labels, num_clusters, sigma_values, title):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    cmap = plt.get_cmap('viridis', num_clusters)

    for i, sigma in enumerate(sigma_values):
        ax = axes[i]
        labels = all_labels[sigma]
        
        for j in range(num_clusters):
            cluster_points = data[labels == j]
            ax.scatter(
                cluster_points[:, 0],
                cluster_points[:, 1],
                color=cmap(j),
                label=f'Cluster {j + 1}'
            )
        ax.set_title(rf'Spectral Clustering with $\sigma$={sigma}', fontsize=14)
        ax.set_xlabel('X-axis', fontsize=10)
        ax.set_ylabel('Y-axis', fontsize=10)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)

    fig.suptitle(title, fontsize=20, y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.show()



def rbf_kernel_arg(X, Y=None, sigma=None):
    if Y is None:
        Y = X
    if sigma is None:
        gamma = X.shape[1] 
    else:
        gamma = 1.0 / (sigma ** 2)

    sq_dists = (
        np.sum(X ** 2, axis=1).reshape(-1, 1)
        + np.sum(Y ** 2, axis=1)
        - 2 * np.dot(X, Y.T)
    )
    return np.exp(-gamma * sq_dists)
def spectral_clustering_arg(X, n_clusters=4, sigma=None):
    K = rbf_kernel_arg(X, sigma=sigma)
    np.fill_diagonal(K, 0)

    D = np.diag(K.sum(axis=1))
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D)))
    L_sym = np.eye(X.shape[0]) - D_inv_sqrt @ K @ D_inv_sqrt

    eigvals, eigvecs = np.linalg.eigh(L_sym)
    idx = np.argsort(eigvals)[:n_clusters]
    eigvecs = eigvecs[:, idx]

    embedding = eigvecs / np.linalg.norm(eigvecs, axis=1, keepdims=True)

    labels = np.argmax(embedding, axis=1)
    return labels

def run_and_plot_kmeans_with_combined_errors(data, k, num_runs):

    fig_clusters, axes_clusters = plt.subplots(1, num_runs, figsize=(25, 5), sharey=True, sharex=True)
    fig_clusters.suptitle(f'K-means Clustering Results (k={k}) for {num_runs} Different Random Initializations', fontsize=16)
    
    fig_error, ax_error = plt.subplots(figsize=(12, 8))
    
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i in range(num_runs):
        print(f"Running iteration {i+1}/{num_runs}...")
        final_centroids, final_labels, error_history = kmeans_from_scratch(data, k=k)
        
        ax = axes_clusters[i]
        ax.scatter(data[:, 0], data[:, 1], c=final_labels, cmap='viridis', alpha=0.7)
        ax.scatter(final_centroids[:, 0], final_centroids[:, 1], c='red', marker='X', s=200, label='Centroids')
        ax.set_title(f'Run {i+1} Final Clusters')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.grid(True)
        ax.legend()
        ax.set_aspect('equal', adjustable='box')
        
        color = colors[i % len(colors)]
        ax_error.plot(range(len(error_history)), error_history, marker='o', linestyle='-', 
                     color=color, linewidth=2, label=f'Run {i+1}', markersize=6)
    
    ax_error.set_title('K-means Objective Function (Error) vs. Iterations - All Runs', fontsize=14)
    ax_error.set_xlabel('Iteration')
    ax_error.set_ylabel('Sum of Squared Distances (Error)')
    ax_error.grid(True, alpha=0.3)
    ax_error.legend()
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def plot_spectral_results(data, sigmas, n_clusters=4, rows=3, cols=5):

    fig, axes = plt.subplots(rows, cols, figsize=(20, 12))
    axes = axes.flatten()

    for i, s in enumerate(sigmas):
        labels = spectral_clustering_arg(data, n_clusters=n_clusters, sigma=s)
        sns.scatterplot(
            x=data[:, 0], y=data[:, 1], hue=labels,
            palette="viridis", s=40, edgecolor="white", linewidth=0.5,
            ax=axes[i], legend=False
        )
        axes[i].set_title(f"σ = {s:.2f}", fontsize=12)
        axes[i].set_xticks([]); axes[i].set_yticks([])

    for j in range(len(sigmas), len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle("Spectral Clustering with different RBF values",
                 fontsize=16, weight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

if __name__ == "__main__":
    df = pd.read_csv("dataset2.csv", header=None)
    data = df.values
    
    # --- Execute for Question 2(a) with new side-by-side layout ---
    k_clusters = 4
    iteration = 5
    run_and_plot_kmeans_experiment(data, k=k_clusters, num_runs=iteration)
 
    fixed_k = [2, 3, 4, 5]
    run_and_plot_voronoi_experiment(data, k_values=fixed_k)
    

    num_clusters = 4 
    sigma_values = [0.1, 0.25, 0.5, 0.75]

    all_cluster_labels = {}
    
    for sigma in sigma_values:
        print(f" - Processing for sigma = {sigma}")
        labels = spectral_clustering(
            data=data,
            n_clusters=num_clusters,
            sigma=sigma
        )
        all_cluster_labels[sigma] = labels
    print("Clustering complete.")

    plotting_spectral_clustering_data(
        data=data,
        all_labels=all_cluster_labels,
        num_clusters=num_clusters,
        sigma_values=sigma_values,
        title=f'Spectral Clustering with RBF Kernel (k={num_clusters})',
    )

    sigmas = np.linspace(0.31, 0.35, 10)
    fig, axes = plt.subplots(2, 5, figsize=(20, 12))
    axes = axes.flatten()

    for i, s in enumerate(sigmas):
        labels = spectral_clustering_arg(data, n_clusters=4, sigma=s)
        sns.scatterplot(
            x=data[:, 0], y=data[:, 1], hue=labels,
            palette="viridis", s=40, edgecolor="white", linewidth=0.5,
            ax=axes[i], legend=False
        )
        axes[i].set_title(f"σ = {s:.2f}", fontsize=12)
        axes[i].set_xticks([]); axes[i].set_yticks([])

    for j in range(len(sigmas), len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle("Spectral Clustering with Sigma Values",
                 fontsize=16, weight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()