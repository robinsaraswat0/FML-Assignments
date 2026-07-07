#!/usr/bin/env python3
# A2Q1: Bernoulli-mixture EM, Gaussian-mixture EM, and K-means (from scratch)
# Requires: numpy, matplotlib, scipy (for multivariate normal pdf in GMM part; optional)
import numpy as np
import matplotlib.pyplot as plt
from math import log, exp
from copy import deepcopy

# ---------- utilities ----------
def load_data(csv_path):
    # Expect shape (N, D)
    return np.loadtxt(csv_path, delimiter=',')  # modify if header present

def logsumexp(a, axis=None):
    a_max = np.max(a, axis=axis, keepdims=True)
    res = a_max + np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True))
    if axis is None:
        return res.squeeze()
    return res.squeeze()

# ---------- Bernoulli mixture EM (from scratch) ----------
def bernoulli_mixture_em(X, K=4, max_iter=200, tol=1e-6, init_seed=None, smoothing=1e-6):
    # X: (N, D) binary 0/1
    N, D = X.shape
    rng = np.random.RandomState(init_seed)
    # initialize pi uniformly + small noise
    pi = np.ones(K) / K
    # initialize theta randomly in (0.25,0.75)
    theta = rng.uniform(0.25, 0.75, size=(K, D))

    log_likes = []
    for it in range(max_iter):
        # --- E-step: compute log responsibilities ---
        # log p(x_i | k) = sum_d x_id log theta_kd + (1-x_id) log(1-theta_kd)
        # shape (K, D) -> we compute for all i via dot product
        # we'll compute log_prob matrix L (N, K)
        log_theta = np.log(theta + smoothing)
        log_one_minus = np.log(1 - theta + smoothing)
        # compute per i,k log likelihood
        L = X.dot(log_theta.T) + (1 - X).dot(log_one_minus.T)  # (N, K)
        log_pi = np.log(pi + 1e-12)
        log_num = L + log_pi  # (N,K)
        # normalize to get responsibilities r_{ik}
        log_denom = logsumexp(log_num, axis=1)  # (N,)
        log_resp = log_num - log_denom[:, None]
        resp = np.exp(log_resp)  # (N, K)

        # --- compute observed log-likelihood ---
        ll = np.sum(log_denom)
        log_likes.append(ll)

        # --- M-step ---
        Nk = resp.sum(axis=0)  # (K,)
        pi = Nk / N
        # update theta: weighted average of x
        theta = (resp.T @ X) / Nk[:, None]  # shape (K, D)
        # clip to avoid exactly 0/1
        eps = 1e-9
        theta = np.clip(theta, eps, 1 - eps)

        # check convergence
        if it > 0 and abs(log_likes[-1] - log_likes[-2]) < tol:
            break

    return {'pi': pi, 'theta': theta, 'loglikes': np.array(log_likes), 'resp': resp, 'iters': len(log_likes)}

# ---------- Gaussian mixture EM (from scratch) ----------
def gaussian_mixture_em(X, K=4, max_iter=200, tol=1e-6, init_seed=None, reg_covar=1e-6):
    N, D = X.shape
    rng = np.random.RandomState(init_seed)
    # init means by randomly picking K data points
    perm = rng.permutation(N)
    mu = X[perm[:K]].copy()
    # init covariances as sample cov + small noise
    covs = np.array([np.cov(X, rowvar=False) + reg_covar * np.eye(D) for _ in range(K)])
    pi = np.ones(K) / K

    log_likes = []
    from numpy.linalg import slogdet, inv

    for it in range(max_iter):
        # E-step: compute log N(x|mu_k, cov_k)
        log_prob = np.zeros((N, K))
        for k in range(K):
            # multivariate normal log-pdf
            diff = X - mu[k]
            sign, ld = slogdet(covs[k])
            invC = inv(covs[k])
            quad = np.einsum('ni,ij,nj->n', diff, invC, diff)
            log_prob[:, k] = -0.5 * (D * np.log(2 * np.pi) + ld + quad)
        log_num = log_prob + np.log(pi + 1e-12)
        log_denom = logsumexp(log_num, axis=1)
        log_resp = log_num - log_denom[:, None]
        resp = np.exp(log_resp)
        ll = np.sum(log_denom)
        log_likes.append(ll)

        Nk = resp.sum(axis=0)
        pi = Nk / N
        mu = (resp.T @ X) / Nk[:, None]
        for k in range(K):
            diff = X - mu[k]
            covs[k] = (resp[:, k][:, None] * diff).T @ diff / Nk[k]
            covs[k] += reg_covar * np.eye(D)
        if it > 0 and abs(log_likes[-1] - log_likes[-2]) < tol:
            break

    return {'pi': pi, 'mu': mu, 'covs': covs, 'loglikes': np.array(log_likes), 'resp': resp, 'iters': len(log_likes)}

# ---------- K-means (from scratch) ----------
def kmeans(X, K=4, max_iter=200, tol=1e-6, init_seed=None):
    N, D = X.shape
    rng = np.random.RandomState(init_seed)
    perm = rng.permutation(N)
    centers = X[perm[:K]].copy()
    obj_vals = []
    labels = np.zeros(N, dtype=int)

    for it in range(max_iter):
        # assign
        dists = np.sum((X[:, None, :] - centers[None, :, :])**2, axis=2)  # (N,K)
        new_labels = np.argmin(dists, axis=1)
        # compute objective
        obj = np.sum(np.min(dists, axis=1))
        obj_vals.append(obj)
        # update centers
        new_centers = np.zeros_like(centers)
        for k in range(K):
            members = X[new_labels == k]
            if len(members) == 0:
                # re-initialize empty cluster
                new_centers[k] = X[rng.randint(N)]
            else:
                new_centers[k] = members.mean(axis=0)
        centers = new_centers
        if it > 0 and abs(obj_vals[-1] - obj_vals[-2]) < tol:
            labels = new_labels
            break
        labels = new_labels
    return {'centers': centers, 'labels': labels, 'obj_vals': np.array(obj_vals), 'iters': len(obj_vals)}

# ---------- helper to run many inits and average log-likelihood curves ----------
def average_runs_loglikes(func, X, K=4, runs=100, max_iter=200, **kwargs):
    curves = np.full((runs, max_iter), np.nan)
    iters_arr = np.zeros(runs, dtype=int)
    for r in range(runs):
        out = func(X, K=K, max_iter=max_iter, init_seed=r, **kwargs)
        ll = out['loglikes']
        L = len(ll)
        curves[r, :L] = ll
        iters_arr[r] = L
    # average across runs per iteration index (ignore NaNs)
    avg = np.nanmean(curves, axis=0)
    return avg, curves, iters_arr

# ---------- Main experiment ----------
def main():
    csv_path = 'A2Q1.csv'   # replace by correct path if needed
    X = load_data(csv_path).astype(float)  # shape (400, 50)
    N, D = X.shape
    K = 4
    runs = 100
    max_iter = 200

    # (i) Bernoulli-mixture EM averaged log-likelihood
    avg_bern, all_bern, iters_bern = average_runs_loglikes(bernoulli_mixture_em, X, K=K, runs=runs, max_iter=max_iter)
    plt.figure(figsize=(6,4))
    plt.plot(avg_bern, label='Bernoulli-mixture EM (avg over {} inits)'.format(runs))
    plt.xlabel('EM iteration'); plt.ylabel('Observed log-likelihood'); plt.title('Bernoulli Mixture EM')
    plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig('bernoully_em_loglik.png', dpi=150)

    # (ii) Gaussian-mixture EM averaged log-likelihood
    # NOTE: treating binary data as real-valued; this is for comparison only
    avg_gauss, all_gauss, iters_gauss = average_runs_loglikes(gaussian_mixture_em, X, K=K, runs=runs, max_iter=max_iter)
    plt.figure(figsize=(6,4))
    plt.plot(avg_gauss, label='Gaussian-mixture EM (avg over {} inits)'.format(runs))
    plt.xlabel('EM iteration'); plt.ylabel('Observed log-likelihood'); plt.title('Gaussian Mixture EM (on binary data)')
    plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig('gaussian_em_loglik.png', dpi=150)

    # (iii) K-means: run kmeans on the same data (we'll use binary vectors as points)
    # average K-means objective over runs
    km_obj_runs = []
    max_km_iter = 100
    all_km_objs = np.full((runs, max_km_iter), np.nan)
    for r in range(runs):
        out = kmeans(X, K=K, max_iter=max_km_iter, init_seed=r)
        L = len(out['obj_vals'])
        all_km_objs[r, :L] = out['obj_vals']
    avg_km = np.nanmean(all_km_objs, axis=0)
    plt.figure(figsize=(6,4))
    plt.plot(avg_km, label='K-means objective (avg over {} inits)'.format(runs))
    plt.xlabel('K-means iteration'); plt.ylabel('Sum squared distances (objective)'); plt.title('K-means objective')
    plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig('kmeans_objective.png', dpi=150)

    # Combined plot for easy visual comparison (first 100 iterations)
    plt.figure(figsize=(7,5))
    it = np.arange(max_iter)
    plt.plot(it, avg_bern, label='Bernoulli EM (avg)')
    plt.plot(it, avg_gauss, label='Gaussian EM (avg)')
    plt.plot(np.arange(len(avg_km)), avg_km, label='K-means objective (avg)')
    plt.xlabel('Iteration'); plt.legend(); plt.grid(True)
    plt.title('Comparison (averaged over {} runs)'.format(runs))
    plt.tight_layout()
    plt.savefig('comparison_all.png', dpi=150)
    print('Saved plots: bernoully_em_loglik.png, gaussian_em_loglik.png, kmeans_objective.png, comparison_all.png')

if __name__ == '__main__':
    main()
