import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    Matern,
    RationalQuadratic,
    WhiteKernel,
    ConstantKernel as C,
    DotProduct,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
# 1. Load data, subsample 1000 training points
# =========================================================
# cols 0-5 = internal coordinates, col 6 = energy
DATA_PATH = Path(r'C:\Users\timot\PycharmProjects\CHEM_V_507\Midterm_Problems\GPR\H3O+ (1).csv')

# Read the CSV normally (detects header names like R1, R2, etc.)
data = pd.read_csv(DATA_PATH)

# Drop any non-numeric rows or columns if present
data = data.apply(pd.to_numeric, errors='coerce').dropna()

# First 6 columns = coordinates, 7th = energy
X_all = data.iloc[:, 0:6].to_numpy()
y_all = data.iloc[:, 6].to_numpy()

# We'll do a train/test split FIRST, then downsample 1000 from the training pool.
# This avoids leaking information from test into model selection/BIC.
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X_all, y_all, test_size=0.2, random_state=0
)

# randomly choose 1000 training points (or all if fewer than 1000)
n_train = min(1000, X_train_full.shape[0])
rng = np.random.default_rng(seed=0)
idx = rng.choice(X_train_full.shape[0], size=n_train, replace=False)
X_train = X_train_full[idx]
y_train = y_train_full[idx]

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
print(f"y range: {y_all.min():.4f} to {y_all.max():.4f}")

# Helper functions

def fit_gp(kernel, Xtr, ytr, n_restarts=5, alpha=0.0):
    """
    Trains a GaussianProcessRegressor with the given kernel.
    Returns the fitted model.
    alpha here can be used for known noise. We'll rely on WhiteKernel instead.
    """
    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=n_restarts,
        alpha=alpha,
        normalize_y=True,
        random_state=0,
    )
    gp.fit(Xtr, ytr)
    return gp

def evaluate_gp(gp, Xte, yte):
    """
    Returns RMSE on the held-out test set (generalization error proxy).
    """
    y_pred, y_std = gp.predict(Xte, return_std=True)
    rmse = np.sqrt(mean_squared_error(yte, y_pred))
    return rmse

def bic_from_gp(gp, Xtr, ytr):
    """
    Bayesian Information Criterion for a GP with a given kernel fit.
    We'll treat BIC = k*ln(n) - 2 * logL, where:
       logL = gp.log_marginal_likelihood_value_
       k    = number of free hyperparameters in the kernel
       n    = number of training samples
    (Some definitions flip sign/add constants; what matters is
     we compare models consistently with the SAME formula.)
    Lower BIC is better.
    """
    n = Xtr.shape[0]
    logL = gp.log_marginal_likelihood_value_
    # count tunable hyperparameters:
    # sklearn kernels expose .theta_ after fitting (log of params).
    # length of theta_ is the number of free params.
    k = gp.kernel_.theta.size
    bic = k * np.log(n) - 2.0 * logL
    return bic, logL, k

# 2 & 3. Train FOUR "simple" kernel choices from sklearn
# We'll define "simple" as single base kernels + white noise:
simple_kernels = {
    "RBF": C(1.0, (1e-3, 1e3)) * RBF(
        length_scale=np.ones(X_train.shape[1]),  # ARD per-dimension lengthscale
        length_scale_bounds=(1e-3, 1e3)
    ) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-6, 1e1)),

    "Matern_3/2": C(1.0, (1e-3, 1e3)) * Matern(
        length_scale=np.ones(X_train.shape[1]),
        length_scale_bounds=(1e-3, 1e3),
        nu=1.5
    ) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-6, 1e1)),

    "RationalQuadratic": C(1.0, (1e-3, 1e3)) * RationalQuadratic(
        length_scale=1.0,
        alpha=1.0,
        length_scale_bounds=(1e-3, 1e3),
        alpha_bounds=(1e-3, 1e3)
    ) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-6, 1e1)),

    "DotProduct": C(1.0, (1e-3, 1e3)) * DotProduct(
        sigma_0=1.0,
        sigma_0_bounds=(1e-3, 1e3)
    ) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-6, 1e1)),
}

results_simple = []

for name, ker in simple_kernels.items():
    print(f"\n=== Fitting simple kernel: {name} ===")
    gp = fit_gp(ker, X_train, y_train, n_restarts=5)
    rmse = evaluate_gp(gp, X_test, y_test)
    bic, logL, k = bic_from_gp(gp, X_train, y_train)
    print(f"Kernel after fit:\n {gp.kernel_}")
    print(f"Test RMSE = {rmse:.4f}")
    print(f"log marginal L = {logL:.3f}, k = {k}, BIC = {bic:.3f}")

    results_simple.append({
        "kernel_name": name,
        "kernel_obj": gp.kernel_,
        "rmse": rmse,
        "bic": bic,
        "k": k,
        "logL": logL,
    })

# 4. Build MORE COMPLEX kernels and compare by BIC
# Strategy:
#   - Add sums/products of base kernels.
#   - Intuition: sums ~= additive physical effects / multiple characteristic length scales.
#                 products ~= interaction terms / nonstationarity-like behavior.
#
# We'll start from a pool of base kernels and create combos with 2 or 3 terms.
# We'll measure "kernel complexity" = number of free hyperparameters k.
# We'll also record test RMSE to see if generalization improves.
base_RBF = C(1.0, (1e-3, 1e3)) * RBF(
    length_scale=np.ones(X_train.shape[1]),
    length_scale_bounds=(1e-3, 1e3)
)

base_Matern = C(1.0, (1e-3, 1e3)) * Matern(
    length_scale=np.ones(X_train.shape[1]),
    length_scale_bounds=(1e-3, 1e3),
    nu=1.5
)

base_RQ = C(1.0, (1e-3, 1e3)) * RationalQuadratic(
    length_scale=1.0, alpha=1.0,
    length_scale_bounds=(1e-3, 1e3),
    alpha_bounds=(1e-3, 1e3)
)

noise = WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-6, 1e1))

complex_kernel_candidates = {
    # sum of two characteristic behaviors
    "RBF + Matern": (base_RBF + base_Matern) + noise,

    # RBF + RationalQuadratic lets GP capture both narrow and broad length scales
    "RBF + RQ": (base_RBF + base_RQ) + noise,

    # Product: RBF * Matern can increase flexibility (effectively sharper local structure)
    "RBF * Matern": (base_RBF * base_Matern) + noise,

    # 3-term additive mixture (very flexible)
    "RBF + Matern + RQ": (base_RBF + base_Matern + base_RQ) + noise,
}

results_complex = []

for name, ker in complex_kernel_candidates.items():
    print(f"\n=== Fitting COMPLEX kernel: {name} ===")
    gp = fit_gp(ker, X_train, y_train, n_restarts=5)
    rmse = evaluate_gp(gp, X_test, y_test)
    bic, logL, k = bic_from_gp(gp, X_train, y_train)
    print(f"Kernel after fit:\n {gp.kernel_}")
    print(f"Test RMSE = {rmse:.4f}")
    print(f"log marginal L = {logL:.3f}, k = {k}, BIC = {bic:.3f}")

    results_complex.append({
        "kernel_name": name,
        "kernel_obj": gp.kernel_,
        "rmse": rmse,
        "bic": bic,
        "k": k,
        "logL": logL,
    })

# 4 (continued). Plot generalization error vs kernel complexity
# ---------------------------------------------------------
# We'll define "complexity" = number of hyperparameters (k).
# We'll combine both simple and complex model results.
# We'll make two plots:
#   (A) RMSE vs k
#   (B) BIC vs k

all_results = results_simple + results_complex

# sort by k to make the plot lines left->right
all_results_sorted = sorted(all_results, key=lambda d: d["k"])

complexities = [r["k"] for r in all_results_sorted]
rmses        = [r["rmse"] for r in all_results_sorted]
bics         = [r["bic"] for r in all_results_sorted]
labels       = [r["kernel_name"] for r in all_results_sorted]

plt.figure(figsize=(7,5))
plt.plot(complexities, rmses, marker='o')
for i, label in enumerate(labels):
    plt.text(complexities[i]*1.01, rmses[i]*1.01, label, fontsize=8)
plt.xlabel("Kernel complexity (number of hyperparameters k)")
plt.ylabel("Test RMSE (generalization error)")
plt.title("Generalization error vs kernel complexity")
plt.tight_layout()
plt.show()

plt.figure(figsize=(7,5))
plt.plot(complexities, bics, marker='o')
for i, label in enumerate(labels):
    plt.text(complexities[i]*1.01, bics[i]*1.01, label, fontsize=8)
plt.xlabel("Kernel complexity (number of hyperparameters k)")
plt.ylabel("BIC (lower is better)")
plt.title("BIC vs kernel complexity")
plt.tight_layout()
plt.show()

# Print a neat summary table in console
print("\n=== SUMMARY (sorted by BIC: best first) ===")
for r in sorted(all_results, key=lambda d: d["bic"]):
    print(f"{r['kernel_name']:>20s} | k={r['k']:2d} | RMSE={r['rmse']:.4f} | BIC={r['bic']:.3f} | logL={r['logL']:.3f}")
