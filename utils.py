import numpy as np

# =====================================================================
# 1. DATA PREPROCESSING HELPER
# =====================================================================


def zscore_normalize_features(X):
    """
    Scales features to have a mean of 0 and standard deviation of 1.
    Prevents large word counts from drowning out small percentage values.
    """
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma


# =====================================================================
# 2. LINEAR REGRESSION ENGINE (For Translation Time)
# =====================================================================

def compute_cost_regression(X, y, w, b):
    """Computes Mean Squared Error cost for Multiple Linear Regression."""
    m = X.shape[0]
    predictions = X @ w + b
    squared_errors = (predictions - y) ** 2
    total_cost = np.sum(squared_errors) / (2 * m)
    return total_cost


def compute_gradient_regression(X, y, w, b):
    """Computes gradients (dj_dw, dj_db) for Multiple Linear Regression."""
    m = X.shape[0]
    predictions = X @ w + b
    error = predictions - y

    dj_dw = (X.T @ error) / m
    dj_db = np.sum(error) / m
    return dj_dw, dj_db


def gradient_descent_regression(X, y, w_in, b_in, alpha, num_iters):
    """Runs gradient descent to optimize weights and bias for time prediction."""
    w = np.copy(w_in)
    b = b_in
    J_history = []

    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient_regression(X, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        cost = compute_cost_regression(X, y, w, b)
        J_history.append(cost)

        if i % (num_iters // 10) == 0:
            print(f"Regression Iteration {i:4d}: Cost = {cost:.4f}")

    return w, b, J_history


# =====================================================================
# 3. LOGISTIC REGRESSION ENGINE (For Difficulty Classification)
# =====================================================================

def sigmoid(z):
    """Maps any value to a probability between 0 and 1."""
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # clip prevents overflow runtime warnings


def compute_cost_classification(X, y, w, b):
    """Computes Binary Cross-Entropy loss for Logistic Regression."""
    m = X.shape[0]
    # Matrix multiplication passed through sigmoid gives a vector of probabilities
    f_wb = sigmoid(X @ w + b)

    # Avoid log(0) issues by adding a tiny epsilon
    epsilon = 1e-15
    f_wb = np.clip(f_wb, epsilon, 1 - epsilon)

    # Vectorized loss computation
    loss = -y * np.log(f_wb) - (1 - y) * np.log(1 - f_wb)
    total_cost = np.sum(loss) / m
    return total_cost


def compute_gradient_classification(X, y, w, b):
    """Computes gradients for Logistic Regression."""
    m = X.shape[0]
    f_wb = sigmoid(X @ w + b)
    error = f_wb - y

    dj_dw = (X.T @ error) / m
    dj_db = np.sum(error) / m
    return dj_dw, dj_db


def gradient_descent_classification(X, y, w_in, b_in, alpha, num_iters):
    """Runs gradient descent to optimize parameters for classification."""
    w = np.copy(w_in)
    b = b_in
    J_history = []

    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient_classification(X, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        cost = compute_cost_classification(X, y, w, b)
        J_history.append(cost)

        if i % (num_iters // 10) == 0:
            print(f"Classification Iteration {i:4d}: Cost = {cost:.4f}")

    return w, b, J_history
