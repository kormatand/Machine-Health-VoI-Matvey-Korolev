import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

DATA_PATH = 'data/processed'


def load_data():
    X = np.load(f'{DATA_PATH}/X_train.npy')
    y = np.load(f'{DATA_PATH}/y_train.npy')
    print(f"Loaded — X: {X.shape}, y: {y.shape}")
    return X, y


def flatten_for_rf(X):
    """Flatten (samples, timesteps, features) → (samples, timesteps*features) for RF input."""
    n_samples, timesteps, n_features = X.shape
    return X.reshape(n_samples, timesteps * n_features)


def train_baseline(X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training RF baseline — train: {X_train.shape}, val: {X_val.shape}")

    rf = RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)

    preds = rf.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    print(f"Baseline RMSE: {rmse:.2f} cycles")

    return y_val, preds, rmse


def plot_results(y_true, y_pred, rmse):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_true, y_pred, alpha=0.3, s=10, color='steelblue')

    upper = max(y_true.max(), y_pred.max()) * 1.05
    ax.plot([0, upper], [0, upper], 'r--', linewidth=1, label='y = x')

    ax.set_xlabel('True RUL (cycles)')
    ax.set_ylabel('Predicted RUL (cycles)')
    ax.set_title(f'Random Forest Baseline  —  RMSE: {rmse:.2f} cycles')
    ax.legend()
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    X, y = load_data()
    X_flat = flatten_for_rf(X)
    y_val, preds, rmse = train_baseline(X_flat, y)
    plot_results(y_val, preds, rmse)