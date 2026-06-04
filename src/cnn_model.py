import os

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

DATA_PATH = 'data/processed'
BATCH_SIZE = 64
LEARNING_RATE = 0.0005
EPOCHS = 50

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_data():
    X = np.load(f'{DATA_PATH}/X_train.npy')
    y = np.load(f'{DATA_PATH}/y_train.npy')

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    def to_loader(X, y, shuffle):
        ds = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32).view(-1, 1),
        )
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)

    return (
        to_loader(X_train, y_train, shuffle=True),
        to_loader(X_val, y_val, shuffle=False),
        X.shape[1],
        X.shape[2],
    )


class CNN1D(nn.Module):
    def __init__(self, sequence_length, num_sensors):
        super().__init__()

        # no pooling after conv1 — preserves local high-frequency fault signatures
        self.conv1 = nn.Conv1d(num_sensors, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

        # infer flattened size via a dummy forward pass
        with torch.no_grad():
            flat_size = self._conv_block(
                torch.zeros(1, num_sensors, sequence_length)
            ).view(1, -1).shape[1]

        self.fc1 = nn.Linear(flat_size, 100)
        self.fc2 = nn.Linear(100, 1)

    def _conv_block(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))
        return x

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch, seq, sensors) → (batch, sensors, seq)
        x = self._conv_block(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(x)


def train_model(model, train_loader, val_loader):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_hist, val_hist = [], []
    best_val_rmse = float('inf')

    print(f"Training on {device}")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                val_loss += criterion(model(X_b), y_b).item()

        rmse_train = np.sqrt(running_loss / len(train_loader))
        rmse_val = np.sqrt(val_loss / len(val_loader))
        train_hist.append(rmse_train)
        val_hist.append(rmse_val)

        if rmse_val < best_val_rmse:
            best_val_rmse = rmse_val
            torch.save(model.state_dict(), 'models/cnn_best.pth')

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}  train: {rmse_train:.2f}  val: {rmse_val:.2f}")

    print(f"Best val RMSE: {best_val_rmse:.2f} cycles")
    return train_hist, val_hist


def plot_training(train_hist, val_hist):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(train_hist, label='Training RMSE', color='#1f77b4', linewidth=2)
    ax.plot(val_hist, label='Validation RMSE', color='#ff7f0e', linewidth=2)
    ax.set_title('Training vs Validation RMSE (1D-CNN)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('RMSE (cycles)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    os.makedirs('models', exist_ok=True)

    train_loader, val_loader, seq_len, num_sensors = load_data()
    print(f"Device: {device}")

    model = CNN1D(seq_len, num_sensors).to(device)
    train_hist, val_hist = train_model(model, train_loader, val_loader)
    plot_training(train_hist, val_hist)