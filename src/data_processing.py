import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

RAW_PATH = 'data/raw/train_FD001.txt'
PROCESSED_DIR = 'data/processed'

# Zero-variance channels confirmed via EDA — excluded to reduce input noise
DROP_SENSORS = ['s_1', 's_5', 's_6', 's_10', 's_16', 's_18', 's_19']

WINDOW_SIZE = 30  # cycles per input sequence
MAX_RUL = 125     # piecewise linear RUL cap (Heimes 2008 benchmark)


def load_data(filepath):
    """Read raw C-MAPPS text file and attach column headers."""
    col_names = (
        ['unit_nr', 'time_cycles']
        + ['setting_1', 'setting_2', 'setting_3']
        + [f's_{i}' for i in range(1, 22)]
    )
    df = pd.read_csv(filepath, sep=r'\s+', header=None, names=col_names)
    print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def process_targets(df):
    """Compute RUL per engine cycle and apply piecewise linear cap at MAX_RUL."""
    max_cycles = df.groupby('unit_nr')['time_cycles'].max().rename('max_life')
    df = df.merge(max_cycles, left_on='unit_nr', right_index=True)
    df['RUL'] = df['max_life'] - df['time_cycles']
    df['RUL_clipped'] = df['RUL'].clip(upper=MAX_RUL)
    return df.drop(columns=['max_life'])


def process_features(df):
    """
    Drop constant-output sensors and operational settings, then
    MinMax-scale on training engines (units 1–80) only to avoid leakage.
    """
    df = df.drop(columns=DROP_SENSORS + ['setting_1', 'setting_2', 'setting_3'])

    sensor_cols = [c for c in df.columns if c.startswith('s_')]
    scaler = MinMaxScaler()

    # scaler fitted on training split only — params must not reflect validation data
    scaler.fit(df[df['unit_nr'] <= 80][sensor_cols])
    df[sensor_cols] = scaler.transform(df[sensor_cols])

    print(f"Retained sensors ({len(sensor_cols)}): {sensor_cols}")
    return df, sensor_cols


def create_sliding_windows(df, window_size, sensor_cols):
    """
    Reshape per-unit time series into (samples, window, features) arrays.
    Units with fewer cycles than window_size are skipped.
    """
    X, y = [], []

    for unit in df['unit_nr'].unique():
        unit_df = df[df['unit_nr'] == unit]
        if len(unit_df) < window_size:
            continue

        data = unit_df[sensor_cols].values
        labels = unit_df['RUL_clipped'].values

        for i in range(len(unit_df) - window_size):
            X.append(data[i:i + window_size])
            y.append(labels[i + window_size])

    return np.array(X), np.array(y)


if __name__ == '__main__':
    df = load_data(RAW_PATH)
    df = process_targets(df)
    df, features = process_features(df)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df.to_csv(f'{PROCESSED_DIR}/train_FD001_processed.csv', index=False)

    # build tensors from training engines only
    train_df = df[df['unit_nr'] <= 80].copy()
    X_train, y_train = create_sliding_windows(train_df, WINDOW_SIZE, features)

    print(f"X_train: {X_train.shape},  y_train: {y_train.shape}")

    np.save(f'{PROCESSED_DIR}/X_train.npy', X_train)
    np.save(f'{PROCESSED_DIR}/y_train.npy', y_train)
    print(f"Tensors saved to {PROCESSED_DIR}")