import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import MinMaxScaler

sys.path.append(os.path.join(os.getcwd(), 'src'))
from cnn_model import CNN1D

RAW_PATH = 'data/raw/train_FD001.txt'
MODEL_PATH = 'models/cnn_best.pth'
WINDOW_SIZE = 30
MAX_LIFE = 125.0
C_MAINT = 1.0

DROP_SENSORS = ['s_1', 's_5', 's_6', 's_10', 's_16', 's_18', 's_19']

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')


def prepare_fleet_data():
    """Load raw FD001, apply training-split scaling, and build per-engine window tensors."""
    col_names = (
        ['unit_nr', 'time_cycles']
        + ['setting_1', 'setting_2', 'setting_3']
        + [f's_{i}' for i in range(1, 22)]
    )
    df = pd.read_csv(RAW_PATH, sep=r'\s+', header=None, names=col_names)

    max_cycles = df.groupby('unit_nr')['time_cycles'].max().rename('max_cycles')
    df = df.merge(max_cycles, on='unit_nr', how='left')
    df['RUL'] = df['max_cycles'] - df['time_cycles']
    df['RUL_clipped'] = df['RUL'].clip(upper=MAX_LIFE)

    sensor_cols = [c for c in df.columns if c.startswith('s_') and c not in DROP_SENSORS]
    scaler = MinMaxScaler()
    scaler.fit(df[df['unit_nr'] <= 80][sensor_cols])  # fit on training engines only
    df[sensor_cols] = scaler.transform(df[sensor_cols])

    fleet_data = {}
    for unit in df[df['unit_nr'] >= 81]['unit_nr'].unique():
        unit_df = df[df['unit_nr'] == unit].sort_values('time_cycles')
        data = unit_df[sensor_cols].values
        labels = unit_df['RUL_clipped'].values

        X_seq, y_seq = [], []
        for i in range(len(unit_df) - WINDOW_SIZE):
            X_seq.append(data[i:i + WINDOW_SIZE])
            y_seq.append(labels[i + WINDOW_SIZE])

        fleet_data[unit] = {
            'X': torch.tensor(np.array(X_seq), dtype=torch.float32).to(DEVICE),
            'y': np.array(y_seq),
        }

    return fleet_data, len(sensor_cols)


def mc_predict(model, X_tensor, n_samples=100):
    """Run MC Dropout inference and return predictive mean and std across n_samples passes."""
    model.eval()
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()

    with torch.no_grad():
        preds = np.array([model(X_tensor).cpu().numpy() for _ in range(n_samples)])

    return preds.mean(axis=0).flatten(), preds.std(axis=0).flatten()


def evaluate_scenario(fleet_predictions, cost_A, l_limit, c_fail, tau_range):
    """
    Compute fleet savings for Policy B (deterministic) and Policy C (probabilistic)
    at a given safety limit and failure cost.
    """
    cost_B = 0
    for preds in fleet_predictions.values():
        mu, y_true = preds['mu'], preds['y_true']
        maintained = False
        for t in range(len(y_true)):
            if y_true[t] <= 0:
                break
            if mu[t] <= l_limit:
                cost_B += C_MAINT + (y_true[t] / MAX_LIFE)
                maintained = True
                break
        if not maintained:
            cost_B += c_fail

    savings_C = []
    for tau in tau_range:
        cost_C = 0
        for preds in fleet_predictions.values():
            mu, sigma, y_true = preds['mu'], preds['sigma'], preds['y_true']
            maintained = False
            for t in range(len(y_true)):
                if y_true[t] <= 0:
                    break
                if (mu[t] - tau * sigma[t]) <= l_limit:
                    cost_C += C_MAINT + (y_true[t] / MAX_LIFE)
                    maintained = True
                    break
            if not maintained:
                cost_C += c_fail
        savings_C.append(cost_A - cost_C)

    return cost_A - cost_B, np.array(savings_C)


def run_sensitivity_analysis(fleet_data, model):
    print(f"Running MC Dropout predictions on {len(fleet_data)} engines...")
    fleet_predictions = {}
    for i, (unit, data) in enumerate(fleet_data.items()):
        mu, sigma = mc_predict(model, data['X'])
        fleet_predictions[unit] = {'mu': mu, 'sigma': sigma, 'y_true': data['y']}
        print(f"  Engine {unit} ({i+1}/{len(fleet_data)})")

    tau_range = np.linspace(0.0, 3.0, 60)
    cost_ratios = [5.0, 15.0, 50.0]

    for c_fail in cost_ratios:
        cost_A = len(fleet_data) * c_fail
        ratio_label = f'{int(c_fail)}:1'

        sav_B_15, sav_C_15 = evaluate_scenario(fleet_predictions, cost_A, 15.0, c_fail, tau_range)
        sav_B_5,  sav_C_5  = evaluate_scenario(fleet_predictions, cost_A,  5.0, c_fail, tau_range)

        max_sav_15 = sav_C_15.max()
        max_sav_5  = sav_C_5.max()
        optimal_tau = tau_range[sav_C_5.argmax()]
        alpha_5 = max_sav_5 - sav_B_5

        print(f"\nCost ratio {ratio_label}")
        print(f"  Scenario A (L=15)  det: {sav_B_15:.1f}  prob: {max_sav_15:.1f}  alpha: +{max_sav_15 - sav_B_15:.1f} CU")
        print(f"  Scenario B (L=5 )  det: {sav_B_5:.1f}  prob: {max_sav_5:.1f}  alpha: +{alpha_5:.1f} CU  optimal τ: {optimal_tau:.2f}")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

        ax1.axhline(y=sav_B_15, color='tab:orange', linestyle='--', linewidth=2, label='Policy B (Deterministic)')
        ax1.plot(tau_range, sav_C_15, color='tab:green', linewidth=2.5, label='Policy C (Probabilistic)')
        ax1.set_title(f'Scenario A: L = 15 Cycles (Cost Ratio {ratio_label})')
        ax1.set_xlabel('Risk Aversion Multiplier (τ)')
        ax1.set_ylabel('Total Fleet Savings (CU)')
        ax1.legend(loc='lower left')
        ax1.grid(True, alpha=0.3)

        ax2.axhline(y=sav_B_5, color='tab:orange', linestyle='--', linewidth=2, label='Policy B (Deterministic)')
        ax2.plot(tau_range, sav_C_5, color='tab:green', linewidth=2.5, label='Policy C (Probabilistic)')
        ax2.axvline(x=optimal_tau, color='tab:red', linestyle=':', linewidth=1.5, label=f'Optimal τ ({optimal_tau:.2f})')
        ax2.plot(optimal_tau, max_sav_5, 'o', color='tab:red', markersize=8)
        ax2.annotate(
            f'Peak Savings: {max_sav_5:.1f} CU\nAlpha: +{alpha_5:.1f} CU\nOptimal τ: {optimal_tau:.2f}σ',
            xy=(optimal_tau, max_sav_5),
            xytext=(optimal_tau + 0.2, max_sav_5 - max_sav_5 * 0.15),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
            fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='gray', alpha=0.9),
        )
        ax2.set_title(f'Scenario B: L = 5 Cycles (Cost Ratio {ratio_label})')
        ax2.set_xlabel('Risk Aversion Multiplier (τ)')
        ax2.set_ylabel('Total Fleet Savings (CU)')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        filename = f'voi_stress_test_{int(c_fail)}_to_1.png'
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"  Saved: {filename}")


if __name__ == '__main__':
    fleet_data, num_sensors = prepare_fleet_data()
    print(f"Device: {DEVICE}  |  Fleet: {len(fleet_data)} engines")

    model = CNN1D(WINDOW_SIZE, num_sensors)
    state_dict = torch.load(MODEL_PATH, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(DEVICE)

    run_sensitivity_analysis(fleet_data, model)