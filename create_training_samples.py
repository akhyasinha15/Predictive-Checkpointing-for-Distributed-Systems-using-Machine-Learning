import pandas as pd
import numpy as np
from tqdm import tqdm
import os

# ======================================================
# CONFIGURATION
# ======================================================
INPUT_FILE = r"C:\Users\Shobit\PycharmProjects\Hackathons\Predictive Checkpointing and Proactive Fault Tolerance in Distributed Systems\processed_borg_data.csv"
OUTPUT_FEATURES = r"C:\Users\Shobit\PycharmProjects\Hackathons\Predictive Checkpointing and Proactive Fault Tolerance in Distributed Systems\samples_features.npy"
OUTPUT_LABELS = r"C:\Users\Shobit\PycharmProjects\Hackathons\Predictive Checkpointing and Proactive Fault Tolerance in Distributed Systems\samples_labels.npy"

# Window size and step for LSTM training
WINDOW_SIZE = 10          # number of time steps per sample
PREDICT_AHEAD = 3         # how many steps ahead to check for failure (lead time)
STEP_SIZE = 1             # step between sliding windows

# ======================================================
# LOAD PROCESSED DATA
# ======================================================
print(f"📥 Loading processed data from {INPUT_FILE}")
df = pd.read_csv(INPUT_FILE)

# Ensure proper dtypes
df['time'] = pd.to_datetime(df['time'], errors='coerce')
df = df.dropna(subset=['machine_id', 'time', 'cpu', 'memory'])
df = df.sort_values(['machine_id', 'time']).reset_index(drop=True)

# Make sure event and failed columns exist (in case missing)
if 'failed' not in df.columns:
    df['failed'] = 0
if 'event' not in df.columns:
    df['event'] = 'normal'

# Convert failed column to binary
df['failed'] = df['failed'].apply(lambda x: 1 if str(x).strip().lower() in ['1', 'true', 'fail', 'failed'] else 0)

# ======================================================
# FUNCTION TO CREATE SAMPLES PER MACHINE
# ======================================================
def generate_samples_for_machine(machine_df, window_size, predict_ahead, step):
    """
    Generate time window samples for a single machine
    """
    features = []
    labels = []

    cpu = machine_df['cpu'].values
    memory = machine_df['memory'].values
    failed = machine_df['failed'].values

    for i in range(0, len(machine_df) - window_size - predict_ahead, step):
        window_cpu = cpu[i:i+window_size]
        window_mem = memory[i:i+window_size]
        X = np.stack([window_cpu, window_mem], axis=1)  # shape (window, 2)

        # Label = 1 if failure occurs in next predict_ahead timesteps
        y = 1 if np.any(failed[i+window_size:i+window_size+predict_ahead]) else 0

        features.append(X)
        labels.append(y)

    return features, labels


# ======================================================
# GENERATE DATASET
# ======================================================
print("⚙️ Generating samples per machine...")

all_features = []
all_labels = []

for machine_id, machine_df in tqdm(df.groupby('machine_id'), total=df['machine_id'].nunique()):
    ftrs, lbls = generate_samples_for_machine(machine_df, WINDOW_SIZE, PREDICT_AHEAD, STEP_SIZE)
    all_features.extend(ftrs)
    all_labels.extend(lbls)

# Convert to numpy arrays
X = np.array(all_features, dtype=np.float32)
y = np.array(all_labels, dtype=np.int8)

# ======================================================
# SAVE OUTPUTS
# ======================================================
np.save(OUTPUT_FEATURES, X)
np.save(OUTPUT_LABELS, y)

print(f"✅ Created samples for LSTM training")
print(f"   → Features shape: {X.shape}")
print(f"   → Labels shape:   {y.shape}")
print(f"💾 Saved to:")
print(f"   {OUTPUT_FEATURES}")
print(f"   {OUTPUT_LABELS}")
