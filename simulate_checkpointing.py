"""
Step 4: simulate_checkpointing.py
Simulates job execution on a single node with:
  - Fixed-interval checkpointing
  - Adaptive checkpointing using LSTM failure probability model
"""

import numpy as np
import tensorflow as tf
import random
import time

# ------------------------
# Load trained LSTM model
# ------------------------
print("📥 Loading trained LSTM model...")
model = tf.keras.models.load_model("lstm_model_final.keras")
print("✅ LSTM model loaded successfully.")

# ------------------------
# Simulation Parameters
# ------------------------
JOB_DURATION = 1000          # Total simulated time units for a job
FIXED_INTERVAL = 100         # Fixed checkpoint interval
CHECKPOINT_TIME = 5          # Time taken to perform a checkpoint
RECOVERY_TIME = 20           # Time lost when recovering from failure
FAILURE_PROB_BASE = 0.02     # Base random failure probability

# ------------------------
# Generate Synthetic Node Metrics (simulation input)
# ------------------------
def generate_node_metrics(num_steps=JOB_DURATION):
    """
    Simulate realistic node metrics (CPU, memory) over time.
    Values fluctuate to mimic load variation.
    """
    cpu_usage = np.clip(np.random.normal(0.6, 0.1, num_steps), 0, 1)
    mem_usage = np.clip(np.random.normal(0.7, 0.1, num_steps), 0, 1)
    metrics = np.stack([cpu_usage, mem_usage], axis=1)
    return metrics

# ------------------------
# Failure Simulation
# ------------------------
def simulate_failure(predicted_prob):
    """
    Decide if a failure happens based on both random chance and model-predicted risk.
    """
    adjusted_prob = FAILURE_PROB_BASE + (predicted_prob * 0.5)
    return random.random() < adjusted_prob

# ------------------------
# Fixed Interval Checkpointing
# ------------------------
def simulate_fixed_interval(metrics):
    total_time = 0
    wasted_work = 0
    num_recoveries = 0

    for t in range(JOB_DURATION):
        total_time += 1  # Job progresses

        # Perform checkpoint
        if t % FIXED_INTERVAL == 0 and t != 0:
            total_time += CHECKPOINT_TIME

        # Random failure occurs
        if random.random() < FAILURE_PROB_BASE:
            wasted_work += t % FIXED_INTERVAL
            total_time += RECOVERY_TIME
            num_recoveries += 1

    return {
        "strategy": "Fixed Interval",
        "total_time": total_time,
        "wasted_work": wasted_work,
        "num_recoveries": num_recoveries,
        "checkpoint_overhead": (JOB_DURATION // FIXED_INTERVAL) * CHECKPOINT_TIME,
    }

# ------------------------
# Adaptive Checkpointing
# ------------------------
def simulate_adaptive_checkpointing(metrics, model, window_size=10):
    total_time = 0
    wasted_work = 0
    num_recoveries = 0
    next_checkpoint = 0
    t = 0

    while t < JOB_DURATION:
        total_time += 1

        # Predict failure probability based on last few metrics
        if t >= window_size:
            X_window = metrics[t - window_size:t].reshape(1, window_size, metrics.shape[1])
            predicted_prob = model.predict(X_window, verbose=0)[0][0]
        else:
            predicted_prob = 0.0

        # Adjust checkpoint interval dynamically
        if predicted_prob > 0.5:
            checkpoint_interval = 50
        elif predicted_prob > 0.3:
            checkpoint_interval = 75
        else:
            checkpoint_interval = 100

        # Perform checkpoint when needed
        if t >= next_checkpoint:
            total_time += CHECKPOINT_TIME
            next_checkpoint = t + checkpoint_interval

        # Simulate failure
        if simulate_failure(predicted_prob):
            wasted_work += t % checkpoint_interval
            total_time += RECOVERY_TIME
            num_recoveries += 1
            next_checkpoint = t + checkpoint_interval  # reschedule checkpoint

        t += 1

    return {
        "strategy": "Adaptive (LSTM)",
        "total_time": total_time,
        "wasted_work": wasted_work,
        "num_recoveries": num_recoveries,
        "checkpoint_overhead": (JOB_DURATION / np.mean([50, 75, 100])) * CHECKPOINT_TIME,
    }

# ------------------------
# Run Simulation
# ------------------------
print("⚙️ Running job simulations...")
metrics = generate_node_metrics()

fixed_results = simulate_fixed_interval(metrics)
adaptive_results = simulate_adaptive_checkpointing(metrics, model)

# ------------------------
# Display Results
# ------------------------
print("\n📊 Simulation Results:")
print("-" * 60)
for r in [fixed_results, adaptive_results]:
    print(f"Strategy: {r['strategy']}")
    print(f"  Total Time: {r['total_time']:.2f}")
    print(f"  Wasted Work: {r['wasted_work']:.2f}")
    print(f"  Checkpoint Overhead: {r['checkpoint_overhead']:.2f}")
    print(f"  Recoveries: {r['num_recoveries']}")
    print("-" * 60)
