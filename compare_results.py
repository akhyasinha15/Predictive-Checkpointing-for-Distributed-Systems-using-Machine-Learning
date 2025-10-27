"""
Step 5: compare_metrics.py
Compares Fixed Interval vs Adaptive (LSTM) checkpointing strategies
across multiple simulation runs and visualizes results.
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from simulate_checkpointing import generate_node_metrics, simulate_fixed_interval, simulate_adaptive_checkpointing

# ------------------------
# Configuration
# ------------------------
RUNS = 10                   # Number of simulation iterations
JOB_DURATION = 1000          # Consistent with previous scripts

# ------------------------
# Load trained model
# ------------------------
print("📥 Loading trained LSTM model...")
model = tf.keras.models.load_model("lstm_model_final.keras")
print("✅ Model loaded successfully.")

# ------------------------
# Containers for results
# ------------------------
fixed_times, adaptive_times = [], []
fixed_waste, adaptive_waste = [], []
fixed_recoveries, adaptive_recoveries = [], []

# ------------------------
# Run simulations multiple times
# ------------------------
print(f"⚙️ Running {RUNS} comparative simulations...")

for i in range(RUNS):
    print(f"\n▶️ Run {i+1}/{RUNS}")
    metrics = generate_node_metrics(JOB_DURATION)

    fixed = simulate_fixed_interval(metrics)
    adaptive = simulate_adaptive_checkpointing(metrics, model)

    fixed_times.append(fixed['total_time'])
    adaptive_times.append(adaptive['total_time'])

    fixed_waste.append(fixed['wasted_work'])
    adaptive_waste.append(adaptive['wasted_work'])

    fixed_recoveries.append(fixed['num_recoveries'])
    adaptive_recoveries.append(adaptive['num_recoveries'])

# ------------------------
# Compute averages
# ------------------------
avg_results = {
    "Strategy": ["Fixed Interval", "Adaptive (LSTM)"],
    "Avg_Total_Time": [np.mean(fixed_times), np.mean(adaptive_times)],
    "Avg_Wasted_Work": [np.mean(fixed_waste), np.mean(adaptive_waste)],
    "Avg_Recoveries": [np.mean(fixed_recoveries), np.mean(adaptive_recoveries)],
}

print("\n📊 Average Simulation Metrics:")
print("-" * 70)
for i in range(2):
    print(f"Strategy: {avg_results['Strategy'][i]}")
    print(f"  Avg Total Time: {avg_results['Avg_Total_Time'][i]:.2f}")
    print(f"  Avg Wasted Work: {avg_results['Avg_Wasted_Work'][i]:.2f}")
    print(f"  Avg Recoveries: {avg_results['Avg_Recoveries'][i]:.2f}")
    print("-" * 70)

# ------------------------
# Visualization
# ------------------------
def plot_comparison(title, fixed_values, adaptive_values, ylabel):
    x = np.arange(2)
    plt.bar(x, [np.mean(fixed_values), np.mean(adaptive_values)], color=['red', 'green'], width=0.5)
    plt.xticks(x, ["Fixed", "Adaptive (LSTM)"])
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.show()

# ------------------------
# Plot results
# ------------------------
print("📈 Generating comparison plots...")

plot_comparison("Average Total Execution Time", fixed_times, adaptive_times, "Total Time (units)")
plot_comparison("Average Wasted Work", fixed_waste, adaptive_waste, "Wasted Work (units)")
plot_comparison("Average Recoveries", fixed_recoveries, adaptive_recoveries, "Recoveries (count)")

print("\n✅ Comparison complete. Plots generated successfully.")
