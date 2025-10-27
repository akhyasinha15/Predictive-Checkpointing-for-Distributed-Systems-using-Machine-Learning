import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import tensorflow as tf

# ======================================================
# CONFIGURATION
# ======================================================
INPUT_FEATURES = r"C:\Users\Shobit\PycharmProjects\Hackathons\Predictive Checkpointing and Proactive Fault Tolerance in Distributed Systems\samples_features.npy"
INPUT_LABELS = r"C:\Users\Shobit\PycharmProjects\Hackathons\Predictive Checkpointing and Proactive Fault Tolerance in Distributed Systems\samples_labels.npy"
MODEL_SAVE_PATH = r"C:\Users\Shobit\PycharmProjects\Hackathons\Predictive Checkpointing and Proactive Fault Tolerance in Distributed Systems\models\lstm_failure_predictor.h5"
RESULTS_DIR = os.path.dirname(MODEL_SAVE_PATH)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ======================================================
# LOAD DATA
# ======================================================
print("📥 Loading training data...")
X = np.load(INPUT_FEATURES)
y = np.load(INPUT_LABELS)

print(f"✅ Loaded features: {X.shape}")
print(f"✅ Loaded labels: {y.shape}")

# ======================================================
# TRAIN / TEST SPLIT
# ======================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"📊 Train samples: {X_train.shape[0]} | Test samples: {X_test.shape[0]}")

# ======================================================
# BUILD LSTM MODEL
# ======================================================

model = tf.keras.models.Sequential([
    tf.keras.layers .Input(shape=(X.shape[1], X.shape[2])),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)
model.summary()


# ======================================================
# TRAINING
# ======================================================
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "best_lstm_model.keras",
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

print("🚀 Training model...")
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[checkpoint, early_stop],
    verbose=1
)

# ======================================================
# EVALUATION
# ======================================================
print("🔍 Evaluating model on test data...")
y_pred_probs = model.predict(X_test)
y_pred = (y_pred_probs > 0.5).astype(int).flatten()

print("\n📈 Classification Report:")
print(classification_report(y_test, y_pred, digits=4))

print("\n📊 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Final Test Accuracy: {acc:.4f}")

# ======================================================
# PLOT TRAINING CURVES
# ======================================================
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss', color='blue')
plt.plot(history.history['val_loss'], label='Val Loss', color='orange')
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Acc', color='blue')
plt.plot(history.history['val_accuracy'], label='Val Acc', color='orange')
plt.title('Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'training_curves.png'))
plt.show()

model.save("lstm_model_final.keras")
print("💾 Model saved as lstm_model_final.keras")
print(f"📊 Training plots saved to: {os.path.join(RESULTS_DIR, 'training_curves.png')}")
