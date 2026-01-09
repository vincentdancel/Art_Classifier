import tensorflow as tf
import os

MODEL_DIR = os.path.join("models", "art_classifier_final")
TFLITE_PATH = os.path.join("models", "art_classifier.tflite")

if not os.path.exists(MODEL_DIR):
    raise FileNotFoundError(
        f"❌ Model not found at {MODEL_DIR}\n"
        "Make sure training finished successfully."
    )

print("📦 Loading SavedModel...")
model = tf.keras.models.load_model(MODEL_DIR)

print("⚙️ Converting to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_model)

print(f"✅ TFLite model saved to: {TFLITE_PATH}")
