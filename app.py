from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io, json, os, base64, time
from functools import lru_cache
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO

# ======================
# TensorFlow Speed Boost
# ======================
tf.config.optimizer.set_jit(True)

# ======================
# Flask Setup
# ======================
app = Flask(__name__)
model = None
class_indices = None
idx_to_class = None
artist_info = None

IMG_SIZE = (160, 160)

# ======================
# Caching Functions
# ======================
@lru_cache(maxsize=1)
def load_artist_info():
    try:
        with open("artist_info.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {}

@lru_cache(maxsize=1)
def load_class_indices():
    try:
        with open("class_indices.json", "r", encoding="utf-8") as f:
            indices = json.load(f)
            return indices, {v: k for k, v in indices.items()}
    except:
        return {}, {}

# ======================
# Load Everything ONCE
# ======================
def load_all():
    global model, class_indices, idx_to_class, artist_info

    BASE = os.path.dirname(os.path.abspath(__file__))
    
    try:
        # Load model
        model_path = os.path.join(BASE, "models", "art_classifier_best")
        if not os.path.exists(model_path):
            model_path = os.path.join(BASE, "models", "art_classifier_final")
        
        model = tf.keras.models.load_model(model_path)
        print(f"✅ Model loaded from: {model_path}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        # Create a dummy model for testing
        model = None

    # Load class indices
    class_indices, idx_to_class = load_class_indices()
    
    # Load artist info
    artist_info = load_artist_info()
    
    # 🔥 Warm-up the model
    if model:
        warmup_input = np.zeros((1, 160, 160, 3), dtype=np.float32)
        model.predict(warmup_input, verbose=0)
        print("🔥 Model warmed up & ready")
    else:
        print("⚠️  Using dummy model for testing")

    print(f"✅ Loaded {len(class_indices)} artists")

# ======================
# Chart Generation
# ======================
def generate_prediction_chart(top_predictions, num_predictions=5):
    """
    Generate a bar chart of top predictions
    Returns: base64 encoded image string
    """
    try:
        # Get top N predictions
        top_n = top_predictions[:num_predictions]
        artists = [p['artist'] for p in top_n]
        confidences = [p['confidence'] * 100 for p in top_n]  # Convert to percentage
        
        # Create figure with custom style
        plt.figure(figsize=(10, 6))
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # Create horizontal bar chart
        bars = plt.barh(artists, confidences, color='#4CAF50', alpha=0.8)
        
        # Add value labels on bars
        for i, (bar, conf) in enumerate(zip(bars, confidences)):
            plt.text(conf + 1, i, f'{conf:.1f}%', 
                    va='center', fontsize=10, fontweight='bold')
        
        # Customize chart
        plt.xlabel('Confidence (%)', fontsize=12, fontweight='bold')
        plt.ylabel('Artist', fontsize=12, fontweight='bold')
        plt.title('Top Prediction Results', fontsize=14, fontweight='bold', pad=20)
        plt.xlim(0, 100)
        
        # Invert y-axis so highest confidence is at top
        plt.gca().invert_yaxis()
        
        # Tight layout
        plt.tight_layout()
        
        # Save to buffer
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        
        # Encode to base64
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Close plot to free memory
        plt.close()
        
        return f"data:image/png;base64,{image_base64}"
        
    except Exception as e:
        print(f"❌ Error generating chart: {e}")
        return None


def generate_confidence_pie_chart(top_predictions, num_predictions=5):
    """
    Generate a pie chart of confidence distribution
    Returns: base64 encoded image string
    """
    try:
        # Get top N predictions
        top_n = top_predictions[:num_predictions]
        
        # Prepare data
        artists = [p['artist'] for p in top_n]
        confidences = [p['confidence'] * 100 for p in top_n]
        
        # Calculate "Other" category
        other_confidence = 100 - sum(confidences)
        if other_confidence > 0:
            artists.append('Others')
            confidences.append(other_confidence)
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Color palette
        colors = ['#4CAF50', '#66BB6A', '#81C784', '#A5D6A7', '#C8E6C9', '#E8F5E9']
        
        # Create pie chart
        wedges, texts, autotexts = plt.pie(
            confidences, 
            labels=artists,
            autopct='%1.1f%%',
            colors=colors,
            startangle=90,
            textprops={'fontsize': 10, 'fontweight': 'bold'}
        )
        
        # Make percentage text white for better visibility
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(11)
        
        # Title
        plt.title('Confidence Distribution', fontsize=14, fontweight='bold', pad=20)
        
        # Equal aspect ratio ensures circular pie
        plt.axis('equal')
        
        # Save to buffer
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        
        # Encode to base64
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Close plot
        plt.close()
        
        return f"data:image/png;base64,{image_base64}"
        
    except Exception as e:
        print(f"❌ Error generating pie chart: {e}")
        return None

# ======================
# Image Preprocessing
# ======================
def preprocess(image):
    image = image.convert("RGB").resize(IMG_SIZE)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return arr[None, ...]

# ======================
# Routes
# ======================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get_artists")
def get_artists():
    try:
        # Load from labels.json or class_indices.json
        try:
            with open("labels.json", "r", encoding="utf-8") as f:
                labels = json.load(f)
                artists = labels.get("artists", [])
        except:
            artists = list(class_indices.keys()) if class_indices else []
        
        return jsonify({
            "success": True,
            "artists": sorted(artists)
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/predict", methods=["POST"])
def predict():
    start_time = time.time()
    
    try:
        if model is None:
            return jsonify({
                "success": False,
                "error": "Model not loaded. Please train the model first."
            }), 500

        if "file" in request.files:
            file = request.files["file"]
            if file.filename == "":
                return jsonify({"success": False, "error": "No file selected"}), 400
            image = Image.open(file.stream)
        else:
            data = request.get_json()
            if not data or "image" not in data:
                return jsonify({"success": False, "error": "No image data provided"}), 400
            
            img_base64 = data["image"].split(",")[1] if "," in data["image"] else data["image"]
            image = Image.open(io.BytesIO(base64.b64decode(img_base64)))

        img = preprocess(image)

        preds = model.predict(img, verbose=0)[0]
        idx = int(np.argmax(preds))

        artist = idx_to_class.get(idx, "Unknown")
        info = artist_info.get(artist, {})
        
        # Get top predictions (top 5 or 10)
        top_n = 10  # Get top 10 for charts
        top_indices = np.argsort(preds)[-top_n:][::-1]
        top_predictions = []
        
        for i in top_indices:
            artist_name = idx_to_class.get(i, f"Artist_{i}")
            top_predictions.append({
                "artist": artist_name,
                "confidence": float(preds[i])
            })

        # Generate charts
        print("📊 Generating prediction charts...")
        bar_chart = generate_prediction_chart(top_predictions, num_predictions=5)
        pie_chart = generate_confidence_pie_chart(top_predictions, num_predictions=5)

        processing_time = time.time() - start_time
        print(f"✅ Prediction completed in {processing_time:.2f} seconds")

        return jsonify({
            "success": True,
            "processing_time": processing_time,
            "prediction": {
                "artist": artist,
                "confidence": float(preds[idx]),
                "genre": info.get("genre", []),
                "history": info.get("history", "No information available."),
                "nationality": info.get("nationality", "Unknown"),
                "years": info.get("years", "Unknown"),
                "famous_works": info.get("famous_works", [])
            },
            "top_predictions": top_predictions[:3],  # Top 3 for text display
            "charts": {
                "bar_chart": bar_chart,
                "pie_chart": pie_chart
            }
        })

    except Exception as e:
        processing_time = time.time() - start_time
        print(f"❌ Error in prediction: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
            "processing_time": processing_time
        }), 500

@app.route("/status")
def status():
    return jsonify({
        "success": True,
        "model_loaded": model is not None,
        "num_artists": len(class_indices),
        "artists": list(class_indices.keys()) if class_indices else []
    })

# ======================
# Run
# ======================
if __name__ == "__main__":
    print("🚀 Starting Art Classifier Server...")
    load_all()
    print("✅ Server ready!")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)