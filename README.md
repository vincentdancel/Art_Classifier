🎨 Art Style Classifier
A web-based machine learning application that identifies artists from uploaded or captured images of paintings. It predicts the artist and provides detailed information including genre, period, nationality, and history.

✨ Features
🎯 Artist Prediction: Identifies artists from uploaded images using a trained TensorFlow model

📚 Artist Information: Displays genre, period, nationality, history, and famous works

📸 Multiple Input Methods:

File upload (drag & drop or click)

Camera capture (mobile/desktop)

📊 Visual Results: Shows confidence scores and top predictions

🎨 Beautiful UI: Modern, artistic design with animated elements

⚡ Fast Performance: Optimized for quick predictions



🏗️ Project Structure
text
art-classifier/
├── app.py                    # Flask web server
├── train_model.py           # Model training script
├── export_tflite.py         # Export model to TFLite
├── artist_info.json         # Artist metadata (genre, history, etc.)
├── class_indices.json       # Model class indices
├── labels.json              # Artist labels
├── requirements.txt         # Python dependencies
├── static/
│   ├── style.css           # Frontend styling
│   └── script.js           # Frontend functionality
├── templates/
│   └── index.html          # Main HTML page
└── models/                  # Saved models (generated after training)


Dataset Setup
The project uses the Best Artworks of All Time dataset from Kaggle.

bash
# Download and extract the dataset
# Expected structure:
dataset/
├── train/
│   ├── Edgar_Degas/
│   ├── Francisco_Goya/
│   ├── Pablo_Picasso/
│   ├── Rembrandt/
│   └── Vincent_van_Gogh/
└── test/ (optional)
    └── ... (similar structure)


🎮 How to Use
Upload an Image
Click "Choose File" or drag & drop an image
Images are automatically resized to 160x160 pixels
Click "Analyze Painting"

Use Camera
Click "Use Camera" (requires browser permissions)
Position the artwork in frame
Click "Capture"
Click "Analyze Painting"

View Results
Artist Prediction: Top predicted artist with confidence score
Artist Details: Genre, nationality, active years
History: Brief biography
Famous Works: List of notable paintings
Other Predictions: Alternative artist predictions


Acknowledgments
Dataset: https://www.kaggle.com/datasets/ikarus777/best-artworks-of-all-time from Kaggle
TensorFlow and Keras for machine learning framework
Flask for web framework
All artists whose works inspire this project
