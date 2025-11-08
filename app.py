import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from flask import Flask, request, jsonify
from flask_cors import CORS   # NEW

# Load models
rf_model = joblib.load("D:/BE Project/AI-dvriven-Driver-behaviour-Monitoring-System/NoteBooks/random_forest_driver.pkl")
scaler = joblib.load("D:/BE Project/AI-dvriven-Driver-behaviour-Monitoring-System/NoteBooks/scaler.pkl")
label_encoder = joblib.load("D:/BE Project/AI-dvriven-Driver-behaviour-Monitoring-System/NoteBooks/label_encoder.pkl")
cnn_model = tf.keras.models.load_model("D:/BE Project/Updated preprocessing files/driver_custom_cnn.h5")

print("✅ Models loaded successfully!")

# Flask app
app = Flask(__name__)
CORS(app)   # ✅ allow frontend requests

# --- CNN Prediction ---
def predict_cnn(img_path, target_size=(224,224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    preds = cnn_model.predict(img_array)
    pred_class = np.argmax(preds, axis=1)[0]
    class_labels = ['closed_eyes', 'not_yawn', 'open_eyes', 'yawn']
    return class_labels[pred_class], preds[0].tolist()

# --- RF Prediction ---
def predict_rf(num_input: dict):
    X = pd.DataFrame([num_input])
    X_encoded = pd.get_dummies(X, drop_first=True)
    X_encoded = X_encoded.reindex(columns=scaler.feature_names_in_, fill_value=0)
    X_scaled = scaler.transform(X_encoded)
    pred = rf_model.predict(X_scaled)[0]
    pred_label = label_encoder.inverse_transform([pred])[0]
    return pred_label

# --- Fusion ---
def fusion_logic(cnn_pred, rf_pred, cnn_probs, threshold=0.7):
    class_labels = ['closed_eyes', 'not_yawn', 'open_eyes', 'yawn']
    cnn_index = class_labels.index(cnn_pred)
    cnn_confidence = cnn_probs[cnn_index]

    if cnn_pred in ["closed_eyes", "yawn"] and cnn_confidence >= threshold:
        if rf_pred in ["drowsy", "distracted"]:
            return "🚨 Critical: Driver Drowsy"
        else:
            return f"⚠ Warning: Possible Drowsiness (Eyes Closed, {cnn_confidence:.2f})"

    if cnn_pred in ["open_eyes", "not_yawn"] and rf_pred == "attentive" and cnn_confidence >= threshold:
        return "✅ Driver Alert"

    if rf_pred in ["drowsy", "distracted"] and cnn_pred in ["open_eyes", "not_yawn"]:
        return "⚠ Warning: Driving Pattern Risky"

    return f"⚠ General Warning (Low CNN Confidence: {cnn_confidence:.2f})"

# Routes
@app.route("/predict_image", methods=["POST"])
def predict_image():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})
    file = request.files["file"]
    os.makedirs("uploads", exist_ok=True)
    file_path = os.path.join("uploads", file.filename)
    file.save(file_path)
    cnn_pred, cnn_probs = predict_cnn(file_path)
    return jsonify({"cnn_prediction": cnn_pred, "cnn_probabilities": cnn_probs})

@app.route("/predict_numerical", methods=["POST"])
def predict_numerical():
    data = request.get_json()
    rf_pred = predict_rf(data)
    return jsonify({"rf_prediction": rf_pred})

@app.route("/predict_fusion", methods=["POST"])
def predict_fusion():
    if "file" not in request.files or "numerical" not in request.form:
        return jsonify({"error": "Image and numerical data required"})
    file = request.files["file"]
    os.makedirs("uploads", exist_ok=True)
    file_path = os.path.join("uploads", file.filename)
    file.save(file_path)
    cnn_pred, cnn_probs = predict_cnn(file_path)
    num_data = eval(request.form["numerical"])
    rf_pred = predict_rf(num_data)
    fusion_result = fusion_logic(cnn_pred, rf_pred, cnn_probs, threshold=0.7)
    return jsonify({
        "cnn_prediction": cnn_pred,
        "cnn_probabilities": cnn_probs,
        "rf_prediction": rf_pred,
        "fusion_result": fusion_result
    })

if __name__ == "__main__":
    app.run(debug=True)
