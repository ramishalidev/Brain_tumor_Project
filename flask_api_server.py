"""
Flask API Server for Brain Tumor Classification Model
Integrates with n8n workflow for AI-powered RAG and LLM analysis
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.efficientnet import preprocess_input
import os
import base64
from io import BytesIO
from PIL import Image
import json
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Configuration
MODEL_PATH = os.path.join("results", "models", "best_model.keras")
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
IMG_SIZE = (224, 224)

# Load model
print("Loading brain tumor classification model...")
try:
    model = load_model(MODEL_PATH)
    print("✓ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None


def decode_image(image_data):
    """Decode base64 image or file upload"""
    try:
        if isinstance(image_data, str) and image_data.startswith('data:image'):
            # Base64 encoded image
            image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
        else:
            # Direct bytes
            image_bytes = image_data
        
        img = Image.open(BytesIO(image_bytes))
        img = img.convert('RGB')
        img = img.resize(IMG_SIZE)
        return img
    except Exception as e:
        raise ValueError(f"Error decoding image: {e}")


def preprocess_image(img):
    """Preprocess image for model prediction"""
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array


def generate_gradcam_heatmap(img_batch, pred_idx):
    """Generate Grad-CAM heatmap for visualization"""
    try:
        # Get the target layer
        gradcam_layer = model.get_layer('gradcam_target_conv')
        
        # Create grad model
        grad_model = tf.keras.Model(
            inputs=model.input,
            outputs=[gradcam_layer.output, model.output[0] if isinstance(model.output, list) else model.output]
        )
        
        # Compute gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_batch, training=False)
            tape.watch(conv_outputs)
            class_score = predictions[:, pred_idx]
        
        grads = tape.gradient(class_score, conv_outputs)
        
        if grads is None:
            return None
        
        # Guided Grad-CAM
        grads_positive = tf.maximum(grads, 0)
        pooled_grads = tf.reduce_mean(grads_positive, axis=(1, 2))[0]
        conv_outputs = conv_outputs[0]
        
        # Create heatmap
        heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1).numpy()
        heatmap = np.maximum(heatmap, 0)
        
        # Normalize
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        # Resize to image size
        heatmap = tf.image.resize(heatmap[..., np.newaxis], IMG_SIZE, method='bicubic')
        heatmap = tf.squeeze(heatmap).numpy()
        
        return heatmap.tolist()
    except Exception as e:
        print(f"Grad-CAM error: {e}")
        return None


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict brain tumor classification
    
    Expected JSON:
    {
        "image": "base64_encoded_image" or file upload,
        "patient_id": "optional_patient_id"
    }
    """
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Get data
        data = request.get_json() if request.is_json else request.form.to_dict()
        
        # Handle file upload
        if 'file' in request.files:
            file = request.files['file']
            img = Image.open(file.stream).convert('RGB')
            img = img.resize(IMG_SIZE)
        elif 'image' in data:
            img = decode_image(data['image'])
        else:
            return jsonify({'error': 'No image provided'}), 400
        
        patient_id = data.get('patient_id', 'unknown')
        
        # Preprocess
        img_array = preprocess_image(img)
        
        # Predict
        predictions = model.predict(img_array, verbose=0)[0]
        pred_idx = int(np.argmax(predictions))
        pred_class = CLASS_NAMES[pred_idx]
        confidence = float(predictions[pred_idx])
        
        # Generate Grad-CAM
        gradcam = generate_gradcam_heatmap(img_array, pred_idx)
        
        # Build response
        response = {
            'patient_id': patient_id,
            'timestamp': datetime.now().isoformat(),
            'prediction': {
                'class': pred_class,
                'confidence': confidence,
                'probabilities': {
                    CLASS_NAMES[i]: float(predictions[i]) 
                    for i in range(len(CLASS_NAMES))
                },
                'is_tumor': pred_class != 'notumor'
            },
            'gradcam': gradcam,
            'metadata': {
                'model_version': '1.0',
                'image_size': IMG_SIZE,
                'analysis_method': 'EfficientNetB0 + Improved Grad-CAM'
            }
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


@app.route('/batch-predict', methods=['POST'])
def batch_predict():
    """
    Batch prediction for multiple images
    
    Expected JSON:
    {
        "images": ["base64_1", "base64_2", ...],
        "patient_ids": ["id1", "id2", ...]
    }
    """
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        data = request.get_json()
        images = data.get('images', [])
        patient_ids = data.get('patient_ids', [f'patient_{i}' for i in range(len(images))])
        
        if not images:
            return jsonify({'error': 'No images provided'}), 400
        
        results = []
        
        for idx, (img_data, patient_id) in enumerate(zip(images, patient_ids)):
            try:
                img = decode_image(img_data)
                img_array = preprocess_image(img)
                
                predictions = model.predict(img_array, verbose=0)[0]
                pred_idx = int(np.argmax(predictions))
                pred_class = CLASS_NAMES[pred_idx]
                confidence = float(predictions[pred_idx])
                
                results.append({
                    'patient_id': patient_id,
                    'prediction': {
                        'class': pred_class,
                        'confidence': confidence,
                        'is_tumor': pred_class != 'notumor'
                    },
                    'status': 'success'
                })
            except Exception as e:
                results.append({
                    'patient_id': patient_id,
                    'status': 'error',
                    'error': str(e)
                })
        
        return jsonify({
            'total': len(images),
            'successful': sum(1 for r in results if r['status'] == 'success'),
            'results': results,
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/model-info', methods=['GET'])
def model_info():
    """Get model information"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify({
        'model_name': 'Brain Tumor Classifier',
        'architecture': 'EfficientNetB0',
        'classes': CLASS_NAMES,
        'input_size': IMG_SIZE,
        'features': [
            'Multi-class classification',
            'Grad-CAM visualization',
            'High accuracy prediction',
            'Supports: Glioma, Meningioma, Pituitary, No Tumor'
        ],
        'version': '1.0',
        'model_path': MODEL_PATH
    })


if __name__ == '__main__':
    print("="*70)
    print("Brain Tumor Classification API Server")
    print("="*70)
    print(f"Model: {MODEL_PATH}")
    print(f"Classes: {CLASS_NAMES}")
    print(f"Image Size: {IMG_SIZE}")
    print("="*70)
    print("\nEndpoints:")
    print("  - GET  /health        : Health check")
    print("  - POST /predict       : Single image prediction")
    print("  - POST /batch-predict : Batch image prediction")
    print("  - GET  /model-info    : Model information")
    print("="*70)
    print("\nStarting server on http://localhost:5000")
    print("="*70)
    
    app.run(host='0.0.0.0', port=5000, debug=True)
