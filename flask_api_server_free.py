"""
Flask API Server for Brain Tumor Classification Model
FREE VERSION - Uses Groq LLM + Local Embeddings (No OpenAI costs)
FIXED: Using tensorflow.keras for model loading
"""

import os
# Set environment variables BEFORE importing TensorFlow
os.environ['TF_USE_LEGACY_KERAS'] = '0'  # Use modern Keras 3
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model  # Use standard TensorFlow Keras
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np
from io import BytesIO
from PIL import Image
import base64
from datetime import datetime
from dotenv import load_dotenv

# Import OpenCV for better colormap and image processing
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

# Load embeddings
from sentence_transformers import SentenceTransformer

load_dotenv()

app = Flask(__name__)
CORS(app)

# Print OpenCV status
print("\n" + "="*70)
print("🎨 Grad-CAM Visualization Setup")
print("="*70)
if HAS_OPENCV:
    print("✅ OpenCV (cv2) detected - Using high-quality colormaps")
    print("   - COLORMAP_JET for medical imaging standard")
    print("   - Gaussian blur smoothing for cleaner heatmaps")
    print("   - Contrast enhancement for better visibility")
else:
    print("⚠️  OpenCV not found - Using basic colormap")
    print("   Install for better quality: pip install opencv-python")
print("="*70 + "\n")

# Configuration - Get absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "results", "models")
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
IMG_SIZE = (224, 224)

# Toggle to use strictly paper-style Grad-CAM (Selvaraju et al., 2017) without extra smoothing/thresholds.
GRADCAM_STRICT = os.getenv('GRADCAM_STRICT', '0') == '1'

# Class-specific Grad-CAM tuning values (see GRADCAM_IMPLEMENTATION.md)
CLASS_HEATMAP_CONFIG = {
    # Use JET-like for classic "hotspot" look - tightened to focus on actual tumor
    'glioma':     {'threshold': 0.15, 'power': 0.65, 'alpha': 0.50, 'colormap': 'jet',     'kernel': 5, 'topk_ratio': 0.08},
    # Use HOT for strong core
    'meningioma': {'threshold': 0.12, 'power': 0.85, 'alpha': 0.60, 'colormap': 'hot',     'kernel': 5, 'topk_ratio': 0.10},
    # Use TURBO (smooth jet-like) for small regions
    'pituitary':  {'threshold': 0.10, 'power': 0.60, 'alpha': 0.60, 'colormap': 'turbo',   'kernel': 3, 'topk_ratio': 0.06},
    # Use VIRIDIS muted for no-tumor (very small allowed area)
    'notumor':    {'threshold': 0.30, 'power': 1.00, 'alpha': 0.35, 'colormap': 'viridis', 'kernel': 5, 'topk_ratio': 0.03},
}

# Try multiple model paths with absolute paths
MODEL_PATHS = [
    os.path.join(MODEL_DIR, "final_model.keras"),
    os.path.join(MODEL_DIR, "best_model_finetuned.keras"),
    os.path.join(MODEL_DIR, "best_model.keras"),
]

# Load models with tensorflow.keras
print("Loading brain tumor classification model...")
print(f"Base directory: {BASE_DIR}")
print(f"Model directory: {MODEL_DIR}")
print(f"TensorFlow version: {tf.__version__}")

if os.path.exists(MODEL_DIR):
    print(f"\nFiles in {MODEL_DIR}:")
    for f in os.listdir(MODEL_DIR):
        print(f"  - {f}")

model = None
model_path_used = None

for path in MODEL_PATHS:
    abs_path = os.path.abspath(path)
    print(f"\nChecking: {abs_path}")
    print(f"  Exists: {os.path.exists(abs_path)}")
    
    if os.path.exists(abs_path):
        try:
            print(f"  Attempting to load with tensorflow.keras.models.load_model()...")
            # Use standard TensorFlow Keras load_model
            model = load_model(abs_path, compile=False)  # Don't compile initially
            
            # Recompile with correct optimizer
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            model_path_used = abs_path
            print(f"  ✓ Model loaded successfully!")
            print(f"  Input shape: {model.input_shape}")
            print(f"  Output shape: {model.output_shape}")
            break
        except Exception as e:
            print(f"  ✗ Failed to load: {str(e)[:200]}")
            import traceback
            traceback.print_exc()
            continue

if model is None:
    print("\n" + "="*70)
    print("❌ ERROR: Could not load any model!")
    print("="*70)
    print("Searched paths:")
    for path in MODEL_PATHS:
        exists = "✓ EXISTS" if os.path.exists(path) else "✗ NOT FOUND"
        print(f"  {exists}: {path}")
    print("\nTroubleshooting:")
    print("  1. Check TensorFlow version: pip install tensorflow --upgrade")
    print("  2. Verify model was saved correctly")
    print("  3. Try re-training the model in your Jupyter notebook")
else:
    print("\n" + "="*70)
    print(f"✓ Model loaded successfully!")
    print(f"  Path: {model_path_used}")
    print(f"  Architecture: {model.name if hasattr(model, 'name') else 'Unknown'}")
    print("="*70)

print("\nLoading free embedding model (sentence-transformers)...")
try:
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✓ Embedding model loaded successfully!")
    print(f"  Model: all-MiniLM-L6-v2")
    print(f"  Dimension: 384 (vs OpenAI's 1536)")
    print(f"  Cost: $0 (runs locally)")
except Exception as e:
    print(f"❌ Error loading embedding model: {e}")
    embedding_model = None

# Initialize Groq for free LLM reports
GROQ_API_KEY = os.getenv('GROQ_API_KEY', '')
print("\nChecking Groq API for LLM reports...")
if GROQ_API_KEY:
    print("✓ Groq API key found (free, fast LLM)")
else:
    print("⚠ Groq API key not set - LLM reports will be limited")
    print("  Get free key at: https://console.groq.com/")


def decode_image(image_data):
    """Decode base64 image or file upload"""
    if isinstance(image_data, str):
        # Base64 encoded
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        img_bytes = base64.b64decode(image_data)
        img = Image.open(BytesIO(img_bytes))
    else:
        # File upload
        img = Image.open(image_data)
    
    # Convert to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    return img


def preprocess_image(img):
    """Preprocess image for EfficientNet"""
    img = img.resize(IMG_SIZE)
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array


def generate_gradcam(img_array, pred_idx):
    """Generate Grad-CAM (Selvaraju et al., 2017) on EfficientNetB0.

    If GRADCAM_STRICT=1, apply the classic formulation: weights via GAP over
    gradients, ReLU on heatmap, min-max normalize, resize + colormap, blend.
    """
    try:
        pred_label = CLASS_NAMES[pred_idx]
        cfg = CLASS_HEATMAP_CONFIG.get(pred_label, CLASS_HEATMAP_CONFIG['notumor'])

        # Prefer Grad-CAM friendly layer from notebook: gradcam_target_conv → top_activation → last conv
        target_layer = None
        for candidate in ['gradcam_target_conv', 'top_activation']:
            try:
                target_layer = model.get_layer(candidate)
                if target_layer is not None:
                    print(f"🔭 Found target layer: {candidate}")
                    break
            except Exception:
                continue

        if target_layer is None:
            # Fallback: last conv-like layer
            for layer in reversed(model.layers):
                if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D, tf.keras.layers.SeparableConv2D)):
                    target_layer = layer
                    print(f"🔭 Fallback target layer: {layer.name}")
                    break

        if target_layer is None:
            print('❌ No convolutional layer found for Grad-CAM')
            return None

        print(f"✅ Grad-CAM using layer: {target_layer.name}")
        print(f"📊 Class config: threshold={cfg['threshold']}, power={cfg['power']}, alpha={cfg['alpha']}")

        grad_model = tf.keras.Model(inputs=model.input, outputs=[target_layer.output, model.output])

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            if isinstance(predictions, list):
                predictions = predictions[0]
            class_score = predictions[:, pred_idx]

        grads = tape.gradient(class_score, conv_outputs)
        if grads is None:
            print('❌ Gradients are None - layer may be frozen')
            return None

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
        heatmap = tf.maximum(heatmap, 0).numpy()  # ReLU per Grad-CAM paper

        if GRADCAM_STRICT:
            # Classic Grad-CAM: min-max normalize only
            heatmap = heatmap / (heatmap.max() + 1e-8)
            active_ratio = (heatmap > 0).mean() * 100
            threshold_eff = None
            print("🧭 GRADCAM_STRICT enabled: no thresholds/power/morphology")
        else:
            # Enhanced: percentile scaling + power + adaptive threshold + top-k clamp
            scale_val = np.percentile(heatmap, 98.0)
            if scale_val < 1e-6:
                scale_val = heatmap.max() + 1e-8
            heatmap = heatmap / (scale_val + 1e-8)
            heatmap = np.power(heatmap, cfg['power'])
            heatmap = np.clip(heatmap, 0, 1)

            active_ratio_raw = (heatmap > 0).mean() * 100
            threshold_eff = cfg['threshold']

            # Top-k clamp to bound area per class
            topk = cfg.get('topk_ratio', 0.1)
            q = np.quantile(heatmap, 1 - topk)
            adaptive_thresh = max(threshold_eff, q)

            heatmap = np.where(heatmap >= adaptive_thresh, heatmap, 0)
            active_ratio = (heatmap > 0).mean() * 100
            if active_ratio < 0.5:
                threshold_eff = max(0.03, threshold_eff * 0.6)
                heatmap = np.where(heatmap >= threshold_eff, heatmap, 0)
                active_ratio = (heatmap > 0).mean() * 100
                print(f"⚠ Low activation, lowering threshold to {threshold_eff:.3f}")

            print(f"✓ Active raw: {active_ratio_raw:.2f}% | post-threshold: {active_ratio:.2f}% | topk target: {topk*100:.1f}%")

        # Morphological noise removal for tumor cases (skip in strict mode)
        if (not GRADCAM_STRICT) and pred_label != 'notumor' and HAS_OPENCV and active_ratio > 0.3:
            k = cfg['kernel']
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            heatmap_uint8 = np.uint8(heatmap * 255)
            opened = cv2.morphologyEx(heatmap_uint8, cv2.MORPH_OPEN, kernel)
            closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
            if (closed > 0).mean() * 100 < 0.1:
                print("⚠ Morphology would remove signal; skipping")
            else:
                heatmap = closed.astype(np.float32) / 255.0

        # Resize and smooth
        heatmap_uint8 = np.uint8(255 * heatmap)
        if HAS_OPENCV:
            heatmap_resized = cv2.resize(heatmap_uint8, IMG_SIZE, interpolation=cv2.INTER_CUBIC)
            heatmap_resized = cv2.GaussianBlur(heatmap_resized, (9, 9), 0)

            colormap_map = {
                'inferno': cv2.COLORMAP_INFERNO,
                'hot': cv2.COLORMAP_HOT,
                'plasma': cv2.COLORMAP_PLASMA,
                'viridis': cv2.COLORMAP_VIRIDIS,
                'turbo': cv2.COLORMAP_TURBO,
                'jet': cv2.COLORMAP_JET,
            }
            colormap = colormap_map.get(cfg['colormap'], cv2.COLORMAP_JET)
            heatmap_color = cv2.applyColorMap(heatmap_resized, colormap)
            heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        else:
            # Simple fallback colormap
            heatmap_resized = np.array(Image.fromarray(heatmap_uint8).resize(IMG_SIZE, resample=Image.BICUBIC))
            heatmap_color = np.stack([heatmap_resized]*3, axis=-1)

        # Recover original image in RGB 0-255
        original_img = img_array[0].copy()
        if original_img.max() <= 1.0:
            original_img = (original_img * 127.5 + 127.5).astype(np.uint8)
        else:
            original_img = ((original_img - original_img.min()) / (original_img.max() - original_img.min() + 1e-8) * 255).astype(np.uint8)
        original_rgb = np.array(Image.fromarray(original_img).resize(IMG_SIZE).convert('RGB'))

        if HAS_OPENCV:
            blended = cv2.addWeighted(original_rgb, 1 - cfg['alpha'], heatmap_color, cfg['alpha'], 0)
            blended = cv2.convertScaleAbs(blended, alpha=1.05, beta=10)
            blended_img = Image.fromarray(blended)
        else:
            blended_img = Image.blend(Image.fromarray(original_rgb), Image.fromarray(heatmap_color), alpha=cfg['alpha'])

        buffer = BytesIO()
        blended_img.save(buffer, format='PNG')
        heatmap_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        print(f"✅ Grad-CAM generated successfully (layer={target_layer.name}, label={pred_label})")
        return heatmap_base64

    except Exception as e:
        print(f"❌ Grad-CAM generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy' if model is not None else 'degraded',
        'model_loaded': model is not None,
        'model_path': model_path_used if model is not None else None,
        'embedding_model_loaded': embedding_model is not None,
        'timestamp': datetime.now().isoformat(),
        'tensorflow_version': tf.__version__,
        'llm_provider': 'groq (free)',
        'embedding_provider': 'sentence-transformers (local, free)',
        'version': '2.0-free'
    })


@app.route('/predict', methods=['POST'])
def predict():
    """Single image prediction endpoint"""
    try:
        if model is None:
            return jsonify({
                'error': 'Model not loaded',
                'message': 'Please check model files and TensorFlow installation',
                'searched_paths': MODEL_PATHS
            }), 500
        
        # Get image from request
        if 'image' in request.files:
            img = decode_image(request.files['image'])
        elif request.is_json:
            # Accept both 'image_base64' and 'image' keys for flexibility
            image_data = request.json.get('image_base64') or request.json.get('image')
            if image_data:
                img = decode_image(image_data)
            else:
                return jsonify({'error': 'No image provided. Send as form-data or JSON with image_base64 or image key'}), 400
        else:
            return jsonify({'error': 'No image provided. Send as form-data or JSON with image_base64'}), 400
        
        # Preprocess and predict
        img_array = preprocess_image(img)
        predictions = model.predict(img_array, verbose=0)[0]
        
        pred_idx = int(np.argmax(predictions))
        pred_label = CLASS_NAMES[pred_idx]
        confidence = float(predictions[pred_idx])
        
        # Determine severity
        is_tumor = pred_label.lower() != 'notumor'
        severity = 'none'
        urgency = 'none'
        
        if is_tumor:
            if confidence > 0.95:
                severity = 'high'
                urgency = 'immediate'
            elif confidence > 0.85:
                severity = 'medium'
                urgency = 'urgent'
            else:
                severity = 'low'
                urgency = 'routine'
        
        # Generate Grad-CAM heatmap
        gradcam_base64 = generate_gradcam(img_array, pred_idx)
        
        if gradcam_base64:
            print(f"✓ Grad-CAM generated successfully ({len(gradcam_base64)} bytes)")
        else:
            print("⚠️ Grad-CAM not available for this prediction")
        
        # Build response (compatible with both frontend and WhatsApp)
        response = {
            'prediction': {
                'class': pred_label,
                'predicted_class': pred_label,
                'confidence': confidence,
                'confidence_percentage': round(confidence * 100, 2),
                'probabilities': {
                    CLASS_NAMES[i]: float(predictions[i]) 
                    for i in range(len(CLASS_NAMES))
                },
                'all_probabilities': {
                    CLASS_NAMES[i]: float(predictions[i]) 
                    for i in range(len(CLASS_NAMES))
                },
                'is_tumor': is_tumor
            },
            'severity': severity,
            'urgency': urgency,
            'gradcam': gradcam_base64,
            'gradcam_data': gradcam_base64,
            'patient_id': request.json.get('patient_id', f'web_{datetime.now().timestamp()}') if request.is_json else None,
            'timestamp': datetime.now().isoformat(),
            'model_version': '1.0'
        }
        
        return jsonify(response)
    
    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/batch-predict', methods=['POST'])
def batch_predict():
    """Batch prediction endpoint"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        images_data = request.json.get('images', [])
        if not images_data:
            return jsonify({'error': 'No images provided'}), 400
        
        results = []
        for idx, img_data in enumerate(images_data):
            try:
                img = decode_image(img_data)
                img_array = preprocess_image(img)
                predictions = model.predict(img_array, verbose=0)[0]
                
                pred_idx = int(np.argmax(predictions))
                pred_label = CLASS_NAMES[pred_idx]
                confidence = float(predictions[pred_idx])
                
                results.append({
                    'index': idx,
                    'prediction': pred_label,
                    'confidence': confidence,
                    'is_tumor': pred_label.lower() != 'notumor'
                })
            except Exception as e:
                results.append({
                    'index': idx,
                    'error': str(e)
                })
        
        return jsonify({
            'results': results,
            'total': len(images_data),
            'successful': sum(1 for r in results if 'error' not in r),
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/generate-embedding', methods=['POST'])
def generate_embedding():
    """
    Generate embeddings using FREE local sentence-transformers model
    """
    try:
        if embedding_model is None:
            return jsonify({'error': 'Embedding model not loaded'}), 500
        
        data = request.json
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Generate embedding (local, instant, free)
        embedding = embedding_model.encode(text).tolist()
        
        return jsonify({
            'embedding': embedding,
            'dimension': len(embedding),
            'model': 'all-MiniLM-L6-v2',
            'provider': 'sentence-transformers (local)',
            'cost': 0.0
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/model-info', methods=['GET'])
def model_info():
    """Get model information"""
    if model is None:
        return jsonify({
            'error': 'Model not loaded',
            'message': 'Check TensorFlow installation and model files',
            'searched_paths': MODEL_PATHS,
            'available_files': os.listdir(MODEL_DIR) if os.path.exists(MODEL_DIR) else []
        }), 500
    
    return jsonify({
        'status': 'ready',
        'model_name': 'EfficientNetB0-TransformerHead',
        'model_path': model_path_used,
        'version': '1.0',
        'classes': CLASS_NAMES,
        'num_classes': len(CLASS_NAMES),
        'input_size': list(IMG_SIZE),
        'framework': f'TensorFlow {tf.__version__}',
        'llm_provider': 'groq (free)',
        'embedding_provider': 'sentence-transformers (local, free)',
        'embedding_dimension': 384,
        'total_cost_per_request': '$0'
    })


if __name__ == '__main__':
    print("\n" + "="*70)
    print("🆓 FREE Brain Tumor AI API Server")
    print("="*70)
    print("Configuration:")
    print(f"  • Model Path: {model_path_used if model else 'NOT LOADED'}")
    print(f"  • Classes: {CLASS_NAMES}")
    print(f"  • TensorFlow: {tf.__version__}")
    print(f"  • LLM: Groq (free, fast)")
    print(f"  • Embeddings: sentence-transformers (local, free)")
    print(f"  • Total cost: $0/month")
    print("="*70)
    print("\nEndpoints:")
    print("  • GET  /health - Health check")
    print("  • POST /predict - Single prediction")
    print("  • POST /batch-predict - Batch prediction")
    print("  • POST /generate-embedding - Free embeddings")
    print("  • GET  /model-info - Model information")
    print("="*70)
    
    if model is None:
        print("\n⚠️  WARNING: Model not loaded!")
        print("\nTroubleshooting:")
        print("  1. Check TensorFlow: pip install tensorflow --upgrade")
        print("  2. Verify model files exist in results/models/")
        print("  3. Try re-saving models from Jupyter notebook")
        exit(1)
    
    print("\n✅ All systems ready!")
    print("🚀 Starting server on http://localhost:5000")
    print("="*70 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)