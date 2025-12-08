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

# Load embeddings
from sentence_transformers import SentenceTransformer

load_dotenv()

app = Flask(__name__)
CORS(app)

# Configuration - Get absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "results", "models")
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
IMG_SIZE = (224, 224)

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
        elif request.is_json and 'image_base64' in request.json:
            img = decode_image(request.json['image_base64'])
        else:
            return jsonify({'error': 'No image provided. Send as form-data or JSON with image_base64'}), 400
        
        # Preprocess and predict
        img_array = preprocess_image(img)
        predictions = model.predict(img_array, verbose=0)[0]
        
        pred_idx = int(np.argmax(predictions))
        pred_label = CLASS_NAMES[pred_idx]
        confidence = float(predictions[pred_idx])
        
        # Build response
        response = {
            'prediction': pred_label,
            'confidence': confidence,
            'all_probabilities': {
                CLASS_NAMES[i]: float(predictions[i]) 
                for i in range(len(CLASS_NAMES))
            },
            'is_tumor': pred_label.lower() != 'notumor',
            'timestamp': datetime.now().isoformat(),
            'model_version': '1.0'
        }
        
        # Add severity assessment
        if response['is_tumor']:
            if confidence > 0.95:
                response['severity'] = 'high'
                response['urgency'] = 'immediate'
            elif confidence > 0.85:
                response['severity'] = 'medium'
                response['urgency'] = 'urgent'
            else:
                response['severity'] = 'low'
                response['urgency'] = 'routine'
        else:
            response['severity'] = 'none'
            response['urgency'] = 'none'
        
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