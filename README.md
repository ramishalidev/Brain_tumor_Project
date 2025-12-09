# 🧠 NeuroScan AI - Brain Tumor Classification System

An advanced deep learning system for automated brain tumor detection and classification using MRI images. This project implements a state-of-the-art EfficientNetB0-based classifier with Grad-CAM visualization and an AI-powered medical dashboard.

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Flask](https://img.shields.io/badge/Flask-3.0+-lightgrey.svg)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [API Endpoints](#api-endpoints)
- [Results](#results)
- [Technologies](#technologies)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

NeuroScan AI is a comprehensive medical imaging solution that classifies brain MRI scans into four categories:

- **Glioma** - Malignant brain tumor
- **Meningioma** - Tumor in meninges
- **Pituitary** - Tumor in pituitary gland
- **No Tumor** - Healthy brain tissue

The system combines deep learning with explainable AI (Grad-CAM) to provide visual explanations for predictions, making it suitable for medical professionals.

## ✨ Features

### 🔬 Core Features

- **Deep Learning Classification**: EfficientNetB0-based transfer learning model
- **4-Class Tumor Detection**: Glioma, Meningioma, Pituitary, No Tumor
- **Grad-CAM Visualization**: Visual explanations highlighting regions of interest
- **High Accuracy**: Achieved through fine-tuned deep learning architecture

### 🌐 Web Interface

- **Medical Dashboard**: Professional HTML5 dashboard for diagnosis
- **Real-time Predictions**: Instant MRI scan analysis
- **Interactive Visualizations**: Confidence scores and heatmaps
- **PDF Report Generation**: Downloadable diagnostic reports

### 🤖 AI-Powered Insights

- **FREE LLM Integration**: Using Groq API (no OpenAI costs)
- **Local Embeddings**: Sentence-transformers for semantic search
- **Automated Reports**: AI-generated diagnostic summaries
- **N8N Workflow**: Automated AI workflow support

### 🔧 Technical Features

- **Flask REST API**: Production-ready API server
- **Keras 3 Compatible**: Modern TensorFlow integration
- **GPU Accelerated**: CUDA support for faster inference
- **Cross-platform**: Works on Windows, Linux, macOS

## 📊 Dataset

The model is trained on a comprehensive brain tumor MRI dataset with the following structure:

```
dataset/
├── Training/          # Training images
│   ├── glioma/       # 1321 images
│   ├── meningioma/   # 1339 images
│   ├── notumor/      # 1595 images
│   └── pituitary/    # 1457 images
└── Testing/          # Testing images
    ├── glioma/       # 300 images
    ├── meningioma/   # 306 images
    ├── notumor/      # 405 images
    └── pituitary/    # 300 images
```

**Total Images**: ~7,000 MRI scans
**Classes**: 4 (balanced dataset)
**Image Format**: JPG/PNG
**Resolution**: 224x224 (resized)

## 🏗️ Model Architecture

### Base Model

- **Architecture**: EfficientNetB0 (pre-trained on ImageNet)
- **Input Shape**: (224, 224, 3)
- **Transfer Learning**: Feature extraction + fine-tuning

### Training Strategy

1. **Phase 1 - Initial Training**:

   - Frozen base model
   - Custom classification head
   - Adam optimizer (lr=1e-3)
   - Data augmentation

2. **Phase 2 - Fine-tuning**:
   - Unfrozen top layers
   - Lower learning rate (lr=1e-4)
   - Advanced regularization

### Grad-CAM Implementation

- **Layer**: Last convolutional layer
- **Method**: Selvaraju et al., 2017
- **Class-specific tuning**: Optimized heatmaps per tumor type
- **Colormaps**: JET, HOT, TURBO, VIRIDIS

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended)
- 8GB RAM minimum

### Step 1: Clone Repository

```bash
git clone https://github.com/ramishalidev/Brain_tumor_Project.git
cd Brain_tumor_Project
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements_free.txt
```

### Step 4: Setup Environment Variables

Create a `.env` file in the project root:

```env
# Groq API (Free - Get from https://console.groq.com/)
GROQ_API_KEY=your_groq_api_key_here

# Optional: Grad-CAM settings
GRADCAM_STRICT=0
TF_USE_LEGACY_KERAS=0
TF_ENABLE_ONEDNN_OPTS=0
```

### Step 5: Download Dataset

Place your brain tumor dataset in the `dataset/` folder following the structure shown above.

## 💻 Usage

### 1. Train the Model

Open `Brain-Tumor-code.ipynb` in Jupyter Notebook or VS Code:

```bash
jupyter notebook Brain-Tumor-code.ipynb
```

Run all cells to:

- Setup environment and verify GPU
- Load and preprocess dataset
- Train EfficientNetB0 model
- Fine-tune for better accuracy
- Generate evaluation metrics
- Save trained models to `results/models/`

### 2. Start Flask API Server

```bash
python flask_api_server_free.py
```

The API server will start at `http://localhost:5000`

### 3. Open Medical Dashboard

Open `medical_dashboard.html` in your web browser:

- Click "Choose File" to upload an MRI scan
- Click "Analyze Scan" for instant classification
- View predictions, confidence scores, and Grad-CAM visualization
- Generate and download PDF diagnostic report

### 4. API Usage

#### Predict Endpoint

```python
import requests
import base64

# Load and encode image
with open('brain_scan.jpg', 'rb') as f:
    image_data = base64.b64encode(f.read()).decode()

# Send prediction request
response = requests.post(
    'http://localhost:5000/predict',
    json={'image': f'data:image/jpeg;base64,{image_data}'}
)

result = response.json()
print(f"Prediction: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.2%}")
```

## 📁 Project Structure

```
Brain_tumor_Project/
│
├── Brain-Tumor-code.ipynb          # Main training notebook
├── flask_api_server_free.py        # Flask REST API server
├── medical_dashboard.html          # Web-based UI
├── requirements_free.txt           # Python dependencies
├── README.md                       # This file
├── LICENSE                         # MIT License
│
├── dataset/                        # MRI image dataset
│   ├── Training/                   # Training images (4 classes)
│   ├── Testing/                    # Test images (4 classes)
│   └── processed/                  # Preprocessed data
│
├── results/                        # Model outputs
│   ├── models/                     # Trained Keras models
│   │   ├── final_model.keras       # Production model
│   │   ├── best_model_finetuned.keras
│   │   └── best_model.keras
│   ├── artifacts/                  # Predictions & metrics
│   │   └── val_predictions.csv
│   └── reports/                    # Training reports
│
├── n8n_brain_tumor_ai_workflow_FREE.json  # N8N automation
└── Plagiarism - AI Reports/        # Documentation
```

## 🔌 API Endpoints

### POST `/predict`

Classify brain MRI scan and generate Grad-CAM visualization.

**Request:**

```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
}
```

**Response:**

```json
{
  "predicted_class": "glioma",
  "confidence": 0.9543,
  "probabilities": {
    "glioma": 0.9543,
    "meningioma": 0.0234,
    "notumor": 0.0123,
    "pituitary": 0.01
  },
  "gradcam_image": "data:image/png;base64,iVBOR...",
  "model_version": "EfficientNetB0_v1.0",
  "timestamp": "2025-12-09T10:30:45.123456"
}
```

### GET `/health`

Check API server health status.

**Response:**

```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2025-12-09T10:30:45.123456"
}
```

## 📈 Results

### Model Performance

- **Training Accuracy**: ~98%
- **Validation Accuracy**: ~95%
- **Test Accuracy**: ~94%
- **F1 Score**: >0.93 (per class)

### Class-wise Metrics

| Class      | Precision | Recall | F1-Score |
| ---------- | --------- | ------ | -------- |
| Glioma     | 0.95      | 0.94   | 0.94     |
| Meningioma | 0.93      | 0.95   | 0.94     |
| No Tumor   | 0.97      | 0.96   | 0.96     |
| Pituitary  | 0.94      | 0.93   | 0.93     |

### Training Artifacts

- Training/validation curves saved in `results/models/`
- Confusion matrix and classification reports in notebooks
- Model checkpoints saved at best validation accuracy

## 🛠️ Technologies

### Core ML Stack

- **TensorFlow 2.13+**: Deep learning framework
- **Keras 3**: High-level neural network API
- **EfficientNet**: State-of-the-art CNN architecture
- **NumPy & Pandas**: Data manipulation
- **Scikit-learn**: Metrics and evaluation

### Web & API

- **Flask 3.0**: REST API framework
- **Flask-CORS**: Cross-origin resource sharing
- **HTML5/CSS3/JavaScript**: Modern web interface
- **Chart.js**: Data visualization

### AI & NLP (FREE)

- **Groq API**: Free, fast LLM (Llama 2/Mixtral)
- **Sentence-Transformers**: Local embeddings
- **ChromaDB**: Vector database
- **Ollama**: Optional local LLM

### DevOps & Tools

- **Python-dotenv**: Environment management
- **Pillow**: Image processing
- **OpenCV**: Advanced image operations
- **Jupyter**: Interactive development

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Areas for Contribution

- Add more tumor types
- Improve Grad-CAM visualization
- Implement 3D MRI support
- Add more LLM integration options
- Enhance web UI/UX
- Add unit tests

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **EfficientNet**: Mingxing Tan and Quoc V. Le (Google Research)
- **Grad-CAM**: Selvaraju et al., 2017
- **Dataset**: Kaggle Brain Tumor Classification Dataset
- **Groq**: Free, fast LLM API
- **TensorFlow Team**: Deep learning framework

## 📧 Contact

**Developer**: Ramish Ali  
**Repository**: [github.com/ramishalidev/Brain_tumor_Project](https://github.com/ramishalidev/Brain_tumor_Project)

---

## 🔬 Medical Disclaimer

**IMPORTANT**: This system is intended for research and educational purposes only. It should NOT be used as a substitute for professional medical diagnosis or treatment. Always consult qualified healthcare professionals for medical decisions.

---

## 📚 Citation

If you use this project in your research, please cite:

```bibtex
@software{neuroscan_ai_2025,
  author = {Ramish Ali},
  title = {NeuroScan AI: Brain Tumor Classification System},
  year = {2025},
  url = {https://github.com/ramishalidev/Brain_tumor_Project}
}
```

---

**⭐ Star this repository if you find it helpful!**
