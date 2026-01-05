# REASON 🔍
Image-based REliability & SOuRce iNspectioN for Deep Learning Models
## 📌 Overview
REASON is a reliability-aware, introspective framework designed to predict whether a deep learning model’s prediction is correct, rather than relying on poorly calibrated softmax confidence scores. It operates as a post-hoc, architecture-agnostic reliability estimator that enhances trust in image classification systems.
The framework augments a ResNet50 backbone with a lightweight meta-model (MetaNet) that analyzes internal signals of the base model to estimate prediction correctness.

## 🎯 Key Contributions
- Predicts correctness probability, not just class confidence
- Mitigates overconfident wrong predictions
- Enables selective prediction through safe rejection
- Uses introspective meta-features extracted from internal model behavior
- Improves trust and safety in deployment-critical AI systems

## 🧠 Core Idea
Softmax confidence is often miscalibrated and unreliable, especially for ambiguous or out-of-distribution inputs.
REASON addresses this by learning a secondary correctness predictor using enriched signals from inside the network.

## 📂 Dataset
REASON is evaluated using the **Imagenette** dataset, a curated subset of ImageNet designed for rapid experimentation while preserving ImageNet semantics.

🔗 **Imagenette Dataset:**  
https://github.com/fastai/imagenette


## 🧩 Meta-Features Used
REASON constructs a high-dimensional reliability vector using:
- Entropy of class probabilities
- Logit gap between top predictions
- Top-k logits
- Monte Carlo dropout variance
- Gradient sensitivity (input-level fragility)
- PCA-compressed penultimate embeddings
- K-Nearest Neighbor distances in feature space
These signals are combined and fed into a MetaNet (MLP) to estimate correctness probability.

## ⚙️ Architecture
```
Input Image
     │
ResNet50 Backbone
     │
Internal Signals (logits, features, gradients)
     │
Meta-Feature Construction
     │
MetaNet (Correctness Predictor)
     │
Prediction + Reliability Score
```

## 📂 Repository Structure
```
REASON/
│
├── index.html              # Interactive frontend demo
├── reason_api.py           # Flask backend for inference
├── reason_saved/           # Trained models, PCA, KNN, calibrators
├── eval_plots/             # ROC curves, diagnostics, visualizations
├── README.md               # Project documentation
```

## 🚀 How to Run (Local / Demo)
### 1. Install dependencies
```bash
pip install torch torchvision flask flask-cors numpy joblib pillow matplotlib
```

### 2. Place model artifacts
Ensure the folder reason_saved/ contains:
- best_base.pth
- best_meta_enhanced.pth
- penultimate_pca.pkl
- knn_bank.pkl
- dataset_meta.json

### 3. Start backend
```bash
python reason_api.py
```
Backend runs at:
```
http://127.0.0.1:5000
```

### 4. Open demo UI
Open index.html in your browser and upload an image.

## 📊 Results Summary
- Correctness Prediction AUC: 0.817
- Average Precision (AP): 0.954
- Effectively detects high-confidence incorrect predictions
- Improves trusted accuracy via selective rejection

## 🔍 Explainability & Diagnostics
- Class probability bar charts
- Meta-vector visualization (first 60 dimensions)
- Gradient sensitivity and uncertainty diagnostics
- Raw JSON outputs for debugging and analysis

## 🧪 Generalization
REASON demonstrates strong robustness on:
- Unseen real-world images
- Out-of-distribution samples
- Ambiguous inputs with misleading softmax confidence

## 📄 Research Paper
REASON: A Reliability-Aware Introspective Image Classification Framework Using Enhanced Meta-Feature Modeling
- Accepted at IEEE International Conference on Sustainable and Futuristic Technologies (ICSFT 2026)

## 🔮 Future Work
- Multi-task reliability learning
- Attention-based introspection
- Active learning with reliability feedback
- Extension to other modalities (medical imaging, surveillance)

## 📄 License
This project is intended for academic and research use.
