# reason_api.py
"""
REASON Flask backend (single-file).
Place this file in the same folder as your `reason_saved/` directory (or adjust SAVE_DIR).
Runs a simple Flask server that exposes:
 - GET  /health
 - GET  /        (serves index.html from same dir)
 - POST /api/infer (multipart/form-data, field name "file")

The server loads:
 - best_base.pth
 - best_meta_enhanced.pth
 - penultimate_pca.pkl
 - knn_bank.pkl
 - optional: temp_scaler.pkl, iso_calibrator.pkl
 - dataset_meta.json (human-readable labels)

Responses: JSON with pred_label (human), pred_synset (imagenet synset), pred_idx, probs,
           base_conf_calibrated, meta_prob, meta_vector, preview_png, classbar_png, meta_png, diag
"""
import os
import io
import json
import math
import base64
import joblib
import numpy as np
from io import BytesIO
from PIL import Image

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt

# ----------------- CONFIG -----------------
# By default expect a folder named "reason_saved" in the same directory as this file.
# You can set an absolute path here if your artifacts live elsewhere.
SAVE_DIR = os.environ.get("REASON_SAVE_DIR", "./reason_saved")

BASE_CKPT = os.path.join(SAVE_DIR, "best_base.pth")
META_CKPT = os.path.join(SAVE_DIR, "best_meta_enhanced.pth")
PCA_PATH  = os.path.join(SAVE_DIR, "penultimate_pca.pkl")
KNN_PATH  = os.path.join(SAVE_DIR, "knn_bank.pkl")
TS_PATH   = os.path.join(SAVE_DIR, "temp_scaler.pkl")
ISO_PATH  = os.path.join(SAVE_DIR, "iso_calibrator.pkl")
LABELS_PATH = os.path.join(SAVE_DIR, "dataset_meta.json")
META_INFO_PATH = os.path.join(SAVE_DIR, "meta_vector_info.json")

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Starting REASON backend. Device:", DEVICE)
print("Expecting artifacts in:", os.path.abspath(SAVE_DIR))

# ----------------- MODEL CLASSES -----------------
class ResNet50Base(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        from torchvision import models
        self.backbone = models.resnet50(pretrained=False)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)
        self.penultimate_dim = in_features

    def forward(self, x, return_features=False):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.backbone.avgpool(x)
        feats = torch.flatten(x, 1)
        logits = self.backbone.fc(feats)
        return (logits, feats) if return_features else logits

class MetaNetBig(nn.Module):
    def __init__(self, input_dim, hidden=[512,256,128], dropout=0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)

# ----------------- UTILITIES -----------------
def load_torch_model(model, path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    sd = torch.load(path, map_location=DEVICE)
    sd_used = sd.get("state_dict", sd)
    clean = {k.replace("module.", ""): v for k,v in sd_used.items()}
    model.load_state_dict(clean, strict=False)
    model = model.to(DEVICE).eval()
    return model

def fig_to_datauri(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode("ascii")
    plt.close(fig)
    return f"data:image/png;base64,{data}"

def make_preview_png(pil_img, size=(320,320)):
    img = pil_img.copy().convert("RGB")
    img.thumbnail(size)
    buf = BytesIO()
    img.save(buf, format='PNG')
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"

def make_classbar_png(class_probs, class_names=None, topk=5):
    idx = np.argsort(class_probs)[::-1][:topk]
    probs = class_probs[idx]
    labels = [class_names[i] if class_names else str(i) for i in idx]
    fig, ax = plt.subplots(figsize=(4,2.2))
    y = np.arange(len(labels))
    ax.barh(y, probs[::-1], align='center')
    ax.set_yticks(y)
    ax.set_yticklabels(labels[::-1])
    ax.set_xlim(0,1)
    ax.set_xlabel('Probability')
    ax.set_title('Top-{} predictions'.format(topk))
    plt.tight_layout()
    return fig_to_datauri(fig)

def make_meta_png(meta_vector, max_dims=60):
    vec = np.array(meta_vector[:max_dims])
    fig, ax = plt.subplots(figsize=(6,2))
    ax.bar(np.arange(len(vec)), vec)
    ax.set_xlabel('Meta-dim index')
    ax.set_ylabel('Value')
    ax.set_title('Meta vector (first {} dims)'.format(len(vec)))
    plt.tight_layout()
    return fig_to_datauri(fig)

# ----------------- Load class names (human-readable) -----------------
class_names = None
num_classes = None
if os.path.exists(LABELS_PATH):
    try:
        dd = json.load(open(LABELS_PATH))
        if dd.get('class_names'):
            class_names = dd['class_names']
            num_classes = len(class_names)
    except Exception as e:
        print("Warning reading dataset_meta.json:", e)

# Fallback detailed mapping (Imagenette order)
FALLBACK_CLASS_NAMES = [
    "Tench (fish, Tinca tinca)",                  # n01440764
    "English springer (dog)",                     # n02102040
    "Cassette player (audio equipment)",          # n02979186
    "Chain saw (chainsaw, power tool)",           # n03000684
    "Church (building)",                          # n03028079
    "French horn (musical instrument)",           # n03394916
    "Garbage truck (refuse collection vehicle)",  # n03417042
    "Gas mask (protective mask)",                 # n03425413
    "Golf ball (sports equipment)",               # n03445777
    "Parachute (aerial deceleration device)"      # n03888257
]
if class_names is None:
    class_names = FALLBACK_CLASS_NAMES
if num_classes is None:
    num_classes = len(class_names)

# SYNSET order (Imagenette synsets)
SYNSET_ORDER = [
    "n01440764", "n02102040", "n02979186", "n03000684", "n03028079",
    "n03394916", "n03417042", "n03425413", "n03445777", "n03888257"
]
if len(SYNSET_ORDER) != num_classes:
    print("Warning: SYNSET_ORDER length != num_classes")

# ----------------- Load artifacts (models, pca, knn, calibrators) -----------------
# Base model
try:
    base_model = ResNet50Base(num_classes=num_classes)
    base_model = load_torch_model(base_model, BASE_CKPT)
    print("Loaded base model from", BASE_CKPT)
except Exception as e:
    print("ERROR loading base model:", e)
    base_model = None

# PCA & KNN (optional)
pca = joblib.load(PCA_PATH) if os.path.exists(PCA_PATH) else None
knn = joblib.load(KNN_PATH) if os.path.exists(KNN_PATH) else None
if pca is not None:
    print("Loaded PCA (n_components):", getattr(pca, "n_components", None))
if knn is not None:
    print("Loaded KNN bank.")

# Meta model
meta_model = None
try:
    if os.path.exists(META_INFO_PATH):
        meta_dim = int(json.load(open(META_INFO_PATH))['meta_dim'])
    else:
        # fallback: meta_vector_info.json missing - try to infer (dangerous)
        meta_dim = None
    if meta_dim is not None:
        mm = MetaNetBig(input_dim=meta_dim)
        meta_model = load_torch_model(mm, META_CKPT)
        print("Loaded meta model from", META_CKPT)
    else:
        print("meta_vector_info.json missing; meta_model not loaded.")
except Exception as e:
    print("Could not load meta model:", e)
    meta_model = None

# Calibrators (optional)
temp_scaler = joblib.load(TS_PATH) if os.path.exists(TS_PATH) else None
iso_cal = joblib.load(ISO_PATH) if os.path.exists(ISO_PATH) else None
if temp_scaler is not None:
    print("Loaded temperature scaler.")
if iso_cal is not None:
    print("Loaded isotonic calibrator.")

# ----------------- Preprocessing -----------------
IMG_SIZE = 224
infer_transform = transforms.Compose([
    transforms.Resize(int(IMG_SIZE * 1.14)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

# ----------------- Meta feature computation (single-image version) -----------------
def compute_meta_single(img_tensor, model=base_model, k=5, mc_iterations=6, pca_obj=None, knn_obj=None, device=DEVICE):
    """
    img_tensor: (1,C,H,W) normalized torch tensor
    returns dict with meta_vec (1D), entropy, logit_gap, max_prob, mc_var, grad_norm, knn_mean, pen_feats
    """
    if model is None:
        raise RuntimeError("Base model not loaded")
    model.eval()
    with torch.no_grad():
        logits, feats = model(img_tensor.to(device), return_features=True)
        logits_np = logits.cpu().numpy()[0]
        probs = np.exp(logits_np - logits_np.max()); probs /= probs.sum()
        entropy = float(-(probs * np.log(probs + 1e-12)).sum())
        max_prob = float(probs.max())
        num_classes = logits.shape[1]
        k_eff = min(max(2, min(k, num_classes)), num_classes)
        topk_vals = np.sort(logits_np)[::-1][:k_eff]
        logit_gap = float(topk_vals[0] - topk_vals[1]) if k_eff >= 2 else 0.0

    # MC dropout variance (best-effort)
    mc_var = 0.0
    try:
        model.train()
        preds = []
        with torch.no_grad():
            for _ in range(mc_iterations):
                l, _ = model(img_tensor.to(device), return_features=True)
                p = F.softmax(l, dim=1).cpu().numpy()
                preds.append(p)
        preds = np.stack(preds, axis=0)
        mc_var = float(preds.var(axis=0).mean())
    except Exception:
        mc_var = 0.0
    finally:
        model.eval()

    # grad norm
    grad_norm = 0.0
    try:
        inp = img_tensor.clone().detach().requires_grad_(True).to(device)
        l, _ = model(inp, return_features=True)
        topv, _ = l.topk(1, dim=1)
        topv.sum().backward()
        grad_norm = float(inp.grad.view(-1).norm().item())
    except Exception:
        grad_norm = 0.0

    # penultimate and PCA
    with torch.no_grad():
        _, feats_now = model(img_tensor.to(device), return_features=True)
        pen_feats = feats_now.cpu().numpy()[0]
    if pca_obj is not None:
        pca_feats = pca_obj.transform(pen_feats.reshape(1, -1))[0]
    else:
        pca_feats = np.array([])

    knn_mean = 0.0
    if knn_obj is not None and pca_feats.size > 0:
        try:
            d, idxs = knn_obj.kneighbors(pca_feats.reshape(1, -1), n_neighbors=min(knn_obj.n_neighbors, 6))
            knn_mean = float(d[:,1:].mean()) if d.shape[1] > 1 else float(d.mean())
        except Exception:
            knn_mean = 0.0

    small_feats = np.array([entropy, logit_gap, max_prob, mc_var, grad_norm, knn_mean], dtype=float)
    meta_vec = np.concatenate([small_feats, topk_vals, pca_feats]).astype(float)
    return {
        "meta_vec": meta_vec,
        "entropy": entropy,
        "logit_gap": logit_gap,
        "max_prob": max_prob,
        "mc_var": mc_var,
        "grad_norm": grad_norm,
        "knn_mean": knn_mean,
        "pen_feats": pen_feats
    }

# ----------------- Inference wrapper -----------------
def run_inference(pil_img, mc_iterations=6):
    """
    Input: PIL.Image
    Returns: dict with predictions, images (data URIs), metrics, meta vector
    """
    input_t = infer_transform(pil_img).unsqueeze(0)  # (1,C,H,W)
    # Base forward
    with torch.no_grad():
        logits, _ = base_model(input_t.to(DEVICE), return_features=True)
        logits_np = logits.cpu().numpy()[0]
        probs = np.exp(logits_np - logits_np.max()); probs /= probs.sum()
    pred_idx = int(np.argmax(probs))
    pred_label = class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)
    pred_synset = SYNSET_ORDER[pred_idx] if pred_idx < len(SYNSET_ORDER) else None

    # Temperature scaling if available
    calibrated_prob = float(np.max(probs))
    probs_for_display = probs.copy()
    if temp_scaler is not None:
        try:
            scaled_logits = logits.cpu().numpy() / (float(temp_scaler.T) + 1e-12)
            scaled = np.exp(scaled_logits - scaled_logits.max()); scaled = scaled / scaled.sum()
            calibrated_prob = float(np.max(scaled))
            probs_for_display = scaled
        except Exception:
            pass

    # Isotonic calibrator
    iso_prob = calibrated_prob
    if iso_cal is not None:
        try:
            iso_prob = float(iso_cal.predict([calibrated_prob])[0])
        except Exception:
            iso_prob = calibrated_prob

    # Meta vector + meta probability
    meta_dict = compute_meta_single(input_t, model=base_model, k=5, mc_iterations=mc_iterations, pca_obj=pca, knn_obj=knn, device=DEVICE)
    meta_vec = meta_dict['meta_vec'].astype(float)
    meta_prob = None
    if meta_model is not None:
        with torch.no_grad():
            mv = torch.tensor(meta_vec, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            meta_logit = meta_model(mv).cpu().numpy()[0]
            meta_prob = float(1 / (1 + math.exp(-meta_logit)))

    # images
    preview_png = make_preview_png(pil_img)
    classbar_png = None
    meta_png = None
    try:
        classbar_png = make_classbar_png(probs_for_display, class_names=class_names, topk=min(8, len(probs_for_display)))
    except:
        classbar_png = None
    try:
        meta_png = make_meta_png(meta_vec, max_dims=60)
    except:
        meta_png = None

    diag = {
        "entropy": float(meta_dict.get("entropy")),
        "logit_gap": float(meta_dict.get("logit_gap")),
        "mc_var": float(meta_dict.get("mc_var")),
        "grad_norm": float(meta_dict.get("grad_norm")),
        "knn_mean": float(meta_dict.get("knn_mean")),
        "pen_feats_len": int(len(meta_dict.get("pen_feats")) if meta_dict.get("pen_feats") is not None else 0)
    }

    return {
        "pred_label": pred_label,
        "pred_synset": pred_synset,
        "pred_idx": pred_idx,
        "probs": probs_for_display.tolist(),
        "base_conf_calibrated": float(iso_prob),
        "meta_prob": meta_prob,
        "meta_vector": meta_vec.tolist(),
        "classbar_png": classbar_png,
        "meta_png": meta_png,
        "preview_png": preview_png,
        "diag": diag
    }

# ----------------- Flask app -----------------
app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status":"ok", "device": str(DEVICE)})

@app.route("/", methods=["GET"])
def index():
    # serve index.html if present in same folder
    idx = os.path.join(os.getcwd(), "index.html")
    if os.path.exists(idx):
        return send_from_directory(".", "index.html")
    return "<h3>REASON backend running. Place index.html next to reason_api.py to serve the UI.</h3>"

@app.route("/api/infer", methods=["POST"])
def api_infer():
    if "file" not in request.files:
        return jsonify({"error": "No file part, expected 'file' form field"}), 400
    f = request.files["file"]
    try:
        img = Image.open(f.stream).convert("RGB")
    except Exception as e:
        return jsonify({"error": "Failed to read uploaded image", "detail": str(e)}), 400
    try:
        resp = run_inference(img)
        return jsonify(resp)
    except Exception as e:
        return jsonify({"error": "Inference failed", "detail": str(e)}), 500

# Serve static assets (if you store CSS/JS separately)
@app.route("/<path:filename>")
def serve_static(filename):
    if os.path.exists(filename):
        return send_from_directory(".", filename)
    return "", 404

if __name__ == "__main__":
    # Suggestion: run with `python reason_api.py`
    print("REASON API serving at http://127.0.0.1:5000/  (SAVE_DIR = {})".format(os.path.abspath(SAVE_DIR)))
    app.run(host="0.0.0.0", port=5000, debug=True)
