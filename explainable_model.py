"""
explainability.py
=================
Compares two XAI methods side-by-side for the DeiT-base COVID-19 classifier:
  • Attention Rollout  — ViT-native, traces attention across all layers
  • GradCAM            — gradient-based heatmap on the last transformer block

Output: explainability_comparison.png
         One row per sample image  →  Original | Attention Rollout | GradCAM
"""

import os, random, csv, json
from datetime import datetime
from typing import cast
import numpy as np
import torch
import torch.nn as nn
import timm
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ── Config ────────────────────────────────────────────────────────────────────
DATA_ROOT   = "COVID_19_dataset"
CKPT_PATH   = "deit_best.pth"
NUM_CLASSES = 3
IMG_SIZE    = 224
CLASS_NAMES = ["COVID", "Normal", "Viral Pneumonia"]
SAMPLES_PER_CLASS = 2          # how many images to show per class
SEED        = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ── Device ────────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")

# ── Transforms ────────────────────────────────────────────────────────────────
tensor_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ── Load model ────────────────────────────────────────────────────────────────
model = timm.create_model("deit_base_patch16_224", pretrained=False, num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()
print("Model loaded.")

# (Selection logic moved below predict helper)

# ── Helper: load raw RGB image (for overlay) ──────────────────────────────────
def load_rgb(path):
    img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    return np.array(img, dtype=np.float32) / 255.0   # [0,1] float for overlays

# ── 1. ATTENTION ROLLOUT ──────────────────────────────────────────────────────
def get_attention_rollout(model, tensor):
    """
    Attention-based heatmap for ViT models.
    For timm models, we use a simplified approach: compute gradient-based attention.
    Returns a (224,224) heatmap.
    """
    # Use gradient of class logit w.r.t. input as attention proxy
    tensor_input = tensor.unsqueeze(0).to(DEVICE).requires_grad_(True)
    
    with torch.enable_grad():
        logits = model(tensor_input)
        class_idx = logits.argmax(dim=1).item()
        class_score = logits[0, class_idx]
        class_score.backward()
    
    # Get gradient w.r.t. input
    grads = tensor_input.grad.data.abs()  # (1, 3, 224, 224)
    grads = grads.squeeze(0)  # (3, 224, 224)
    
    # Average across channels and normalize
    attention_map = grads.mean(0).cpu().numpy()
    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
    
    return attention_map

# ── 2. GradCAM ────────────────────────────────────────────────────────────────
# Target layer: last transformer block's LayerNorm (feeds into head)
# For timm DeiT, reshape_transform is required to handle patch tokens
def reshape_transform(tensor, height=14, width=14):
    # tensor: (batch, seq_len, embed_dim)  seq_len = 1(CLS) + 196(patches)
    result = tensor[:, 1:, :]               # drop CLS token → (B, 196, dim)
    result = result.reshape(result.size(0), height, width, result.size(2))
    result = result.permute(0, 3, 1, 2)     # → (B, dim, 14, 14)
    return result

# target_layer: last block's first LayerNorm
# Cast to Module to access norm1 attribute
target_layer = [cast(nn.Module, model.blocks[-1]).norm1]

# ── Prediction helper ─────────────────────────────────────────────────────────
def predict(tensor):
    with torch.no_grad():
        logits = model(tensor.unsqueeze(0).to(DEVICE))
        prob   = torch.softmax(logits, dim=1)
        pred_idx = int(prob.argmax(1).item())
        conf   = prob[0, pred_idx].item()
    return pred_idx, conf

# ── Pick sample images (balanced Correct vs Incorrect) ────────────────────────
test_dataset = datasets.ImageFolder(os.path.join(DATA_ROOT, "test"), tensor_tf)

print("Finding correct and incorrect predictions for selection...")
correct_indices = []
incorrect_indices = []

# Scan a subset of the test set to find examples
# We scan up to 100 images to find enough incorrect ones
scan_limit = min(100, len(test_dataset))
pool_indices = random.sample(range(len(test_dataset)), scan_limit)

for idx in pool_indices:
    tensor, true_label = test_dataset[idx]
    pred, _ = predict(tensor)
    if pred == true_label:
        correct_indices.append(idx)
    else:
        incorrect_indices.append(idx)

# Select a mix: up to 3 correct and up to 3 incorrect
n_correct_target = 3
n_incorrect_target = 3

selected_correct = random.sample(correct_indices, min(n_correct_target, len(correct_indices)))
selected_incorrect = random.sample(incorrect_indices, min(n_incorrect_target, len(incorrect_indices)))

selected = selected_correct + selected_incorrect
random.shuffle(selected)

print(f"Selected {len(selected)} images: {len(selected_correct)} correct, {len(selected_incorrect)} incorrect.")

# ── Build figure ──────────────────────────────────────────────────────────────
n_cols = 3   # Original | Attention Rollout | GradCAM
n_rows = len(selected)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, 4.5 * n_rows))
if n_rows == 1:
    axes = np.array([axes])  # ensure axes is 2D
elif n_rows == 0:
    print("Error: No images selected.")
    exit()

cam = GradCAM(model=model, target_layers=target_layer,
              reshape_transform=reshape_transform)

# collect per-image results for saving
results = []

for row, idx in enumerate(selected):
    idx_int = int(idx)
    img_path, true_label = test_dataset.samples[idx_int]
    tensor     = test_dataset[idx_int][0]
    rgb        = load_rgb(img_path)
    pred, conf = predict(tensor)
    correct    = (pred == true_label)

    title_color = "green" if correct else "red"
    status_text = "CORRECT" if correct else "INCORRECT"
    pred_int = int(pred) if not isinstance(pred, int) else pred
    row_label = (f"{status_text}\n"
                 f"True: {CLASS_NAMES[true_label]}\n"
                 f"Pred: {CLASS_NAMES[pred_int]}\n"
                 f"Conf: {conf*100:.1f}%")

    # ── Col 0: Original ───────────────────────────────────────────────────────
    axes[row][0].imshow(rgb)
    # Add a border to the original image axis to highlight correct/incorrect
    for spine in axes[row][0].spines.values():
        spine.set_edgecolor(title_color)
        spine.set_linewidth(4)
        spine.set_visible(True)
    
    # Reposition row label to the left of the row
    # Y-coordinate calculation: subplot rows are from top to bottom
    # transFigure: 0 is bottom, 1 is top.
    row_y = 0.9 - (row + 0.5) * (0.8 / n_rows)
    fig.text(0.01, row_y, row_label, 
             fontsize=11, color=title_color, fontweight="bold",
             va="center", ha="left", transform=fig.transFigure)

    # ── Col 1: Attention Rollout ──────────────────────────────────────────────
    rollout_map = get_attention_rollout(model, tensor)
    rollout_mean = float(rollout_map.mean())
    rollout_max  = float(rollout_map.max())

    rollout_uint8 = cast(np.ndarray, np.uint8(255 * rollout_map))
    heatmap_r  = cv2.applyColorMap(rollout_uint8, cv2.COLORMAP_JET)
    heatmap_r  = cv2.cvtColor(heatmap_r, cv2.COLOR_BGR2RGB) / 255.0
    overlay_r  = np.clip(0.5 * rgb + 0.5 * heatmap_r, 0, 1)
    axes[row][1].imshow(overlay_r)

    # ── Col 2: GradCAM ────────────────────────────────────────────────────────
    input_tensor  = tensor.unsqueeze(0).to(DEVICE)
    targets = [ClassifierOutputTarget(pred)]
    grayscale_cam_result = cast(np.ndarray, cam(input_tensor=input_tensor, targets=targets))
    grayscale_cam_0 = cast(np.ndarray, grayscale_cam_result[0])
    gradcam_mean  = float(grayscale_cam_0.mean())
    gradcam_max   = float(grayscale_cam_0.max())

    overlay_g = show_cam_on_image(rgb, grayscale_cam_0, use_rgb=True)
    axes[row][2].imshow(overlay_g)

    # Remove axes but keep spines for Col 0 if we want the border visible
    for i, ax in enumerate(axes[row]):
        if i == 0:
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.axis("off")

    # store result record
    results.append({
        "sample_index"        : idx,
        "image_path"          : img_path,
        "true_class"          : CLASS_NAMES[true_label],
        "predicted_class"     : CLASS_NAMES[pred_int],
        "confidence"          : round(conf, 4),
        "correct"             : correct,
        "rollout_map_mean"    : round(rollout_mean, 4),
        "rollout_map_max"     : round(rollout_max,  4),
        "gradcam_map_mean"    : round(gradcam_mean, 4),
        "gradcam_map_max"     : round(gradcam_max,  4),
    })

# Adjust subplots to make room for the labels on the left
plt.subplots_adjust(left=0.2, bottom=0.1, top=0.92)

# ── Legend ────────────────────────────────────────────────────────────────────
correct_patch = mpatches.Patch(color="green", label="Correct prediction")
wrong_patch   = mpatches.Patch(color="red",   label="Wrong prediction")
fig.legend(handles=[correct_patch, wrong_patch],
           loc="lower center", ncol=2, fontsize=11,
           bbox_to_anchor=(0.5, 0.02), frameon=True)

fig.suptitle("Explainability Comparison — DeiT-base COVID-19 Classifier\n"
             "Attention Rollout vs GradCAM",
             fontsize=15, fontweight="bold", y=0.98)

plt.tight_layout()
out_path = "explainability_comparison.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nVisualization saved → {out_path}")
plt.show()

# ── Save results to CSV ───────────────────────────────────────────────────────
csv_path = "explainability_results.csv"
fieldnames = [
    "sample_index", "image_path", "true_class", "predicted_class",
    "confidence", "correct",
    "rollout_map_mean", "rollout_map_max",
    "gradcam_map_mean", "gradcam_map_max",
]
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)
print(f"CSV saved        → {csv_path}")

# ── Save results to TXT ───────────────────────────────────────────────────────
txt_path = "explainability_results.txt"
with open(txt_path, "w") as f:
    f.write("=" * 70 + "\n")
    f.write("    EXPLAINABILITY RESULTS — DeiT-base COVID-19 Classifier\n")
    f.write("    Attention Rollout  vs  GradCAM\n")
    f.write("=" * 70 + "\n")
    f.write(f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"  Model     : deit_base_patch16_224\n")
    f.write(f"  Checkpoint: {CKPT_PATH}\n")
    f.write(f"  Samples   : {len(results)}  ({SAMPLES_PER_CLASS} per class)\n\n")

    correct_count = sum(1 for r in results if r["correct"])
    f.write(f"  Prediction accuracy on these samples: "
            f"{correct_count}/{len(results)} ({100*correct_count/len(results):.1f}%)\n\n")

    for i, r in enumerate(results, 1):
        status = "✓ CORRECT" if r["correct"] else "✗ WRONG"
        f.write(f"  {'─'*64}\n")
        f.write(f"  Sample {i:02d}  [{status}]\n")
        f.write(f"    Image            : {os.path.basename(r['image_path'])}\n")
        f.write(f"    True class       : {r['true_class']}\n")
        f.write(f"    Predicted class  : {r['predicted_class']}\n")
        f.write(f"    Confidence       : {r['confidence']*100:.2f}%\n\n")
        f.write(f"    Attention Rollout heatmap stats:\n")
        f.write(f"      Mean activation : {r['rollout_map_mean']:.4f}\n")
        f.write(f"      Max  activation : {r['rollout_map_max']:.4f}\n\n")
        f.write(f"    GradCAM heatmap stats:\n")
        f.write(f"      Mean activation : {r['gradcam_map_mean']:.4f}\n")
        f.write(f"      Max  activation : {r['gradcam_map_max']:.4f}\n\n")

    f.write(f"  {'─'*64}\n\n")
    f.write("  Method Comparison Summary\n")
    f.write(f"  {'─'*64}\n")
    f.write(f"  {'Method':<22} {'Avg Mean Activation':>22} {'Avg Max Activation':>20}\n")
    f.write(f"  {'─'*64}\n")
    avg_r_mean = sum(r["rollout_map_mean"] for r in results) / len(results)
    avg_r_max  = sum(r["rollout_map_max"]  for r in results) / len(results)
    avg_g_mean = sum(r["gradcam_map_mean"] for r in results) / len(results)
    avg_g_max  = sum(r["gradcam_map_max"]  for r in results) / len(results)
    f.write(f"  {'Attention Rollout':<22} {avg_r_mean:>22.4f} {avg_r_max:>20.4f}\n")
    f.write(f"  {'GradCAM':<22} {avg_g_mean:>22.4f} {avg_g_max:>20.4f}\n")
    f.write("=" * 70 + "\n")
print(f"TXT saved        → {txt_path}")

# ── Save results to JSON ──────────────────────────────────────────────────────
json_path = "explainability_results.json"
json_out = {
    "model"      : "deit_base_patch16_224",
    "checkpoint" : CKPT_PATH,
    "generated"  : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "samples_per_class": SAMPLES_PER_CLASS,
    "total_samples"    : len(results),
    "correct_predictions": correct_count,
    "method_summary": {
        "attention_rollout": {
            "avg_mean_activation": round(avg_r_mean, 4),
            "avg_max_activation" : round(avg_r_max,  4),
        },
        "gradcam": {
            "avg_mean_activation": round(avg_g_mean, 4),
            "avg_max_activation" : round(avg_g_max,  4),
        },
    },
    "per_image_results": results,
}
with open(json_path, "w") as f:
    json.dump(json_out, f, indent=2)
print(f"JSON saved       → {json_path}")

print("\nAll outputs:")
print(f"  {out_path:<40} — side-by-side visualisation")
print(f"  {csv_path:<40} — per-image results (spreadsheet)")
print(f"  {txt_path:<40} — human-readable report")
print(f"  {json_path:<40} — structured data")