import os
import csv
import json
from datetime import datetime
from typing import cast
import torch
import timm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
)
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ── Config (must match train.py) ──────────────────────────────────────────────
DATA_ROOT   = "COVID_19_dataset"
CKPT_PATH   = "deit_best.pth"
NUM_CLASSES = 3
IMG_SIZE    = 224
BATCH_SIZE  = 64
CLASS_NAMES = ["COVID", "Normal", "Viral Pneumonia"]

# ── Device ────────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")

# ── Test transforms (same as val_transforms in train.py) ─────────────────────
test_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

test_dataset = datasets.ImageFolder(
    os.path.join(DATA_ROOT, "test"), test_transforms
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)
print(f"Test samples: {len(test_dataset)}")

# ── Load model ────────────────────────────────────────────────────────────────
model = timm.create_model(
    "deit_base_patch16_224", pretrained=False, num_classes=NUM_CLASSES
)
model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()
print("Model loaded from", CKPT_PATH)

# ── Inference ─────────────────────────────────────────────────────────────────
all_preds, all_labels = [], []

with torch.no_grad(), tqdm(total=len(test_dataset), desc="Evaluating", unit="img") as pbar:
    for imgs, labels in test_loader:
        preds = model(imgs.to(DEVICE)).argmax(1).cpu()
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())
        pbar.update(len(imgs))

# ── Metrics ───────────────────────────────────────────────────────────────────
acc = accuracy_score(all_labels, all_preds)
precision, recall, f1, support = cast(
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    precision_recall_fscore_support(
        all_labels, all_preds, average=None, labels=[0, 1, 2]
    )
)
macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
    all_labels, all_preds, average="macro"
)
weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
    all_labels, all_preds, average="weighted"
)

# ── Console report ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("          TEST SET EVALUATION REPORT — DeiT-base")
print("=" * 60)
print(f"\n  Overall Accuracy : {acc:.4f}  ({acc*100:.2f}%)\n")
print(f"  {'Class':<20} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
print("  " + "-" * 54)
for i, cls in enumerate(CLASS_NAMES):
    print(f"  {cls:<20} {precision[i]:>10.4f} {recall[i]:>10.4f} {f1[i]:>10.4f} {int(support[i]):>10}")
print("  " + "-" * 54)
print(f"  {'Macro avg':<20} {macro_p:>10.4f} {macro_r:>10.4f} {macro_f1:>10.4f} {len(all_labels):>10}")
print(f"  {'Weighted avg':<20} {weighted_p:>10.4f} {weighted_r:>10.4f} {weighted_f1:>10.4f} {len(all_labels):>10}")
print("=" * 60)
print("\nFull Classification Report:")
print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES, digits=4, output_dict=False))

# ── Save metrics to CSV ───────────────────────────────────────────────────────
csv_path = "evaluation_metrics.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Model", "DeiT-base (deit_base_patch16_224)"])
    writer.writerow(["Checkpoint", CKPT_PATH])
    writer.writerow(["Evaluated on", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
    writer.writerow(["Test samples", len(test_dataset)])
    writer.writerow([])
    writer.writerow(["Overall Accuracy", f"{acc:.4f}"])
    writer.writerow([])
    writer.writerow(["Class", "Precision", "Recall", "F1-Score", "Support"])
    for i, cls in enumerate(CLASS_NAMES):
        writer.writerow([cls, f"{precision[i]:.4f}", f"{recall[i]:.4f}", f"{f1[i]:.4f}", int(support[i])])
    writer.writerow(["Macro avg", f"{macro_p:.4f}", f"{macro_r:.4f}", f"{macro_f1:.4f}", len(all_labels)])
    writer.writerow(["Weighted avg", f"{weighted_p:.4f}", f"{weighted_r:.4f}", f"{weighted_f1:.4f}", len(all_labels)])
print(f"Metrics saved to {csv_path}")

# ── Save metrics to TXT ───────────────────────────────────────────────────────
txt_path = "evaluation_metrics.txt"
with open(txt_path, "w") as f:
    f.write("=" * 60 + "\n")
    f.write("     TEST SET EVALUATION REPORT — DeiT-base\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"  Model      : DeiT-base (deit_base_patch16_224)\n")
    f.write(f"  Checkpoint : {CKPT_PATH}\n")
    f.write(f"  Evaluated  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"  Test set   : {len(test_dataset)} images\n\n")
    f.write(f"  Overall Accuracy : {acc:.4f}  ({acc*100:.2f}%)\n\n")
    f.write(f"  {'Class':<20} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}\n")
    f.write("  " + "-" * 54 + "\n")
    for i, cls in enumerate(CLASS_NAMES):
        f.write(f"  {cls:<20} {precision[i]:>10.4f} {recall[i]:>10.4f} {f1[i]:>10.4f} {int(support[i]):>10}\n")
    f.write("  " + "-" * 54 + "\n")
    f.write(f"  {'Macro avg':<20} {macro_p:>10.4f} {macro_r:>10.4f} {macro_f1:>10.4f} {len(all_labels):>10}\n")
    f.write(f"  {'Weighted avg':<20} {weighted_p:>10.4f} {weighted_r:>10.4f} {weighted_f1:>10.4f} {len(all_labels):>10}\n")
    f.write("=" * 60 + "\n\n")
    f.write("Full Classification Report:\n\n")
    f.write(cast(str, classification_report(all_labels, all_preds, target_names=CLASS_NAMES, digits=4, output_dict=False)))
print(f"Metrics saved to {txt_path}")

# ── Save metrics to JSON ──────────────────────────────────────────────────────
json_path = "evaluation_metrics.json"
metrics_dict = {
    "model": "DeiT-base (deit_base_patch16_224)",
    "checkpoint": CKPT_PATH,
    "evaluated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "test_samples": len(test_dataset),
    "overall_accuracy": round(float(acc), 4),
    "per_class": {
        cls: {
            "precision": round(float(precision[i]), 4),
            "recall":    round(float(recall[i]), 4),
            "f1_score":  round(float(f1[i]), 4),
            "support":   int(support[i]),
        }
        for i, cls in enumerate(CLASS_NAMES)
    },
    "macro_avg": {
        "precision": round(float(macro_p), 4),
        "recall":    round(float(macro_r), 4),
        "f1_score":  round(float(macro_f1), 4),
    },
    "weighted_avg": {
        "precision": round(float(weighted_p), 4),
        "recall":    round(float(weighted_r), 4),
        "f1_score":  round(float(weighted_f1), 4),
    },
}
with open(json_path, "w") as f:
    json.dump(metrics_dict, f, indent=2)
print(f"Metrics saved to {json_path}")

# ── Confusion Matrix ──────────────────────────────────────────────────────────
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(7, 6))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
    linewidths=0.5, linecolor="gray",
)
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.title("Confusion Matrix — DeiT-base COVID-19 Classifier", fontsize=13, pad=12)
plt.tight_layout()
plt.savefig("confusion_matrix_eval.png", dpi=150)
print("Confusion matrix saved to confusion_matrix_eval.png")
plt.show()

print("\nAll outputs saved:")
print(f"  {csv_path}  — spreadsheet-friendly metrics")
print(f"  {txt_path}  — human-readable report")
print(f"  {json_path} — structured data for programmatic use")
print(f"  confusion_matrix_eval.png — visual confusion matrix")