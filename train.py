from matplotlib import colorizer
import os                       
import subprocess
import torch                   
import torch.nn as nn  
from torch.optim import AdamW    
from torch.optim.lr_scheduler import CosineAnnealingLR  
from torchvision import datasets, transforms  
from torch.utils.data import DataLoader       
from collections import Counter  
import timm                      
import matplotlib.pyplot as plt 
import seaborn as sns            
from sklearn.metrics import classification_report,confusion_matrix  
from tqdm import tqdm

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("Using MPS")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("Using CUDA")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU")

# Prevent laptop from sleeping during training
try:
    subprocess.Popen(["caffeinate", "-i"])
    print("Caffeinate enabled — laptop won't sleep during training")
except FileNotFoundError:
    print("Warning: caffeinate not found, laptop may sleep during training")

DATA_ROOT    = "COVID_19_dataset"   # folder that contains train/ val/ test/
CKPT_PATH    = "deit_best.pth"      # where to save the best model weights
NUM_CLASSES  = 3                    # COVID, Normal, Viral Pneumonia
IMG_SIZE     = 224                  # DeiT expects a 224 x 224 image 
BATCH_SIZE   = 64                   # how many images to process at once
EPOCHS       = 25                   # how many full passes through the data
LR           = 3e-4                 # learning rate — how big each update step is
WEIGHT_DECAY = 1e-4                 # regularization — prevents overfitting
CLASS_NAMES  = ["COVID", "Normal", "Viral Pneumonia"]
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),        
    transforms.RandomRotation(10),            
    transforms.ColorJitter(brightness=0.2),   
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(os.path.join(DATA_ROOT, "train"),train_transforms)
val_dataset   = datasets.ImageFolder(os.path.join(DATA_ROOT, "val"),val_transforms)
test_dataset  = datasets.ImageFolder(os.path.join(DATA_ROOT, "test"),val_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

class_names = train_dataset.classes
print("Class names:", class_names)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

count = Counter(train_dataset.targets)
total = sum(count.values())
weights = torch.tensor( [total / (NUM_CLASSES * count[i]) for i in range(NUM_CLASSES)], dtype=torch.float).to(DEVICE) # for class imbalance and focussing more on minority classes

model = timm.create_model("deit_base_patch16_224", pretrained=True, num_classes=NUM_CLASSES) 
model = model.to(DEVICE) 
# freezing the weights of the model so that pretrained knowledge is not lost
for param in model.parameters(): 
    param.requires_grad = False
# now unfreezing the weights of the last few layers because they are relevant for classfication task
for name, param in model.named_parameters():
    if any(f"blocks.{i}" in name for i in [8, 9, 10, 11]):
        param.requires_grad = True
    if "head" in name or "norm" in name:
        param.requires_grad = True
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters()) 
print(f"Trainable parameters: {trainable} of {total_params} ({100*trainable/total_params:.2f}%)")

criterion = nn.CrossEntropyLoss(weight=weights).to(DEVICE)
optimizer = AdamW(filter(lambda p : p.requires_grad, model.parameters()), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

# We track these to plot curves later
history = {"train_loss": [], "val_loss": [], "val_acc": []}
best_val_acc = 0.0

for epoch in range(1, EPOCHS + 1):

    # Training phase
    model.train()          # tells model to enable dropout etc.
    running_loss = 0.0

    for imgs, labels in train_loader:
        imgs   = imgs.to(DEVICE)    # move images to GPU
        labels = labels.to(DEVICE)  # move labels to GPU

        optimizer.zero_grad()        # clear gradients from last step
        logits = model(imgs)         # forward pass: model makes predictions
        loss   = criterion(logits, labels)  # measure the error
        loss.backward()              # backward pass: compute gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()             # update weights using gradients

        running_loss += loss.item() * imgs.size(0)

    train_loss = running_loss / len(train_dataset)

    # Validation phase 
    model.eval()           # tells model to disable dropout etc.
    val_loss, correct, total_n = 0.0, 0, 0

    with torch.no_grad():  # no gradients needed — saves memory and time
        for imgs, labels in val_loader:
            imgs   = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            logits = model(imgs)
            loss   = criterion(logits, labels)
            val_loss += loss.item() * imgs.size(0)
            correct  += (logits.argmax(1) == labels).sum().item()
            total_n  += labels.size(0)

    val_loss /= len(val_dataset)
    val_acc   = correct / total_n
    scheduler.step()

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    # Save model only when val accuracy improves
    saved = ""
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), CKPT_PATH)
        saved = "saved"

    print(f"Epoch {epoch:02d}/{EPOCHS} | "
          f"train loss {train_loss:.4f} | "
          f"val loss {val_loss:.4f} | "
          f"val acc {val_acc:.4f}{saved}")

print(f"\nDone. Best val acc: {best_val_acc:.4f}")

# Plot training curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
x = range(1, EPOCHS + 1)

ax1.plot(x, history["train_loss"], label="Train loss")
ax1.plot(x, history["val_loss"],   label="Val loss")
ax1.set_title("Loss curves")
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
ax1.legend(); ax1.grid(alpha=0.3)

ax2.plot(x, history["val_acc"], color="seagreen", label="Val accuracy")
ax2.set_title("Validation accuracy")
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
ax2.legend(); ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
plt.show()