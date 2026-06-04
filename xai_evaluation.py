import torch
import numpy as np
import torch.nn.functional as F
from scipy.stats import entropy as scipy_entropy

def calculate_entropy(heatmap):
    """
    Calculates the Shannon entropy of the normalized heatmap.
    A more localized/focused heatmap will have lower entropy.
    """
    # Flatten and normalize to a probability distribution
    probs = heatmap.flatten()
    probs = probs / (probs.sum() + 1e-10)
    return scipy_entropy(probs)

def deletion_insertion_aopc(model, image_tensor, heatmap, device, target_class=None, steps=20):
    """
    Calculates Deletion AUC, Insertion AUC, and AOPC.
    
    Steps:
    1. Flatten the heatmap and get sorting indices (descending importance).
    2. Deletion: Start with original image, gradually mask most important pixels.
    3. Insertion: Start with blank image, gradually add most important pixels.
    4. AOPC: Measure drop in confidence for the first few steps.
    """
    model.eval()
    if target_class is None:
        with torch.no_grad():
            logits = model(image_tensor.unsqueeze(0).to(device))
            target_class = logits.argmax(dim=1).item()
    
    heatmap_flat = heatmap.flatten()
    indices = np.argsort(-heatmap_flat)  # High to low
    
    n_pixels = len(heatmap_flat)
    step_size = n_pixels // steps
    
    deletion_scores = []
    insertion_scores = []
    
    # Deletion
    current_img = image_tensor.clone().to(device) # (3, 224, 224)
    # Background for insertion (zeros)
    insertion_img = torch.zeros_like(image_tensor).to(device)
    
    img_flat = current_img.view(3, -1)
    ins_flat = insertion_img.view(3, -1)
    
    with torch.no_grad():
        # Baseline confidence
        base_logits = model(image_tensor.unsqueeze(0).to(device))
        base_conf = torch.softmax(base_logits, dim=1)[0, target_class].item()
        deletion_scores.append(base_conf)
        
        # Start with empty for insertion
        start_logits = model(insertion_img.unsqueeze(0).to(device))
        start_conf = torch.softmax(start_logits, dim=1)[0, target_class].item()
        insertion_scores.append(start_conf)
        
        for i in range(steps):
            # Range of pixels to mask/unmask
            start_idx = i * step_size
            end_idx = min((i + 1) * step_size, n_pixels)
            pixel_indices = indices[start_idx:end_idx]
            
            # Update images
            img_flat[:, pixel_indices] = 0 # Masking with black
            ins_flat[:, pixel_indices] = image_tensor.view(3, -1)[:, pixel_indices].to(device)
            
            # Predict
            d_logits = model(current_img.view(3, 224, 224).unsqueeze(0))
            i_logits = model(insertion_img.view(3, 224, 224).unsqueeze(0))
            
            d_conf = torch.softmax(d_logits, dim=1)[0, target_class].item()
            i_conf = torch.softmax(i_logits, dim=1)[0, target_class].item()
            
            deletion_scores.append(d_conf)
            insertion_scores.append(i_conf)

    deletion_auc = np.mean(deletion_scores)
    insertion_auc = np.mean(insertion_scores)
    
    # AOPC: Average drop in confidence over the steps
    # Usually calculated as 1/L * sum(f(x) - f(x_removed))
    # Here we take the average drop from the baseline for all steps
    aopc = np.mean([base_conf - s for s in deletion_scores[1:]])
    
    return {
        "deletion_auc": deletion_auc,
        "insertion_auc": insertion_auc,
        "aopc": aopc,
        "deletion_curve": deletion_scores,
        "insertion_curve": insertion_scores
    }
