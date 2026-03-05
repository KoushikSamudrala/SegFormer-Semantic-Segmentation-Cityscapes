# SegFormer for Semantic Segmentation on Cityscapes 🏙️🚗



An end-to-end, modular PyTorch pipeline implementing a Vision Transformer (`nvidia/mit-b0` SegFormer) to perform pixel-level semantic segmentation on urban street scenes. 

## 🚀 Project Overview

Instead of relying on pre-packaged high-level Trainer APIs, this project features a fully custom implementation written natively in PyTorch and Hugging Face. It demonstrates deep tensor manipulation, custom training loops, and robust data sanitization.

**Key Engineering Highlights:**
* **Engineered Robust Data Pipelines:** Architected custom `train_transforms` to dynamically handle dimensionality reduction across PIL Images, PyTorch Tensors, and NumPy arrays.
* **Sanitized Labels & Tensors:** Intercepted RGB masks, forced channel standardization to 2D Grayscale (`L` mode), and strictly mapped out-of-bounds pixel classes to a `255` ignore index to prevent cross-entropy calculation errors.
* **Architected Modular Systems:** Separated concerns into a decoupled factory pattern (`data_setup.py`, `model_builder.py`, `engine.py`) to easily swap architectures and datasets.

---

## 🏗️ Project Structure & Code Implementation

This project is divided into modular files for clean coding practices. Below is the complete code for each module:

### 1. `data_setup.py`
Handles dataset downloading, the `SegformerImageProcessor`, and custom tensor dimension squeezing.

```python
import torch
import numpy as np
from PIL import Image
from datasets import load_dataset
from transformers import SegformerImageProcessor
from torch.utils.data import DataLoader

def create_dataloaders(batch_size=4):
    # Load Dataset
    dataset = load_dataset("Chris1/cityscapes")
    train_data = dataset["train"]
    val_data = dataset["validation"]

    # Load Processor
    processor = SegformerImageProcessor.from_pretrained("nvidia/mit-b0")

    # Custom Transforms
    def train_transforms(batch):
        images = batch["image"] 
        labels = batch["semantic_segmentation"] 

        processed_images = []
        processed_segmentation_maps = []

      
```

### 2. `model_builder.py`
Initializes the model architecture. Isolated to allow for easy model swapping in the future.

```python
from transformers import AutoModelForSemanticSegmentation

def create_model(num_labels=19, device="cpu"):
    model = AutoModelForSemanticSegmentation.from_pretrained("nvidia/mit-b0", num_labels=num_labels)
    model.to(device)
    return model
```

### 3. `engine.py`
The training and validation logic. Strictly handles the forward passes, backpropagation, and metric evaluation.

```python
import torch
from evaluate import load

def train_step(model, train_dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in train_dataloader:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        
        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        total_loss += loss.item()
    return total_loss / len(train_dataloader)


```

### 4. `train.py`
The main execution script. Orchestrates the modules and saves the fully trained weights.

```python

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Preparing DataLoaders...")
    train_dataloader, val_dataloader = data_setup.create_dataloaders(batch_size=4)

    print("Initializing Model...")
    model = model_builder.create_model(num_labels=19, device=device)
    optimizer = AdamW(model.parameters(), lr=5e-5)


    checkpoint = "./my_segformer_model"
    model.save_pretrained(checkpoint)
    print(f"Model saved to {checkpoint}")

if __name__ == "__main__":
    main()
```

### 5. `requirements.txt`
```text
torch
datasets
transformers
evaluate
numpy
Pillow
```

---

## ⚙️ How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/SegFormer-Cityscapes-Segmentation.git](https://github.com/YOUR_USERNAME/SegFormer-Cityscapes-Segmentation.git)
   cd SegFormer-Cityscapes-Segmentation
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Execute the pipeline:**
   ```bash
   python train.py
   ```

## 📊 Evaluation Metrics
The model is evaluated using **Mean Intersection over Union (mIoU)**. The pipeline is explicitly configured to assess accuracy across 19 distinct urban classes while disregarding unclassified background pixels (Index 255).
