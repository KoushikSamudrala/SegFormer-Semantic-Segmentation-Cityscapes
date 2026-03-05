import torch
import numpy as np
from PIL import Image
from datasets import load_dataset
from transformers import SegformerImageProcessor
from torch.utils.data import DataLoader

def create_dataloaders(batch_size=4):
    dataset = load_dataset("Chris1/cityscapes") ##Load the Cityscapes dataset from Hugging Face Datasets
    train_data = dataset["train"]
    val_data = dataset["validation"]
    

    ##Initialize the Image processor that will be used to preprocess the images and segmentation maps for the Segformer model. This processor will handle resizing, normalization, and other necessary transformations to prepare the data for training and validation.
    processor = SegformerImageProcessor.from_pretrained("nvidia/mit-b0")
    
    ##Define a transformation function that will be applied to each batch of data during training and validation. This function processes the images and their corresponding segmentation maps, ensuring that they are in the correct format and shape for the Segformer model. It also handles any necessary conversions and adjustments to the segmentation maps, such as setting invalid labels to an ignore index.
    def train_transforms(batch):
        images = batch["image"]
        labels = batch["semantic_segmentation"]
        
        processed_images = []
        processed_segmentation_maps = []
        
        NUM_LABELS = 19
        IGNORE_INDEX = 255

        for img, lbl in zip(images, labels):
            processed_images.append(img)
            
            labels_np = None
            if isinstance(lbl, Image.Image):
                labels_np = np.array(lbl)
            elif isinstance(lbl, torch.Tensor):
                labels_np = lbl.squeeze().cpu().numpy()
            else:
                labels_np = np.array(lbl).squeeze()

            if labels_np.ndim == 3 and labels_np.shape[-1] == 1:
                labels_np = labels_np.squeeze(-1)
                
            labels_np = labels_np.astype(np.uint8)

            if labels_np.ndim == 3 and labels_np.shape[-1] == 3:
                labels_pil = Image.fromarray(labels_np, mode='RGB')
            elif labels_np.ndim == 2:
                labels_pil = Image.fromarray(labels_np, mode='L')
            else:
                raise ValueError(f"Unexpected shape: {labels_np.shape}")

            if labels_pil.mode != "L":
                labels_pil = labels_pil.convert("L")

            final_lbl_array = np.array(labels_pil, dtype=np.uint8)
            final_lbl_array[final_lbl_array >= NUM_LABELS] = IGNORE_INDEX
            processed_segmentation_maps.append(final_lbl_array)

        return processor(images=processed_images, segmentation_maps=processed_segmentation_maps, return_tensors="pt")

    # Apply the defined transformations to the training and validation datasets. This ensures that both datasets are preprocessed in the same way, making them suitable for training and evaluating the Segformer model.
    train_data.set_transform(train_transforms)
    val_data.set_transform(train_transforms)

    # Create DataLoaders for the training and validation datasets. The DataLoader will handle batching and shuffling of the data during training, while ensuring that the validation data is not shuffled to maintain consistency during evaluation.
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader