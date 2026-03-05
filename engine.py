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

def val_step(model, val_dataloader, device):
    model.eval()
    metric = load("mean_iou")
    
    with torch.no_grad():
        for batch in val_dataloader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(pixel_values=pixel_values, labels=labels)
            predicted_masks = outputs.logits.argmax(dim=1)
            
            metric.add_batch(predictions=predicted_masks, references=labels)
            
    final_score = metric.compute(num_labels=19, ignore_index=255)
    return final_score["mean_iou"]