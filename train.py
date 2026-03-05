import torch
from torch.optim import AdamW
import data_setup
import model_builder
import engine

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Preparing DataLoaders...")
    train_dataloader, val_dataloader = data_setup.create_dataloaders(batch_size=4)

    print("Initializing Model...")
    model = model_builder.create_model(num_labels=19, device=device)
    optimizer = AdamW(model.parameters(), lr=5e-5)

    epochs = 10
    print("Starting Training...")
    for epoch in range(epochs):
        train_loss = engine.train_step(model, train_dataloader, optimizer, device)
        print(f"Epoch: {epoch} | Train Loss: {train_loss:.4f}")
        
    print("Evaluating Model...")
    mIoU = engine.val_step(model, val_dataloader, device)
    print(f"Final mIoU: {mIoU:.4f}")

    checkpoint = "./my_segformer_model"
    model.save_pretrained(checkpoint)
    print(f"Model saved to {checkpoint}")

if __name__ == "__main__":
    main()