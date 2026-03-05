from transformers import AutoModelForSemanticSegmentation

def create_model(num_labels=19, device='torch.device("cuda" if torch.cuda.is_available() else "cpu")'):
    model = AutoModelForSemanticSegmentation.from_pretrained("nvidia/mit-b0", num_labels=num_labels)
    model.to(device)
    return model