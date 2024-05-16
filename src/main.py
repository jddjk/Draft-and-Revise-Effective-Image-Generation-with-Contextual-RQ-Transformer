import torch
import torch.optim as optim
import torch.nn as nn
from models import RQVAE, ContextualRQTransformer
from dataset import train_loader, test_loader
from training import train_model
from evaluation import evaluate_model, save_model

if __name__ == "__main__":
    codebook_size = 512
    code_dim = 256
    num_layers = 6
    num_heads = 8
    num_epochs = 3

    vae = RQVAE(codebook_size, code_dim)
    model = ContextualRQTransformer(codebook_size, code_dim, num_layers, num_heads)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-7)
    criterion = nn.CrossEntropyLoss()

    train_model(model, vae, train_loader, optimizer, criterion, num_epochs)
    
    # Save the model
    save_model(model, vae)

    # Evaluate and save images after training
    evaluate_model(model, vae, test_loader)