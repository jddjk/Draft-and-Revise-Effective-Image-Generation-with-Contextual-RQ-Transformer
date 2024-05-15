import torch
import torch.nn as nn
import torch.optim as optim
from models import ContextualRQTransformer, RQVAE

def train_model(model, vae, dataloader, optimizer, criterion, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for images, _ in dataloader:
            optimizer.zero_grad()
            codes = vae.encode(images)
            mask = torch.rand(codes.size()) < 0.15
            output = model(codes, mask)
            loss = criterion(output.view(-1, output.size(-1)), codes.view(-1))  # Reshape for cross_entropy
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.7f}')
