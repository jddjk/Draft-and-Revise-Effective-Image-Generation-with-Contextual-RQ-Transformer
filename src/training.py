import torch
import torch.nn as nn
import torch.optim as optim
from models import ContextualRQTransformer, RQVAE
import logging

logging.basicConfig(filename='training.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def train_model(model, vae, dataloader, optimizer, criterion, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_idx, (images, _) in enumerate(dataloader):
            optimizer.zero_grad()
            codes = vae.encode(images)
            mask = torch.rand(codes.size()) < 0.15
            output = model(codes, mask)
            loss = criterion(output.view(-1, output.size(-1)), codes.view(-1))  # Reshape for cross_entropy
            loss.backward()
            optimizer.step()
            batch_loss = loss.item()
            epoch_loss += batch_loss
            logging.info(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {batch_loss:.7f}')
        avg_loss = epoch_loss / len(dataloader)
        logging.info(f'Epoch [{epoch+1}/{num_epochs}] completed. Average Loss: {avg_loss:.7f}')
        print(f'Epoch [{epoch+1}/{num_epochs}] completed. Average Loss: {avg_loss:.7f}')
