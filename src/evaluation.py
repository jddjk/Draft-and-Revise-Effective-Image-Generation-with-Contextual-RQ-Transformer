import torch
from models import ContextualRQTransformer, RQVAE
from torchvision.utils import save_image
import os
import logging

logging.basicConfig(filename='training.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def update(S, partitions, model):
    for mask in partitions:
        masked_S = S.clone()
        masked_S[mask] = model.encode(masked_S, mask)
        S[mask] = model.decode(masked_S[mask], masked_S)
    return S

def draft_and_revise(model, initial_S, draft_steps, revise_steps, Tdraft, Trevise):
    S = initial_S.clone()
    draft_partitions = [torch.randint(0, S.size(1), (Tdraft,)) for _ in range(draft_steps)]
    S = update(S, draft_partitions, model)
    
    for _ in range(revise_steps):
        revise_partitions = [torch.randint(0, S.size(1), (Trevise,)) for _ in range(Trevise)]
        S = update(S, revise_partitions, model)
    
    return S


def evaluate_model(model, vae, dataloader, output_dir='./generated_images'):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    vae.eval()
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(dataloader):
            codes = vae.encode(images)
            initial_S = torch.full_like(codes, fill_value=model.codebook.padding_idx)
            generated_codes = draft_and_revise(model, initial_S, draft_steps=2, revise_steps=2, Tdraft=64, Trevise=2)
            generated_images = vae.decode(generated_codes)
            for i, img in enumerate(generated_images):
                save_image(img, os.path.join(output_dir, f'image_{batch_idx * dataloader.batch_size + i}.png'))
            logging.info(f'Saved batch {batch_idx + 1}')
            print(f'Saved batch {batch_idx + 1}')

def save_model(model, vae, model_path='model.pth', vae_path='vae.pth'):
    torch.save(model.state_dict(), model_path)
    torch.save(vae.state_dict(), vae_path)
    logging.info('Model and VAE saved')
    print('Model and VAE saved')

def load_model(model, vae, model_path='model.pth', vae_path='vae.pth'):
    model.load_state_dict(torch.load(model_path))
    vae.load_state_dict(torch.load(vae_path))
    model.eval()
    vae.eval()
    logging.info('Model and VAE loaded')
    print('Model and VAE loaded')
