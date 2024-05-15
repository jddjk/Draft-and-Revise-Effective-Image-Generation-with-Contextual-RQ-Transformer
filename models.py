import torch
import torch.nn as nn

class RQVAE(nn.Module):
    def __init__(self, codebook_size, code_dim):
        super(RQVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, code_dim, kernel_size=4, stride=2, padding=1)
        )
        self.codebook = nn.Embedding(codebook_size, code_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(code_dim, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, x):
        z_e = self.encoder(x)
        z_e_flat = z_e.view(z_e.size(0), z_e.size(1), -1).permute(0, 2, 1)  # (batch_size, num_patches, code_dim)
        z_q = self.codebook(z_e_flat.argmax(dim=2))
        z_q = z_q.permute(0, 2, 1).view(z_e.size())
        recon_x = self.decoder(z_q)
        return recon_x, z_q

    def encode(self, x):
        z_e = self.encoder(x)
        return z_e.view(z_e.size(0), z_e.size(1), -1).permute(0, 2, 1).argmax(dim=2)  # (batch_size, num_patches)

    def decode(self, z_q):
        z_q = z_q.permute(0, 2, 1).view(z_q.size(0), -1, int(z_q.size(1)**0.5), int(z_q.size(1)**0.5))
        return self.decoder(z_q)
    
class ContextualRQTransformer(nn.Module):
    def __init__(self, codebook_size, code_dim, num_layers, num_heads):
        super(ContextualRQTransformer, self).__init__()
        self.codebook = nn.Embedding(codebook_size, code_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, codebook_size, code_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=code_dim, nhead=num_heads, batch_first=True)
        decoder_layer = nn.TransformerDecoderLayer(d_model=code_dim, nhead=num_heads, batch_first=True)
        self.bidirectional_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.depth_transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def forward(self, x, mask):
        x = self.codebook(x) + self.positional_encoding[:, :x.size(1), :]
        context = self.bidirectional_transformer(x, src_key_padding_mask=mask)
        depth_outputs = []
        for i in range(x.size(1)):
            depth_input = x[:, i, :].unsqueeze(1)
            depth_output = self.depth_transformer(depth_input, context)
            depth_outputs.append(depth_output)
        depth_outputs = torch.cat(depth_outputs, dim=1)
        return depth_outputs

    def encode(self, x, mask):
        x = self.codebook(x) + self.positional_encoding[:, :x.size(1), :]
        return self.bidirectional_transformer(x, src_key_padding_mask=mask)

    def decode(self, x, context):
        depth_outputs = []
        for i in range(x.size(1)):
            depth_input = x[:, i, :].unsqueeze(1)
            depth_output = self.depth_transformer(depth_input, context)
            depth_outputs.append(depth_output)
        return torch.cat(depth_outputs, dim=1)
