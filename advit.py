import numpy as np
import torch
import torch.nn as nn

class ADViT(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ADViT, self).__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.diffusion_steps = 10
        self.lambda_t = 0.01
        self.gamma_t = 0.001
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4)
        self.fc = nn.Linear(hidden_dim, 1)
        self.softmax = nn.Softmax(dim=1)

    def adaptive_diffusion(self, x):
        for t in range(self.diffusion_steps):
            grad = torch.autograd.grad(outputs=x.sum(), inputs=x, create_graph=True)[0]
            noise = torch.randn_like(x)
            x = x - self.lambda_t * grad + self.gamma_t * noise
        return x

    def forward(self, ecg, pcg):
        ecg_encoded = torch.relu(self.encoder(ecg))
        pcg_encoded = torch.relu(self.encoder(pcg))

        ecg_denoised = self.adaptive_diffusion(ecg_encoded)
        pcg_denoised = self.adaptive_diffusion(pcg_encoded)

        combined = torch.stack([ecg_denoised, pcg_denoised], dim=1)
        attn_output, _ = self.attention(combined, combined, combined)
        attn_output = attn_output.mean(dim=1)

        logits = self.fc(attn_output)
        predictions = self.softmax(logits)

        return predictions