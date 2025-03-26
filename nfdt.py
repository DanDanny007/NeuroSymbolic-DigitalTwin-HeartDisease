import numpy as np
import torch
import torch.nn as nn

class NFDT(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(NFDT, self).__init__()
        self.local_model = nn.Linear(input_dim, hidden_dim)
        self.global_model = nn.Linear(hidden_dim, hidden_dim)
        self.privacy_noise_scale = 0.01

    def local_training(self, local_data):
        local_output = torch.relu(self.local_model(local_data))
        noisy_update = local_output + torch.randn_like(local_output) * self.privacy_noise_scale
        return noisy_update

    def global_aggregation(self, updates):
        aggregated_update = torch.mean(torch.stack(updates), dim=0)
        global_update = torch.relu(self.global_model(aggregated_update))
        return global_update

    def forward(self, distributed_data):
        updates = []
        for local_data in distributed_data:
            update = self.local_training(local_data)
            updates.append(update)
        global_model_output = self.global_aggregation(updates)

        return global_model_output
