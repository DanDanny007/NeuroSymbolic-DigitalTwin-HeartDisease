import numpy as np
import torch
import torch.nn as nn

class SCDN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SCDN, self).__init__()
        self.causal_layer = nn.Linear(input_dim, hidden_dim)
        self.structure_layer = nn.Linear(hidden_dim, hidden_dim)
        self.symbolic_layer = nn.Linear(hidden_dim, hidden_dim)
        self.decision_layer = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, X):
        C = torch.sigmoid(self.causal_layer(X))
        latent = torch.relu(self.structure_layer(torch.matmul(C, X)))
        symbolic_representation = torch.relu(self.symbolic_layer(latent))
        decision_output = torch.softmax(self.decision_layer(symbolic_representation), dim=1)

        threshold = 0.5
        final_causal_graph = (decision_output > threshold).float() * C

        return final_causal_graph
