import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorEncoder(nn.Module):
    def __init__(self, param_dim, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(param_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.mu = nn.Linear(256, latent_dim)
        self.logvar = nn.Linear(256, latent_dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.mu(h), self.logvar(h)

class ActorDecoder(nn.Module):
    def __init__(self, latent_dim, param_dim):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, param_dim)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        return self.fc3(h)

from collections import deque

class ActorArchiveBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def add(self, flat_params):
        self.buffer.append(flat_params.detach().clone())

    def sample(self, batch_size):
        indices = torch.randint(0, len(self.buffer), (batch_size,))
        return torch.stack([self.buffer[i] for i in indices])

class ActorVAE(nn.Module):
    def __init__(self, param_dim, latent_dim):
        super().__init__()
        self.encoder = ActorEncoder(param_dim, latent_dim)
        self.decoder = ActorDecoder(latent_dim, param_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar
