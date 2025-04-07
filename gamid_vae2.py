import copy
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from vae_model import ActorVAE, ActorArchiveBuffer
from torch.nn.utils import parameters_to_vector, vector_to_parameters

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        #self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        self.max_action = max_action
        
    def forward(self, state):
        a = F.relu(self.l1(state))
        #a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))
    
    def get_param_dim(self):
        return sum(p.numel() for p in self.parameters())

    def flatten_actor(self):
        return parameters_to_vector(self.parameters()).detach()
    
    def load_from_flat(self, flat_vector):
        vector_to_parameters(flat_vector, self.parameters())

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        
        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)
        
        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        
        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1

class GamidVae2(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2
    ):
        self.initial_actor_num = 3
        self.actors = nn.ModuleList([
            Actor(state_dim, action_dim, max_action) for _ in range(self.initial_actor_num)
        ]).to(device)
        param_dim = self.actors[0].get_param_dim()
        self.latent_dim = 256
        self.actor_vae = ActorVAE(param_dim=param_dim, latent_dim=self.latent_dim)
        self.vae_buffer = ActorArchiveBuffer(capacity=1000)
        lr = torch.tensor(1e-3, device=device)
        self.vae_optimizer = torch.optim.Adam(self.actor_vae.parameters(), lr=lr)
        self.actor_optimizers = [torch.optim.Adam(actor.parameters(), lr=lr) for actor in self.actors]
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        
        self.action_dim = action_dim
        self.max_action = max_action
        self.criteria = self.max_action * math.sqrt(self.action_dim)
        print('self.criteria: ', self.criteria)
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.total_it = 0

    def convert_flat_idx_to_pair(self, flat_idx, n_actors):
        pairs = [(i, j) for i in range(n_actors) for j in range(i+1, n_actors)]
        return pairs[flat_idx]
    
    def get_idx_pair(self, flat_idx, n_actors):
        # Compute upper-triangular indices (i < j) of pairwise combinations
        idx_i, idx_j = torch.triu_indices(n_actors, n_actors, offset=1)
        return idx_i[flat_idx].item(), idx_j[flat_idx].item()

    @torch.no_grad()
    def calculate_actor_distance(self, next_state):
        #if self.total_it > 5e3 and self.total_it % 1000 == 0:
        if self.total_it % 15000 == 0:
            '''
            next_actions = torch.stack([actor(next_state) for actor in self.actors]).permute(1, 0, 2)
            noise = (torch.randn_like(next_actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_actions = (next_actions + noise).clamp(-self.max_action, self.max_action)
            q_values = torch.stack([
                torch.min(*self.critic_target(next_state, next_actions[:,i,:])) 
                for i in range(next_actions.shape[1]) ], dim=1)
            mean_q_values = q_values.mean(dim=0)
            #min_q = torch.min(mean_q_values)
            #max_q = torch.max(mean_q_values)
            actor_to_elimi = torch.argmin(mean_q_values).item()
            z = torch.randn(1, self.latent_dim).to(device)
            flat_theta = self.actor_vae.decoder(z).squeeze(0)
            self.actors[actor_to_elimi].load_from_flat(flat_theta)
            print('We have actors from new sampling space')
            '''
            actor_to_elimi = 0
            z = torch.randn(1, self.latent_dim).to(device)
            flat_theta = self.actor_vae.decoder(z).squeeze(0)
            self.actors[actor_to_elimi].load_from_flat(flat_theta)
            print('We have actors from new sampling space')

            
        chosen_actor = random.choice(self.actors)
        return chosen_actor(next_state)

    def select_action(self, state):
        # print('self.total_it in VAE code: ', self.total_it)
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        chosen_actor = random.choice(self.actors)
        if self.total_it % 10 == 0:
            self.vae_buffer.add(chosen_actor.flatten_actor())
        return chosen_actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.calculate_actor_distance(next_state) + noise).clamp(-self.max_action, self.max_action)
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        if self.total_it > 1000:
            self.train_vae()
        if self.total_it % self.policy_freq == 0:
            for idx in range(self.initial_actor_num):
                self.actor_optimizers[idx].zero_grad()
            active_actions = torch.stack([actor(state) for actor in self.actors])
            actor_losses = -torch.stack([self.critic.Q1(state, actions).mean() for actions in active_actions])
            actor_losses.sum().backward()
            for idx in range(self.initial_actor_num):
                self.actor_optimizers[idx].step()
            with torch.no_grad():
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def train_vae(self, batch_size=32, beta=0.001):
        self.actor_vae.train()
        x = self.vae_buffer.sample(batch_size).to(device)
        recon, mu, logvar = self.actor_vae(x)

        recon_loss = F.mse_loss(recon, x)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.shape[0]
        loss = recon_loss + beta * kl_loss

        self.vae_optimizer.zero_grad()
        loss.backward()
        self.vae_optimizer.step()

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        torch.save(self.actor_vae.state_dict(), filename + "_vae")
        
        for i, actor in enumerate(self.actors):
            torch.save(actor.state_dict(), filename + f"_actor_{i}")
            torch.save(self.actor_optimizers[i].state_dict(), filename + f"_actor_optimizer_{i}")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        for i, actor in enumerate(self.actors):
            actor.load_state_dict(torch.load(filename + f"_actor_{i}"))
            self.actor_optimizers[i].load_state_dict(torch.load(filename + f"_actor_optimizer_{i}"))
