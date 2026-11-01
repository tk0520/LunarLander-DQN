from torchrl.modules import BatchRenorm1d
from torch.nn.utils import weight_norm
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch

class DDPGActorModel:
    def __init__(self, state_space, action_space, device):
        super().__init__()

        self.state_space = state_space
        self.action_space = action_space
        self.device = device

        self.actor = nn.Sequential(
            nn.Linear(state_space, 256),  
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_space)  
        )

class DDPGCriticModel:
    def __init__(self, state_space, action_space, device):
        self.device = device
        self.state_space = state_space
        self.action_space = action_space

        self.value = nn.Sequential(
            nn.Linear(state_space, 256),  
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1) 
        )
