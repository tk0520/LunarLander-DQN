from torch.optim import RMSprop
import torch.nn.functional as F
import numpy as np
import random
import torch
import time

from agent.memory import ReplayMemory
from agent.model import DQN
from agent.value import QLerning
import settings

class SpaceshipAgent:
    def __init__(self, state_space, action_space):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_space = state_space
        self.action_space = action_space

        self.current_model = DQN(self.state_space, self.action_space, self.device).to(device=self.device)
        self.target_model = DQN(self.state_space, self.action_space, self.device).to(device=self.device)

        self.optimizer = RMSprop(self.current_model.parameters(), lr=settings.LEARNING_RATE, alpha=0.95)
        self.target_model.load_state_dict(self.current_model.state_dict())
        
        self.replay_memory = ReplayMemory()
        self.learner = QLerning(self)        
        self.epsilon = settings.EPSILON_START

        self.env_steps = 0
        self.last_sync_step = 0

    def sync_models(self, env_steps):
        if env_steps - self.last_sync_step >= settings.SYNC_INTERVAL:
            self.target_model.load_state_dict(self.current_model.state_dict())
            self.last_sync_step = env_steps
            print(f"[Sync] Target model synced at env_step={env_steps}")

    def update_epsilon(self, env_steps):
        learning_steps = max(0, env_steps - settings.INITIAL_MEMORY)
        frac = min(1.0, learning_steps / settings.EPSILON_DECAY_STEPS)
        self.epsilon = settings.EPSILON_START + frac * (settings.EPSILON_END - settings.EPSILON_START)

    def current_evaluate(self, observation):
        return self.current_model.forward(observation)
    
    def target_evaluate(self, observation):
        with torch.no_grad():
            return self.target_model.forward(observation)

    def select_action(self, observation):
        q_values = self.current_model(observation)
        action = torch.argmax(q_values).item()
        return action

    def update(self, samples):
        observations, actions, rewards, next_observations, terminated = zip(*samples)
        observations = torch.tensor(np.array(observations), dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_observations = torch.tensor(np.array(next_observations), dtype=torch.float32).to(self.device)
        terminated = torch.tensor(terminated, dtype=torch.float32).to(self.device)
        
        current_q = self.current_evaluate(observations)
        current_q_selected = current_q.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_current_q = self.current_evaluate(next_observations)
            best_actions = next_current_q.argmax(dim=1)

            next_target_q = self.target_evaluate(next_observations)
            next_target_q = next_target_q.gather(1, best_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + settings.DISCOUNT * next_target_q * (1 - terminated)
        
        # Huber Loss
        loss = F.smooth_l1_loss(current_q_selected, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(
            self.current_model.parameters(), settings.MAX_GRAD_NORM
        )
        self.optimizer.step()

        return loss.item()

        # if observation_batch.ndim == 3:
        #     observation_batch = np.expand_dims(observation_batch, axis=0)
        #     obs_tensor = torch.from_numpy(observation_batch).to(next(self.current_model.parameters()).device)
        #     q_values = self.current_model(obs_tensor)
        #     actions = torch.argmax(q_values, dim=1).cpu().numpy()
        #     return actions
        
        # p = np.random.rand(self.batch_size)  
        
        # with torch.no_grad():
        #     obs_tensor = torch.from_numpy(observation_batch).to(next(self.current_model.parameters()).device)
        #     q_values = self.current_model(obs_tensor)
        #     actions = torch.argmax(q_values, dim=1).cpu().numpy()

        # explore_mask = p < self.epsilon
        # if np.any(explore_mask):
        #     actions[explore_mask] = np.random.randint(0, self.action_size, size=explore_mask.sum())
        
        # actions = actions.astype(np.int32)
        # return actions
