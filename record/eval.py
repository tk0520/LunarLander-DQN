from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
from gymnasium.wrappers import RecordVideo, Autoreset
import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np
import torch
import os

import settings

BASE_DIR = "./RESCO/grid4x4"
NET_FILE = os.path.join(BASE_DIR, "grid4x4.net.xml")
ROUTE_FILE = os.path.join(BASE_DIR, "grid4x4_1.rou.xml")

def setup_eval_env():
    eval_env = gym.make(
        "LunarLander-v3",
        render_mode="rgb_array",
        continuous=False,           
        gravity=-5,            
        enable_wind=True,           
        wind_power=10,           
        turbulence_power=1.0,       
        max_episode_steps=1500      
    )
    return eval_env

def setup_video_record(eval_env):
    eval_env = RecordVideo(
        eval_env,
        video_folder="./videos",
        episode_trigger=lambda x: True,
        name_prefix="Space"
    )
    return eval_env

def record_video(eval_env, eval_agent):
    observation, info = eval_env.reset()
    total_reward = 0
    done = False
    
    while not done:
        with torch.no_grad():
            action = eval_agent.select_action(observation)

        observation, reward, terminated, truncated, info = eval_env.step(action)
        total_reward += reward
        done = terminated or truncated

    print("Eval total reward:", total_reward)

def record_stat(episode_avg_rewards, replay_losses):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 5))

    # --- 1. Reward Plot ---
    ax1.plot(episode_avg_rewards[::50], color='blue', linewidth=1)
    ax1.set_title("Avg Rewards")
    ax1.set_xlabel("Per 50 Episode")
    ax1.set_ylabel("Avg Reward")

    # # Critic Loss plot
    ax2.plot(replay_losses[::50], color='red', linewidth=1)
    ax2.set_title("Loss Per 50 Episode")
    ax2.set_xlabel("Per 50 Episodes")
    ax2.set_ylabel("Loss")
    ax2.set_yscale('log')

    # # Actor Loss plot
    # ax3.plot(actor_losses, color='green', linewidth=1)
    # ax3.set_title("Actor Loss Per Episode")
    # ax3.set_xlabel("Per Episode")
    # ax3.set_ylabel("Loss")
    # ax3.set_yscale('log')

    plt.tight_layout()
    plt.savefig(f"./stats/{len(episode_avg_rewards)}_graph.png")
    plt.close()