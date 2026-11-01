import numpy as np
import torch
import os

from record.eval import record_stat, record_video
import settings

def start_train(train_envs, eval_env, agent):
    episodes, steps = 0, 0
    episode_avg_rewards = []
    episode_avg_losses = []
    episode_rewards = []
    replay_losses = []

    observation, _ = train_envs.reset()
    total_reward = 0

    while episodes < settings.EPISODES:
        action = agent.select_action(observation)
        next_observation, reward, terminated, truncated, _ = train_envs.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1

        agent.replay_memory.store((
            observation,
            action,
            reward,
            next_observation,
            done,
        ))

        if done:
            print("Total Reward:", total_reward)
            print("Completed Episodes:", episodes)
            print("Memory Size:", len(agent.replay_memory))
            print("Agent Epsilon:", agent.epsilon)

            episodes += 1
            episode_rewards.append(total_reward)
            last_10_avg_rewards = sum(episode_rewards[-10:]) / min(len(episode_rewards), 10)
            episode_avg_rewards.append(last_10_avg_rewards)

            if replay_losses:
                last_10_avg_losses = sum(replay_losses[-10:]) / min(len(replay_losses), 10)
                episode_avg_losses.append(last_10_avg_losses)

            if episodes % settings.STAT_INTERVAL == 0:
                record_stat(episode_avg_rewards, episode_avg_losses)

            if episodes % settings.VIDEO_RECORD_INTERVAL == 0:
                record_video(eval_env, agent)

            observation, _ = train_envs.reset()
            total_reward = 0
        else:
            observation = next_observation

        if len(agent.replay_memory) > settings.INITIAL_MEMORY:
            samples = agent.replay_memory.get_samples()
            loss = agent.update(samples)
            replay_losses.append(loss)

        agent.update_epsilon(steps)
        agent.sync_models(steps)