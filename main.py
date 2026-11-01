import gymnasium as gym

from agent.agent import SpaceshipAgent 
from record.eval import setup_eval_env, setup_video_record
from train import start_train

train_env = gym.make(
    "LunarLander-v3",
    render_mode=None,
    continuous=False,           
    gravity=-5,            
    enable_wind=True,           
    wind_power=10,           
    turbulence_power=1.0,       
    max_episode_steps=1500      
)
agent = SpaceshipAgent(
    state_space=train_env.observation_space.shape,
    action_space=train_env.action_space.n,
)

eval_env = setup_eval_env()
eval_env = setup_video_record(eval_env)

start_train(train_env, eval_env, agent)
