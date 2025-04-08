
import gymnasium as gym
from stable_baselines3 import PPO
import wandb
from wandb.integration.sb3 import WandbCallback

group_name = "Names"


env = gym.make('Ant-v5', render_mode ="human")

wandb.init(project="Main_training", sync_tensorboard="True", name="PPO")

ppo_model = PPO('MlpPolicy', env, device="cpu",  verbose = 1, tensorboard_log='log')








ppo_model.learn(total_timesteps=100000, log_interval=4, callback=WandbCallback())
ppo_model.save("ant_model")

ppo_model = PPO.load("ant_model", env=env)

env = gym.make("Ant-v5")
obs, info = env.reset()
while True:
    action, _states = ppo_model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

wandb.finish()

env.close()


