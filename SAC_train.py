import gymnasium as gym

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import SAC
import wandb
from wandb.integration.sb3 import WandbCallback


env = gym.make('Ant-v5', render_mode='human')

wandb.init(project="Main_training", sync_tensorboard="True", name="SAC")

SAC_model = SAC("MlpPolicy", env, verbose=1, tensorboard_log='log')


SAC_model.learn(total_timesteps= 100000, log_interval=4, callback=WandbCallback())

SAC_model.save("ant_model", env=env)

env = gym.make("Ant-v5")
obs, info = env.reset()
while True:
    action, _states = SAC_model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

wandb.finish()

env.close()


