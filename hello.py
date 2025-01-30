import gymnasium as gym
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback

wandb.login()

run = wandb.init(project="Ant learning", config={"learning_rate": 0.01, "epochs": 10})

class WandBCallback(BaseCallback):
    def __init__(self):
        super(WandBCallback, self).__init__()

    def _on_step(self) -> bool:
        if "rewards" in self.locals:
            wandb.log({"Training Reward": self.locals["rewards"]})
        return True

# Initialise the environment
env = gym.make("Ant-v4", ctrl_cost_weight=0.5, render_mode="human")
env = Monitor(env)


model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1_000_000, callback=WandBCallback())


model.save("ppo_ant_model")

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
wandb.log({"Mean Reward": mean_reward, "Reward Std Dev": std_reward})

# Reset the environment to generate the first observation
observation, info = env.reset(seed=42)
for _ in range(1000):
    # this is where you would insert your policy
    action, _ = model.predict(observation)

    # step (transition) through the environment with the action
    # receiving the next observation, reward and if the episode has terminated or truncated
    observation, reward, terminated, truncated, info = env.step(action)

    wandb.log({"Reward": reward})
    

    # If the episode has ended then we can reset to start a new episode
    if terminated or truncated:
        observation, info = env.reset()

env.close()
wandb.finish()