import os
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common import env_checker
from stable_baselines3.common.utils import get_schedule_fn
from torch.nn import LeakyReLU
from Gym_CosmicVoyager_Frame_GPTo1_stack_v3 import CosmicVoyageEnv  # Import your environment
from dotenv import load_dotenv
import wandb
import imageio
import cv2  # Needed for image processing

# Set up the project directory
save_dir = r'C:\Users\Andy\Documents\GitHub\RL-Training_CosmicVoyager'  # Replace with your actual path
os.chdir(save_dir)
# Load environment variables from .env file
dotenv_path = r'C:\Users\Andy\Documents\GitHub\RL-Training_CosmicVoyager\rl_env\apikeys.env'  # Replace with the actual path to your .env file
load_dotenv(dotenv_path=dotenv_path)

# Now you can access WANDB_API_KEY as before
wandb_api_key = os.getenv('WANDB_API_KEY')

# Check if the API key is available
if wandb_api_key is None:
    raise ValueError("WandB API key not found. Please set the WANDB_API_KEY environment variable.")

print("CUDA Available:", torch.cuda.is_available())

# Initialize WandB project
# Replace 'YOUR_ACTUAL_WANDB_API_KEY' with your actual API key
wandb.login(key=wandb_api_key)
wandb.init(project='Cosmic Voyager RL', entity='andiB1293', config={
    "learning_rate": 5e-5,            # Reduced learning rate
    "batch_size": 256,                # Increased batch size
    "n_steps": 2048,                  # Keeping n_steps the same
    "n_epochs": 20,                   # Increased number of epochs
    "max_grad_norm": 0.75,             # Adjusted max_grad_norm
    "gamma": 0.95,
    "gae_lambda": 0.95,
    "clip_range": 0.1,                # Reduced clip range
    "ent_coef": 0.02,                 # Increased entropy coefficient
    "target_kl": 0.025,                # Added target KL divergence
    "total_timesteps": 1000000,       # Additional training timesteps
    "policy": "CnnPolicy"
})

# Create the environment with adjusted observation dimensions and frame stacking
env = CosmicVoyageEnv(
    width=400,
    height=600,
    observation_width=100,
    observation_height=150,
    cooldown_steps=2,
    num_stacked_frames=8
)

# Verify the environment
env_checker.check_env(env)

# Create the evaluation environment with the same dimensions
eval_env = CosmicVoyageEnv(
    width=400,
    height=600,
    observation_width=100,
    observation_height=150,
    cooldown_steps=2,
    num_stacked_frames=8
)

# Define the policy keyword arguments (must match the saved model's architecture)
policy_kwargs = dict(
    features_extractor_kwargs=dict(features_dim=1024),  # Must match the original features_dim
    net_arch=[dict(pi=[1024, 512, 256], vf=[1024, 512, 256])],  # Must match the original network architecture
    activation_fn=LeakyReLU,
    normalize_images=False,
)

# Load the previously trained model
model_path = r'C:\Users\Andy\Documents\GitHub\RL-Training_CosmicVoyager\checkpoints\model_RL_Cosmic_latest.zip'  # Replace with your actual path

model = PPO.load(model_path, env=env)

# Update hyperparameters
model.learning_rate = wandb.config.learning_rate
model.ent_coef = wandb.config.ent_coef
model.clip_range = wandb.config.clip_range
model.n_epochs = wandb.config.n_epochs
model.batch_size = wandb.config.batch_size
model.target_kl = wandb.config.target_kl
model.max_grad_norm = wandb.config.max_grad_norm

# Update the learning rate schedule
model.lr_schedule = get_schedule_fn(model.learning_rate)

# **Update the clip range schedule**
model.clip_range = get_schedule_fn(model.clip_range)

# Reinitialize the optimizer with the new learning rate
model.optimizer = model.policy.optimizer_class(
    model.policy.parameters(),
    lr=model.learning_rate,
    **model.policy.optimizer_kwargs
)

# Configure the logger to include desired metrics
from stable_baselines3.common.logger import configure
new_logger = configure(folder='logs', format_strings=['stdout', 'csv'])
model.set_logger(new_logger)

# Custom callback to log additional metrics
class CustomRLCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CustomRLCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.total_rewards = 0
        self.episode_lengths = []
        self.total_steps = 0

    def _on_step(self) -> bool:
        self.total_rewards += self.locals["rewards"][0]
        self.total_steps += 1

        if self.locals["dones"][0]:
            self.episode_rewards.append(self.total_rewards)
            self.episode_lengths.append(self.total_steps)
            # Log metrics every 10 episodes
            if len(self.episode_rewards) % 10 == 0:
                average_reward = np.mean(self.episode_rewards[-10:])
                average_length = np.mean(self.episode_lengths[-10:])
                print(f"Episode {len(self.episode_rewards)}: Average Reward: {average_reward}, Average Length: {average_length}")

                # Access training logs
                logs = self.model.logger.name_to_value
                policy_loss = logs.get("train/policy_loss")
                entropy_loss = logs.get("train/entropy_loss")

                # Prepare data for logging
                log_data = {
                    "Average Reward per Episode": average_reward,
                    "Average Episode Length": average_length,
                }
                if policy_loss is not None:
                    log_data["Policy Loss"] = policy_loss
                if entropy_loss is not None:
                    log_data["Entropy Loss"] = entropy_loss

                # Log metrics to WandB
                wandb.log(log_data)

            self.total_rewards = 0
            self.total_steps = 0

        return True

    def _on_training_end(self):
        print("Training completed!")

# Video recorder callback (assuming you have this defined)
class VideoRecorderCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq=5000, video_length=200, video_folder='./videos/', verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.video_length = video_length
        self.video_folder = video_folder
        os.makedirs(video_folder, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            print(f"Recording video at step {self.n_calls}...")
            images = []
            obs, _ = self.eval_env.reset()
            for _ in range(self.video_length):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, done, _, _ = self.eval_env.step(action)
                img = self.eval_env.render(mode='rgb_array')
                images.append(img)
                if done:
                    obs, _ = self.eval_env.reset()
            # Save the video
            video_path = os.path.join(self.video_folder, f"step_{self.n_calls}.mp4")
            imageio.mimsave(video_path, [np.array(img) for img in images], fps=10)
            print(f"Saved video at {video_path}")
            # Optionally, log the video to WandB
            wandb.log({
                "training_video": wandb.Video(video_path, fps=10, format="mp4")
            })
        return True

# Observation logger callback (assuming you have this defined)
class ObservationLoggerCallback(BaseCallback):
    def __init__(self, eval_freq=5000, observation_folder='./observations/', verbose=0):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.observation_folder = observation_folder
        os.makedirs(observation_folder, exist_ok=True)
    
    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq == 0:
            print(f"Logging agent's observation at step {self.num_timesteps}...")
            # Access the current environment
            env = self.training_env.envs[0]
            # Get the current observation using the public method
            obs = env.get_current_observation()
            if obs is not None:
                try:
                    # obs shape is (num_stacked_frames, H, W)
                    # Concatenate frames for visualization
                    obs_imgs = [(obs[i] * 255).astype(np.uint8) for i in range(obs.shape[0])]
                    # Stack images horizontally
                    obs_img = np.concatenate(obs_imgs, axis=1)
                    # Save the observation as an image
                    obs_image_path = os.path.join(self.observation_folder, f"obs_{self.num_timesteps}.png")
                    imageio.imwrite(obs_image_path, obs_img)
                    # Optionally, log the observation to WandB
                    wandb.log({
                        "agent_observation": wandb.Image(obs_img, caption=f"Observation at step {self.num_timesteps}")
                    })
                except Exception as e:
                    print(f"Error saving observation image: {e}")
            else:
                print("No observation available to log.")
        return True

# Checkpoint callback to save the model every 200,000 steps
class CheckpointCallback(BaseCallback):
    def __init__(self, save_freq, save_path, verbose=0):
        super(CheckpointCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)
    
    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            checkpoint_path = os.path.join(self.save_path, f"model_{self.num_timesteps}_steps.zip")
            self.model.save(checkpoint_path)
            print(f"Saved checkpoint at {checkpoint_path}")
        return True

# Create the callbacks
custom_callback = CustomRLCallback()
video_recorder = VideoRecorderCallback(
    eval_env=eval_env,
    eval_freq=5000,      # Record every 5000 steps
    video_length=200,    # Record 200 steps per video
    video_folder='./videos/',
)
observation_logger = ObservationLoggerCallback(
    eval_freq=5000,
    observation_folder='./observations/'
)
checkpoint_callback = CheckpointCallback(
    save_freq=200000,  # Save every 200,000 steps
    save_path='./checkpoints/'
)

# Combine all callbacks
callback = CallbackList([
    custom_callback,
    video_recorder,
    observation_logger,
    checkpoint_callback
])

# Continue training
additional_timesteps = wandb.config.total_timesteps
model.learn(total_timesteps=additional_timesteps, callback=callback)

# Save the updated model
model.save("cosmic_voyage_ppo_final_continued")

# Close the environments
env.close()
eval_env.close()

# Finish WandB session
wandb.finish()
