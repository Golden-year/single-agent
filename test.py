from stable_baselines3 import PPO
import torch
import gymnasium as gym
from env import CustomEnv  # 确保你的环境文件正确导入

# 检查 CUDA 是否可用
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 加载环境
env = CustomEnv()  # 你的自定义环境
obs, _ = env.reset()

# 加载训练好的模型
model_path = "./ppo_custom_env_final.zip"  # 确保路径正确
model = PPO.load(model_path, device=device)

# 运行测试
for _ in range(1000):  
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, done, truncated, info = env.step(action)
    env.render()
    
    if done or truncated:
        obs, _ = env.reset()

env.close()
print("Model testing complete.")
