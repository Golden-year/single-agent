from stable_baselines3 import PPO
from gymnasium.utils.env_checker import check_env
from env import CustomEnv

# 创建环境
env = CustomEnv()
check_env(env)

# 让 PPO 运行在 GPU 上
model = PPO("MultiInputPolicy", env, verbose=1, device="cuda", learning_rate=1e-4,
            ent_coef=0.01, vf_coef=0.5, clip_range=0.3, policy_kwargs={"net_arch": dict(pi=[256, 256], vf=[256, 256])})  # 添加 `device="cuda"`

# 训练模型
model.learn(total_timesteps=int(1024))

# 保存模型
model.save("ppo_custom_env")

# 删除模型以测试加载
del model

# 加载训练好的模型
model = PPO.load("ppo_custom_env", device="cuda")  # 确保在 GPU 上加载

# 测试训练好的智能体
obs, _ = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()
    env.render()
