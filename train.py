from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import VecNormalize
from env import CustomEnv
import os
def main():
    # 创建环境
    num_envs = 4  # 根据硬件性能调整
    env = make_vec_env(CustomEnv, n_envs=num_envs, vec_env_cls=DummyVecEnv)
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    # TensorBoard日志目录
    log_dir = "./ppo_tensorboard/"
    os.makedirs(log_dir, exist_ok=True)

    # 让 PPO 运行在 GPU 上
    model = PPO("MultiInputPolicy",
                env, 
                verbose=1, 
                device="cuda", 
                learning_rate=5e-4,
                ent_coef=0.01, 
                vf_coef=0.5, 
                policy_kwargs={"net_arch": dict(pi=[256, 256], vf=[256, 256])},
                tensorboard_log=log_dir
            ) 

    # 回调函数
    eval_callback = EvalCallback(
        env,
        best_model_save_path="./best_model/",  # 保存最佳模型的路径
        log_path="./eval_logs/",  # 评估日志路径
        eval_freq=10000,  # 每500步评估一次
        deterministic=True,
        render=False
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,  # 每1000步保存一次模型
        save_path="./checkpoints/",  # 模型保存路径
        name_prefix="ppo_custom_env"
    )

    # 训练模型
    model.learn(total_timesteps=int(1024),
                callback=[eval_callback, checkpoint_callback])

    # 保存模型
    model.save("ppo_custom_env_final")

    # 删除模型以测试加载
    del model


if __name__ == "__main__":
    main()
