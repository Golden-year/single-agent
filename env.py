import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np
import torch

class CustomEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(CustomEnv, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.steps = 0

        # 设置随机种子
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        self.__init__users()

        # 定义动作空间和状态空间
        self.action_space = spaces.MultiDiscrete([3] * 15)
        self.observation_space = spaces.Dict({
            "drone_positions": spaces.Box(
                low=np.array([0, 0, 40] * 5),
                high=np.array([1000, 1000, 300] * 5),
                dtype=np.float32
            ),
            "user_task_means": spaces.Box(
                low=1.75,  # (3+0.5)/2
                high=3.0,  # (5+1)/2
                shape=(441,),
                dtype=np.float32
            )
        })

    def __init__users(self):
        """初始化用户相关数据（GPU版本）"""
        # 用户位置（形状 [441, 3]）
        self.user_positions = torch.empty((441, 3), device=self.device)
        self.user_positions[:, :2] = torch.rand((441, 2), device=self.device) * 1000
        self.user_positions[:, 2] = 0  # 用户高度固定为0
        
        # 用户计算能力 [441]
        self.user_computing_power = torch.rand(441, device=self.device) * 0.5 + 0.5
        
        # 用户任务缓存
        self.user_tasks = torch.empty((441, 2), device=self.device)
        self.user_choices = torch.full((441,), -1, dtype=torch.long, device=self.device)  # -1表示local      

    def reset(self, seed=None, options=None):
        self.np_random, seed = seeding.np_random(seed)
        super().reset(seed=seed)
        self.steps = 0
        # 初始化无人机位置
        self.drone_positions = torch.tensor([
            [0, 0, 40],
            [0, 1000, 40],
            [1000, 0, 40],
            [1000, 1000, 40],
            [500, 500, 40]
        ], dtype=torch.float32, device=self.device)

        # 生成用户任务（批量操作）
        self.user_tasks[:, 0] = torch.tensor(self.np_random.uniform(3, 5, size=441), device=self.device)  # [3,5)
        self.user_tasks[:, 1] = torch.tensor(self.np_random.uniform(0.5, 1.0, size=441), device=self.device)   # [0.5,1.0)
        
        # 重置用户选择
        self.user_choices.fill_(-1)
        user_means = torch.mean(self.user_tasks, dim=1).cpu().numpy()
        
        return ({"drone_positions": self.drone_positions.cpu().numpy().flatten(), "user_task_means": user_means}, {})

    def step(self, action):
        action = torch.tensor(action, dtype=torch.long, device=self.device)  
        drone_actions = (action - 1).view(5, 3) 

        self.drone_positions = self.drone_positions + drone_actions

        distance = torch.norm(self.drone_positions.unsqueeze(1) - self.drone_positions.unsqueeze(0), dim=2)
        distance = distance + torch.eye(self.drone_positions.size(0), device=self.device) * 1000
        out_of_bounds = ((self.drone_positions < torch.tensor([0, 0, 40], device=self.device)) |
            (self.drone_positions > torch.tensor([1000, 1000, 300], device=self.device))).any()
        collision = (distance < 20).any()
        truncated = out_of_bounds or collision
        
        done = self.steps >= 1000
        reward = self._calculate_reward()
        if truncated:
            reward -= 100
        self.steps += 1
        info = {}
        return ({"drone_positions": self.drone_positions.flatten().cpu().numpy(), "user_task_means": torch.mean(self.user_tasks, dim=1).cpu().numpy()}, reward, done, truncated, info)
        #ToDo 返回信息还可以更改

    def _calculate_reward(self):
        local_energy = 0.1 * (self.user_computing_power ** 2) * self.user_tasks[:, 0]
        path_loss_matrix = self.calculate_path_loss_matrix()
        trans_energy_all = self._batch_calculate_trans_energy(path_loss_matrix)  # [441,5]

        self.match_users_to_drones(path_loss_matrix, trans_energy_all, local_energy)

        is_local = (self.user_choices == -1)
        uav_mask = (self.user_choices != -1)
        total_local = local_energy[is_local].sum()
        total_trans = trans_energy_all[uav_mask].gather(1, self.user_choices[uav_mask].view(-1, 1)).sum()

        return - (total_local + total_trans)

    def _batch_calculate_trans_energy(self, path_loss_matrix):
        """批量计算传输能耗（全矩阵）"""
        snr = 10 ** (-path_loss_matrix / 10) / 1e-11
        rate = torch.log2(1 + snr)
        trans_time = self.user_tasks[:, 1].unsqueeze(1) / rate
        return 0.1 * trans_time

    def calculate_path_loss_matrix(self):
        delta = self.user_positions.unsqueeze(1) - self.drone_positions.unsqueeze(0)
        dist = torch.norm(delta, dim=2)

        H = self.drone_positions[:, 2]
        H_factor = torch.clamp(23.9 - 1.8 * torch.log10(H), min=20)
        d_factor = torch.log10(dist)
        return H_factor * d_factor + 20 * torch.log10(torch.tensor(4 * np.pi * 80, device=self.device))

    def render(self, mode='human'):
        print(f"Drone positions: {self.drone_positions.cpu().numpy()}")

    def close(self):
        pass
    def match_users_to_drones(self, path_loss_matrix, trans_energy_all, local_energy):
        """优化后的用户匹配（使用GPU张量操作）"""
        threshold = 111  # 设置路径损耗阈值

        valid_mask = (path_loss_matrix <= threshold) & (trans_energy_all < local_energy.unsqueeze(1))

        energy_saving = local_energy.unsqueeze(1) - trans_energy_all

        masked_saving = torch.where(valid_mask, energy_saving, torch.tensor(-float('inf'), device=self.device))
        
        # 步骤4：找到每个用户的最大节能量及其对应无人机索引
        max_saving_values, max_saving_indices = torch.max(masked_saving, dim=1)
        
        # 步骤5：创建最终选择
        # 条件1：最大节能量 > 0（实际有节能）
        # 条件2：索引有效（非-1）
        final_mask = (max_saving_values > 0)
        
        # 初始化选择为-1（local）
        self.user_choices = torch.full((441,), -1, 
            dtype=torch.long, 
            device=self.device)
        
        # 应用有效选择
        self.user_choices[final_mask] = max_saving_indices[final_mask]