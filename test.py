import torch

trans_energy_all = torch.tensor([[10, 20, 30, 40, 50],
                                 [60, 70, 80, 90, 100],
                                 [110, 120, 130, 140, 150]])

user_choices = torch.tensor([2, -1, 1])  # 选择索引（-1 代表不选）
uav_mask = torch.tensor([True, False,True])  # 只选第一行
total_trans = trans_energy_all[uav_mask].gather(1, user_choices[uav_mask].view(-1, 1)).sum()

print(user_choices[uav_mask].view(-1, 1))  
# 预期应该是 tensor([[2], [1]])

print(total_trans)
