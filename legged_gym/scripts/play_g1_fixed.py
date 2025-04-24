import sys
sys.path.append("/home/yth/project/IK_humanoid/unitree_rl_gym")

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry

import torch
import time

def play_g1_fixed(args):
    # 获取现有的G1配置
    env_cfg, _ = task_registry.get_cfgs(name="g1")
    
    # 修改配置使机器人固定
    env_cfg.asset.fix_base_link = True  # 固定基座
    env_cfg.env.num_envs = 1            # 单个环境
    env_cfg.domain_rand.push_robots = False  # 禁用推动
    env_cfg.terrain.curriculum = False     # 禁用地形课程
    
    # 修改机器人初始姿态，使其更稳定
    env_cfg.init_state.pos = [0.0, 0.0, 0.6]  # 调整初始高度
    
    # 创建环境
    print("正在创建环境...")
    env, _ = task_registry.make_env(name="g1", args=args, env_cfg=env_cfg)
    
    # 获取初始观察
    print("获取初始观察...")
    obs = env.get_observations()
    
    print("G1机器人已固定，保持站立状态")
    
    # 创建零动作向量
    zero_action = torch.zeros((env.num_envs, env.num_actions), device=env.device)
    
    try:
        # 让机器人保持站立状态
        print("保持机器人站立...")
        for i in range(1000):  # 运行大约10秒
            obs, _, _, _, _ = env.step(zero_action)
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("手动停止")
    except Exception as e:
        print(f"执行过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
    
    print("演示完成！")

if __name__ == "__main__":
    args = get_args()
    play_g1_fixed(args)