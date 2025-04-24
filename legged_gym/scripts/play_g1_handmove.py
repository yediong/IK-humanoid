import sys
sys.path.append("/home/yth/project/IK_humanoid/unitree_rl_gym")

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry

import torch
import time
import math

def play_g1_fixed(args):
    # 获取现有的G1配置
    env_cfg, _ = task_registry.get_cfgs(name="g1")
    
    # 修改配置使机器人固定
    env_cfg.asset.fix_base_link = True  # 固定基座
    env_cfg.env.num_envs = 1            # 单个环境
    env_cfg.domain_rand.push_robots = False  # 禁用推动
    env_cfg.domain_rand.randomize_base_mass = False  # 禁用质量随机化
    env_cfg.domain_rand.randomize_friction = False  # 禁用摩擦随机化
    env_cfg.terrain.curriculum = False     # 禁用地形课程
    
    # 修改机器人初始姿态，使其更稳定
    env_cfg.init_state.pos = [0.0, 0.0, 0.6]  # 调整初始高度
    
    # 创建环境
    print("正在创建环境...")
    env, _ = task_registry.make_env(name="g1", args=args, env_cfg=env_cfg)
    
    # 获取初始观察
    print("获取初始观察...")
    obs = env.get_observations()
    
    print(f"G1机器人已固定，总关节数量: {env.num_actions}")
    
    # 创建零动作向量
    zero_action = torch.zeros((env.num_envs, env.num_actions), device=env.device)
    
    # 23自由度模型的关节索引（基于配置文件）
    # 参考官网关节名称
    joint_indices = {
        # 腿部关节 (保持固定)
        'left_hip_pitch': 0, 'left_hip_roll': 1, 'left_hip_yaw': 2,
        'left_knee': 3, 'left_ankle_pitch': 4, 'left_ankle_roll': 5,
        'right_hip_pitch': 6, 'right_hip_roll': 7, 'right_hip_yaw': 8,
        'right_knee': 9, 'right_ankle_pitch': 10, 'right_ankle_roll': 11,
        
        # 躯干和手臂关节 - 修正索引值
        'waist_yaw': 12,
        'left_shoulder_pitch': 13, 'left_shoulder_roll': 14, 'left_shoulder_yaw': 15,
        'left_elbow': 16, 'left_wrist_roll': 17, 
        'right_shoulder_pitch': 18, 'right_shoulder_roll': 19, 'right_shoulder_yaw': 20,
        'right_elbow': 21, 'right_wrist_roll': 22
    }
    
    try:
        # 先让机器人稳定一下
        print("让机器人稳定下来...")
        for i in range(300):
            obs, _, _, _, _ = env.step(zero_action)
            time.sleep(0.01)
        
        # 演示1: 简单的挥手动作
        print("演示1: 左手挥手")
        for t in range(300):
            actions = zero_action.clone()
            phase = t / 100.0
            
            # 左肩前举
            actions[0, joint_indices['left_shoulder_pitch']] = -2
            
            # # 左肘弯曲，做一个挥手动作
            # actions[0, joint_indices['left_elbow']] = 0.8 + 0.3 * math.sin(phase * 2 * math.pi)
            
            # # 手腕旋转
            # actions[0, joint_indices['left_wrist_roll']] = 0.5 * math.sin(phase * 4 * math.pi)
            
            obs, _, _, _, _ = env.step(actions)
            time.sleep(0.01)
        
        # 回到零位
        print("回到初始位置...")
        for i in range(100):
            obs, _, _, _, _ = env.step(zero_action)
            time.sleep(0.01)
            
        # # 演示2: 双手同步动作
        # print("演示2: 双手举起")
        # for t in range(200):
        #     actions = zero_action.clone()
        #     progress = min(t / 100.0, 1.0)  # 0到1的渐进值
            
        #     # 双肩逐渐举起
        #     target_angle = 0.8 * progress
        #     actions[0, joint_indices['left_shoulder_pitch']] = target_angle
        #     actions[0, joint_indices['right_shoulder_pitch']] = target_angle
            
        #     # 双肘弯曲
        #     elbow_angle = 0.5 * progress
        #     actions[0, joint_indices['left_elbow']] = elbow_angle
        #     actions[0, joint_indices['right_elbow']] = elbow_angle
            
        #     obs, _, _, _, _ = env.step(actions)
        #     time.sleep(0.01)
        
        # # 双手在空中挥动
        # print("双手在空中挥动...")
        # for t in range(300):
        #     actions = zero_action.clone()
        #     phase = t / 100.0
            
        #     # 保持肩部举起
        #     actions[0, joint_indices['left_shoulder_pitch']] = 0.8
        #     actions[0, joint_indices['right_shoulder_pitch']] = 0.8
            
        #     # 肘部摆动
        #     actions[0, joint_indices['left_elbow']] = 0.5 + 0.3 * math.sin(phase * 2 * math.pi)
        #     actions[0, joint_indices['right_elbow']] = 0.5 + 0.3 * math.sin(phase * 2 * math.pi + math.pi)  # 反相
            
        #     # 手腕旋转
        #     actions[0, joint_indices['left_wrist_roll']] = 0.5 * math.sin(phase * 3 * math.pi)
        #     actions[0, joint_indices['right_wrist_roll']] = 0.5 * math.sin(phase * 3 * math.pi + math.pi)  # 反相
            
        #     obs, _, _, _, _ = env.step(actions)
        #     time.sleep(0.01)
        
        # # 回到零位
        # print("回到初始位置...")
        # for i in range(100):
        #     obs, _, _, _, _ = env.step(zero_action)
        #     time.sleep(0.01)
        
        # # 演示3: 躯干旋转
        # print("演示3: 躯干旋转")
        # for t in range(300):
        #     actions = zero_action.clone()
        #     phase = t / 300.0 * 2 * math.pi
            
        #     # 躯干左右旋转
        #     actions[0, joint_indices['waist_yaw']] = 0.5 * math.sin(phase)
            
        #     # 手臂姿势
        #     actions[0, joint_indices['left_shoulder_pitch']] = 0.3
        #     actions[0, joint_indices['right_shoulder_pitch']] = 0.3
        #     actions[0, joint_indices['left_elbow']] = 0.5
        #     actions[0, joint_indices['right_elbow']] = 0.5
            
        #     obs, _, _, _, _ = env.step(actions)
        #     time.sleep(0.01)
        
        # # 回到零位
        # print("回到初始位置...")
        # for i in range(100):
        #     obs, _, _, _, _ = env.step(zero_action)
        #     time.sleep(0.01)
            
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