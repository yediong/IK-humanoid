import sys
sys.path.append("/home/yth/project/IK_humanoid/unitree_rl_gym")

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry

import torch
import numpy as np
import time
import math

def analytical_inverse_kinematics(target_pos, initial_joint_angles=None, arm_joint_indices=None):
    """使用解析方法求解左臂的逆运动学（修正版）
    
    参数:
    target_pos - 目标位置坐标 [x, y, z]
    initial_joint_angles - 初始关节角度（可选）
    arm_joint_indices - 手臂关节索引列表（可选）
    
    返回:
    关节角度 [shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll]
    """
    # 定义手臂参数
    L1 = 0.1032  # shoulder_roll到shoulder_yaw的距离
    L2 = 0.080518  # shoulder_yaw到elbow的距离
    L3 = 0.10  # elbow到wrist的距离
    
    # 肩部偏移
    shoulder_offset_x = 0.0039563
    shoulder_offset_y = 0.10022
    shoulder_offset_z = 0.24778
    
    # 肩部初始旋转（从URDF读取）
    shoulder_rpy = torch.tensor([0.27931, 5.4949E-05, -0.00019159], device=target_pos.device)
    
    # 转换到肩部坐标系
    shoulder_pos = torch.tensor([shoulder_offset_x, shoulder_offset_y, shoulder_offset_z], device=target_pos.device)
    
    # 应用肩部初始旋转的修正
    # 创建旋转矩阵（这里简化处理，实际应该使用完整的旋转矩阵）
    cos_p, sin_p = torch.cos(shoulder_rpy[0]), torch.sin(shoulder_rpy[0])
    R_pitch = torch.tensor([
        [cos_p, 0, sin_p],
        [0, 1, 0],
        [-sin_p, 0, cos_p]
    ], device=target_pos.device)
    
    # 转换目标到肩部坐标系并应用旋转修正
    local_target = target_pos - shoulder_pos
    local_target = torch.matmul(R_pitch, local_target)
    
    # 计算到目标的距离
    r = torch.norm(local_target)
    
    # 检查是否在可达范围内
    max_reach = L1 + L2 + L3
    if r > max_reach:
        scale = max_reach / r
        local_target = local_target * scale
        r = max_reach
    
    # 计算肩部pitch角 - 控制前后方向
    q0 = torch.atan2(-local_target[0], local_target[2])
    
    # 计算肩部roll角 - 控制侧向运动
    projection_yz = torch.sqrt(local_target[1]**2 + local_target[2]**2)
    q1 = torch.atan2(local_target[1], local_target[2])
    
    # 计算肘部角度
    # 使用余弦定理：cos(theta) = (a² + b² - c²) / (2ab)
    # 这里a=L1, b=L2+L3, c=r
    cos_elbow = torch.tensor((r**2 - L1**2 - (L2+L3)**2) / (2 * L1 * (L2+L3)), device=target_pos.device)
    cos_elbow = torch.clamp(cos_elbow, min=-1.0, max=1.0)
    q3 = math.pi - torch.acos(cos_elbow)
    
    # 计算肩部yaw角 - 使手臂能达到目标平面
    # 这里计算方式需要与其他角度配合
    # 基于手臂的工作平面方向
    planar_dist = torch.sqrt(local_target[0]**2 + projection_yz**2)
    q2 = torch.atan2(local_target[0], projection_yz) - q0/2.0  # 修正，避免与q0冗余
    
    # 保持腕部角度或使用默认值
    q4 = torch.tensor(0.0, device=target_pos.device)
    if initial_joint_angles is not None and len(initial_joint_angles) > 4:
        q4 = initial_joint_angles[4]
    
    return torch.tensor([q0, q1, q2, q3, q4], device=target_pos.device)

def play_g1_ik(args):
    try:
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
        
        # 使用23DOF模型
        urdf_path = "unitree_rl_gym/resources/robots/g1_description/g1_23dof_rev_1_0.urdf"
        
        # 创建环境
        print("正在创建环境...")
        env, _ = task_registry.make_env(name="g1", args=args, env_cfg=env_cfg)
        
        # 获取初始观察
        print("获取初始观察...")
        obs = env.get_observations()
        
        print(f"G1机器人已固定，总关节数量: {env.num_actions}")
        
        # 创建零动作向量
        zero_action = torch.zeros((env.num_envs, env.num_actions), device=env.device)
        
        # 找到左右手臂关节和手部链接
        joint_names = [
            # 腿部关节
            'left_hip_pitch', 'left_hip_roll', 'left_hip_yaw',
            'left_knee', 'left_ankle_pitch', 'left_ankle_roll',
            'right_hip_pitch', 'right_hip_roll', 'right_hip_yaw',
            'right_knee', 'right_ankle_pitch', 'right_ankle_roll',
            
            # 躯干和手臂关节 
            'waist_yaw',
            'left_shoulder_pitch', 'left_shoulder_roll', 'left_shoulder_yaw',
            'left_elbow', 'left_wrist_roll',
            'right_shoulder_pitch', 'right_shoulder_roll', 'right_shoulder_yaw',
            'right_elbow', 'right_wrist_roll'
        ]
        
        # 查找左右手臂关节索引
        left_arm_joint_indices = [
            joint_names.index('left_shoulder_pitch'),
            joint_names.index('left_shoulder_roll'),
            joint_names.index('left_shoulder_yaw'),
            joint_names.index('left_elbow'),
            joint_names.index('left_wrist_roll')
        ]
        
        right_arm_joint_indices = [
            joint_names.index('right_shoulder_pitch'),
            joint_names.index('right_shoulder_roll'),
            joint_names.index('right_shoulder_yaw'),
            joint_names.index('right_elbow'),
            joint_names.index('right_wrist_roll')
        ]
        
        # 查找左右手链接索引
        rigid_body_names = env.gym.get_actor_rigid_body_names(env.envs[0], env.actor_handles[0])

        left_hand_name = "left_wrist"  # 根据URDF中的名称调整
        right_hand_name = "right_wrist"  # 根据URDF中的名称调整
        
        left_hand_idx = -1
        right_hand_idx = -1
        
        for i, name in enumerate(rigid_body_names):
            if left_hand_name in name:
                left_hand_idx = i
            if right_hand_name in name:
                right_hand_idx = i
        
        print(f"左手索引: {left_hand_idx}, 右手索引: {right_hand_idx}")
        
        if left_hand_idx == -1 or right_hand_idx == -1:
            print("无法找到手部链接，请检查URDF文件中的链接名称")
            return
        
        # 让机器人稳定一下
        print("让机器人稳定下来...")
        for i in range(100):
            obs, _, _, _, _ = env.step(zero_action)
            time.sleep(0.01)
        
        # 获取刚体状态
        env.gym.refresh_rigid_body_state_tensor(env.sim)
        
        # 确认刚体数量
        num_bodies = env.rigid_body_states.shape[0]
        print(f"刚体总数: {num_bodies}")

        # 获取初始手部位置
        left_hand_init_pos = env.rigid_body_states[left_hand_idx, :3].clone()
        right_hand_init_pos = env.rigid_body_states[right_hand_idx, :3].clone()
        
        print(f"左手初始位置: {left_hand_init_pos}")
        print(f"右手初始位置: {right_hand_init_pos}")
        
        # 测试动作：简单的上下移动（明显的大幅度动作）
        print("开始测试：左手大幅度上下移动")
        
        for t in range(400):
            actions = zero_action.clone()
            
            # 计算当前阶段
            stage = t // 100  # 0-3
            
            # 左手目标位置（明显的上下移动）
            left_target = left_hand_init_pos.clone()
            
            # 四个阶段的动作
            if stage == 0:
                # 第一阶段：手向上移动30cm
                left_target[2] += 0.1
                print(f"阶段1: 左手上移 目标位置: {left_target}") if t % 100 == 0 else None
            elif stage == 1:
                # 第二阶段：手向下移动30cm
                left_target[2] += 0
                print(f"阶段2: 左手下移 目标位置: {left_target}") if t % 100 == 0 else None
            elif stage == 2:
                # 第三阶段：手向前移动30cm
                left_target[0] += 0.1
                print(f"阶段3: 左手前移 目标位置: {left_target}") if t % 100 == 0 else None
            elif stage == 3:
                # 第四阶段：回到初始位置
                left_target = left_hand_init_pos.clone()
                print(f"阶段4: 左手回到初始位置 目标位置: {left_target}") if t % 100 == 0 else None
            
            # 使用解析IK求解左手关节角度
            # 获取当前关节角度作为初始值
            current_joint_angles = env.dof_pos[0, left_arm_joint_indices].clone()
            joint_angles = analytical_inverse_kinematics(left_target, current_joint_angles, left_arm_joint_indices)
            
            # 设置关节角度
            for i, joint_idx in enumerate(left_arm_joint_indices):
                actions[0, joint_idx] = joint_angles[i]
            
            # 执行动作
            obs, _, _, _, _ = env.step(actions)
            
            # 显示当前位置
            if t % 50 == 0:
                left_current = env.rigid_body_states[left_hand_idx, :3]
                print(f"左手当前/目标: {left_current}/{left_target}")
            
            time.sleep(0.01)
        
        # 回到初始位置
        print("回到初始位置...")
        for i in range(100):
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
    play_g1_ik(args)