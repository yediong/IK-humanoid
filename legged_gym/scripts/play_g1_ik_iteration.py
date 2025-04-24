import sys
sys.path.append("/home/yth/project/IK_humanoid/unitree_rl_gym")

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry

import torch
import numpy as np
import time
import math
import xml.etree.ElementTree as ET

def compute_hand_jacobian(env, hand_idx, arm_joint_indices):
    """通过解析方法计算手部雅可比矩阵
    
    参数:
    env - 仿真环境
    arm_joint_indices - 手臂关节索引
    
    返回:
    位置雅可比矩阵 (3 x 关节数)
    """
    device = env.device
    
    # 获取当前关节角度
    q = env.dof_pos[0, arm_joint_indices].clone()
    
    # 链接长度参数
    L1 = 0.1032  # shoulder_roll到shoulder_yaw的距离
    L2 = 0.080518  # shoulder_yaw到elbow的距离
    L3 = 0.10  # elbow到wrist的距离
    
    # 雅可比矩阵初始化
    J = torch.zeros((3, len(arm_joint_indices)), device=device)
    
    # 关节角度和三角函数值
    q0, q1, q2, q3, q4 = q[0], q[1], q[2], q[3], q[4]
    
    # 预计算三角函数值
    c0, s0 = torch.cos(q0), torch.sin(q0)
    c1, s1 = torch.cos(q1), torch.sin(q1)
    c2, s2 = torch.cos(q2), torch.sin(q2)
    c3, s3 = torch.cos(q3), torch.sin(q3)
    c4, s4 = torch.cos(q4), torch.sin(q4)
    
    # 计算各关节轴的旋转矩阵
    
    # 对shoulder_pitch (q0)的雅可比列
    J[0, 0] = -s0*(L1 + L2*c3 + L3*c3)
    J[1, 0] = 0
    J[2, 0] = -c0*(L1 + L2*c3 + L3*c3)
    
    # 对shoulder_roll (q1)的雅可比列
    J[0, 1] = c0*s1*(L2*c3 + L3*c3)
    J[1, 1] = c1*(L2*c3 + L3*c3)
    J[2, 1] = s0*s1*(L2*c3 + L3*c3)
    
    # 对shoulder_yaw (q2)的雅可比列
    J[0, 2] = c0*c1*L2*s3 + c0*c1*L3*s3
    J[1, 2] = -s1*L2*s3 - s1*L3*s3
    J[2, 2] = s0*c1*L2*s3 + s0*c1*L3*s3
    
    # 对elbow (q3)的雅可比列
    J[0, 3] = -c0*c1*s2*L2*s3 - c0*c1*s2*L3*s3 + c0*c1*c2*L2*c3 + c0*c1*c2*L3*c3
    J[1, 3] = s1*s2*L2*s3 + s1*s2*L3*s3 - s1*c2*L2*c3 - s1*c2*L3*c3
    J[2, 3] = -s0*c1*s2*L2*s3 - s0*c1*s2*L3*s3 + s0*c1*c2*L2*c3 + s0*c1*c2*L3*c3
    
    # 对wrist_roll (q4)的雅可比列 - 纯旋转关节，对位置没有影响
    J[0, 4] = 0
    J[1, 4] = 0
    J[2, 4] = 0
    # print(f"J_pos norm: {torch.norm(J)}")
    return J



def ik_solve_damped_least_squares(jacobian, error, damping=0.01):
    """使用阻尼最小二乘法求解逆运动学"""
    # 确保维度正确
    # jacobian:(3, num_joints)
    # error:(3,)
    
    # J^T * J + λ^2 * I
    j_t = torch.transpose(jacobian, 0, 1)  # 转置为(num_joints, 3)
    reg = damping * torch.eye(jacobian.shape[1], device=jacobian.device)
    lhs = torch.matmul(j_t, jacobian) + reg
    
    # J^T * error
    rhs = torch.matmul(j_t, error)
    
    # 求解线性方程组
    try:
        delta_theta = torch.linalg.solve(lhs, rhs)
    except:
        print("求解线性方程组失败，使用伪逆作为后备")
        delta_theta = torch.matmul(torch.linalg.pinv(lhs), rhs)

    # print(f"delta_theta: {delta_theta}")
    
    return delta_theta

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
        
        env_cfg.control.control_type = "P"  # Position control模式

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
        for i in range(200):
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
        
        # IK参数
        damping = 0.5
        max_iterations = 10
        error_threshold = 0.01
      
        # 测试动作：简单的上下移动（明显的大幅度动作）
        print("开始测试：左手大幅度上下移动")
        current_actions = torch.zeros((1, env.num_actions), device=env.device)
        # 保存原始关节角度
        for j in range(env.num_actions):
            current_actions[0, j] = env.dof_pos[0, j]

        for t in range(80):
            actions = zero_action.clone()
            print(f"t: {t}")

            # current_actions = env.dof_pos.clone()
            # print(f"current_actions: {current_actions[0, 13]}")
            # print(f"env.dof_pos: {env.dof_pos[0, 13]}")

            # 计算当前阶段
            stage = t // 20  # 0-3
        
            # 左手目标位置（明显的上下移动）
            left_target = left_hand_init_pos.clone()
            
            # 四个阶段的动作
            if stage == 0:
                # 第一阶段：手向上移动10cm
                left_target[2] += 0.1
                print(f"阶段1: 左手上移 目标位置: {left_target}") if t % 20 == 0 else None
            elif stage == 1:
                # 第二阶段：手向下移动10cm(即回到原状态)
                left_target[2] += 0
                print(f"阶段2: 左手下移 目标位置: {left_target}") if t % 20 == 0 else None
            elif stage == 2:
                # 第三阶段：手向左移动10cm
                left_target[1] += 0.1
                print(f"阶段3: 左手左移 目标位置: {left_target}") if t % 20 == 0 else None
            elif stage == 3:
                # 第四阶段：回到初始位置
                left_target = left_hand_init_pos.clone()
                print(f"阶段4: 左手回到初始位置 目标位置: {left_target}") if t % 20 == 0 else None
            

            # 左手IK求解
            # print(f"开始求解IK，当前时间步: {t}, 目标位置: {left_target}")
            for iteration in range(max_iterations):
                # 当前手部位置
                current_left_pos = env.rigid_body_states[left_hand_idx, :3].clone()
                
                # 计算位置误差
                left_pos_error = left_target - current_left_pos
                error_magnitude = torch.norm(left_pos_error).item()
                
                # 如果误差足够小，退出迭代
                if error_magnitude < error_threshold:
                    print(f"  迭代{iteration}: 误差{error_magnitude:.4f} < 阈值，收敛")
                    break
                
                # 计算左手雅可比矩阵
                jacobian_pos = compute_hand_jacobian(env, left_hand_idx, left_arm_joint_indices)
                
                # 求解IK
                delta_theta = ik_solve_damped_least_squares(jacobian_pos, left_pos_error, damping)*0.6
                
                # delta_theta[0] = -0.02
                # delta_theta[1] = 0
                # delta_theta[2] = 0
                # delta_theta[3] = 0
                # delta_theta[4] = 0

                # 更新动作
                for i, joint_idx in enumerate(left_arm_joint_indices):
                    current_actions[0, joint_idx] += delta_theta[i]
                
                # 执行动作并更新环境
                obs, _, _, _, _ = env.step(current_actions)
                
                env.gym.refresh_rigid_body_state_tensor(env.sim)
                env.gym.refresh_dof_state_tensor(env.sim)  # ← 新增关键行
                # print(f"  迭代{iteration}: 误差={error_magnitude:.4f}, 关节角度变化={[delta_theta[i].item() for i in range(len(delta_theta))]}")
            
            
            # 显示当前位置
            if t % 20 == 0:
                left_current = env.rigid_body_states[left_hand_idx, :3]
                right_current = env.rigid_body_states[right_hand_idx, :3]
                # print(f"左手当前/目标: {left_current}/{left_target}, 右手当前/目标: {right_current}/{right_target}")
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