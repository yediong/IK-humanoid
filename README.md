# Overview
Using inverse kinematics (IK) solving to achieve partial hand movements for the Unitree G1 robot, based on the [RL example code framework](https://github.com/unitreerobotics/unitree_rl_gym) from Unitree's official website. 

# Quick Start
The following describes how this project will be deployed. Configuration method reference for IsaacGym, rsl_rl and raw file (unitree_rl_gym): [rl_control_routine](https://support.unitree.com/home/zh/G1_developer/rl_control_routine)

## environment
1. Users are recommanded to run this project in Ubuntu 20.04 and ROS noetic environment.
2. Users are recommanded to use the following versions of the software:
    - Python 3.8
    - torch==1.10.0+cu113

## Dependencies
1. [IssacGym](https://developer.nvidia.com/isaac-gym)
2. [rsl_rl](https://github.com/leggedrobotics/rsl_rl)

Put these two folders in the same project directory as the project files.

## Instructions
Creating a Virtual Environment:
```bash
conda create -n rl-g1 python=3.8
conda activate rl-g1
```

Installing CUDA, pytorch:
```bash
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

Download the Isaac Gym Preview 4 simulation platform, unzip it into the python directory, and install it using pip:
```bash
# current directory: isaacgym/python
pip install -e .
```
Install the rsl_rl library (please use version 1.0.2):
```bash
git clone https://github.com/leggedrobotics/rsl_rl
cd rsl_rl
git checkout v1.0.2
pip install -e .
```


## Run
1. Deploy the code for this project：

```bash
git clone https://github.com/yediong/IK_humanoid.git
```

2. Modify sys.path.append(“/home/unitree/h1/legged_gym”) in legged_gym/scripts/play_g1_handmove.py , legged_gym/scripts/play_g1_ik.py for your own path.

3. Activate RL environments:
```bash
conda activate rl-g1
```

4. Switch to the legged_gym/scripts directory and start test.
- Control the specific joint to move an angle through the code of setting the actions and 'step':
```bash
python play_g1_handmove.py
```

- Inverse Kinematic: Move the left hand to a specific position in 3d environment, using Jacobi matrices and damped least squares methods:
```bash
python play_g1_ik.py
```

## usage
Modify the parameters(positions, angles, etc) in the related pyfiles to adjust the robot's behavior.


# Notes
This project provides beginners with a basic routine for deploying the Unitree G1 robot operation using the Isaacgym and rsl_rl platforms. Additional fine-tuning of parameters or more advanced methods may be required for better performance. Questions are welcome.
