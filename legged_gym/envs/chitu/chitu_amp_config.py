# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin
import glob

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

MOTION_FILES = glob.glob('datasets/chitu/mocap_motions/*')


class ChituAMPCfg(LeggedRobotCfg):
    # 完全仿照 A1AMPCfg 结构：直接继承 LeggedRobotCfg，平铺式定义所有参数。
    # 不再依赖 ChituRoughCfg，避免 AMP 与 Rough 任务参数耦合。

    class env(LeggedRobotCfg.env):
        num_envs = 5480
        include_history_steps = None  # Number of steps of history to include.
        num_observations = 42
        num_privileged_obs = 48
        reference_state_initialization = True
        reference_state_initialization_prob = 0.85
        amp_motion_files = MOTION_FILES
        # mocap 文件腿序与 A1 相同（PyBullet [FR,FL,RR,RL]），permute [1,3,0,2] → Chitu URDF [LF,LH,RF,RH]
        amp_joint_reorder = 'chitu'
        # 使用仿真足端位置（Chitu 腿长与 A1 解析 FK 不兼容）；A1 仍用解析 FK
        amp_use_sim_foot_pos = True

    class init_state(LeggedRobotCfg.init_state):
        # 与 legged_control chitu reference.info comHeight 对齐
        pos = [0.0, 0.0, 0.60]  # x,y,z [m]
        # OCS2 站立零位，关节名与原版 chitu.urdf (LF/LH/RF/RH) 一致
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'LF_HAA': 0.0,
            'LF_HFE': 0.0,
            'LF_KFE': 0.24,

            'LH_HAA': 0.0,
            'LH_HFE': 0.0,
            'LH_KFE': 0.24,

            'RF_HAA': 0.0,
            'RF_HFE': 0.0,
            'RF_KFE': 0.24,

            'RH_HAA': 0.0,
            'RH_HFE': 0.0,
            'RH_KFE': 0.24,
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'
        # legged_gym 用 dof_name in joint_name 子串匹配；chitu 关节名为 LF_HAA 等，不能用 A1 的 'joint' 键
        # 仿真 AMP 与 a1_amp 同档 (kp=80)；实机 swingLegTask 为 kp=280/kd=10，勿直接用于 Isaac 训练
        stiffness = {'HAA': 180., 'HFE': 180., 'KFE': 180.}  # [N*m/rad]
        damping = {'HAA': 8.0, 'HFE': 8.0, 'KFE': 8.0}    # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'
        measure_heights = False

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/chitu/urdf/chitu.urdf'
        foot_name = "FOOT"
        joint_friction = 0.  # 仿真训练与 A1 一致为 0；实机约 2.0 N·m
        flip_visual_attachments = False  # chitu STL 为 Z-up；勿用 A1 的 y-up 翻转
        # AMP 训练：chitu 腿长大，thigh/calf 触地终止过于频繁会导致 episode 极短、reward 抖动
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter

    class domain_rand:
        randomize_friction = True
        friction_range = [1.5, 3.75]
        randomize_base_mass = True
        added_mass_range = [-1., 1.]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.0
        randomize_gains = True
        stiffness_multiplier_range = [0.9, 1.1]
        damping_multiplier_range = [0.9, 1.1]

    class noise:
        add_noise = True
        noise_level = 1.0  # scales other values
        class noise_scales:
            dof_pos = 0.03
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.3
            gravity = 0.05
            height_measurements = 0.1

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.60
        class scales(LeggedRobotCfg.rewards.scales):
            termination = 0.0
            tracking_lin_vel = 1.5 * 1. / (.005 * 4)
            tracking_ang_vel = 0.5 * 1. / (.005 * 4)
            lin_vel_z = 0.0
            ang_vel_xy = 0.0
            orientation = 0.0
            torques = 0.0
            dof_vel = 0.0
            dof_acc = 0.0
            base_height = 0.0
            feet_air_time = 0.0
            collision = 0.0
            feet_stumble = 0.0
            action_rate = 0.0
            stand_still = 0.0
            dof_pos_limits = 0.0

    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 4  # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading
        resampling_time = 10.  # time before command are changed[s]
        heading_command = False  # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [-1.0, 1.0]  # min max [m/s]
            lin_vel_y = [-0.8, 0.8]  # min max [m/s]
            ang_vel_yaw = [-1.57, 1.57]  # min max [rad/s]
            heading = [-3.14, 3.14]


class ChituAMPCfgPPO(LeggedRobotCfgPPO):
    runner_class_name = 'AMPOnPolicyRunner'
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
        amp_replay_buffer_size = 1000000
        num_learning_epochs = 5
        num_mini_batches = 4

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'chitu_amp_example'
        algorithm_class_name = 'AMPPPO'
        policy_class_name = 'ActorCritic'
        max_iterations = 500000  # number of policy updates

        amp_reward_coef = 2.0
        amp_motion_files = MOTION_FILES
        # mocap 文件腿序与 A1 相同（PyBullet [FR,FL,RR,RL]），permute [1,3,0,2] → Chitu URDF [LF,LH,RF,RH]
        amp_joint_reorder = 'chitu'
        amp_num_preload_transitions = 2000000
        amp_task_reward_lerp = 0.5
        amp_discr_hidden_dims = [1024, 512]

        min_normalized_std = [0.05, 0.02, 0.05] * 4