# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
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

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class ChituRoughCfg(LeggedRobotCfg):

    class env(LeggedRobotCfg.env):
        num_envs = 4096
        include_history_steps = None
        num_observations = 235
        num_privileged_obs = 235
        episode_length_s = 20  # episode length in seconds

    class init_state(LeggedRobotCfg.init_state):
        # 与 legged_control chitu reference.info comHeight 对齐
        pos = [0.0, 0.0, 0.60]  # x,y,z [m]
        # OCS2 站立零位，关节名与原版 chitu.urdf (LF/LH/RF/RH) 一致
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'LF_HAA': 0.00,
            'LF_HFE': 0.00,
            'LF_KFE': 0.24,

            'LH_HAA': 0.00,
            'LH_HFE': 0.00,
            'LH_KFE': 0.24,

            'RF_HAA': 0.00,
            'RF_HFE': 0.00,
            'RF_KFE': 0.24,

            'RH_HAA': 0.00,
            'RH_HFE': 0.00,
            'RH_KFE': 0.24,
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'
        # legged_gym 用 dof_name in joint_name 子串匹配；chitu 关节名为 LF_HAA 等，不能用 A1 的 'joint' 键
        # 实机 legged_control swingLegTask 为 kp=280/kd=10；仿真训练沿用 A1 思路用较低增益，否则大质量 + 高 kp 在 rough 上极易失稳
        stiffness = {'HAA': 180., 'HFE': 180., 'KFE': 180.}  # [N*m/rad]
        damping = {'HAA': 12., 'HFE': 12., 'KFE': 12.}        # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/chitu/urdf/chitu.urdf'
        foot_name = "FOOT"
        # 实机滑动摩擦约 2.0 N·m；仿真训练与 A1 一致设为 0，否则会对抗 PD、站不稳
        joint_friction = 0.
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False  # chitu STL 为 Z-up；勿用 A1 的 y-up 翻转

    class terrain(LeggedRobotCfg.terrain):
        # 86kg 大机器先从较平地形成起步，curriculum 会随表现升降级
        max_init_terrain_level = 2

    class commands(LeggedRobotCfg.commands):
        heading_command = True  # 跟踪heading

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.60  # 与 init_state.pos[2] / comHeight 对齐
        class scales(LeggedRobotCfg.rewards.scales):
            torques = -0.000002
            dof_pos_limits = -10.0


class ChituRoughCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'rough_chitu'