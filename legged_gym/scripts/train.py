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

import numpy as np
import os
from datetime import datetime

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch

def train(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    # 续训时不要随机打乱 episode 进度，避免恢复后短期奖励抖动
    init_random = not train_cfg.runner.resume
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations,
                     init_at_random_ep_len=init_random)

if __name__ == '__main__':
    args = get_args()
    # ============ 训练配置（可通过命令行参数覆盖）============
    # 任务名称：未通过 --task 指定时默认训练 a1
    #   可选: "a1", "a1_amp", "chitu", "chitu_amp"
    #   用法: python train.py --task chitu_amp
    args.task = args.task or 'chitu_amp'

    # 并行环境数量：未通过 --num_envs 指定时默认 4096
    # - 4096: 推荐，显存占用约7GB，训练速度快
    # - 2048: 显存不足时可降低，训练时间会增加
    if args.num_envs is None:
        args.num_envs = 4096

    # 地形：chitu rough 与 A1 相同，继承 LeggedRobotCfg 默认 trimesh；平地请改 chitu_config.py

    # 可视化设置：
    # - True:  无头模式（默认），训练速度快，推荐用于正式训练
    # - False: 开启显示（运行时加 --headless 反而开启，这里强制无头）
    args.headless = True

    # 是否恢复训练（断点续训）：
    # 方式一：命令行（推荐）
    #   python legged_gym/scripts/train.py --task chitu --resume
    #   python legged_gym/scripts/train.py --task chitu --resume --load_run Jun28_14-20-00_
    #   python legged_gym/scripts/train.py --task chitu --resume --load_run Jun28_14-20-00_ --checkpoint 500
    # 方式二：在下方取消注释（等效于命令行参数）
    # args.resume = True
    # args.load_run = 'Jun28_14-20-00_'   # logs/<experiment_name>/ 下的子目录名；不设则用最近一次 run
    # args.checkpoint = 500               # model_500.pt；不设则用该 run 下最新的 model_*.pt
    # 说明：
    # - checkpoint 路径: logs/<experiment_name>/<run_name>/model_<iter>.pt
    #   chitu 默认 experiment_name='rough_chitu'，chitu_amp 为 'chitu_amp_example'
    # - 续训会从 checkpoint 的 iter 接着训到 max_iterations（默认 1500）
    # - 续训会新建一个带时间戳的 log 目录，权重写入新目录（旧 run 保留）
    train(args)
