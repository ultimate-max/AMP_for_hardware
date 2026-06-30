# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

"""Open-loop kinematic playback of AMP mocap trajectories in Isaac Gym."""

import argparse
import os
import sys
import time

import isaacgym
from isaacgym import gymtorch
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry

import numpy as np
import torch


def parse_motion_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--motion_file", type=str, default=None,
        help="Path to a single mocap JSON file. If omitted, uses all files from task config.")
    parser.add_argument(
        "--traj_idx", type=int, default=0,
        help="Index into loaded trajectories (see printed list on startup).")
    parser.add_argument(
        "--loop", action="store_true",
        help="Loop playback when the trajectory ends.")
    parser.add_argument(
        "--speed", type=float, default=1.0,
        help="Playback speed multiplier (1.0 = real-time).")
    parser.add_argument(
        "--list", action="store_true",
        help="List available trajectories and exit (requires env init).")
    motion_args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return motion_args


def resolve_motion_files(env_cfg, motion_file):
    if motion_file is None:
        return env_cfg.env.amp_motion_files
    path = os.path.abspath(motion_file)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Motion file not found: {path}")
    return [path]


def print_trajectories(loader):
    print("\nLoaded mocap trajectories:")
    for i, name in enumerate(loader.trajectory_names):
        n_frames = int(loader.trajectory_num_frames[i])
        duration = loader.trajectory_lens[i]
        fd = loader.trajectory_frame_durations[i]
        print(f"  [{i}] {name}  ({n_frames} frames, {duration:.2f}s, dt={fd:.4f}s)")


def apply_kinematic_frame(env, env_ids, frame):
    """Write one mocap frame into sim without PD control."""
    env._reset_dofs_amp(env_ids, frame)
    env._reset_root_states_amp(env_ids, frame)

    env_ids_int32 = env_ids.to(dtype=torch.int32)
    env.gym.set_dof_state_tensor_indexed(
        env.sim,
        gymtorch.unwrap_tensor(env.dof_state),
        gymtorch.unwrap_tensor(env_ids_int32),
        len(env_ids_int32),
    )
    env.gym.set_actor_root_state_tensor_indexed(
        env.sim,
        gymtorch.unwrap_tensor(env.root_states),
        gymtorch.unwrap_tensor(env_ids_int32),
        len(env_ids_int32),
    )


def play_motion(args, motion_args):
    if args.headless:
        print("WARNING: --headless disables the viewer; playback is meant for visual inspection.")

    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = 1
    env_cfg.terrain.mesh_type = "plane"
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_gains = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.env.reference_state_initialization = True
    env_cfg.env.amp_motion_files = resolve_motion_files(env_cfg, motion_args.motion_file)
    env_cfg.asset.disable_gravity = True
    env_cfg.asset.self_collisions = 1  # playback: skip self-collision for faster stepping

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    loader = env.amp_loader
    print(f"Isaac Gym DOF order: {env.dof_names}")
    print(f"AMP joint reorder: {loader.joint_reorder}")
    print_trajectories(loader)

    if motion_args.list:
        return

    traj_idx = motion_args.traj_idx
    if traj_idx < 0 or traj_idx >= loader.num_motions:
        raise ValueError(
            f"traj_idx={traj_idx} out of range [0, {loader.num_motions - 1}]")

    traj = loader.trajectories_full[traj_idx]
    frame_dt = float(loader.trajectory_frame_durations[traj_idx])
    num_frames = traj.shape[0]
    name = loader.trajectory_names[traj_idx]
    root_xy0 = traj[0, :2].clone()
    print(f"\nPlaying [{traj_idx}] {name}  speed={motion_args.speed}x  loop={motion_args.loop}")
    print("Press Q / close window to quit.\n")

    env_ids = torch.tensor([0], device=env.device, dtype=torch.long)
    frame_idx = 0
    sleep_dt = frame_dt / max(motion_args.speed, 1e-6)
    # Chitu mocap is often 100 Hz (dt=0.01); A1 is ~48 Hz (dt=0.021). Heavy URDF +
    # simulate/render may exceed 10 ms/frame → uneven pacing looks "choppy".
    if frame_dt < 0.015:
        print(f"Note: mocap dt={frame_dt:.4f}s ({1/frame_dt:.0f} Hz). "
              f"If playback stutters, try --speed=0.5 or a lighter sim (--sim_device=cpu).")

    while env.viewer is None or not env.gym.query_viewer_has_closed(env.viewer):
        loop_start = time.perf_counter()

        frame = traj[frame_idx].clone().unsqueeze(0)
        frame[:, 0:2] -= root_xy0.to(frame.device)

        apply_kinematic_frame(env, env_ids, frame)

        env.gym.simulate(env.sim)
        if env.device != "cpu":
            env.gym.fetch_results(env.sim, True)

        env.gym.refresh_actor_root_state_tensor(env.sim)
        env.gym.refresh_rigid_body_state_tensor(env.sim)
        env.gym.refresh_dof_state_tensor(env.sim)

        look_at = env.root_states[0, :3].cpu().numpy()
        camera_pos = look_at + np.array([2.5, -2.5, 1.2], dtype=np.float64)
        env.set_camera(camera_pos, look_at)
        # Do not sync to sim dt (5 ms); we pace with mocap frame_dt below.
        env.render(sync_frame_time=False)

        frame_idx += 1
        if frame_idx >= num_frames:
            if motion_args.loop:
                frame_idx = 0
            else:
                print("Playback finished.")
                break

        if sleep_dt > 0:
            elapsed = time.perf_counter() - loop_start
            time.sleep(max(0.0, sleep_dt - elapsed))


if __name__ == "__main__":
    motion_args = parse_motion_args()
    args = get_args()
    args.task = args.task or "chitu_amp"
    play_motion(args, motion_args)
