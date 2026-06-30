# AMP 运动先验数据格式说明

本文档描述 `a1_amp` / `chitu_amp` 训练所用的 mocap JSON 数据格式，以及文件内关节顺序与 Isaac Gym URDF 的对应关系。

相关代码：

- 加载与重排：`rsl_rl/rsl_rl/datasets/motion_loader.py`（`AMPLoader`）
- A1 配置：`legged_gym/envs/a1/a1_amp_config.py`（`amp_joint_reorder = 'a1'`，默认）
- Chitu 配置：`legged_gym/envs/chitu/chitu_amp_config.py`（`amp_joint_reorder = 'chitu'`）
- 可视化校验：`legged_gym/scripts/play_motion.py`

---

## 目录结构

```
datasets/
├── a1/
│   ├── mocap_motions/     # A1 AMP 轨迹（trot、pace、turn 等）
│   └── hopturn/
└── chitu/
    └── mocap_motions/     # Chitu AMP 轨迹（LeggedController 导出）
```

---

## 文件格式（JSON）

每个 `.txt` 文件为 JSON，结构如下：

```json
{
  "LoopMode": "Wrap",
  "FrameDuration": 0.021,
  "EnableCycleOffsetPosition": true,
  "EnableCycleOffsetRotation": true,
  "MotionWeight": 0.5,
  "Frames": [
    [ /* 61 个 float */ ],
    [ /* 61 个 float */ ]
  ]
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `LoopMode` | string | 循环模式（如 `"Wrap"`），当前 loader 未使用 |
| `FrameDuration` | float | 相邻帧时间间隔（秒）。A1 多为 `0.021`；Chitu 多为 `0.01` |
| `EnableCycleOffsetPosition` | bool | 导出工具字段，loader 未使用 |
| `EnableCycleOffsetRotation` | bool | 同上 |
| `MotionWeight` | float | 多轨迹采样权重，loader 加载后会归一化 |
| `Frames` | float[][] | 轨迹帧列表，**每帧必须恰好 61 维** |

---

## 单帧 61 维布局（通用）

所有机器人共用同一帧结构；差异仅在 **关节/足端的腿顺序**（见下文 A1 / Chitu 专节）。

```
索引    长度   字段                    单位     坐标系
────────────────────────────────────────────────────────────
[0:3]     3    root_pos               m        世界系 (x, y, z)
[3:7]     4    root_rot               -        世界系四元数 (x, y, z, w)
[7:19]   12    joint_pos              rad      文件内腿序，每腿 3 DoF
[19:31]  12    tar_toe_pos_local      m        机身/base 系，4 足 × (x,y,z)
[31:34]   3    linear_vel             m/s      见 AMPLoader 存储方式
[34:37]   3    angular_vel            rad/s    同上
[37:49]  12    joint_vel              rad/s    与 joint_pos 同腿序
[49:61]  12    tar_toe_vel_local      m/s      base 系；AMP 训练未使用
────────────────────────────────────────────────────────────
合计     61
```

代码中的索引常量（`motion_loader.py`）：

| 常量 | 值 | 含义 |
|------|-----|------|
| `ROOT_POS_START_IDX` | 0 | 根位置起始 |
| `ROOT_ROT_START_IDX` | 3 | 根姿态起始 |
| `JOINT_POSE_START_IDX` | 7 | 关节角起始 |
| `TAR_TOE_POS_LOCAL_START_IDX` | 19 | 足端位置起始 |
| `LINEAR_VEL_START_IDX` | 31 | 线速度起始 |
| `ANGULAR_VEL_START_IDX` | 34 | 角速度起始 |
| `JOINT_VEL_START_IDX` | 37 | 关节角速度起始 |
| `TAR_TOE_VEL_LOCAL_START_IDX` | 49 | 足端速度起始 |
| `JOINT_VEL_END_IDX` | 49 | AMP 有效数据上界（不含 foot_vel） |

---

## AMP 训练实际使用的 43 维观测

Discriminator 与 `get_amp_observations()` 对齐，**不使用完整 61 维**：

| 顺序 | 内容 | 维数 | 文件索引（重排后） |
|------|------|------|-------------------|
| 1 | 关节角 `joint_pos` | 12 | `[7:19]` |
| 2 | 足端位置 `foot_pos_local` | 12 | `[19:31]` |
| 3 | 线速度 | 3 | `[31:34]` |
| 4 | 角速度 | 3 | `[34:37]` |
| 5 | 关节角速度 `joint_vel` | 12 | `[37:49]` |
| 6 | 根高度 `root_z` | 1 | `[2]` |
| **合计** | | **43** | |

未参与 AMP 观测：`root_x/y`、四元数、`tar_toe_vel_local[49:61]`。

根状态初始化（`reference_state_initialization`）会使用完整帧的前 49 维（含 root_pos、root_rot）。

---

## A1：文件腿序 ↔ URDF 映射

### 配置

- 数据路径：`datasets/a1/mocap_motions/*`
- 任务：`a1_amp`
- 重排模式：`amp_joint_reorder = 'a1'`（默认）

### 文件中的原始腿顺序（PyBullet 导出）

每腿 3 关节，4 腿块顺序为：

```
文件 block index:   0      1      2      3
                    FR     FL     RR     RL
每腿关节顺序:      hip → thigh → calf
在 joint_pos 内:   [0:3]  [3:6]  [6:9]  [9:12]
```

### Isaac Gym / `a1.urdf` DOF 顺序（重排后）

```
index  URDF 关节名
[ 0]   FL_hip_joint
[ 1]   FL_thigh_joint
[ 2]   FL_calf_joint
[ 3]   FR_hip_joint
[ 4]   FR_thigh_joint
[ 5]   FR_calf_joint
[ 6]   RL_hip_joint
[ 7]   RL_thigh_joint
[ 8]   RL_calf_joint
[ 9]   RR_hip_joint
[10]   RR_thigh_joint
[11]   RR_calf_joint
```

规律：**按腿排列**，腿顺序 **FL → FR → RL → RR**，每腿 **hip → thigh → calf**。

### 重排规则

`AMPLoader.reorder_from_pybullet_to_isaac()` 对 `joint_pos`、`foot_pos`、`joint_vel`、`foot_vel` 四组 12 维数据做相同的腿块置换：

```
permute = [1, 0, 3, 2]
即：新顺序 = 原 [FL, FR, RL, RR]  ←  原文件 [FR, FL, RR, RL]
```

| 重排后 block | URDF 腿 | 原文件 block |
|-------------|---------|-------------|
| 0 (index 0–2) | FL | 1 (FL) |
| 1 (index 3–5) | FR | 0 (FR) |
| 2 (index 6–8) | RL | 3 (RL) |
| 3 (index 9–11) | RR | 2 (RR) |

### A1 关节符号与 URDF 轴（以 FL 为例）

| 关节 | 旋转轴 | URDF limit (rad) | 典型站立参考 |
|------|--------|------------------|-------------|
| `FL_hip_joint` | X | [-0.80, +0.80] | ≈ 0 |
| `FL_thigh_joint` | Y | [-1.05, +4.19] | ≈ +0.9 |
| `FL_calf_joint` | Y | [-2.70, -0.92] | ≈ -1.8（**始终为负**） |

`calf` 上限为负是 URDF 约定，不是数据错误。

### A1 示例（`trot0.txt` 第 0 帧，原始 → 重排）

**原始 joint_pos [7:19]，PyBullet 顺序 [FR, FL, RR, RL]：**

```
FR: hip=+0.205  thigh=+0.359  calf=-1.630
FL: hip=-0.107  thigh=+1.190  calf=-1.748
RR: hip=+0.124  thigh=+1.149  calf=-1.405
RL: hip=+0.144  thigh=+0.154  calf=-1.206
```

**重排后，与 Isaac Gym DOF 一致 [FL, FR, RL, RR]：**

```
FL_hip=-0.107  FL_thigh=+1.190  FL_calf=-1.748
FR_hip=+0.205  FR_thigh=+0.359  FR_calf=-1.630
RL_hip=+0.144  RL_thigh=+0.154  RL_calf=-1.206
RR_hip=+0.124  RR_thigh=+1.149  RR_calf=-1.405
```

---

## Chitu：文件腿序 ↔ URDF 映射

### 配置

- 数据路径：`datasets/chitu/mocap_motions/*`
- 任务：`chitu_amp`
- 重排模式：`amp_joint_reorder = 'chitu'`

### 文件中的原始腿顺序（与 A1 mocap 相同，PyBullet 约定）

```
文件 block index:   0      1      2      3
                    FR     FL     RR     RL
每腿关节顺序:      HAA → HFE → KFE
```

与 A1 的 `[FR, FL, RR, RL]` 布局一致；**不要**对 Chitu 使用 `amp_joint_reorder='a1'`（那会映到 A1 URDF 顺序 `[FL,FR,RL,RR]`）。

### Isaac Gym / `chitu.urdf` DOF 顺序（重排后）

```
index  URDF 关节名
[ 0]   LF_HAA    [ 1]   LF_HFE    [ 2]   LF_KFE
[ 3]   LH_HAA    [ 4]   LH_HFE    [ 5]   LH_KFE
[ 6]   RF_HAA    [ 7]   RF_HFE    [ 8]   RF_KFE
[ 9]   RH_HAA    [10]   RH_HFE    [11]   RH_KFE
```

规律：腿顺序 **LF → LH → RF → RH**，每腿 **HAA → HFE → KFE**。

### 重排规则

`AMPLoader.reorder_from_amp_mocap_to_chitu()`（`amp_joint_reorder='chitu'`）：

```
permute = [1, 3, 0, 2]
即：Chitu URDF [LF, LH, RF, RH]  ←  文件 [FR, FL, RR, RL]
```

| 重排后 block | URDF 腿 | 原文件 block | A1 命名 |
|-------------|---------|-------------|---------|
| 0 | LF | 1 | FL |
| 1 | LH | 3 | RL |
| 2 | RF | 0 | FR |
| 3 | RH | 2 | RR |

若导出端已按 Chitu URDF 顺序写入文件，设 `amp_joint_reorder='none'`。

---

## 数据流概览

```
┌─────────────────┐     permute 腿块      ┌──────────────────────┐
│  JSON 文件       │  ──────────────────►  │  Isaac Gym URDF 顺序  │
│  61-dim / frame  │   (a1 或 chitu 模式)   │  dof_names 一致       │
└─────────────────┘                        └──────────────────────┘
         │                                           │
         │  取 [7:49] + root_z                       │  get_amp_observations()
         ▼                                           ▼
┌─────────────────────────────────────────────────────────────┐
│  AMP 43-dim：joint_pos + foot_pos + lin/ang_vel + joint_vel + z │
└─────────────────────────────────────────────────────────────┘
```

---

## 校验方法

### 1. 播放 mocap：`play_motion.py`（推荐）

在仓库根目录、已激活 `amp_hw` 等含 Isaac Gym 的环境中运行。**不要加 `--headless`**，需要窗口目视检查。

#### 基本用法

```bash
# Chitu（默认 --task=chitu_amp）
python legged_gym/scripts/play_motion.py \
  --task=chitu_amp \
  --motion_file=datasets/chitu/mocap_motions/trot_fwd_slow.txt \
  --loop

# A1
python legged_gym/scripts/play_motion.py \
  --task=a1_amp \
  --motion_file=datasets/a1/mocap_motions/trot0.txt \
  --loop
```

#### 常用命令

```bash
# 列出 config 加载的全部轨迹（不播放）
python legged_gym/scripts/play_motion.py --task=chitu_amp --list

# 按索引播放（索引见 --list 或启动时打印）
python legged_gym/scripts/play_motion.py --task=chitu_amp --traj_idx=2 --loop

# 2 倍速播放
python legged_gym/scripts/play_motion.py \
  --task=chitu_amp \
  --motion_file=datasets/chitu/mocap_motions/trot_fwd_slow.txt \
  --speed=2.0

# 加载 config 中全部 mocap 文件，播放第 0 条
python legged_gym/scripts/play_motion.py --task=chitu_amp --traj_idx=0
```

#### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--task` | `chitu_amp` | AMP 任务名，决定 URDF 与 `amp_joint_reorder`（`a1_amp` / `chitu_amp`） |
| `--motion_file` | 无 | 单个 mocap JSON（`.txt`）；不指定则加载 config 里 `glob` 的全部文件 |
| `--traj_idx` | `0` | 轨迹索引，见启动时打印的 `[0] ... [1] ...` 列表 |
| `--loop` | 关 | 播完后循环 |
| `--speed` | `1.0` | 播放倍速（相对 JSON 内 `FrameDuration`） |
| `--list` | 关 | 只打印轨迹信息并退出 |

另支持 Isaac Gym 通用参数，例如 `--sim_device=cpu`、`--rl_device=cpu`。

#### 脚本行为

- 单环境、平地、关闭域随机与重力；**开环运动学回放**（逐帧写入 root/关节状态，不跑 RL 策略、不用 PD 跟踪）。
- 将轨迹起点平移到原点附近，相机跟随机器人。
- 关闭窗口或按 **Q** 退出。

#### 启动时应确认

- `Isaac Gym DOF order` — 与 URDF 关节顺序一致（Chitu: `LF_HAA … RH_KFE`）。
- `AMP joint reorder` — Chitu 为 `chitu`，A1 为 `a1`。

#### 目视检查

- 四足步态连贯，无单帧跳变。
- 无「串腿」、穿地、严重关节限位 violation。
- 前进/转弯 clip 有明显位移。

与 `play.py` 的区别：`play.py` 加载**已训练策略**做闭环控制；`play_motion.py` 只回放**专家 mocap**，用于检查数据集质量。

#### 播放卡顿（Chitu 比 A1 更明显）

常见原因：

| 因素 | A1 | Chitu |
|------|-----|-------|
| `FrameDuration` | 0.021 s（约 48 Hz） | 0.01 s（100 Hz） |
| 每帧渲染预算 | ~21 ms | ~10 ms |
| 模型 | 轻 | URDF/碰撞更重 |

100 Hz 时若 `simulate + render` 超过 10 ms，再叠加 `sleep(0.01)` 和 `sync_frame_time`，节奏会不均匀，看起来像卡顿（数据本身通常比 A1 更平滑）。

**处理办法：**

```bash
# 略放慢播放，给 GPU 留时间
python legged_gym/scripts/play_motion.py --task=chitu_amp \
  --motion_file=datasets/chitu/mocap_motions/trot_fwd_slow.txt --speed=0.5 --loop

# 或改用 CPU 仿真（有时更稳）
python legged_gym/scripts/play_motion.py --task=chitu_amp \
  --motion_file=datasets/chitu/mocap_motions/trot_fwd_slow.txt \
  --sim_device=cpu --rl_device=cpu --loop
```

当前脚本已关闭 `render(sync_frame_time)`、关闭回放自碰撞，并用自适应 `sleep` 减轻卡顿。

### 2. 检查清单

- [ ] 每帧 **61** 维，JSON 可解析
- [ ] `FrameDuration` 与采集频率一致
- [ ] 关节/足端使用 **PyBullet 文件腿顺序** `[FR, FL, RR, RL]`（与 A1 相同）；Chitu 配置 `amp_joint_reorder='chitu'`
- [ ] 每腿内部关节顺序与 URDF 语义一致（A1: hip/thigh/calf；Chitu: HAA/HFE/KFE）
- [ ] 四元数为 **(x, y, z, w)**，loader 会自动归一化
- [ ] 配置中 `amp_joint_reorder` 与数据来源匹配

---

## 新增轨迹文件

1. 将 JSON 文件放入对应目录（`datasets/a1/mocap_motions/` 或 `datasets/chitu/mocap_motions/`）。
2. 确认 `MotionWeight` 设置合理（多条轨迹时控制采样比例）。
3. 用 `play_motion.py` 目视确认。
4. 无需改代码：配置已通过 `glob.glob('datasets/.../mocap_motions/*')` 自动加载。

---

## 参考

- NVIDIA Isaac Gym AMP 原始格式（PyBullet 四足，61-dim frame）
- A1 URDF：`resources/robots/a1/urdf/a1.urdf`
- Chitu URDF：`resources/robots/chitu/urdf/chitu.urdf`
