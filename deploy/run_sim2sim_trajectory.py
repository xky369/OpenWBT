#!/usr/bin/env python3
"""
Sim2sim trajectory tracking benchmark for OpenWBT (MuJoCo + squat policy + Pinocchio arm IK).

Reference trajectory: CSV with columns matching ``TrajectorySampler`` in
``rl_ik_solver/.../trajectory.py`` (torso-frame wrist targets as x,y,z,roll,pitch,yaw).

Metrics: same ``TrackingErrorMonitor`` math as ``trajectory.py``. By default
(``--metrics_actual_source rl_fk``) **actual** pose is ``larm_forward`` /
``rarm_forward`` on arm joints, with the same ``joint2motor_idx`` reordering and
``LEFT_ARM_JOINT_IDX`` / ``RIGHT_ARM_JOINT_IDX`` as that file (see
``deploy/evaluation/g1_rl_trajectory_fk.py``). Optional ``mujoco`` uses
``torso_link``-expressed ``*_wrist_yaw_link`` bodies instead.

CSV targets remain 6D torso-frame wrist (same as RL command). IK still maps
wrist targets to Pinocchio ``L_ee`` / ``R_ee`` before ``solve_ik``.

The summary JSON and sibling ``*.npz`` trace are written in the same schema as
``rl_ik_solver/.../trajectory.py`` (``tracking_error`` / ``smoothness`` /
``compute`` dicts, 14-D ``joint_cmd`` in ``LEFT_ARM_POLICY_IDX`` ||
``RIGHT_ARM_POLICY_IDX`` order) so R2S2's ``plot_tracking_trace.py`` and any
RL-vs-WBT side-by-side scripts can consume them unchanged. ``runtime_type`` is
``"openwbt_sim2sim"`` and ``compute.time_ms_*`` reports per-control-step IK
solve time (the R2S2 equivalent of policy inference time).

Run from repository root::

    python deploy/run_sim2sim_trajectory.py \
        --trajectory_csv traj1.csv \
        --viewer --realtime --wait_for_keypress \
        --settle_sec 3.0 --settle_hold_zero_sec 1.0 --cmd_ramp_sec 1.0 \
        --stabilize_qvel_thresh 0.3 --stabilize_max_sec 5.0 \
        --ramp_in_sec 2.0 \
        --output_json deploy/evaluation/results/wbt_traj1.json

Requires ``ckpts/squat.onnx`` (see ``deploy/configs/g1_squat.yaml``). Do not import
``deploy.controllers.controller`` here (USB / DDS side effects).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

os.chdir(_REPO_ROOT)

from deploy.config import Config
from deploy.evaluation.g1_rl_trajectory_fk import (
    LEFT_ARM_POLICY_IDX,
    RIGHT_ARM_POLICY_IDX,
    fk_actual_pose7_pair_from_mujoco_qj,
    motor_q_from_qj_order,
    policy_joint_vector_from_motor_q,
)
from deploy.evaluation.tracking_metrics import (
    CommandSmoothnessMonitor,
    TrackingErrorMonitor,
    TrajectorySampler,
    euler_xyz_to_T,
    matrix_to_rpy,
    quat_wxyz_slerp,
    quaternion_wxyz_to_matrix,
    rpy_to_quaternion_wxyz,
    t_trans,
    T_to_pose7_xyz_quat_wxyz,
)
from deploy.helpers.policy_unified import SquatLowLevelPolicy
from deploy.helpers.rotation_helper import get_gravity_orientation
from deploy.teleop.robot_control.robot_arm_ik import G1_29_ArmIK

# Pinocchio ``L_ee`` / ``R_ee`` frames attach to ``left_wrist_yaw_joint`` with this
# translation from the joint (same as ``G1_29_ArmIK``). CSV / metrics use
# ``wrist_yaw_link``; multiply before IK so the solver tracks the intended wrist pose.
_T_WRIST_YAW_TO_IK_EE = t_trans(np.array([0.05, 0.0, 0.0], dtype=np.float64))

# ``G1_29_ArmIK`` builds a reduced Pinocchio model rooted at ``pelvis`` with the
# three waist joints locked at 0, so the solver lives in a **pelvis frame** with
# pelvis at the origin (no floating base). CSV targets and ``larm_forward`` /
# ``rarm_forward`` are defined in **torso_link frame**. With waist = 0 the two
# frames only differ by the fixed chain pelvis -> waist_yaw -> waist_roll ->
# torso_link translation from the URDF used by the IK (see
# ``deploy/assets/g1/g1_body29_hand14.urdf``): waist_yaw_joint xyz=(0,0,0),
# waist_roll_joint xyz=(-0.0039635, 0, 0.044), waist_pitch_joint xyz=(0,0,0).
# Feed the solver ``T_pelvis_target = T_PELVIS_TORSO @ T_torso_target`` otherwise
# the target lands outside the arm workspace and the FK metric explodes.
_T_PELVIS_TORSO = t_trans(np.array([-0.0039635, 0.0, 0.044], dtype=np.float64))


def mj_body_T_world(m: mujoco.MjModel, d: mujoco.MjData, body_name: str) -> np.ndarray:
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        raise ValueError(f"Body not found in MJCF: {body_name}")
    R = d.xmat[bid].reshape(3, 3).copy()
    p = d.xpos[bid].copy()
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def torso_frame_wrist_poses(m: mujoco.MjModel, d: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
    T_w_t = mj_body_T_world(m, d, "torso_link")
    T_w_inv = np.linalg.inv(T_w_t)
    T_l = T_w_inv @ mj_body_T_world(m, d, "left_wrist_yaw_link")
    T_r = T_w_inv @ mj_body_T_world(m, d, "right_wrist_yaw_link")
    return T_to_pose7_xyz_quat_wxyz(T_l), T_to_pose7_xyz_quat_wxyz(T_r)


def actual_arm_pose7_for_metrics(
    m: mujoco.MjModel,
    d: mujoco.MjData,
    dof_idx: np.ndarray,
    source: str,
) -> tuple[np.ndarray, np.ndarray]:
    if source == "rl_fk":
        qj = d.qpos[7:][dof_idx].astype(np.float64)
        return fk_actual_pose7_pair_from_mujoco_qj(qj, dof_idx)
    if source == "mujoco":
        l7, r7 = torso_frame_wrist_poses(m, d)
        return l7.astype(np.float64), r7.astype(np.float64)
    raise ValueError(f"Unknown --metrics_actual_source {source!r} (use rl_fk or mujoco).")


def pd_torques(
    kps: np.ndarray,
    kds: np.ndarray,
    dof_idx: np.ndarray,
    target_q: np.ndarray,
    m: mujoco.MjModel,
    d: mujoco.MjData,
) -> np.ndarray:
    q = d.qpos[7:][dof_idx].astype(np.float64)
    dq = d.qvel[6:][dof_idx].astype(np.float64)
    return (target_q - q) * kps + (0.0 - dq) * kds


def resolve_policy_path(squat_cfg: Config) -> str:
    p = Path(squat_cfg.policy_path)
    if not p.is_absolute():
        p = (_REPO_ROOT / p).resolve()
    return str(p)


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenWBT sim2sim trajectory tracking (MuJoCo + IK + squat policy).")
    parser.add_argument(
        "--runner_config",
        type=str,
        default="deploy/configs/run_teleoperation.yaml",
        help="YAML with xml_path, dof_idx, action_hl_idx, default_angles, control_dt, ...",
    )
    parser.add_argument("--trajectory_csv", type=str, required=True, help="Trajectory CSV (torso-frame wrists).")
    parser.add_argument("--output_json", type=str, default="", help="Where to write result summary JSON.")
    parser.add_argument(
        "--output_npz",
        type=str,
        default="",
        help=(
            "Where to write the trace NPZ (t / left_target / right_target / left_actual / "
            "right_actual / joint_cmd / joint_cmd_t, same schema as the R2S2 RL trajectory "
            "runner). Defaults to --output_json with suffix replaced by .npz."
        ),
    )
    parser.add_argument(
        "--no_save_trace",
        action="store_true",
        help="Skip writing the trace NPZ (summary JSON is still written).",
    )
    parser.add_argument("--error_log_csv", type=str, default="", help="Optional per-step tracking error CSV.")
    parser.add_argument("--loop_trajectory", action="store_true", help="Loop CSV in time; default plays once.")
    parser.add_argument("--max_duration_sec", type=float, default=600.0, help="Safety cap when --loop_trajectory is set.")
    parser.add_argument(
        "--settle_sec",
        type=float,
        default=3.0,
        help=(
            "Total squat-only pre-roll (wall clock) before arm ramp-in / tracking. "
            "Split into: --settle_hold_zero_sec at cmd=0, then --cmd_ramp_sec "
            "ramping to --squat_cmd / cmd_debug, then the remainder holding "
            "cmd_target. Arms are PD-held at their initial qpos throughout."
        ),
    )
    parser.add_argument("--ramp_in_sec", type=float, default=2.0, help="Blend measured torso wrist pose to traj[0] before metrics clock.")
    parser.add_argument("--error_warmup_sec", type=float, default=0.0, help="Skip tracking stats for this many seconds after ramp-in.")
    parser.add_argument("--error_print_hz", type=float, default=2.0, help="<=0 disables periodic tracking prints.")
    parser.add_argument("--error_lag_search_sec", type=float, default=1.0, help="Lag search window for lag-compensated RMSE.")
    parser.add_argument("--error_spatial_resample_count", type=int, default=200, help="Spatial resampling count.")
    parser.add_argument("--smoothness_print_hz", type=float, default=0.0, help="<=0 disables smoothness prints.")
    parser.add_argument("--disable_smoothness", action="store_true")
    parser.add_argument(
        "--squat_cmd",
        type=float,
        nargs=2,
        default=None,
        metavar=("H", "P"),
        help=(
            "Squat policy command [height_delta, pitch_delta]. Defaults to [0, 0] "
            "(nominal standing pose controlled by the squat policy). For a shallow "
            "squat try e.g. '-0.2 0.0'. Values must stay within the policy's "
            "training range (see max_cmd in the squat YAML); the default cmd_debug "
            "is usually a deep debug pose outside that range and breaks tracking."
        ),
    )
    parser.add_argument(
        "--use_cmd_debug",
        action="store_true",
        help=(
            "Force --squat_cmd = squat YAML cmd_debug (legacy deep-squat default). "
            "Off by default because cmd_debug is often outside max_cmd training "
            "range and causes base instability + tracking failure."
        ),
    )
    parser.add_argument(
        "--metrics_actual_source",
        type=str,
        choices=("rl_fk", "mujoco"),
        default="rl_fk",
        help=(
            "How to build 7D actual pose for TrackingErrorMonitor: "
            "'rl_fk' matches trajectory.py (larm_forward/rarm_forward + g1_upper_ik joint2motor_idx); "
            "'mujoco' uses torso_link <- wrist_yaw_link bodies."
        ),
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help=(
            "Open mujoco.viewer.launch_passive during the run. Requires a display "
            "(local / X-forwarding / VNC); will fail on a pure headless server."
        ),
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help=(
            "When --viewer is set, pace each control step to wall clock so the "
            "viewer shows the sim at (approximately) real time."
        ),
    )
    parser.add_argument(
        "--viewer_sync_every_n_substeps",
        type=int,
        default=1,
        help=(
            "Call viewer.sync() every N physics substeps (1 = every substep, smoothest; "
            "larger values reduce GUI overhead)."
        ),
    )
    parser.add_argument(
        "--wait_for_keypress",
        action="store_true",
        help=(
            "After the robot is placed at the initial pose, block on stdin until the "
            "user presses Enter before advancing simulation. Useful when combined "
            "with --viewer so the viewer opens, settles on the default pose, and you "
            "can aim the camera before the squat policy engages."
        ),
    )
    parser.add_argument(
        "--settle_hold_zero_sec",
        type=float,
        default=1.0,
        help=(
            "At the start of settle (right after --wait_for_keypress), step physics "
            "with the squat policy at cmd=[0, 0] for this long BEFORE ramping to "
            "--squat_cmd / cmd_debug. Gives the robot time to land on its feet and "
            "have the policy regulate legs at neutral stance. Counts against "
            "--settle_sec (i.e. settle = hold_zero + cmd_ramp + hold_target)."
        ),
    )
    parser.add_argument(
        "--cmd_ramp_sec",
        type=float,
        default=1.0,
        help=(
            "After --settle_hold_zero_sec, ramp the squat command from [0, 0] up to "
            "--squat_cmd / cmd_debug over this many seconds. Also counts against "
            "--settle_sec. Set 0 to jump straight to the target command."
        ),
    )
    parser.add_argument(
        "--stabilize_qvel_thresh",
        type=float,
        default=0.0,
        help=(
            "If > 0, after settle finishes, keep stepping physics with "
            "cmd=cmd_target and arms held until max |qvel| on --dof_idx falls below "
            "this threshold (rad/s), or --stabilize_max_sec elapses, whichever "
            "first. 0 disables this wait. Typical value 0.2-0.5 rad/s."
        ),
    )
    parser.add_argument(
        "--stabilize_max_sec",
        type=float,
        default=5.0,
        help=(
            "Hard cap on the --stabilize_qvel_thresh wait after settle. Ignored "
            "when --stabilize_qvel_thresh <= 0."
        ),
    )
    parser.add_argument(
        "--policy_warmup_iters",
        type=int,
        default=20,
        help=(
            "Before any physics step, run the squat policy this many times at "
            "cmd=[0,0] with the initial joint state so its recurrent hidden state "
            "leaves the zero-initialised transient. Not stepping physics during "
            "warm-up keeps the viewer visually still. Set 0 to disable."
        ),
    )
    parser.add_argument(
        "--disable_squat_policy",
        action="store_true",
        help=(
            "Skip the squat policy entirely: legs and waist are PD-held at "
            "run_cfg.default_angles for the whole run. Robot just stands. "
            "Overrides --squat_cmd / --cmd_ramp_sec / --policy_warmup_iters."
        ),
    )
    args = parser.parse_args()

    run_cfg = Config(str(Path(args.runner_config).resolve()))
    squat_cfg_path = _REPO_ROOT / "deploy" / "configs" / getattr(run_cfg, "squat_config", "g1_squat.yaml")
    squat_cfg = Config(str(squat_cfg_path.resolve()))
    squat_cfg.policy_path = resolve_policy_path(squat_cfg)

    traj_path = Path(args.trajectory_csv).expanduser().resolve()
    trajectory = TrajectorySampler(traj_path)

    xml_path = (_REPO_ROOT / run_cfg.xml_path).resolve() if not Path(run_cfg.xml_path).is_absolute() else Path(run_cfg.xml_path)
    if not xml_path.is_file():
        raise FileNotFoundError(f"MJCF not found: {xml_path}")

    m = mujoco.MjModel.from_xml_path(str(xml_path))
    d = mujoco.MjData(m)
    m.opt.timestep = float(run_cfg.simulation_dt)

    dof_idx = np.array(run_cfg.dof_idx, dtype=np.int64)
    action_hl_idx = np.array(run_cfg.action_hl_idx, dtype=np.int64)
    n_dof = int(run_cfg.num_dof)

    arm_ik = G1_29_ArmIK(Unit_Test=False, Visualization=False)
    squat_policy = SquatLowLevelPolicy(squat_cfg)

    mujoco.mj_resetData(m, d)
    d.qpos[0:3] = [0.0, 0.0, 0.793]
    d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    for i in range(n_dof):
        d.qpos[7 + int(dof_idx[i])] = float(run_cfg.default_angles[i])
    mujoco.mj_forward(m, d)

    target_dof_pos = np.array(run_cfg.default_angles, dtype=np.float64).copy()
    kps = np.array(squat_cfg.kps, dtype=np.float64)
    kds = np.array(squat_cfg.kds, dtype=np.float64)

    decimation = int(run_cfg.control_decimation)
    sim_dt = float(run_cfg.simulation_dt)
    control_dt = float(run_cfg.control_dt)

    # Resolve the squat policy command. Default (no args) is [0, 0] = nominal
    # standing, which keeps the base stable and leaves the arm full workspace
    # for tracking. cmd_debug from the YAML is often a debug deep-squat value
    # outside the policy's max_cmd training range and destabilises tracking, so
    # it's opt-in via --use_cmd_debug.
    if args.squat_cmd is not None:
        cmd_target = np.array(args.squat_cmd, dtype=np.float32)
        cmd_source = "--squat_cmd"
    elif args.use_cmd_debug:
        cmd_target = np.array(squat_cfg.cmd_debug, dtype=np.float32).copy()
        cmd_source = "squat_cfg.cmd_debug (--use_cmd_debug)"
    else:
        cmd_target = np.zeros(2, dtype=np.float32)
        cmd_source = "default [0, 0] (nominal standing)"
    cmd_now = np.zeros_like(cmd_target)

    max_cmd = np.asarray(getattr(squat_cfg, "max_cmd", [np.inf, np.inf]), dtype=np.float32)
    if not args.disable_squat_policy:
        print(f"[sim2sim] squat cmd = {cmd_target.tolist()} (from {cmd_source})")
        if np.any(np.abs(cmd_target) > max_cmd + 1e-6):
            print(
                f"[sim2sim] WARNING: |cmd_target|={np.abs(cmd_target).tolist()} exceeds "
                f"max_cmd={max_cmd.tolist()} from squat YAML; policy is being driven "
                "out of its training distribution and may destabilise the base + break tracking."
            )
    else:
        print("[sim2sim] squat policy disabled; legs + waist PD-held at run_cfg.default_angles")

    viewer = mujoco.viewer.launch_passive(m, d) if args.viewer else None
    sync_every = max(1, int(args.viewer_sync_every_n_substeps))
    realtime_pacing = bool(args.viewer and args.realtime)

    counter = 0
    # Per-control-step IK solve time, collected only while ``record_ik_time`` is
    # True (main trajectory tracking phase, after ramp-in and
    # ``error_warmup_sec``). Same role as ``infer_times_ms`` in the R2S2
    # trajectory runner, so ``compute.time_ms_*`` is apples-to-apples.
    ik_times_ms: list[float] = []
    record_ik_time = False

    def physics_substep(
        hold_arms: bool,
        left_euler6: np.ndarray,
        right_euler6: np.ndarray,
    ) -> None:
        nonlocal target_dof_pos, counter
        if counter % decimation == 0:
            if args.disable_squat_policy:
                # Keep legs & waist at run_cfg.default_angles; only the arms are
                # touched below (hold or IK). target_dof_pos already starts at
                # run_cfg.default_angles, so no-op for the leg / waist slice.
                pass
            else:
                qj = d.qpos[7:][dof_idx].astype(np.float32)
                dqj = d.qvel[6:][dof_idx].astype(np.float32)
                quat = d.qpos[3:7].astype(np.float32)
                omega = d.qvel[3:6].astype(np.float32)
                g = get_gravity_orientation(quat).astype(np.float32)
                td = target_dof_pos.astype(np.float32).copy()
                _, _, leg_slice = squat_policy.inference(cmd_now, g, omega, qj, dqj)
                target_dof_pos[:] = td.astype(np.float64)
                target_dof_pos[squat_cfg.action_idx] = leg_slice.astype(np.float64)

            if hold_arms:
                target_dof_pos[action_hl_idx] = d.qpos[7:][action_hl_idx].astype(np.float64)
            else:
                # CSV target is torso_link-frame wrist_yaw_link pose. Pinocchio IK
                # expects pelvis-frame L_ee / R_ee targets (L_ee = wrist_yaw_joint
                # + 0.05 m along x). Compose: pelvis <- torso <- wrist_yaw <- L_ee.
                T_torso_wrist_l = euler_xyz_to_T(left_euler6[:3], left_euler6[3:6])
                T_torso_wrist_r = euler_xyz_to_T(right_euler6[:3], right_euler6[3:6])
                T_pelvis_ik_l = _T_PELVIS_TORSO @ T_torso_wrist_l @ _T_WRIST_YAW_TO_IK_EE
                T_pelvis_ik_r = _T_PELVIS_TORSO @ T_torso_wrist_r @ _T_WRIST_YAW_TO_IK_EE

                q_arm = d.qpos[7:][action_hl_idx].astype(np.float64)
                dq_arm = d.qvel[6:][action_hl_idx].astype(np.float64)
                _t_ik_start = time.perf_counter()
                sol_q, _ = arm_ik.solve_ik(T_pelvis_ik_l, T_pelvis_ik_r, q_arm, dq_arm)
                _ik_ms = (time.perf_counter() - _t_ik_start) * 1000.0
                target_dof_pos[action_hl_idx] = sol_q
                if record_ik_time:
                    ik_times_ms.append(float(_ik_ms))

        tau = pd_torques(kps, kds, dof_idx, target_dof_pos, m, d)
        d.ctrl[:] = 0.0
        d.ctrl[dof_idx] = tau
        mujoco.mj_step(m, d)
        counter += 1
        if viewer is not None and (counter % sync_every == 0):
            if not viewer.is_running():
                raise KeyboardInterrupt("viewer closed by user")
            viewer.sync()

    def pace_realtime(wall_deadline: float) -> None:
        if not realtime_pacing:
            return
        remaining = wall_deadline - time.perf_counter()
        if remaining > 0.0:
            time.sleep(remaining)

    # Warm up the squat policy's recurrent hidden state with cmd=[0,0] and the
    # stationary initial joint state, BEFORE stepping physics. Without this the
    # first ~10 policy outputs are garbage (all-zero hidden state) and the legs
    # jerk violently the moment we start integrating. Skipped if we're running
    # without the squat policy at all.
    if not args.disable_squat_policy:
        for _ in range(max(0, int(args.policy_warmup_iters))):
            qj0 = d.qpos[7:][dof_idx].astype(np.float32)
            dqj0 = d.qvel[6:][dof_idx].astype(np.float32)
            quat0 = d.qpos[3:7].astype(np.float32)
            omega0 = d.qvel[3:6].astype(np.float32)
            g0 = get_gravity_orientation(quat0).astype(np.float32)
            squat_policy.inference(np.zeros_like(cmd_target), g0, omega0, qj0, dqj0)

    if args.wait_for_keypress:
        if viewer is not None:
            viewer.sync()
        try:
            input(
                "[sim2sim] Press Enter to start the squat policy + settle "
                "(tracking will NOT begin until the robot is stable) ... "
            )
        except EOFError:
            pass

    wall_anchor = time.perf_counter()
    wall_phase = 0.0

    def advance_one_control_step(
        hold_arms: bool,
        le: np.ndarray,
        re: np.ndarray,
    ) -> None:
        nonlocal wall_phase
        for _ in range(decimation):
            physics_substep(hold_arms=hold_arms, left_euler6=le, right_euler6=re)
        wall_phase += control_dt
        pace_realtime(wall_anchor + wall_phase)

    settle_control_steps = max(0, int(round(args.settle_sec / control_dt)))
    # Split settle_sec into three back-to-back sub-phases, all with physics +
    # squat policy active and arms PD-held:
    #   [0, hold_zero_steps)        -> cmd_now = 0        (robot lands + locks
    #                                                      legs at neutral)
    #   [hold_zero_steps, +ramp)    -> cmd_now = a*target (smooth ramp)
    #   [+ramp, settle_end)         -> cmd_now = target   (hold target pose)
    # Irrelevant when the squat policy is disabled (cmd_now stays 0 throughout
    # and legs are PD-held at default_angles).
    if args.disable_squat_policy:
        hold_zero_steps = 0
        cmd_ramp_steps = 0
    else:
        hold_zero_steps = min(
            settle_control_steps,
            max(0, int(round(max(0.0, float(args.settle_hold_zero_sec)) / control_dt))),
        )
        cmd_ramp_steps = min(
            settle_control_steps - hold_zero_steps,
            max(0, int(round(max(0.0, float(args.cmd_ramp_sec)) / control_dt))),
        )
    le_hold = np.zeros(6, dtype=np.float64)
    re_hold = np.zeros(6, dtype=np.float64)
    for k in range(settle_control_steps):
        if args.disable_squat_policy:
            cmd_now[:] = 0.0
        elif k < hold_zero_steps:
            cmd_now[:] = 0.0
        elif cmd_ramp_steps > 0 and k < hold_zero_steps + cmd_ramp_steps:
            alpha = float(k - hold_zero_steps + 1) / float(cmd_ramp_steps)
            cmd_now[:] = alpha * cmd_target
        else:
            cmd_now[:] = cmd_target
        advance_one_control_step(hold_arms=True, le=le_hold, re=re_hold)
    if not args.disable_squat_policy:
        cmd_now[:] = cmd_target

    # Wait for the robot to actually stop moving before touching arms or
    # starting metric collection. Without this, ramp-in starts while the legs
    # are still oscillating and the IK/tracking loop inherits that transient.
    if float(args.stabilize_qvel_thresh) > 0.0:
        stab_max_steps = max(0, int(round(max(0.0, float(args.stabilize_max_sec)) / control_dt)))
        stab_v = float("inf")
        stab_done = False
        for stab_k in range(stab_max_steps):
            advance_one_control_step(hold_arms=True, le=le_hold, re=re_hold)
            stab_v = float(np.max(np.abs(d.qvel[6:][dof_idx])))
            if stab_v < float(args.stabilize_qvel_thresh):
                print(
                    f"[sim2sim] stabilized after extra {(stab_k + 1) * control_dt:.2f}s "
                    f"(max|qvel|={stab_v:.3f} < {args.stabilize_qvel_thresh:.3f})"
                )
                stab_done = True
                break
        if not stab_done:
            print(
                f"[sim2sim] stabilize wait hit --stabilize_max_sec={args.stabilize_max_sec:.2f}s "
                f"(max|qvel|={stab_v:.3f} >= {args.stabilize_qvel_thresh:.3f}); "
                "proceeding to ramp-in anyway"
            )

    left_pose7_0, right_pose7_0 = actual_arm_pose7_for_metrics(m, d, dof_idx, args.metrics_actual_source)
    left_start_pos = left_pose7_0[:3].copy()
    right_start_pos = right_pose7_0[:3].copy()
    left_start_q = left_pose7_0[3:7].copy()
    right_start_q = right_pose7_0[3:7].copy()

    left_g = trajectory.left_traj[0].astype(np.float64)
    right_g = trajectory.right_traj[0].astype(np.float64)
    left_tgt_q = rpy_to_quaternion_wxyz(left_g[3:6])
    right_tgt_q = rpy_to_quaternion_wxyz(right_g[3:6])
    if float(np.dot(left_start_q, left_tgt_q)) < 0.0:
        left_tgt_q = -left_tgt_q
    if float(np.dot(right_start_q, right_tgt_q)) < 0.0:
        right_tgt_q = -right_tgt_q

    ramp_sec = max(0.0, float(args.ramp_in_sec))
    ramp_control_steps = max(0, int(round(ramp_sec / control_dt))) if ramp_sec > 0 else 0

    for k in range(ramp_control_steps):
        a = float(k + 1) / float(max(ramp_control_steps, 1))
        lp = (1.0 - a) * left_start_pos + a * left_g[:3]
        rp = (1.0 - a) * right_start_pos + a * right_g[:3]
        lq = quat_wxyz_slerp(left_start_q, left_tgt_q, a)
        rq = quat_wxyz_slerp(right_start_q, right_tgt_q, a)
        rpy_l = matrix_to_rpy(quaternion_wxyz_to_matrix(lq))
        rpy_r = matrix_to_rpy(quaternion_wxyz_to_matrix(rq))
        le = np.concatenate([lp, rpy_l]).astype(np.float64)
        re = np.concatenate([rp, rpy_r]).astype(np.float64)
        advance_one_control_step(hold_arms=False, le=le, re=re)

    error_monitor = TrackingErrorMonitor(
        print_hz=args.error_print_hz,
        log_csv_path=Path(args.error_log_csv).resolve() if args.error_log_csv else None,
        warmup_sec=args.error_warmup_sec,
        lag_search_max_sec=args.error_lag_search_sec,
        spatial_resample_count=args.error_spatial_resample_count,
    )
    smooth_monitor = (
        None
        if args.disable_smoothness
        else CommandSmoothnessMonitor(
            print_hz=args.smoothness_print_hz,
            warmup_sec=args.error_warmup_sec,
            fixed_dt=control_dt,
        )
    )

    t_wall0 = time.perf_counter()
    control_step_idx = 0
    trajectory_finished = False
    stop_reason = "running"
    # Per-control-step joint command trace, arm-only, in the
    # ``LEFT_ARM_POLICY_IDX || RIGHT_ARM_POLICY_IDX`` order used by R2S2 so the
    # resulting NPZ is byte-compatible with its plotters.
    joint_cmd_times: list[float] = []
    joint_cmd_samples: list[np.ndarray] = []
    warmup_sec = float(args.error_warmup_sec)
    try:
        while True:
            elapsed = control_step_idx * control_dt
            if not args.loop_trajectory and elapsed > trajectory.duration + 1e-9:
                trajectory_finished = True
                stop_reason = "trajectory_finished"
                break
            if args.loop_trajectory and (time.perf_counter() - t_wall0) > args.max_duration_sec:
                stop_reason = "max_duration_reached"
                break

            left_e, right_e = trajectory.sample(elapsed, loop=args.loop_trajectory)
            le = left_e.astype(np.float64)
            re = right_e.astype(np.float64)
            record_ik_time = elapsed >= warmup_sec
            advance_one_control_step(hold_arms=False, le=le, re=re)
            record_ik_time = False

            if smooth_monitor is not None:
                smooth_monitor.update(elapsed, target_dof_pos)
            l_act, r_act = actual_arm_pose7_for_metrics(m, d, dof_idx, args.metrics_actual_source)
            error_monitor.update(
                elapsed_time=elapsed,
                left_target_euler=le,
                right_target_euler=re,
                left_actual_pose=l_act.astype(np.float64),
                right_actual_pose=r_act.astype(np.float64),
            )

            if elapsed >= warmup_sec:
                motor_q_cmd = motor_q_from_qj_order(target_dof_pos, dof_idx)
                pol_cmd = policy_joint_vector_from_motor_q(motor_q_cmd)
                arm_cmd = np.concatenate(
                    [
                        pol_cmd[LEFT_ARM_POLICY_IDX].astype(np.float64),
                        pol_cmd[RIGHT_ARM_POLICY_IDX].astype(np.float64),
                    ],
                    axis=0,
                )
                joint_cmd_times.append(float(elapsed))
                joint_cmd_samples.append(arm_cmd.copy())

            control_step_idx += 1
    except KeyboardInterrupt:
        print("Interrupted; writing partial summary.")
        stop_reason = "keyboard_interrupt"
    finally:
        error_monitor.close()
        if viewer is not None:
            try:
                viewer.close()
            except Exception:
                pass

    elapsed_control_sec = float(control_step_idx * control_dt)

    error_monitor.print_summary()
    if smooth_monitor is not None:
        smooth_monitor.print_summary()

    out_path = (
        Path(args.output_json).expanduser().resolve()
        if args.output_json
        else _REPO_ROOT
        / "deploy"
        / "evaluation"
        / "results"
        / f"sim2sim_{traj_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if ik_times_ms:
        _ik_arr = np.asarray(ik_times_ms, dtype=np.float64)
        compute_summary = {
            "sample_count": int(_ik_arr.size),
            "time_ms_mean": float(_ik_arr.mean()),
            "time_ms_std": float(_ik_arr.std(ddof=0)),
            "time_ms_p50": float(np.percentile(_ik_arr, 50)),
            "time_ms_p95": float(np.percentile(_ik_arr, 95)),
            "time_ms_max": float(_ik_arr.max()),
        }
    else:
        compute_summary = {
            "sample_count": 0,
            "time_ms_mean": None,
            "time_ms_std": None,
            "time_ms_p50": None,
            "time_ms_p95": None,
            "time_ms_max": None,
        }

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "runtime_type": "openwbt_sim2sim",
        "trajectory_csv": str(traj_path),
        "trajectory_name": traj_path.stem,
        "runner_config": str(Path(args.runner_config).resolve()),
        "squat_config": str(squat_cfg_path.resolve()),
        "xml_path": str(xml_path),
        "loop_trajectory": args.loop_trajectory,
        "auto_stop_on_finish": not args.loop_trajectory,
        "trajectory_duration_sec": float(trajectory.duration),
        "trajectory_sample_count": int(trajectory.sample_count),
        "control_dt_sec": control_dt,
        "simulation_dt_sec": sim_dt,
        "control_decimation": decimation,
        "ramp_in_sec": ramp_sec,
        "settle_sec": args.settle_sec,
        "elapsed_control_sec": elapsed_control_sec,
        "trajectory_finished": trajectory_finished,
        "stop_reason": stop_reason,
        "tracking_error": error_monitor.get_summary(),
        "smoothness": smooth_monitor.get_summary() if smooth_monitor is not None else None,
        "compute": compute_summary,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote summary: {out_path}")

    # R2S2-compatible NPZ trace (same keys / shapes as
    # ``trajectory.py::save_trace``). Disabled with --no_save_trace.
    if not args.no_save_trace and error_monitor.count > 0:
        if args.output_npz:
            npz_path = Path(args.output_npz).expanduser().resolve()
        else:
            npz_path = out_path.with_suffix(".npz")
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        em = error_monitor
        times_arr = np.asarray(em.sample_times, dtype=np.float64)
        lp_arr = np.stack(em.left_target_pos_samples, axis=0).astype(np.float64)
        rp_arr = np.stack(em.right_target_pos_samples, axis=0).astype(np.float64)
        lq_arr = np.stack(em.left_target_quat_samples, axis=0).astype(np.float64)
        rq_arr = np.stack(em.right_target_quat_samples, axis=0).astype(np.float64)
        lap_arr = np.stack(em.left_actual_pose_samples, axis=0).astype(np.float64)
        rap_arr = np.stack(em.right_actual_pose_samples, axis=0).astype(np.float64)
        left_target_arr = np.concatenate([lp_arr, lq_arr], axis=1)
        right_target_arr = np.concatenate([rp_arr, rq_arr], axis=1)
        if joint_cmd_samples:
            joint_cmd_arr = np.stack(joint_cmd_samples, axis=0).astype(np.float64)
            joint_cmd_t_arr = np.asarray(joint_cmd_times, dtype=np.float64)
        else:
            joint_cmd_arr = np.zeros((0, 14), dtype=np.float64)
            joint_cmd_t_arr = np.zeros((0,), dtype=np.float64)
        np.savez_compressed(
            npz_path,
            t=times_arr,
            left_target=left_target_arr,
            right_target=right_target_arr,
            left_actual=lap_arr,
            right_actual=rap_arr,
            joint_cmd=joint_cmd_arr,
            joint_cmd_t=joint_cmd_t_arr,
        )
        print(f"Wrote trace: {npz_path}")


if __name__ == "__main__":
    main()
