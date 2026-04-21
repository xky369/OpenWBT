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

Run from repository root::

    python deploy/run_sim2sim_trajectory.py --trajectory_csv /path/to/traj1.csv

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
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

os.chdir(_REPO_ROOT)

from deploy.config import Config
from deploy.evaluation.g1_rl_trajectory_fk import fk_actual_pose7_pair_from_mujoco_qj
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
    parser.add_argument("--error_log_csv", type=str, default="", help="Optional per-step tracking error CSV.")
    parser.add_argument("--loop_trajectory", action="store_true", help="Loop CSV in time; default plays once.")
    parser.add_argument("--max_duration_sec", type=float, default=600.0, help="Safety cap when --loop_trajectory is set.")
    parser.add_argument("--settle_sec", type=float, default=2.0, help="Squat-only pre-roll before tracking (wall clock).")
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
        help="Override squat policy command [height, pitch]; default uses squat config cmd_debug.",
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

    cmd_raw = squat_cfg.cmd_debug.copy() if args.squat_cmd is None else np.array(args.squat_cmd, dtype=np.float32)

    counter = 0

    def physics_substep(
        hold_arms: bool,
        left_euler6: np.ndarray,
        right_euler6: np.ndarray,
    ) -> None:
        nonlocal target_dof_pos, counter
        if counter % decimation == 0:
            qj = d.qpos[7:][dof_idx].astype(np.float32)
            dqj = d.qvel[6:][dof_idx].astype(np.float32)
            quat = d.qpos[3:7].astype(np.float32)
            omega = d.qvel[3:6].astype(np.float32)
            g = get_gravity_orientation(quat).astype(np.float32)
            td = target_dof_pos.astype(np.float32).copy()
            _, _, leg_slice = squat_policy.inference(cmd_raw, g, omega, qj, dqj)
            target_dof_pos[:] = td.astype(np.float64)
            target_dof_pos[squat_cfg.action_idx] = leg_slice.astype(np.float64)

            if hold_arms:
                target_dof_pos[action_hl_idx] = d.qpos[7:][action_hl_idx].astype(np.float64)
            else:
                T_torso_wrist_l = euler_xyz_to_T(left_euler6[:3], left_euler6[3:6])
                T_torso_wrist_r = euler_xyz_to_T(right_euler6[:3], right_euler6[3:6])
                T_torso_ik_l = T_torso_wrist_l @ _T_WRIST_YAW_TO_IK_EE
                T_torso_ik_r = T_torso_wrist_r @ _T_WRIST_YAW_TO_IK_EE
                T_w_t = mj_body_T_world(m, d, "torso_link")
                T_tgt_l_w = T_w_t @ T_torso_ik_l
                T_tgt_r_w = T_w_t @ T_torso_ik_r

                q_arm = d.qpos[7:][action_hl_idx].astype(np.float64)
                dq_arm = d.qvel[6:][action_hl_idx].astype(np.float64)
                sol_q, _ = arm_ik.solve_ik(T_tgt_l_w, T_tgt_r_w, q_arm, dq_arm)
                target_dof_pos[action_hl_idx] = sol_q

        tau = pd_torques(kps, kds, dof_idx, target_dof_pos, m, d)
        d.ctrl[:] = 0.0
        d.ctrl[dof_idx] = tau
        mujoco.mj_step(m, d)
        counter += 1

    settle_physics_steps = max(0, int(args.settle_sec / sim_dt))
    le_hold = np.zeros(6, dtype=np.float64)
    re_hold = np.zeros(6, dtype=np.float64)
    for _ in range(settle_physics_steps):
        physics_substep(hold_arms=True, left_euler6=le_hold, right_euler6=re_hold)

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
        for _ in range(decimation):
            physics_substep(hold_arms=False, left_euler6=le, right_euler6=re)

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
    try:
        while True:
            elapsed = control_step_idx * control_dt
            if not args.loop_trajectory and elapsed > trajectory.duration + 1e-9:
                break
            if args.loop_trajectory and (time.perf_counter() - t_wall0) > args.max_duration_sec:
                break

            left_e, right_e = trajectory.sample(elapsed, loop=args.loop_trajectory)
            le = left_e.astype(np.float64)
            re = right_e.astype(np.float64)
            for _ in range(decimation):
                physics_substep(hold_arms=False, left_euler6=le, right_euler6=re)

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

            control_step_idx += 1
    finally:
        error_monitor.close()

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
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "runtime_type": "openwbt_sim2sim",
        "trajectory_csv": str(traj_path),
        "runner_config": str(Path(args.runner_config).resolve()),
        "squat_config": str(squat_cfg_path.resolve()),
        "xml_path": str(xml_path),
        "control_dt_sec": control_dt,
        "simulation_dt_sec": sim_dt,
        "control_decimation": decimation,
        "ramp_in_sec": ramp_sec,
        "settle_sec": args.settle_sec,
        "loop_trajectory": args.loop_trajectory,
        "tracking_error": error_monitor.get_summary(),
        "smoothness": smooth_monitor.get_summary() if smooth_monitor is not None else None,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote summary: {out_path}")


if __name__ == "__main__":
    main()
