#!/usr/bin/env python3
"""
Real-robot trajectory tracking benchmark for OpenWBT on the Unitree G1 (DDS).

Mirrors ``deploy/run_sim2sim_trajectory.py`` but drives a real G1 over
``unitree_sdk2py`` instead of MuJoCo. Same CSV schema, same metrics
(``TrackingErrorMonitor`` + ``CommandSmoothnessMonitor``), same summary JSON
and ``*.npz`` trace schema as ``rl_ik_solver/.../trajectory.py`` so R2S2's
plotters consume sim2sim / real runs interchangeably. ``runtime_type`` in the
summary is ``"openwbt_real"``.

Pipeline (per control step, same as sim2sim):

    1. DDS refresh proprioception.
    2. Leg policy (loco / squat / none) produces leg joint targets.
    3. Pinocchio IK maps the CSV torso-frame wrist target (per
       ``TrajectorySampler``) into pelvis-frame ``L_ee`` / ``R_ee`` targets
       and solves for 14-D arm joint targets.
    4. PD gains from the leg policy YAML drive the motors via ``low_cmd``.
    5. ``TrackingErrorMonitor.update(target, actual=larm_forward/rarm_forward)``.

Startup is explicit and safe: damping -> slow ramp to ``default_angles`` ->
wait for Enter -> policy warmup -> settle (hold_zero + cmd_ramp +
hold_target) -> stabilize wait -> arm ramp_in -> tracking loop -> damping.
``--safety_tau_limit`` force-damps if any commanded torque exceeds the limit.

Run from repository root (default leg policy is ``loco`` in stance mode)::

    python deploy/run_real_trajectory.py \
        --net enp0s31f6 --trajectory_csv /path/to/traj1.csv \
        --settle_sec 3.0 --ramp_in_sec 2.0 --stabilize_qvel_thresh 0.3 \
        --output_json deploy/evaluation/results/real_traj1.json

Requires ``unitree_sdk2py`` on the host that is connected to the robot DDS
network, plus the same policy ckpts as the sim2sim script.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

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
)
from deploy.helpers.command_helper import (
    MotorMode,
    create_damping_cmd,
    init_cmd_hg,
)
from deploy.helpers.policy_unified import LocoLowLevelPolicy, SquatLowLevelPolicy
from deploy.helpers.rotation_helper import get_gravity_orientation, transform_imu_data
from deploy.teleop.robot_control.robot_arm_ik import G1_29_ArmIK

# Pinocchio ``L_ee`` / ``R_ee`` frames attach to ``*_wrist_yaw_joint`` with this
# translation (same as ``G1_29_ArmIK``). CSV / metrics use ``wrist_yaw_link``;
# multiply before IK so the solver tracks the intended wrist pose.
_T_WRIST_YAW_TO_IK_EE = t_trans(np.array([0.05, 0.0, 0.0], dtype=np.float64))

# Pinocchio reduced model is rooted at ``pelvis`` with waist joints locked. CSV
# targets are expressed in ``torso_link`` frame. With waist = 0 the two frames
# differ only by the fixed chain pelvis -> waist_yaw -> waist_roll -> torso_link
# translation from the URDF used by the IK (see
# ``deploy/assets/g1/g1_body29_hand14.urdf``). Feed the solver
# ``T_pelvis_target = T_PELVIS_TORSO @ T_torso_target`` otherwise the target
# lands outside the arm workspace and the FK metric explodes.
_T_PELVIS_TORSO = t_trans(np.array([-0.0039635, 0.0, 0.044], dtype=np.float64))


def resolve_policy_path(cfg: Config) -> str:
    p = Path(cfg.policy_path)
    if not p.is_absolute():
        p = (_REPO_ROOT / p).resolve()
    return str(p)


class RealG1:
    """Minimal DDS bridge to a G1 via ``unitree_sdk2py``.

    Only provides what the trajectory runner needs: async state polling, cmd
    publish, damping, smooth move to a target pose. Intentionally independent
    of ``deploy.controllers.controller`` because that module opens USB handles
    at import time which we don't want for headless benchmarks.
    """

    def __init__(self, net: str, run_cfg: Config) -> None:
        # Delay SDK import until construction so ``--help`` works without the
        # robot SDK installed. Same import style as controller.py.
        from unitree_sdk2py.core.channel import (
            ChannelFactoryInitialize,
            ChannelPublisher,
            ChannelSubscriber,
        )
        from unitree_sdk2py.idl.default import (
            unitree_hg_msg_dds__LowCmd_,
            unitree_hg_msg_dds__LowState_,
        )
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
        from unitree_sdk2py.utils.crc import CRC

        if getattr(run_cfg, "msg_type", "hg") != "hg":
            raise NotImplementedError(
                "Only msg_type='hg' is supported here; add a 'go' branch if you need H1 / Go2."
            )

        self.run_cfg = run_cfg
        self.control_dt = float(run_cfg.control_dt)
        self.num_dof = int(run_cfg.num_dof)
        self.imu_on_torso = (getattr(run_cfg, "imu_type", "pelvis") == "torso")
        self.arm_waist_joint2motor_idx = list(getattr(run_cfg, "arm_waist_joint2motor_idx", []))

        self._crc = CRC()

        ChannelFactoryInitialize(0, net)
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state = unitree_hg_msg_dds__LowState_()
        self.mode_pr_ = MotorMode.PR
        self.mode_machine_ = 0

        # State published by the poller thread; latest copies are read by the
        # control loop via refresh_prop().
        self.qj = np.zeros(self.num_dof, dtype=np.float32)
        self.dqj = np.zeros(self.num_dof, dtype=np.float32)
        self.quat = np.zeros(4, dtype=np.float32)
        self.ang_vel = np.zeros(3, dtype=np.float32)
        self.tau_record = np.zeros(self.num_dof, dtype=np.float64)

        # DDS topic objects. Subscriber is polled in a background thread,
        # publisher is called from the main loop.
        self._lowcmd_publisher = ChannelPublisher(run_cfg.lowcmd_topic, LowCmdHG)
        self._lowcmd_publisher.Init()
        self._lowstate_subscriber = ChannelSubscriber(run_cfg.lowstate_topic, LowStateHG)
        self._lowstate_subscriber.Init()

        self._stop = threading.Event()
        self._state_thread = threading.Thread(target=self._state_loop, daemon=True)
        self._state_thread.start()

        print("[real] Waiting for initial LowState packet ...")
        t0 = time.perf_counter()
        while self.low_state.tick == 0:
            if time.perf_counter() - t0 > 10.0:
                raise RuntimeError("No LowState received on DDS within 10s; check --net")
            time.sleep(self.control_dt)
        print(f"[real] Connected to robot (mode_machine={self.mode_machine_}).")

        init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)

    # Thread body: keep pulling latest LowState from DDS.
    def _state_loop(self) -> None:
        while not self._stop.is_set():
            try:
                msg = self._lowstate_subscriber.Read()
            except Exception:
                time.sleep(0.001)
                continue
            if msg is None:
                time.sleep(0.001)
                continue
            self.low_state = msg
            self.mode_machine_ = self.low_state.mode_machine
            time.sleep(0.001)

    def close(self) -> None:
        self._stop.set()

    def refresh_prop(self) -> None:
        """Snapshot joint / IMU state into numpy arrays for the control step."""
        for i in range(self.num_dof):
            self.qj[i] = self.low_state.motor_state[i].q
            self.dqj[i] = self.low_state.motor_state[i].dq
        self.quat[:] = self.low_state.imu_state.quaternion
        self.ang_vel[:] = np.array(self.low_state.imu_state.gyroscope, dtype=np.float32)
        if self.imu_on_torso and self.arm_waist_joint2motor_idx:
            waist_yaw = self.low_state.motor_state[self.arm_waist_joint2motor_idx[0]].q
            waist_yaw_omega = self.low_state.motor_state[self.arm_waist_joint2motor_idx[0]].dq
            self.quat, self.ang_vel = transform_imu_data(
                waist_yaw=waist_yaw,
                waist_yaw_omega=waist_yaw_omega,
                imu_quat=self.quat,
                imu_omega=self.ang_vel,
            )

    def publish_damping(self) -> None:
        create_damping_cmd(self.low_cmd)
        self._publish()

    def publish_pd(
        self,
        target_q: np.ndarray,
        kps: np.ndarray,
        kds: np.ndarray,
        motor_idx: np.ndarray,
        tau_limit: float,
    ) -> None:
        """Write PD target + kp/kd for each controlled motor, publish, record tau."""
        # Populate motor_cmd only for the controlled indices; other motors keep
        # whatever last cmd we had (init_cmd_hg left them at zero with kp/kd/tau
        # zero so they won't move).
        for i, m in enumerate(motor_idx):
            idx = int(m)
            self.low_cmd.motor_cmd[idx].q = float(target_q[i])
            self.low_cmd.motor_cmd[idx].dq = 0.0
            self.low_cmd.motor_cmd[idx].kp = float(kps[i])
            self.low_cmd.motor_cmd[idx].kd = float(kds[i])
            self.low_cmd.motor_cmd[idx].tau = 0.0
            self.tau_record[idx] = (float(target_q[i]) - float(self.low_state.motor_state[idx].q)) * float(kps[i]) \
                - float(self.low_state.motor_state[idx].dq) * float(kds[i])
        # Safety cutoff: if estimated PD torque exceeds the limit on any motor,
        # replace the whole cmd with damping this tick.
        if tau_limit > 0.0 and float(np.max(np.abs(self.tau_record))) > tau_limit:
            bad = np.where(np.abs(self.tau_record) > tau_limit)[0].tolist()
            print(f"[real] SAFETY: |tau|>{tau_limit:.0f} on motors {bad}; forcing damping this tick.")
            create_damping_cmd(self.low_cmd)
        self._publish()

    def _publish(self) -> None:
        self.low_cmd.crc = self._crc.Crc(self.low_cmd)
        self._lowcmd_publisher.Write(self.low_cmd)

    def hold_damping_for(self, seconds: float) -> None:
        n = max(1, int(round(seconds / self.control_dt)))
        for _ in range(n):
            self.publish_damping()
            time.sleep(self.control_dt)

    def move_to_default(
        self,
        default_angles: np.ndarray,
        kps: np.ndarray,
        kds: np.ndarray,
        motor_idx: np.ndarray,
        ramp_sec: float,
        hold_sec: float,
        tau_limit: float,
    ) -> None:
        """Slowly ramp the controlled joints from current state to default_angles."""
        self.refresh_prop()
        start_q = np.array([self.qj[int(m)] for m in motor_idx], dtype=np.float64)
        n_ramp = max(1, int(round(ramp_sec / self.control_dt)))
        for k in range(n_ramp):
            alpha = float(k + 1) / float(n_ramp)
            target = (1.0 - alpha) * start_q + alpha * default_angles.astype(np.float64)
            self.publish_pd(target, kps, kds, motor_idx, tau_limit)
            time.sleep(self.control_dt)
        n_hold = max(0, int(round(hold_sec / self.control_dt)))
        for _ in range(n_hold):
            self.publish_pd(default_angles.astype(np.float64), kps, kds, motor_idx, tau_limit)
            time.sleep(self.control_dt)


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenWBT real-robot trajectory tracking (G1 + DDS + IK + leg policy).")
    parser.add_argument("--net", type=str, required=True, help="DDS network interface (e.g. eth0, enp0s31f6).")
    parser.add_argument(
        "--runner_config",
        type=str,
        default="deploy/configs/run_teleoperation.yaml",
        help="YAML with dof_idx, action_hl_idx, default_angles, control_dt, ...",
    )
    parser.add_argument("--trajectory_csv", type=str, required=True, help="Trajectory CSV (torso-frame wrists).")
    parser.add_argument("--output_json", type=str, default="", help="Where to write result summary JSON.")
    parser.add_argument(
        "--output_npz",
        type=str,
        default="",
        help=(
            "Trace NPZ path. Defaults to --output_json with suffix replaced by .npz. "
            "Same schema as the R2S2 and sim2sim runners."
        ),
    )
    parser.add_argument("--no_save_trace", action="store_true", help="Skip writing the trace NPZ.")
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
        "--leg_policy",
        type=str,
        choices=("loco", "squat", "none"),
        default="loco",
        help=(
            "Which low-level policy regulates the legs. 'loco' uses ckpts/loco.onnx "
            "at cmd=[0,0,0] + stance=True (recommended standing mode for arm "
            "tracking). 'squat' uses ckpts/squat.onnx. 'none' PD-holds legs + "
            "waist at run_cfg.default_angles with no policy."
        ),
    )
    parser.add_argument(
        "--squat_cmd",
        type=float,
        nargs=2,
        default=None,
        metavar=("H", "P"),
        help="Squat policy command [height_delta, pitch_delta]; default [0, 0].",
    )
    parser.add_argument(
        "--use_cmd_debug",
        action="store_true",
        help="When --leg_policy squat, force cmd = squat YAML cmd_debug.",
    )
    parser.add_argument(
        "--loco_cmd",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("VX", "VY", "WZ"),
        help="Loco policy body-frame cmd (m/s, m/s, rad/s); default [0,0,0]. "
             "Only used when --loco_walk is set.",
    )
    parser.add_argument(
        "--loco_walk",
        action="store_true",
        help="When --leg_policy loco, walk with --loco_cmd instead of stance.",
    )
    parser.add_argument(
        "--settle_hold_zero_sec",
        type=float,
        default=1.0,
        help="Settle sub-phase: hold cmd=0 for this long before ramping. Part of --settle_sec.",
    )
    parser.add_argument(
        "--cmd_ramp_sec",
        type=float,
        default=1.0,
        help="Settle sub-phase: ramp cmd 0 -> target after hold_zero. Part of --settle_sec.",
    )
    parser.add_argument(
        "--stabilize_qvel_thresh",
        type=float,
        default=0.3,
        help=(
            "After settle, keep stepping until max |qvel| (on dof_idx) is below "
            "this threshold (rad/s), or --stabilize_max_sec elapses. 0 disables. "
            "Default 0.3 rad/s (on real hardware start conservative)."
        ),
    )
    parser.add_argument("--stabilize_max_sec", type=float, default=5.0, help="Cap on stabilize wait.")
    parser.add_argument(
        "--policy_warmup_iters",
        type=int,
        default=20,
        help="Policy inference iterations before any cmd is published, to warm up hidden state.",
    )
    parser.add_argument(
        "--move_to_default_sec",
        type=float,
        default=3.0,
        help="Duration of the safe ramp from current pose to run_cfg.default_angles at start.",
    )
    parser.add_argument(
        "--move_to_default_hold_sec",
        type=float,
        default=0.5,
        help="Hold default pose for this many seconds after the ramp, before policy warmup.",
    )
    parser.add_argument(
        "--initial_damping_sec",
        type=float,
        default=0.5,
        help="Publish damping cmds for this many seconds right after DDS connect.",
    )
    parser.add_argument(
        "--final_damping_sec",
        type=float,
        default=1.5,
        help="Publish damping cmds for this many seconds at the end of the run.",
    )
    parser.add_argument(
        "--safety_tau_limit",
        type=float,
        default=120.0,
        help=(
            "Abort a cmd (fall back to damping that tick) if any controlled motor's "
            "estimated PD torque exceeds this Nm. Set 0 to disable."
        ),
    )
    parser.add_argument(
        "--wait_for_keypress",
        action="store_true",
        help=(
            "Block on stdin after moving to default pose until the user presses "
            "Enter. Strongly recommended the first time you run on real hardware."
        ),
    )
    args = parser.parse_args()

    run_cfg = Config(str(Path(args.runner_config).resolve()))

    # --------------------------------------------------------------------- #
    # Leg policy resolution (same logic as run_sim2sim_trajectory.py).
    # --------------------------------------------------------------------- #
    leg_policy_kind = args.leg_policy
    squat_cfg_path = _REPO_ROOT / "deploy" / "configs" / getattr(run_cfg, "squat_config", "g1_squat.yaml")
    loco_cfg_path = _REPO_ROOT / "deploy" / "configs" / getattr(run_cfg, "loco_config", "g1_loco.yaml")
    if leg_policy_kind == "squat":
        leg_cfg_path = squat_cfg_path
    elif leg_policy_kind == "loco":
        leg_cfg_path = loco_cfg_path
    else:
        leg_cfg_path = squat_cfg_path
    leg_cfg = Config(str(leg_cfg_path.resolve()))
    leg_cfg.policy_path = resolve_policy_path(leg_cfg)

    if leg_policy_kind == "squat":
        leg_policy_obj: SquatLowLevelPolicy | LocoLowLevelPolicy | None = SquatLowLevelPolicy(leg_cfg)
    elif leg_policy_kind == "loco":
        leg_policy_obj = LocoLowLevelPolicy(leg_cfg)
    else:
        leg_policy_obj = None
    leg_cfg_dof_idx = np.asarray(leg_cfg.dof_idx, dtype=np.int64)
    leg_cfg_action_idx = np.asarray(leg_cfg.action_idx, dtype=np.int64)

    if leg_policy_kind == "squat":
        if args.squat_cmd is not None:
            cmd_target = np.array(args.squat_cmd, dtype=np.float32)
            cmd_source = "--squat_cmd"
        elif args.use_cmd_debug:
            cmd_target = np.array(leg_cfg.cmd_debug, dtype=np.float32).copy()
            cmd_source = "squat YAML cmd_debug (--use_cmd_debug)"
        else:
            cmd_target = np.zeros(2, dtype=np.float32)
            cmd_source = "default [0, 0]"
    elif leg_policy_kind == "loco":
        cmd_target = np.array(args.loco_cmd, dtype=np.float32)
        cmd_source = "--loco_cmd"
        if not args.loco_walk and np.any(cmd_target != 0.0):
            print(
                f"[real] NOTE: --loco_cmd={cmd_target.tolist()} ignored because "
                "--loco_walk is off; loco runs stance with cmd=0."
            )
            cmd_target = np.zeros(3, dtype=np.float32)
    else:
        cmd_target = np.zeros(0, dtype=np.float32)
        cmd_source = "n/a"
    cmd_now = np.zeros_like(cmd_target)

    max_cmd = np.asarray(
        getattr(leg_cfg, "max_cmd", [np.inf] * cmd_target.shape[0]),
        dtype=np.float32,
    )
    if leg_policy_kind == "squat":
        print(f"[real] leg_policy=squat, cmd={cmd_target.tolist()} (from {cmd_source})")
        if np.any(np.abs(cmd_target) > max_cmd + 1e-6):
            print(
                f"[real] WARNING: |cmd_target|={np.abs(cmd_target).tolist()} exceeds "
                f"max_cmd={max_cmd.tolist()} from squat YAML; out-of-distribution."
            )
    elif leg_policy_kind == "loco":
        mode = "walking with --loco_cmd" if args.loco_walk else "stance (cmd forced to 0)"
        print(f"[real] leg_policy=loco, mode={mode}, cmd={cmd_target.tolist()}")
    else:
        print("[real] leg_policy=none; legs + waist PD-held at run_cfg.default_angles")

    # --------------------------------------------------------------------- #
    # Shared artefacts: trajectory, IK, error monitors, I/O paths.
    # --------------------------------------------------------------------- #
    traj_path = Path(args.trajectory_csv).expanduser().resolve()
    trajectory = TrajectorySampler(traj_path)

    arm_ik = G1_29_ArmIK(Unit_Test=False, Visualization=False)

    dof_idx = np.array(run_cfg.dof_idx, dtype=np.int64)
    action_hl_idx = np.array(run_cfg.action_hl_idx, dtype=np.int64)
    default_angles = np.array(run_cfg.default_angles, dtype=np.float64)
    kps = np.array(leg_cfg.kps, dtype=np.float64)
    kds = np.array(leg_cfg.kds, dtype=np.float64)
    control_dt = float(run_cfg.control_dt)

    # target_dof_pos is indexed by the 29-D dof ordering (= motor index for G1).
    # We only ever publish PD commands for motors in dof_idx, so uncontrolled
    # motors stay at kp=kd=0 (no torque) from init_cmd_hg.
    target_dof_pos = default_angles.copy()

    # --------------------------------------------------------------------- #
    # Robot + clean shutdown plumbing.
    # --------------------------------------------------------------------- #
    robot = RealG1(args.net, run_cfg)

    shutdown = {"requested": False, "reason": "running"}
    orig_sigint = signal.getsignal(signal.SIGINT)

    def _handle_sigint(signum, frame):  # noqa: ARG001
        shutdown["requested"] = True
        shutdown["reason"] = "keyboard_interrupt"
        # Restore default so a second Ctrl-C aborts hard.
        signal.signal(signal.SIGINT, orig_sigint)

    signal.signal(signal.SIGINT, _handle_sigint)

    # Pre-initialise everything the summary / NPZ writers may access, so an
    # early failure in settle / ramp-in still produces a partial summary.
    error_monitor: TrackingErrorMonitor | None = None
    smooth_monitor: CommandSmoothnessMonitor | None = None
    ik_times_ms: list[float] = []
    joint_cmd_times: list[float] = []
    joint_cmd_samples: list[np.ndarray] = []
    control_step_idx = 0
    elapsed_control_sec = 0.0
    trajectory_finished = False
    stop_reason = "aborted_before_tracking"

    def apply_target(target: np.ndarray) -> None:
        """Publish PD cmd for the controlled motors and tick the safety limit."""
        robot.publish_pd(
            target_q=target[dof_idx],
            kps=kps,
            kds=kds,
            motor_idx=dof_idx,
            tau_limit=float(args.safety_tau_limit),
        )

    def measure_ik_solve(
        left_euler6: np.ndarray,
        right_euler6: np.ndarray,
        q_arm_seed: np.ndarray,
        dq_arm: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        T_torso_wrist_l = euler_xyz_to_T(left_euler6[:3], left_euler6[3:6])
        T_torso_wrist_r = euler_xyz_to_T(right_euler6[:3], right_euler6[3:6])
        T_pelvis_ik_l = _T_PELVIS_TORSO @ T_torso_wrist_l @ _T_WRIST_YAW_TO_IK_EE
        T_pelvis_ik_r = _T_PELVIS_TORSO @ T_torso_wrist_r @ _T_WRIST_YAW_TO_IK_EE
        t0 = time.perf_counter()
        sol_q, _ = arm_ik.solve_ik(T_pelvis_ik_l, T_pelvis_ik_r, q_arm_seed, dq_arm)
        return sol_q, (time.perf_counter() - t0) * 1000.0

    def actual_arm_pose7_from_qj(qj29: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Same FK + clip + joint2motor_idx path as R2S2 trajectory.py and sim2sim
        # 'rl_fk' mode. qj29 must be indexed by dof_idx (motor order for G1).
        qj_in_dof_order = qj29[dof_idx]
        return fk_actual_pose7_pair_from_mujoco_qj(qj_in_dof_order, dof_idx)

    try:
        # ----------------------------------------------------------------- #
        # Safety phase 0: damping so the robot hangs limp briefly.
        # ----------------------------------------------------------------- #
        robot.hold_damping_for(float(args.initial_damping_sec))

        # ----------------------------------------------------------------- #
        # Safety phase 1: slow ramp to default_angles so the current pose
        # doesn't jump to whatever the policy wants on its first inference.
        # ----------------------------------------------------------------- #
        print("[real] Moving to run_cfg.default_angles ...")
        default_q_on_dof = default_angles[dof_idx]
        robot.move_to_default(
            default_angles=default_q_on_dof,
            kps=kps,
            kds=kds,
            motor_idx=dof_idx,
            ramp_sec=float(args.move_to_default_sec),
            hold_sec=float(args.move_to_default_hold_sec),
            tau_limit=float(args.safety_tau_limit),
        )

        if args.wait_for_keypress:
            try:
                input(
                    "[real] Robot is at default pose. Make sure it is stable and "
                    "supported; press Enter to START the leg policy + settle + "
                    "trajectory tracking ... "
                )
            except EOFError:
                pass

        # ----------------------------------------------------------------- #
        # Policy warmup: inference only, no new cmd - target_dof_pos is still
        # default_angles so PD is already holding pose while we warm up.
        # ----------------------------------------------------------------- #
        if leg_policy_obj is not None:
            robot.refresh_prop()
            for _ in range(max(0, int(args.policy_warmup_iters))):
                qj_leg = robot.qj[leg_cfg_dof_idx].astype(np.float32)
                dqj_leg = robot.dqj[leg_cfg_dof_idx].astype(np.float32)
                g0 = get_gravity_orientation(robot.quat).astype(np.float32)
                omega0 = robot.ang_vel.astype(np.float32)
                if leg_policy_kind == "loco":
                    leg_policy_obj.gait_planner.update_gait_phase(stop=not args.loco_walk)
                leg_policy_obj.inference(np.zeros_like(cmd_target), g0, omega0, qj_leg, dqj_leg)
                # Keep holding default_angles on the motors during warmup.
                apply_target(target_dof_pos)
                time.sleep(control_dt)
                robot.refresh_prop()

        # ----------------------------------------------------------------- #
        # Control step primitive: identical behaviour to sim2sim's
        # advance_one_control_step(). 'hold_arms=True' freezes arms at
        # current qpos; otherwise IK the CSV target.
        # ----------------------------------------------------------------- #
        ik_times_ms_warmup: list[float] = []  # discarded, only main-loop IK counts

        def step_control(
            hold_arms: bool,
            left_euler6: np.ndarray,
            right_euler6: np.ndarray,
            record_ik: bool,
        ) -> float | None:
            """Run one control step. Returns IK solve ms if record_ik else None."""
            robot.refresh_prop()
            ik_ms: float | None = None

            if leg_policy_obj is not None:
                qj_leg = robot.qj[leg_cfg_dof_idx].astype(np.float32)
                dqj_leg = robot.dqj[leg_cfg_dof_idx].astype(np.float32)
                g = get_gravity_orientation(robot.quat).astype(np.float32)
                omega = robot.ang_vel.astype(np.float32)
                if leg_policy_kind == "loco":
                    leg_policy_obj.gait_planner.update_gait_phase(stop=not args.loco_walk)
                _, _, leg_slice = leg_policy_obj.inference(cmd_now, g, omega, qj_leg, dqj_leg)
                target_dof_pos[leg_cfg_action_idx] = leg_slice.astype(np.float64)
            # else: target_dof_pos leg / waist slice stays at default_angles.

            if hold_arms:
                target_dof_pos[action_hl_idx] = robot.qj[action_hl_idx].astype(np.float64)
            else:
                q_arm = robot.qj[action_hl_idx].astype(np.float64)
                dq_arm = robot.dqj[action_hl_idx].astype(np.float64)
                sol_q, ms = measure_ik_solve(left_euler6, right_euler6, q_arm, dq_arm)
                target_dof_pos[action_hl_idx] = sol_q
                if record_ik:
                    ik_ms = float(ms)

            apply_target(target_dof_pos)
            return ik_ms

        def wait_control_deadline(deadline: float) -> None:
            remaining = deadline - time.perf_counter()
            if remaining > 0.0:
                time.sleep(remaining)

        # ----------------------------------------------------------------- #
        # Settle: hold_zero + cmd_ramp + hold_target with arms PD-held.
        # ----------------------------------------------------------------- #
        wall_anchor = time.perf_counter()
        wall_phase = 0.0

        settle_control_steps = max(0, int(round(args.settle_sec / control_dt)))
        ramp_meaningful = leg_policy_obj is not None and np.any(cmd_target != 0.0)
        if not ramp_meaningful:
            hold_zero_steps = settle_control_steps
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
            if shutdown["requested"]:
                break
            if not ramp_meaningful:
                cmd_now[:] = 0.0
            elif k < hold_zero_steps:
                cmd_now[:] = 0.0
            elif cmd_ramp_steps > 0 and k < hold_zero_steps + cmd_ramp_steps:
                alpha = float(k - hold_zero_steps + 1) / float(cmd_ramp_steps)
                cmd_now[:] = alpha * cmd_target
            else:
                cmd_now[:] = cmd_target
            step_control(hold_arms=True, left_euler6=le_hold, right_euler6=re_hold, record_ik=False)
            wall_phase += control_dt
            wait_control_deadline(wall_anchor + wall_phase)
        if ramp_meaningful:
            cmd_now[:] = cmd_target

        # ----------------------------------------------------------------- #
        # Stabilize wait (optional): keep arms held, wait for max|qvel| to
        # drop below threshold so we don't start tracking while legs still
        # oscillate.
        # ----------------------------------------------------------------- #
        if not shutdown["requested"] and float(args.stabilize_qvel_thresh) > 0.0:
            stab_max_steps = max(0, int(round(max(0.0, float(args.stabilize_max_sec)) / control_dt)))
            stab_v = float("inf")
            stab_done = False
            for stab_k in range(stab_max_steps):
                if shutdown["requested"]:
                    break
                step_control(hold_arms=True, left_euler6=le_hold, right_euler6=re_hold, record_ik=False)
                wall_phase += control_dt
                wait_control_deadline(wall_anchor + wall_phase)
                robot.refresh_prop()
                stab_v = float(np.max(np.abs(robot.dqj[dof_idx])))
                if stab_v < float(args.stabilize_qvel_thresh):
                    print(
                        f"[real] stabilized after extra {(stab_k + 1) * control_dt:.2f}s "
                        f"(max|qvel|={stab_v:.3f} < {args.stabilize_qvel_thresh:.3f})"
                    )
                    stab_done = True
                    break
            if not stab_done and not shutdown["requested"]:
                print(
                    f"[real] stabilize wait hit --stabilize_max_sec={args.stabilize_max_sec:.2f}s "
                    f"(max|qvel|={stab_v:.3f}); proceeding anyway."
                )

        # ----------------------------------------------------------------- #
        # Arm ramp-in: blend measured torso-frame wrist pose to traj[0].
        # Legs keep running the leg policy at cmd_target the whole time.
        # ----------------------------------------------------------------- #
        if not shutdown["requested"]:
            robot.refresh_prop()
            left_pose7_0, right_pose7_0 = actual_arm_pose7_from_qj(robot.qj)
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
                if shutdown["requested"]:
                    break
                a = float(k + 1) / float(max(ramp_control_steps, 1))
                lp = (1.0 - a) * left_start_pos + a * left_g[:3]
                rp = (1.0 - a) * right_start_pos + a * right_g[:3]
                lq = quat_wxyz_slerp(left_start_q, left_tgt_q, a)
                rq = quat_wxyz_slerp(right_start_q, right_tgt_q, a)
                rpy_l = matrix_to_rpy(quaternion_wxyz_to_matrix(lq))
                rpy_r = matrix_to_rpy(quaternion_wxyz_to_matrix(rq))
                le = np.concatenate([lp, rpy_l]).astype(np.float64)
                re = np.concatenate([rp, rpy_r]).astype(np.float64)
                step_control(hold_arms=False, left_euler6=le, right_euler6=re, record_ik=False)
                wall_phase += control_dt
                wait_control_deadline(wall_anchor + wall_phase)

        # ----------------------------------------------------------------- #
        # Tracking loop with metrics collection (same schema as sim2sim /
        # R2S2 trajectory.py).
        # ----------------------------------------------------------------- #
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

        warmup_sec = float(args.error_warmup_sec)

        t_wall0 = time.perf_counter()
        control_step_idx = 0
        trajectory_finished = False
        stop_reason = shutdown["reason"] if shutdown["requested"] else "running"

        while not shutdown["requested"]:
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
            record_ik = elapsed >= warmup_sec
            ik_ms = step_control(hold_arms=False, left_euler6=le, right_euler6=re, record_ik=record_ik)
            wall_phase += control_dt
            wait_control_deadline(wall_anchor + wall_phase)

            if record_ik and ik_ms is not None:
                ik_times_ms.append(ik_ms)

            if smooth_monitor is not None:
                smooth_monitor.update(elapsed, target_dof_pos)
            l_act, r_act = actual_arm_pose7_from_qj(robot.qj)
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

        if shutdown["requested"]:
            stop_reason = shutdown["reason"]

        elapsed_control_sec = float(control_step_idx * control_dt)

        # Tear-down: always damp down from whatever state we were in so the
        # robot doesn't try to hold the last target forever.
        if error_monitor is not None:
            error_monitor.close()
            error_monitor.print_summary()
        if smooth_monitor is not None:
            smooth_monitor.print_summary()

    except Exception as e:  # noqa: BLE001
        print(f"[real] ERROR during control: {type(e).__name__}: {e}")
        stop_reason = f"exception: {type(e).__name__}"
    finally:
        try:
            robot.hold_damping_for(float(args.final_damping_sec))
        except Exception as e:  # noqa: BLE001
            print(f"[real] WARNING: final damping failed: {e}")
        try:
            robot.close()
        except Exception:
            pass

    # ------------------------------------------------------------------- #
    # Write summary JSON + trace NPZ (R2S2-compatible schema).
    # ------------------------------------------------------------------- #
    out_path = (
        Path(args.output_json).expanduser().resolve()
        if args.output_json
        else _REPO_ROOT
        / "deploy"
        / "evaluation"
        / "results"
        / f"real_{traj_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
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
        "runtime_type": "openwbt_real",
        "trajectory_csv": str(traj_path),
        "trajectory_name": traj_path.stem,
        "runner_config": str(Path(args.runner_config).resolve()),
        "leg_policy": leg_policy_kind,
        "leg_policy_config": str(leg_cfg_path.resolve()),
        "leg_policy_cmd": cmd_target.tolist(),
        "loco_walk": bool(args.loco_walk) if leg_policy_kind == "loco" else None,
        "loop_trajectory": args.loop_trajectory,
        "auto_stop_on_finish": not args.loop_trajectory,
        "trajectory_duration_sec": float(trajectory.duration),
        "trajectory_sample_count": int(trajectory.sample_count),
        "control_dt_sec": control_dt,
        "ramp_in_sec": max(0.0, float(args.ramp_in_sec)),
        "settle_sec": args.settle_sec,
        "elapsed_control_sec": elapsed_control_sec,
        "trajectory_finished": trajectory_finished,
        "stop_reason": stop_reason,
        "net": args.net,
        "tracking_error": error_monitor.get_summary() if error_monitor is not None else None,
        "smoothness": smooth_monitor.get_summary() if smooth_monitor is not None else None,
        "compute": compute_summary,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[real] Wrote summary: {out_path}")

    if not args.no_save_trace and error_monitor is not None and error_monitor.count > 0:
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
        print(f"[real] Wrote trace: {npz_path}")


if __name__ == "__main__":
    main()
