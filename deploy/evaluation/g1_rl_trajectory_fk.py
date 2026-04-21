"""
Analytic arm FK and joint indexing aligned with
``rl_ik_solver/.../g1_rl_controller_py/trajectory.py`` (``larm_forward`` / ``rarm_forward``,
``joint2motor_idx`` from ``g1_upper_ik.yaml``, ``LEFT_ARM_JOINT_IDX`` / ``RIGHT_ARM_JOINT_IDX``).

Used so sim2sim ``TrackingErrorMonitor`` sees the same ``actual`` pose definition as the RL deploy script.
"""

from __future__ import annotations

import math

import numpy as np

from deploy.evaluation.tracking_metrics import (
    matrix_to_quaternion_wxyz,
    t_rot_rpy,
    t_rot_x,
    t_rot_y,
    t_rot_z,
    t_trans,
)

# From g1_upper_ik.yaml: IsaacLab policy index p -> Unitree motor index m
G1_UPPER_IK_JOINT2MOTOR_IDX = np.array(
    [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28],
    dtype=np.int64,
)

LEFT_ARM_POLICY_IDX = np.array([11, 15, 19, 21, 23, 25, 27], dtype=np.int64)
RIGHT_ARM_POLICY_IDX = np.array([12, 16, 20, 22, 24, 26, 28], dtype=np.int64)

G1_JOINT_LIMITS = [
    (-3.0892, 2.6704),
    (-1.5882, 2.2515),
    (-2.6180, 2.6180),
    (-1.0472, 2.0944),
    (-1.972222054, 1.972222054),
    (-1.614429558, 1.614429558),
    (-1.614429558, 1.614429558),
    (-3.0892, 2.6704),
    (-2.2515, 1.5882),
    (-2.6180, 2.6180),
    (-1.0472, 2.0944),
    (-1.972222054, 1.972222054),
    (-1.614429558, 1.614429558),
    (-1.614429558, 1.614429558),
]
LEFT_JOINT_LIMITS_RAD = np.array(G1_JOINT_LIMITS[:7], dtype=np.float32)
RIGHT_JOINT_LIMITS_RAD = np.array(G1_JOINT_LIMITS[7:], dtype=np.float32)


def larm_forward(theta: np.ndarray) -> np.ndarray:
    shoulder = (
        t_trans(np.array([0.0039563, 0.10022, 0.24778], dtype=np.float64))
        @ t_rot_rpy(np.array([0.27931, 0.000054949, -0.00019159], dtype=np.float64))
        @ t_rot_y(float(theta[0]))
        @ t_trans(np.array([0.0, 0.038, -0.013831], dtype=np.float64))
        @ t_rot_rpy(np.array([-0.27925, 0.0, 0.0], dtype=np.float64))
        @ t_rot_x(float(theta[1]))
        @ t_trans(np.array([0.0, 0.00624, -0.1032], dtype=np.float64))
        @ t_rot_z(float(theta[2]))
    )
    elbow = t_trans(np.array([0.015783, 0.0, -0.080518], dtype=np.float64)) @ t_rot_y(float(theta[3]))
    wrist = (
        t_trans(np.array([0.1, 0.00188791, -0.01], dtype=np.float64))
        @ t_rot_x(float(theta[4]))
        @ t_trans(np.array([0.038, 0.0, 0.0], dtype=np.float64))
        @ t_rot_y(float(theta[5]))
        @ t_trans(np.array([0.046, 0.0, 0.0], dtype=np.float64))
        @ t_rot_z(float(theta[6]))
    )
    hand = shoulder @ elbow @ wrist
    pos = hand[:3, 3]
    quat = matrix_to_quaternion_wxyz(hand[:3, :3])
    return np.concatenate([pos, quat]).astype(np.float32)


def rarm_forward(theta: np.ndarray) -> np.ndarray:
    shoulder = (
        t_trans(np.array([0.0039563, -0.10021, 0.24778], dtype=np.float64))
        @ t_rot_rpy(np.array([-0.27931, 0.000054949, 0.00019159], dtype=np.float64))
        @ t_rot_y(float(theta[0]))
        @ t_trans(np.array([0.0, -0.038, -0.013831], dtype=np.float64))
        @ t_rot_rpy(np.array([0.27925, 0.0, 0.0], dtype=np.float64))
        @ t_rot_x(float(theta[1]))
        @ t_trans(np.array([0.0, -0.00624, -0.1032], dtype=np.float64))
        @ t_rot_z(float(theta[2]))
    )
    elbow = t_trans(np.array([0.015783, 0.0, -0.080518], dtype=np.float64)) @ t_rot_y(float(theta[3]))
    wrist = (
        t_trans(np.array([0.1, -0.00188791, -0.01], dtype=np.float64))
        @ t_rot_x(float(theta[4]))
        @ t_trans(np.array([0.038, 0.0, 0.0], dtype=np.float64))
        @ t_rot_y(float(theta[5]))
        @ t_trans(np.array([0.046, 0.0, 0.0], dtype=np.float64))
        @ t_rot_z(float(theta[6]))
    )
    hand = shoulder @ elbow @ wrist
    pos = hand[:3, 3]
    quat = matrix_to_quaternion_wxyz(hand[:3, :3])
    return np.concatenate([pos, quat]).astype(np.float32)


def policy_joint_vector_from_motor_q(q_motor: np.ndarray) -> np.ndarray:
    """Match ``trajectory.py``: ``joint_pos[p] = motor_q[joint2motor_idx[p]]``."""
    q = np.asarray(q_motor, dtype=np.float64).reshape(29)
    return q[G1_UPPER_IK_JOINT2MOTOR_IDX].copy()


def motor_q_from_qj_order(qj: np.ndarray, dof_idx: np.ndarray) -> np.ndarray:
    """Build ``motor_q[m]`` from OpenWBT ``qj[i] = qpos[7+dof_idx[i]]`` (same convention as Unitree id m)."""
    qj = np.asarray(qj, dtype=np.float64).reshape(-1)
    dof_idx = np.asarray(dof_idx, dtype=np.int64).reshape(-1)
    if qj.shape[0] != dof_idx.shape[0]:
        raise ValueError("qj and dof_idx length mismatch")
    n = int(qj.shape[0])
    motor = np.zeros(n, dtype=np.float64)
    for i in range(n):
        motor[int(dof_idx[i])] = qj[i]
    return motor


def fk_actual_pose7_pair_from_mujoco_qj(qj: np.ndarray, dof_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Same ``actual`` definition as ``trajectory.py`` ``_policy_step`` tracking branch:
    clip arm joints, ``larm_forward`` / ``rarm_forward``, return ``[x,y,z,qw,qx,qy,qz]`` float64.
    """
    motor_q = motor_q_from_qj_order(qj, dof_idx)
    pol = policy_joint_vector_from_motor_q(motor_q)
    left_theta = np.clip(
        pol[LEFT_ARM_POLICY_IDX],
        LEFT_JOINT_LIMITS_RAD[:, 0],
        LEFT_JOINT_LIMITS_RAD[:, 1],
    )
    right_theta = np.clip(
        pol[RIGHT_ARM_POLICY_IDX],
        RIGHT_JOINT_LIMITS_RAD[:, 0],
        RIGHT_JOINT_LIMITS_RAD[:, 1],
    )
    left_pose = larm_forward(left_theta).astype(np.float64)
    right_pose = rarm_forward(right_theta).astype(np.float64)
    return left_pose, right_pose
