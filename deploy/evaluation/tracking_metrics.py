"""
Tracking and smoothness metrics aligned with
``rl_ik_solver/.../g1_rl_controller_py/trajectory.py`` (same definitions:
position L2 error, quaternion geodesic angle, RMSE / max, lag search,
spatial resample, joint-command smoothness).
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np


def t_trans(offset_xyz: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = offset_xyz
    return transform


def t_rot_x(theta: float) -> np.ndarray:
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, c, -s, 0.0], [0.0, s, c, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def t_rot_y(theta: float) -> np.ndarray:
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array(
        [[c, 0.0, s, 0.0], [0.0, 1.0, 0.0, 0.0], [-s, 0.0, c, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def t_rot_z(theta: float) -> np.ndarray:
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array(
        [[c, -s, 0.0, 0.0], [s, c, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def t_rot_rpy(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = float(rpy[0]), float(rpy[1]), float(rpy[2])
    cr = math.cos(roll)
    sr = math.sin(roll)
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cy = math.cos(yaw)
    sy = math.sin(yaw)

    transform = np.eye(4, dtype=np.float64)
    transform[0, 0] = cy * cp
    transform[0, 1] = cy * sp * sr - sy * cr
    transform[0, 2] = cy * sp * cr + sy * sr
    transform[1, 0] = sy * cp
    transform[1, 1] = sy * sp * sr + cy * cr
    transform[1, 2] = sy * sp * cr - cy * sr
    transform[2, 0] = -sp
    transform[2, 1] = cp * sr
    transform[2, 2] = cp * cr
    return transform


def matrix_to_quaternion_wxyz(matrix: np.ndarray) -> np.ndarray:
    trace = matrix[0, 0] + matrix[1, 1] + matrix[2, 2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (matrix[2, 1] - matrix[1, 2]) / s
        qy = (matrix[0, 2] - matrix[2, 0]) / s
        qz = (matrix[1, 0] - matrix[0, 1]) / s
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        s = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        qw = (matrix[2, 1] - matrix[1, 2]) / s
        qx = 0.25 * s
        qy = (matrix[0, 1] + matrix[1, 0]) / s
        qz = (matrix[0, 2] + matrix[2, 0]) / s
    elif matrix[1, 1] > matrix[2, 2]:
        s = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        qw = (matrix[0, 2] - matrix[2, 0]) / s
        qx = (matrix[0, 1] + matrix[1, 0]) / s
        qy = 0.25 * s
        qz = (matrix[1, 2] + matrix[2, 1]) / s
    else:
        s = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
        qw = (matrix[1, 0] - matrix[0, 1]) / s
        qx = (matrix[0, 2] + matrix[2, 0]) / s
        qy = (matrix[1, 2] + matrix[2, 1]) / s
        qz = 0.25 * s
    quat = np.array([qw, qx, qy, qz], dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return quat / norm


def rpy_to_quaternion_wxyz(rpy: np.ndarray) -> np.ndarray:
    rot = t_rot_rpy(rpy)
    return matrix_to_quaternion_wxyz(rot[:3, :3])


def quaternion_wxyz_to_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(quat_wxyz, dtype=np.float64)
    n = math.sqrt(float(q[0] ** 2 + q[1] ** 2 + q[2] ** 2 + q[3] ** 2))
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = q[0] / n, q[1] / n, q[2] / n, q[3] / n
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_rpy(matrix: np.ndarray) -> np.ndarray:
    sp = -float(matrix[2, 0])
    sp = max(min(sp, 1.0), -1.0)
    pitch = math.asin(sp)
    if abs(sp) < 1.0 - 1e-6:
        roll = math.atan2(float(matrix[2, 1]), float(matrix[2, 2]))
        yaw = math.atan2(float(matrix[1, 0]), float(matrix[0, 0]))
    else:
        yaw = 0.0
        roll = math.atan2(-float(matrix[1, 2]), float(matrix[1, 1]))
    return np.array([roll, pitch, yaw], dtype=np.float64)


def quat_wxyz_slerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    a = np.asarray(q0, dtype=np.float64)
    b = np.asarray(q1, dtype=np.float64)
    a = a / max(np.linalg.norm(a), 1e-12)
    b = b / max(np.linalg.norm(b), 1e-12)
    dot = float(np.dot(a, b))
    if dot < 0.0:
        b = -b
        dot = -dot
    if dot > 0.9995:
        q = (1.0 - alpha) * a + alpha * b
        return q / max(np.linalg.norm(q), 1e-12)
    dot = max(min(dot, 1.0), -1.0)
    theta = math.acos(dot)
    s = math.sin(theta)
    w0 = math.sin((1.0 - alpha) * theta) / s
    w1 = math.sin(alpha * theta) / s
    return w0 * a + w1 * b


def quaternion_angle_error_rad(quat_ref_wxyz: np.ndarray, quat_act_wxyz: np.ndarray) -> float:
    ref = quat_ref_wxyz / max(np.linalg.norm(quat_ref_wxyz), 1e-12)
    act = quat_act_wxyz / max(np.linalg.norm(quat_act_wxyz), 1e-12)
    cos_half = float(np.clip(abs(np.dot(ref, act)), -1.0, 1.0))
    return float(2.0 * math.acos(cos_half))


def euler_xyz_to_T(pos: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    return t_trans(np.asarray(pos, dtype=np.float64).reshape(3)) @ t_rot_rpy(np.asarray(rpy, dtype=np.float64).reshape(3))


def T_to_pose7_xyz_quat_wxyz(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64)
    q = matrix_to_quaternion_wxyz(T[:3, :3])
    return np.concatenate([T[:3, 3].copy(), q], axis=0)


class TrajectorySampler:
    REQUIRED_COLUMNS = (
        "t",
        "left_x",
        "left_y",
        "left_z",
        "left_roll",
        "left_pitch",
        "left_yaw",
        "right_x",
        "right_y",
        "right_z",
        "right_roll",
        "right_pitch",
        "right_yaw",
    )

    def __init__(self, csv_path: Path) -> None:
        self.csv_path = csv_path
        self.timestamps, self.left_traj, self.right_traj = self._load(csv_path)
        self.duration = float(self.timestamps[-1])
        self.sample_count = int(self.timestamps.shape[0])

    def _load(self, csv_path: Path):
        rows = []
        with csv_path.open("r", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            if reader.fieldnames is None:
                raise ValueError(f"Trajectory CSV is empty: {csv_path}")

            missing_columns = [name for name in self.REQUIRED_COLUMNS if name not in reader.fieldnames]
            if missing_columns:
                raise ValueError(f"Trajectory CSV is missing required columns: {missing_columns}")

            for row in reader:
                rows.append(row)

        if len(rows) < 2:
            raise ValueError("Trajectory CSV must contain at least two samples.")

        timestamps = np.array([float(row["t"]) for row in rows], dtype=np.float32)
        if not np.all(np.diff(timestamps) > 0.0):
            raise ValueError("Trajectory timestamps must be strictly increasing.")

        left_traj = np.array(
            [
                [
                    float(row["left_x"]),
                    float(row["left_y"]),
                    float(row["left_z"]),
                    float(row["left_roll"]),
                    float(row["left_pitch"]),
                    float(row["left_yaw"]),
                ]
                for row in rows
            ],
            dtype=np.float32,
        )
        right_traj = np.array(
            [
                [
                    float(row["right_x"]),
                    float(row["right_y"]),
                    float(row["right_z"]),
                    float(row["right_roll"]),
                    float(row["right_pitch"]),
                    float(row["right_yaw"]),
                ]
                for row in rows
            ],
            dtype=np.float32,
        )
        return timestamps, left_traj, right_traj

    @staticmethod
    def _wrap_pi(x: np.ndarray) -> np.ndarray:
        return np.mod(x + math.pi, 2.0 * math.pi) - math.pi

    @classmethod
    def _shortest_arc_interp(cls, a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
        pos = (1.0 - alpha) * a[:3] + alpha * b[:3]
        delta_rpy = cls._wrap_pi(b[3:6] - a[3:6])
        rpy = cls._wrap_pi(a[3:6] + alpha * delta_rpy)
        return np.concatenate([pos, rpy], axis=0)

    def sample(self, elapsed_time: float, loop: bool):
        if loop and self.duration > 0.0:
            elapsed_time = elapsed_time % self.duration
        else:
            elapsed_time = np.clip(elapsed_time, 0.0, self.duration)

        upper_idx = int(np.searchsorted(self.timestamps, elapsed_time, side="right"))
        if upper_idx <= 0:
            return self.left_traj[0].copy(), self.right_traj[0].copy()
        if upper_idx >= self.sample_count:
            return self.left_traj[-1].copy(), self.right_traj[-1].copy()

        lower_idx = upper_idx - 1
        t0 = float(self.timestamps[lower_idx])
        t1 = float(self.timestamps[upper_idx])
        alpha = 0.0 if t1 <= t0 else (elapsed_time - t0) / (t1 - t0)

        left_pose = self._shortest_arc_interp(
            self.left_traj[lower_idx].astype(np.float64),
            self.left_traj[upper_idx].astype(np.float64),
            alpha,
        )
        right_pose = self._shortest_arc_interp(
            self.right_traj[lower_idx].astype(np.float64),
            self.right_traj[upper_idx].astype(np.float64),
            alpha,
        )
        return left_pose.astype(np.float32), right_pose.astype(np.float32)


class TrackingErrorMonitor:
    def __init__(
        self,
        print_hz: float,
        log_csv_path: Path | None,
        warmup_sec: float,
        lag_search_max_sec: float,
        spatial_resample_count: int,
    ):
        self.print_interval = 1.0 / max(print_hz, 1e-6) if print_hz > 0.0 else None
        self.next_print_time = 0.0
        self.warmup_sec = max(0.0, warmup_sec)
        self.lag_search_max_sec = max(0.0, float(lag_search_max_sec))
        self.spatial_resample_count = max(20, int(spatial_resample_count))
        self.warmup_notified = False
        self.count = 0
        self.sum_left_pos = 0.0
        self.sum_right_pos = 0.0
        self.sum_left_rot = 0.0
        self.sum_right_rot = 0.0
        self.sum_left_pos_sq = 0.0
        self.sum_right_pos_sq = 0.0
        self.sum_left_rot_sq = 0.0
        self.sum_right_rot_sq = 0.0
        self.max_left_pos = 0.0
        self.max_right_pos = 0.0
        self.max_left_rot = 0.0
        self.max_right_rot = 0.0
        self.summary_printed = False
        self.sample_times: list[float] = []
        self.left_target_pos_samples: list[np.ndarray] = []
        self.right_target_pos_samples: list[np.ndarray] = []
        self.left_target_quat_samples: list[np.ndarray] = []
        self.right_target_quat_samples: list[np.ndarray] = []
        self.left_actual_pose_samples: list[np.ndarray] = []
        self.right_actual_pose_samples: list[np.ndarray] = []
        self.log_file = None
        self.csv_writer = None
        if log_csv_path is not None:
            log_csv_path.parent.mkdir(parents=True, exist_ok=True)
            self.log_file = log_csv_path.open("w", newline="")
            self.csv_writer = csv.writer(self.log_file)
            self.csv_writer.writerow(
                [
                    "t",
                    "left_pos_err_m",
                    "left_rot_err_deg",
                    "right_pos_err_m",
                    "right_rot_err_deg",
                    "left_target_x",
                    "left_target_y",
                    "left_target_z",
                    "left_actual_x",
                    "left_actual_y",
                    "left_actual_z",
                    "right_target_x",
                    "right_target_y",
                    "right_target_z",
                    "right_actual_x",
                    "right_actual_y",
                    "right_actual_z",
                ]
            )

    def update(
        self,
        elapsed_time: float,
        left_target_euler: np.ndarray,
        right_target_euler: np.ndarray,
        left_actual_pose: np.ndarray,
        right_actual_pose: np.ndarray,
    ) -> None:
        if elapsed_time < self.warmup_sec:
            if not self.warmup_notified:
                self.warmup_notified = True
                print(
                    f"[TrackingError] warmup {self.warmup_sec:.2f}s, skip early transient in stats."
                )
            return

        left_target_pos = left_target_euler[:3]
        right_target_pos = right_target_euler[:3]
        left_target_quat = rpy_to_quaternion_wxyz(left_target_euler[3:6])
        right_target_quat = rpy_to_quaternion_wxyz(right_target_euler[3:6])

        left_pos_err = float(np.linalg.norm(left_target_pos - left_actual_pose[:3]))
        right_pos_err = float(np.linalg.norm(right_target_pos - right_actual_pose[:3]))
        left_rot_err = quaternion_angle_error_rad(left_target_quat, left_actual_pose[3:7])
        right_rot_err = quaternion_angle_error_rad(right_target_quat, right_actual_pose[3:7])

        self.count += 1
        self.sum_left_pos += left_pos_err
        self.sum_right_pos += right_pos_err
        self.sum_left_rot += left_rot_err
        self.sum_right_rot += right_rot_err
        self.sum_left_pos_sq += left_pos_err * left_pos_err
        self.sum_right_pos_sq += right_pos_err * right_pos_err
        self.sum_left_rot_sq += left_rot_err * left_rot_err
        self.sum_right_rot_sq += right_rot_err * right_rot_err
        self.max_left_pos = max(self.max_left_pos, left_pos_err)
        self.max_right_pos = max(self.max_right_pos, right_pos_err)
        self.max_left_rot = max(self.max_left_rot, left_rot_err)
        self.max_right_rot = max(self.max_right_rot, right_rot_err)
        self.sample_times.append(float(elapsed_time))
        self.left_target_pos_samples.append(left_target_pos.astype(np.float64).copy())
        self.right_target_pos_samples.append(right_target_pos.astype(np.float64).copy())
        self.left_target_quat_samples.append(left_target_quat.astype(np.float64).copy())
        self.right_target_quat_samples.append(right_target_quat.astype(np.float64).copy())
        self.left_actual_pose_samples.append(left_actual_pose.astype(np.float64).copy())
        self.right_actual_pose_samples.append(right_actual_pose.astype(np.float64).copy())

        if self.csv_writer is not None:
            self.csv_writer.writerow(
                [
                    f"{elapsed_time:.6f}",
                    f"{left_pos_err:.6f}",
                    f"{math.degrees(left_rot_err):.6f}",
                    f"{right_pos_err:.6f}",
                    f"{math.degrees(right_rot_err):.6f}",
                    *[f"{v:.6f}" for v in left_target_pos],
                    *[f"{v:.6f}" for v in left_actual_pose[:3]],
                    *[f"{v:.6f}" for v in right_target_pos],
                    *[f"{v:.6f}" for v in right_actual_pose[:3]],
                ]
            )

        if self.print_interval is not None and elapsed_time >= self.next_print_time:
            self.next_print_time = elapsed_time + self.print_interval
            print(
                "[TrackingError] "
                f"L_pos={left_pos_err:.4f}m L_rot={math.degrees(left_rot_err):.2f}deg | "
                f"R_pos={right_pos_err:.4f}m R_rot={math.degrees(right_rot_err):.2f}deg"
            )

    def close(self):
        if self.log_file is not None:
            self.log_file.flush()
            self.log_file.close()
            self.log_file = None
            self.csv_writer = None

    def get_summary(self) -> dict[str, float | int | None]:
        if self.count == 0:
            return {
                "sample_count": 0,
                "warmup_sec": self.warmup_sec,
                "lag_search_max_sec": self.lag_search_max_sec,
                "lag_compensated": None,
                "spatial_resample_count": self.spatial_resample_count,
                "spatial_resampled": None,
                "left_pos_mean_m": None,
                "left_pos_rmse_m": None,
                "left_pos_max_m": None,
                "right_pos_mean_m": None,
                "right_pos_rmse_m": None,
                "right_pos_max_m": None,
                "left_rot_mean_deg": None,
                "left_rot_rmse_deg": None,
                "left_rot_max_deg": None,
                "right_rot_mean_deg": None,
                "right_rot_rmse_deg": None,
                "right_rot_max_deg": None,
            }

        left_pos_mean = self.sum_left_pos / self.count
        right_pos_mean = self.sum_right_pos / self.count
        left_rot_mean = self.sum_left_rot / self.count
        right_rot_mean = self.sum_right_rot / self.count
        left_pos_rmse = math.sqrt(self.sum_left_pos_sq / self.count)
        right_pos_rmse = math.sqrt(self.sum_right_pos_sq / self.count)
        left_rot_rmse = math.sqrt(self.sum_left_rot_sq / self.count)
        right_rot_rmse = math.sqrt(self.sum_right_rot_sq / self.count)
        lag_comp = self._compute_lag_compensated_summary()
        spatial = self._compute_spatial_resampled_summary()
        return {
            "sample_count": self.count,
            "warmup_sec": self.warmup_sec,
            "lag_search_max_sec": self.lag_search_max_sec,
            "lag_compensated": lag_comp,
            "spatial_resample_count": self.spatial_resample_count,
            "spatial_resampled": spatial,
            "left_pos_mean_m": left_pos_mean,
            "left_pos_rmse_m": left_pos_rmse,
            "left_pos_max_m": self.max_left_pos,
            "right_pos_mean_m": right_pos_mean,
            "right_pos_rmse_m": right_pos_rmse,
            "right_pos_max_m": self.max_right_pos,
            "left_rot_mean_deg": math.degrees(left_rot_mean),
            "left_rot_rmse_deg": math.degrees(left_rot_rmse),
            "left_rot_max_deg": math.degrees(self.max_left_rot),
            "right_rot_mean_deg": math.degrees(right_rot_mean),
            "right_rot_rmse_deg": math.degrees(right_rot_rmse),
            "right_rot_max_deg": math.degrees(self.max_right_rot),
        }

    @staticmethod
    def _resample_by_arclength(
        pos: np.ndarray, quat_wxyz: np.ndarray, sample_count: int
    ) -> tuple[np.ndarray, np.ndarray]:
        if pos.shape[0] < 2:
            return pos.copy(), quat_wxyz.copy()
        seg = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        s = np.concatenate([np.array([0.0], dtype=np.float64), np.cumsum(seg)])
        if s[-1] < 1e-9:
            idx = np.linspace(0.0, float(pos.shape[0] - 1), sample_count)
            i0 = np.floor(idx).astype(np.int64)
            i1 = np.clip(i0 + 1, 0, pos.shape[0] - 1)
            a = (idx - i0).reshape(-1, 1)
        else:
            st = np.linspace(0.0, float(s[-1]), sample_count)
            i1 = np.searchsorted(s, st, side="right")
            i1 = np.clip(i1, 1, pos.shape[0] - 1)
            i0 = i1 - 1
            denom = np.maximum(s[i1] - s[i0], 1e-12)
            a = ((st - s[i0]) / denom).reshape(-1, 1)
        pr = (1.0 - a) * pos[i0] + a * pos[i1]
        q0 = quat_wxyz[i0]
        q1 = quat_wxyz[i1].copy()
        flip = np.sum(q0 * q1, axis=1) < 0.0
        q1[flip] *= -1.0
        qr = (1.0 - a) * q0 + a * q1
        qr = qr / np.clip(np.linalg.norm(qr, axis=1, keepdims=True), 1e-12, None)
        return pr, qr

    def _compute_spatial_resampled_summary(self) -> dict[str, float | int] | None:
        if len(self.left_actual_pose_samples) < 5:
            return None
        n = int(self.spatial_resample_count)

        ltp = np.stack(self.left_target_pos_samples, axis=0)
        rtp = np.stack(self.right_target_pos_samples, axis=0)
        ltq = np.stack(self.left_target_quat_samples, axis=0)
        rtq = np.stack(self.right_target_quat_samples, axis=0)
        lap = np.stack(self.left_actual_pose_samples, axis=0)
        rap = np.stack(self.right_actual_pose_samples, axis=0)

        def quat_angle_error_batch(ref_wxyz: np.ndarray, act_wxyz: np.ndarray) -> np.ndarray:
            ref = ref_wxyz / np.clip(np.linalg.norm(ref_wxyz, axis=1, keepdims=True), 1e-12, None)
            act = act_wxyz / np.clip(np.linalg.norm(act_wxyz, axis=1, keepdims=True), 1e-12, None)
            cos_half = np.clip(np.abs(np.sum(ref * act, axis=1)), -1.0, 1.0)
            return 2.0 * np.arccos(cos_half)

        ltp_r, ltq_r = self._resample_by_arclength(ltp, ltq, n)
        rtp_r, rtq_r = self._resample_by_arclength(rtp, rtq, n)
        lap_r, laq_r = self._resample_by_arclength(lap[:, :3], lap[:, 3:7], n)
        rap_r, raq_r = self._resample_by_arclength(rap[:, :3], rap[:, 3:7], n)

        l_pos = np.linalg.norm(ltp_r - lap_r, axis=1)
        r_pos = np.linalg.norm(rtp_r - rap_r, axis=1)
        l_rot = quat_angle_error_batch(ltq_r, laq_r)
        r_rot = quat_angle_error_batch(rtq_r, raq_r)
        return {
            "sample_count": n,
            "left_pos_rmse_m": float(np.sqrt(np.mean(l_pos * l_pos))),
            "right_pos_rmse_m": float(np.sqrt(np.mean(r_pos * r_pos))),
            "left_rot_rmse_deg": float(np.rad2deg(np.sqrt(np.mean(l_rot * l_rot)))),
            "right_rot_rmse_deg": float(np.rad2deg(np.sqrt(np.mean(r_rot * r_rot)))),
        }

    def _compute_lag_compensated_summary(self) -> dict[str, float | int] | None:
        if self.lag_search_max_sec <= 0.0 or len(self.sample_times) < 5:
            return None

        times = np.asarray(self.sample_times, dtype=np.float64)
        dts = np.diff(times)
        dts = dts[dts > 1e-9]
        if dts.size == 0:
            return None
        dt = float(np.median(dts))
        max_shift = int(self.lag_search_max_sec / max(dt, 1e-6))
        if max_shift < 1:
            return None

        ltp = np.stack(self.left_target_pos_samples, axis=0)
        rtp = np.stack(self.right_target_pos_samples, axis=0)
        ltq = np.stack(self.left_target_quat_samples, axis=0)
        rtq = np.stack(self.right_target_quat_samples, axis=0)
        lap = np.stack(self.left_actual_pose_samples, axis=0)
        rap = np.stack(self.right_actual_pose_samples, axis=0)

        def quat_angle_error_batch(ref_wxyz: np.ndarray, act_wxyz: np.ndarray) -> np.ndarray:
            ref = ref_wxyz / np.clip(np.linalg.norm(ref_wxyz, axis=1, keepdims=True), 1e-12, None)
            act = act_wxyz / np.clip(np.linalg.norm(act_wxyz, axis=1, keepdims=True), 1e-12, None)
            cos_half = np.clip(np.abs(np.sum(ref * act, axis=1)), -1.0, 1.0)
            return 2.0 * np.arccos(cos_half)

        best = None
        for shift in range(-max_shift, max_shift + 1):
            if shift >= 0:
                tgt_slice = slice(0, len(times) - shift)
                act_slice = slice(shift, len(times))
            else:
                tgt_slice = slice(-shift, len(times))
                act_slice = slice(0, len(times) + shift)
            if (tgt_slice.stop - tgt_slice.start) < 5:
                continue

            l_pos = np.linalg.norm(ltp[tgt_slice] - lap[act_slice, :3], axis=1)
            r_pos = np.linalg.norm(rtp[tgt_slice] - rap[act_slice, :3], axis=1)
            l_rot = quat_angle_error_batch(ltq[tgt_slice], lap[act_slice, 3:7])
            r_rot = quat_angle_error_batch(rtq[tgt_slice], rap[act_slice, 3:7])
            pos_cost = float(np.mean(0.5 * (l_pos * l_pos + r_pos * r_pos)))
            if best is None or pos_cost < best["pos_cost"]:
                best = {
                    "shift": shift,
                    "count": int(l_pos.size),
                    "l_pos_rmse": float(np.sqrt(np.mean(l_pos * l_pos))),
                    "r_pos_rmse": float(np.sqrt(np.mean(r_pos * r_pos))),
                    "l_rot_rmse_rad": float(np.sqrt(np.mean(l_rot * l_rot))),
                    "r_rot_rmse_rad": float(np.sqrt(np.mean(r_rot * r_rot))),
                    "pos_cost": pos_cost,
                }

        if best is None:
            return None
        lag_sec = float(best["shift"] * dt)
        return {
            "sample_count": int(best["count"]),
            "best_lag_sec": lag_sec,
            "best_lag_samples": int(best["shift"]),
            "left_pos_rmse_m": float(best["l_pos_rmse"]),
            "right_pos_rmse_m": float(best["r_pos_rmse"]),
            "left_rot_rmse_deg": float(math.degrees(best["l_rot_rmse_rad"])),
            "right_rot_rmse_deg": float(math.degrees(best["r_rot_rmse_rad"])),
        }

    def print_summary(self):
        if self.summary_printed:
            return
        self.summary_printed = True

        summary = self.get_summary()
        if summary["sample_count"] == 0:
            print(
                "Tracking error summary: no samples collected "
                f"(warmup_sec={self.warmup_sec:.2f})."
            )
            return

        print(
            "Tracking error summary: "
            f"L_pos(mean/rmse/max)={summary['left_pos_mean_m']:.4f}/{summary['left_pos_rmse_m']:.4f}/{summary['left_pos_max_m']:.4f} m, "
            f"R_pos(mean/rmse/max)={summary['right_pos_mean_m']:.4f}/{summary['right_pos_rmse_m']:.4f}/{summary['right_pos_max_m']:.4f} m, "
            f"L_rot(mean/rmse/max)={summary['left_rot_mean_deg']:.2f}/{summary['left_rot_rmse_deg']:.2f}/{summary['left_rot_max_deg']:.2f} deg, "
            f"R_rot(mean/rmse/max)={summary['right_rot_mean_deg']:.2f}/{summary['right_rot_rmse_deg']:.2f}/{summary['right_rot_max_deg']:.2f} deg."
        )
        lag_comp = summary.get("lag_compensated")
        if lag_comp is not None:
            print(
                "[TrackingError LagComp] "
                f"best_lag={lag_comp['best_lag_sec']:.3f}s, "
                f"L_pos_rmse={lag_comp['left_pos_rmse_m']:.4f}m, "
                f"R_pos_rmse={lag_comp['right_pos_rmse_m']:.4f}m, "
                f"L_rot_rmse={lag_comp['left_rot_rmse_deg']:.2f}deg, "
                f"R_rot_rmse={lag_comp['right_rot_rmse_deg']:.2f}deg."
            )
        spatial = summary.get("spatial_resampled")
        if spatial is not None:
            print(
                "[TrackingError Spatial] "
                f"L_pos_rmse={spatial['left_pos_rmse_m']:.4f}m, "
                f"R_pos_rmse={spatial['right_pos_rmse_m']:.4f}m, "
                f"L_rot_rmse={spatial['left_rot_rmse_deg']:.2f}deg, "
                f"R_rot_rmse={spatial['right_rot_rmse_deg']:.2f}deg."
            )


class CommandSmoothnessMonitor:
    def __init__(self, print_hz: float, warmup_sec: float, fixed_dt: float):
        self.print_interval = 1.0 / max(print_hz, 1e-6) if print_hz > 0.0 else None
        self.next_print_time = 0.0
        self.warmup_sec = max(0.0, warmup_sec)
        self.fixed_dt = max(1e-6, float(fixed_dt))

        self.prev_joint_q = None
        self.prev_delta_q = None
        self.summary_printed = False

        self.count = 0
        self.max_delta_q_abs = 0.0
        self.sum_delta_q_sq = 0.0
        self.sum_delta_q_vec_sq = 0.0
        self.max_delta_q_vec_norm = 0.0
        self.max_delta_dq_abs = 0.0
        self.sum_delta_dq_sq = 0.0
        self.delta_dq_count = 0
        self.delta_q_abs_samples: list[float] = []

    def update(self, elapsed_time: float, joint_target_q: np.ndarray) -> None:
        joint_target_q = joint_target_q.astype(np.float64)

        if self.prev_joint_q is None:
            self.prev_joint_q = joint_target_q.copy()
            return

        delta_q = joint_target_q - self.prev_joint_q
        delta_q_abs = float(np.max(np.abs(delta_q)))
        delta_q_vec_norm = float(np.linalg.norm(delta_q))

        delta_dq_abs = None
        if self.prev_delta_q is not None:
            delta_dq = (delta_q - self.prev_delta_q) / (self.fixed_dt * self.fixed_dt)
            delta_dq_abs = float(np.max(np.abs(delta_dq)))
            delta_dq_vec_norm = float(np.linalg.norm(delta_dq))
        else:
            delta_dq_vec_norm = 0.0

        if elapsed_time >= self.warmup_sec:
            self.count += 1
            self.max_delta_q_abs = max(self.max_delta_q_abs, delta_q_abs)
            self.max_delta_q_vec_norm = max(self.max_delta_q_vec_norm, delta_q_vec_norm)
            self.sum_delta_q_sq += delta_q_abs * delta_q_abs
            self.sum_delta_q_vec_sq += delta_q_vec_norm * delta_q_vec_norm
            self.delta_q_abs_samples.append(delta_q_abs)

            if delta_dq_abs is not None:
                self.max_delta_dq_abs = max(self.max_delta_dq_abs, delta_dq_abs)
                self.sum_delta_dq_sq += delta_dq_vec_norm * delta_dq_vec_norm
                self.delta_dq_count += 1

            if self.print_interval is not None and elapsed_time >= self.next_print_time:
                self.next_print_time = elapsed_time + self.print_interval
                print(
                    "[Smoothness] "
                    f"max|dq|={delta_q_abs:.4f} rad, "
                    f"||dq||={delta_q_vec_norm:.4f}, "
                    f"max|ddq|={0.0 if delta_dq_abs is None else delta_dq_abs:.3f} rad/s^2"
                )

        self.prev_joint_q = joint_target_q.copy()
        self.prev_delta_q = delta_q.copy()

    def print_summary(self) -> None:
        if self.summary_printed:
            return
        self.summary_printed = True

        summary = self.get_summary()
        if summary["sample_count"] == 0:
            print("Smoothness summary: no valid samples " f"(warmup_sec={self.warmup_sec:.2f}).")
            return

        print(
            "Smoothness summary (lower is smoother): "
            f"max|dq|={summary['max_delta_q_abs']:.4f} rad, "
            f"p95|dq|={summary['p95_delta_q_abs']:.4f} rad, "
            f"rms|dq|={summary['rms_delta_q_abs']:.4f} rad, "
            f"rms||dq||={summary['rms_delta_q_vec_norm']:.4f}, "
            f"max|ddq|={summary['max_delta_dq_abs']:.2f} rad/s^2, "
            f"rms||ddq||={summary['rms_delta_dq_vec_norm']:.2f} rad/s^2."
        )

    def get_summary(self) -> dict[str, float | int | None]:
        if self.count == 0:
            return {
                "sample_count": 0,
                "warmup_sec": self.warmup_sec,
                "max_delta_q_abs": None,
                "p95_delta_q_abs": None,
                "rms_delta_q_abs": None,
                "max_delta_q_vec_norm": None,
                "rms_delta_q_vec_norm": None,
                "max_delta_dq_abs": None,
                "rms_delta_dq_vec_norm": None,
            }

        rms_delta_q_abs = math.sqrt(self.sum_delta_q_sq / self.count)
        rms_delta_q_vec_norm = math.sqrt(self.sum_delta_q_vec_sq / self.count)
        if self.delta_dq_count > 0:
            rms_delta_dq_vec_norm = math.sqrt(self.sum_delta_dq_sq / self.delta_dq_count)
            max_delta_dq_abs: float | None = self.max_delta_dq_abs
        else:
            rms_delta_dq_vec_norm = None
            max_delta_dq_abs = None
        p95_delta_q_abs = float(np.percentile(np.array(self.delta_q_abs_samples, dtype=np.float64), 95))

        return {
            "sample_count": self.count,
            "warmup_sec": self.warmup_sec,
            "max_delta_q_abs": self.max_delta_q_abs,
            "p95_delta_q_abs": p95_delta_q_abs,
            "rms_delta_q_abs": rms_delta_q_abs,
            "max_delta_q_vec_norm": self.max_delta_q_vec_norm,
            "rms_delta_q_vec_norm": rms_delta_q_vec_norm,
            "max_delta_dq_abs": max_delta_dq_abs,
            "rms_delta_dq_vec_norm": rms_delta_dq_vec_norm,
        }
