# kalman_tracker.py
import numpy as np
import logging

logger = logging.getLogger(__name__)


class KalmanPointTracker:
    """
    一个简单的卡尔曼滤波器，用于跟踪二维点的位置和速度。
    它假设一个近似恒定速度模型，并通过过程噪声来解释未建模的加速度或机动。
    """

    def __init__(self, initial_pos, dt=1.0, process_noise_std=1.0, measurement_noise_std=10.0):
        """
        初始化卡尔曼滤波器。

        参数:
            initial_pos (tuple): (x, y) 被跟踪点的初始位置。
            dt (float): 帧之间的时间步长 (通常对于逐帧处理为1.0)。
            process_noise_std (float): 过程噪声的标准差。
                                       较高的值允许滤波器适应更不稳定的运动。
            measurement_noise_std (float): 测量噪声的标准差。
                                           较高的值表示对输入测量的信任度较低。
        """
        self.dt = dt  # 时间步长

        # 状态向量 [x_pos, y_pos, x_vel, y_vel]^T
        self.x = np.array([initial_pos[0], initial_pos[1], 0, 0], dtype=np.float64)

        # 状态转移矩阵 A (模拟状态在没有外部输入的情况下如何演变)
        # 假设恒定速度: x_k = x_{k-1} + vx * dt, y_k = y_{k-1} + vy * dt
        self.A = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)

        # 测量矩阵 H (将状态映射到测量空间)
        # 我们只直接测量位置 (x, y)。
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)

        # 状态协方差矩阵 P (状态估计的不确定性)
        # 初始化时速度的不确定性较高。
        self.P = np.diag([20.0, 20.0, 100.0, 100.0])

        # 过程噪声协方差矩阵 Q (来自运动模型本身的不确定性)
        # 表示未建模的加速度或速度变化。
        self.Q = np.diag([
            (dt ** 4) / 4, (dt ** 4) / 4,  # 位置方差分量
            (dt ** 2), (dt ** 2)  # 速度方差分量
        ]) * process_noise_std ** 2  # sigma_w^2 是 process_noise_std^2

        # 测量噪声协方差矩阵 R (测量本身的不确定性/噪声)
        self.R = np.eye(2, dtype=np.float64) * (measurement_noise_std ** 2)

        self.age = 0  # 此跟踪器已存在的帧数（预测步骤）
        self.time_since_update = 0  # 自上次成功测量更新以来的帧数

    def predict(self):
        """
        预测下一时间步的状态和协方差。
        应在尝试更新之前每帧调用一次。
        """
        self.x = self.A @ self.x  # 预测状态: x_k|k-1 = A * x_k-1|k-1
        self.P = self.A @ self.P @ self.A.T + self.Q  # 预测协方差: P_k|k-1 = A*P_k-1|k-1*A^T + Q
        self.age += 1
        self.time_since_update += 1
        return self.x[:2]  # 返回预测的位置 (x, y)

    def update(self, z_measurement):
        """
        用新的测量值更新状态估计。

        参数:
            z_measurement (tuple or None): (x_measured, y_measured) 来自传感器 (例如NCC)。
                                           如果为None，则不执行更新。
        """
        if z_measurement is None: return

        z = np.array(z_measurement, dtype=np.float64).reshape(2, 1)  # 测量向量

        # 测量残差 (innovation): y = z - H*x_pred
        # 确保 x 也被视为列向量进行矩阵数学运算（如果当前是1D）
        y_tilde = z - self.H @ self.x.reshape(4, 1)

        # 残差 (innovation) 协方差: S = H*P_pred*H^T + R
        S = self.H @ self.P @ self.H.T + self.R

        # 卡尔曼增益: K = P_pred*H^T*inv(S)
        try:
            # 使用 solve 计算 K = (S.T \ (H @ P.T).T ).T，等效于 P @ H.T @ inv(S)
            # 这在数值上可能更稳定。
            K = np.linalg.solve(S.T, self.H @ self.P.T).T
        except np.linalg.LinAlgError:
            logger.warning("卡尔曼滤波器: 更新期间 S 矩阵奇异或病态。跳过更新。")
            return

        # 更新状态估计: x_updated = x_pred + K*y_tilde
        self.x = self.x + (K @ y_tilde).flatten()  # 将 K@y_tilde 展平回1D数组

        # 更新估计协方差: P_updated = (I - K*H)*P_pred
        I = np.eye(self.H.shape[1])  # 4x4 单位矩阵
        self.P = (I - K @ self.H) @ self.P

        self.time_since_update = 0

    def get_state(self):
        """返回状态向量中当前估计的位置 (x, y)。"""
        return self.x[:2]