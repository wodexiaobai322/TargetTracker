# myTask1.py
import os
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import shutil
import logging

# 尝试导入必要的自定义模块
try:
    from correlation_matcher_fft import match_template_ncc_fft as match_template_ncc_custom
except ImportError:
    # 如果导入失败，配置基本日志并退出，因为这是核心依赖
    logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.critical("致命错误: 无法从 'correlation_matcher_fft.py' 导入 'match_template_ncc_fft'。请确保已安装 scipy。")
    exit()

try:
    from kalman_tracker import KalmanPointTracker
except ImportError:
    logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.critical("致命错误: 无法从 'kalman_tracker.py' 导入 KalmanPointTracker。")
    exit()

LOGGING_LEVEL = logging.INFO
logging.basicConfig(level=LOGGING_LEVEL, format='%(asctime)s - %(levelname)s - [%(module)s.%(funcName)s] - %(message)s')
logger = logging.getLogger(__name__)

# 尝试加载默认字体用于绘图
try:
    common_font = ImageFont.load_default()
except IOError:
    common_font = ImageFont.load_default()  # 更稳健的备选方案


class VideoTracker:
    """
    使用基于FFT的归一化互相关 (NCC)、卡尔曼滤波器进行状态预测与平滑，
    以及多模板匹配，在视频中跟踪目标。
    """

    def __init__(self, template_paths, temp_dir="temp_frames",
                 # NCC 相关阈值
                 ncc_threshold=0.22,  # NCC峰值被视为KF候选者的最低分数
                 reliable_ncc_threshold=0.38,  # 通过门限的检测需要达到此分数才被视为“强命中”
                 re_acquire_ncc_threshold=0.50,  # KF丢失后重新初始化所需的较高分数
                 # 卡尔曼滤波器调优参数
                 kf_measurement_noise_std=18.0,  # NCC中心点的预期误差 (小图坐标系下的像素)
                 kf_process_noise_std=3.5,  # KF运动模型的灵活性 (小图坐标系下的像素/帧^2)
                 detection_gate_factor=5.0,  # 用于门限检测的标准差倍数 (基于测量噪声)
                 # 跟踪器生命周期和行为
                 max_consecutive_misses=10,  # KF在重置前可以连续“丢失”目标（没有可靠命中）的帧数
                 min_hits_to_activate=2,  # KF输出被完全信任所需的“强命中”次数
                 re_acquire_timeout_frames=25,  # KF丢失后，尝试重新捕获信号的帧数
                 # 帧处理参数
                 frame_downsample_factor=2,  # 帧下采样因子，加快处理速度
                 search_roi_padding_factor=2.0  # KF预测ROI周围的填充因子
                 ):

        self.frame_downsample_factor = max(1, int(frame_downsample_factor))
        self.templates_processed = []  # 存储 {'array', 'orig_pil_size', 'id'}
        logger.info("加载并预处理模板...")
        for i, p in enumerate(template_paths):
            try:
                np_arr, orig_pil_size = self.load_template(p)  # 注意：load_template现在不接收template_idx
                self.templates_processed.append({'array': np_arr, 'orig_pil_size': orig_pil_size, 'id': i})
            except Exception as e:
                logger.error(f"跳过模板 {p}，加载错误: {e}", exc_info=True)
        if not self.templates_processed: raise ValueError("未能成功加载任何模板。")
        for t_data in self.templates_processed:
            logger.info(
                f"  模板 {t_data['id']}: 处理后形状 {t_data['array'].shape}, 原始PIL尺寸 {t_data['orig_pil_size']}")

        self.temp_dir = temp_dir
        self.trajectory = []  # 存储轨迹点 (原始坐标)
        # 存储初始化参数
        self.ncc_threshold = ncc_threshold
        self.reliable_ncc_threshold = reliable_ncc_threshold
        self.re_acquire_ncc_threshold = re_acquire_ncc_threshold
        self.kf_measurement_noise_std = kf_measurement_noise_std
        self.kf_process_noise_std = kf_process_noise_std
        self.detection_gate_factor = detection_gate_factor
        self.search_roi_padding_factor = search_roi_padding_factor
        self.max_consecutive_misses = max_consecutive_misses
        self.min_hits_to_activate = min_hits_to_activate
        self.re_acquire_timeout_frames = re_acquire_timeout_frames

        # 内部状态变量
        self.kalman_filter = None
        self.active_template_data = None  # 当前最佳匹配的模板数据
        self.consecutive_misses = 0
        self.hits = 0  # 可靠命中的计数
        self.frames_since_kf_lost = 0  # KF丢失后的帧计数，用于重新捕获超时
        self.current_frame_idx = 0
        self.last_successful_match_score = -1.0  # 上一个被接受的NCC测量的分数

    def load_template(self, path):
        """加载模板图像，转换为灰度，并根据需要进行尺寸缩放。"""
        img_pil = Image.open(path).convert('L')
        original_pil_size = img_pil.size
        processed_pil = img_pil  # 默认为原始PIL图像
        if self.f_dsf() > 1:  # f_dsf() 是 frame_downsample_factor 的辅助函数
            new_w = original_pil_size[0] // self.f_dsf()
            new_h = original_pil_size[1] // self.f_dsf()
            if new_w > 0 and new_h > 0:  # 确保缩放后尺寸有效
                try:
                    processed_pil = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)  # Pillow >= 9.0
                except AttributeError:
                    processed_pil = img_pil.resize((new_w, new_h), Image.LANCZOS)  # Pillow < 9.0
            else:
                logger.warning(f"模板 {path} 缩放后尺寸为零。将使用原始尺寸。")
        img_np = np.array(processed_pil, dtype=np.uint8)
        if img_np.size < 1: raise ValueError(f"模板 {path} 处理后无效 (形状:{img_np.shape})。")
        return img_np, original_pil_size

    def _get_search_region(self, frame_gray_np_shape_small):
        """根据卡尔曼滤波器的预测（如果可用）确定NCC匹配的搜索区域 (ROI)。"""
        fh_small, fw_small = frame_gray_np_shape_small
        if self.kalman_filter:
            predicted_center_small = self.kalman_filter.predict()  # KF状态在此处推进

            # 使用活动模板的尺寸确定ROI大小，如果活动模板不存在则使用第一个模板作为备选
            tpl_h_s, tpl_w_s = (self.active_template_data['array'].shape if self.active_template_data
                                else self.templates_processed[0]['array'].shape if self.templates_processed else (
                20, 20))  # 最后的(20,20)是极端备选

            roi_w = max(int(tpl_w_s * self.search_roi_padding_factor), tpl_w_s + 20)  # 保证最小填充
            roi_h = max(int(tpl_h_s * self.search_roi_padding_factor), tpl_h_s + 20)

            roi_x1_ideal = predicted_center_small[0] - roi_w / 2.0
            roi_y1_ideal = predicted_center_small[1] - roi_h / 2.0
            # 将ROI裁剪到帧边界内
            roi_y1 = max(0, int(round(roi_y1_ideal)))
            roi_x1 = max(0, int(round(roi_x1_ideal)))
            roi_y2 = min(fh_small, int(round(roi_y1_ideal + roi_h)))
            roi_x2 = min(fw_small, int(round(roi_x1_ideal + roi_w)))

            if roi_x2 > roi_x1 and roi_y2 > roi_y1:  # 有效的ROI
                return (roi_y1, roi_x1), (roi_y2 - roi_y1, roi_x2 - roi_x1)  # 返回 offset_yx, size_hw
            else:
                logger.debug(f"F{self.current_frame_idx:04d}: KF预测的ROI裁剪后无效。将使用全帧搜索。")
        return (0, 0), frame_gray_np_shape_small  # 默认：全帧搜索

    def process_frame(self, frame_pil_small):
        """处理单个视频帧以检测和跟踪目标。"""
        self.current_frame_idx += 1
        frame_id_str = f"F{self.current_frame_idx:04d}"
        logger.debug(f"{frame_id_str}: 处理帧 (小图尺寸: {frame_pil_small.size})")
        frame_gray_np = np.array(frame_pil_small.convert('L'), dtype=np.uint8)

        # 判断是否处于重新捕获阶段 (KF丢失，正在搜索强信号)
        is_re_acquiring_phase = (self.kalman_filter is None) and (self.frames_since_kf_lost > 0)

        search_offset_yx, search_size_hw = self._get_search_region(frame_gray_np.shape)
        search_y1, search_x1 = search_offset_yx
        search_h, search_w = search_size_hw
        search_image_np = frame_gray_np[search_y1: search_y1 + search_h, search_x1: search_x1 + search_w]

        if search_image_np.size < 25:  # 例如，小于5x5像素的搜索区域太小
            logger.debug(f"{frame_id_str}: 搜索图像区域过小 ({search_image_np.shape})。")
            if self.kalman_filter:
                self.consecutive_misses += 1
            elif is_re_acquiring_phase:
                self.frames_since_kf_lost += 1  # 重新捕获阶段也计数“丢失”
            return None, None

        # --- NCC匹配: 尝试所有模板以找到最佳候选者 ---
        best_match_score, best_s_center, best_tpl_data = -np.inf, None, None
        for t_data in self.templates_processed:
            tpl_np = t_data['array']
            tpl_h_s, tpl_w_s = tpl_np.shape
            if tpl_h_s > search_h or tpl_w_s > search_w: continue  # 模板大于搜索区域

            ncc_map = match_template_ncc_custom(search_image_np, tpl_np)
            if ncc_map is None or ncc_map.size == 0: continue

            current_max_score = np.max(ncc_map)
            if current_max_score > best_match_score:
                best_match_score = current_max_score
                best_tpl_data = t_data
                my_roi, mx_roi = np.unravel_index(np.argmax(ncc_map), ncc_map.shape)
                # 将ROI局部匹配坐标转换为小图全局坐标
                best_s_center = ((mx_roi + tpl_w_s / 2.0) + search_x1, (my_roi + tpl_h_s / 2.0) + search_y1)

        logger.debug(f"{frame_id_str}: NCC最佳总分: {best_match_score:.3f}")
        self.last_successful_match_score = -1.0  # 为当前帧重置

        detected_center_sm, detected_size_orig, is_reliable_measurement = None, None, False

        # 根据当前跟踪状态确定NCC接受阈值
        current_acceptance_threshold = self.re_acquire_ncc_threshold if is_re_acquiring_phase else self.ncc_threshold

        if best_match_score >= current_acceptance_threshold and best_s_center is not None:
            self.last_successful_match_score = best_match_score
            detected_center_sm = best_s_center
            detected_size_orig = best_tpl_data['orig_pil_size']
            if best_match_score >= self.reliable_ncc_threshold:
                is_reliable_measurement = True  # 是否为“强”命中

            logger.debug(
                f"{frame_id_str}: NCC候选分数 {best_match_score:.3f} vs 接受阈值 {current_acceptance_threshold:.2f}。是否可靠: {is_reliable_measurement}")

            if self.kalman_filter is None:  # 尝试初始化或重新捕获KF
                can_initialize_kf = (is_re_acquiring_phase and best_match_score >= self.re_acquire_ncc_threshold) or \
                                    (not is_re_acquiring_phase and is_reliable_measurement)
                if can_initialize_kf:
                    init_reason = "重新捕获" if is_re_acquiring_phase else "初始可靠"
                    logger.info(
                        f"{frame_id_str}: 初始化KF ({init_reason}) 于小图坐标 {detected_center_sm} (模板 {best_tpl_data['id']})")
                    self.kalman_filter = KalmanPointTracker(detected_center_sm, 1.0, self.kf_process_noise_std,
                                                            self.kf_measurement_noise_std)
                    self.active_template_data = best_tpl_data
                    self.hits = 1  # 第一个可靠命中
                    self.consecutive_misses = 0
                    self.frames_since_kf_lost = 0  # 成功 (重新)初始化
                else:  # 分数不足以初始化/重新捕获
                    logger.info(
                        f"{frame_id_str}: NCC分数 {best_match_score:.3f} 不足以初始化/重新捕获KF (状态: {'重新捕获' if is_re_acquiring_phase else '初始'})。")
                    detected_center_sm = None  # 不足以初始化，视为未检测到以供输出
                    if is_re_acquiring_phase: self.frames_since_kf_lost += 1  # 如果在重新捕获阶段失败，则增加超时计数

            elif detected_center_sm:  # KF存在，并且当前帧有NCC候选检测
                predicted_state_small = self.kalman_filter.get_state()  # 此状态是KF predict()之后的状态
                dx = detected_center_sm[0] - predicted_state_small[0]
                dy = detected_center_sm[1] - predicted_state_small[1]
                distance_sq = dx ** 2 + dy ** 2
                gate_radius_sq = (self.detection_gate_factor * self.kf_measurement_noise_std) ** 2

                # 可选：更详细的门限检查日志
                # logger.debug(f"{frame_id_str}: 门限检查: 距离平方 {distance_sq:.1f}, 门限平方 {gate_radius_sq:.1f}")

                if distance_sq <= gate_radius_sq:  # NCC检测在KF的门限内
                    logger.debug(f"{frame_id_str}: NCC通过门限。")
                    self.kalman_filter.update(detected_center_sm)
                    self.active_template_data = best_tpl_data  # 如果更好的模板匹配成功，则更新活动模板
                    if is_reliable_measurement: self.hits += 1  # 只有可靠的测量才能使KF成熟
                    self.consecutive_misses = 0
                    self.frames_since_kf_lost = 0  # 如果KF处于活动状态并已更新，则重置
                else:  # NCC检测未通过KF的门限 (距离预测太远)
                    logger.debug(
                        f"{frame_id_str}: NCC未通过门限。预测{predicted_state_small} 测量{detected_center_sm}。视为丢失。")  # 使用 debug 级别
                    detected_center_sm = None  # 此测量对输出无效
                    self.consecutive_misses += 1
                    # self.hits = 0 # 可选：在门限失败时重置命中数，或让它们衰减/以不同方式管理
        else:  # 没有NCC匹配高于 current_acceptance_threshold
            if self.kalman_filter:
                self.consecutive_misses += 1
            elif is_re_acquiring_phase:
                self.frames_since_kf_lost += 1
            logger.debug(f"{frame_id_str}: 没有NCC匹配高于当前阈值 {current_acceptance_threshold:.2f}。")
            detected_center_sm = None  # 没有有效的NCC检测

        # --- 输出决策逻辑 ---
        out_c_orig, out_s_orig = None, None  # 初始化输出为None
        if self.kalman_filter:  # KF处于活动状态
            if self.consecutive_misses >= self.max_consecutive_misses:
                logger.info(
                    f"{frame_id_str}: 目标丢失 (KF连续丢失:{self.consecutive_misses})。重置KF。进入重新捕获阶段。")
                self.kalman_filter, self.active_template_data, self.hits, self.consecutive_misses = None, None, 0, 0
                self.frames_since_kf_lost = 1  # 立即开始重新捕获超时计数
            # 如果KF成熟 或 正在滑行（有连续丢失但尚未丢失），则输出KF状态
            elif self.hits >= self.min_hits_to_activate or self.consecutive_misses > 0:
                kf_state_sm = self.kalman_filter.get_state()
                out_c_orig = (int(round(kf_state_sm[0] * self.f_dsf())), int(round(kf_state_sm[1] * self.f_dsf())))
                out_s_orig = self.active_template_data['orig_pil_size'] if self.active_template_data else (
                    30, 30)  # 备选尺寸
            # KF存在但不成熟，但当前检测可靠并通过了门限
            elif detected_center_sm and is_reliable_measurement:
                out_c_orig = (
                    int(round(detected_center_sm[0] * self.f_dsf())), int(round(detected_center_sm[1] * self.f_dsf())))
                out_s_orig = detected_size_orig
            # 否则 (KF存在，不成熟，当前检测不可靠/未通过门限): out_c_orig/out_s_orig 保持 None (KF静默滑行)

        elif is_re_acquiring_phase:  # 没有KF，已丢失，正在尝试重新捕获
            if self.frames_since_kf_lost >= self.re_acquire_timeout_frames:
                logger.info(f"{frame_id_str}: 重新捕获超时 ({self.frames_since_kf_lost} 帧)。目标确实丢失。")
                self.frames_since_kf_lost = 0  # 为下一次可能的丢失重置超时计数器
            # 在重新捕获期间不输出，除非KF刚刚被成功重新初始化 (这种情况将由上面的 'if self.kalman_filter:' 块处理)

        elif detected_center_sm and is_reliable_measurement:  # 初始状态 (无KF，非重新捕获)，首次可靠检测
            # 这种情况理想情况下应导致KF初始化。如果此时KF仍为None，则为逻辑边缘情况或第一帧。
            logger.debug(f"{frame_id_str}: 输出初始可靠的原始NCC (KF应该已初始化)。")
            out_c_orig = (
                int(round(detected_center_sm[0] * self.f_dsf())), int(round(detected_center_sm[1] * self.f_dsf())))
            out_s_orig = detected_size_orig

        return out_c_orig, out_s_orig

    def f_dsf(self):
        """辅助函数：返回帧下采样因子。"""
        return self.frame_downsample_factor

    def lsms(self):
        """辅助函数：返回上一个成功的匹配分数。"""
        return self.last_successful_match_score

    def draw_on_frame_scaled(self, frame_pil_rgb_small, center_pos_orig, box_size_orig, frame_id_str=""):
        """在（小）帧上绘制跟踪信息（框、轨迹、状态）。"""
        draw = ImageDraw.Draw(frame_pil_rgb_small)
        sm_w, sm_h = frame_pil_rgb_small.size

        # 绘制轨迹
        if center_pos_orig: self.trajectory.append(center_pos_orig)  # 添加当前输出点
        if len(self.trajectory) > 70: self.trajectory.pop(0)  # 限制轨迹长度
        if len(self.trajectory) > 1:
            traj_sm = []  # 存储缩放后的轨迹点
            for p_orig in self.trajectory:
                p_sm_x = np.clip(int(round(p_orig[0] / self.f_dsf())), 0, sm_w - 1)
                p_sm_y = np.clip(int(round(p_orig[1] / self.f_dsf())), 0, sm_h - 1)
                traj_sm.append((p_sm_x, p_sm_y))
            if len(traj_sm) > 1: draw.line(traj_sm, fill='lime', width=2)

        # 确定状态文本和框颜色
        status_text, box_color = "状态: 搜索中", 'yellow'
        is_re_acq_draw = (self.kalman_filter is None) and (self.frames_since_kf_lost > 0)

        if self.kalman_filter:  # KF处于活动状态
            if self.consecutive_misses == 0 and self.hits >= self.min_hits_to_activate:
                status_text = f"跟踪中 (H:{self.hits} S:{self.lsms():.2f})"
                box_color = 'lime'  # 石灰绿表示稳定跟踪
            elif self.consecutive_misses > 0:
                status_text = f"滑行中 (M:{self.consecutive_misses} H:{self.hits})"
                box_color = 'orange'  # 橙色表示滑行
            else:  # KF存在，命中数 < min_hits_to_activate，无丢失
                status_text = f"捕获中 (H:{self.hits} S:{self.lsms():.2f})"
                box_color = 'cyan'  # 青色表示正在捕获
        elif is_re_acq_draw:  # 无KF，处于重新捕获阶段
            status_text = f"重新捕获中 (丢F:{self.frames_since_kf_lost})"
            box_color = 'violet'  # 紫罗兰色表示重新捕获
            # 如果center_pos_orig不为None，意味着KF在本帧刚刚被成功重新初始化
            if center_pos_orig:
                status_text = f"重新锁定 (S:{self.lsms():.2f})"  # KF刚刚重新建立
                box_color = 'magenta'  # 洋红色表示成功重新锁定
        elif center_pos_orig:  # 无KF，非重新捕获，但本帧有初始输出
            status_text = f"初始锁定 (S:{self.lsms():.2f})"
            box_color = 'magenta'

        # 如果有目标位置，则绘制边界框和中心点
        if center_pos_orig and box_size_orig:
            s_cx = int(round(center_pos_orig[0] / self.f_dsf()))  # 缩放后的中心x
            s_cy = int(round(center_pos_orig[1] / self.f_dsf()))  # 缩放后的中心y
            s_bw = int(round(box_size_orig[0] / self.f_dsf()))  # 缩放后的框宽度
            s_bh = int(round(box_size_orig[1] / self.f_dsf()))  # 缩放后的框高度

            s_l = max(0, s_cx - s_bw // 2)  # 左边界
            s_t = max(0, s_cy - s_bh // 2)  # 上边界
            # 根据左/上边界和宽度/高度计算右/下边界，以避免累积舍入误差
            s_r = min(sm_w, s_l + s_bw)
            s_b = min(sm_h, s_t + s_bh)

            if s_r > s_l and s_b > s_t:  # 确保框尺寸有效
                # PIL矩形的第二个点(x1,y1)是排除在外的：绘制到x1-1, y1-1
                draw.rectangle([s_l, s_t, s_r - 1, s_b - 1], outline=box_color, width=2)

            if 0 <= s_cx < sm_w and 0 <= s_cy < sm_h:  # 如果中心点在边界内，则绘制中心点
                draw.ellipse([(s_cx - 3, s_cy - 3), (s_cx + 3, s_cy + 3)], fill=box_color, outline='white')

            draw.text((5, 5), f"中心(原):({center_pos_orig[0]},{center_pos_orig[1]})", fill=box_color, font=common_font)
        else:  # 没有有效的center_pos_orig用于绘制
            # 仅当不处于重新捕获阶段 或 重新捕获已超时时，才显示“目标丢失”
            if not is_re_acq_draw or (is_re_acq_draw and self.frames_since_kf_lost >= self.re_acquire_timeout_frames):
                status_text = "状态: 目标丢失"
                box_color = "gray"  # 灰色表示丢失

        draw.text((5, 20), status_text, fill=box_color, font=common_font)  # 绘制状态文本
        if self.active_template_data:  # 如果有活动模板，显示其ID
            draw.text((5, 35), f"活动模板ID:{self.active_template_data['id']}", fill='white', font=common_font)
        return frame_pil_rgb_small

    def _ffmpeg_run(self, cmd_list, stage_name="ffmpeg_process"):
        """辅助函数：运行FFmpeg命令并记录stderr。"""
        logger.info(f"FFmpeg {stage_name} 命令: {' '.join(cmd_list)}")
        try:
            p = subprocess.run(cmd_list, check=True, capture_output=True, text=True, encoding='utf-8')
            if p.stderr: logger.warning(f"FFmpeg stderr ({stage_name}): {p.stderr.strip()}")
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg {stage_name} 错误 (返回码 {e.returncode}): {e.stderr.strip()}")
            raise RuntimeError(e)

    def _extract_frames(self, input_video_path):
        """使用FFmpeg从视频中提取帧并进行下采样。"""
        input_frames_dir = os.path.join(self.temp_dir, "input_frames")
        os.makedirs(input_frames_dir, exist_ok=True)
        scale_val = f'iw/{self.f_dsf()}:ih/{self.f_dsf()}'
        cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', input_video_path]
        if self.f_dsf() != 1: cmd.extend(['-vf', f'scale={scale_val}'])  # 如果需要下采样
        cmd.extend(['-vsync', '0', '-qscale:v', '2', os.path.join(input_frames_dir, 'frame_%05d.png')])  # 高质量PNG输出
        self._ffmpeg_run(cmd, "extract")
        return input_frames_dir

    def _compile_video(self, processed_frames_dir, output_video_path, fps=30.0):
        """使用FFmpeg将处理后的帧合成为视频。"""
        cmd = ['ffmpeg', '-y', '-framerate', str(fps), '-i', os.path.join(processed_frames_dir, 'frame_%05d.png'),
               '-c:v', 'libx264', '-preset', 'medium', '-pix_fmt', 'yuv420p', output_video_path]  # 标准H.264输出
        self._ffmpeg_run(cmd, "compile")

    def process_video(self, input_video_path, output_video_path):
        """主要的视频处理流程。"""
        if os.path.exists(self.temp_dir): shutil.rmtree(self.temp_dir)  # 清理旧的临时文件
        os.makedirs(self.temp_dir, exist_ok=True)
        output_frames_dir = os.path.join(self.temp_dir, "output_frames")
        os.makedirs(output_frames_dir, exist_ok=True)

        # 为每个新的视频处理重置跟踪器状态变量
        self.current_frame_idx = 0
        self.kalman_filter = None
        self.active_template_data = None
        self.trajectory = []
        self.consecutive_misses = 0
        self.hits = 0
        self.frames_since_kf_lost = 0

        try:
            input_frames_dir = self._extract_frames(input_video_path)  # 提取并下采样帧
            frame_files = sorted(
                [f for f in os.listdir(input_frames_dir) if f.startswith('frame_') and f.endswith('.png')])
            if not frame_files: logger.error("未能从视频中提取任何帧。"); return

            total_frames = len(frame_files)
            logger.info(f"开始为 {total_frames} 帧进行目标跟踪...")

            for i, fname in enumerate(frame_files):
                frame_path = os.path.join(input_frames_dir, fname)
                try:
                    with Image.open(frame_path) as frame_pil_small:  # 已由_extract_frames下采样
                        center_orig, size_orig = self.process_frame(frame_pil_small.copy())  # 主要跟踪逻辑

                        rgb_frame_to_draw = frame_pil_small.convert('RGB')  # 转换为RGB用于绘制
                        tracked_annotated_frame = self.draw_on_frame_scaled(
                            rgb_frame_to_draw, center_orig, size_orig,
                            frame_id_str=f"F{self.current_frame_idx:04d}"  # current_frame_idx在process_frame中更新
                        )
                        tracked_annotated_frame.save(os.path.join(output_frames_dir, fname))
                except Exception as e:
                    logger.error(f"处理视频帧文件 {fname} 时出错: {e}", exc_info=True)
                    # 备选方案：如果此帧处理失败，则复制原始帧到输出目录
                    if os.path.exists(frame_path):
                        logger.info(f"因错误，将原始（有问题的）帧 {fname} 复制到输出目录。")
                        shutil.copy(frame_path, os.path.join(output_frames_dir, fname))

                if (i + 1) % 50 == 0 or (i + 1) == total_frames:  # 定期记录进度
                    logger.info(f"进度: {i + 1}/{total_frames} 帧 ({((i + 1) / total_frames) * 100:.1f}%)")

            # 检测原始视频的FPS用于输出编译
            video_fps = 30.0  # 默认FPS
            try:
                probe_cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                             'stream=r_frame_rate', '-of', 'csv=p=0', input_video_path]
                fps_str = subprocess.check_output(probe_cmd, text=True, encoding='utf-8').strip()
                if '/' in fps_str:
                    num, den = map(int, fps_str.split('/'))
                    if den != 0: video_fps = num / den  # 计算FPS分数
                elif fps_str:  # 确保字符串非空再转换
                    video_fps = float(fps_str)
                logger.info(f"检测到视频FPS: {video_fps:.2f}")
            except Exception as e:
                logger.warning(f"无法检测视频FPS (使用默认 {video_fps:.2f} fps)。错误: {e}")

            self._compile_video(output_frames_dir, output_video_path, fps=video_fps)  # 合成最终视频
            logger.info(f"视频处理完成。输出已保存至: {output_video_path}")
        finally:
            # 除非处于调试模式，否则清理临时帧目录
            if os.path.exists(self.temp_dir) and not logger.isEnabledFor(logging.DEBUG):
                logger.info(f"清理临时目录: {self.temp_dir}")
                shutil.rmtree(self.temp_dir)
            elif logger.isEnabledFor(logging.DEBUG) and os.path.exists(self.temp_dir):
                logger.info(f"临时目录 {self.temp_dir} 已保留用于调试 (日志级别为DEBUG)。")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)  # 确保相对路径正确工作

    # 定义基础目录，假设 'src/templates' 和 'src/videos' 结构
    # 如果您的项目布局不同，请调整。
    project_base_dir = "."
    src_sub_dir = "src"
    template_base_dir = os.path.join(project_base_dir, src_sub_dir, "templates/anime")
    video_base_dir = os.path.join(project_base_dir, src_sub_dir, "videos")
    output_base_dir = os.path.join(project_base_dir, "outputs")  # 输出与脚本在同一级别

    os.makedirs(template_base_dir, exist_ok=True)
    os.makedirs(video_base_dir, exist_ok=True)
    os.makedirs(output_base_dir, exist_ok=True)

    # --- 配置: 模板和视频文件 ---
    template_filenames = ["template1.png", "template2.png", "template3.png"]  # 您的模板文件名
    input_video_filename = "anime.mp4"  # 您的输入视频文件名
    # --- 结束配置 ---

    template_paths = []
    for fname in template_filenames:
        path = os.path.join(template_base_dir, fname)
        if os.path.exists(path):
            template_paths.append(path)
        else:  # 如果特定模板丢失，则创建一个占位符虚拟模板
            logger.warning(f"模板文件 '{path}' 未找到。正在创建一个占位符虚拟模板。")
            s = (np.random.randint(30, 51), np.random.randint(30, 51))  # 虚拟模板尺寸
            try:
                img = Image.new('L', s, color=np.random.randint(100, 151))
                d = ImageDraw.Draw(img)
                cx, cy = s[0] // 2, s[1] // 2
                o = min(s) // 3  # 虚拟模板中的特征尺寸
                d.rectangle([cx - o, cy - o, cx + o, cy + o], fill=np.random.randint(0, 51))
                img.save(path)
                template_paths.append(path)
                logger.info(f"已创建虚拟模板: {path}")
            except Exception as e:
                logger.error(f"为 {path} 创建虚拟模板失败: {e}")

    if not template_paths:
        logger.critical("没有可用的模板 (真实的或虚拟的)。跟踪无法继续。正在退出。")
        exit()

    input_video_path = os.path.join(video_base_dir, input_video_filename)
    # 输出视频将基于输入命名，并带有前缀
    output_video_path = os.path.join(output_base_dir, f"tracked_{input_video_filename}")

    if not os.path.exists(input_video_path):
        logger.error(f"输入视频 '{input_video_path}' 未找到。尝试创建虚拟视频。")
        try:
            dummy_tmp_dir = os.path.join(output_base_dir, "_dummy_video_frames")
            os.makedirs(dummy_tmp_dir, exist_ok=True)
            fw, fh = 320, 240
            n_frames = 120  # 4 秒 @ 30fps
            for i in range(n_frames):
                bg_val = max(0, min(255, int(120 + 100 * np.sin(i * 2 * np.pi / n_frames))))  # 脉动背景
                img = Image.new('RGB', (fw, fh), color=(bg_val // 3, bg_val // 2, bg_val))
                d = ImageDraw.Draw(img)
                sq_s = 35  # 虚拟目标尺寸
                # 虚拟目标更复杂的运动
                x = fw // 2 + int((fw * 0.35) * np.sin(i * 4 * np.pi / n_frames)) - sq_s // 2
                y = fh // 2 + int((fh * 0.25) * np.cos(i * 6 * np.pi / n_frames)) - sq_s // 2
                d.rectangle([x, y, x + sq_s, y + sq_s], fill="white")  # 白色方块作为目标
                d.text((10, 10), f"帧 {i}", fill=(50, 50, 50), font=common_font)  # 帧号文本
                img.save(os.path.join(dummy_tmp_dir, f"frame_{i:04d}.png"))  # 一致的4位数帧编号

            cmd_dummy_vid = ['ffmpeg', '-y', '-framerate', '30', '-i', os.path.join(dummy_tmp_dir, f"frame_%04d.png"),
                             '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-t', str(n_frames / 30),
                             input_video_path]  # -t 指定时长
            subprocess.run(cmd_dummy_vid, check=True, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)  # 隐藏ffmpeg输出
            shutil.rmtree(dummy_tmp_dir)
            logger.info(f"成功创建虚拟视频: {input_video_path}")
        except Exception as e:
            logger.error(f"虚拟视频创建失败: {e}。FFmpeg可能未安装或未在PATH中。")
            logger.critical("无法在没有输入视频的情况下继续。正在退出。")
            exit()

    try:
        # 这些是您认为对任务1（卡通表情符号）效果良好的参数。
        # 请根据您的最终测试调整这些值。
        tracker = VideoTracker(
            template_paths=template_paths,
            temp_dir=os.path.join(output_base_dir, "_temp_processing_final"),  # 临时文件目录
            ncc_threshold=0.22,
            reliable_ncc_threshold=0.38,
            re_acquire_ncc_threshold=0.50,
            kf_measurement_noise_std=18.0,
            kf_process_noise_std=3.5,
            detection_gate_factor=5.0,
            frame_downsample_factor=2,
            search_roi_padding_factor=2.0,
            max_consecutive_misses=10,
            min_hits_to_activate=2,
            re_acquire_timeout_frames=25
        )
        tracker.process_video(input_video_path, output_video_path)
    except ValueError as ve:
        logger.critical(f"VideoTracker 初始化错误: {ve}")
    except RuntimeError as rte:  # 特别捕获FFmpeg相关的运行时错误
        logger.critical(f"运行时错误 (可能与FFmpeg相关): {rte}")
    except Exception as e:
        logger.error(f"视频处理过程中发生意外错误: {str(e)}", exc_info=True)
