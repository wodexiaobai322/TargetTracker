# myTask2.py
import os
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import shutil
import logging

# 尝试导入必要的自定义模块
try:
    # 从 correlation_matcher_fft.py 导入自定义的NCC FFT匹配函数
    from correlation_matcher_fft import match_template_ncc_fft as match_template_ncc_custom
except ImportError:
    logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.critical("致命错误: 无法从 'correlation_matcher_fft.py' 导入 'match_template_ncc_fft'。请确保已安装 scipy。")
    exit()

try:
    # 从 kalman_tracker.py 导入卡尔曼点跟踪器
    from kalman_tracker import KalmanPointTracker
except ImportError:
    logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.critical("致命错误: 无法从 'kalman_tracker.py' 导入 KalmanPointTracker。")
    exit()

LOGGING_LEVEL = logging.INFO  # 可调整为 logging.DEBUG 以获取更详细的日志
logging.basicConfig(level=LOGGING_LEVEL,
                    format='%(asctime)s - %(levelname)s - [%(module)s.%(funcName)s:%(lineno)d] - %(message)s')
logger = logging.getLogger(__name__)

# 尝试加载Arial字体，如果失败则使用默认字体
try:
    common_font = ImageFont.truetype("arial.ttf", 12)
except IOError:
    common_font = ImageFont.load_default()


class VideoTracker:
    """
    视频跟踪器类，用于在视频序列中跟踪指定的目标模板。
    它结合了模板匹配（NCC FFT）、卡尔曼滤波和多阶段模板选择策略。
    """

    def __init__(self, template_paths, temp_dir="temp_frames",
                 ncc_threshold=0.28, reliable_ncc_threshold=0.38, re_acquire_ncc_threshold=0.50,
                 initial_grace_period_frames=20, initial_lock_threshold=0.60,
                 small_template_area_threshold_px=250, small_template_ncc_boost=0.05,
                 kf_measurement_noise_std=40.0, kf_process_noise_std=12.0,
                 detection_gate_factor=4.5, max_consecutive_misses=22,
                 min_hits_to_activate=2, re_acquire_timeout_frames=45,
                 frame_downsample_factor=1, search_roi_padding_factor=2.8,
                 turn_phase_starts_at_frame=50,  # 转弯阶段开始的帧号
                 distant_phase_starts_at_frame=100  # 远距离阶段开始的帧号
                 ):
        """
        初始化 VideoTracker。

        参数:
            template_paths (list): 模板图像文件路径列表。
            temp_dir (str): 用于存储临时帧的目录。
            ncc_threshold (float): NCC匹配的最低接受阈值。
            reliable_ncc_threshold (float): NCC匹配的可靠测量阈值 (用于更新KF或激活)。
            re_acquire_ncc_threshold (float): 目标丢失后重新捕获时的NCC阈值。
            initial_grace_period_frames (int): 初始宽限期帧数，期间使用更高的锁定阈值。
            initial_lock_threshold (float): 初始宽限期内用于首次锁定目标的NCC阈值。
            small_template_area_threshold_px (int): 小模板的面积阈值 (像素平方)。
            small_template_ncc_boost (float): 对小模板的可靠NCC阈值的提升量。
            kf_measurement_noise_std (float): 卡尔曼滤波器测量噪声的标准差。
            kf_process_noise_std (float): 卡尔曼滤波器过程噪声的标准差。
            detection_gate_factor (float): 卡尔曼滤波器检测门限因子 (乘以测量噪声标准差)。
            max_consecutive_misses (int): 连续丢失目标多少帧后认为目标彻底丢失并重置KF。
            min_hits_to_activate (int): 卡尔曼滤波器激活前需要的最小命中次数。
            re_acquire_timeout_frames (int): 目标丢失后尝试重新捕获的最大帧数。
            frame_downsample_factor (int): 帧和模板的缩放因子，用于加速处理。
            search_roi_padding_factor (float): 搜索区域相对于模板尺寸的填充因子。
            turn_phase_starts_at_frame (int): "转弯/中距离"阶段开始的帧号，此后才考虑使用中距离模板。
            distant_phase_starts_at_frame (int): "远距离"阶段开始的帧号，此后才考虑使用远距离模板。
        """
        self.frame_downsample_factor = max(1, int(frame_downsample_factor))  # 确保缩放因子至少为1
        self.templates_processed = []  # 存储预处理后的模板数据
        self.small_template_area_threshold_px = small_template_area_threshold_px  # 小模板面积阈值
        self.small_template_ncc_boost = small_template_ncc_boost  # 小模板NCC分数提升
        self.initial_grace_period_frames = initial_grace_period_frames  # 初始宽限期
        self.initial_lock_threshold = initial_lock_threshold  # 初始锁定阈值

        # 新增：存储延迟激活帧号，用于分阶段启用不同模板
        self.turn_phase_starts_at_frame = turn_phase_starts_at_frame
        self.distant_phase_starts_at_frame = distant_phase_starts_at_frame

        logger.info("加载并预处理模板...")
        for p in template_paths:
            try:
                np_arr, orig_pil_size, size_cat = self.load_template(p)
                self.templates_processed.append({
                    'array': np_arr,  # 模板的NumPy数组 (可能已缩放)
                    'orig_pil_size': orig_pil_size,  # 模板原始PIL图像尺寸
                    'id': os.path.basename(p),  # 模板文件名作为ID
                    'size_cat': size_cat,  # 模板尺寸分类 ('small', 'medium', 'large')
                    'path': p  # 模板原始路径
                })
            except Exception as e:
                logger.error(f"跳过模板 {p}，加载错误: {e}", exc_info=LOGGING_LEVEL == logging.DEBUG)
        if not self.templates_processed: raise ValueError("未能成功加载任何模板。")

        # 按模板面积从大到小排序，优先尝试大模板
        self.templates_processed.sort(key=lambda t: t['array'].shape[0] * t['array'].shape[1], reverse=True)

        for t_data in self.templates_processed:
            logger.info(
                f"  模板ID: {t_data['id']}, 处理后形状: {t_data['array'].shape}, "
                f"原始PIL尺寸: {t_data['orig_pil_size']}, 尺寸分类: {t_data['size_cat']}")

        self.temp_dir = temp_dir  # 临时文件目录
        self.trajectory = []  # 存储目标轨迹点 (原始尺寸坐标)
        self.ncc_threshold = ncc_threshold  # NCC基础接受阈值
        self.reliable_ncc_threshold = reliable_ncc_threshold  # 可靠NCC阈值
        self.re_acquire_ncc_threshold = re_acquire_ncc_threshold  # 重新捕获时的NCC阈值
        self.kf_measurement_noise_std = kf_measurement_noise_std  # KF测量噪声
        self.kf_process_noise_std = kf_process_noise_std  # KF过程噪声
        self.detection_gate_factor = detection_gate_factor  # KF检测门限因子
        self.search_roi_padding_factor = search_roi_padding_factor  # 搜索区域填充因子
        self.max_consecutive_misses = max_consecutive_misses  # 最大连续丢失次数
        self.min_hits_to_activate = min_hits_to_activate  # KF激活所需最小命中数
        self.re_acquire_timeout_frames = re_acquire_timeout_frames  # 重新捕获超时帧数

        # 跟踪器状态变量
        self.kalman_filter = None  # 卡尔曼滤波器实例
        self.active_template_data = None  # 当前用于跟踪的模板数据
        self.consecutive_misses = 0  # 连续丢失计数
        self.hits = 0  # 命中计数 (KF更新次数)
        self.frames_since_kf_lost = 0  # KF丢失后经过的帧数 (用于重捕获计时)
        self.current_frame_idx = 0  # 当前处理的帧序号 (从1开始)
        self.last_successful_match_score = -1.0  # 上一次成功匹配的NCC分数
        self.last_kf_state_small = None  # 上一个KF状态 (缩放后坐标)，用于KF丢失后指导ROI

    def load_template(self, path):
        """
        加载单个模板图像，转换为灰度，并根据 frame_downsample_factor 进行缩放。
        同时对模板进行尺寸分类。
        """
        img_pil = Image.open(path).convert('L')  # 打开图像并转为灰度
        original_pil_size = img_pil.size  # 记录原始尺寸
        processed_pil = img_pil
        if self.f_dsf() > 1:  # 如果设置了缩放因子
            new_w = original_pil_size[0] // self.f_dsf()
            new_h = original_pil_size[1] // self.f_dsf()
            if new_w > 0 and new_h > 0:  # 确保缩放后尺寸有效
                try:
                    # 使用LANCZOS高质量缩放
                    processed_pil = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
                except AttributeError:  # PIL/Pillow老版本兼容
                    processed_pil = img_pil.resize((new_w, new_h), Image.LANCZOS)
            else:
                logger.warning(f"模板 {path} 缩放后尺寸为零。将使用原始尺寸（未缩放）。")

        img_np = np.array(processed_pil, dtype=np.uint8)  # 转换为NumPy数组
        if img_np.size < 1: raise ValueError(f"模板 {path} 处理后无效 (形状:{img_np.shape})。")

        template_area = img_np.shape[0] * img_np.shape[1]  # 计算缩放后模板面积
        size_category = 'large'  # 默认为大模板
        if template_area < self.small_template_area_threshold_px:
            size_category = 'small'
        elif template_area < self.small_template_area_threshold_px * 2.5:  # 中等模板阈值
            size_category = 'medium'
        return img_np, original_pil_size, size_category

    def _get_search_region(self, frame_gray_np_shape_small):
        """
        根据卡尔曼滤波器的预测（如果可用）确定搜索区域(ROI)。
        如果KF不可用，则使用上一次KF状态；如果都没有，则全图搜索。
        返回搜索区域的左上角坐标 (y1, x1) 和尺寸 (h, w) (均为缩放后尺寸)。
        """
        fh_small, fw_small = frame_gray_np_shape_small  # 缩放后帧的高度和宽度
        center_for_roi_small = None  # 用于ROI计算的中心点 (缩放后坐标)

        if self.kalman_filter:
            # 如果KF存在，使用其预测值作为ROI中心
            center_for_roi_small = self.kalman_filter.predict()
            self.last_kf_state_small = center_for_roi_small  # 保存当前预测，以备KF丢失后使用
        elif self.last_kf_state_small is not None:
            # 如果KF丢失，但有上一次KF状态，则使用它
            center_for_roi_small = self.last_kf_state_small
            logger.debug(
                f"F{self.current_frame_idx:04d}: KF lost, using last known KF state for ROI: {center_for_roi_small}")

        if center_for_roi_small is not None:
            # 确定模板尺寸 (缩放后) 以计算ROI大小
            tpl_h_s, tpl_w_s = (20, 20)  # 默认/最小模板尺寸
            current_search_padding_factor = self.search_roi_padding_factor
            if self.active_template_data:  # 如果有当前活动模板
                tpl_h_s, tpl_w_s = self.active_template_data['array'].shape
            elif self.templates_processed:  # 否则使用第一个（通常是最大的）模板的尺寸
                tpl_h_s, tpl_w_s = self.templates_processed[0]['array'].shape

            # 如果KF处于“滑行”状态（连续未更新），稍微扩大搜索区域
            if self.kalman_filter and self.kalman_filter.time_since_update > 2:  # 超过2帧未更新
                current_search_padding_factor *= 1.2  # 增加填充因子
                logger.debug(
                    f"F{self.current_frame_idx:04d}: KF coasting, increasing ROI padding factor to {current_search_padding_factor:.2f}")

            # 计算ROI的理想宽度和高度
            roi_w = max(int(tpl_w_s * current_search_padding_factor), tpl_w_s + 20)  # 至少比模板宽20像素
            roi_h = max(int(tpl_h_s * current_search_padding_factor), tpl_h_s + 20)  # 至少比模板高20像素

            # 计算ROI的理想左上角坐标
            roi_x1_ideal = center_for_roi_small[0] - roi_w / 2.0
            roi_y1_ideal = center_for_roi_small[1] - roi_h / 2.0

            # 确保ROI在帧边界内
            roi_y1 = max(0, int(round(roi_y1_ideal)))
            roi_x1 = max(0, int(round(roi_x1_ideal)))
            roi_y2 = min(fh_small, int(round(roi_y1_ideal + roi_h)))
            roi_x2 = min(fw_small, int(round(roi_x1_ideal + roi_w)))

            if roi_x2 > roi_x1 and roi_y2 > roi_y1:  # 确保ROI有效
                logger.debug(
                    f"F{self.current_frame_idx:04d}: Search ROI: TL({roi_x1},{roi_y1}) Size({roi_x2 - roi_x1},{roi_y2 - roi_y1}) from Tpl({tpl_w_s},{tpl_h_s})")
                return (roi_y1, roi_x1), (roi_y2 - roi_y1, roi_x2 - roi_x1)  # 返回左上角和尺寸
            else:
                logger.debug(
                    f"F{self.current_frame_idx:04d}: KF ROI invalid ({roi_x1},{roi_y1},{roi_x2},{roi_y2}). Full search.")

        # 如果无法确定ROI中心，或者计算出的ROI无效，则进行全图搜索
        logger.debug(f"F{self.current_frame_idx:04d}: No KF/last_kf_state. Full search.")
        self.last_kf_state_small = None  # 清除上一个KF状态，因为它可能已失效
        return (0, 0), frame_gray_np_shape_small  # 返回全图作为搜索区域

    def process_frame(self, frame_pil_small):
        """
        处理单帧图像。

        参数:
            frame_pil_small (PIL.Image): 已经过缩小的输入帧图像 (灰度或彩色均可，内部会转灰度)。

        返回:
            tuple: (center_orig, size_orig)
                center_orig (tuple or None): 目标在原始尺寸帧中的中心坐标 (x, y)，如果未检测到则为None。
                size_orig (tuple or None): 目标在原始尺寸帧中的包围盒尺寸 (width, height)，如果未检测到则为None。
        """
        self.current_frame_idx += 1
        frame_id_str = f"F{self.current_frame_idx:04d}"  # 用于日志的帧ID字符串
        logger.debug(f"{frame_id_str}: Processing frame (Downsampled size: {frame_pil_small.size})")
        frame_gray_np = np.array(frame_pil_small.convert('L'), dtype=np.uint8)  # 转换为灰度NumPy数组

        # 判断当前跟踪阶段
        is_re_acquiring_phase = (self.kalman_filter is None) and (self.frames_since_kf_lost > 0)  # 是否处于重新捕获阶段
        is_initial_lock_phase = (self.kalman_filter is None) and \
                                (not is_re_acquiring_phase) and \
                                (self.current_frame_idx <= self.initial_grace_period_frames)  # 是否处于初始锁定宽限期

        # 根据当前帧号和预设的阶段开始帧号，选择当前要使用的模板
        templates_to_use_now = []
        for t_data in self.templates_processed:
            template_id_str = str(t_data['id']).lower()  # 模板ID转小写，方便匹配
            # 识别中距离/转弯模板 (通常文件名包含 t7, t8 或 "medium")
            is_turn_template = ("t7.png" in template_id_str or "t8.png" in template_id_str
                                or "medium" in t_data.get('size_cat', ''))  # 使用size_cat作为补充
            # 识别远距离模板 (通常文件名包含 t9 或 "distant", "far")
            is_distant_template = ("t9.png" in template_id_str or "distant" in template_id_str
                                   or "far" in template_id_str)

            # 如果是转弯模板，但当前帧还没到转弯阶段开始帧，则跳过
            if is_turn_template and self.current_frame_idx < self.turn_phase_starts_at_frame:
                continue
            # 如果是远距离模板，但当前帧还没到远距离阶段开始帧，则跳过
            if is_distant_template and self.current_frame_idx < self.distant_phase_starts_at_frame:
                continue
            templates_to_use_now.append(t_data)  # 将符合条件的模板加入当前使用列表

        # 如果阶段性过滤后没有模板可选，但总模板列表不为空，则使用所有模板作为后备
        if not templates_to_use_now and self.templates_processed:
            logger.warning(
                f"{frame_id_str}: No templates selected by phase logic for frame {self.current_frame_idx}. Using all templates as fallback.")
            templates_to_use_now = list(self.templates_processed)
        elif not self.templates_processed:  # 如果一开始就没有任何模板加载
            logger.error(f"{frame_id_str}: No templates loaded at all.")
            return None, None  # 无法处理

        if LOGGING_LEVEL == logging.DEBUG:
            logger.debug(
                f"{frame_id_str}: Templates considered for matching: {[t['id'] for t in templates_to_use_now]}")

        # 获取搜索区域 (ROI)
        search_offset_yx, search_size_hw = self._get_search_region(frame_gray_np.shape)
        search_y1, search_x1 = search_offset_yx
        search_h, search_w = search_size_hw
        search_image_np = frame_gray_np[search_y1: search_y1 + search_h, search_x1: search_x1 + search_w]  # 从帧中裁剪ROI

        if search_image_np.size < 25:  # 如果ROI过小 (例如，小于5x5)，可能无法进行有效匹配
            logger.debug(f"{frame_id_str}: Search ROI too small ({search_image_np.shape}).")
            if self.kalman_filter:
                self.consecutive_misses += 1  # 如果KF存在，计为一次丢失
            elif is_re_acquiring_phase:
                self.frames_since_kf_lost += 1  # 如果在重捕获，增加重捕获计时
            return None, None

        # 在ROI内对选定的模板进行NCC匹配
        best_match_score_overall, best_s_center_overall, best_tpl_data_overall = -np.inf, None, None
        for t_data in templates_to_use_now:
            tpl_np = t_data['array']  # 缩放后的模板NumPy数组
            tpl_h_s, tpl_w_s = tpl_np.shape  # 缩放后模板尺寸

            # 如果模板大于搜索区域，则跳过此模板
            if tpl_h_s > search_h or tpl_w_s > search_w:
                if LOGGING_LEVEL == logging.DEBUG:
                    logger.debug(
                        f"{frame_id_str}: Template {t_data['id']} ({tpl_h_s}x{tpl_w_s}) too large for ROI ({search_h}x{search_w}). Skipping.")
                continue

            # 进行NCC匹配
            ncc_map = match_template_ncc_custom(search_image_np, tpl_np)
            if ncc_map is None or ncc_map.size == 0:  # NCC图无效
                if LOGGING_LEVEL == logging.DEBUG:
                    logger.debug(f"{frame_id_str}: NCC map for template {t_data['id']} is empty. Skipping.")
                continue

            current_max_score_for_template = np.max(ncc_map)  # 当前模板在ROI内的最高NCC分
            if LOGGING_LEVEL == logging.DEBUG:
                logger.debug(
                    f"{frame_id_str}: Tpl {t_data['id']} ({t_data['size_cat']}) ROI Max NCC: {current_max_score_for_template:.4f}")

            # 更新全局最佳匹配
            if current_max_score_for_template > best_match_score_overall:
                best_match_score_overall = current_max_score_for_template
                best_tpl_data_overall = t_data  # 最佳匹配的模板数据
                # 计算最佳匹配位置在ROI内的坐标 (中心点)
                my_roi, mx_roi = np.unravel_index(np.argmax(ncc_map), ncc_map.shape)
                # 转换到缩放后完整帧的坐标系 (中心点)
                best_s_center_overall = ((mx_roi + tpl_w_s / 2.0) + search_x1, (my_roi + tpl_h_s / 2.0) + search_y1)

        if best_tpl_data_overall:  # 如果找到了最佳匹配模板
            logger.info(
                f"{frame_id_str}: NCC Best: {best_tpl_data_overall['id']} ({best_tpl_data_overall['size_cat']}), "
                f"Score: {best_match_score_overall:.3f} at {best_s_center_overall}")
        else:  # 如果在ROI内没有找到任何模板的匹配
            logger.debug(f"{frame_id_str}: No template match found in ROI.")
            if self.kalman_filter:
                self.consecutive_misses += 1
            elif is_re_acquiring_phase:
                self.frames_since_kf_lost += 1
            return None, None  # 无匹配，返回

        self.last_successful_match_score = -1.0  # 重置上次成功匹配分数
        detected_center_sm, detected_size_orig, is_reliable_measurement = None, None, False  # 初始化检测结果

        # 根据当前阶段确定NCC接受阈值和可靠阈值
        current_acceptance_threshold = self.ncc_threshold
        current_reliable_threshold = self.reliable_ncc_threshold

        if is_re_acquiring_phase:  # 重新捕获阶段，使用更高的阈值
            current_acceptance_threshold = self.re_acquire_ncc_threshold
            current_reliable_threshold = self.re_acquire_ncc_threshold
        elif is_initial_lock_phase:  # 初始锁定阶段，使用特定的初始锁定阈值
            current_acceptance_threshold = self.initial_lock_threshold
            current_reliable_threshold = self.initial_lock_threshold
            logger.debug(
                f"{frame_id_str}: Initial lock phase. Thresh: Acc={current_acceptance_threshold:.2f}, Rel={current_reliable_threshold:.2f}")

        # 如果最佳匹配是小模板，并且不是初始锁定阶段，则稍微提高其可靠阈值
        if best_tpl_data_overall['size_cat'] == 'small' and not is_initial_lock_phase:
            effective_reliable_threshold_for_small = self.reliable_ncc_threshold + self.small_template_ncc_boost
            # 取当前可靠阈值和为小模板调整后的阈值中的较大者
            current_reliable_threshold = max(current_reliable_threshold, effective_reliable_threshold_for_small)
            logger.debug(
                f"{frame_id_str}: Small Tpl {best_tpl_data_overall['id']}. BaseRel: {self.reliable_ncc_threshold:.2f}, Boost: {self.small_template_ncc_boost:.2f}, EffRel: {current_reliable_threshold:.2f}")

        # 判断NCC分数是否满足当前接受阈值
        if best_match_score_overall >= current_acceptance_threshold and best_s_center_overall is not None:
            self.last_successful_match_score = best_match_score_overall  # 记录成功的匹配分数
            detected_center_sm = best_s_center_overall  # 检测到的中心点 (缩放后)
            detected_size_orig = best_tpl_data_overall['orig_pil_size']  # 检测到的模板原始尺寸
            if best_match_score_overall >= current_reliable_threshold:  # 是否为可靠测量
                is_reliable_measurement = True

            logger.info(
                f"{frame_id_str}: NCC Candidate {best_tpl_data_overall['id']} Score {best_match_score_overall:.3f} vs "
                f"AccTh {current_acceptance_threshold:.2f}. "
                f"Reliable if > {current_reliable_threshold:.2f}. IsReliable: {is_reliable_measurement}")

            # 处理卡尔曼滤波器 (KF)
            if self.kalman_filter is None:  # 如果KF尚未初始化
                can_initialize_kf = False
                init_reason = "Unknown"
                if is_reliable_measurement:  # 只有可靠测量才能初始化KF
                    if is_re_acquiring_phase:  # 在重捕获阶段成功找到可靠匹配
                        can_initialize_kf, init_reason = True, "Re-acquired"
                    elif is_initial_lock_phase:  # 在初始锁定阶段成功找到可靠匹配 (分数高于初始锁定阈值)
                        can_initialize_kf, init_reason = True, "Initial Strong Lock"
                    else:  # 非特定阶段，但有可靠测量 (通常发生在视频刚开始，但已过宽限期)
                        can_initialize_kf, init_reason = True, "Initial Reliable"

                if can_initialize_kf:
                    logger.info(
                        f"{frame_id_str}: Init KF ({init_reason}) at {detected_center_sm} (Tpl: {best_tpl_data_overall['id']}) Score: {best_match_score_overall:.2f}")
                    # 初始化KF
                    self.kalman_filter = KalmanPointTracker(detected_center_sm, 1.0, self.kf_process_noise_std,
                                                            self.kf_measurement_noise_std)
                    self.active_template_data = best_tpl_data_overall  # 设置当前活动模板
                    self.hits, self.consecutive_misses, self.frames_since_kf_lost = 1, 0, 0  # 重置计数器
                    self.last_kf_state_small = detected_center_sm  # 保存KF状态
                else:  # NCC分数不足以初始化KF
                    if LOGGING_LEVEL == logging.DEBUG:
                        logger.debug(
                            f"{frame_id_str}: NCC Score {best_match_score_overall:.3f} not sufficient to init/re-acq KF. Phase: {'Re-acq' if is_re_acquiring_phase else 'Initial' if is_initial_lock_phase else 'Normal'}. Needed Rel: {current_reliable_threshold:.2f}. IsReliable: {is_reliable_measurement}")
                    detected_center_sm = None  # 清除检测结果，因为未满足初始化条件
                    if is_re_acquiring_phase: self.frames_since_kf_lost += 1  # 如果在重捕获，增加计时

            elif detected_center_sm:  # KF已存在，并且有新的检测
                predicted_state_small = self.kalman_filter.get_state()  # 获取KF当前状态(预测)
                # 计算检测到的位置与KF预测位置之间的距离 (平方)
                dx = detected_center_sm[0] - predicted_state_small[0]
                dy = detected_center_sm[1] - predicted_state_small[1]
                distance_sq = dx ** 2 + dy ** 2
                gate_radius_sq = (self.detection_gate_factor * self.kf_measurement_noise_std) ** 2  # 计算检测门限半径 (平方)

                if distance_sq <= gate_radius_sq:  # 如果检测在门限内 (有效关联)
                    logger.debug(
                        f"{frame_id_str}: NCC passed gate (Dist^2: {distance_sq:.1f} <= Gate^2: {gate_radius_sq:.1f}). Updating KF with {best_tpl_data_overall['id']}.")
                    if is_reliable_measurement:  # 只有可靠测量才用于更新KF
                        self.kalman_filter.update(detected_center_sm)  # 更新KF
                        self.active_template_data = best_tpl_data_overall  # 更新活动模板
                        self.hits += 1
                        self.consecutive_misses = 0
                        self.frames_since_kf_lost = 0  # 重置丢失计数
                        self.last_kf_state_small = self.kalman_filter.get_state()  # 保存KF状态
                    else:  # 在门限内，但测量不可靠
                        logger.debug(
                            f"{frame_id_str}: Passed gate but measurement not reliable (Score {best_match_score_overall:.2f} < RelTh {current_reliable_threshold:.2f}). KF not updated. Miss.")
                        self.consecutive_misses += 1  # 计为一次丢失 (因为未更新KF)
                else:  # 检测在门限外 (关联失败)
                    logger.debug(
                        f"{frame_id_str}: NCC failed gate. Pred {predicted_state_small}, Meas {detected_center_sm} (Dist^2: {distance_sq:.1f} > Gate^2: {gate_radius_sq:.1f}). Miss.")
                    detected_center_sm = None  # 清除检测结果，因为它未通过门限
                    self.consecutive_misses += 1  # 计为一次丢失
        else:  # NCC分数低于当前接受阈值
            if self.kalman_filter:
                self.consecutive_misses += 1  # 如果KF存在，计为丢失
            elif is_re_acquiring_phase:
                self.frames_since_kf_lost += 1  # 如果在重捕获，增加计时
            logger.debug(
                f"{frame_id_str}: No NCC match above AccTh {current_acceptance_threshold:.2f} (Best was {best_match_score_overall:.3f}).")
            detected_center_sm = None  # 清除检测结果

        # 根据KF状态和检测结果，确定最终输出
        out_c_orig, out_s_orig = None, None  # 初始化输出 (原始尺寸坐标和尺寸)
        if self.kalman_filter:  # 如果KF存在
            if self.consecutive_misses >= self.max_consecutive_misses:  # 连续丢失次数达到上限
                logger.info(
                    f"{frame_id_str}: Target lost (KF Misses: {self.consecutive_misses}). Reset KF. Re-acquiring.")
                # 重置KF和相关状态，进入重新捕获模式
                self.kalman_filter, self.active_template_data, self.hits, self.consecutive_misses = None, None, 0, 0
                self.frames_since_kf_lost = 1
            elif self.hits >= self.min_hits_to_activate or self.consecutive_misses > 0:  # KF已激活或正在滑行
                kf_state_sm = self.kalman_filter.get_state()  # 获取KF状态 (缩放后)
                # 将KF状态转换回原始帧尺寸
                out_c_orig = (int(round(kf_state_sm[0] * self.f_dsf())), int(round(kf_state_sm[1] * self.f_dsf())))
                out_s_orig = self.active_template_data['orig_pil_size'] if self.active_template_data else (
                30, 30)  # 使用活动模板的原始尺寸
            elif detected_center_sm and is_reliable_measurement:  # KF刚被初始化 (hits < min_hits_to_activate, misses = 0)
                # 输出当前可靠的NCC检测结果 (原始尺寸)
                out_c_orig = (
                    int(round(detected_center_sm[0] * self.f_dsf())), int(round(detected_center_sm[1] * self.f_dsf())))
                out_s_orig = detected_size_orig
        elif is_re_acquiring_phase:  # 如果KF不存在，且处于重新捕获阶段
            if self.frames_since_kf_lost >= self.re_acquire_timeout_frames:  # 重新捕获超时
                logger.info(f"{frame_id_str}: Re-acq timeout ({self.frames_since_kf_lost} frames). Target lost.")
                self.frames_since_kf_lost = 0  # 重置计时
                self.last_kf_state_small = None  # 清除最后KF状态，避免影响后续ROI
        elif detected_center_sm and is_reliable_measurement:  # KF不存在，非重捕获，但有可靠的初始检测
            # 这种情况理论上应该在上面KF初始化逻辑中处理并建立了KF。
            # 但作为一种保障，如果KF未建立但有可靠检测，也输出它。
            logger.debug(f"{frame_id_str}: Outputting initial reliable NCC (KF should be init).")
            out_c_orig = (
                int(round(detected_center_sm[0] * self.f_dsf())), int(round(detected_center_sm[1] * self.f_dsf())))
            out_s_orig = detected_size_orig

        return out_c_orig, out_s_orig

    def f_dsf(self):
        """返回帧下采样因子 (frame_downsample_factor) 的便捷方法。"""
        return self.frame_downsample_factor

    def lsms(self):
        """返回最后一次成功匹配分数 (last_successful_match_score) 的便捷方法。"""
        return self.last_successful_match_score

    def draw_on_frame_scaled(self, frame_pil_rgb_small, center_pos_orig, box_size_orig, frame_id_str=""):
        """
        在缩放后的RGB帧上绘制跟踪结果（包围盒、轨迹、状态文本）。

        参数:
            frame_pil_rgb_small (PIL.Image): 缩放后的RGB帧图像。
            center_pos_orig (tuple or None): 目标在原始帧中的中心坐标 (x, y)。
            box_size_orig (tuple or None): 目标在原始帧中的尺寸 (width, height)。
            frame_id_str (str): 当前帧的ID字符串，用于显示。

        返回:
            PIL.Image: 绘制了跟踪信息后的帧图像。
        """
        draw = ImageDraw.Draw(frame_pil_rgb_small)  # 获取绘图对象
        sm_w, sm_h = frame_pil_rgb_small.size  # 缩放后帧的尺寸

        # 绘制轨迹
        if center_pos_orig: self.trajectory.append(center_pos_orig)  # 添加当前点到轨迹 (原始坐标)
        if len(self.trajectory) > 100: self.trajectory.pop(0)  # 保持轨迹长度上限
        if len(self.trajectory) > 1:
            # 将轨迹点转换为缩放后坐标
            traj_sm = [(np.clip(int(round(p[0] / self.f_dsf())), 0, sm_w - 1),
                        np.clip(int(round(p[1] / self.f_dsf())), 0, sm_h - 1)) for p in self.trajectory]
            if len(traj_sm) > 1: draw.line(traj_sm, fill='lime', width=1 if self.f_dsf() > 1 else 2)  # 绘制轨迹线

        # 根据跟踪器状态确定状态文本和包围盒颜色
        status_text, box_color = "Status: Searching", 'yellow'  # 默认状态
        is_re_acq_draw = (self.kalman_filter is None) and (self.frames_since_kf_lost > 0)  # 是否在重捕获
        is_def_lost = is_re_acq_draw and (
                    self.frames_since_kf_lost >= self.re_acquire_timeout_frames)  # 是否已确认丢失 (重捕获超时)

        if is_def_lost:
            status_text, box_color, center_pos_orig = "Status: Target Lost", "gray", None  # 目标丢失
        elif self.kalman_filter:  # KF 存在
            if self.consecutive_misses == 0 and self.hits >= self.min_hits_to_activate:  # 稳定跟踪
                status_text, box_color = f"Tracking (H:{self.hits} S:{self.lsms():.2f})", 'lime'
            elif self.consecutive_misses > 0:  # KF 滑行 (有丢失，但未到上限)
                status_text, box_color = f"Coasting (M:{self.consecutive_misses} H:{self.hits})", 'orange'
            else:  # KF 刚初始化，正在积累命中数
                status_text, box_color = f"Acquiring (H:{self.hits} S:{self.lsms():.2f})", 'cyan'
        elif is_re_acq_draw:  # 正在重新捕获
            status_text = f"Re-Acquiring (LostF:{self.frames_since_kf_lost})"
            box_color = 'violet'
            if center_pos_orig:  # 如果在重捕获阶段有了可靠的检测结果
                status_text, box_color = f"Re-Locked! (S:{self.lsms():.2f})", 'magenta'
        elif center_pos_orig:  # 初始锁定 (KF未建立，非重捕获，但有可靠检测)
            status_text, box_color = f"Initial Lock (S:{self.lsms():.2f})", 'magenta'

        # 绘制包围盒和中心点 (如果目标被检测到)
        if center_pos_orig and box_size_orig:
            # 将原始坐标和尺寸转换为缩放后坐标和尺寸
            s_cx = int(round(center_pos_orig[0] / self.f_dsf()))
            s_cy = int(round(center_pos_orig[1] / self.f_dsf()))
            s_bw = int(round(box_size_orig[0] / self.f_dsf()))
            s_bh = int(round(box_size_orig[1] / self.f_dsf()))
            # 计算包围盒的左上右下角坐标 (缩放后)
            s_l = max(0, s_cx - s_bw // 2)
            s_t = max(0, s_cy - s_bh // 2)
            s_r = min(sm_w, s_cx + s_bw // 2)
            s_b = min(sm_h, s_cy + s_bh // 2)
            if s_r > s_l and s_b > s_t:  # 确保包围盒有效
                draw.rectangle([s_l, s_t, s_r - 1, s_b - 1], outline=box_color,
                               width=1 if self.f_dsf() > 1 else 2)  # 绘制矩形框
            if 0 <= s_cx < sm_w and 0 <= s_cy < sm_h:  # 确保中心点在帧内
                draw.ellipse([(s_cx - 2, s_cy - 2), (s_cx + 2, s_cy + 2)],
                             fill=box_color, outline='white')  # 绘制中心小圆点

        # 绘制状态文本
        draw.text((5, 5), status_text, fill=box_color, font=common_font)
        if self.active_template_data:  # 如果有活动模板，显示其信息
            draw.text((5, 20), f"Tpl: {self.active_template_data['id']} ({self.active_template_data['size_cat']})",
                      fill='white', font=common_font)

        # 绘制帧ID (右上角)
        try:  # PIL/Pillow新版本使用textbbox
            bbox = draw.textbbox((0, 0), frame_id_str, font=common_font)
            text_w = bbox[2] - bbox[0]
        except AttributeError:  # PIL/Pillow老版本使用textsize
            text_w, _ = draw.textsize(frame_id_str, font=common_font) if hasattr(draw,
                                                                                 'textsize') else common_font.getsize(
                frame_id_str)
        draw.text((sm_w - text_w - 5, 5), frame_id_str, fill='white', font=common_font)
        return frame_pil_rgb_small

    def _ffmpeg_run(self, cmd_list, stage_name="ffmpeg_process"):
        """
        执行FFmpeg命令的辅助函数。
        """
        logger.info(f"FFmpeg {stage_name} 命令: {' '.join(cmd_list)}")
        try:
            p = subprocess.run(cmd_list, check=True, capture_output=True, text=True, encoding='utf-8')
            if p.stderr: logger.warning(f"FFmpeg stderr ({stage_name}): {p.stderr.strip()}")
        except FileNotFoundError:
            logger.error(f"FFmpeg 可执行文件未找到 ({stage_name})。请确保FFmpeg已安装并在系统路径中。")
            raise RuntimeError(f"FFmpeg not found for {stage_name}")
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg {stage_name} 错误 (返回码 {e.returncode}): {e.stderr.strip()}")
            raise RuntimeError(f"FFmpeg {stage_name} failed")

    def _extract_frames(self, input_video_path):
        """
        使用FFmpeg从输入视频中提取帧，并进行缩放。
        返回提取帧存放的目录路径。
        """
        input_frames_dir = os.path.join(self.temp_dir, "input_frames")  # 输入帧存储路径
        os.makedirs(input_frames_dir, exist_ok=True)
        # FFmpeg缩放参数：iw和ih代表输入宽度和高度
        scale_val = f'iw/{self.f_dsf()}:ih/{self.f_dsf()}'
        cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', input_video_path]
        if self.f_dsf() != 1:  # 如果设置了缩放，添加缩放滤镜
            cmd.extend(['-vf', f'scale={scale_val}'])
        # -vsync 0: 尽可能快地提取帧; -qscale:v 2: 高质量JPEG编码 (用于PNG输出)
        cmd.extend(['-vsync', '0', '-qscale:v', '2', os.path.join(input_frames_dir, 'frame_%05d.png')])
        self._ffmpeg_run(cmd, "extract")  # 执行命令
        return input_frames_dir

    def _compile_video(self, processed_frames_dir, output_video_path, fps=30.0):
        """
        使用FFmpeg将处理后的帧序列编译成视频。
        """
        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)  # 确保输出目录存在
        cmd = ['ffmpeg', '-y',  # -y: 覆盖输出文件
               '-framerate', str(fps),  # 设置输入帧率
               '-i', os.path.join(processed_frames_dir, 'frame_%05d.png'),  # 输入帧序列
               '-c:v', 'libx264',  # 使用H.264编码器
               '-preset', 'medium',  # H.264编码预设 (速度与质量的平衡)
               '-pix_fmt', 'yuv420p',  # 像素格式，确保广泛兼容性
               output_video_path]
        self._ffmpeg_run(cmd, "compile")  # 执行命令

    def reset_tracker_state(self):
        """
        重置跟踪器的内部状态，以便处理新的视频或重新开始。
        """
        self.current_frame_idx = 0
        self.kalman_filter = None
        self.active_template_data = None
        self.trajectory = []
        self.consecutive_misses = 0
        self.hits = 0
        self.frames_since_kf_lost = 0
        self.last_successful_match_score = -1.0
        self.last_kf_state_small = None
        logger.info("跟踪器状态已重置。")

    def process_video(self, input_video_path, output_video_path):
        """
        处理整个视频文件。
        包括：清理/创建临时目录、重置状态、提取帧、逐帧处理、编译输出视频、清理临时文件。
        """
        # 清理并创建临时工作目录
        if os.path.exists(self.temp_dir): shutil.rmtree(self.temp_dir)
        os.makedirs(self.temp_dir, exist_ok=True)
        output_frames_dir = os.path.join(self.temp_dir, "output_frames")  # 处理后帧的存储路径
        os.makedirs(output_frames_dir, exist_ok=True)
        self.reset_tracker_state()  # 重置跟踪器状态

        # 获取视频实际FPS
        video_fps = 30.0  # 默认FPS
        try:
            # 使用ffprobe获取视频流的平均帧率
            probe_cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                         '-show_entries', 'stream=avg_frame_rate', '-of', 'csv=p=0',
                         input_video_path]
            fps_str = subprocess.check_output(probe_cmd, text=True, encoding='utf-8').strip()
            if '/' in fps_str:  # 帧率可能是分数形式，如 "30000/1001"
                num, den = map(float, fps_str.split('/'))
                if den != 0: video_fps = num / den
            elif fps_str:  # 帧率是单个数字
                video_fps = float(fps_str)
            logger.info(f"检测到视频平均FPS: {video_fps:.2f}")
        except Exception as e:
            logger.warning(f"无法检测视频FPS (使用默认 {video_fps:.2f} fps)。错误: {e}")

        try:
            input_frames_dir = self._extract_frames(input_video_path)  # 提取帧
            # 获取所有提取的帧文件名，并排序
            frame_files = sorted(
                [f for f in os.listdir(input_frames_dir) if f.startswith('frame_') and f.endswith('.png')])
            if not frame_files:
                logger.error("未能从视频中提取任何帧。")
                return

            total_frames = len(frame_files)
            logger.info(f"开始为 {total_frames} 帧进行目标跟踪...")

            # 逐帧处理
            for i, fname in enumerate(frame_files):
                frame_path = os.path.join(input_frames_dir, fname)
                try:
                    with Image.open(frame_path) as frame_pil_small:  # 打开缩放后的帧
                        # 调用核心处理逻辑
                        center_orig, size_orig = self.process_frame(frame_pil_small.copy())  # 传递副本以防被修改
                        # 将帧转为RGB以便绘制彩色标记
                        rgb_frame_to_draw = frame_pil_small.convert('RGB')
                        # 在帧上绘制跟踪结果
                        tracked_annotated_frame = self.draw_on_frame_scaled(
                            rgb_frame_to_draw, center_orig, size_orig,
                            frame_id_str=f"F{self.current_frame_idx:04d}"  # 使用 self.current_frame_idx
                        )
                        # 保存处理并绘制后的帧
                        tracked_annotated_frame.save(os.path.join(output_frames_dir, fname))
                except Exception as e:
                    logger.error(f"处理视频帧文件 {fname} 时出错: {e}", exc_info=LOGGING_LEVEL == logging.DEBUG)
                    if os.path.exists(frame_path):  # 如果处理失败，复制原始帧到输出目录，避免视频编译中断
                        logger.info(f"因错误，将原始（有问题的）帧 {fname} 复制到输出目录。")
                        shutil.copy(frame_path, os.path.join(output_frames_dir, fname))

                # 定期打印进度
                if (i + 1) % 50 == 0 or (i + 1) == total_frames:
                    logger.info(f"进度: {i + 1}/{total_frames} 帧 ({((i + 1) / total_frames) * 100:.1f}%)")

            # 将处理后的帧编译成视频，使用检测到的FPS
            self._compile_video(output_frames_dir, output_video_path, fps=video_fps)
            logger.info(f"视频处理完成。输出已保存至: {output_video_path}")
        finally:
            # 清理临时目录 (除非日志级别为DEBUG，以便检查中间文件)
            if os.path.exists(self.temp_dir) and not logger.isEnabledFor(logging.DEBUG):
                logger.info(f"清理临时目录: {self.temp_dir}")
                shutil.rmtree(self.temp_dir)
            elif logger.isEnabledFor(logging.DEBUG) and os.path.exists(self.temp_dir):
                logger.info(f"临时目录 {self.temp_dir} 已保留用于调试 (日志级别为DEBUG)。")


if __name__ == "__main__":
    # 获取脚本所在目录，并切换工作目录到此，确保相对路径正确
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # 定义项目相关的基本路径
    project_base_dir = "."  # 当前目录作为项目基础
    src_sub_dir = "src"  # 存放模板和视频的子目录名
    template_base_dir = os.path.join(project_base_dir, src_sub_dir, "templates/car")  # 模板存放目录
    video_base_dir = os.path.join(project_base_dir, src_sub_dir, "videos")  # 视频存放目录
    output_base_dir = os.path.join(project_base_dir, "outputs")  # 输出文件存放目录

    # 创建必要的目录 (如果不存在)
    os.makedirs(template_base_dir, exist_ok=True)
    os.makedirs(video_base_dir, exist_ok=True)
    os.makedirs(output_base_dir, exist_ok=True)

    # 定义模板文件名列表
    template_filenames = [
        "t1.png", "t5.png", "t3.png",  # 初始/通用模板
        "t7.png", "t8.png",  # 中距离/转弯模板
        "t9.png",  # 远距离模板
    ]
    input_video_filename = "car.mp4"  # 输入视频文件名

    # 构建完整的模板文件路径列表
    template_paths = []
    for fname in template_filenames:
        path = os.path.join(template_base_dir, fname)
        if os.path.exists(path):
            template_paths.append(path)
        else:
            # 如果模板文件不存在，记录警告。实际应用中可能需要更严格的错误处理。
            logger.warning(f"模板文件 '{path}' 未找到。")

    if not template_paths:
        logger.critical("没有可用的模板。跟踪无法继续。正在退出。")
        exit()

    # 构建输入输出视频的完整路径
    input_video_path = os.path.join(video_base_dir, input_video_filename)
    base_vid_name, vid_ext = os.path.splitext(input_video_filename)  # 分离视频文件名和扩展名
    output_video_path = os.path.join(output_base_dir, f"tracked_{base_vid_name}{vid_ext}")

    if not os.path.exists(input_video_path):
        logger.error(f"输入视频 '{input_video_path}' 未找到。请确保视频文件存在。")
        exit()

    # 在创建 VideoTracker 实例前获取视频的FPS，以便计算基于时间的参数
    detected_fps = 30.0  # 默认FPS值
    try:
        probe_cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                     '-show_entries', 'stream=avg_frame_rate', '-of', 'csv=p=0', input_video_path]
        fps_str = subprocess.check_output(probe_cmd, text=True, encoding='utf-8').strip()
        if '/' in fps_str:  # 处理如 "30000/1001" 的帧率
            num, den = map(float, fps_str.split('/'))
            if den != 0: detected_fps = num / den
        elif fps_str:  # 处理如 "29.97" 的帧率
            detected_fps = float(fps_str)
        logger.info(f"为模板延迟激活逻辑检测到视频平均FPS: {detected_fps:.2f}")
    except Exception as e:
        logger.warning(f"无法检测视频FPS以进行模板延迟。将使用默认帧号（基于约25-30FPS的估算）。错误: {e}")
        # 如果无法获取FPS，detected_fps 将保持为默认值 30.0

    # ################################################################################
    # ########## 重要：根据视频实际情况和FPS，调整下面的秒数和计算得到的帧号 ##########
    # ################################################################################
    # 定义模板切换阶段的时间点 (以秒为单位)，这些是用户需要根据视频内容调整的关键参数
    seconds_before_turn_phase = 5.0  # 假设约5秒后目标开始转弯或场景变化，需要启用中距离模板
    seconds_before_distant_phase = 10.0  # 假设约10秒后目标变远，需要启用远距离模板

    # 根据检测到的FPS和设定的秒数，计算对应的帧号
    calculated_turn_phase_frame = int(seconds_before_turn_phase * detected_fps)
    calculated_distant_phase_frame = int(seconds_before_distant_phase * detected_fps)

    logger.info(f"基于 {detected_fps:.2f} FPS, 转弯阶段约从第 {calculated_turn_phase_frame} 帧开始。")
    logger.info(f"基于 {detected_fps:.2f} FPS, 远距离阶段约从第 {calculated_distant_phase_frame} 帧开始。")

    try:
        # 创建 VideoTracker 实例，传入所有配置参数
        tracker = VideoTracker(
            template_paths=template_paths,  # 模板路径列表
            # 临时文件目录，加上后缀以区分不同运行
            temp_dir=os.path.join(output_base_dir, f"_temp_{base_vid_name}_processing_video"),

            # 传递计算得到的阶段开始帧号
            turn_phase_starts_at_frame=calculated_turn_phase_frame,
            distant_phase_starts_at_frame=calculated_distant_phase_frame,

            # NCC 相关阈值
            ncc_threshold=0.28,  # 基础NCC接受阈值
            reliable_ncc_threshold=0.38,  # 可靠NCC阈值 (用于KF更新/初始化)
            re_acquire_ncc_threshold=0.50,  # 重新捕获时的NCC阈值 (通常更高)
            initial_lock_threshold=0.60,  # 初始锁定时的高阈值

            # 初始阶段和模板尺寸相关参数
            # 初始宽限期，至少15帧或2秒的帧数，期间使用 initial_lock_threshold
            initial_grace_period_frames=max(15, int(2 * detected_fps)),
            small_template_area_threshold_px=250,  # 小模板面积阈值
            small_template_ncc_boost=0.05,  # 小模板可靠阈值提升 (使小模板更难被确认为可靠)

            # 卡尔曼滤波器相关参数
            kf_measurement_noise_std=40.0,  # 测量噪声标准差 (影响对NCC检测结果的信任度)
            kf_process_noise_std=12.0,  # 过程噪声标准差 (影响KF预测的平滑性/响应速度)
            # 若FPS低，目标帧间位移大，可适当调大此值
            detection_gate_factor=4.5,  # 检测门限因子 (NCC检测结果与KF预测的允许偏差范围)

            # 跟踪鲁棒性参数
            # 最大连续丢失帧数，至少15帧或1秒的帧数，超过则重置KF
            max_consecutive_misses=max(15, int(1.0 * detected_fps)),
            min_hits_to_activate=2,  # KF激活(开始输出结果)前需要的最小命中数
            # 重新捕获超时帧数，至少30帧或1.5秒的帧数
            re_acquire_timeout_frames=max(30, int(1.5 * detected_fps)),

            # 性能与搜索参数
            frame_downsample_factor=1,  # 帧缩放因子 (1为不缩放，>1则缩小以提速)
            search_roi_padding_factor=2.8  # ROI搜索区域相对于模板尺寸的填充倍数
        )
        # 处理视频
        tracker.process_video(input_video_path, output_video_path)
    except ValueError as ve:
        logger.critical(f"VideoTracker 初始化错误: {ve}")
    except RuntimeError as rte:
        logger.critical(f"运行时错误 (可能与FFmpeg相关): {rte}")
    except Exception as e:
        logger.error(f"视频处理过程中发生意外错误: {str(e)}", exc_info=LOGGING_LEVEL == logging.DEBUG)