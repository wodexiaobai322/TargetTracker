# Combined Video Tracker File (with video output, no trailing semicolons)

import numpy as np
import cv2
import sys
from collections import deque
import time
import traceback  # For detailed error reporting


# --- Content from correlation_matcher.py ---
def match_template_ncc_custom(image, template):
    if image is None or template is None or image.ndim != 2 or template.ndim != 2:
        return None
    img_h, img_w = image.shape
    tmpl_h, tmpl_w = template.shape
    if tmpl_h > img_h or tmpl_w > img_w:
        return None
    result_h = img_h - tmpl_h + 1
    result_w = img_w - tmpl_w + 1
    ncc_map = np.full((result_h, result_w), -1.0, dtype=np.float64)
    template_f64 = template.astype(np.float64)
    template_mean = np.mean(template_f64)
    template_std = np.std(template_f64)
    epsilon = 1e-6
    if template_std < epsilon:
        return ncc_map
    template_norm = template_f64 - template_mean
    image_f64 = image.astype(np.float64)
    for y in range(result_h):
        for x in range(result_w):
            patch = image_f64[y:y + tmpl_h, x:x + tmpl_w]
            patch_mean = np.mean(patch)
            patch_std = np.std(patch)
            if patch_std < epsilon:
                ncc_map[y, x] = 0.0
                continue
            patch_norm = patch - patch_mean
            correlation = np.sum(patch_norm * template_norm)
            norm_factor = (tmpl_h * tmpl_w) * patch_std * template_std
            if norm_factor < epsilon:
                ncc_map[y, x] = 0.0
            else:
                ncc_val = correlation / norm_factor
                ncc_map[y, x] = np.clip(ncc_val, -1.0, 1.0)
    return ncc_map


# --- Content from tracker_utils.py ---
selecting_roi = False
roi_start_point = None
roi_end_point = None
selected_bbox = None


def _roi_select_callback(event, x, y, flags, param):
    global selecting_roi, roi_start_point, roi_end_point, selected_bbox
    if 'frame' not in param or param['frame'] is None: return
    frame_copy = param['frame'].copy()
    if event == cv2.EVENT_LBUTTONDOWN:
        selecting_roi = True
        roi_start_point = (x, y)
        roi_end_point = (x, y)
        selected_bbox = None
    elif event == cv2.EVENT_MOUSEMOVE:
        if selecting_roi:
            roi_end_point = (x, y)
            try:
                cv2.rectangle(frame_copy, roi_start_point, roi_end_point, (0, 255, 0), 1)
                cv2.imshow(param['window_name'], frame_copy)
            except cv2.error as e:
                print(f"警告 (_roi_select_callback): 绘制或显示时出错: {e}")
    elif event == cv2.EVENT_LBUTTONUP:
        selecting_roi = False
        roi_end_point = (x, y)
        x1 = min(roi_start_point[0], roi_end_point[0])
        y1 = min(roi_start_point[1], roi_end_point[1])
        x2 = max(roi_start_point[0], roi_end_point[0])
        y2 = max(roi_start_point[1], roi_end_point[1])
        w = x2 - x1
        h = y2 - y1
        if w > 0 and h > 0:
            selected_bbox = (x1, y1, w, h)
            try:
                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame_copy, "Selection OK. Press ENTER/SPACE", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 255), 1)
                cv2.imshow(param['window_name'], frame_copy)
            except cv2.error as e:
                print(f"警告 (_roi_select_callback): 绘制或显示确认框时出错: {e}")
                selected_bbox = None
        else:
            print("选择的 ROI 无效。")
            selected_bbox = None
            try:
                cv2.imshow(param['window_name'], param['frame'])
            except cv2.error as e:
                print(f"警告 (_roi_select_callback): 重置显示时出错: {e}")


def select_target_roi(frame, window_name="Select Target ROI"):
    global selecting_roi, roi_start_point, roi_end_point, selected_bbox
    selecting_roi = False
    roi_start_point = None
    roi_end_point = None
    selected_bbox = None
    if frame is None or frame.size == 0:
        print("错误 (select_target_roi): 输入帧无效，无法选择 ROI。")
        return None
    original_window_exists = False
    try:
        cv2.namedWindow(window_name)
        original_window_exists = True
        callback_param = {'frame': frame, 'window_name': window_name}
        cv2.setMouseCallback(window_name, _roi_select_callback, callback_param)
        print("\n请在窗口中用鼠标左键拖拽选择目标区域，完成后按 Enter 或 Space 确认，Esc 取消。")
        while True:
            if not selecting_roi and roi_start_point is None:
                try:
                    cv2.imshow(window_name, frame)
                except cv2.error as e:
                    print(f"错误 (select_target_roi): 初始显示失败: {e}")
                    return None
            key = cv2.waitKey(20) & 0xFF
            if key == 13 or key == 32:  # Enter or Space
                if selected_bbox is not None:
                    print("选择已确认。")
                    break
                else:
                    print("尚未选择有效区域。")
            elif key == 27:  # Esc
                print("选择已取消。")
                selected_bbox = None
                break
    except cv2.error as e:
        print(f"错误 (select_target_roi): OpenCV 窗口操作失败: {e}")
        selected_bbox = None
    except Exception as e:
        print(f"错误 (select_target_roi): 发生未知错误: {e}")
        selected_bbox = None
    finally:
        if original_window_exists:
            try:
                cv2.destroyWindow(window_name)
            except cv2.error:
                pass
    return selected_bbox


def draw_bounding_box(frame, bbox, color=(0, 255, 0), thickness=2, status_text=""):
    if frame is None: return frame
    display_frame = frame
    if bbox is not None:
        try:
            x, y, w, h = map(int, bbox)
            fh, fw = frame.shape[:2]
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(fw, x + w), min(fh, y + h)
            if x2 > x1 and y2 > y1:
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, thickness)
                if status_text:
                    (tw, th), bl = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    text_y = y1 - 10 if y1 - 10 > th else y1 + th + bl
                    text_y = max(th + bl, text_y)
                    text_y = min(text_y, fh - bl)
                    cv2.putText(display_frame, status_text, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                                cv2.LINE_AA)
        except Exception as e:
            print(f"警告 (draw_bounding_box): 绘制出错 - bbox={bbox}, error={e}")
    return display_frame


# --- Content from video_tracker.py ---
class CorrelationTracker:
    HIGH_SCORE_THRESHOLD_FOR_EARLY_EXIT = 0.95
    MAX_CONSECUTIVE_MISSES = 3

    def __init__(self, confidence_threshold=0.5, update_threshold=0.7,
                 search_window_scale=2.5, scales=[0.9, 1.0, 1.1],
                 min_variance_ratio_threshold=0.4, use_clahe=True,
                 clahe_clip_limit=2.0, clahe_tile_grid_size=(8, 8),
                 max_template_pool_size=5, disable_variance_check_with_clahe=True,
                 target_dims_update_rate=0.1, min_target_dims=(10, 10)):
        self.templates = deque(maxlen=max_template_pool_size)
        self.current_bbox = None
        self.target_visible = False
        self.initialized = False
        self.trajectory = []
        self.last_bbox = None
        self.initial_template_dims = None
        self.current_target_dims = None
        self.consecutive_misses = 0
        self.confidence_threshold = confidence_threshold
        self.update_threshold = update_threshold
        self.search_window_scale = search_window_scale
        self.scales = scales
        self.min_variance_ratio_threshold = min_variance_ratio_threshold
        self.use_clahe_param = use_clahe
        self.max_template_pool_size = max_template_pool_size
        self.target_dims_update_rate = target_dims_update_rate
        self.min_target_dims = min_target_dims
        self.clahe = None
        clahe_created_successfully = False
        if self.use_clahe_param:
            try:
                self.clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_grid_size)
                clahe_created_successfully = True
            except cv2.error as e:
                print(f"警告: 创建 CLAHE 失败 ({e})")
                self.clahe = None
        self.disable_variance_check = disable_variance_check_with_clahe and clahe_created_successfully
        print("-" * 30)
        print("跟踪器配置 (CPU):")
        print(f"  使用 CLAHE: {'是' if clahe_created_successfully else '否'}")
        print(f"  置信阈值(单帧): {self.confidence_threshold}")
        print(f"  置信阈值(更新): {self.update_threshold}")
        print(f"  最大模板池: {self.max_template_pool_size}")
        print(f"  搜索窗口系数: {self.search_window_scale}")
        print(f"  搜索尺度: {self.scales}")
        if not self.disable_variance_check:
            print(f"  方差检查: 是 (阈值: {self.min_variance_ratio_threshold})")
        else:
            print("  方差检查: 否")
        print(f"  提前退出阈值: {self.HIGH_SCORE_THRESHOLD_FOR_EARLY_EXIT}")
        print(f"  尺寸更新率: {self.target_dims_update_rate}")
        print(f"  最小目标尺寸: {self.min_target_dims}")
        print(f"  最大连续丢失: {self.MAX_CONSECUTIVE_MISSES}")
        print("-" * 30)

    def initialize(self, first_frame, initial_bbox):
        if initial_bbox is None or first_frame is None: return False
        x, y, w, h = map(int, initial_bbox)
        frame_h, frame_w = first_frame.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(frame_w, x + w), min(frame_h, y + h)
        w_act, h_act = x2 - x1, y2 - y1
        if w_act <= 0 or h_act <= 0: return False
        gray_cpu = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        proc_gray_cpu = gray_cpu
        if self.clahe is not None:
            try:
                proc_gray_cpu = self.clahe.apply(gray_cpu)
            except cv2.error as e:
                print(f"警告: 初始化 CLAHE 失败: {e}.")
        init_tmpl_cpu = proc_gray_cpu[y1:y2, x1:x2].copy()
        if init_tmpl_cpu is None or init_tmpl_cpu.size == 0: return False
        self.initial_template_dims = init_tmpl_cpu.shape[:2]
        self.current_target_dims = self.initial_template_dims
        self.templates.clear()
        self.templates.append(init_tmpl_cpu)
        tmpl_var = np.var(init_tmpl_cpu.astype(np.float64))
        if tmpl_var < 1.0: print(f"警告: 初始模板方差低 ({tmpl_var:.2f}).")
        self.current_bbox = (x1, y1, w_act, h_act)
        self.last_bbox = self.current_bbox
        self.target_visible = True
        self.initialized = True
        self.trajectory = [self._get_center(self.current_bbox)]
        self.consecutive_misses = 0
        print(f"跟踪器初始化成功。尺寸: {w_act}x{h_act} (池: {len(self.templates)})")
        return True

    def _get_center(self, bbox):
        if bbox is None: return None
        x, y, w, h = map(int, bbox)
        return (x + w // 2, y + h // 2)

    def update(self, frame):
        if not self.initialized or not self.templates:
            self.target_visible = False
            return self.current_bbox
        self.last_bbox = self.current_bbox
        gray_cpu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        proc_gray_cpu = gray_cpu
        if self.clahe is not None:
            try:
                proc_gray_cpu = self.clahe.apply(gray_cpu)
            except cv2.error as e:
                print(f"警告: CLAHE 应用失败: {e}.")
        frame_h, frame_w = proc_gray_cpu.shape
        epsilon = 1e-6
        pred_center = self._get_center(self.current_bbox)
        if self.last_bbox is not None and self.current_bbox != self.last_bbox:
            last_center = self._get_center(self.last_bbox)
            if last_center is not None and pred_center is not None:
                dx = pred_center[0] - last_center[0]
                dy = pred_center[1] - last_center[1]
                pred_center = (pred_center[0] + dx, pred_center[1] + dy)
        if pred_center is None: pred_center = self._get_center(self.current_bbox)
        if pred_center is None:
            self.target_visible = False
            self.consecutive_misses += 1
            return self.current_bbox

        calc_h, calc_w = (0, 0)
        if self.current_target_dims and self.target_dims_update_rate > 0:
            calc_h, calc_w = self.current_target_dims
        elif self.initial_template_dims:
            calc_h, calc_w = self.initial_template_dims
        elif self.current_bbox:
            calc_w, calc_h = self.current_bbox[2], self.current_bbox[3]
        else:
            calc_h, calc_w = (50, 50)
        if calc_h <= 0 or calc_w <= 0: calc_h, calc_w = (50, 50)

        max_scale = max(self.scales) if self.scales else 1.0
        search_w = int(calc_w * max_scale * self.search_window_scale)
        search_h = int(calc_h * max_scale * self.search_window_scale)
        min_search_w = int(calc_w * max_scale * 1.5)
        min_search_h = int(calc_h * max_scale * 1.5)
        search_w = max(search_w, min_search_w, 1)
        search_h = max(search_h, min_search_h, 1)
        search_x = int(pred_center[0] - search_w / 2)
        search_y = int(pred_center[1] - search_h / 2)
        search_x1 = max(0, search_x)
        search_y1 = max(0, search_y)
        search_x2 = min(frame_w, search_x + search_w)
        search_y2 = min(frame_h, search_y + search_h)
        search_w_actual = search_x2 - search_x1
        search_h_actual = search_y2 - search_y1
        if search_w_actual <= 0 or search_h_actual <= 0:
            self.target_visible = False
            self.consecutive_misses += 1
            return self.current_bbox
        search_region_cpu = proc_gray_cpu[search_y1:search_y2, search_x1:search_x2]
        search_offset_x = search_x1
        search_offset_y = search_y1
        overall_best_score = -1.0
        overall_best_bbox = None
        overall_best_patch_cpu = None
        best_tmpl_idx = -1
        for t_idx, cur_tmpl_cpu in reversed(list(enumerate(self.templates))):
            tmpl_h, tmpl_w = cur_tmpl_cpu.shape[:2]
            if tmpl_h == 0 or tmpl_w == 0: continue
            cur_best_score = -1.0
            cur_best_bbox = None
            cur_best_patch = None
            for scale in self.scales:
                sc_h = int(tmpl_h * scale)
                sc_w = int(tmpl_w * scale)
                if not (sc_h > 0 and sc_w > 0 and sc_h <= search_h_actual and sc_w <= search_w_actual): continue
                try:
                    sc_tmpl_cpu = cv2.resize(cur_tmpl_cpu, (sc_w, sc_h), interpolation=cv2.INTER_LINEAR)
                except cv2.error:
                    continue
                try:
                    ncc_map = match_template_ncc_custom(search_region_cpu, sc_tmpl_cpu)
                except Exception as e:
                    print(f"错误: 匹配出错: {e}")
                    ncc_map = None
                if ncc_map is not None and ncc_map.size > 0:
                    min_v, max_v, min_l, max_l = cv2.minMaxLoc(ncc_map)
                    score = max_v
                    loc_rel = max_l
                    if score > cur_best_score:
                        cur_best_score = score
                        abs_x = search_offset_x + loc_rel[0]
                        abs_y = search_offset_y + loc_rel[1]
                        cur_best_bbox = (abs_x, abs_y, sc_w, sc_h)
                        p_y1 = abs_y
                        p_y2 = min(frame_h, abs_y + sc_h)
                        p_x1 = abs_x
                        p_x2 = min(frame_w, abs_x + sc_w)
                        if p_y2 > p_y1 and p_x2 > p_x1:
                            cur_best_patch = proc_gray_cpu[p_y1:p_y2, p_x1:p_x2]
                        else:
                            cur_best_patch = None
            if cur_best_score > overall_best_score:
                overall_best_score = cur_best_score
                overall_best_bbox = cur_best_bbox
                overall_best_patch_cpu = cur_best_patch
                best_tmpl_idx = t_idx
                if overall_best_score >= self.HIGH_SCORE_THRESHOLD_FOR_EARLY_EXIT: break
        patch_var = 0.0
        tmpl_var = 0.0
        var_ratio = 0.0
        is_var_ok = False
        perform_var_check = not self.disable_variance_check
        if perform_var_check and overall_best_patch_cpu is not None and overall_best_patch_cpu.size > 0 and best_tmpl_idx != -1:
            try:
                patch_var = np.var(overall_best_patch_cpu.astype(np.float64))
                best_tmpl_cpu = self.templates[best_tmpl_idx]
                if best_tmpl_cpu is not None and best_tmpl_cpu.size > 0:
                    tmpl_var = np.var(best_tmpl_cpu.astype(np.float64))
                    if tmpl_var > epsilon: var_ratio = patch_var / (tmpl_var + epsilon)
                    is_var_ok = (var_ratio >= self.min_variance_ratio_threshold)
            except IndexError:
                print(f"警告: 访问模板索引 {best_tmpl_idx} 失败.")
                is_var_ok = False
            except Exception as e:
                print(f"警告: 方差计算出错: {e}")
                is_var_ok = False
        elif not perform_var_check:
            is_var_ok = True
        single_frame_match_ok = (
                    overall_best_score >= self.confidence_threshold and overall_best_bbox is not None and is_var_ok)
        if single_frame_match_ok:
            self.consecutive_misses = 0
            self.target_visible = True
            new_h, new_w = int(overall_best_bbox[3]), int(overall_best_bbox[2])
            if self.target_dims_update_rate > 0 and self.current_target_dims:
                cur_h, cur_w = self.current_target_dims
                up_h = int(self.target_dims_update_rate * new_h + (1 - self.target_dims_update_rate) * cur_h)
                up_w = int(self.target_dims_update_rate * new_w + (1 - self.target_dims_update_rate) * cur_w)
                min_h, min_w = self.min_target_dims
                self.current_target_dims = (max(up_h, min_h), max(up_w, min_w))
            elif self.target_dims_update_rate > 0:
                self.current_target_dims = (max(new_h, self.min_target_dims[0]), max(new_w, self.min_target_dims[1]))
            best_cx = overall_best_bbox[0] + overall_best_bbox[2] / 2
            best_cy = overall_best_bbox[1] + overall_best_bbox[3] / 2
            if self.target_dims_update_rate > 0 and self.current_target_dims:
                adapt_h, adapt_w = self.current_target_dims
            else:
                adapt_h, adapt_w = new_h, new_w
            new_x = int(best_cx - adapt_w / 2)
            new_y = int(best_cy - adapt_h / 2)
            new_x = max(0, new_x)
            new_y = max(0, new_y)
            new_w_c = min(adapt_w, frame_w - new_x)
            new_h_c = min(adapt_h, frame_h - new_y)
            self.current_bbox = (new_x, new_y, new_w_c, new_h_c)
            new_center = self._get_center(self.current_bbox)
            if new_center: self.trajectory.append(new_center)
            if overall_best_score >= self.update_threshold and overall_best_patch_cpu is not None and overall_best_patch_cpu.size > 0:
                if self.initial_template_dims:
                    tgt_h, tgt_w = self.initial_template_dims
                    patch_h, patch_w = overall_best_patch_cpu.shape[:2]
                    bbox_w_c, bbox_h_c = overall_best_bbox[2], overall_best_bbox[3]
                    if abs(patch_h - bbox_h_c) < 5 and abs(patch_w - bbox_w_c) < 5 and patch_h > 0 and patch_w > 0:
                        try:
                            up_patch_cpu = cv2.resize(overall_best_patch_cpu, (tgt_w, tgt_h),
                                                      interpolation=cv2.INTER_LINEAR)
                            self.templates.append(up_patch_cpu.astype(np.uint8))
                        except Exception as e:
                            print(f"警告: 更新模板池出错: {e}")
        else:
            self.consecutive_misses += 1
            if self.consecutive_misses >= self.MAX_CONSECUTIVE_MISSES:
                self.target_visible = False
        # print(f"DEBUG: Score={overall_best_score:.4f} | VarOK={is_var_ok} | Misses={self.consecutive_misses} | Visible={self.target_visible}")
        return self.current_bbox

    def get_status(self):
        return self.target_visible, self.current_bbox

    def get_trajectory(self):
        return self.trajectory


# --- Main Application (formerly main.py) ---
VIDEO_PATH = './src/videos/rider.mp4'  # Replace with your video path or 0 for webcam
# VIDEO_PATH = 0 # For webcam
FRAME_SKIP = 1

# --- **NEW: Video Output Configuration** ---
OUTPUT_VIDEO_FILENAME = "./outputs/tracked_rider.mp4"  # or .avi
# For .mp4, 'MP4V' or 'avc1' (H.264 if available). For .avi, 'XVID' is common.
OUTPUT_VIDEO_CODEC = 'MP4V'  # Try 'XVID' if 'MP4V' fails or gives errors.

tracker_params = {
    'confidence_threshold': 0.6,
    'disable_variance_check_with_clahe': False,
    'min_variance_ratio_threshold': 0.15,
    'max_template_pool_size': 1,
    'scales': [1.0],
    'search_window_scale': 2.0,
    'use_clahe': False,
    'target_dims_update_rate': 0.0,  # 0.0 means fixed size after init
    'update_threshold': 0.65,
    'min_target_dims': (15, 15),
    'clahe_clip_limit': 2.0,
    'clahe_tile_grid_size': (8, 8)
}


def run():
    video_source = VIDEO_PATH
    if isinstance(VIDEO_PATH, str) and VIDEO_PATH.isdigit():
        video_source = int(VIDEO_PATH)

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"错误: 无法打开视频: {video_source}")
        sys.exit()

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if (fps is not None and 0 < fps <= 1000) else 30.0
    except:
        fps = 30.0
    print(f"视频加载: {frame_width}x{frame_height} @ {fps:.2f} FPS, Frame Skip: {FRAME_SKIP}")
    frame_delay_ms = int(1000 / fps) if fps > 0 else 30

    tracker = None
    tracking_active = False
    initial_bbox = None
    last_good_bbox = None
    total_processing_time = 0
    processed_frame_count = 0
    frame_count = 0
    window_title = "Video Tracking (CPU - Combined)"

    out_video = None  # --- **NEW: VideoWriter object** ---

    while True:
        ret, frame = cap.read()
        if not ret:
            print("视频结束或无法读取帧。")
            break
        frame_count += 1

        if frame_count % FRAME_SKIP != 0:
            continue

        display_frame = frame.copy()
        start_time = time.time()
        current_bbox_to_draw = None
        box_color = (0, 0, 0)
        status_text = ""

        if tracking_active and tracker is not None:
            current_bbox_from_tracker = tracker.update(frame)
            target_visible, _ = tracker.get_status()
            processing_time = time.time() - start_time
            total_processing_time += processing_time
            processed_frame_count += 1

            if target_visible and current_bbox_from_tracker is not None:
                box_color = (0, 255, 0)
                status_text = "Tracking"
                if tracker_params.get('target_dims_update_rate', 0.1) == 0.0:
                    status_text += " (Fixed Size)"
                current_bbox_to_draw = current_bbox_from_tracker
                last_good_bbox = current_bbox_from_tracker
            else:
                box_color = (0, 0, 255)
                status_text = "Target Lost"
                if tracker and tracker.consecutive_misses < tracker.MAX_CONSECUTIVE_MISSES and not target_visible:
                    status_text = f"Searching... (Miss: {tracker.consecutive_misses})"
                current_bbox_to_draw = last_good_bbox

        elif not tracking_active:
            prompt_text = "Press 's' to select, 'q' to quit"
            cv2.putText(display_frame, prompt_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            current_bbox_to_draw = None
            last_good_bbox = None

        # --- 绘制 ---
        if current_bbox_to_draw is not None:
            draw_bbox_int = tuple(map(int, current_bbox_to_draw))
            draw_bounding_box(display_frame, draw_bbox_int, box_color, status_text=status_text)

        if tracker and tracking_active:  # Only draw trajectory if tracking
            trajectory_points = tracker.get_trajectory()
            if len(trajectory_points) >= 2:
                try:
                    pts = np.array(trajectory_points, np.int32).reshape((-1, 1, 2))
                    cv2.polylines(display_frame, [pts], isClosed=False, color=(255, 100, 0), thickness=2)
                except Exception as e:
                    print(f"警告: 绘制轨迹出错: {e}")

        if processed_frame_count > 0 and total_processing_time > 0:
            avg_fps = processed_frame_count / total_processing_time
            cv2.putText(display_frame, f"Proc FPS: {avg_fps:.1f}", (frame_width - 200, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 255), 2)
        elif tracking_active:
            cv2.putText(display_frame, "Proc FPS: 0.0", (frame_width - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 255), 2)

        # --- **NEW: Write frame to output video if recording** ---
        if tracking_active and out_video is not None and out_video.isOpened():
            try:
                out_video.write(display_frame)
            except Exception as e:
                print(f"错误: 写入视频帧失败: {e}")
                out_video.release()  # Stop trying to write if it fails
                out_video = None

        cv2.imshow(window_title, display_frame)

        elapsed_time_ms = int((time.time() - start_time) * 1000)
        wait_time = max(1, frame_delay_ms - elapsed_time_ms)
        key = cv2.waitKey(wait_time) & 0xFF

        if key == ord('q'):
            print("用户请求退出...")
            break
        elif key == ord('p'):
            print("暂停...")
            paused_text = "Paused"
            text_font = cv2.FONT_HERSHEY_SIMPLEX
            text_scale = 1
            text_thickness = 3
            paused_text_size, _ = cv2.getTextSize(paused_text, text_font, text_scale, text_thickness)
            paused_text_x = (frame_width - paused_text_size[0]) // 2
            paused_text_y = (frame_height + paused_text_size[1]) // 2
            pause_display = display_frame.copy()
            cv2.putText(pause_display, paused_text, (paused_text_x, paused_text_y), text_font, text_scale,
                        (0, 255, 255), text_thickness)
            cv2.imshow(window_title, pause_display)
            while True:
                pause_key = cv2.waitKey(0) & 0xFF
                if pause_key == ord('q'):
                    print("用户在暂停时请求退出...")
                    if out_video is not None: out_video.release()
                    cap.release()
                    cv2.destroyAllWindows()
                    sys.exit()
                if pause_key != -1: break
            print("继续...")
            total_processing_time = 0
            processed_frame_count = 0

        elif key == ord('s') and not tracking_active:
            print("\n请求选择目标...")
            initial_bbox_frame = frame.copy()
            roi_window_name = f"{window_title} - Select ROI"
            initial_bbox = select_target_roi(initial_bbox_frame, window_name=roi_window_name)

            if initial_bbox is not None and initial_bbox[2] > 0 and initial_bbox[3] > 0:
                print(f"用户选择: {initial_bbox}")
                try:
                    tracker = CorrelationTracker(**tracker_params)
                    if tracker.initialize(frame, initial_bbox):
                        tracking_active = True
                        last_good_bbox = tracker.current_bbox
                        print("跟踪器初始化成功!")
                        total_processing_time = 0
                        processed_frame_count = 0

                        if out_video is None:
                            output_fps_val = float(fps)
                            fourcc = cv2.VideoWriter_fourcc(*OUTPUT_VIDEO_CODEC)
                            out_video = cv2.VideoWriter(OUTPUT_VIDEO_FILENAME, fourcc, output_fps_val,
                                                        (frame_width, frame_height))
                            if not out_video.isOpened():
                                print(
                                    f"错误: 无法打开 VideoWriter for {OUTPUT_VIDEO_FILENAME}. Codec: {OUTPUT_VIDEO_CODEC}, FPS: {output_fps_val}, Size: ({frame_width},{frame_height})")
                                out_video = None
                            else:
                                print(f"开始录制追踪视频到: {OUTPUT_VIDEO_FILENAME}")
                    else:
                        print("错误: 跟踪器初始化失败。")
                        tracker = None
                        tracking_active = False
                        last_good_bbox = None
                except Exception as e:
                    print(f"错误: 创建或初始化 Tracker 时出错: {e}")
                    traceback.print_exc()
                    tracker = None
                    tracking_active = False
                    last_good_bbox = None
            else:
                print("目标选择取消或无效。")
                tracker = None
                tracking_active = False
                last_good_bbox = None
            cv2.imshow(window_title, display_frame)

    print("处理结束。")
    if processed_frame_count > 0 and total_processing_time > 0:
        print(f"平均处理速度: {processed_frame_count / total_processing_time:.2f} FPS")
    else:
        print("未处理任何有效跟踪帧或处理时间为零。")

    if out_video is not None:
        print(f"完成录制，视频已保存到: {OUTPUT_VIDEO_FILENAME}")
        out_video.release()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        run()
    except SystemExit:
        pass
    except Exception as e:
        print(f"\n发生未捕获的异常: {e}")
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()