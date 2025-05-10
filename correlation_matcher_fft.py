# correlation_matcher_fft.py
import numpy as np
from scipy.signal import fftconvolve  # 用于通过FFT高效计算卷积
import logging

logger = logging.getLogger(__name__)


def match_template_ncc_fft(image, template):
    """
    使用基于FFT的快速归一化互相关 (NCC) 计算模板在图像上的匹配。
    此函数在较大的 'image' 中搜索 'template'。

    所使用的NCC公式与OpenCV的TM_CCOEFF_NORMED一致：
    NCC(x,y) = sum_{x',y'} (T'(x',y') * I'(x+x', y+y')) /
               sqrt( sum_{x',y'} T'(x',y')^2 * sum_{x',y'} I'(x+x', y+y')^2 )
    其中 T' = T - mean(T)，I' = I_window - mean(I_window)。
    这是通过将公式分解为可通过卷积计算的项来实现的。

    参数:
        image (numpy.ndarray): 执行搜索的灰度图像。
        template (numpy.ndarray): 要搜索的灰度模板图像。

    返回:
        numpy.ndarray or None: 一个二维数组，表示模板在图像中每个可能的左上角位置的NCC得分。
                               得分范围从-1.0到1.0。如果出错或输入无效，则返回None。
    """
    if image is None or template is None or image.ndim != 2 or template.ndim != 2 \
            or image.size == 0 or template.size == 0:
        logger.debug("NCC_FFT: 无效的输入图像或模板。")
        return None

    image_h, image_w = image.shape
    template_h, template_w = template.shape

    if template_h > image_h or template_w > image_w or template_h == 0 or template_w == 0:
        logger.debug(f"NCC_FFT: 模板形状 ({template_h}x{template_w}) 与图像形状 ({image_h}x{image_w}) 不兼容。")
        return None

    # 确保使用浮点数类型以保证计算精度
    image_f = image.astype(np.float64)
    template_f = template.astype(np.float64)

    num_template_pixels = template_h * template_w
    if num_template_pixels == 0: return None  # 应该已被前面的检查捕获

    # --- 预计算与模板相关的项 ---
    template_mean = np.mean(template_f)
    sum_template_sq = np.sum(template_f ** 2)  # sum(t_i^2)

    # 用于计算局部和（移动平均）的核（全1矩阵）
    kernel_ones = np.ones_like(template_f, dtype=np.float64)

    # 'valid' 模式卷积的预期输出形状
    expected_conv_h = image_h - template_h + 1
    expected_conv_w = image_w - template_w + 1

    # --- 使用卷积计算NCC分母和分子所需的项 ---
    # 检查图像是否小于模板（防御性检查）
    if not (image_f.shape[0] >= kernel_ones.shape[0] and image_f.shape[1] >= kernel_ones.shape[1]):
        logger.warning(f"NCC_FFT: 图像形状 {image_f.shape} 对于核 {kernel_ones.shape} 来说太小。")
        return None

    # 图像窗口像素总和: sum(f_win)
    local_sum_image = fftconvolve(image_f, kernel_ones, mode='valid')
    if local_sum_image.shape != (expected_conv_h, expected_conv_w): return None

    # 图像窗口像素均值: f_win_mean = sum(f_win) / N
    image_window_mean = local_sum_image / num_template_pixels

    # 图像窗口像素平方和: sum(f_win^2)
    local_sum_image_sq = fftconvolve(image_f ** 2, kernel_ones, mode='valid')
    if local_sum_image_sq.shape != (expected_conv_h, expected_conv_w): return None

    # 互相关项: sum(f_win * t)
    # 对于通过卷积计算互相关，模板需要被翻转。
    template_flipped = template_f[::-1, ::-1]
    if not (image_f.shape[0] >= template_flipped.shape[0] and image_f.shape[1] >= template_flipped.shape[1]):
        return None

    cross_correlation_term = fftconvolve(image_f, template_flipped, mode='valid')
    if cross_correlation_term.shape != (expected_conv_h, expected_conv_w): return None

    # --- 组装NCC分子 ---
    # 分子 = sum(f_win*t) - N * f_win_mean * t_mean
    #      = cross_correlation_term - local_sum_image * template_mean
    # (因为 local_sum_image = N * image_window_mean)
    numerator = cross_correlation_term - local_sum_image * template_mean

    # --- 组装NCC分母项 ---
    # 图像窗口方差 * N: sum( (f_win_i - f_win_mean)^2 ) = sum(f_win_i^2) - N * f_win_mean^2
    variance_image_window_N = local_sum_image_sq - num_template_pixels * (image_window_mean ** 2)
    variance_image_window_N = np.maximum(0, variance_image_window_N)  # 确保由于精度问题不会出现负值

    # 模板方差 * N: sum( (t_i - t_mean)^2 ) = sum(t_i^2) - N * t_mean^2
    variance_template_N = sum_template_sq - num_template_pixels * (template_mean ** 2)
    variance_template_N = np.maximum(0, variance_template_N)  # 确保非负

    # 分母 = sqrt( variance_image_window_N * variance_template_N )
    denominator = np.sqrt(variance_image_window_N * variance_template_N)

    epsilon = 1e-7  # 防止除以零的小值

    ncc_map = np.zeros_like(numerator, dtype=np.float64)
    valid_mask = denominator > epsilon  # 创建有效分母的掩码

    if np.any(valid_mask):  # 仅在分母有效的地方计算NCC
        ncc_map[valid_mask] = numerator[valid_mask] / denominator[valid_mask]

    # 将值裁剪到理论上的NCC范围 [-1, 1]
    return np.clip(ncc_map, -1.0, 1.0)