import cv2 as cv
import numpy as np


def cosine_similarity(roi, template):

    """
    在roi区域中进行滑窗计算与模板的余弦相似度
    :param roi: roi区域图像
    :param template: 模板图像
    :return: 各窗口的余弦相似度值组成的矩阵
    """

    temp_w, temp_h, temp_channel = template.shape
    roi = roi.astype('float64')
    template = template.astype('float64')
    roi_bgr = cv.split(roi)
    temp_bgr = cv.split(template)

    # cv.filter2D函数会进行边缘处理，返回与roi区域等大的图片
    # 计算没有边缘处理的有效区域的上下左右坐标
    top = temp_h // 2
    down = temp_h - 1 - top
    left = temp_w // 2
    right = temp_w - 1 - left

    # 计算roi区域和template的模长
    norm_kernal = np.ones((temp_h, temp_w))
    roi_channel_square = [np.multiply(roi_c, roi_c) for roi_c in roi_bgr]
    roi_channel_square_sum = np.sum(roi_channel_square, axis=0)
    roi_norm = np.sqrt(cv.filter2D(roi_channel_square_sum, -1, norm_kernal)[top:-down, left:-right])
    temp_channel_square = [np.multiply(temp_c, temp_c) for temp_c in temp_bgr]
    temp_channel_square_sum = np.sum(temp_channel_square, axis=0)
    temp_norm = np.sqrt(temp_channel_square_sum.sum())

    # 计算roi区域和template区域的内积
    inner_product = np.sum([cv.filter2D(roi_bgr[i], -1, temp_bgr[i])[top:-down, left:-right] for i in range(3)], axis=0)

    # 计算余弦相似度
    res = inner_product / (roi_norm * temp_norm)

    return res
