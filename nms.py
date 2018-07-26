import numpy as np


def non_maximum_suppression(boxes, IoUThresh):
    """
    按照bbox的置信度进行nms
    :param boxes: (x1,y1,x2,y2,score)
    :param IoUThresh: nms的阈值 
    :return: nms后符合条件的bbox
    """
    # 输入boxes为空则返一个空列表
    if len(boxes) == 0:
        return []

    # 如果输入的数据为int型数据，则将其转化为float
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float64")

    # 得到boxes的坐标和置信度得分
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    score = boxes[:, 4]

    # 计算boxes的面积
    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    # 按照置信度从大到小排序，得到对应的boxes的index
    idxs = np.argsort(score)[::-1]

    # 初始化pick列表，用于存放最终的boxes所对应的index
    pick = []

    # 循环直至idxs中不再有值
    while len(idxs) > 0:
        # 得到最大score对应的实际boxes里的index, 并加入pick
        i = idxs[0]
        pick.append(i)

        # 找到剩余的boxes与该box相交的左上角（xx1,yy1)和右下角（xx2,yy2)
        # 不用担心没有交集，没有交集时，下一步w,h会为0
        xx1 = np.maximum(x1[i], x1[idxs[1:]])
        yy1 = np.maximum(y1[i], y1[idxs[1:]])
        xx2 = np.minimum(x2[i], x2[idxs[1:]])
        yy2 = np.minimum(y2[i], y2[idxs[1:]])

        # 计算相交区域的长宽w,h
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # 计算IoU
        intersection = w * h
        iou = intersection / (area[i] + area[idxs[1:]] - intersection)

        # 删除掉IoU大于阈值的boxes
        idxs = np.delete(idxs, np.concatenate(([0], np.where(iou > IoUThresh)[0])))

    return boxes[pick].astype("int")
