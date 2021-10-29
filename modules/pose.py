import time

import cv2
import numpy as np
import numba as nb

from modules.keypoints import BODY_PARTS_KPT_IDS, BODY_PARTS_PAF_IDS
from modules.one_euro_filter import OneEuroFilter



@nb.jit()
def get_true_angel(value):
    '''
    转转得到角度值
    '''
    return value/np.pi*180

@nb.jit()
def get_angle(x1, y1, x2, y2):
    '''
    计算旋转角度
    '''
    dx = abs(x1- x2)
    dy = abs(y1- y2)
    result_angele = 0
    if x1 == x2:
        if y1 > y2:
            result_angele = 180
    else:
        if y1!=y2:
            the_angle = int(get_true_angel(np.arctan(dx/dy)))
        if x1 < x2:
            if y1>y2:
                result_angele = -(180 - the_angle)
            elif y1<y2:
                result_angele = -the_angle
            elif y1==y2:
                result_angele = -90
        elif x1 > x2:
            if y1>y2:
                result_angele = 180 - the_angle
            elif y1<y2:
                result_angele = the_angle
            elif y1==y2:
                result_angele = 90
    
    if result_angele<0:
        result_angele = 360 + result_angele
    return result_angele

def rotate_bound(image, angle, key_point_y):
    '''
    旋转图像，并取得关节点偏移量
    '''
    #获取图像的尺寸
    (h,w) = image.shape[:2]
    #旋转中心
    (cx,cy) = (w/2,h/2)
    # 关键点必须在中心的y轴上
    (kx,ky) = cx, key_point_y
    d = abs(ky - cy)
    
    #设置旋转矩阵
    M = cv2.getRotationMatrix2D((cx,cy), -angle, 1.0)
    cos = np.abs(M[0,0])
    sin = np.abs(M[0,1])
    
    # 计算图像旋转后的新边界
    nW = int((h*sin)+(w*cos))
    nH = int((h*cos)+(w*sin))
    
    # 计算旋转后的相对位移
    move_x = nW/2 + np.sin(angle/180*np.pi)*d 
    move_y = nH/2 - np.cos(angle/180*np.pi)*d
    
    # 调整旋转矩阵的移动距离（t_{x}, t_{y}）
    M[0,2] += (nW/2) - cx
    M[1,2] += (nH/2) - cy

    return cv2.warpAffine(image,M,(nW,nH)), int(move_x), int(move_y)

@nb.jit()
def get_distences(x1, y1, x2, y2):
    return ((x1-x2)**2 + (y1-y2)**2)**0.5


class Pose:
    num_kpts = 18
    kpt_names = ['nose', 'neck',
                 'r_sho', 'r_elb', 'r_wri', 'l_sho', 'l_elb', 'l_wri',
                 'r_hip', 'r_knee', 'r_ank', 'l_hip', 'l_knee', 'l_ank',
                 'r_eye', 'l_eye',
                 'r_ear', 'l_ear']
    sigmas = np.array([.26, .79, .79, .72, .62, .79, .72, .62, 1.07, .87, .89, 1.07, .87, .89, .25, .25, .35, .35],
                      dtype=np.float32) / 10.0
    vars = (sigmas * 2) ** 2
    last_id = -1
    color = [0, 224, 255]

    def __init__(self, keypoints, confidence):
        super().__init__()
        self.keypoints = keypoints
        self.confidence = confidence
        self.bbox = Pose.get_bbox(self.keypoints)
        self.id = None
        self.filters = [[OneEuroFilter(), OneEuroFilter()] for _ in range(Pose.num_kpts)]

    @staticmethod
    def get_bbox(keypoints):
        found_keypoints = np.zeros((np.count_nonzero(keypoints[:, 0] != -1), 2), dtype=np.int32)
        found_kpt_id = 0
        for kpt_id in range(Pose.num_kpts):
            if keypoints[kpt_id, 0] == -1:
                continue
            found_keypoints[found_kpt_id] = keypoints[kpt_id]
            found_kpt_id += 1
        bbox = cv2.boundingRect(found_keypoints)
        return bbox

    def update_id(self, id=None):
        self.id = id
        if self.id is None:
            self.id = Pose.last_id + 1
            Pose.last_id += 1

    def draw(self, img):
        assert self.keypoints.shape == (Pose.num_kpts, 2)

        for part_id in range(len(BODY_PARTS_PAF_IDS) - 2):
            kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0]
            global_kpt_a_id = self.keypoints[kpt_a_id, 0]
            if global_kpt_a_id != -1:
                x_a, y_a = self.keypoints[kpt_a_id]
                cv2.circle(img, (int(x_a), int(y_a)), 3, Pose.color, -1)
            kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]
            global_kpt_b_id = self.keypoints[kpt_b_id, 0]
            if global_kpt_b_id != -1:
                x_b, y_b = self.keypoints[kpt_b_id]
                cv2.circle(img, (int(x_b), int(y_b)), 3, Pose.color, -1)
            if global_kpt_a_id != -1 and global_kpt_b_id != -1:
                cv2.line(img, (int(x_a), int(y_a)), (int(x_b), int(y_b)), Pose.color, 2)


def get_similarity(a, b, threshold=0.5):
    return get_similarity_(a.keypoints, b.keypoints, a.bbox, b.bbox, Pose.num_kpts, Pose.vars, threshold)
    '''
    num_similar_kpt = 0
    for kpt_id in range(Pose.num_kpts):
        if a.keypoints[kpt_id, 0] != -1 and b.keypoints[kpt_id, 0] != -1:
            distance = np.sum((a.keypoints[kpt_id] - b.keypoints[kpt_id]) ** 2)
            area = max(a.bbox[2] * a.bbox[3], b.bbox[2] * b.bbox[3])
            similarity = np.exp(-distance / (2 * (area + np.spacing(1)) * Pose.vars[kpt_id]))
            if similarity > threshold:
                num_similar_kpt += 1
    return num_similar_kpt
    '''
@nb.jit(nopython=True)
def get_similarity_(a, b, ab, bb, num_kpts, vars,threshold):
    num_similar_kpt = 0
    for kpt_id in range(num_kpts):
        if a[kpt_id, 0] != -1 and b[kpt_id, 0] != -1:
            distance = np.sum((a[kpt_id] - b[kpt_id]) ** 2)
            area = max(ab[2] * ab[3], bb[2] * bb[3])
            similarity = np.exp(-distance / (2 * (area + np.spacing(1)) * vars[kpt_id]))
            if similarity > threshold:
                num_similar_kpt += 1
    return num_similar_kpt

@nb.jit()
def img_mask(img, append_image, zero_x, zero_y, r, img_width, img_height):
    for i in range(0, r.shape[0]):
        for j in range(0, r.shape[1]):
            if 230>r[i][j]>200 and 0<=zero_y+i<img_height and 0<=zero_x+j<img_width:
                img[zero_y+i][zero_x+j] = append_image[i][j]


def append_img_by_sk_points(img, append_image, key_point_y, first_point, second_point, append_img_reset_width=None,
                                        append_img_max_height_rate=1, middle_flip=False, append_img_max_height=None):
    '''
    将需要添加的肢体图片进行缩放
    '''
    if first_point[0] == -1 or second_point[0] == -1:
        return img
    #width, height = img.shape[:2]

    # 根据长度进行缩放
    sk_height = int(get_distences(first_point[0], first_point[1], second_point[0], second_point[1])*append_img_max_height_rate)
    # 缩放制约
    if append_img_max_height:
        sk_height = min(sk_height, append_img_max_height)

    sk_width = int(sk_height/append_image.shape[0]*append_image.shape[1]) if append_img_reset_width is None else int(append_img_reset_width)
    if sk_width <= 0:
        sk_width = 1
    if sk_height <= 0:
        sk_height = 1
    # 关键点映射
    key_point_y_new = int(key_point_y/append_image.shape[0]*append_image.shape[1])
    # 缩放图片
    append_image = cv2.resize(append_image, (sk_width, sk_height))

    img_height, img_width, _ = img.shape
    # 是否根据骨骼节点位置在 图像中间的左右来控制是否进行 左右翻转图片
    # 主要处理头部的翻转, 默认头部是朝左
    if middle_flip:
        middle_x = int(img_width/2)
        if first_point[0] < middle_x and second_point[0] < middle_x:
            append_image = cv2.flip(append_image, 1)

    # 旋转角度
    angle = get_angle(first_point[0], first_point[1], second_point[0], second_point[1])
    append_image, move_x, move_y = rotate_bound(append_image, angle=angle, key_point_y=key_point_y_new)

    zero_x = first_point[0] - move_x
    zero_y = first_point[1] - move_y



    #cv2.seamlessClone(img, append_image, mask, (zero_y + width // 2, zero_x + height // 2), cv2.MIXED_CLONE)
    #img_mask(img, append_image, zero_x, zero_y)
    #img[zero_y: zero_y + width, zero_x: zero_x + height]  =  append_image
    (b, g, r) = cv2.split(append_image) 
    img_mask(img, append_image, zero_x, zero_y, r, img_width, img_height)
    '''
    for i in range(0, r.shape[0]):
        for j in range(0, r.shape[1]):
            if 230>r[i][j]>200 and 0<=zero_y+i<img_height and 0<=zero_x+j<img_width:
                img[zero_y+i][zero_x+j] = append_image[i][j]
    '''
    
    
    return img



def get_combine_img(image, current_poses, body_img, backgroup_img_path= './background.jpg'):
    '''
    识别图片中的关节点，并将皮影的肢体进行对应，最后与原图像拼接后输出
    '''
    
    # 背景图片
    
    backgroup_image = cv2.imread(backgroup_img_path)
    
    image_flag = cv2.resize(backgroup_image, (image.shape[1], image.shape[0]))
    
    # 最小宽度
    pose = current_poses[0]
    
    result = pose.keypoints
        
    min_width = int(get_distences(result[0][0], result[0][1],
                    result[1][0], result[1][1])/3)
    max_width = image.shape[0] / 10
        
    

        #右大腿
    append_img_reset_width = min(max(int(get_distences((result[8][0] + result[11][0]) / 2, (result[8][1] + result[11][1]) / 2,
                                                result[11][0], result[8][1])*1.6), min_width), max_width)

    image_flag = append_img_by_sk_points(image_flag, body_img['right_hip'], key_point_y=10, first_point=result[8],
                                                second_point=result[9], append_img_reset_width=append_img_reset_width)

        # 右小腿
    append_img_reset_width = min(max(int(get_distences((result[8][0] + result[11][0]) / 2, (result[8][1] + result[11][1]) / 2,
                                                result[11][0], result[11][1])*1.5), min_width), max_width)
    
    image_flag = append_img_by_sk_points(image_flag, body_img['right_knee'], key_point_y=10, first_point=result[9],
                                                    second_point=result[10], append_img_reset_width=append_img_reset_width)
    
        # 左大腿
    append_img_reset_width = min(max(int(get_distences((result[8][0] + result[11][0]) / 2, (result[8][1] + result[11][1]) / 2,
                                                result[11][0], result[11][1])*1.6), min_width), max_width)
    

    image_flag = append_img_by_sk_points(image_flag, body_img['left_hip'], key_point_y=0, first_point=result[11],
                                            second_point=result[12], append_img_reset_width=append_img_reset_width)

        # 左小腿
    append_img_reset_width = min(max(int(get_distences((result[8][0] + result[11][0]) / 2, (result[8][1] + result[11][1]) / 2,
                                            result[11][0], result[11][1])*1.5), min_width), max_width)
    image_flag = append_img_by_sk_points(image_flag, body_img['left_knee'], key_point_y=10, first_point=result[12],
                                                second_point=result[13], append_img_reset_width=append_img_reset_width)

        # 右手臂
    image_flag = append_img_by_sk_points(image_flag, body_img['left_elbow'], key_point_y=25, first_point=result[2],
                                            second_point=result[3], append_img_max_height_rate=1.2)

    
    
        # 右手肘
    append_img_max_height = int(get_distences(result[2][0], result[2][1],
                                                result[3][0], result[3][1])*1.6)
    image_flag = append_img_by_sk_points(image_flag, body_img['left_wrist'], key_point_y=10, first_point=result[3],
                                                second_point=result[4], append_img_max_height_rate=1.5, 
                                                append_img_max_height=append_img_max_height)

        # 左手臂
    image_flag = append_img_by_sk_points(image_flag, body_img['right_elbow'], key_point_y=25, first_point=result[5], 
                                            second_point=result[6],  append_img_max_height_rate=1.2)
        # 左手肘
    append_img_max_height = int(get_distences(result[5][0], result[5][1],
                                            result[6][0], result[6][1])*1.6)
    image_flag = append_img_by_sk_points(image_flag, body_img['right_wrist'], key_point_y=10, first_point=result[6],
                                            second_point=result[7], append_img_max_height_rate=1.5, 
                                            append_img_max_height=append_img_max_height)



        # 身体
    append_img_reset_width = max(int(get_distences(result[5][0], result[5][1],
                                                result[2][0], result[2][1])*1.2), min_width*3)
    image_flag = append_img_by_sk_points(image_flag, body_img['body'], key_point_y=20, first_point=result[1],
                        second_point=(result[8] + result[11]) / 2, append_img_reset_width=append_img_reset_width, append_img_max_height_rate=1.2)

        # 头
    append_img_max_height = int(get_distences(result[1][0], result[1][1], (result[8][0] + result[11][0]) / 2, (result[8][1] + result[11][1]) / 2))
    image_flag = append_img_by_sk_points(image_flag, body_img['head'], key_point_y=10, first_point=result[0],
                        second_point=result[1], append_img_max_height_rate=1.2, middle_flip=True, append_img_max_height = int(append_img_max_height / 2))

    
    result_img =  np.concatenate((image, image_flag), axis=1) 
    
    return result_img



def track_poses(previous_poses, current_poses, threshold=3, smooth=False):
    """Propagate poses ids from previous frame results. Id is propagated,
    if there are at least `threshold` similar keypoints between pose from previous frame and current.
    If correspondence between pose on previous and current frame was established, pose keypoints are smoothed.
    :param previous_poses: poses from previous frame with ids
    :param current_poses: poses from current frame to assign ids
    :param threshold: minimal number of similar keypoints between poses
    :param smooth: smooth pose keypoints between frames
    :return: None
    """
    current_poses = sorted(current_poses, key=lambda pose: pose.confidence, reverse=True)  # match confident poses first
    mask = np.ones(len(previous_poses), dtype=np.int32)
    for current_pose in current_poses:
        best_matched_id = None
        best_matched_pose_id = None
        best_matched_iou = 0
        for id, previous_pose in enumerate(previous_poses):
            if not mask[id]:
                continue
            iou = get_similarity(current_pose, previous_pose)
            if iou > best_matched_iou:
                best_matched_iou = iou
                best_matched_pose_id = previous_pose.id
                best_matched_id = id
        if best_matched_iou >= threshold:
            mask[best_matched_id] = 0
        else:  # pose not similar to any previous
            best_matched_pose_id = None
        current_pose.update_id(best_matched_pose_id)

        if smooth:
            for kpt_id in range(Pose.num_kpts):
                if current_pose.keypoints[kpt_id, 0] == -1 and len(previous_poses):
                    current_pose.keypoints[kpt_id] = previous_poses[0].keypoints[kpt_id]
                # reuse filter if previous pose has valid filter
                if (best_matched_pose_id is not None
                        and previous_poses[best_matched_id].keypoints[kpt_id, 0] != -1):
                    current_pose.filters[kpt_id] = previous_poses[best_matched_id].filters[kpt_id]
                current_pose.keypoints[kpt_id, 0] = current_pose.filters[kpt_id][0](current_pose.keypoints[kpt_id, 0])
                current_pose.keypoints[kpt_id, 1] = current_pose.filters[kpt_id][1](current_pose.keypoints[kpt_id, 1])
            current_pose.bbox = Pose.get_bbox(current_pose.keypoints)
    return current_poses