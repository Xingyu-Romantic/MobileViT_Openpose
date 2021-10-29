import cv2
import math
import numpy as np




def append_image_and_resize(bg, append_img, first_point, second_point, combine_point):
    length = math.sqrt((first_point[0] - second_point[0])**2, (first_point[1] - second_point[1])**2)
    width, height = append_img.shape[:2]
    scale = length / width
    
    append_img = cv2.imread(append_img, fx = scale, fy = scale)

    cv2.add(bg)



def get_combine_img(image, current_poses, body_img, bg_img_path = './background.jpg'):
    
    # 背景图片
    bg_img = cv2.imread(bg_img_path)
    image_flag = cv2.resize(bg_img, (image.shape[0], image.shape[1]))

    pose = current_poses[0]
    result = pose.keypoints

    # Head
    L_ear, R_ear, Neck = result[15], result[14], result[1]
    append_image_and_resize(bg_img, body_img['head'], L_ear, R_ear, Neck)
    
    # Body







    

    result_img = np.concatenate((image, image_flag), axis = 1)