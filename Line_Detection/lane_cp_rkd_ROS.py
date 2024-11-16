#! /usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np

from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from math import *
from collections import deque

distance_threshold = 20
theta_threshold = 10
max_queue_size = 5  # 큐의 최대 크기 설정
weights = [1/85, 3/85, 7/85, 15/85, 60/85]

distance_L_queue = deque([], maxlen=5)
distance_R_queue = deque([], maxlen=5)
theta_L_queue = deque([], maxlen=5)
theta_R_queue = deque([], maxlen=5)

def warpping(image):
    """
        차선을 BEV로 변환하는 함수
        
        Return
        1) _image : BEV result image
        2) minv : inverse matrix of BEV conversion matrix
    """

    source = np.float32([[140, 100], [0, 480], [500, 100], [640, 480]])
    destination = np.float32([[0, 0], [0, 480], [480, 0], [480, 480]])
    
    M = cv2.getPerspectiveTransform(source, destination)
    Minv = cv2.getPerspectiveTransform(destination, source)
    
    
    warp_image = cv2.warpPerspective(image, M, (480, 480), flags=cv2.INTER_LINEAR)
    # cv2.rectangle(warp_image, (195, 445), (480, 480), (0, 0, 0), -1)

    return warp_image, Minv

def color_filter(image):
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

    lower = np.array([200, 200, 200])
    upper = np.array([255, 255, 255])

    mask = cv2.inRange(hls, lower, upper)
    masked = cv2.bitwise_and(image, image, mask = mask)
    
    return masked

class lane_detect():
    def __init__(self):
        self.bridge = CvBridge()
        rospy.init_node('lane_detection_node', anonymous=False)
        rospy.Subscriber('/main_camera/image_raw/compressed', CompressedImage, self.camera_callback)
        self.pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)

    def camera_callback(self, data):
        self.image = self.bridge.compressed_imgmsg_to_cv2(data, desired_encoding="bgr8")
        self.lane_detect()
        
    def high_level_detect(self, hough_img):

        nwindows = 10       # window 개수
        margin = 75         # window 가로 길이
        minpix = 30          # 차선 인식을 판정하는 최소 픽셀 수
       
        histogram = np.sum(hough_img[hough_img.shape[0]//2:,:],   axis=0)
    
        midx_current = np.argmax(histogram[:])

        # 쌓을 window의 height 설정
        window_height = np.int32(hough_img.shape[0]/nwindows)
        
        # 240*320 픽셀에 담긴 값중 0이 아닌 값을 저장한다.
        # nz[0]에는 index[row][col] 중에 row파트만 담겨있고 nz[1]에는 col이 담겨있다.
        nz = hough_img.nonzero()

        mid_lane_inds = []

        global x,y
        x,y = [],[]

        global out_img
        out_img = np.dstack((hough_img, hough_img, hough_img))*255

        mid_sum = 0

        total_loop = nwindows-4

        for window in range(total_loop):
            
            # bounding box 크기 설정
            win_yl = hough_img.shape[0] - (window+1)*window_height
            win_yh = hough_img.shape[0] - window*window_height

            win_xl = midx_current - margin
            win_xh = midx_current + margin

            # out image에 bounding box 시각화
            cv2.rectangle(out_img,(win_xl,win_yl),(win_xh,    win_yh),    (0,255,0), 2) 

            # 흰점의 픽셀들 중에 window안에 들어오는 픽셀인지 여부를 판단하여 
            # good_left_inds와 good_right_inds에 담는다.
            good_inds = ((nz[0] >= win_yl)&(nz[0] < win_yh)&   (nz    [1] >= win_xl)&(nz[1] < win_xh)).nonzero()    [0]
            
            mid_lane_inds.append(good_inds)

            # nz[1]값들 중에 good_left_inds를 index로 삼는 nz[1]들의 평균을 구해서 leftx_current를 갱신한다.
            if len(good_inds) > minpix:
                midx_current = np.int32(np.mean(nz[1]    [good_inds])   )

            #lx ly rx ry에 x,y좌표들의 중심점들을 담아둔다.
            x.append(midx_current)
            y.append((win_yl + win_yh)/2)

            # left_sum += leftx_current
            mid_sum += midx_current

        mid_lane_inds = np.concatenate(mid_lane_inds)

        fit = np.polyfit(np.array(y[1:]),np.array(x[1:]),2)

        #out_img에서 왼쪽 선들의 픽셀값을 BLUE로, 
        #오른쪽 선들의 픽셀값을 RED로 바꿔준다.
        out_img[nz[0][mid_lane_inds], nz[1][mid_lane_inds]] = [255, 0, 0]

        mid_avg = mid_sum / total_loop

        return fit, mid_avg
    
    def lane_detect(self):
        
        cv2.imshow('Original', self.image)

        warpped_img, minv = warpping(self.image)
        blurred_img = cv2.GaussianBlur(warpped_img, (7, 7), 5)
        w_f_img = color_filter(blurred_img)
        cv2.rectangle(w_f_img, (0, 0), (480, 200), (0, 0, 0), -1)
        
        grayscale = cv2.cvtColor(w_f_img, cv2.COLOR_BGR2GRAY)
        print(grayscale[240][240])
        ret, thresh = cv2.threshold(grayscale, 50, 255, cv2.THRESH_BINARY) #170, 255
        
        canny_img = cv2.Canny(thresh, 10, 100)

        fit, avg = self.high_level_detect(canny_img)
        
        fit = np.polyfit(np.array(y),np.array(x),1)
        print(fit)
        
        line = np.poly1d(fit)
        
        # 좌,우측 차선의 휘어진 각도
        line_angle = degrees(atan(line[1]))


        cv2.namedWindow('Sliding Window')
        cv2.moveWindow('Sliding Window', 1400, 0)
        cv2.imshow("Sliding Window", out_img)
        cv2.waitKey(1)
        if fit[0]==0 and fit[1]==0:
            distance=0
        else:
            distance = -(np.polyval(fit,480) - 240)

        k= 0.001
        theta_err = radians(line_angle)
        lat_err = distance * cos(line_angle)

        speed = Twist()
        speed.linear.x = 0.1
        speed.angular.z = theta_err + atan(k*lat_err)
        self.pub.publish(speed)
        print(degrees(theta_err),degrees(atan(k*lat_err)))
        print(speed.angular.z)

if __name__ == "__main__":

    if not rospy.is_shutdown():
        lane_detect()
        rospy.spin()
