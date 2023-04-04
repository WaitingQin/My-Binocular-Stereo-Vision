# -*- coding: utf-8 -*-
import sys
import cv2
import numpy as np
import stereoconfig
import pcl
# import pcl.pcl_visualization
import open3d as o3d
import socket

# 预处理
def preprocess(img1, img2):
    # 彩色图->灰度图
    if (img1.ndim == 3):
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)  # 通过OpenCV加载的图像通道顺序是BGR
    if (img2.ndim == 3):
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 直方图均衡
    img1 = cv2.equalizeHist(img1)
    img2 = cv2.equalizeHist(img2)

    return img1, img2


# 消除畸变
def undistortion(image, camera_matrix, dist_coeff):
    undistortion_image = cv2.undistort(image, camera_matrix, dist_coeff)

    return undistortion_image


# 获取畸变校正和立体校正的映射变换矩阵、重投影矩阵
# @param：config是一个类，存储着双目标定的参数:config = stereoconfig.stereoCamera()
def getRectifyTransform(height, width, config):
    # 读取内参和外参
    left_K = config.cam_matrix_left
    right_K = config.cam_matrix_right
    left_distortion = config.distortion_l
    right_distortion = config.distortion_r
    R = config.R
    T = config.T

    # 计算校正变换
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(left_K, left_distortion, right_K, right_distortion,
                                                      (width, height), R, T, alpha=0)

    map1x, map1y = cv2.initUndistortRectifyMap(left_K, left_distortion, R1, P1, (width, height), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(right_K, right_distortion, R2, P2, (width, height), cv2.CV_32FC1)

    return map1x, map1y, map2x, map2y, Q


# 畸变校正和立体校正
def rectifyImage(image1, image2, map1x, map1y, map2x, map2y):
    rectifyed_img1 = cv2.remap(image1, map1x, map1y, cv2.INTER_AREA)
    rectifyed_img2 = cv2.remap(image2, map2x, map2y, cv2.INTER_AREA)

    return rectifyed_img1, rectifyed_img2


# 立体校正检验----画线
def draw_line(image1, image2):
    # 建立输出图像
    height = max(image1.shape[0], image2.shape[0])
    width = image1.shape[1] + image2.shape[1]

    output = np.zeros((height, width, 3), dtype=np.uint8)
    output[0:image1.shape[0], 0:image1.shape[1]] = image1
    output[0:image2.shape[0], image1.shape[1]:] = image2

    # 绘制等间距平行线
    line_interval = 50  # 直线间隔：50
    for k in range(height // line_interval):
        cv2.line(output, (0, line_interval * (k + 1)), (2 * width, line_interval * (k + 1)), (0, 255, 0), thickness=2,
                 lineType=cv2.LINE_AA)

    return output


# 视差计算
def stereoMatchSGBM(left_image, right_image, down_scale=False):
    # SGBM匹配参数设置
    if left_image.ndim == 2:
        img_channels = 1
    else:
        img_channels = 3
    blockSize = 3
    paraml = {'minDisparity': 0,
              'numDisparities': 64,
              'blockSize': blockSize,
              'P1': 8 * img_channels * blockSize ** 2,
              'P2': 32 * img_channels * blockSize ** 2,
              'disp12MaxDiff': 1,
              'preFilterCap': 63,
              'uniquenessRatio': 15,
              'speckleWindowSize': 100,
              'speckleRange': 1,
              'mode': cv2.STEREO_SGBM_MODE_SGBM_3WAY
              }

    # 构建SGBM对象
    left_matcher = cv2.StereoSGBM_create(**paraml)
    paramr = paraml
    paramr['minDisparity'] = -paraml['numDisparities']
    right_matcher = cv2.StereoSGBM_create(**paramr)

    # 计算视差图
    size = (left_image.shape[1], left_image.shape[0])
    if down_scale == False:
        disparity_left = left_matcher.compute(left_image, right_image)
        disparity_right = right_matcher.compute(right_image, left_image)
    else:
        left_image_down = cv2.pyrDown(left_image)
        right_image_down = cv2.pyrDown(right_image)
        factor = left_image.shape[1] / left_image_down.shape[1]

        disparity_left_half = left_matcher.compute(left_image_down, right_image_down)
        disparity_right_half = right_matcher.compute(right_image_down, left_image_down)
        disparity_left = cv2.resize(disparity_left_half, size, interpolation=cv2.INTER_AREA)
        disparity_right = cv2.resize(disparity_right_half, size, interpolation=cv2.INTER_AREA)
        disparity_left = factor * disparity_left
        disparity_right = factor * disparity_right

    # 真实视差（因为SGBM算法得到的视差是×16的）
    trueDisp_left = disparity_left.astype(np.float32) / 16.
    trueDisp_right = disparity_right.astype(np.float32) / 16.

    return trueDisp_left, trueDisp_right


# 将h×w×3数组转换为N×3的数组
def hw3ToN3(points):
    height, width = points.shape[0:2]

    points_1 = points[:, :, 0].reshape(height * width, 1)
    points_2 = points[:, :, 1].reshape(height * width, 1)
    points_3 = points[:, :, 2].reshape(height * width, 1)

    points_ = np.hstack((points_1, points_2, points_3))

    return points_


# 深度、颜色转换为点云
def DepthColor2Cloud(points_3d, colors):
    rows, cols = points_3d.shape[0:2]
    size = rows * cols

    points_ = hw3ToN3(points_3d)
    colors_ = hw3ToN3(colors).astype(np.int64)

    # 颜色信息
    blue = colors_[:, 0].reshape(size, 1)
    green = colors_[:, 1].reshape(size, 1)
    red = colors_[:, 2].reshape(size, 1)

    rgb = np.left_shift(blue, 0) + np.left_shift(green, 8) + np.left_shift(red, 16)

    # 将坐标+颜色叠加为点云数组
    pointcloud = np.hstack((points_, rgb)).astype(np.float32)

    # 删掉一些不合适的点
    X = pointcloud[:, 0]
    Y = pointcloud[:, 1]
    Z = pointcloud[:, 2]

    # 下面参数是经验性取值，需要根据实际情况调整
    remove_idx1 = np.where(Z <= 0)
    remove_idx2 = np.where(Z > 15000)
    remove_idx3 = np.where(X > 10000)
    remove_idx4 = np.where(X < -10000)
    remove_idx5 = np.where(Y > 10000)
    remove_idx6 = np.where(Y < -10000)
    remove_idx = np.hstack(
        (remove_idx1[0], remove_idx2[0], remove_idx3[0], remove_idx4[0], remove_idx5[0], remove_idx6[0]))

    pointcloud_1 = np.delete(pointcloud, remove_idx, 0)

    return pointcloud_1


# 点云显示
def view_cloud(pointcloud):
    cloud = pcl.PointCloud_PointXYZRGBA()
    cloud.from_array(pointcloud)

    try:
        visual = pcl.pcl_visualization.CloudViewing()
        visual.ShowColorACloud(cloud)
        v = True
        while v:
            v = not (visual.WasStopped())
    except:
        pass


def getDepthMapWithQ(disparityMap: np.ndarray, Q: np.ndarray) -> np.ndarray:
    points_3d = cv2.reprojectImageTo3D(disparityMap, Q)
    depthMap = points_3d[:, :, 2]
    reset_index = np.where(np.logical_or(depthMap < 0.0, depthMap > 65535.0))
    depthMap[reset_index] = 0

    return depthMap.astype(np.float32)


def getDepthMapWithConfig(disparityMap: np.ndarray, config: stereoconfig.stereoCamera) -> np.ndarray:
    fb = config.cam_matrix_left[0, 0] * (-config.T[0])
    doffs = config.doffs
    depthMap = np.divide(fb, disparityMap + doffs)
    reset_index = np.where(np.logical_or(depthMap < 0.0, depthMap > 65535.0))
    depthMap[reset_index] = 0
    reset_index2 = np.where(disparityMap < 0.0)
    depthMap[reset_index2] = 0
    return depthMap.astype(np.float32)

def callbackFunc(e, x, y, f, p):
    if e == cv2.EVENT_LBUTTONDOWN:
        # print("Depth: ", depthMap[y][x])
        print(depthMap[y][x]/100.0, "cm")

cv2.namedWindow(winname='DepthMap')
cv2.namedWindow(winname='OriginImage')
cv2.createTrackbar("DepthMax", "DepthMap", 100, 2000, lambda x: None)
cv2.createTrackbar("DepthMin", "DepthMap", 0, 2000, lambda x: None)
cv2.setMouseCallback("OriginImage", callbackFunc)

camera = cv2.VideoCapture(1)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

if __name__ == '__main__':
    host = "172.18.16.77"
    port = 5000
    s = socket.socket()
    s.bind((host, port))
    print("waiting for connection......")
    s.listen(1)  # 只能同时连接一个
    my_server, address = s.accept()
    while True:
        _, frame = camera.read()
        # 裁剪坐标为[y0:y1, x0:x1] HEIGHT*WIDTH
        iml = frame[0:240, 0:320]
        imr = frame[0:240, 320:640]
        # iml_temp = iml.copy()
        # imr_temp = imr.copy()
        # M = cv2.getRotationMatrix2D((iml_temp.shape[1]/2, iml_temp.shape[0]/2), 180, 1)
        # imr = cv2.warpAffine(iml_temp, M, (iml_temp.shape[1], iml_temp.shape[0]))
        # M = cv2.getRotationMatrix2D((imr_temp.shape[1]/2, imr_temp.shape[0]/2), 180, 1)
        # iml = cv2.warpAffine(imr_temp, M, (imr_temp.shape[1], imr_temp.shape[0]))
        height, width = iml.shape[0:2]
        depthMax = cv2.getTrackbarPos("DepthMax", "DepthMap")
        depthMin = cv2.getTrackbarPos("DepthMin", "DepthMap")
        
        # 读取相机内参和外参
        # 使用之前先将标定得到的内外参数填写到stereoconfig.py中的StereoCamera类中
        config = stereoconfig.stereoCamera()
        config.setMiddleBurryParams()
        # print("config.cam_matrix_left: ", config.cam_matrix_left)
        #
        # # 立体校正
        # map1x, map1y, map2x, map2y, Q = getRectifyTransform(height, width, config)  # 获取用于畸变校正和立体校正的映射矩阵以及用于计算像素空间坐标的重投影矩阵
        # iml_rectified, imr_rectified = rectifyImage(iml, imr, map1x, map1y, map2x, map2y)
        # print("Q: ", Q)
        #
        # # 绘制等间距平行线，检查立体校正的效果
        # line = draw_line(iml_rectified, imr_rectified)
        # cv2.imwrite('data/check_rectification.png', line)

        # 立体匹配
        iml, imr = preprocess(iml, imr)  # 预处理，一般可以削弱光照不均的影响，不做也可以
        disp, _ = stereoMatchSGBM(iml, imr, False)  # 这里传入的是未经立体校正的图像，因为我们使用的middleburry图片已经是校正过的了
        # cv2.imwrite('data/disaprity.png', disp * 4)

        # 计算深度图
        # depthMap = getDepthMapWithQ(disp, Q)
        depthMap = getDepthMapWithConfig(disp, config)
        temp = depthMap.copy()
        reset_index2 = np.where(depthMap == 0.0)
        temp[reset_index2] = float('inf')
        minDepth = np.min(temp)
        maxDepth = np.max(depthMap)
        reset_index3 = np.where(depthMap >= minDepth+float(depthMax))
        depthMap[reset_index3] = 0.0
        reset_index4 = np.where(depthMap <= minDepth+float(depthMin))
        depthMap[reset_index4] = 0.0
        # print("minDepth: ", minDepth)
        # print("maxDepth: ", maxDepth)
        # depthtemp = depthMap.copy()
        # for n in range(len(depthtemp)):
        #     for m in range(len(depthtemp[n])):
        #         if depthtemp[n][m] - minDepth >= 1200:
        #             depthtemp[n][m] = 0
        reset_index3 = np.where(depthMap == 0.0)
        depthMap[reset_index3] = minDepth
        depthMapVis = -255.0 * ((depthMap - minDepth) / (maxDepth - 0))*4
        depthMapVis = depthMapVis.astype(np.uint8)
        ret, binary = cv2.threshold(depthMapVis, 1, 255, 0)
        kernel = np.ones((6, 6), np.uint8)  #膨胀
        binary = cv2.dilate(binary, kernel, iterations=2)
        # cv2.imshow("", binary)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        if len(contours) != 0:
            contours = max(contours, key=cv2.contourArea)
            M = cv2.moments(contours)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            # print("depth: ", depthMap[cy][cx])
            print("%.2f"%(depthMap[cy][cx]/100.0), "cm")
            data = str(cx) + " " + str(cy) + " " + str(depthMap[cy][cx])
            my_server.send(data.encode())
        else:
            data = str(0.0) + " " + str(0.0) + " " + str(0.0)
            my_server.send(data.encode())
        depthMapVis = cv2.drawContours(depthMapVis, contours, -1, (255, 255, 255), 3)
        # depthMapVis = cv2.drawContours(iml, contours, -1, (255, 255, 255), 3)
        cv2.imshow("DepthMap", depthMapVis)
        cv2.imshow("OriginImage", iml)
        cv2.imshow("", imr)
        key = cv2.waitKey(1)
        if key == ord("q"):
            # my_server.close()
            exit(0)

        # # 使用open3d库绘制点云
        # colorImage = o3d.geometry.Image(iml)
        # depthImage = o3d.geometry.Image(depthMap)
        # rgbdImage = o3d.geometry.RGBDImage().create_from_color_and_depth(colorImage, depthImage, depth_scale=1000.0,
        #                                                                  depth_trunc=np.inf)
        # intrinsics = o3d.camera.PinholeCameraIntrinsic()
        # # fx = Q[2, 3]
        # # fy = Q[2, 3]
        # # cx = Q[0, 3]
        # # cy = Q[1, 3]
        # fx = config.cam_matrix_left[0, 0]
        # fy = fx
        # cx = config.cam_matrix_left[0, 2]
        # cy = config.cam_matrix_left[1, 2]
        # print(fx, fy, cx, cy)
        # intrinsics.set_intrinsics(width, height, fx=fx, fy=fy, cx=cx, cy=cy)
        # extrinsics = np.array([[1., 0., 0., 0.],
        #                        [0., 1., 0., 0.],
        #                        [0., 0., 1., 0.],
        #                        [0., 0., 0., 1.]])
        # pointcloud = o3d.geometry.PointCloud().create_from_rgbd_image(rgbdImage, intrinsic=intrinsics, extrinsic=extrinsics)
        # o3d.io.write_point_cloud("PointCloud.pcd", pointcloud=pointcloud)
        # o3d.visualization.draw_geometries([pointcloud], width=720, height=480)
        # sys.exit(0)
        #
        # # 计算像素点的3D坐标（左相机坐标系下）
        # points_3d = cv2.reprojectImageTo3D(disp, Q)  # 参数中的Q就是由getRectifyTransform()函数得到的重投影矩阵
        #
        # # 构建点云--Point_XYZRGBA格式
        # pointcloud = DepthColor2Cloud(points_3d, iml)
        #
        # # 显示点云
        # view_cloud(points_3d)
        