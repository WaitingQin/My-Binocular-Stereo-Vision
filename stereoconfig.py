import numpy as np


# 双目相机参数
class stereoCamera(object):
    def __init__(self):
        # 左相机内参
        self.cam_matrix_left = np.array([[574.3784, 0., 0.],
                                         [0., 574.3105, 0.],
                                         [370.0067, 220.0164, 1.]])
        # 右相机内参
        self.cam_matrix_right = np.array([[573.2131, 0., 0.],
                                          [0., 574.1315, 0.],
                                          [356.0212, 234.8006, 1.]])

        # 左右相机畸变系数:[k1, k2, p1, p2, k3]
        self.distortion_l = np.array([[0.0353, -0.1002, 0., 0., 0.]])
        self.distortion_r = np.array([[0.0472, -0.1018, 0., 0., 0.]])

        # 旋转矩阵
        self.R = np.array([[1.0000, 0.0023, -0.0084],
                           [-0.0022, 0.9998, 0.0181],
                           [0.0084, -0.0181, 0.9998]])

        # 平移矩阵
        self.T = np.array([[-107.2232], [0.0709], [-0.0571]])

        # 主点列坐标的差
        self.doffs = 0.0

        # 指示上述内外参是否为经过立体校正后的结果
        self.isRectified = False

    def setMiddleBurryParams(self):
        self.cam_matrix_left = np.array([[3997.684, 0, 225.0],
                                         [0., 3997.684, 187.5],
                                         [0., 0., 1.]])
        self.cam_matrix_right = np.array([[3997.684, 0, 225.0],
                                          [0., 3997.684, 187.5],
                                          [0., 0., 1.]])
        self.distortion_l = np.zeros(shape=(5, 1), dtype=np.float64)
        self.distortion_r = np.zeros(shape=(5, 1), dtype=np.float64)
        self.R = np.identity(3, dtype=np.float64)
        self.T = np.array([[-193.001], [0.0], [0.0]])
        self.doffs = 131.111
        self.isRectified = True