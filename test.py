import cv2

camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
num = 0

if __name__ == '__main__':
    while True:
        ret, frame = camera.read()
        # 裁剪坐标为[y0:y1, x0:x1] HEIGHT*WIDTH
        if frame is None:
            print("No Camera")
            break
        iml = frame[0:240, 0:320]
        imr = frame[0:240, 320:640]
        cv2.imshow("left", iml)
        cv2.imshow("right", imr)
        key = cv2.waitKey(1)
        if key == ord("s"):
            cv2.imwrite("shot/left_%s.jpg" % num, iml)
            cv2.imwrite("shot/right_%s.jpg" % num, imr)
            num += 1
            # cv2.imwrite("./snapshot/BM_depth.jpg", disp)
        if key == ord("q"):
            exit(0)