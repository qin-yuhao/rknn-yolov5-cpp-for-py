import objdetecter
import cv2
import time
runner = objdetecter.Rknn_yolo_runner("./yolov5s-640-640.rknn",12)# 模型路径，线程数

img = cv2.imread("bus.jpg")
star = time.time()
count = 0
while True:
    count += 1
    if count > 1000:
        count = 0
        end = time.time()
        delta_avg_ms = (end - star) * 1000 / 1000
        print("1000 帧平均耗时：{} ms".format(delta_avg_ms))
        print("1000 帧 FPS: {}".format(1000 / delta_avg_ms))
        star = time.time()

    resu = runner.get_put(img) # 传入要检测的图片，返回结果为检测到的图片切片列表，初始化线程时返回[-1]
    if len(resu) == 1 and type(resu[0]) == int:
        print("初始化线程池")
    # else:
    #     for i,j in enumerate(resu):
    #         cv2.imwrite("./resu_{}.jpg".format(i),j)
        
            
    
time.sleep(1)#释放太快会导致崩溃
runner.clean_pool()