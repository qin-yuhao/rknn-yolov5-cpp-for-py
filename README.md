# 简介
* Accelerate the running speed of the rknn model by compiling the cpp dynamic library for python
* 通过[pybind11](https://github.com/leafqycc/rknn-cpp-Multithreading)为py编译cpp动态库来加速
* 此仓库大体改自[rknn-cpp-Multithreading](https://github.com/leafqycc/rknn-cpp-Multithreading)
* rknn驱动，rknn模型均为1.4.0，若要使用1.5.0等版本，请更换./lib中的rknn驱动
* 运行1.6.0版本的库和模型会卡住
# 使用说明
* 运行bash build.sh编译出动态库.so文件
* python test.py(若为python3.10，应该可以直接用本项目的.so文件)
# 多线程模型帧率测试
* 测试模型来源: 
* [yolov5s-relu](https://github.com/rockchip-linux/rknpu2/tree/master/examples/rknn_yolov5_demo/model/RK3588)
* 测试时未进行CPU/NPU定频
* 返回值是检测到的目标在原图中的切片
  
 |  模型\线程数   |  12  |
 |  ----  | ----  |
 | Yolov5s - relu  |  120 |
# Acknowledgements
* https://github.com/leafqycc/rknn-cpp-Multithreading
* https://github.com/rockchip-linux/rknpu2
* https://github.com/senlinzhan/dpool
* https://github.com/ultralytics/yolov5
* https://github.com/airockchip/rknn_model_zoo
* https://github.com/leafqycc/rknn-cpp-Multithreading
