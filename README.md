# nanodet-opncv-dnn-cpp-python
用opencv部署nanodet目标检测，包含C++和python两种版本程序的实现，
使用opencv里的dnn模块加载网络模型，图像预处理和后处理模块用C++和python编程实现。
整个程序运行，不依赖任何深度学习框架，
在windows系统和ubuntu系统，在cpu和gpu机器上都能运行

python版本的主程序是main_nanodet.py， c++版本的主程序是main.cpp

类别置信度阈值confThreshold，nms重叠率阈值nmsThreshold可自行调整
