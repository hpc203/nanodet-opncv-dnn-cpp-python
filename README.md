# nanodet-opncv-dnn-cpp-python
用opencv部署nanodet目标检测，包含C++和python两种版本程序的实现，
使用opencv里的dnn模块加载网络模型，图像预处理和后处理模块是使用C++和python编程实现。
整个程序运行，不依赖任何深度学习框架，
在windows系统和ubuntu系统，在cpu和gpu机器上都能运行。

python版本的主程序是main_nanodet.py， c++版本的主程序是main.cpp

程序里提供输入图片尺寸320和416这两种选择，类别置信度阈值confThreshold，nms重叠率阈值nmsThreshold可自行调整

nanodet的后处理模块的解读，可以参阅我写的csdn博客文章
https://blog.csdn.net/nihate/article/details/113850913#comments_15084984
