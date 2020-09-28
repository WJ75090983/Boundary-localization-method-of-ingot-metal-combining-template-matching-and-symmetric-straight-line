# -
模板匹配、对称直线分割、金属溶液边缘检测

代码是用python编写的，主要使用的库有opencv4.2和pyrealsese。opencv4.2用来进行图像处理，pyrealsense用来读取相机的视频流。

程序的思路如下图所示。




![image](https://github.com/WJ75090983/Boundary-localization-method-of-ingot-metal-combining-template-matching-and-symmetric-straight-line/blob/master/1.jpg)



从图中可以看出，首先利用模板匹配法对原图像进行检测，得到单个铸锭的粗略位置。然后根据铸锭间距相等的特点，由已知的单块铸锭位置得到多块铸锭的位置。如下图所示。




![image](https://github.com/WJ75090983/Boundary-localization-method-of-ingot-metal-combining-template-matching-and-symmetric-straight-line/blob/master/2.jpg)

















