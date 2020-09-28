import numpy as np
import cv2
import pyrealsense2 as rs

#realsense的配置参数
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)
config.enable_device_from_file('C:\\Users\\wujie\\Videos\\20200618_Xin_He_Jin\\20200625_133732.bag')
pipeline.start(config)

#模板匹配的参数
template_image_1 = cv2.imread("Xin_template_empty_2.png", 0)
template_image_2 = cv2.imread("Xin_template_good_2.png", 0)
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
meth = methods[1]
method = eval(meth)

# 获取列表的第二个元素
def takeSecond(elem):
    return elem[1]

def Point_differnce(image, step, points, state):
    height, width = image.shape[:]
    if state == 1:                          #在竖直方向上进行比较
        diff_list = []
        for i in range(len(points)):
            point = points[i]
            Point_up = (point[0], point[1]-step)
            Point_down = (point[0], point[1]+step)
            Point_up = np.uint0(Point_up)
            Point_down = np.uint0(Point_down)
            if Point_up[1] >= 0 and Point_down[1] <= height-1:
                diff = abs(image[Point_up[1]][Point_up[0]] - image[Point_down[1]][Point_down[0]])
                diff_list.append(diff)
        arr_var = np.var(diff_list)           #计算方差
        return arr_var

    if state == 0:                         #在水平方向上进行比较
        diff_list = []
        for i in range(len(points)):
            point = points[i]
            Point_right = (point[0] + step, point[1])
            Point_left = (point[0] - step, point[1])
            Point_right = np.uint0(Point_right)
            Point_left = np.uint0(Point_left)
            if Point_left[0] >= 0 and Point_right[0] <= width-1:
                diff = abs(image[Point_left[1]][Point_left[0]] - image[Point_right[1]][Point_right[0]])
                diff_list.append(diff)
        arr_var = np.var(diff_list)           #计算方差
        return arr_var

def cross_point(line1, line2):  # 计算交点函数
    x1 = line1[0]  # 取四点坐标
    y1 = line1[1]
    x2 = line1[2]
    y2 = line1[3]

    x3 = line2[0]
    y3 = line2[1]
    x4 = line2[2]
    y4 = line2[3]

    k1 = (y2 - y1) * 1.0 / (x2 - x1)  # 计算k1,由于点均为整数，需要进行浮点数转化
    b1 = y1 * 1.0 - x1 * k1 * 1.0  # 整型转浮点型是关键
    if (x4 - x3) == 0:  # L2直线斜率不存在操作
        k2 = None
        b2 = 0
    else:
        k2 = (y4 - y3) * 1.0 / (x4 - x3)  # 斜率存在操作
        b2 = y3 * 1.0 - x3 * k2 * 1.0
    if k2 == None:
        x = x3
    else:
        x = (b2 - b1) * 1.0 / (k1 - k2)
    y = k1 * x * 1.0 + b1 * 1.0
    return (x, y)

def Line_Difference_v2(image, step, interal, point_number, state):
    height, width = image.shape[:]
    if state == 1:
        sum_line_difference = []
        for i in range( int(height/2) // 2 ):
            if (i + 1) * interal < int(height/2):
                Pt1 = (1, interal * (i + 1))
                Pt2 = (width - 1, interal * (i + 1))
                Pt3 = (1, height -1 - interal * (i + 1))
                Pt4 = (width - 1, height - 1 - interal * (i + 1))
                Points12 = np.uint0(np.linspace(Pt1, Pt2, point_number))
                Points34 = np.uint0(np.linspace(Pt3, Pt4, point_number))
                diff_var12 = Point_differnce(image, step=step, points=Points12, state=state)
                diff_var34 = Point_differnce(image, step=step, points=Points34, state=state)
                diff_var = (diff_var12+diff_var34)/2
                sum_line_difference.append([diff_var, Pt1, Pt2, Pt3, Pt4])
        sum_diff = [x[0] for x in sum_line_difference]
        max_index = sum_diff.index(max(sum_diff))
        return [sum_line_difference[max_index][1], sum_line_difference[max_index][2], \
               sum_line_difference[max_index][3], sum_line_difference[max_index][4]]

    if state == 0:
        sum_line_difference = []
        for i in range( int(width/2) // 2 ):
            if (i + 1) * interal < int(width/2):
                Pt1 = (interal*(i+1), 1)
                Pt2 = (interal*(i+1), height - 1)
                Pt3 = (width - 1 - interal*(i+1), 1)
                Pt4 = (width - 1 - interal*(i+1), height - 1)
                Points12 = np.uint0(np.linspace(Pt1, Pt2, point_number))
                Points34 = np.uint0(np.linspace(Pt3, Pt4, point_number))
                diff_var12 = Point_differnce(image, step=step, points=Points12, state=state)
                diff_var34 = Point_differnce(image, step=step, points=Points34, state=state)
                diff_var = (diff_var12+diff_var34)/2
                sum_line_difference.append([diff_var, Pt1, Pt2, Pt3, Pt4])
        sum_diff = [x[0] for x in sum_line_difference]
        max_index = sum_diff.index(max(sum_diff))
        return [sum_line_difference[max_index][1], sum_line_difference[max_index][2], \
               sum_line_difference[max_index][3], sum_line_difference[max_index][4]]

def Box_inside_detect(image, top_left_x, bottom_right_x):   #输入图像必须是单通道的灰度图
    roi = image[top_left_x[1]:bottom_right_x[1], top_left_x[0]:bottom_right_x[0]]
    h, w = roi.shape[:]

    #sobel算子检测
    x = cv2.Sobel(roi, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(roi, cv2.CV_16S, 0, 1)
    absX = cv2.convertScaleAbs(x)  # 转回uint8
    absY = cv2.convertScaleAbs(y)
    mask = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    ret, mask = cv2.threshold(mask, 50, 250, cv2.THRESH_BINARY)

    #对sobel算子的结果进行腐蚀
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.dilate(mask, kernel)

    new_Pt_1 = Line_Difference_v2(mask, step=2, interal=5, point_number=20, state=1)
    new_Pt_2 = Line_Difference_v2(mask, step=2, interal=5, point_number=20, state=0)

    height_pt = abs(new_Pt_1[0][1] - new_Pt_1[2][1])
    width_pt = abs(new_Pt_2[0][0] - new_Pt_2[2][0])

    # 输出面积占比
    area = (height_pt * width_pt) / (w * h) * 100

    print(area)

    rect_corner = []
    if 60.0 > area > 50.0 :
        for i in range(len(new_Pt_1) // 2):
            j = i * 2
            line1 = (new_Pt_1[j][0], new_Pt_1[j][1], new_Pt_1[j + 1][0], new_Pt_1[j + 1][1])
            line2 = (new_Pt_2[j][0], new_Pt_2[j][1], new_Pt_2[j + 1][0], new_Pt_2[j + 1][1])
            point_corner = np.uint0(cross_point(line1, line2))
            rect_corner.append(point_corner)

    if len(rect_corner) == 2:
        rect_corner[0][0] += top_left_x[0]
        rect_corner[0][1] += top_left_x[1]
        rect_corner[1][0] += top_left_x[0]
        rect_corner[1][1] += top_left_x[1]
        return  rect_corner
    else:
        return None

def Generate_multi_box(image, top_left_x, top_left_y, state ,INTERVAL):        #由一个矩形框生成多个矩形框
    height, width = image.shape[:]
    center = (int((top_left[0] + bottom_right[0]) / 2), int((top_left[1] + bottom_right[1]) / 2))
    w = bottom_right[0] - top_left[0]
    h = bottom_right[1] - top_left[1]

    if state == 1:
        up_number = center[1] // INTERVAL
        down_number = (height - center[1]) // INTERVAL

        all_center = []
        if up_number > 0:
            for i in range(up_number):
                displacement = (i + 1) * INTERVAL
                # beacause it is on the left, so use the x value of center point sub the displacement
                temp_x = center[0]
                temp_y = center[1] - displacement
                all_center.append((temp_x, temp_y))
        all_center.append(center)
        if down_number > 0:
            for j in range(down_number):
                displacement = (j + 1) * INTERVAL
                # beacause it is on the right, so use the x value of center point add the displacement
                temp_x = center[0]
                temp_y = center[1] + displacement
                all_center.append((temp_x, temp_y))

        #对第二列元素进行排序
        all_center.sort(key=takeSecond)

        all_box = []
        number = 0
        for i in range(len(all_center)):
            center_temp = all_center[i]
            top_left_temp = (int(center_temp[0] - w / 2), int(center_temp[1] - h / 2))
            bottom_right_temp = (int(center_temp[0] + w / 2), int(center_temp[1] + h / 2))
            if top_left_temp[1] > 0 and  bottom_right_temp[1] < height:
                all_box.append((number, top_left_temp, bottom_right_temp))
                number += 1

        return all_box

    if state == 0:
        left_number = center[0] // INTERVAL
        right_number = (width - center[0]) // INTERVAL

        all_center = []
        if left_number > 0:
            for i in range(left_number):
                displacement = (i + 1) * INTERVAL
                # beacause it is on the left, so use the x value of center point sub the displacement
                temp_x = center[0] - displacement
                temp_y = center[1]
                all_center.append((temp_x, temp_y))
        all_center.append(center)
        if right_number > 0:
            for j in range(right_number):
                displacement = (j + 1) * INTERVAL
                # beacause it is on the right, so use the x value of center point add the displacement
                temp_x = center[0] + displacement
                temp_y = center[1]
                all_center.append((temp_x, temp_y))

        #对第一列元素进行排序
        all_center.sort()

        all_box = []
        number = 0
        for i in range(len(all_center)):
            center_temp = all_center[i]
            top_left_temp = (int(center_temp[0] - w / 2), int(center_temp[1] - h / 2))
            bottom_right_temp = (int(center_temp[0] + w / 2), int(center_temp[1] + h / 2))
            if top_left_temp[0] > 0 and bottom_right_temp[0] < width:
                all_box.append((number, top_left_temp, bottom_right_temp))
                number += 1
        return all_box

number_index = 0

try:
    while True:

        frames = pipeline.wait_for_frames()
        align = rs.align(rs.stream.color)
        frames = align.process(frames)
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Intrinsics & Extrinsics
        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
        color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
        depth_to_color_extrin = depth_frame.profile.get_extrinsics_to(color_frame.profile)

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Resize the color_image and depth_image from 1.0 to 0.8
        factor = 0.5
        color_image = cv2.resize(color_image, None, fx=factor, fy=factor, interpolation=0)

        #get the width and height of image
        color_image = cv2.GaussianBlur(color_image, (7, 7), 1)
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        w, h = template_image_2.shape[::-1]
        width, height = gray_image.shape[::-1]

        # 模板匹配
        res = cv2.matchTemplate(gray_image, template_image_2, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)

        if max_val < 0.6:
            res_2 = cv2.matchTemplate(gray_image, template_image_1, method)
            min_val_2, max_val_2, min_loc_2, max_loc_2 = cv2.minMaxLoc(res_2)

            if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                top_left_2 = min_loc_2
            else:
                top_left_2 = max_loc_2
            bottom_right_2 = (top_left_2[0] + w, top_left_2[1] + h)

            if max_val_2 > max_val:
                top_left = top_left_2
                bottom_right = bottom_right_2

        # all_box = Generate_multi_box(gray_image, top_left, bottom_right, state=1, INTERVAL=125)
        #
        # for i in range(len(all_box)):
        #     box = all_box[i]
        #     index = box[0]
        #     temp_top_left = box[1]
        #     temp_bottom_right = box[2]
        #     roi = color_image[temp_top_left[1]:temp_bottom_right[1], temp_top_left[0]:temp_bottom_right[0]]
        #     cv2.rectangle(color_image, temp_top_left, temp_bottom_right, (0,0,0), 2)
        #     cv2.imwrite()

        cv2.rectangle(color_image, top_left, bottom_right, (0, 255, 0), 2)

        roi = color_image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
        filename = "roi_Xin/roi3_" + str(number_index)+".jpg"
        cv2.imwrite(filename, roi)
        number_index += 1

        # rect_corner = Box_inside_detect(image=gray_image, top_left_x=top_left, bottom_right_x=bottom_right)
        # if rect_corner is not None:
        #     # 画出铸锭的边缘
        #     cv2.rectangle(color_image, (rect_corner[0][0], rect_corner[0][1]), \
        #                   (rect_corner[1][0], rect_corner[1][1]), (0, 0, 0), 2)
        #     #cv2.putText(color_image, str(i), top_left, cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', color_image)

        key = cv2.waitKey(1)
        if key == 27:
            cv2.imwrite("test_2.png", color_image)
            break;

finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()