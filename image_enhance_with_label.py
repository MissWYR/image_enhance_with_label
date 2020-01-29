# -*- coding: utf-8 -*-
#作者：卫毅然（老卫）
#邮箱：2205492446@qq.com
#该脚本用于增加图片以及label标签，适用于于深度学习数据集
#根据自身需要去更改对应地址以及代码



import cv2
import math
import numpy as np
import os
import PIL.Image as pilimg
import xml.etree.ElementTree as ET
import pdb

h_flip = True  # horizontal flip
v_flip = True  # vertical flip
hv_flip = True  # both horizontal and vertical flip

#旋转图像的函数
def rotate_image(src, angle, scale=1.):
    w = src.shape[1]
    h = src.shape[0]
    # 角度变弧度
    rangle = np.deg2rad(angle)  # angle in radians
    # now calculate new image width and height
    nw = (abs(np.sin(rangle)*h) + abs(np.cos(rangle)*w))*scale
    nh = (abs(np.cos(rangle)*h) + abs(np.sin(rangle)*w))*scale
    # ask OpenCV for the rotation matrix
    rot_mat = cv2.getRotationMatrix2D((nw*0.5, nh*0.5), angle, scale)
    # calculate the move from the old center to the new center combined
    # with the rotation
    rot_move = np.dot(rot_mat, np.array([(nw-w)*0.5, (nh-h)*0.5,0]))
    # the move only affects the translation, so update the translation
    # part of the transform
    rot_mat[0,2] += rot_move[0]
    rot_mat[1,2] += rot_move[1]
    # 仿射变换
    return cv2.warpAffine(src, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)

#对应修改xml文件
def rotate_xml(src, xmin, ymin, xmax, ymax, angle, scale=1.):
    w = src.shape[1]
    h = src.shape[0]
    rangle = np.deg2rad(angle)  # angle in radians
    # now calculate new image width and height
    # 获取旋转后图像的长和宽
    nw = (abs(np.sin(rangle)*h) + abs(np.cos(rangle)*w))*scale
    nh = (abs(np.cos(rangle)*h) + abs(np.sin(rangle)*w))*scale
    # ask OpenCV for the rotation matrix
    rot_mat = cv2.getRotationMatrix2D((nw*0.5, nh*0.5), angle, scale)
    # calculate the move from the old center to the new center combined
    # with the rotation
    rot_move = np.dot(rot_mat, np.array([(nw-w)*0.5, (nh-h)*0.5,0]))
    # the move only affects the translation, so update the translation
    # part of the transform
    rot_mat[0,2] += rot_move[0]
    rot_mat[1,2] += rot_move[1]
    # rot_mat是最终的旋转矩阵
    # 获取原始矩形的四个中点，然后将这四个点转换到旋转后的坐标系下
    point1 = np.dot(rot_mat, np.array([(xmin+xmax)/2, ymin, 1]))
    point2 = np.dot(rot_mat, np.array([xmax, (ymin+ymax)/2, 1]))
    point3 = np.dot(rot_mat, np.array([(xmin+xmax)/2, ymax, 1]))
    point4 = np.dot(rot_mat, np.array([xmin, (ymin+ymax)/2, 1]))
    # 合并np.array
    concat = np.vstack((point1, point2, point3, point4))
    # 改变array类型
    concat = concat.astype(np.int32)
    print(concat)
    rx, ry, rw, rh = cv2.boundingRect(concat)
    return rx, ry, rw, rh


def boundary_control(value, min, max):
    # 类型检查
    value = int(value)
    # 边界溢出检查
    value = min if value < min else value
    value = max if value > max else value
    return value

def enhance_label(src, xmin, ymin, xmax, ymax):
    h_flip = True  # horizontal flip
    v_flip = True  # vertical flip
    hv_flip = False  # both horizontal and vertical flip

    w = src.shape[1]
    h = src.shape[0]
    print(w,h)
    x1=xmin
    y1=ymin
    x2=xmax
    y2=ymax
    x1_new,y1_new,x2_new,y2_new=0,0,0,0
    if hv_flip:
        # Flipped Horizontally & Vertically 水平垂直翻转
                x1, x2 = boundary_control(x1, 0, w), boundary_control(x2, 0, w)
                y1, y2 = boundary_control(y1, 0, h), boundary_control(y2, 0, h)
                # flip
                x1_new = w - x2
                y1_new = h-y2
                x2_new = w - x1
                y2_new = h-y1
                # new_mess = "{0} {1} {2} {3}\n".format(x1_new, y1_new, x2_new, y2_new)
                # h_file.writelines(new_mess)
        # h_file.close()
                return x1_new,y1_new,x2_new,y2_new
    if h_flip:

        # Flipped Horizontally & Vertically 水平垂直翻转
        x1, x2 = boundary_control(x1, 0, w), boundary_control(x2, 0, w)
        y1, y2 = boundary_control(y1, 0, h), boundary_control(y2, 0, h)
        x1_new = x1
        y1_new = h - y2
        x2_new = x2
        y2_new = h - y1
        # new_mess = "{0} {1} {2} {3}\n".format(x1_new, y1_new, x2_new, y2_new)
        # h_file.writelines(new_mess)
        # h_file.close()
        return x1_new, y1_new, x2_new, y2_new

    if v_flip:

        # Flipped Horizontally & Vertically 水平垂直翻转
        x1, x2 = boundary_control(x1, 0, w), boundary_control(x2, 0, w)
        y1, y2 = boundary_control(y1, 0, h), boundary_control(y2, 0, h)
        x1_new = w-x2
        y1_new = y1
        x2_new = w-x1
        y2_new = y2
        # new_mess = "{0} {1} {2} {3}\n".format(x1_new, y1_new, x2_new, y2_new)
        # h_file.writelines(new_mess)
        # h_file.close()
        return x1_new, y1_new, x2_new, y2_new

def hv_flip_func():
    # 指向图片所在的文件夹
    for i in os.listdir("/home/weiyiran/Public/src_img"):
        # 分离文件名与后缀
        a, b = os.path.splitext(i)
        # 如果后缀名是“.jpg”就旋转图像
        if b == ".jpg":
            img_path = os.path.join("/home/weiyiran/Public/src_img", i)
            img = cv2.imread(img_path)
            rotated_img = rotate_image(img,180)
            # 写入图像
            cv2.imwrite("/home/weiyiran/Public/dis_img/" + a + "_d.jpg", rotated_img)
            print("log: %s is processed." % (i))
        else:
            xml_path = os.path.join("/home/weiyiran/Public/src_img", i)
            img_path = "/home/weiyiran/Public/src_img/" + a + ".jpg"
            src = cv2.imread(img_path)
            tree = ET.parse(xml_path)
            root = tree.getroot()
            for box in root.iter('bndbox'):
                xmin = float(box.find('xmin').text)
                ymin = float(box.find('ymin').text)
                xmax = float(box.find('xmax').text)
                ymax = float(box.find('ymax').text)
                print(xmin,ymin,xmax,ymax)
                # x, y, w, h = rotate_xml(src, xmin, ymin, xmax, ymax, angle)
                # 改变xml中的人脸坐标值
                x, y, w, h = enhance_label(src, xmin, ymin, xmax, ymax)
                print(x,y,w,h)
                box.find('xmin').text = str(x)
                box.find('ymin').text = str(y)
                box.find('xmax').text = str(w)
                box.find('ymax').text = str(h)
                box.set('updated', 'yes')
            # 写入新的xml
            tree.write("/home/weiyiran/Public/dis_img/" + a + "_d.xml")
            print("%s is processed." % (i))

def h_flip_func():
    # 指向图片所在的文件夹
    for i in os.listdir("/home/weiyiran/Public/src_img"):
        # 分离文件名与后缀
        a, b = os.path.splitext(i)
        # 如果后缀名是“.jpg”就旋转图像
        if b == ".jpg":
            img_path = os.path.join("/home/weiyiran/Public/src_img", i)
            img = pilimg.open(img_path)
            ng = img.transpose(pilimg.FLIP_TOP_BOTTOM)  # 上下对换。
            # cv2.imwrite("/home/weiyiran/Public/dis_img/" + a + "_h.jpg", ng)
            ng.save("/home/weiyiran/Public/dis_img/"+ a + "_h.jpg")
            print("log: %s is processed." % (i))
        else:
            xml_path = os.path.join("/home/weiyiran/Public/src_img", i)
            img_path = "/home/weiyiran/Public/src_img/" + a + ".jpg"
            src = cv2.imread(img_path)
            tree = ET.parse(xml_path)
            root = tree.getroot()
            for box in root.iter('bndbox'):
                xmin = float(box.find('xmin').text)
                ymin = float(box.find('ymin').text)
                xmax = float(box.find('xmax').text)
                ymax = float(box.find('ymax').text)
                print(xmin,ymin,xmax,ymax)
                # x, y, w, h = rotate_xml(src, xmin, ymin, xmax, ymax, angle)
                # 改变xml中的人脸坐标值
                x, y, w, h = enhance_label(src, xmin, ymin, xmax, ymax)
                print(x,y,w,h)
                box.find('xmin').text = str(x)
                box.find('ymin').text = str(y)
                box.find('xmax').text = str(w)
                box.find('ymax').text = str(h)
                box.set('updated', 'yes')
            # 写入新的xml
            tree.write("/home/weiyiran/Public/dis_img/" + a + "_h.xml")
            print("%s is processed." % (i))

def v_flip_func():
    # 指向图片所在的文件夹
    for i in os.listdir("/home/weiyiran/Public/src_img"):
        # 分离文件名与后缀
        a, b = os.path.splitext(i)
        # 如果后缀名是“.jpg”就旋转图像
        if b == ".jpg":
            img_path = os.path.join("/home/weiyiran/Public/src_img", i)
            img = pilimg.open(img_path)
            ng = img.transpose(pilimg.FLIP_LEFT_RIGHT)  # 左右对换。
            # cv2.imwrite("/home/weiyiran/Public/dis_img/" + a + "_h.jpg", ng)
            ng.save("/home/weiyiran/Public/vflip_img/"+ a + "_v.jpg")
            print("log: %s is processed." % (i))
        else:
            xml_path = os.path.join("/home/weiyiran/Public/src_img", i)
            img_path = "/home/weiyiran/Public/src_img/" + a + ".jpg"
            src = cv2.imread(img_path)
            tree = ET.parse(xml_path)
            root = tree.getroot()
            for box in root.iter('bndbox'):
                xmin = float(box.find('xmin').text)
                ymin = float(box.find('ymin').text)
                xmax = float(box.find('xmax').text)
                ymax = float(box.find('ymax').text)
                print(xmin,ymin,xmax,ymax)
                # x, y, w, h = rotate_xml(src, xmin, ymin, xmax, ymax, angle)
                # 改变xml中的人脸坐标值
                x, y, w, h = enhance_label(src, xmin, ymin, xmax, ymax)
                print(x,y,w,h)
                box.find('xmin').text = str(x)
                box.find('ymin').text = str(y)
                box.find('xmax').text = str(w)
                box.find('ymax').text = str(h)
                box.set('updated', 'yes')
            # 写入新的xml
            tree.write("/home/weiyiran/Public/vflip_img/" + a + "_v.xml")
            print("%s is processed." % (i))


def rotate90():
    angle=90
    # 指向图片所在的文件夹
    for i in os.listdir("/home/weiyiran/Public/src_img"):
        # 分离文件名与后缀
        a, b = os.path.splitext(i)
        # 如果后缀名是“.jpg”就旋转图像
        if b == ".jpg":
            img_path = os.path.join("/home/weiyiran/Public/src_img", i)
            img = cv2.imread(img_path)
            rotated_img = rotate_image(img, angle)
            # ng = img.transpose(pilimg.ROTATE_90)  # 90'
            cv2.imwrite("/home/weiyiran/Public/r90flip_img/" + a + "_r90.jpg", rotated_img)
            # ng.save("/home/weiyiran/Public/r90flip_img/"+ a + "_r90.jpg")
            print("log: %s is processed." % (i))
        else:
            xml_path = os.path.join("/home/weiyiran/Public/src_img", i)
            img_path = "/home/weiyiran/Public/src_img/" + a + ".jpg"
            src = cv2.imread(img_path)
            tree = ET.parse(xml_path)
            root = tree.getroot()
            for box in root.iter('bndbox'):
                xmin = float(box.find('xmin').text)
                ymin = float(box.find('ymin').text)
                xmax = float(box.find('xmax').text)
                ymax = float(box.find('ymax').text)
                print(xmin,ymin,xmax,ymax)
                x, y, w, h = rotate_xml(src, xmin, ymin, xmax, ymax, angle)
                # 改变xml中的人脸坐标值
                # x, y, w, h = enhance_label(src, xmin, ymin, xmax, ymax)
                print(x,y,w,h)
                box.find('xmin').text = str(x)
                box.find('ymin').text = str(y)
                box.find('xmax').text = str(x+w)
                box.find('ymax').text = str(y+h)
                box.set('updated', 'yes')
            # 写入新的xml
            tree.write("/home/weiyiran/Public/r90flip_img/" + a + "_r90.xml")
            print("%s is processed." % (i))

def rotate270():
    angle=270
    # 指向图片所在的文件夹
    for i in os.listdir("/home/weiyiran/Public/src_img"):
        # 分离文件名与后缀
        a, b = os.path.splitext(i)
        # 如果后缀名是“.jpg”就旋转图像
        if b == ".jpg":
            img_path = os.path.join("/home/weiyiran/Public/src_img", i)
            img = cv2.imread(img_path)
            rotated_img = rotate_image(img, angle)
            # ng = img.transpose(pilimg.ROTATE_90)  # 90'
            cv2.imwrite("/home/weiyiran/Public/r270flip_img/" + a + "_r270.jpg", rotated_img)
            # ng.save("/home/weiyiran/Public/r90flip_img/"+ a + "_r90.jpg")
            print("log: %s is processed." % (i))
        else:
            xml_path = os.path.join("/home/weiyiran/Public/src_img", i)
            img_path = "/home/weiyiran/Public/src_img/" + a + ".jpg"
            src = cv2.imread(img_path)
            tree = ET.parse(xml_path)
            root = tree.getroot()
            for box in root.iter('bndbox'):
                xmin = float(box.find('xmin').text)
                ymin = float(box.find('ymin').text)
                xmax = float(box.find('xmax').text)
                ymax = float(box.find('ymax').text)
                print(xmin,ymin,xmax,ymax)
                x, y, w, h = rotate_xml(src, xmin, ymin, xmax, ymax, angle)
                # 改变xml中的人脸坐标值
                # x, y, w, h = enhance_label(src, xmin, ymin, xmax, ymax)
                print(x,y,w,h)
                box.find('xmin').text = str(x)
                box.find('ymin').text = str(y)
                box.find('xmax').text = str(x+w)
                box.find('ymax').text = str(y+h)
                box.set('updated', 'yes')
            # 写入新的xml
            tree.write("/home/weiyiran/Public/r270flip_img/" + a + "_r270.xml")
            print("%s is processed." % (i))




if __name__=="__main__":
    #Choose corresponding code accroding to your needs
    h_flip = True  # 1.horizontal flip
    h_flip_func()

    # v_flip = True  # 2.vertical flip
    # v_flip_func()
    #
    # hv_flip = True  # 3.both horizontal and vertical flip
    # hv_flip_func()
    #
    # rotate270() # 4.rotate 90
    #
    # rotate90()  # 5.rotate 270
