import csv
import math
import operator
from operator import itemgetter

import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
from collections import Counter
from yolo import yolo

input_dir = "python_split_image_by_ch/data"
output_dir = "python_split_image_by_ch/output_data"
output_dir_colors = "color_cars_train/output_data"
input_dir_color = "color_cars_train/data"
img_count_path_color = "color_cars_train/image_counter.txt"
output_dir_csv = "output.csv"
output_color_csv = "C_V/output_color.csv"
img_count_path = "python_split_image_by_ch/image_counter.txt"
f = open(img_count_path, "r")
img_count = f.readlines()
f.close()
car_classifier = cv2.CascadeClassifier('haarcascade_car.xml')
zeros = "00000000"
zero_count = int(math.log10(int(img_count[0])))


def path_to_list(input):
    files = os.listdir(input)
    temp = map(lambda name: os.path.join(input, name), files)
    return temp


def merge_channels(input_dir, output_dir):
    path = path_to_list(input_dir)
    path = list(path)
    for index in range(0, int(len(path) / 3)):
        imgb = cv2.imread(path[index * 3])
        imgg = cv2.imread(path[index * 3 + 1])
        imgr = cv2.imread(path[index * 3 + 2])
        height, width = imgb.shape[:2]
        img = np.zeros((height, width, 3), np.uint8)
        for line in range(0, height):
            for pixel in range(0, width):
                img[line][pixel][2] = imgr[line][pixel][2]
                img[line][pixel][1] = imgg[line][pixel][1]
                img[line][pixel][0] = imgb[line][pixel][0]

        if (len(str(index)) != zero_count):
            output_path = output_dir + "\\img_" + zeros[-(zero_count - len(str(index))):] + str(index) + ".png"
        else:
            output_path = output_dir + "\\img_" + str(index) + ".png"

        cv2.imwrite(output_path, img)


def find_car(input_dir, output_cars="output.csv"):
    path = path_to_list(input_dir)
    path = list(path)
    path.sort()
    vd = yolo()
    l = list()
    col = ['name', 'cars']
    for image_path in path:
        image = cv2.imread(image_path)
        vehicle_boxes = vd.detect_vechicles(image)
        symbol = ''
        z = 1
        while symbol != "i":
            symbol = image_path[len(image_path) - z]
            z += 1
        name = image_path[-z:]
        name = name[1:]
        dict = {}
        if (vehicle_boxes):
            dict["name"] = name
            dict["cars"] = "TRUE"
        else:
            dict["name"] = name
            dict["cars"] = "FALSE"
        l.append(dict)
    with open(output_cars, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=col)
        writer.writeheader()
        writer.writerows(l)


def calc_metric(image, x, y, w, h):
    image = image[y:y + h, x:x + w]
    # cv2.imshow("Cropped image", image)
    average_color_row = np.average(image, axis=0)
    average_color = np.average(average_color_row, axis=0)
    d_img = np.ones((150, 300, 3), dtype=np.uint8)
    d_img[:, :] = average_color
    cv2.imshow("Average Color", d_img)
    clt = KMeans(n_clusters=4)
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    labels = clt.fit_predict(image)
    label_counts = Counter(labels)
    dominant_color = clt.cluster_centers_[label_counts.most_common(1)[0][0]]
    img = np.zeros((150, 300, 3), np.uint8)
    img[:, :] = (list(dominant_color)[0], list(dominant_color)[1], list(dominant_color)[2])
    cv2.imshow("Dominant colo", dominant_color)
    cv2.waitKey()
    return list(dominant_color)


def find_color(input_dir, output_file="output_color.csv"):
    path = path_to_list(input_dir)
    path = list(path)
    path.sort()
    vd = yolo()
    l = list()
    col = ['name', 'color']
    i = 0
    for image_path in path:
        if i > 63:
            break
        i += 1
        print(i)
        image = cv2.imread(image_path)
        vehicle_boxes = vd.detect_vechicles(image)
        symbol = ''
        z = 1
        while symbol != "i":
            symbol = image_path[len(image_path) - z]
            z += 1
        name = image_path[-z:]
        name = name[1:]
        dict = {}
        xmax = 1
        ymax = 1
        wmax = 1
        hmax = 1
        Smax = 1
        for (x, y, w, h) in vehicle_boxes:
            s = w * h
            if (s > Smax):
                Smax = s
                xmax = x
                ymax = y
                wmax = w
                hmax = h
        d_img = np.ones((150, 300, 3), dtype=np.uint8)
        dominant_color = calc_metric(image, xmax, ymax, wmax, hmax)
        d_img[:, :] = (list(dominant_color)[0], list(dominant_color)[1], list(dominant_color)[2])
        d_img = cv2.cvtColor(d_img, cv2.COLOR_BGR2HSV)
        dict["name"] = name
        dict["color"] = "black"
        if (d_img[0][0][0] > 22 and d_img[0][0][0] < 38):
            dict["name"] = name
            dict["color"] = "yellow"
        if (d_img[0][0][0] > 38 and d_img[0][0][0] < 85):
            dict["name"] = name
            dict["color"] = "green"
        if (d_img[0][0][0] > 75 and d_img[0][0][0] < 130):
            dict["name"] = name
            dict["color"] = "blue_cyan"
        if (d_img[0][0][0] > 130 and d_img[0][0][0] < 179):
            dict["name"] = name
            dict["color"] = "red"
        if (d_img[0][0][1] < 35):
            dict["name"] = name
            dict["color"] = "white_silver"
        if (d_img[0][0][2] < 35):
            dict["name"] = name
            dict["color"] = "black"
        l.append(dict)
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=col)
        writer.writeheader()
        writer.writerows(l)


merge_channels(input_dir, output_dir)