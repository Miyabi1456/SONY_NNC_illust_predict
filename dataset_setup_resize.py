#coding: utf-8
import cv2
import numpy as np
import os
from itertools import chain
import glob

#画像ファイルへのパスをリスト形式でimg_data,ファイル名をimg_nameに代入
input_dir = "./original_size/other_data_20180812"
output_dir = "./other_data_20180812"

ext_list = ["jpg", "png","jpeg"]
img_path_list = list(chain.from_iterable([glob.glob(os.path.join(input_dir, "*." + ext)) for ext in ext_list]))

size_x = 0 #画像の横解像度
size_y = 0 #画像の縦解像度
x_amp = 1.0 #リサイズ倍率
y_amp = 1.0 #リサイズ倍率
img_width = 256 #目的サイズ
img_height = 256 #目的サイズ
x = 0 #トリミング始点
y = 0 #トリミング始点

"""
短辺側の解像度が目的のサイズになるようにリサイズ,その後中央をトリミングする.
"""

for i,img_path in enumerate(img_path_list):
    img = cv2.imread(img_path)
    size_x = img.shape[1]
    size_y = img.shape[0]


    #短辺側を基準にリサイズ
    if size_x > size_y:
        amp = img_height / size_y
        img = cv2.resize(img,(int(size_x*amp),img_height))
        x = (img.shape[1] - img_width)//2
        img = img[0:img_height,x:x+img_width]

    elif size_x < size_y:
        amp = img_width / size_x
        img = cv2.resize(img,(img_width,int(size_y*amp)))
        y = (img.shape[0] - img_height)//2
        img = img[y:y+img_height,0:img_width]

    else: #size_x = size_y
        img = cv2.resize(img,(img_width,img_height))


    cv2.imwrite(os.path.join(output_dir,os.path.basename(img_path)),img)

    #プログレスバー
    bar = "#" * int(50*i/len(img_path_list)) + " " * int((50*(1-i/len(img_path_list))))
    print("\r[{0}]{1}/{2} 処理中".format(bar,i+1,len(img_path_list)),end="")
