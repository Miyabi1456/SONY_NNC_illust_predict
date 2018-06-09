#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import os
import shutil
import sys
import glob
from itertools import chain
import numpy as np
import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF

########################################保存先の指定など#########################################################
current_dir = os.path.dirname(__file__) #このファイルまでの絶対パス
input_dir = os.path.join(current_dir,"predict_input") #推論を実行したいファイルを保存しておく
illust_dir = os.path.join(current_dir,"predict_output/illust")
other_dir = os.path.join(current_dir,"predict_output/other")
################################################################################################################

def network(x, test=False):
    # Input -> 3,256,256
    # Convolution -> 32,255,255
    with nn.parameter_scope('Convolution'):
        h = PF.convolution(x, 32, (2,2), (0,0))
    # BatchNormalization_5
    with nn.parameter_scope('BatchNormalization_5'):
        h = PF.batch_normalization(h, (1,), 0.9, 0.0001, not test)
    # MaxPooling_3 -> 32,127,127
    h = F.max_pooling(h, (2,2), (2,2), True)
    # ReLU
    h = F.relu(h, True)
    # Convolution_4 -> 32,126,126
    with nn.parameter_scope('Convolution_4'):
        h = PF.convolution(h, 32, (2,2), (0,0))
    # BatchNormalization_2
    with nn.parameter_scope('BatchNormalization_2'):
        h = PF.batch_normalization(h, (1,), 0.9, 0.0001, not test)
    # MaxPooling_2 -> 32,63,63
    h = F.max_pooling(h, (2,2), (2,2))
    # ReLU_2
    h = F.relu(h, True)

    # Convolution_2 -> 32,62,62
    with nn.parameter_scope('Convolution_2'):
        h = PF.convolution(h, 32, (2,2), (0,0))
    # BatchNormalization_4
    with nn.parameter_scope('BatchNormalization_4'):
        h = PF.batch_normalization(h, (1,), 0.9, 0.0001, not test)
    # ReLU_3
    h = F.relu(h, True)
    # MaxPooling -> 32,31,31
    h = F.max_pooling(h, (2,2), (2,2))
    # Convolution_3 -> 32,30,30
    with nn.parameter_scope('Convolution_3'):
        h = PF.convolution(h, 32, (2,2), (0,0))
    # BatchNormalization
    with nn.parameter_scope('BatchNormalization'):
        h = PF.batch_normalization(h, (1,), 0.9, 0.0001, not test)
    # ReLU_4
    h = F.relu(h, True)
    # MaxPooling_4 -> 32,15,15
    h = F.max_pooling(h, (2,2), (2,2))

    # Affine_2 -> 256
    with nn.parameter_scope('Affine_2'):
        h = PF.affine(h, (256,))
    # BatchNormalization_6
    with nn.parameter_scope('BatchNormalization_6'):
        h = PF.batch_normalization(h, (1,), 0.9, 0.0001, not test)
    # ReLU_5
    h = F.relu(h, True)
    # Affine_3 -> 128
    with nn.parameter_scope('Affine_3'):
        h = PF.affine(h, (128,))
    # BatchNormalization_7
    with nn.parameter_scope('BatchNormalization_7'):
        h = PF.batch_normalization(h, (1,), 0.9, 0.0001, not test)
    # ReLU_6
    h = F.relu(h, True)
    # Affine -> 1
    with nn.parameter_scope('Affine'):
        h = PF.affine(h, (1,))
    # BatchNormalization_3
    with nn.parameter_scope('BatchNormalization_3'):
        h = PF.batch_normalization(h, (1,), 0.9, 0.0001, not test)
    # Sigmoid
    h = F.sigmoid(h)
    return h

def image_preproccess(image_path):
    """
    画像を読み込んで,短辺側の解像度を256pixelにリサイズして,中央を切り抜きして正方サイズにする.
    cv2とNNCの画像の取扱の仕様により,転置する.
    戻り値はトリミングされた画像(色,高さ,幅).
    """
    size_x = 0 #画像の横解像度
    size_y = 0 #画像の縦解像度
    x_amp = 1.0 #リサイズ倍率
    y_amp = 1.0 #リサイズ倍率
    img_width = 256 #目的サイズ
    img_height = 256 #目的サイズ
    x = 0 #トリミング始点
    y = 0 #トリミング始点

    img = cv2.imread(image_path)
    size_x = img.shape[1]
    size_y = img.shape[0]

    #短辺側を基準にリサイズ
    if size_x > size_y:
        y_amp = img_height / size_y
        x_amp = y_amp
        img = cv2.resize(img,(int(size_x*x_amp),img_height))
        x = (img.shape[1] - img_width)//2
        img = img[0:img_height,x:x+img_width] #中央をトリミング
    else:
        x_amp = img_width / size_x #x_amp<0
        y_amp = x_amp
        img = cv2.resize(img,(img_width,int(size_y*y_amp)))
        y = (img.shape[0] - img_height)//2
        img = img[y:y+img_height,0:img_width] #中央をトリミング

    img = img / 255.0 #学習時に正規化したための処理
    img = img.transpose(2,0,1) #openCVは(高さ,幅,色)なので転置する必要あり.

    return img

class Predict():
    """
    与えられた画像に対して推論を行う.
    戻り値には,その推論の結果を返す.
    """
    def __init__(self):
        #パラメタの初期化
        nn.clear_parameters()

        #入力変数の準備
        self.x = nn.Variable((1,3,256,256)) #(枚数,色,高さ,幅)

        #パラメタの読み込み
        nn.load_parameters(os.path.join(current_dir,"parameters.h5"))

        #推論ネットワークの構築
        self.y = network(self.x,test=True) 

    def pred(self,image_path):
        img = image_preproccess(image_path) #入力の画像
        self.x.d = img.reshape(self.x.shape) #画像を(1,3,256,256)の行列に整形し,x.dに代入する.
        self.y.forward() #推論の実行

        return self.y.d[0]

def main():
    #jpg拡張子のファイルへのパスをリスト形式でimagesに代入する
    ext_list = ["jpg", "png","jpeg"]
    images_path = list(chain.from_iterable([glob.glob(os.path.join(input_dir, "*." + ext)) for ext in ext_list]))
    print("フォルダ内の写真は"+str(len(images_path))+"枚")

    pred = Predict() #ネットワークが形成される.
    for image_path in images_path:
        y = pred.pred(image_path)

        if y<0.5:
            print("イラスト"+str(y))
            shutil.move(image_path,illust_dir)
        else:
            print("その他  "+str(y))
            shutil.move(image_path,other_dir)

if __name__ == "__main__":
    main()