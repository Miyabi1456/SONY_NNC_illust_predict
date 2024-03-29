﻿#!/usr/bin/env python
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

        #パラメタの読み込みparam
        nn.load_parameters(os.path.join(current_dir,"parameters.h5"))

        #推論ネットワークの構築
        self.y = self.network(self.x,test=True) 


    def network(self, x, test=False):
        # Input:x -> 3,256,256
        # Convolution_5 -> 16,256,256
        h = PF.convolution(x, 16, (3,3), (1,1), name='Convolution_5')
        # BatchNormalization_9
        h = PF.batch_normalization(h, (1,), 0.9, 0.0001, not test, name='BatchNormalization_9')
        # PReLU_8
        h = PF.prelu(h, 1, False, name='PReLU_8')
        # Convolution_6
        h = PF.convolution(h, 16, (3,3), (1,1), name='Convolution_6')
        # BatchNormalization_5
        h = PF.batch_normalization(h, (1,), 0.9, 0.0001, not test, name='BatchNormalization_5')
        # PReLU_7
        h = PF.prelu(h, 1, False, name='PReLU_7')
        # Convolution_4
        h = PF.convolution(h, 16, (3,3), (1,1), name='Convolution_4')
        # BatchNormalization_2
        h = PF.batch_normalization(h, (1,), 0.9, 0.0001, not test, name='BatchNormalization_2')
        # PReLU_6
        h = PF.prelu(h, 1, False, name='PReLU_6')
        # MaxPooling_2 -> 16,128,128
        h = F.max_pooling(h, (2,2), (2,2), False)
        # Convolution_2 -> 32,128,128
        h = PF.convolution(h, 32, (3,3), (1,1), name='Convolution_2')
        # BatchNormalization_4
        h = PF.batch_normalization(h, (1,), 0.9, 0.0001, not test, name='BatchNormalization_4')
        # PReLU_5
        h = PF.prelu(h, 1, False, name='PReLU_5')
        # MaxPooling -> 32,64,64
        h = F.max_pooling(h, (2,2), (2,2), False)

        # Convolution_3 -> 64,64,64
        h = PF.convolution(h, 64, (3,3), (1,1), name='Convolution_3')
        # BatchNormalization
        h = PF.batch_normalization(h, (1,), 0.9, 0.0001, not test, name='BatchNormalization')
        # PReLU_4
        h = PF.prelu(h, 1, False, name='PReLU_4')
        # MaxPooling_4 -> 64,32,32
        h = F.max_pooling(h, (2,2), (2,2), False)
        # Convolution_7 -> 128,32,32
        h = PF.convolution(h, 128, (3,3), (1,1), name='Convolution_7')
        # BatchNormalization_7
        h = PF.batch_normalization(h, (1,), 0.9, 0.0001, not test, name='BatchNormalization_7')
        # PReLU_3
        h = PF.prelu(h, 1, False, name='PReLU_3')
        # MaxPooling_3 -> 128,16,16
        h = F.max_pooling(h, (2,2), (2,2), False)
        # Convolution_8 -> 256,16,16
        h = PF.convolution(h, 256, (3,3), (1,1), name='Convolution_8')
        # BatchNormalization_10
        h = PF.batch_normalization(h, (1,), 0.9, 0.0001, not test, name='BatchNormalization_10')
        # PReLU_2
        h = PF.prelu(h, 1, False, name='PReLU_2')
        # MaxPooling_5 -> 256,8,8
        h = F.max_pooling(h, (2,2), (2,2), False)
        # Convolution -> 512,8,8
        h = PF.convolution(h, 512, (3,3), (1,1), name='Convolution')
        # BatchNormalization_8
        h = PF.batch_normalization(h, (1,), 0.9, 0.0001, not test, name='BatchNormalization_8')
        # PReLU
        h = PF.prelu(h, 1, False, name='PReLU')

        # AveragePooling -> 512,1,1
        h = F.average_pooling(h, (8,8), (8,8))
        # BatchNormalization_6
        h = PF.batch_normalization(h, (1,), 0.9, 0.0001, not test, name='BatchNormalization_6')
        # PReLU_9
        h = PF.prelu(h, 1, False, name='PReLU_9')
        # Affine -> 1
        h = PF.affine(h, (1,), name='Affine')
        # BatchNormalization_3
        h = PF.batch_normalization(h, (1,), 0.9, 0.0001, not test, name='BatchNormalization_3')
        # y'
        h = F.sigmoid(h)
        return h



    def image_preproccess(self, image_path):
        """
        画像を読み込んで,短辺側の解像度を256pixelにリサイズして,中央を切り抜きして正方サイズにする.
        cv2とNNCの画像の取扱の仕様により,転置する.
        戻り値はトリミングされた画像(色,高さ,幅).
        """
        size_x = 0 #画像の横解像度
        size_y = 0 #画像の縦解像度
        amp = 1.0 #リサイズ倍率
        img_width = 256 #目的サイズ
        img_height = 256 #目的サイズ
        x = 0 #トリミング始点
        y = 0 #トリミング始点

        img = cv2.imread(image_path)
        size_x = img.shape[1]
        size_y = img.shape[0]

        #短辺側を基準にリサイズ
        if size_x > size_y:
            amp = img_height / size_y
            img = cv2.resize(img,(int(size_x*amp),img_height))
            x = (img.shape[1] - img_width)//2
            img = img[0:img_height,x:x+img_width] #中央をトリミング

        elif size_x < size_y:
            amp = img_width / size_x
            img = cv2.resize(img,(img_width,int(size_y*amp)))
            y = (img.shape[0] - img_height)//2
            img = img[y:y+img_height,0:img_width] #中央をトリミング

        else: #size_x = size_y
            img = cv2.resize(img,(img_width,img_height))

        img = img / 255.0 #学習時に正規化したための処理
        img = img.transpose(2,0,1) #openCVは(高さ,幅,色)なので転置する必要あり.

        return img

    def pred(self,image_path):
        img = self.image_preproccess(image_path) #入力の画像
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