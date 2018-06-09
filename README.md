# SONY_NNC_illust_predict
必要なライブラリのセットアップ:
    conda install scipy scikit-image ipython
    pip install nnabla
    
概要:
    イラストかそうでないかの2値分類を行う。
    
使い方:
    predict_inputディレクトリに分類したい画像を保存しておく。
    illust_predict.pyを実行すると、predict_output下の2つのフォルダにそれぞれ分類される。
