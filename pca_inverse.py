# 使用pca降维还原原始图像
from sklearn.decomposition import PCA

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def printImage(images):
    # 上面 5 行是原始图片， 下面 5 行是经过 PCA 提取主成分还原后的图片
    plt.figure(figsize=(1000, 1000))
    for i in range(images.shape[0]):
        plt.subplot(10, 10, i + 1)
        # plt.xticks([])
        # plt.yticks([])
        # plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)
    plt.show()


# 读取数据
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# (1) [60000, 28, 28] => [60000, 28 * 28]
x_train_re = x_train.reshape(-1, 28 * 28)
# (2) 保留图片 95% 的特征
pca = PCA(n_components=100, copy=True, iterated_power='auto',
          random_state=None, svd_solver='auto', tol=0.0, whiten=False)
pca.fit(x_train_re)
x_train_redu = pca.transform(x_train_re)  # 提取图片主成分


# 由提取后的主成分还原图片
x_train_inverse = pca.inverse_transform(x_train_redu)
# [60000, 10 * 10] => [60000, 28, 28]
x_train_inverse = x_train_inverse.reshape(-1, 28, 28)

# 输入的前 50 张+重建的前 50 张图片合并
x_concat = np.vstack([x_train[:50], x_train_inverse[:50]])
print(x_concat.shape)

printImage(x_concat)