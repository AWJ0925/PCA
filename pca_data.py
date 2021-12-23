"""
目标：使用封装好的pca算法预处理MNIST数据集
（1）设置pca的参数，主要是n_components（降到多少维度）
（2）多维的数据变为一维度的数据
（2）pca.fit() and pca.transform()
"""
from sklearn.decomposition import PCA
import tensorflow as tf


# 读取数据
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# (1) [60000, 28, 28] => [60000, 28 * 28]
x_train_re = x_train.reshape(-1, 28 * 28)

# (2) 保留图片 95% 的特征
pca = PCA(n_components=100, copy=True, iterated_power='auto',
          random_state=None, svd_solver='auto', tol=0.0, whiten=False)
pca.fit(x_train_re)
x_train_redu = pca.transform(x_train_re)  # 提取图片主成分
print(x_train_redu.shape)

# (3) reshape
x_train_reduction = x_train_redu.reshape(-1, 10, 10)
