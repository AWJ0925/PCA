from tensorflow.keras import layers, models
from sklearn.decomposition import PCA
import os
import datetime
import numpy as np
import tensorflow as tf


# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 禁用GPU

'''-------------------使用GPU----------------'''
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

print('GPU:', tf.test.is_gpu_available())


'''-------------------使用封装好的pca预处理MNIST的28*28--->10*10-----------------'''
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()  # 读取数据
# [60000, 28, 28] => [60000, 28 * 28]
x_train_reshape = x_train.reshape(-1, 28 * 28)
x_test_reshape = x_test.reshape(-1, 28 * 28)

components = 100
# pca
pca = PCA(copy=True, iterated_power='auto', n_components=components, random_state=None,
          svd_solver='auto', tol=0.0, whiten=False)
pca.fit(x_train_reshape)
pca.fit(x_test_reshape)
# 提取图片主成分
x_train_reduction = pca.transform(x_train_reshape)
x_test_reduction = pca.transform(x_test_reshape)

# reshape
x_train_inverse = x_train_reduction.reshape(-1, int(np.sqrt(components)), int(np.sqrt(components)), 1)
x_test_inverse = x_test_reduction.reshape(-1, int(np.sqrt(components)), int(np.sqrt(components)), 1)

# 数据类型转换
x_train_inverse = tf.cast(x_train_inverse, dtype=tf.float32)
x_test_inverse = tf.cast(x_test_inverse, tf.float32)


'''-------------------BUild Model-----------------'''
def cnn_model(x=int(np.sqrt(components)), y=int(np.sqrt(components))):
    # Build Model
    cnn_Model = models.Sequential()
    cnn_Model.add(layers.Conv2D(8, (2, 2), activation='relu', name='conv1', input_shape=(x, y, 1)))
    cnn_Model.add(layers.MaxPooling2D((2, 2)))
    cnn_Model.add(layers.Conv2D(16, (2, 2), activation='relu'))
    cnn_Model.add(layers.Flatten())
    cnn_Model.add(layers.Dense(32, activation='relu'))  # 注意激活函数
    cnn_Model.add(layers.Dense(10, activation='softmax'))
    # Compile Model
    cnn_Model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.0001),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
    return cnn_Model


cnnModel = cnn_model()
cnnModel.summary()

# 统计时间
start_time = datetime.datetime.now()
'''-----------------------Train---------------------'''
cnn_history = cnnModel.fit(x_train_inverse, y_train,
                           batch_size=128,
                           steps_per_epoch=400,
                           epochs=1000,
                           validation_data=(x_test_inverse, y_test))
end_time = datetime.datetime.now()
print("Total training time {}".format((end_time - start_time).total_seconds()))
