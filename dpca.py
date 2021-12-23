# 自己写PCA的算法
import os
import datetime
import numpy as np
import tensorflow as tf


# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 设置不使用GPU

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


def twoDPCA(imgs, dim):
    """imgs 是三维的图像矩阵，[num_images, width, height]"""
    a, b, c = imgs.shape
    average = np.zeros((b, c))
    for i in range(a):
        average += imgs[i, :, :] / (a * 1.0)
    G_t = np.zeros((c, c))
    for j in range(a):
        img = imgs[j, :, :]
        temp = img - average
        G_t = G_t + np.dot(temp.T, temp) / (a * 1.0)
    w, v = np.linalg.eigh(G_t)
    # print('v_shape:{}'.format(v.shape))
    w = w[::-1]
    v = v[::-1]
    '''
    for k in range(c):
        # alpha = sum(w[:k])*1.0/sum(w)
        alpha = 0
        if alpha >= p:
            u = v[:,:k]
            break
    '''
    print('alpha={}'.format(sum(w[:dim] * 1.0 / sum(w))))
    u = v[:, :dim]
    print('u_shape:{}'.format(u.shape))
    return u  # u是投影矩阵


def TTwoDPCA(imgs, dim):
    # TTwoDPCA(images, 15)
    u = twoDPCA(imgs, dim)
    a1, b1, c1 = imgs.shape
    img = []
    for i in range(a1):
        temp1 = np.dot(imgs[i, :, :], u)
        img.append(temp1.T)
    img = np.array(img)
    uu = twoDPCA(img, dim)
    print('uu_shape:{}'.format(uu.shape))
    return u, uu  # uu是投影矩阵


def PCA2D_2D(samples, row_top=10, col_top=10):
    """samples are 2d matrices,[size, row, column]"""
    size = samples[0].shape
    # m*n matrix
    mean = np.zeros(size)

    for s in samples:
        mean = mean + s

    # get the mean of all samples
    mean /= float(len(samples))

    # n*n matrix
    cov_row = np.zeros((size[1], size[1]))
    for s in samples:
        diff = s - mean
        cov_row = cov_row + np.dot(diff.T, diff)
    cov_row /= float(len(samples))
    row_eval, row_evec = np.linalg.eig(cov_row)
    # select the top t evals
    sorted_index = np.argsort(row_eval)
    # using slice operation to reverse
    X = row_evec[:, sorted_index[:-row_top - 1: -1]]

    # m*m matrix
    cov_col = np.zeros((size[0], size[0]))
    for s in samples:
        diff = s - mean
        cov_col += np.dot(diff, diff.T)
    cov_col /= float(len(samples))
    col_eval, col_evec = np.linalg.eig(cov_col)
    sorted_index = np.argsort(col_eval)
    Z = col_evec[:, sorted_index[:-col_top - 1: -1]]

    return X, Z


def image_2D2DPCA(images, u, uu):
    a, b, c = images.shape
    new_images = np.ones((a, uu.shape[1], u.shape[1]))
    for i in range(a):
        Y = np.dot(uu.T, images[i, :, :])
        Y = np.dot(Y, u)
        new_images[i, :, :] = Y

    return new_images


# Testing
if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # 2DPCA
    U, UU = PCA2D_2D(x_train)  # train
    x_train_reduction = image_2D2DPCA(x_train, U, UU)
    x_train_reduction = x_train_reduction.reshape(-1, 10, 10, 1)

    U, UU = PCA2D_2D(x_test)  # test
    x_test_reduction = image_2D2DPCA(x_test, U, UU)
    x_test_reduction = x_test_reduction.reshape(-1, 10, 10, 1)

    # Build Model
    cnnModel = tf.keras.models.Sequential()
    cnnModel.add(tf.keras.layers.Conv2D(8, (2, 2), activation='relu', name='conv1', input_shape=(10, 10, 1)))
    cnnModel.add(tf.keras.layers.MaxPooling2D((2, 2)))
    cnnModel.add(tf.keras.layers.Conv2D(16, (2, 2), activation='relu'))
    cnnModel.add(tf.keras.layers.Flatten())
    cnnModel.add(tf.keras.layers.Dense(32, activation='relu'))  # 注意激活函数
    cnnModel.add(tf.keras.layers.Dense(10, activation='softmax'))
    # Compile Model
    cnnModel.compile(optimizer=tf.optimizers.Adam(learning_rate=0.0001),
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
    cnnModel.summary()
    # Training
    start_time = datetime.datetime.now()
    cnn_history = cnnModel.fit(x_train_reduction, y_train,
                               batch_size=128,
                               steps_per_epoch=200,
                               epochs=20000,
                               validation_data=(x_test_reduction, y_test))
    end_time = datetime.datetime.now()
    print("Total Train Time".format((end_time - start_time).total_seconds()))
