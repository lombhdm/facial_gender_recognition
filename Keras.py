import  os
os.environ['CUDA_VISIBLE_DEVICES'] = '/gpu:0'
import  numpy as np
import  cv2
import time
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
filename = './data/'


def read_data(filename=filename, rate=0.2):
    # 读取女性
    imgs_female = os.listdir(filename + 'female/')
    # 标记为0
    labels_female = list(map(lambda x: 0, imgs_female))
    X_female = np.zeros(shape=[len(imgs_female), 92, 112, 1])
    y_female = np.zeros(shape=[len(labels_female), 1])
    for index, img in enumerate(imgs_female):
        # 以灰度进行读取
        vector = cv2.imread(filename + 'female/' + img, 0)
        # 归一化
        vector = vector.flatten() / 255
        vector = np.reshape(vector, [92, 112, 1])
        X_female[index, :, :, :] = vector
    for index, label in enumerate(labels_female):
        y_female[index, :] = label

    # 读取男性
    imgs_male = os.listdir(filename + 'male/')
    labels_male = list(map(lambda x: 1, imgs_male))
    X_male = np.zeros(shape=[len(imgs_male), 92, 112, 1])
    y_male = np.zeros(shape=[len(labels_male), 1])
    for index, img in enumerate(imgs_male):
        vector = cv2.imread(filename + 'male/' + img, 0)
        vector = vector.flatten() / 255
        vector = np.reshape(vector, [92, 112, 1])
        X_male[index, :, :, :] = vector
    for index, label in enumerate(labels_male):
        y_male[index, :] = label

    # 合并
    # [399,92,112,1]
    X = np.r_[X_female, X_male]
    # [399,1]
    y = np.r_[y_female, y_male]

    # shuffle
    index = np.random.permutation(len(X))
    X = X[index]
    y = y[index]

    # train:test 8：2
    test_num = int(len(X) * rate)
    # [320,92,112,1]
    train = X[test_num:, :, :, :]
    # [320,1]
    train_labels = y[test_num:, :]
    test = X[:test_num, :, :, :]
    test_labels = y[:test_num, :]

    return train, train_labels, test, test_labels


a = datetime.now() # 获得当前时间

train, train_labels, test, test_labels = read_data()

input_shape = (92, 112, 1)
model = keras.Sequential()
# 卷积1
model.add(keras.layers.Conv2D(16, kernel_size=(3, 3), strides=(1, 1),
                              padding='same', activation='relu', input_shape=input_shape))
# 池化1
model.add(keras.layers.MaxPool2D(strides=(2, 2), padding='same'))

# 卷积2
model.add(keras.layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'))
# 池化2
model.add(keras.layers.MaxPool2D(strides=(2, 2), padding='same'))

# 卷积3
model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
# 池化3
model.add(keras.layers.MaxPool2D(strides=(2, 2), padding='same'))

model.add(keras.layers.Flatten())
# 全连接1
model.add(keras.layers.Dense(512, activation='relu'))
model.add(keras.layers.Dropout(0.4))
# 全连接2
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dropout(0.4))
# 输出
model.add(keras.layers.Dense(1, activation='sigmoid'))

# 编译
model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.binary_crossentropy,
              metrics=['accuracy'])

# 训练
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)
model.fit(train, train_labels, batch_size=32, epochs=20, validation_split=0.2, callbacks=[early_stopping])
model.save('my_model.h5')
score = model.evaluate(test, test_labels, batch_size=32)
print(score)

b = datetime.now()  # 获取当前时间
total_time = (b-a).seconds  # 时间差，以秒显示
print("train_time:",total_time,"seconds")

#随便选取一张图片测试训练效果

#显示选择的图片
img = cv2.imread('./data/face201.bmp',0)
img = cv2.resize(img,(112,92))
cv2.namedWindow('my_img')
cv2.imshow('my_img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#查看识别结果
model = keras.models.load_model('my_model.h5')
img = cv2.imread('./data/face201.bmp',0)  #选取照片的路径，注意0为female，1为male
img = cv2.resize(img,(112,92))
img = img.reshape((1,92,112,1))
print(model.predict(img))

