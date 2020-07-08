from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import np_utils
import keras
import numpy as np

classes = ['monkey', 'boar', 'crow']
num_classes = len(classes)
image_size = 50

#メインの関数を定義する
def main():
    X_train, X_test, Y_train, Y_test = np.load("./animal_inc.npy", allow_pickle=True)

    X_train = X_train.astype("float") / 256
    X_test = X_test.astype("float") / 256
    Y_train = np_utils.to_categorical(Y_train, num_classes)
    Y_test = np_utils.to_categorical(Y_test, num_classes)

    model = model_train(X_train, Y_train)
    model_eval(model, X_test, Y_test)

def model_train(X,Y):
    model = Sequential()
    model.add(Conv2D(32,(3,3), padding="same", input_shape=X.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3))
    model.add(Activation('softmax'))

    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

    model.compile(loss='categorical_crossentropy', optimizer=opt ,metrics=['accuracy'])

    model.fit(X, Y, batch_size=32, epochs=100)


    #モデルの保存
    model.save('./animal_inc_cnn.h5')

    return model

def model_eval(model, X, Y):
    scores = model.evaluate(X, Y, verbose=1)
    print('Test Loss:', scores[0])
    print('Test Accuracy:', scores[1])


if __name__ == "__main__":
    main()