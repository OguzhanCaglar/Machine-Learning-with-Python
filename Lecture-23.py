from tensorflow.keras.datasets import cifar10

(training_dataset_x, training_dataset_y), (test_dataset_x, test_dataset_y) = cifar10.load_data()

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

print(f'training_dataset_x.shape: {training_dataset_x.shape}' )
print(f'training_dataset_y.shape: {training_dataset_y.shape}' )
print(f'test_dataset_x.shape:     {test_dataset_x.shape}' )
print(f'test_dataset_y.shape:     {test_dataset_y.shape}' )

# import matplotlib.pyplot as plt

# figure = plt.gcf()
# figure.set_size_inches(10, 10)
# for i in range(1, 10):
#     plt.subplot(3, 3, i)
#     axis = plt.gca()
#     axis.set_title(class_names[training_dataset_y[i, 0]])
#     plt.imshow(training_dataset_x[i], cmap='gray')

# plt.show()

# min-max ölçeklendirmesi
training_dataset_x = training_dataset_x / 255
test_dataset_x = test_dataset_x / 255

# ohe
from tensorflow.keras.utils import to_categorical

training_dataset_y = to_categorical(training_dataset_y)
test_dataset_y = to_categorical(test_dataset_y)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D

model = Sequential()
model.add(Conv2D(16, kernel_size=(3, 3), input_shape=(32, 32, 3), activation='relu', name='Convolution-1'))
model.add(MaxPooling2D(name='MaxPooling-1'))
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', name='Convolution-2'))
model.add(MaxPooling2D(name='MaxPooling-2'))
model.add(Flatten(name='Flatten'))
model.add(Dense(32, activation='relu', name='Hidden-1'))
model.add(Dense(10, activation='softmax', name='Output'))

model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
hist = model.fit(training_dataset_x, training_dataset_y, epochs=5, batch_size=64, validation_split=0.2)

import matplotlib.pyplot as plt

plt.figure(figsize=(15, 5))
plt.title('Epoch-Loss Graph', fontsize=14, fontweight='bold')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(range(1, len(hist.history['loss']) + 1), hist.history['loss'])
plt.plot(range(1, len(hist.history['val_loss']) + 1), hist.history['val_loss'])
plt.legend(['Loss', 'Validation Loss'])
plt.show()

plt.figure(figsize=(15, 5))
plt.title('Categorical Accuracy - Epoch Graph', fontsize=14, fontweight='bold')
plt.xlabel('Epochs')
plt.ylabel('Categorical Accuracy')
plt.plot(range(1, len(hist.history['categorical_accuracy']) + 1), hist.history['categorical_accuracy'])
plt.plot(range(1, len(hist.history['val_categorical_accuracy']) + 1), hist.history['val_categorical_accuracy'])
plt.legend(['Categorical Accuracy', 'Validation Categorical Accuracy'])
plt.show()

eval_result = model.evaluate(test_dataset_x, test_dataset_y)
for i in range(len(eval_result)):
    print(f'{model.metrics_names[i]} --> {eval_result[i]}')

# Şimdi yeni resim bulup 9 fotoğrafı sınıflandırmaya çalışalım. Resimler 32x32x3

import numpy as np
import itertools
import glob

figure = plt.gcf()
figure.set_size_inches((10,10))
for index, path in enumerate(itertools.islice(glob.glob('test-cifar10/*.jpg'), 9)):
    img_data = plt.imread(path)
    scaled_img_data = img_data / 255
    result = model.predict(scaled_img_data.reshape(1, 32, 32, 3))
    number = np.argmax(result)
    plt.subplot(3, 3, index + 1)
    axis = plt.gca()
    axis.set_title(class_names[number])
    plt.imshow(img_data)

plt.show()