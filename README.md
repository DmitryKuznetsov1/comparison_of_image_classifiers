# **Сравнение методов глубоко обучения для классификации изображений**

В этом ноутбуке я првоеду сравнение нескольких моделей классификации изображений: [VGG16](https://keras.io/api/applications/vgg/#vgg16-function), [ResNet50](https://keras.io/api/applications/resnet/#resnet50-function), [MobileNetV2](https://keras.io/api/applications/mobilenet/#mobilenetv2-function), [NASNetMobile](https://keras.io/api/applications/nasnet/#nasnetmobile-function) по следующим показателям: размер, время работы, ошибка, и точность на тестовом множестве. Также будут проведены шумовые и патч атаки.
Для сравнения используются предобученные версии моделей, а для проверки - [ImageNet mini](https://www.kaggle.com/ifigotin/imagenetmini-1000/notebooks)


```python
import os
import numpy as np
import random
import matplotlib.pyplot as plt
import gc

from pathlib import Path
```

Структура датасета


```python
counter = 0

for dirname, _, filenames in os.walk('/kaggle/input'):
    counter += 1
    if counter == 6:
        break
    for filename in filenames:
        print(os.path.join(dirname, filename))
```

    /kaggle/input/imagenetmini-1000/imagenet-mini/val/n04536866/ILSVRC2012_val_00007249.JPEG
    /kaggle/input/imagenetmini-1000/imagenet-mini/val/n04536866/ILSVRC2012_val_00016531.JPEG


Для глобальной проверки моделей используется тренировочное множество.

Для проверки на зашумленных и неполных данных достаточно ограничиться валидационным


```python
TEST_PATH = Path('/kaggle/input/imagenetmini-1000/imagenet-mini/val/')
TEST_PATH_LARGE = Path('/kaggle/input/imagenetmini-1000/imagenet-mini/train/')
MODELS_NAMES = ['VGG16', 'ResNet50', 'MobileNetV2', 'NASNetMobile']

INPUT_SIZE = 224 # Все предобученные модели с фиксированным входом 224х224х3


label_name = [subdir for subdir in os.listdir(TEST_PATH) if os.path.isdir(os.path.join(TEST_PATH, subdir))]
print(f'Количество классов: {len(label_name)}')

img_name_large = [name for subdir in os.listdir(TEST_PATH_LARGE) for name in os.listdir(os.path.join(TEST_PATH_LARGE, subdir)) if os.path.isfile(os.path.join(TEST_PATH_LARGE, subdir, name))]
steps_large = len(img_name_large)
img_name_small = [name for subdir in os.listdir(TEST_PATH) for name in os.listdir(os.path.join(TEST_PATH, subdir)) if os.path.isfile(os.path.join(TEST_PATH, subdir, name))]
steps_small = len(img_name_small)
print(f'Количество изображений: {steps_large + steps_small}')
```

    Количество классов: 1000
    Количество изображений: 38668


Необходимо создать генератор изображений с двумя типами атак: добавление шума и вырез центральной области


```python
from keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_input_vgg16
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_input_resnet50
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess_input_mobilenet_v2
from tensorflow.keras.applications.nasnet import preprocess_input as preprocess_input_nasnetmobile


def add_noise(img, noise_level=0):
    VARIABILITY = noise_level
    deviation = VARIABILITY * random.random()
    noise = np.random.normal(0, deviation, img.shape)
    img += noise
    np.clip(img, 0., 255.)
    return img

def center_crop(img, center_crop_size=(0, 0)):
    centerw, centerh = img.shape[0] // 2, img.shape[1] // 2
    halfw, halfh = center_crop_size[0] // 2, center_crop_size[1] // 2
    img[centerw - halfw:centerw + halfw, centerh - halfh:centerh + halfh, :] = 255
    return img


def noise_and_preprocess(preprocess_func, noise_level=0, noise_func=add_noise):
    return lambda x: preprocess_func(noise_func(x, noise_level))


def crop_and_preprocess(preprocess_func, center_crop_size=(0, 0), crop_func=center_crop):
    return lambda x: preprocess_func(crop_func(x, center_crop_size))


def preprocessed_generator(preprocess_input, path=TEST_PATH, input_size=INPUT_SIZE, noise_level=0, center_crop_size=(0, 0), noise=True, rescale=False, crop=False):
    if crop:
        datagen = ImageDataGenerator(
        rescale = 1./255 if rescale  else 1,
        preprocessing_function=crop_and_preprocess(preprocess_input, center_crop_size=center_crop_size) if noise else preprocess_input
    )
    else:
        datagen = ImageDataGenerator(
            rescale = 1./255 if rescale  else 1,
            preprocessing_function=noise_and_preprocess(preprocess_input, noise_level=noise_level) if noise else preprocess_input
        )
    
    generator = datagen.flow_from_directory(
        path,
        target_size=(INPUT_SIZE, INPUT_SIZE),
        batch_size=1,
        shuffle=False
    )
    return generator
```

# Визуалицаия данных


```python
test_generator_mobilenet_v2_without = preprocessed_generator(preprocess_input_mobilenet_v2, noise_level=0)
test_generator_mobilenet_v2_noised = preprocessed_generator(preprocess_input_mobilenet_v2, noise_level=50)
test_generator_mobilenet_v2_cropped = preprocessed_generator(preprocess_input_mobilenet_v2, center_crop_size=(60, 60), crop=True)


fig, axes = plt.subplots(1, 3)
fontsize = 20

train_sample, y_train = next(test_generator_mobilenet_v2_without)
axes[0].imshow(train_sample[0])
axes[0].set_title('Only preprocessing', fontsize=fontsize)

train_sample, y_train = next(test_generator_mobilenet_v2_noised)
axes[1].imshow(train_sample[0])
axes[1].set_title('Noise with preprocessing', fontsize=fontsize)

train_sample, y_train = next(test_generator_mobilenet_v2_cropped)
axes[2].imshow(train_sample[0])
axes[2].set_title('Cropped center with preprocessing', fontsize=fontsize)

fig.set_figwidth(20)
fig.set_figheight(8)
plt.show()
```

    Found 3923 images belonging to 1000 classes.
    Found 3923 images belonging to 1000 classes.
    Found 3923 images belonging to 1000 classes.



    
![png](plots/cls-method-comparison_10_1.png)
    


# Глобальное тестирование

Импорт предобученных моделей


```python
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.efficientnet import EfficientNetB7
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.applications.nasnet import  NASNetMobile
from time import time

INPUT_SHAPE = (INPUT_SIZE, INPUT_SIZE, 3)
```

Функция для сбора ключевых параметров моделей:
* $size$ $-$ количество параметров модели 
* $time$ $-$ время обработки тестового множества
* $loss = H(p, q)=-\sum\limits_{x}-p(x)log(q(x))$ $-$ ошибка на тестовом множестве
* $acc(accuracy)=\frac{\sum true}{\sum true + \sum false}$ $-$ точность на тестовом множестве


```python
def get_model_statistics(model, generator, steps=steps_large, optimizer=SGD, lr=5e-5, loss=categorical_crossentropy):
    statistics = dict.fromkeys(['size', 'time', 'loss', 'acc'])
    pretrained_model = model()
    pretrained_model.compile(optimizer=optimizer(learning_rate=lr), loss=loss, metrics = ['acc'])
    
    statistics['size'] = pretrained_model.count_params()
    
    start = time()
    test_scores = pretrained_model.evaluate(generator, steps=steps)
    statistics['time'] = time() - start
    
    statistics['loss'] = test_scores[0]
    statistics['acc'] = test_scores[1]
    
    return statistics
```

Сравнение размера, точности, ошибки и времени выполнения предсказаний на оригинальных данных


```python
test_generator_vgg16 = preprocessed_generator(preprocess_input_vgg16, noise=False, path=TEST_PATH_LARGE)
test_generator_resnet50 = preprocessed_generator(preprocess_input_resnet50, noise=False, path=TEST_PATH_LARGE)
test_generator_mobilenet_v2 = preprocessed_generator(preprocess_input_mobilenet_v2, noise=False, path=TEST_PATH_LARGE)
test_generator_nasnetmobile = preprocessed_generator(preprocess_input_nasnetmobile, noise=False, path=TEST_PATH_LARGE)
```

    Found 34745 images belonging to 1000 classes.
    Found 34745 images belonging to 1000 classes.
    Found 34745 images belonging to 1000 classes.
    Found 34745 images belonging to 1000 classes.



```python
with tf.device('/GPU:0'):
    stat_vgg16 = get_model_statistics(VGG16, test_generator_vgg16)
    stat_resnet50 = get_model_statistics(ResNet50, test_generator_resnet50)
    stat_mobilenet_v2 = get_model_statistics(MobileNetV2, test_generator_mobilenet_v2)
    stat_nasnetmobile = get_model_statistics(NASNetMobile, test_generator_nasnetmobile)

noise_statistics = {0: [stat_vgg16, stat_resnet50, stat_mobilenet_v2, stat_nasnetmobile]}
```

    Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels.h5
    553467904/553467096 [==============================] - 5s 0us/step
    34745/34745 [==============================] - 423s 12ms/step - loss: 1.1523 - acc: 0.7019
    Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/resnet/resnet50_weights_tf_dim_ordering_tf_kernels.h5
    102973440/102967424 [==============================] - 1s 0us/step
    34745/34745 [==============================] - 461s 13ms/step - loss: 0.8474 - acc: 0.7741
    Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224.h5
    14540800/14536120 [==============================] - 0s 0us/step
    34745/34745 [==============================] - 365s 11ms/step - loss: 0.8480 - acc: 0.8049
    Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/nasnet/NASNet-mobile.h5
    24231936/24227760 [==============================] - 0s 0us/step
    34745/34745 [==============================] - 868s 25ms/step - loss: 0.8239 - acc: 0.7997



```python
sizes = [model['size'] for model in noise_statistics[0]]
times = [model['time'] for model in noise_statistics[0]]
losses = [model['loss'] for model in noise_statistics[0]]
accuracies = [model['acc'] for model in noise_statistics[0]]
```

Втзуализация сравнения


```python
from matplotlib.ticker import FuncFormatter

N = np.arange(len(MODELS_NAMES))

def sizes_format(x, pos):
    return '%1.1f млн' % (x * 1e-6)


def times_format(x, pos):
    return '%1.1f сек' % (x)


fig = plt.figure(figsize=(20,10))


def plot_bar(stat, i, title, formatter=None, x=np.arange(len(MODELS_NAMES))):
    fontsize = 20
    ax = fig.add_subplot(2, 2,i + 1)
    if formatter:
        formatter = FuncFormatter(formatter)
        ax.yaxis.set_major_formatter(formatter)
    ax.set_title(title, fontsize=fontsize)
    plt.bar(x, stat, color=['r', 'g', 'b', 'yellow'])
    plt.xticks(x, MODELS_NAMES)
    

plot_bar(sizes, title='Sizes', formatter=sizes_format, i=0)
plot_bar(times, title='Times', formatter=times_format, i=1)
plot_bar(losses, title='Losses', i=2)
plot_bar(accuracies, title='Accuracies', i=3)
plt.show()
```


    
![png](plots/cls-method-comparison_21_0.png)
    


Выводы:
1. VGG16 значительно превосходит другие модели по размеру 
2. NASNetMobile имеет самый низкий показатель ошибки и сравнительно долгое время выполнения, однако, как показано ниже, это связанно с архитектурой
3. MobileNetV2 обходит все модели по точности и времени выполнения (даже учитыыая, что вычисления на GPU)

Сравнение времени работы моделей при использовании CPU


```python
with tf.device('/CPU:0'):
    stat_vgg16 = get_model_statistics(VGG16, test_generator_vgg16, steps=200)
    stat_resnet50 = get_model_statistics(ResNet50, test_generator_resnet50, steps=200)
    stat_mobilenet_v2 = get_model_statistics(MobileNetV2, test_generator_mobilenet_v2, steps=200)
    stat_nasnetmobile = get_model_statistics(NASNetMobile, test_generator_nasnetmobile, steps=200)
```

    200/200 [==============================] - 84s 422ms/step - loss: 0.8270 - acc: 0.7650
    200/200 [==============================] - 29s 143ms/step - loss: 0.4683 - acc: 0.8400
    200/200 [==============================] - 11s 55ms/step - loss: 0.5960 - acc: 0.8850
    200/200 [==============================] - 26s 128ms/step - loss: 0.5046 - acc: 0.8850


 Вывод: NASNetMobile хорошо подходит для вычислений на CPU, в отличие от абсолютно для этого непригодного VGG16, по прежнему самый быстрый - MobileNetV2

# Зависимость точностей моделей от уровня шума

Для этого можно использовать валидационное множество, которое меньше тренировочного почти в 10 раз


```python
noise_statistics = {}

for noise_level in np.linspace(0, 100, 11):
    test_generator_vgg16 = preprocessed_generator(preprocess_input_vgg16, noise_level=noise_level)
    test_generator_resnet50 = preprocessed_generator(preprocess_input_resnet50, noise_level=noise_level)
    test_generator_mobilenet_v2 = preprocessed_generator(preprocess_input_mobilenet_v2, noise_level=noise_level)
    test_generator_nasnetmobile = preprocessed_generator(preprocess_input_nasnetmobile, noise_level=noise_level)
    
    stat_vgg16 = get_model_statistics(VGG16, test_generator_vgg16, steps=steps_small)
    stat_resnet50 = get_model_statistics(ResNet50, test_generator_resnet50, steps=steps_small)
    stat_mobilenet_v2 = get_model_statistics(MobileNetV2, test_generator_mobilenet_v2, steps=steps_small)
    stat_nasnetmobile = get_model_statistics(NASNetMobile, test_generator_nasnetmobile, steps=steps_small)
    noise_statistics[noise_level] = [stat_vgg16, stat_resnet50, stat_mobilenet_v2, stat_nasnetmobile]
```

    Found 3923 images belonging to 1000 classes.
    Found 3923 images belonging to 1000 classes.
    Found 3923 images belonging to 1000 classes.
    Found 3923 images belonging to 1000 classes.
    3923/3923 [==============================] - 75s 19ms/step - loss: 1.5045 - acc: 0.6378
    3923/3923 [==============================] - 79s 20ms/step - loss: 1.3807 - acc: 0.6778
    3923/3923 [==============================] - 74s 19ms/step - loss: 1.3864 - acc: 0.6844
    3923/3923 [==============================] - 114s 29ms/step - loss: 1.2536 - acc: 0.7099
    Found 3923 images belonging to 1000 classes.
    Found 3923 images belonging to 1000 classes.
    Found 3923 images belonging to 1000 classes.
    Found 3923 images belonging to 1000 classes.
    3923/3923 [==============================] - 75s 19ms/step - loss: 1.5648 - acc: 0.6309
    3923/3923 [==============================] - 81s 21ms/step - loss: 1.4617 - acc: 0.6630
    3923/3923 [==============================] - 75s 19ms/step - loss: 1.4912 - acc: 0.6610
    3923/3923 [==============================] - 113s 29ms/step - loss: 1.3096 - acc: 0.6974
    Found 3923 images belonging to 1000 classes.
    Found 3923 images belonging to 1000 classes.
    Found 3923 images belonging to 1000 classes.
    Found 3923 images belonging to 1000 classes.
    3923/3923 [==============================] - 76s 19ms/step - loss: 1.7248 - acc: 0.5937
    3923/3923 [==============================] - 82s 21ms/step - loss: 1.6375 - acc: 0.6391
    3923/3923 [==============================] - 75s 19ms/step - loss: 1.8082 - acc: 0.5985
    3923/3923 [==============================] - 115s 29ms/step - loss: 1.4225 - acc: 0.6763
    Found 3923 images belonging to 1000 classes.
    Found 3923 images belonging to 1000 classes.
    Found 3923 images belonging to 1000 classes.
    Found 3923 images belonging to 1000 classes.
    3923/3923 [==============================] - 76s 19ms/step - loss: 1.9788 - acc: 0.5582
    3923/3923 [==============================] - 81s 21ms/step - loss: 1.8522 - acc: 0.5896
    3923/3923 [==============================] - 74s 19ms/step - loss: 2.2581 - acc: 0.5205
    3923/3923 [==============================] - 113s 29ms/step - loss: 1.5800 - acc: 0.6472
    Found 3923 images belonging to 1000 classes.
    Found 3923 images belonging to 1000 classes.
    Found 3923 images belonging to 1000 classes.
    Found 3923 images belonging to 1000 classes.
    3923/3923 [==============================] - 77s 20ms/step - loss: 2.2959 - acc: 0.5220
    3923/3923 [==============================] - 81s 21ms/step - loss: 2.2111 - acc: 0.5394
    3923/3923 [==============================] - 75s 19ms/step - loss: 2.8486 - acc: 0.4369
    3923/3923 [==============================] - 114s 29ms/step - loss: 1.7893 - acc: 0.6087
    Found 3923 images belonging to 1000 classes.
    Found 3923 images belonging to 1000 classes.
    Found 3923 images belonging to 1000 classes.
    Found 3923 images belonging to 1000 classes.
    3923/3923 [==============================] - 77s 20ms/step - loss: 2.6563 - acc: 0.4596
    3923/3923 [==============================] - 81s 21ms/step - loss: 2.5636 - acc: 0.4973
    3923/3923 [==============================] - 76s 19ms/step - loss: 3.4910 - acc: 0.3650
    3923/3923 [==============================] - 114s 29ms/step - loss: 2.0244 - acc: 0.5667
    Found 3923 images belonging to 1000 classes.
    Found 3923 images belonging to 1000 classes.
    Found 3923 images belonging to 1000 classes.
    Found 3923 images belonging to 1000 classes.
    3923/3923 [==============================] - 76s 19ms/step - loss: 3.0360 - acc: 0.4168
    3923/3923 [==============================] - 83s 21ms/step - loss: 3.0895 - acc: 0.4425
    3923/3923 [==============================] - 76s 19ms/step - loss: 4.1339 - acc: 0.3207
    3923/3923 [==============================] - 115s 29ms/step - loss: 2.3337 - acc: 0.5192
    Found 3923 images belonging to 1000 classes.
    Found 3923 images belonging to 1000 classes.
    Found 3923 images belonging to 1000 classes.
    Found 3923 images belonging to 1000 classes.
    3923/3923 [==============================] - 79s 20ms/step - loss: 3.5265 - acc: 0.3691
    3923/3923 [==============================] - 82s 21ms/step - loss: 3.7879 - acc: 0.3915
    3923/3923 [==============================] - 76s 19ms/step - loss: 4.7741 - acc: 0.2717
    3923/3923 [==============================] - 116s 30ms/step - loss: 2.6647 - acc: 0.4744
    Found 3923 images belonging to 1000 classes.
    Found 3923 images belonging to 1000 classes.
    Found 3923 images belonging to 1000 classes.
    Found 3923 images belonging to 1000 classes.
    3923/3923 [==============================] - 77s 20ms/step - loss: 3.9547 - acc: 0.3395
    3923/3923 [==============================] - 82s 21ms/step - loss: 4.4250 - acc: 0.3500
    3923/3923 [==============================] - 76s 19ms/step - loss: 5.3707 - acc: 0.2322
    3923/3923 [==============================] - 115s 29ms/step - loss: 3.0097 - acc: 0.4298
    Found 3923 images belonging to 1000 classes.
    Found 3923 images belonging to 1000 classes.
    Found 3923 images belonging to 1000 classes.
    Found 3923 images belonging to 1000 classes.
    3923/3923 [==============================] - 78s 20ms/step - loss: 4.3612 - acc: 0.3151
    3923/3923 [==============================] - 84s 21ms/step - loss: 5.2328 - acc: 0.3166
    3923/3923 [==============================] - 76s 19ms/step - loss: 5.7340 - acc: 0.2167
    3923/3923 [==============================] - 116s 30ms/step - loss: 3.3146 - acc: 0.3890
    Found 3923 images belonging to 1000 classes.
    Found 3923 images belonging to 1000 classes.
    Found 3923 images belonging to 1000 classes.
    Found 3923 images belonging to 1000 classes.
    3923/3923 [==============================] - 79s 20ms/step - loss: 4.7393 - acc: 0.2743
    3923/3923 [==============================] - 82s 21ms/step - loss: 6.2186 - acc: 0.2812
    3923/3923 [==============================] - 76s 19ms/step - loss: 6.2850 - acc: 0.1899
    3923/3923 [==============================] - 117s 30ms/step - loss: 3.5984 - acc: 0.3658



```python
def plot_noises_and_metric(statistics, models_names, metric):
    
    for (i, name) in enumerate(models_names):
        noises = [key for key in statistics.keys()]
        metrics = [statistics[nl][i][metric] for nl in noises]
        plt.plot(noises, metrics, label=name)
    plt.title(f'{metric} with noise')
    plt.legend()
    plt.xlabel("noises")
    plt.ylabel(f"{metric}")
```


```python
plt.figure(figsize=(30,8))
plt.subplot(1, 2, 1)
plot_noises_and_metric(noise_statistics, MODELS_NAMES, metric='acc')
plt.subplot(1, 2, 2)
plot_noises_and_metric(noise_statistics, MODELS_NAMES, metric='loss')
plt.show()
```


    
![png](plots/cls-method-comparison_30_0.png)
    


Выводы:
1. MobileNetV2 очень чувствителен к добавлению шума
2. NASNetMobile показывает отличные результаты
3. VGG16 и ResNet50 стабильны

# Зависимость очностей моделей от размера вырезанного в центре квадрата


```python
crop_statistics = {}

with tf.device('/GPU:0'):
    for crop_size in np.linspace(20, 160, 6):
        crop_size = int(crop_size)
        
        test_generator_vgg16 = preprocessed_generator(preprocess_input_vgg16, center_crop_size=(crop_size, crop_size), crop=True)
        test_generator_resnet50 = preprocessed_generator(preprocess_input_resnet50, center_crop_size=(crop_size, crop_size), crop=True)
        test_generator_mobilenet_v2 = preprocessed_generator(preprocess_input_mobilenet_v2, center_crop_size=(crop_size, crop_size), crop=True)
        test_generator_nasnetmobile = preprocessed_generator(preprocess_input_nasnetmobile, center_crop_size=(crop_size, crop_size), crop=True)

        stat_vgg16 = get_model_statistics(VGG16, test_generator_vgg16, steps=steps_small)
        stat_resnet50 = get_model_statistics(ResNet50, test_generator_resnet50, steps=steps_small)
        stat_mobilenet_v2 = get_model_statistics(MobileNetV2, test_generator_mobilenet_v2, steps=steps_small)
        stat_nasnetmobile = get_model_statistics(NASNetMobile, test_generator_nasnetmobile, steps=steps_small)
        crop_statistics[crop_size] = [stat_vgg16, stat_resnet50, stat_mobilenet_v2, stat_nasnetmobile]
```

    Found 3923 images belonging to 1000 classes.
    Found 3923 images belonging to 1000 classes.
    Found 3923 images belonging to 1000 classes.
    Found 3923 images belonging to 1000 classes.
    3923/3923 [==============================] - 39s 10ms/step - loss: 1.6550 - acc: 0.6092
    3923/3923 [==============================] - 48s 12ms/step - loss: 1.5798 - acc: 0.6406
    3923/3923 [==============================] - 39s 10ms/step - loss: 1.5812 - acc: 0.6447
    3923/3923 [==============================] - 89s 23ms/step - loss: 1.4186 - acc: 0.6796
    Found 3923 images belonging to 1000 classes.
    Found 3923 images belonging to 1000 classes.
    Found 3923 images belonging to 1000 classes.
    Found 3923 images belonging to 1000 classes.
    3923/3923 [==============================] - 41s 10ms/step - loss: 2.0341 - acc: 0.5343
    3923/3923 [==============================] - 47s 12ms/step - loss: 1.9884 - acc: 0.5753
    3923/3923 [==============================] - 38s 10ms/step - loss: 2.1135 - acc: 0.5486
    3923/3923 [==============================] - 90s 23ms/step - loss: 1.8992 - acc: 0.5942
    Found 3923 images belonging to 1000 classes.
    Found 3923 images belonging to 1000 classes.
    Found 3923 images belonging to 1000 classes.
    Found 3923 images belonging to 1000 classes.
    3923/3923 [==============================] - 39s 10ms/step - loss: 2.6594 - acc: 0.4328
    3923/3923 [==============================] - 47s 12ms/step - loss: 2.6692 - acc: 0.4683
    3923/3923 [==============================] - 37s 9ms/step - loss: 2.8653 - acc: 0.4180
    3923/3923 [==============================] - 82s 21ms/step - loss: 2.7032 - acc: 0.4522
    Found 3923 images belonging to 1000 classes.
    Found 3923 images belonging to 1000 classes.
    Found 3923 images belonging to 1000 classes.
    Found 3923 images belonging to 1000 classes.
    3923/3923 [==============================] - 38s 10ms/step - loss: 3.5871 - acc: 0.2980
    3923/3923 [==============================] - 47s 12ms/step - loss: 3.7252 - acc: 0.3375
    3923/3923 [==============================] - 36s 9ms/step - loss: 3.9358 - acc: 0.2786
    3923/3923 [==============================] - 81s 21ms/step - loss: 3.6705 - acc: 0.3140
    Found 3923 images belonging to 1000 classes.
    Found 3923 images belonging to 1000 classes.
    Found 3923 images belonging to 1000 classes.
    Found 3923 images belonging to 1000 classes.
    3923/3923 [==============================] - 39s 10ms/step - loss: 4.7542 - acc: 0.1769
    3923/3923 [==============================] - 47s 12ms/step - loss: 4.8841 - acc: 0.1861
    3923/3923 [==============================] - 36s 9ms/step - loss: 5.2158 - acc: 0.1341
    3923/3923 [==============================] - 80s 20ms/step - loss: 4.8803 - acc: 0.1764
    Found 3923 images belonging to 1000 classes.
    Found 3923 images belonging to 1000 classes.
    Found 3923 images belonging to 1000 classes.
    Found 3923 images belonging to 1000 classes.
    3923/3923 [==============================] - 38s 10ms/step - loss: 6.0904 - acc: 0.0716
    3923/3923 [==============================] - 46s 12ms/step - loss: 6.5071 - acc: 0.0739
    3923/3923 [==============================] - 35s 9ms/step - loss: 6.6898 - acc: 0.0359
    3923/3923 [==============================] - 83s 21ms/step - loss: 6.1749 - acc: 0.0716



```python
def plot_crop_and_metric(statistics, models_names, metric):
    for (i, name) in enumerate(models_names):
        noises = [key for key in statistics.keys()]
        metrics = [statistics[nl][i][metric] for nl in noises]
        plt.plot(noises, metrics, label=name)
    plt.title(f'{metric} with cropped center')
    plt.legend()
    plt.xlabel("crop sizes")
    plt.ylabel(f"{metric}")
```


```python
plt.figure(figsize=(30,8))
plt.subplot(1, 2, 1)
plot_crop_and_metric(crop_statistics, MODELS_NAMES, metric='acc')
plt.subplot(1, 2, 2)
plot_crop_and_metric(crop_statistics, MODELS_NAMES, metric='loss')
plt.show()
```


    
![png](plots/cls-method-comparison_35_0.png)
    


MobileNetV2 показывает худшие результаты, имея при этом довольно высокую стартовую точность

# Распределение точностей моделей на каждом классе


```python
def get_model(model, optimizer=SGD, lr=5e-5, loss=categorical_crossentropy):
    pretrained_model = model()
    pretrained_model.compile(optimizer=optimizer(learning_rate=lr), loss=loss, metrics = ['acc'])
    return pretrained_model
```


```python
from sklearn.metrics import classification_report

test_generator_vgg16 = preprocessed_generator(preprocess_input_vgg16, path=TEST_PATH_LARGE, noise=False)
test_generator_resnet50 = preprocessed_generator(preprocess_input_resnet50, path=TEST_PATH_LARGE, noise=False)
test_generator_mobilenet_v2 = preprocessed_generator(preprocess_input_mobilenet_v2, path=TEST_PATH_LARGE, noise=False)
test_generator_nasnetmobile = preprocessed_generator(preprocess_input_nasnetmobile, path=TEST_PATH_LARGE, noise=False)

generators = [test_generator_vgg16, test_generator_resnet50, test_generator_mobilenet_v2, test_generator_nasnetmobile]
models = [VGG16, ResNet50, MobileNetV2, NASNetMobile]

precisions = []
for model, generator in zip(models, generators):
    test_labels=generator.classes

    predictions=get_model(model).predict_generator(generator, verbose=1)

    y_pred = np.argmax(predictions, axis=-1)

    report = classification_report(test_labels, y_pred, output_dict=True)
    precision_report = [values['precision'] for values in list(report.values())[:-3]][::-1]
    precisions.append(precision_report)
```

    Found 34745 images belonging to 1000 classes.
    Found 34745 images belonging to 1000 classes.
    Found 34745 images belonging to 1000 classes.
    Found 34745 images belonging to 1000 classes.
    34745/34745 [==============================] - 402s 12ms/step
    34745/34745 [==============================] - 375s 11ms/step
    34745/34745 [==============================] - 290s 8ms/step
    34745/34745 [==============================] - 695s 20ms/step



```python
plt.figure(figsize=(30,16))

colors = ['r', 'g', 'b', 'y']

for i, (precision, name) in enumerate(zip(precisions, MODELS_NAMES)):
    plt.subplot(2, 2, i + 1)
    plt.plot(list(range(1, 1001)), precision, label=name, color=colors[i])
    plt.title('Model precision per class')
    plt.legend()
    plt.xlabel('classes')
    plt.ylabel('precision')
    
plt.show()
```


    
![png](plots/cls-method-comparison_40_0.png)
    


Если присмотреться, можно заметить схожие очертания (даже присутствуют одинаковые выбросы).

Среднее абсолютной разности каждой пары распределений


```python
import itertools

for dist1, dist2 in list(itertools.combinations(precisions, 2)):
    print(np.mean(np.abs(np.array(dist1) - np.array(dist2))))
```

    0.0905690797655298
    0.1107288413392988
    0.10418475459401554
    0.06818586738738804
    0.06818431267007204
    0.06635605896452586


Точности моделей по каждому классу слабо отличаются друг от друга. В дополнение можно посмотреть на отсортированные по невозрастанию распределения


```python
plt.figure(figsize=(30,8))

for precision, name in zip((precisions), MODELS_NAMES):
    plt.plot(list(range(1, 1001)), np.sort(precision)[::-1], label=name)
plt.title('Model precision per class')
plt.legend()
plt.xlabel('classes')
plt.ylabel('precision')
plt.show()
```


    
![png](plots/cls-method-comparison_45_0.png)
    


# Выбор модели

Конечно, в первую очередь все зависит от решаемой задачи и используемых ресурсов. Если данные зашумлены или неполны, однозначно не стоит использовать MobileNetV2.

VGG16 неплохо держится, однако 500 Mb $-$ это слишком большой размер, а при самой низкой точности, к сожалению, ни для какого случая не подходит на фоне того же ResNet50.

ResNet50 не сильно уступает MobileNetV2, а по точности на зашумленных или неполных данных сильно превосходит.

Ну и, наконец, NASNetMobile имеет сравнимую с MobileNetV2 точность и идеально подходит для работы с зашумленными или неполными данными (но на CPU MobileNetV2 превосходит по скорости в несколько раз).

Итог: я остановлюсь на NASNetMobile, как на наиболее универсальной.


```python

```
