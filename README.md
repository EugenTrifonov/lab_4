# Использование техник аугментации данных для улучшения сходимости процесса обучения нейронной сети на примере решения задачи классификации Oregon Wildlife
## 1. С использованием техники обучения Transfer Learning  и оптимальной политики изменения темпа обучения обучить нейронную сеть EfficientNet-B0 (предварительно обученную на базе изображений imagenet) для решения задачи классификации изображений Oregon WildLife с использованием техник аугментации данных.
## 1)Манипуляции с яркостью и контрастом
Файл train_contrast.py
 ```python
 def image_preprocess(image,label):
    new_image = tf.image.adjust_contrast(image, 2)
    new_image = tf.image.adjust_brightness(new_image, 0.4)
    return new_image,label
 ```
 
![leg_cont](https://user-images.githubusercontent.com/80068414/113008418-af214780-917f-11eb-84fb-8f7275f95dc6.jpg)
 
Метрика качества на валидации

![acc_1](https://github.com/EugenTrifonov/lab_4/blob/main/graphs/epoch_categorical_accuracy_contrast.svg)

Функция потерь на валидации

![loss_1](https://github.com/EugenTrifonov/lab_4/blob/main/graphs/epoch_loss_contrast.svg)

По полученным результатам можно определить что наиболее оптимальными являются параметры contrast_factor=2 и delta=0.4. При таких параметрах алгоритм сошёлся к 16-й эпохе,как и исходный алгоритм, точность при этом составила 0.880, что больше точности алгоритма с оптимальной политикой изменения темпа обучения на 0.12% 
## 2)Поворот изображения на случайный угол 
Файл train_rotation.py
```python
new_input=tf.keras.layers.experimental.preprocessing.RandomRotation(0.05,fill_mode='reflect')(inputs)
```

## 3)Использование случайной части изображения 
Файл train_random_crop.py

Предварительно изменяем размер изображения в больший чем необходим для входа нейронной сети
```python
 example['image'] = tf.image.resize(example['image'], tf.constant([250,250]))
```
Применяем использование случайной части изображения с размером необходимым для входа сети(RESIZE_TO=224)
```python
def random_crop(image,label):
      return tf.image.random_crop(image,[RESIZE_TO, RESIZE_TO, 3]),label
```

В некоторых случаях были изменены параметры изменения темпа обучения

Исходные параметры : 
```python
def exp_decay(epoch):
   initial_lrate = 0.1
   k = 0.5
   lrate = initial_lrate * exp(-k*epoch)
   return lrate
lrate = LearningRateScheduler(exp_decay)
```
Изменения:

1)initial_lrate=0.01 k=0.5

2)initial_lrate=0.001 k=0.5

3)initial_lrate=0.01 k=0.55


![leg_crop](https://user-images.githubusercontent.com/80068414/113012156-2e644a80-9183-11eb-8b49-710a63b1f552.jpg)

Метрика качества на валидации 

![acc_2](https://github.com/EugenTrifonov/lab_4/blob/main/graphs/epoch_categorical_accuracy_crop.svg)

Функция потерь на валидации

![loss_2](https://github.com/EugenTrifonov/lab_4/blob/main/graphs/epoch_loss_crop.svg)

По результатам можно сделать вывод,что оптимальным размером изображения является 250х250 px при initial_lrate=0.01 и k=0.5 . При таком значении точность алгоритма стала меньше чем у исходного на 0.99%
## 4)Добавление случайного шума


