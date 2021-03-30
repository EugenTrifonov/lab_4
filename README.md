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


## 3)Использование случайной части изображения 
## 4)Добавление случайного шума


