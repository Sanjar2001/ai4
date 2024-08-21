import os
import pandas as pd
import numpy as np
from PIL import Image

# Загрузим датасет
train = pd.read_csv("/content/fashion-mnist_train.csv")

# Отфильтруем только кроссовки (label == 7)
train_snck = train[train["label"] == 7]

# Создадим папку для сохранения изображений
os.makedirs("/content/sneakers", exist_ok=True)

# Перевод массива пикселей в изображения и сохранение
for idx, row in train_snck.iterrows():
    img_array = np.array(row.drop("label")).reshape(28, 28).astype(np.uint8)
    img = Image.fromarray(img_array)
    img.save(f"/content/sneakers/sneaker_{idx}.png")


from fastai.vision.all import *

# Путь к изображениям кроссовок
path = Path("/content/sneakers")

# Создаем DataBlock и DataLoader
dblock = DataBlock(blocks=(ImageBlock(), CategoryBlock()),
                   get_items=get_image_files,
                   splitter=RandomSplitter(0.2),
                   get_y=parent_label,
                   item_tfms=Resize(28))

dls = dblock.dataloaders(path, bs=64)

# Визуализируем изображения
dls.show_batch(max_n=9, figsize=(6,6))


# Создаем генератор и критик
generator = basic_generator(64, n_channels=1, n_features=64, n_extra_layers=1)
critic = basic_critic(64, n_channels=1, n_features=64, n_extra_layers=1)

# Создаем GAN Learner
learn = GANLearner.wgan(dls, generator, critic, opt_func=Adam,
                        loss_func=AdaptiveLoss(nn.BCEWithLogitsLoss), metrics=None)

# Обучаем модель
learn.fit(40, lr=1e-4)

# Сохраняем модель после обучения
learn.save('gan-sneakers-model')


# Эксперименты с обучением
learn.fine_tune(20, base_lr=1e-5)  # Пример дообучения с меньшим learning rate

# Визуализация результатов
learn.show_results(max_n=9, figsize=(8,8))
