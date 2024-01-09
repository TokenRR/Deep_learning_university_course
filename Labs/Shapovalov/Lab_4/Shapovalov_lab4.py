import numpy as np
import tensorflow as tf

from keras import layers, models
from keras.optimizers import SGD 
from keras.datasets import cifar10
from keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score


print('\n\nКМ-03 | Шаповалов Г. Г. | Лаб 4')


print(f'\n\n[LOG]  Завантажуємо та нормалізуємо дані')
batch_size = 64

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train = X_train / 255.0 
X_test  = X_test / 255.0 

y_train = to_categorical(y_train, 10)
y_test  = to_categorical(y_test,  10)

X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

print(f'Розмірність тренувального набору: {len(X_train)}')
print(f'Розмірність тестового набору:     {len(X_test)}')
print(f'Розмірність валідаційного набору: {len(X_val)}')


print(f'\n\n[LOG]  Створюємо модель')
model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation='softmax'))  # 10 класів виводу

sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])


print(f'\n\n[LOG]  Тренуємо модель')
try:
    model = models.load_model('my_model.keras')
except:
    history = model.fit(X_train, y_train, epochs=25, batch_size=batch_size, validation_data=(X_val, y_val))
    model.save('my_model.keras')


print(f'\n\n[LOG]  Тестуємо модель')
test_results = model.evaluate(X_test, y_test, verbose=0)

test_accuracy = test_results[1]
predictions = np.argmax(model.predict(X_test, verbose=0), axis=1)
test_precision = precision_score(np.argmax(y_test, axis=1), predictions, average='weighted')
test_recall = recall_score(np.argmax(y_test, axis=1), predictions, average='weighted')
test_f1_score = f1_score(np.argmax(y_test, axis=1), predictions, average='weighted')

print(f'Accuracy  = {test_accuracy * 100:.2f} %')
print(f'Precision = {test_precision * 100:.2f} %')
print(f'Recall    = {test_recall * 100:.2f} %')
print(f'F1-score  = {test_f1_score * 100:.2f} %')