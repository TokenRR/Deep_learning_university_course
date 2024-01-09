import numpy as np


def pnn(X_train, Y_train, delta, X_test):
    # Створюємо словник для зберігання ймовірностей класів
    class_probs = {}
    # Циклом проходимо по навчальному набору даних
    for x_train, class_label in zip(X_train, Y_train):
        # Якщо клас ще не в словнику, додаємо його
        if class_label not in class_probs:
            class_probs[class_label] = 0
        # Обчислюємо схожість між тестовими даними та поточними навчальними даними
        similarity = np.exp(-np.sum((x_train - X_test) ** 2) / (2 * delta ** 2))
        # Додаємо обчислену схожість до загальної ймовірності поточного класу
        class_probs[class_label] += similarity
    # Повертаємо клас з найбільшою ймовірністю
    return max(class_probs, key=class_probs.get)


if __name__ == '__main__':
    # навчальний набір
    X_train = np.array([[0.10, 0.30], [0.20, 0.10], [0.50, 0.10],
                        [0.30, 0.20], [0.80, 0.10], [0.60, 0.60],
                        [0.28, 0.15], [0.86, 0.54], [0.30, 0.61],
                        [0.85, 0.13], [0.68, 0.98], [0.00, 0.43]])
    # мітки навчального набору
    Y_train = np.array(['A', 'B', 'C',
                        'A', 'B', 'C',
                        'A', 'B', 'C',
                        'A', 'B', 'C'])
    # тестувальний набір
    X_test = np.array([[0.43, 0.88], [0.63, 0.15],
                       [0.57, 0.54], [0.24, 0.25],
                       [0.55, 0.51], [0.45, 0.82]])
    # параметр точності
    delta = 0.1
    # тестування моделі
    for x_test in X_test:
        pred_class = pnn(X_train, Y_train, delta, x_test)
        print(f'{x_test.tolist()} is {pred_class} class')
