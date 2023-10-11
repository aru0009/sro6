# sro6
from sklearn.datasets import load_iris
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Классификация ирисов Фишера
# Загрузка датасета
iris = load_iris()
X_iris = iris.data
y_iris = iris.target
# Разделение данных на обучающий и тестовый набор
X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(X_iris, y_iris, test_size=0.2, random_state=42)
# Создание и обучение модели для ирисов
model_iris = KNeighborsClassifier(n_neighbors=3)
model_iris.fit(X_train_iris, y_train_iris)
# Предсказание для ирисов
y_pred_iris = model_iris.predict(X_test_iris)
# Оценка точности для ирисов
accuracy_iris = accuracy_score(y_test_iris, y_pred_iris)
print("Точность модели для ирисов:", accuracy_iris)
# Классификация рукописных цифр MNIST
# Загрузка датасета MNIST
digits = load_digits()
X_digits = digits.data
y_digits = digits.target
# Разделение данных на обучающий и тестовый набор для цифр
X_train_digits, X_test_digits, y_train_digits, y_test_digits = train_test_split(X_digits, y_digits, test_size=0.2, random_state=42)
# Создание и обучение модели для цифр
model_digits = SVC(kernel='linear')
model_digits.fit(X_train_digits, y_train_digits)
# Предсказание для цифр
y_pred_digits = model_digits.predict(X_test_digits)
# Оценка точности для цифр
accuracy_digits = accuracy_score(y_test_digits, y_pred_digits)
print("Точность модели для цифр MNIST:", accuracy_digits)
