from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

iris_dataset = load_iris()

# print("Keys: \n{}".format(iris_dataset.keys()))
# print(iris_dataset["DESCR"])
# print("Type of data: {}".format(type(iris_dataset['data'])))
# print("Shape of data: {}".format(iris_dataset['data'].shape))
# print("Some rows: {}\n".format(iris_dataset['data'][:5]))
print("Target:\n{}\n".format(iris_dataset['target']))

training_data, testing_data, training_labels, testing_labels = train_test_split(iris_dataset['data'], iris_dataset['target'])
# print(training_data)
# print(testing_data)

# print(len(training_data))
# print(testing_data.shape)

knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(training_data, training_labels)

mystery_iris = np.array([[5, 2.9, 1, 0.6]])
print(mystery_iris.shape)

prediction = knn.predict(mystery_iris)
print(iris_dataset['target_names'][prediction])

test_predictions = knn.predict(testing_data)
print(test_predictions)

print("Score: {:.2f}".format(np.mean(test_predictions == testing_labels)))

#fit, predict, score
