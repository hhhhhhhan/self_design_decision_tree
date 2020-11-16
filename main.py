if __name__ == '__main__':
    import numpy as np
    from sklearn.datasets import load_iris, load_breast_cancer
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from cart_decision_tree import *

    # load iris data from sklearn and retrieve data and labels
    iris = load_iris()
    iris_data = iris.data
    iris_target = iris.target.reshape((iris.target.shape[0], 1))

    # load breast cancer data from sklearn and retrieve data and labels
    bc = load_breast_cancer()
    bc_data = bc.data
    bc_target = bc.target.reshape((bc.target.shape[0], 1))


    iris_train_x, iris_test_x, iris_train_y, iris_test_y = train_test_split(iris_data, iris_target, test_size=0.2, random_state = 21)
    cart_tree = DecisionTree()
    # stack the labels on the data at last column
    iris_train_set = np.hstack((iris_train_x, iris_train_y))
    dt_classifier = cart_tree.fit(iris_train_set, max_iter=100)
    iris_res = np.asarray(cart_tree.predict(iris_test_x, dt_classifier))

    print(accuracy_score(iris_res, iris_test_y))

    bc_train_x, bc_test_x, bc_train_y, bc_test_y = train_test_split(bc_data, bc_target, test_size=0.1, random_state = 0)
    cart_tree = DecisionTree()
    # stack the labels on the data at last column
    bc_train_set = np.hstack((bc_train_x, bc_train_y))
    dt_classifier = cart_tree.fit(bc_train_set, max_iter=100, pre_prune=True)
    bc_res = np.asarray(cart_tree.predict(bc_test_x, dt_classifier))

    print(accuracy_score(bc_res, bc_test_y))