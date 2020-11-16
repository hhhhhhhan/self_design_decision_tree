import numpy as np
import time

class Tree:

    def __init__(self, left_branch=None, right_branch=None, splitting_col=-1, splitting_value=None, classes=None):
        '''
        :param left_branch: left tree
        :param right_branch: right tree
        :param col: the attribute with best splitting evaluation
        :param value: the value point of attribute with best splitting evaluation
        :param labels: the number of each labels in the current sample
        '''

        self.left_branch = left_branch
        self.right_branch = right_branch
        self.splitting_col = splitting_col
        self.splitting_value = splitting_value
        self.classes = classes


class DecisionTree:

    def split_data(self, data, col, value):
        '''
            try to split the data into two parts at this (value) of (col)
        :param data: samples to be split
        :param col: attribute to be split
        :param value: value of attribute to be split
        :return: two parts of samples
        '''
        left = []
        right = []
        for i in data:
            # check the type of value, if it is numeric of nominal
            try:
                i[col].astype(float)
                if i[col] < value:
                    left.append(i)
                else:
                    right.append(i)
            except:
                if i[col] == value:
                    left.append(i)
                else:
                    right.append(i)
        return np.asarray(left), np.asarray(right)

    def cal_no_of_each_class(self, data):
        '''
            calculating the number of each label in the samples
        :param data: samples
        :return: a dictionary that states the numbe of labels
        '''
        classes_count = {}
        # select the last column of data which are labels
        for c in data[:, -1]:
            if c not in classes_count.keys():
                classes_count[c] = 1
            else:
                classes_count[c] += 1

        return classes_count

    def gini(self, data):
        '''
            calculating gini (not gini-index)
        :param data: samples
        :return: gini
        '''
        no_of_sample = len(data)
        if no_of_sample != 0:
            classes = self.cal_no_of_each_class(data)
            res = 1
            for c in classes.keys():
                res -= (classes[c] / no_of_sample) ** 2
            return res
        else:
            return 0

    def get_best_col_value_set(self, data):
        '''
            get the splitting column and value with best evaluation
        :param data: data set
        :return: best evaluation, best splitting column, best splitting value, tuple of left and right splitted data set
        '''
        best_gini_index = 1.0
        best_col = None
        best_value = None
        best_set = None

        # iterate every attribute and every distinct value of the attribute
        for col in range(data.shape[1] - 1):
            value_set = set(data[:, col])
            for value in value_set:
                # try to split the data at current value of the column to calculate the gini-index
                left_data, right_data = self.split_data(data, col, value)
                gini_index = len(left_data) / len(data) * self.gini(left_data) + len(right_data) / len(
                    data) * self.gini(right_data)
                # update the best gini-index and other variables
                if gini_index < best_gini_index:
                    best_gini_index = gini_index
                    best_col = col
                    best_value = value
                    best_set = (left_data, right_data)

        return best_gini_index, best_col, best_value, best_set


    def predict_row(self, row, tree):
        '''
            predict the class label of a single row of data
        :param row: a row of data
        :param tree: fitted tree model
        :return: a class label
        '''
        # find the suitable branch for data until reach the leaf node
        if tree.classes == None:
            # check the type of value, if it is numeric of nominal
            try:
                row[tree.splitting_col].astype(float)
                if row[tree.splitting_col] < tree.splitting_value:
                    return self.predict_row(row, tree.left_branch)
                else:
                    return self.predict_row(row, tree.right_branch)
            except:
                if row[tree.splitting_col] == tree.splitting_value:
                    return self.predict_row(row, tree.left_branch)
                else:
                    return self.predict_row(row, tree.right_branch)
        else:
            return tree.classes

    def predict(self, data, tree):
        '''
            predict the label of data by a fitted decision tree
        :param data: sample to be predicted
        :param tree: fitted tree model
        :return: an array of labels after prediction
        '''
        res = []
        for row in data:
            res.append(self.predict_row(row, tree))
        return res

    def pre_pruning(self, valid_data, best_col, best_value, accu, left_train_data, right_train_data):
        '''
            test the validation set to justify if the splitting need to be pruned
        :param valid_data: validation data set
        :param best_col: attribute to be split
        :param best_value: value of attribute to be split
        :param accu: accuracy of parent validation data set
        :param left_train_data: left train data set splitted at (best_value) of (best_col)
        :param right_train_data: right train data set splitted at (best_value) of (best_col)
        :return: boolean
        '''
        left_valid_data, right_valid_data = self.split_data(valid_data, best_col, best_value)
        # state the classes of the train data
        left_class = self.cal_no_of_each_class(left_train_data)
        right_class = self.cal_no_of_each_class(right_train_data)

        if len(left_valid_data) == 0 or len(right_valid_data) == 0:
            return False, None, None, None, None

        new_accu = (len([i == left_class for i in left_valid_data[:, -1]]) + len([i == right_class for i in right_valid_data[:, -1]])) / (
                len(left_valid_data) + len(right_valid_data))

        left_accu = len([i == left_class for i in left_valid_data[:, -1]]) / len(left_valid_data)
        right_accu = len([i == left_class for i in right_valid_data[:, -1]]) / len(right_valid_data)

        if accu:
            return new_accu > accu, left_valid_data, right_valid_data, left_accu, right_accu
        else:
            return True, left_valid_data, right_valid_data, left_accu, right_accu


    def build_tree(self, data, max_iter):
        '''
            build and train a tree by the train data
        :param data: samples/training set
        :param max_iter: max depth of tree
        :return: a fitted tree/sub-tree
        '''
        max_iter -= 1
        best_gini_index, best_col, best_value, (left_data, right_data) = self.get_best_col_value_set(data)

        # recurrent until the gini-index equals 0 or reach to max iterations
        if best_gini_index > 0 and max_iter > 0:
            left_branch = self.build_tree(left_data, max_iter=max_iter)
            right_branch = self.build_tree(right_data, max_iter=max_iter)
            # record the best col and value at every non-leaf node
            return Tree(left_branch=left_branch, right_branch=right_branch, splitting_col=best_col,
                        splitting_value=best_value)
        else:
            # calculate the number of each class in the samples
            classes = self.cal_no_of_each_class(data)
            max_count = max(zip(classes.values(), classes.keys()))
            return Tree(classes=max_count[1])

    def build_tree_preprune(self, data, max_iter, valid_data=None, accu=None):
        '''
            build and train a tree with pre-prune
        :param data: samples/training set
        :param max_iter: max depth of tree
        :param valid_data: a validation data set
        :param accu: accuracy of parent validation data set
        :return: a fitted tree/sub-tree
        '''
        max_iter -= 1
        # get the best splitting columne and value
        best_gini_index, best_col, best_value, (left_data, right_data) = self.get_best_col_value_set(data)

        if valid_data is not None:
            if_prune, left_valid_data, right_valid_data, l_accu, r_accu = self.pre_pruning(valid_data, best_col, best_value, accu, left_data, right_data)
        else:
            print('there is no validation data input')

        # recurrent until the gini-index equals 0 or reach to max iterations
        if best_gini_index > 0 and max_iter > 0 and if_prune:
            left_branch = self.build_tree_preprune(left_data, max_iter=max_iter, valid_data = left_valid_data, accu=l_accu)
            right_branch = self.build_tree_preprune(right_data, max_iter=max_iter, valid_data = right_valid_data, accu=r_accu)
            # record the best col and value at every non-leaf node
            return Tree(left_branch=left_branch, right_branch=right_branch, splitting_col=best_col,
                        splitting_value=best_value)
        else:
            # record number of labels of remaining data when gini-index is 0
            classes = self.cal_no_of_each_class(data)
            max_count = max(zip(classes.values(), classes.keys()))
            return Tree(classes=max_count[1])

    def fit(self, data, max_iter=100, pre_prune=False):
        '''
            train a tree
        :param data: train data
        :param max_iter: max depth of tree
        :param pre_prune: boolean value
        :return: a fitted tree
        '''
        if not pre_prune:
            return self.build_tree(data, max_iter)
        else:
            split_point = int(np.ceil(len(data) * 0.8))
            train_data = data[:split_point]
            valid_data = data[split_point:]
            return self.build_tree_preprune(train_data, max_iter, valid_data)