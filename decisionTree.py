import sys
import csv
import collections
import numpy as np
import pandas as pd
# print(sys.argv)
def load_tsv(input_file):
    data_file = open(input_file, "r")
    reader = csv.reader(data_file, delimiter='\t')
    headers = next(reader)

    feature = []
    label = []
    attr = []

    for row in reader:
        label.append(row[-1])
        row_dict = {}
        for i in range(0, len(row) - 1):
            row_dict[headers[i]] = row[i]
        feature.append(row_dict)
    data_file.close()

    for key in feature[0]:
        attr.append(key)

    return label, feature, attr

def entropy(label):
    '''
    计算结果列表的熵
    :param label: 标签列表
    :return: 熵
    '''
    first_res = 0
    second_res = 0
    for i in label:
        if i == label[0]:
            first_res += 1
        else:
            second_res += 1

    first_prob = float(first_res)/(first_res+second_res)
    second_prob = float(second_res)/(first_res+second_res)
    entropy = 0.0
    if first_prob != 0:
        entropy -= first_prob * np.log2(first_prob)
    if second_prob != 0:
        entropy -= second_prob * np.log2(second_prob)
    return entropy

def condition_entropy(attr_value_list, result_list):
    """
    计算条件熵
    :param attr_value_list: 属性的特征向量
    :param result_list: 结果列表
    :return: 条件熵
    """
    entropy_dict = collections.defaultdict(list)
    for attr_value, value in zip(attr_value_list, result_list):
        entropy_dict[attr_value].append(value)

    con_ent = 0.0
    attr_len = len(attr_value_list)
    for value in entropy_dict.values():
        p = len(value)/attr_len * entropy(value)
        con_ent += p
    return con_ent

def gain(attr_value_list, result_list):
    '''
    获取某一特征的信息增益
    :param attr_value_list: 属性的特征向量
    :param result_list:  结果列表
    :return: 信息增益
    '''
    ent = entropy(result_list)
    con_ent = condition_entropy(attr_value_list, result_list)
    return ent - con_ent

def data2vec(data_set):
    '''
    将data_set转换为0，1数据的形式
    :param data_set: 数据集
    :return: 转换后的数据集，以及标志列表
    '''
    data_vec = []
    judge = data_set[0]
    for vec in data_set:
        temp_vec = []
        for index in range(len(vec)):
           if vec[index] == judge[index]:
               temp_vec.append(1)
           else:
               temp_vec.append(0)
        data_vec.append(temp_vec)

    table = []
    for j in range(len(data_set[0])):
        table_dict = {}
        for i in range(len(data_set)):
            table_dict[data_vec[i][j]] = data_set[i][j]
            if len(table_dict) == 2:
                break
        table.append(table_dict)

    # print("data_vec:{}".format(data_vec))
    # print("table:{}".format(table))
    return data_vec, table

def data2vec_test(data_test_set, table):
    '''
    根据table表中的信息将data_test_set中的数据进行向量化，转换为0 1 数据
    :param data_test_set: 测试的数据集
    :param table: 转换根据表
    :return: 转换后的数据
    '''
    # print(data_test_set)
    # print(table)
    table_reverse = []
    for i in table:
        dictTmp_2 = dict([val, key] for key, val in i.items())
        table_reverse.append(dictTmp_2)

    # print(table_reverse)
    data_vec_test = []
    for data in data_test_set:
        data_vec = []
        for i in range(len(data)):
            data_vec.append(table_reverse[i][data[i]])
        data_vec_test.append(data_vec)
    # print(data_vec_test)
    return data_vec_test

class DecisionNode(object):
    def __init__(
            self, col=-1, data_set=None, labels=None,
            results=None, tb=None, fb=None, depth=None, max_depth=None, table=None, attr_lst=None, labelset=None):
        self.has_calc_index = []  # 已经计算过的特征索引
        self.col = col  # col 是待检验的判断条件，对应列索引值
        self.data_set = data_set  # 节点的 待检测数据
        self.labels = labels  # 对应当前列必须匹配的值
        self.results = results  # 保存的是针对当前分支的结果，有值则表示该点是叶子节点
        self.tb = tb  # 当信息增益最高的特征为True时的子树
        self.fb = fb  # 当信息增益最高的特征为False时的子树
        self.depth = depth  # 节点处于决策树的深度
        self.max_depth = max_depth  # 节点的最大深度
        self.table = table #0，1对应的属性的判断条件表
        self.attr_lst = attr_lst #判断属性内容列表
        self.label_set = labelset #标签值的集合

def end_condition(result_list, max_depth, current_depth):
    '''
    递归判定的结束条件：树的深度到达要求，或者分支的结果集是相同的分类
    :param result_list: 结果列表
    :param max_depth: 最大深度
    :param current_depth: 当前深度
    :return: true表示结束，false表示结束
    '''
    if current_depth > max_depth:
        return True
    else:
        result = collections.Counter()
        result.update(result_list)
        return len(result) == 1

def choose_best_feature(data_list, labels, ignore_index):
    """
    从特征向量中选取出最好的特征，返回其特征索引
    :param data_list: 数据列表
    :param labels: 结果标签
    :param ignore_index:  忽略数据的索引
    :return: 返回最好特征的特征索引
    """
    result_dict = {} # 索引为信息增益值
    feature_len = len(data_list[0])
    # print("data_set:{}".format(data_list))
    # print("label:{}".format(labels))
    # print("ignore_index:{}".format(ignore_index))

    for i in range(feature_len):
        if i in ignore_index: # 已经计算过的特征属性
            continue
        feature_list = [x[i] for x in data_list]
        result_dict[i] = gain(feature_list, labels) #获得信息增益
    # print("result_dict:{}".format(result_dict))
    ret = sorted(result_dict.items(), key=lambda x: x[1], reverse=True)
    return ret[0][0]

class DecisionTree():
    def __init__(self):
        self.feature_num = 0
        self.root = None

    def majorlabel(self, labels):
        '''
        采用majorvote的方式建立叶子节点
        :param label:
        :return: major结果
        '''
        result1 = labels[0]
        result2 = ""
        num1 = 0
        num2 = 0
        for label in labels:
            if label == result1:
                num1 += 1
            else:
                result2 = label
                num2 += 1
        if num2 > num1:
            return result2
        elif num2 < num1:
            return result1
        else:
            if result1 > result2:
                return result1
            else:
                return result2

    def count_label(self, labels, labelset):
        label_dict = {}
        for label in labelset:
            label_dict[label] = 0
        for label in labels:
            if label in label_dict.keys():
                label_dict[label] += 1
            else:
                label_dict[label] = 1

        print("[{0} {1}/{2} {3}]".format(label_dict[labelset[0]], labelset[0], label_dict[labelset[1]], labelset[1]))

    def build_tree(self, node: DecisionNode):
        '''
        递归建立决策树结构
        :param node: 树的节点
        :return:
        '''

        # print("----------------------------build_tree---------------------------")
        if end_condition(node.labels, node.max_depth, node.depth):
            node.results = self.majorlabel(node.labels)
            self.count_label(node.labels, node.label_set)
            # print("为叶子节点，结果为：{}".format(node.results))
            return
        best_index = choose_best_feature(node.data_set, node.labels, node.has_calc_index)
        node.col = best_index
        # print("best_index =", best_index)

        self.count_label(node.labels, node.label_set)

        #根据信息增益的最大进行划分 采用DFS的方法
        #生成左子树：
        for i in range(node.depth):
            print("|\t",end="")
        print("{} = {}: ".format(node.attr_lst[node.col], node.table[node.col][1]),end="")

        # print("左子树：")
        tb_index = [i for i, value in enumerate(node.data_set) if value[best_index]]
        # print("tb_index = {}".format(tb_index))
        tb_data_set = [node.data_set[x] for x in tb_index]
        # print("tb_data_set = {}".format(tb_data_set))
        tb_data_labels = [node.labels[x] for x in tb_index]
        # print("tb_data_labels = {}".format(tb_data_labels))
        tb_table = node.table
        tb_node = DecisionNode(data_set=tb_data_set, labels=tb_data_labels, table=tb_table, attr_lst=node.attr_lst, labelset=node.label_set)
        tb_node.has_calc_index = list(node.has_calc_index)
        tb_node.has_calc_index.append(best_index)
        # print("tb_node_has_calc_index = {}".format(tb_node.has_calc_index))
        tb_node.depth = node.depth + 1
        # print("tb_node.depth = {}".format(tb_node.depth))
        tb_node.max_depth = node.max_depth
        node.tb = tb_node

        #生成右子树：
        # print("右子树：")
        fb_index = [i for i, value in enumerate(node.data_set) if not value[best_index]]
        # print("fb_index = {}".format(fb_index))
        fb_data_set = [node.data_set[x] for x in fb_index]
        # print("fb_data_set = {}".format(fb_data_set))
        fb_data_labels = [node.labels[x] for x in fb_index]
        # print("fb_data_labels = {}".format(fb_data_labels))
        fb_table = node.table
        fb_node = DecisionNode(data_set=fb_data_set, labels=fb_data_labels, table=fb_table, attr_lst=node.attr_lst, labelset=node.label_set)
        fb_node.has_calc_index = list(node.has_calc_index)
        fb_node.has_calc_index.append(best_index)
        # print("fb_node_has_calc_index = {}".format(fb_node.has_calc_index))
        fb_node.depth = node.depth + 1
        # print("fb_node.depth = {}".format(fb_node.depth))
        fb_node.max_depth = node.max_depth
        node.fb = fb_node

        if tb_index:
            # print("----------------建立左子树------------------")
            self.build_tree(node.tb)
        else:
            self.count_label(node.tb.labels, node.tb.label_set)

        for i in range(node.depth):
            print("|\t",end="")
        print("{} = {}: ".format(node.attr_lst[node.col], node.table[node.col][0]),end="")

        if fb_index:
            # print("----------------建立右子树------------------")
            self.build_tree(node.fb)
        else:
            self.count_label(node.fb.labels, node.fb.label_set)

    def train(self, data_set, label_list, max_dpth, table, attr_list, label_set):
        self.feature_num = len(data_set[0])
        self.root = DecisionNode(data_set=data_set, labels=label_list, max_depth=max_dpth, depth=1, table=table, attr_lst=attr_list, labelset=label_set)
        self.build_tree(self.root)

    def _predict(self, data_test, node):
        '''
        对单个的数据进行预测
        :param data_test: 单个数据
        :param node: 节点
        :return: 预测单个的结果
        '''
        if node.results:
            return node.results
        col = node.col

        if data_test[col]:
            return self._predict(data_test, node.tb)
        else:
            return self._predict(data_test, node.fb)

    def predict(self, data_test_set):
        '''
        对整个数据集进行预测
        :param data_test_set: 所有数据
        :return: 预测结果
        '''
        test_ans = []
        for data_test in data_test_set:
            test_ans.append(self._predict(data_test, self.root))
        return test_ans

def calc_error(ans_list, judge_list):
    error = 0
    for i in range(len(ans_list)):
        if ans_list[i] != judge_list[i]:
            error += 1
    return (error*1.0)/len(ans_list)

def print_out_file(out_file, ans_list):
    out = open(out_file, "w", encoding="utf-8")
    for ans in ans_list:
        print(ans,file=out)
    out.close()

if __name__ == '__main__':
    # train_input_file = "politicians_train.tsv"
    # test_input_file = "politicians_test.tsv"
    # max_dpth = 3
    # train_out_file = "pol_3_train.labels"
    # test_out_file = "pol_3_test.labels"
    # matrics_out_file = "pol_3_metrics.txt"

    train_input_file = sys.argv[1]
    test_input_file = sys.argv[2]
    max_dpth = int(sys.argv[3])
    train_out_file = sys.argv[4]
    test_out_file = sys.argv[5]
    matrics_out_file = sys.argv[6]

# -------------------------训练------------------------------------
    label_list, feature_list, attr_list = load_tsv(train_input_file)
    label_set = list(set(label_list))

    if max_dpth > len(attr_list):
        max_dpth = len(attr_list) #max_dpth最多为属性的个数 不能超过

    data_set = []
    for i in feature_list:
        temp = []
        for j in attr_list:
            temp.append(i[j])
        data_set.append(temp)

    data_set, table = data2vec(data_set)
    # table:[{1: 'n', 0: 'y'}, {1: 'y', 0: 'n'}]

    # print("\n--------------------------------------------\n")
    tree = DecisionTree()
    tree.train(data_set, label_list, max_dpth, table, attr_list, label_set)

# --------------录入test文件中的数据----------------
    label_test_list, feature_test_list, attr_test_list = load_tsv(test_input_file)
    label_test_set = list(set(label_test_list))
    data_test_set = []
    for i in feature_test_list:
        temp = []
        for j in attr_test_list:
            temp.append(i[j])
        data_test_set.append(temp)

    data_test_set = data2vec_test(data_test_set, table)

# ----------------------训练集的结果输出------------------
    matrics_out = open(matrics_out_file, 'w', encoding="utf-8")
    predict_train = tree.predict(data_set)
    print_out_file(train_out_file, predict_train)
    # print(predict_train)
    # print(label_list)
    print("error(train): {}".format(calc_error(predict_train, label_list)),file=matrics_out)

# ----------------------测试集的结果输出--------------------------
    predict_test = tree.predict(data_test_set)
    print_out_file(test_out_file, predict_test)
    # print(predict_test)
    # print(label_test_list)
    print("error(test): {}".format(calc_error(predict_test, label_test_list)),file=matrics_out)
    matrics_out.close()