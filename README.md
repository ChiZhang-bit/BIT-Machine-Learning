# BIT-Machine-Learning-Homework [Decision Tree]

> 北京理工大学机器学习选修课大作业 —— 决策树

## Description

在给定的四个数据集中，使用自己写的决策树程序，跑出相应的实验结果，并按照格式输出。



## Usage

使用命令行接受参数

该程序运行使用命令行进行调用，decisionTree.py程序应该接受命令行参数，命令行参数如下所示：

| 命令行参数  | 参数的含义                                                   |
| ----------- | ------------------------------------------------------------ |
| train_input | 输入相应训练数据的路径，例如：small_train.tsv                |
| test_input  | 输入相应测试数据的路径，例如：small_test.tsv                 |
| max_depth   | 生成决策树的最大深度，例如：2                                |
| train_out   | 将决策树对训练数据的预测写入标签文件的路径，例如：small_2_train.labels |
| test_out    | 将决策树对测试数据的预测写入标签文件的路径，例如：small_2_test.labels |
| metrics_out | 将训练集和测试集分类的错误指标写入文件的路径，例如：small_2_metrics.txt |

​    示例：`python decisionTree.py politicians_train.tsv politicians_test.tsv 2 pol_2_train.labels pol_2_test.labels pol_2_metrics.txt`

### 课程评价

郭宇航老师授课非常细致，讲课也讲得非常好，对于机器学习这门课的讲授通俗易懂。这门课的各项作业也非常简单，推荐学弟学妹们来学习这门课。