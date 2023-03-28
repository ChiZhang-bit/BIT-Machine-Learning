import sys
import csv
import numpy as np

# print(sys.argv)
def load_tsv(inputfile):
    data_file = open(input_file, "r")
    reader = csv.reader(data_file, delimiter='\t')
    headers = next(reader)
    label = []

    for row in reader:
        label.append(row[-1])

    data_file.close()
    return label

def calc_Entropy_Error(label):
    first_res = 0
    second_res = 0
    for i in label:
        if i == label[0]:
            first_res += 1
        else:
            second_res += 1

    if first_res < second_res:
        error = float(first_res)/(first_res + second_res)
    else:
        error = float(second_res)/(first_res + second_res)

    first_prob = float(first_res)/(first_res+second_res)
    second_prob = float(second_res)/(first_res+second_res)
    entropy = 0.0
    entropy -= first_prob * np.log2(first_prob)
    entropy -= second_prob * np.log2(second_prob)
    return entropy, error

if __name__ == '__main__':
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    # input_file = "small_train.tsv"
    # output_file = "small_inspect.txt"
    label_list = load_tsv(input_file)
    print(label_list)

    with open(output_file, 'w', encoding="utf-8") as f:
        print("entropy: {}".format(calc_Entropy_Error(label_list)[0]), file = f)
        print("error: {}".format(calc_Entropy_Error(label_list)[1]), file = f)