import csv
import random

class_num = [0 for i in range(17)]
print(class_num)


with open('labels' + '/' + 'train.csv', newline='') as labels_file:
    reader = csv.reader(labels_file)

    for row in reader:  # Get only first line
        for i, item in enumerate(row):
            if item == '1':
                class_num[i-1] += 1

print(class_num)

class_min = list.copy(class_num)
for i, item in enumerate(class_num):
    if item - (sum(class_num) / len(class_num) / 3) < 0:
        class_min[i] = 1.0
    else:
        class_min[i] = 0.0

print(class_min)

class_list = [random.choice(class_min) for i in class_min]

print(class_list)

equal_labels = 0
for i in range(len(class_min)):
    if class_min[i] == class_list[i] == 1.0:
        equal_labels += 1

print(equal_labels)
