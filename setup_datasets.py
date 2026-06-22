import csv
import random
import traceback


def set_min_max(max_arr, min_arr, row):
    for idx, elem in enumerate(row):
        if elem > max_arr[idx]:
            max_arr[idx] = elem
        if elem < min_arr[idx]:
            min_arr[idx] = elem


def normalize_value(max, min, val):
    if max - min == 0:
        return 0.0
    return (val - min) / (max - min)


def normalize_arr(max_arr, min_arr, data_arr):
    for row in data_arr:
        for i in range(1, len(row)):
            min = min_arr[i - 1]
            max = max_arr[i - 1]
            row[i] = normalize_value(max, min, row[i])


def main():

    data = []
    try:
        with open('data.csv', 'r') as file:
            data_reader = csv.reader(file)
            for row in data_reader:
                if not row:
                    continue
                tmp = row[1:]
                tmp[0] = 1 if tmp[0] == "M" else 0
                tmp[1:] = [float(x) for x in tmp[1:]]
                data.append(tmp)

            random.seed(42)
            random.shuffle(data)

            split_idx = int(len(data) * 0.8)

            train_set = data[:split_idx]
            val_set = data[split_idx:]

            max_arr = train_set[0][1:]
            min_arr = train_set[0][1:]
            for row in train_set:
                set_min_max(max_arr, min_arr, row[1:])
            
            normalize_arr(max_arr, min_arr, train_set)
            normalize_arr(max_arr, min_arr, val_set)

            with open("train_set.csv", mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerows(train_set)
            with open("val_set.csv", mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerows(val_set)
            with open("scaler.csv", mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["max"] + max_arr)
                writer.writerow(["min"] + min_arr)


    except TypeError as e:
        print(f"TypeError: {str(e)}")
        traceback.print_exc()
    except Exception as e:
        print(f"An exception has been caught: {type(e).__name__} - {str(e)}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
