import random




def qsort(list, start, end):
    if start >= end:
        return
    left = start
    right = end
    mid = list[left]
    while left < right:
        while left < right and list[right] >= mid:
            right = right - 1
        list[left] = list[right]
        while left < right and list[left] < mid:
            left = left + 1
        list[right] = list[left]
    qsort(list, start, left - 1)
    qsort(list, left + 1, end)


def msort(list):
    for j in range(0, len(list) - 1):
        for i in range(j, len(list) - 1):
            if list[i] > list[i + 1]:
                temp = list[i]
                list[i] = list[i + 1]
                list[i + 1] = temp


if __name__ == '__main__':
    a = [random.randint(20, 100) for i in range(0, 1000)]
    print(a)
    qsort(a, 0, len(a) - 1)
    msort(a)
    print(a)
    dict = {}
    for num in a:
        dict[num] = dict.get(num, 0) + 1
    print(dict)

