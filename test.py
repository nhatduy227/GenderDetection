import os

malePath = os.path.normpath(os.path.expanduser("~/Desktop/GenderDetection/male"))
arr = os.listdir(malePath)
print(arr)