import pickle
from PIL import Image
import pandas as pd
import numpy as np
y_pred = pickle.load(open("save.p", "rb"))

width = 28
height = 28
new_image = Image.new('L', (width, height))
data = new_image.load()
test_features = pd.read_csv('test.csv')

x1 = input()
#To print one data item as an image
img = list(test_features.iloc[x1])
img_ar = np.array(img[:])
img_ar = img_ar.reshape(28,28)
img = img_ar.tolist()

for y in range(height):
    for x in range(width):
        data[(x, y)] = img[y][x]

new_image.save('1.png', 'png')

print(y_pred[x1])
img = Image.open('1.png')
img = img.resize((400, 400), Image.NEAREST)
img.show()
