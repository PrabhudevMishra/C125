import numpy as np
import pandas as pd
import PIL.ImageOps
from PIL import Image
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

x,y = fetch_openml('mnist_784', version=1, return_X_y=True)
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=4,train_size=7500,test_size=2500)
#to scale the data
x_train_scale = x_train/255.0
x_test_scale = x_test/255.0
# to fit the data into the model to get maximum accuracy
model = LogisticRegression(solver = 'saga', multi_class='multinomial').fit(x_train_scale,y_train)
def get_prediction(image):
    image_PIL = Image.open(image)
    image_bw = image_PIL.convert('L')
    image_resize = image_bw.resize((28,28),Image.ANTIALIAS)
    pixel_filter = 20
    min_pixel = np.percentile(image_resize, pixel_filter)
    img_scale = np.clip(image_resize-min_pixel,0,255)
    max_pixel = np.max(image_resize)
    img_scale = np.asarray(img_scale)/max_pixel
    sample = np.asarray(img_scale).reshape(1,784)
    prediction = model.predict(sample)
    return prediction[0]
