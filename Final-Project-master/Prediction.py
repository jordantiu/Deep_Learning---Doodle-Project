from keras.models import load_model
import numpy as np
from PIL import Image
import PIL.ImageOps

categories = ["Cat", "Sun", "Fish"]

image = Image.open("cat.bmp").convert('L')
image.show()
image = image.resize((28, 28), resample=Image.LANCZOS)
image = PIL.ImageOps.invert(image)
image = np.asarray(image)/255
image = image.reshape(1,28, 28, 1).astype('float32')


model = load_model(('model.h5'))
prediction = model.predict(image)
pred_name = categories[np.argmax(prediction)]
print(prediction)
print(pred_name)
