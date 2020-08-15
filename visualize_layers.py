#visualizing the output from consecutive layers

#import the image classification model 
from img_classification import *


from google.colab import files
uploaded = files.upload()
new_image = plt.imread("1.jpg")

plt.imshow(new_image)

#resizing the input image
from skimage.transform import resize
resized_image = resize(new_image, (20,80,3))
plt.imshow(resized_image)

#making an input-output model using already created image classification model
from keras.models import Model
inputs = model.input
outputs = [model.layers[i].output for i in range(len(model.layers))]
final_model = Model(inputs, outputs)


#normalizing the pixels
resized_image = np.array([resized_image])
resized_image = resized_image/255
all_layers_predictions = final_model.predict(resized_image)


#plotting output image from all channels
for a in all_layers_predictions:
  for i in range(a[-1]):
    try:
      plt.matshow(a[0, :, :, i], cmap='viridis')
    except:
      continue
