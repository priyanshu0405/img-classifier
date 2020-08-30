#code to add gaussian noise to an image and testing it on the model

#import the image classification model 
from img_classification import *

#importing image from system
from google.colab import files
uploaded = files.upload()
new_image = plt.imread("1.jpg")

new_image = np.array(new_image)
new_image = new_image/255

#adding noise to the image 
import skimage
plt.figure(figsize=(18,24))
r=4
c=2
plt.subplot(r,c,1)

#other type of noises can also be added by changing mode to "localvar", "poisson", "salt", "pepper", "s&p", "speckle"

new_image = skimage.util.random_noise(new_image, mode="gaussian") 
plt.imshow(new_image)

#resizing image to test in model
from skimage.transform import resize
resized_image = resize(new_image, (60,100,3))
img = plt.imshow(resized_image)

#prints the class name with maximum prediction value
predictions = model.predict(np.array( [resized_image] ))
category[np.argmax(predictions)]



