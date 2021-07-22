'''
usage: python gen_diff.py -h
'''

from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.layers import Input
from keras_preprocessing import image
from keras import backend as K
import numpy as np


from Util import *

# read the parameter
# argument parsing
#parser = argparse.ArgumentParser( description='Main function for difference-inducing input generation in ImageNet dataset')
#parser.add_argument('seeds', help="number of seeds of input", type=int)

#args = parser.parse_args()


# input image dimensions
img_rows, img_cols = 224, 224
input_shape = (img_rows, img_cols, 3)

# define input tensor as a placeholder
input_tensor = Input(shape=input_shape)

# load multiple models sharing same input tensor
K.set_learning_phase(0)
model1 = VGG16(input_tensor=input_tensor)
model2 = VGG19(input_tensor=input_tensor)

# start gen inputs
img_paths = image.list_pictures('./seeds/', ext='jpeg')

for _ in range(100):
    gen_img = preprocess_image(random.choice(img_paths))
    orig_img = gen_img.copy()
    # first check if input already induces differences
    pred1, pred2 = model1.predict(gen_img), model2.predict(gen_img)
    label1, label2 = np.argmax(pred1[0]), np.argmax(pred2[0])

    #This one here where we check if there is a difference
    if not label1 == label2:
        print("Difference found")
