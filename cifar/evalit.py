import keras
from keras.models import load_model
from PIL import Image
import matplotlib.pylab as plt
import numpy as np
import zipfile
print("Extract")
zip_ref = zipfile.ZipFile("./asset.zip", 'r')
zip_ref.extractall(".")
zip_ref.close()
print("Load Model")
model=load_model("cifar-model.h5")
CIFAR_10_CLASSES=["Plane","Car","bird","cat","deer","dog","frog","horse","ship","truck"]
def calc(imname):
    test_image =Image.open("asset/"+imname)
    test_image=test_image.resize((32,32),Image.ANTIALIAS)
    test_image=np.array(test_image,dtype="float32")
    test_image/=255
    test_image=test_image.reshape(-1,32,32,3)
    predictions=model.predict(test_image)
    index_max_pred=np.argmax(predictions)
    plt.title("Complete: {}".format(CIFAR_10_CLASSES[index_max_pred]))
    plt.imshow(test_image[0].reshape(32,32,3))
    print(predictions)
    plt.show()
print("START TEST")
calc("lkw-image.jpg")
calc("cat.jpg")
calc("frog.jpg")
calc("fog.jpg")
calc("lfog.jpg")
calc("d.jpg")
calc("b.jpg")
calc("bs.jpg")
calc("plapper.jpg")
calc("ds.jpg")
print("Complete")
print("End")
quit(0)
