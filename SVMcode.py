
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from sklearn import svm, metrics, datasets
from sklearn.utils import Bunch
from sklearn.model_selection import GridSearchCV, train_test_split

from skimage.io import imread
from skimage.transform import resize
from skimage import feature
from skimage import io
from sklearn import preprocessing
import cv2

def image_to_feature_vector(image, size=(128, 128)):
		# resize the image to a fixed size, then flatten the image into
		# a list of raw pixel intensities
		return cv2.resize(image, size).flatten()

def load_image_files(container_path, dimension=(128, 128)):
    """
    Load image files with categories as subfolder names 
    which performs like scikit-learn sample dataset
    
    Parameters
    ----------
    container_path : string or unicode
        Path to the main folder holding one subfolder per category
    dimension : tuple
        size to which image are adjusted to
        
    Returns
    
    -------
    Bunch
    """
    image_dir = Path(container_path)
    folders = [directory for directory in image_dir.iterdir() if directory.is_dir()]
    categories = [fo.name for fo in folders]

    descr = "A image classification dataset"
    images = []
    flat_data = []
    target = []
    for i, direc in enumerate(folders):
        for file in direc.iterdir():
            img = io.imread(file)
            input_img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_resized =image_to_feature_vector(input_img,(128,128))
            flat_data.append(img_resized.flatten()) 
            images.append(img_resized)
            target.append(i)
            
    flat_data = np.array(flat_data)
    target = np.array(target)
    images = np.array(images)

    return Bunch(data=flat_data,
                 target=target,
                 target_names=categories,
                 images=images,
                 DESCR=descr)
                 
image_dataset = load_image_files("C:/Users/Anne/Desktop/SVM2/")

X_train, X_test, y_train, y_test = train_test_split(
    image_dataset.data, image_dataset.target, test_size=0.3,random_state=109)

svc = svm.SVC()
#clf = GridSearchCV(svc, param_grid)



svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)
print(y_pred)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
sc=svc.score(X_test,y_test)
print(sc)


'''
print("Classification report for - \n{}:\n{}\n".format(
    svc, metrics.classification_report(y_test, y_pred)))
'''
