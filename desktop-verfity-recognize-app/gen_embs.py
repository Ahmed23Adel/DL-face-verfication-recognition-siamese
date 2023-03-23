import os
import cv2
from keras_facenet import FaceNet
import numpy as np
from PIL import Image
from mtcnn.mtcnn import MTCNN
import json
import pickle


def get_embedding(model, face):
    # scale pixel values
    face = face.astype('float32')
    # standardization
    mean, std = face.mean(), face.std()
    face = (face-mean)/std
    # transfer face into one sample (3 dimension to 4 dimension)
    sample = np.expand_dims(face, axis=0)
    # make prediction to get embedding
    yhat = model.predict(sample)
    return yhat[0]

def extract_face(filename, required_size=(160, 160)):
    # load image from file
    image = Image.open(filename)
    # convert to RGB, if needed
    image = image.convert('RGB')
    # convert to array
    pixels = np.asarray(image)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    # extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']
    # deal with negative pixel index
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = np.asarray(image)
    return face_array



def gen_embds(imgs_path =  r"imgs/single") : 
    embedder = FaceNet()
    facenet_model=embedder.model
    embds = {}
    for img_name in os.listdir(imgs_path):
        full_path = os.path.join(imgs_path, img_name)
        # read the image
        face = extract_face(full_path)
        embidding = get_embedding(facenet_model, face)
        # print("Finished", img_name)        
        embds[img_name.split(".")[0]] = embidding
    with open('embds.p', 'wb') as fp:
        pickle.dump(embds, fp, protocol=pickle.HIGHEST_PROTOCOL)
    return embds


def get_all_embds(read_from_file = False):
    if read_from_file:
        with open('embds.p', 'rb') as fp:
            data = pickle.load(fp)
        return data
    else:
        return gen_embds()

# print(get_all_embds())