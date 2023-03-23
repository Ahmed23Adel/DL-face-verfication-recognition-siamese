import gen_embs as gen_embs
from keras_facenet import FaceNet
import numpy as np
from PIL import Image
from mtcnn.mtcnn import MTCNN
from database_simulator import DatabaseSimulator
from tensorflow.keras import metrics

class recognizer():
    def __init__(self):
        self.embedder = FaceNet()
        self.facenet_model=self.embedder.model
        self.detector = MTCNN()
        self.required_size = (160, 160)
        self.db = DatabaseSimulator()

    def extract_face_from_X(self, img_x):
        """Read the face from image and ignore background

        Args:
            img_x (ndaray): input image 

        Returns:
            ndarray: array of the face only
        """
        # detect faces in the image
        results = self.detector .detect_faces(img_x)
        # extract the bounding box from the first face
        x1, y1, width, height = results[0]['box']
        # deal with negative pixel index
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        # extract the face
        face = img_x[y1:y2, x1:x2]
        # resize pixels to the model size
        image = Image.fromarray(face)
        image = image.resize(self.required_size)
        face_array = np.asarray(image)
        return face_array

    def extract_face_from_name(self, filename):
        """Get face by using MTCNN after reading the image from filepath

        Args:
            filename (str): filepath of the iamge

        Returns:
            ndarry: array having the face only
        """
        # load image from file
        image = Image.open(filename)
        # convert to RGB, if needed
        image = image.convert('RGB')
        # convert to array
        pixels = np.asarray(image)
        return self.extract_face_from_X(pixels)
    
    def get_embedding(self, face):
        """Get embedding of the face

        Args:
            face (ndarray): array, expected that it has been passed through MTCNN

        Returns:
            ndarray: embidding of the face
        """
        # scale pixel values
        face = face.astype('float32')
        # standardization
        mean, std = face.mean(), face.std()
        face = (face-mean)/std
        # transfer face into one sample (3 dimension to 4 dimension)
        sample = np.expand_dims(face, axis=0)
        # make prediction to get embedding
        yhat = self.facenet_model.predict(sample)
        return yhat[0]

    def get_similarity(self, emb1, emb2):
        """Calculate similarity between two embidding using coside

        Args:
            emb1 (ndarray): embidding of image 1
            emb2 (ndarray): embidding of image 2

        Returns:
            int: similary of two embiddings
        """
        cosine_similarity = metrics.CosineSimilarity()
        similarity = cosine_similarity(emb1, emb2).numpy()
        return similarity

    def recognize(self, img_name):
        """Recognizes the image from given simulated database

        Args:
            img_name (str): filepath of image to be recognized
        """
        face = self.extract_face_from_name(img_name)
        emb = self.get_embedding(face)
        max_similarity = 0
        max_similarity_name = ""
        for person_name, person_emb in self.db.users_emb.items():
            sim  = self.get_similarity(emb, person_emb)
            if sim > max_similarity:
                max_similarity = sim
                max_similarity_name = person_name
        print("Most probably it's ", max_similarity_name)

rec = recognizer()
rec.recognize(r"imgs\john2.jpg")
