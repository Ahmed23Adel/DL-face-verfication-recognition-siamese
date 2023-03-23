import gen_embs as gen_embs
from keras_facenet import FaceNet
import numpy as np
from PIL import Image
from mtcnn.mtcnn import MTCNN
from database_simulator import DatabaseSimulator
from tensorflow.keras import metrics


class Verifier():
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


    def verify(self, img_name = None, img_array= None, target_name = None):
        """Verify that input image is same as embidding saved in simulated DB.
        img_name is the filepath of the image
        img_array is the array of images that has been read
        target_name is the name of the person
        if img_name is None:
            then img_array must be not None
        """
        assert not (img_name is None and img_array is None)
        print("img_name", img_name)
        img_emb = None
        print("Getting embidding for the input image...")
        if img_name is None:
            # read the image
            face = self.extract_face_from_X(img_array)
            img_emb = self.get_embedding(face)
        else:
            print("img_name", img_name)
            img_array = Image.open(img_name)
            # convert to RGB, if needed
            img_array = img_array.convert('RGB')
            img_array = np.asarray(img_array)
            face = self.extract_face_from_X(img_array)
            img_emb = self.get_embedding(face)
        targer_emb = self.db.get_emb(target_name)
        cosine_similarity = metrics.CosineSimilarity()
        similarity = cosine_similarity(img_emb, targer_emb).numpy()
        print("similarity: ", similarity)
        if cosine_similarity > 0.5:
            print("Then they are similar")
        else:
            print("Then they are not similar")


    def get_embidding_from_name(self, img_name):
        """Read image from img_name, and return its embidding after MTCNN, FaceNet

        Args:
            img_name (str): filepath to the image

        Returns:
            str: embidding of the image
        """
        img_array = Image.open(img_name)
        img_array = img_array.convert('RGB')
        img_array = np.asarray(img_array)
        face = self.extract_face_from_X(img_array)
        img_emb = self.get_embedding(face)
        return img_emb

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
        
    def verify_imgs(self, img1_name, img2_name):
        """Verify that two images are the same

        Args:
            img1_name (str): filepath of first image
            img2_name (str): filepath of second image

        Returns:
            bool: True if the images are the same, False otherwise
        """
        img1_emb = self.get_embidding_from_name(img1_name)
        img2_emb = self.get_embidding_from_name(img2_name)
        similarity = self.get_similarity(img1_emb, img2_emb)
        if similarity > 0.5:
            print("Same person")
            return True
        else:
            print("Then they are not similar")
            return False
        


verifier = Verifier()
# verifier.verify(img_name = r"imgs\ben2.jpg", target_name="ben")
verifier.verify_imgs( r"imgs\ben1.jpg",  r"imgs\ben2.jpg")
verifier.verify_imgs( r"imgs\ben1.jpg",  r"imgs\jerry2.jpg")


