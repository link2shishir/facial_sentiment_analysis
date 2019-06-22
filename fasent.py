
import dlib
import numpy as np
from sklearn import svm
from sklearn.metrics import classification_report

from dis_calculate import extract_features, extract_features_from_img
from utils import create_dataset

if __name__ == '__main__':
    data_dir = "C:/Users\shish/project/COHN_KANADE"
    train_images, train_labels, test_images, test_labels, id2label = \
        create_dataset(data_dir, 0.2, hot_labels=False, max_classes=3)
    print("Train")
    print(train_images, train_images.shape)
    print(train_labels, train_labels.shape)
    print("Test")
    print(test_images, test_images.shape)
    print(test_labels, test_labels.shape)
    print("Id2label")
    print(id2label)

    train_features = []
    train_y = []

    test_features = []
    test_y = []

    detector = dlib.get_frontal_face_detector()  # Face detector
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Landmark identifier.

    # num_cores = multiprocessing.cpu_count()
    # train_features = Parallel(n_jobs=num_cores)(delayed(extract_features)(detector, predictor, image) for image in train_images)
    # print(train_features)

    for i, image in enumerate(train_images):
        if i % 10 == 0:
            print("Completed " + str(i) + " train samples")
        try:
            feature = extract_features(detector, predictor, image)
            if feature is not None:
                train_features.append(feature)
                train_y.append(train_labels[i])
        except Exception as e:
            print(e)

    for i, image in enumerate(test_images):
        if i % 10 == 0:
            print("Completed " + str(i) + "test samples")
        try:
            feature = extract_features(detector, predictor, image)
            if feature is not None:
                test_features.append(feature)
                test_y.append(test_labels[i])
        except Exception as e:
            print(e)

    train_features = np.array(train_features)
    test_features = np.array(test_features)
    train_y = np.array(train_y)
    test_y = np.array(test_y)

    print("Train features")
    print(train_features, train_features.shape)
    print(train_y, train_y.shape)
    print("Test features")
    print(test_features, test_features.shape)
    print(test_y, test_y.shape)

    classifier = svm.SVC(gamma='scale')
    classifier.fit(train_features, train_y)

    outputs = classifier.predict(test_features)
    report = classification_report(test_y, outputs)
    print(report)

    feature = extract_features(detector, predictor,
                               "C:/Users\shish/project/COHN_KANADE/NEUTRAL/MK.NE2.114.jpg")
    out = classifier.predict([feature])
    print(out, id2label[out[0]])

    import cv2

    video = cv2.VideoCapture(0)

    while True:
        succ, img = video.read()
        feature = extract_features_from_img(detector, predictor, img)
        out = classifier.predict([feature])
        print(out, id2label[out[0]])
