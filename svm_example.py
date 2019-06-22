"""
-- User: Ashok Kumar Pant (ashokpant@treeleaf.ai)
-- Treeleaf Technologies Pvt. Ltd.
-- Date: 5/20/19
-- Time: 1:41 PM
"""
from sklearn import svm
from sklearn.metrics import classification_report

if __name__ == '__main__':
    features = [[0, 0], [1, 1], [2, 2]]
    labels = [0, 1, 2]
    classifier = svm.SVC(gamma='scale')
    classifier.fit(features, labels)
    outputs = classifier.predict(features)

    report = classification_report(labels, outputs)
    print(report)

    out = classifier.predict([[0.5, 0.5]])
    print(out)
