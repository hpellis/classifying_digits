import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics

# load the digits dataset
digits = datasets.load_digits()

# look at the first 4 images in the dataset, which are stored in the 'images' attribute of the dataset
# these images are labelled with the digit they represent
images_and_labels = list(zip(digits.images, digits.target))
for index, (image, label) in enumerate(images_and_labels[:4]):
    plt.subplot(2, 4, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Training: %i' % label)

# in order to use a classifer on this data, need to flatten the images
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# create a classifier, which is a support vector classifier
classifier = svm.SVC(gamma=0.001)

# fit this classifier to the target attribute (the digit) on the first half of the dataset
classifier.fit(data[:n_samples // 2], digits.target[:n_samples // 2])

# predict the value of the target attribute on the test dataset
expected = digits.target[n_samples // 2:]
predicted = classifier.predict(data[n_samples // 2:])

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

images_and_predictions = list(zip(digits.images[n_samples // 2:], predicted))
for index, (image, prediction) in enumerate(images_and_predictions[:4]):
    plt.subplot(2, 4, index + 5)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Prediction: %i' % prediction)

plt.show()
