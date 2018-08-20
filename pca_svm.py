import h5py
import numpy as np
import cv2

TRAIN_PERC = 0.85

hdf5_file = h5py.File("muzzle_USP.hdf5", "r") # Carrega o arquivo do dataset

images = hdf5_file["bois_img"] # Carrega as imagens salvas no arquivo
labels = hdf5_file["bois_label"] # Carrega os labels salvos no arquivo

num_bois = hdf5_file["bois_qtd"][0] # Carrega o numero de bois
num_fotos = hdf5_file["bois_nfotos"][0] # Carrega o numero de fotos por boi

images = np.float32(images)
labels = np.int32(labels)

images = np.reshape(images, (images.shape[0], images.shape[1] * images.shape[2] * images.shape[3]))
labels = np.reshape(labels, (labels.shape[0], 1))

mean, eigenVectors = cv2.PCACompute(images, mean = None, retainedVariance = TRAIN_PERC)
images_pca = cv2.PCAProject(images, mean, eigenVectors)

tam_train = int(TRAIN_PERC * num_fotos) * num_bois
tam_test = images.shape[0] - tam_train

img_train = np.zeros((tam_train, images_pca.shape[1]), dtype = np.float32)
img_test = np.zeros((tam_test, images_pca.shape[1]), dtype = np.float32)
labels_test = np.zeros((tam_test, 1), dtype = np.int32)
labels_train = np.zeros((tam_train, 1), dtype = np.int32)

pos_train = 0
pos_test = 0
for i in range(images_pca.shape[0]):
        if (i % num_fotos) < int(TRAIN_PERC * num_fotos):
                img_train[pos_train] = images_pca[i]
                labels_train[pos_train] = labels[i]
                pos_train += 1

        else:
                img_test[pos_test] = images_pca[i]
                labels_test[pos_test] = labels[i]
                pos_test += 1

svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_LINEAR)

svm.trainAuto(img_train, cv2.ml.ROW_SAMPLE, labels_train)

prev_result = svm.predict(img_test)
mask = prev_result[1] == np.reshape(labels_test, (labels_test.shape[0], 1))

print mask

acertos = 0
erros = 0
for result in mask:
	if result:
		acertos += 1
	else:
		erros += 1

print "Acertos: %s Erros: %s" % (acertos,erros)
print "Taxa de acertos: %s" % (acertos * 100.0/float(acertos + erros))
