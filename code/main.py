import h5py
import numpy as np

from code.model import UNetClassifier


def load_dataset(covid_file_path, normal_file_path):
    covid = h5py.File(covid_file_path, 'r')['covid']
    normal = h5py.File(normal_file_path, 'r')['normal']

    all_images = np.expand_dims(np.concatenate([covid, normal]), axis=3)
    all_labels = np.concatenate([[1]*covid.shape[0], [0]*normal.shape[0]])

    shuffled_indices = np.random.permutation(np.arange(all_images.shape[0]))

    all_images = all_images[shuffled_indices]
    all_labels = all_labels[shuffled_indices]

    return all_images, all_labels

if __name__ == '__main__':
    model = Classifier((512, 512, 1), 2, True)
        
    all_images, all_labels = load_dataset()

    print(all_images.shape, all_labels.shape)

    model.train(all_images, all_labels, 15, 16, 0.2)

