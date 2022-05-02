from datasets import load_dataset
from sklearn.metrics import accuracy_score
import numpy as np
import time

from text_classification import TextClassifier


def main(name):
    print(name)

    language_model = 'distilbert-base-uncased'
    train_set = load_dataset("emotion", split='train')
    validation_set = load_dataset("emotion", split='validation')
    test_set = load_dataset("emotion", split='test')

    emotions_classifier = TextClassifier(language_model=language_model)

    t1 = time.time()
    emotions_classifier.train(train_set, validation_set)
    t2 = time.time()
    print(f'Elapsed Time for Training: {t2 - t1} seconds')

    y_pred = emotions_classifier.predict(validation_set)
    y_val = np.array(validation_set['label'])
    accuracy = accuracy_score(y_val, y_pred)

    print(f'Accuracy: {accuracy}')

    dummy = -32


if __name__ == '__main__':
    main('Emotion Classification')
