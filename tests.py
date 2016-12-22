import unittest

import numpy as np

from train_and_eval_direction_faced import FaceDirectionClassifier


class TestFaceDirectionClassifier(unittest.TestCase):

    def setUp(self):
        self.fc = FaceDirectionClassifier()
        self.fc.read_data()
        self.fc.train()

    def test_predict(self):
        predictions = self.fc.predict([np.random.rand(1, 960)])
        self.assertIsNotNone(predictions)
        self.assertIsNotNone(predictions[0])

if __name__ == "__main__":
    unittest.main()
