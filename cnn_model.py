from train_and_eval_emotion_felt import FaceEmationClassifier
from network3 import Network, ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer


def model_evaluation(pic):
    """
    :param pic: np array of dimensions 120 x 128 representing an image
    :return: String specifying emotion that the subject is feeling
    """
    example_output = "sad"
    return example_output


class MyFaceEmationClassifier(FaceEmationClassifier):

    def __init__(self):
        super(self.__class__, self).__init__()

        self.net = Network([
            ConvPoolLayer(
                image_shape=(self.mini_batch_size, 1,
                             self._pic_size[0], self._pic_size[1]),
                filter_shape=(20, 1,
                              self._filter_window_size[0],
                              self._filter_window_size[1]),
                poolsize=(2, 2)
            ),
            FullyConnectedLayer(
                n_in=self._conved_pic_size(),
                n_out=100
            ),
            SoftmaxLayer(
                n_in=100,
                n_out=4
            )
        ], self.mini_batch_size)

    def _pic_to_features(self, pic):
        return pic.reshape(-1)

if __name__ == '__main__':
    fc = MyFaceEmationClassifier()
    fc.read_data()
    fc.train()
