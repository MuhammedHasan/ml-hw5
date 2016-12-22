from train_and_eval_direction_faced import FaceDirectionClassifier


def model_evaluation(pic):
    """

    :param pic: np array of dimensions 30 x 32 representing an image
    :return: String specifying emotion that the subject is feeling
    """
    example_output = "sad"
    return example_output


class FaceEmationClassifier(FaceDirectionClassifier):

    def __init__(self, labels=('neutral', 'sad', 'angry', 'happy')):
        super(self.__class__, self).__init__(labels=labels)
        self.file_label_number = 2

    def predict():
        raise NotImplemented()

if __name__ == '__main__':
    fc = FaceEmationClassifier()
    fc.read_data()
    fc.train()
