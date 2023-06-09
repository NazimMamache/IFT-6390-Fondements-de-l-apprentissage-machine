import numpy as np


class SVM:
    def __init__(self, eta, C, niter, batch_size, verbose):
        self.eta = eta
        self.C = C
        self.niter = niter
        self.batch_size = batch_size
        self.verbose = verbose

    def make_one_versus_all_labels(self, y, m):
        """
        Par exemple, si le tableau que l’on donne en entrée est [1, 0, 2] et que m = 4, la fonction retournera le tableausuivant:[[−1,1,−1,−1],[1,−1,−1,−1],[−1,−1,1,−1]].
        y : numpy array of shape (n,)
        m : int (num_classes)
        returns : numpy array of shape (n, m)
        """
        y_one_versus_all = -np.ones((y.shape[0], m))
        for i in range(y.shape[0]):
            y_one_versus_all[i, y[i]] = 1
        return y_one_versus_all

    def compute_loss(self, x, y):
        """
        Étant donné un minibatch d’exemples, cette fonction devrait calculer la perte. Les entrées de la fonction sont x (un tableau numpy de dimension (minibatch size, 562)) et y (un tableau numpy de dimension (minibatch size, 6)) et la sortie devrait être la perte calculée, un scalaire.
        Loss for entire x_train before training should equal: 24.0
        x : numpy array of shape (minibatch size, num_features)
        y : numpy array of shape (minibatch size, num_classes)
        returns : float
        """
        return (((2 - x.dot(self.w) * y).clip(0)) ** 2).sum(1).mean() + self.C/2 *  np.linalg.norm(self.w.flatten()) ** 2
                

    def compute_gradient(self, x, y):
        """
        x : numpy array of shape (minibatch size, num_features)
        y : numpy array of shape (minibatch size, num_classes)
        returns : numpy array of shape (num_features, num_classes)
        """
        return 2 / x.shape[0] * np.dot(-x.T, y*(2 - np.dot(x,self.w) * y).clip(0)) + self.C * self.w
        
    # Batcher function
    def minibatch(self, iterable1, iterable2, size=1):
        l = len(iterable1)
        n = size
        for ndx in range(0, l, n):
            index2 = min(ndx + n, l)
            yield iterable1[ndx: index2], iterable2[ndx: index2]

    def infer(self, x):
        """
        x : numpy array of shape (num_examples_to_infer, num_features)
        returns : numpy array of shape (num_examples_to_infer, num_classes)
        """
        y_inferred = - np.ones((x.shape[0], self.w.shape[1]))
        for i in range(x.shape[0]):
            y_inferred[i, np.argmax(np.dot(x[i], self.w))] = 1
        return y_inferred

    def compute_accuracy(self, y_inferred, y):
        """
        y_inferred : numpy array of shape (num_examples, num_classes)
        y : numpy array of shape (num_examples, num_classes)
        returns : float
        """
        acc = 0
        for i in range(y.shape[0]):
            if np.array_equal(y_inferred[i], y[i]) :
                acc += 1
        return acc / y.shape[0]

    def fit(self, x_train, y_train, x_test, y_test):
        """
        x_train : numpy array of shape (number of training examples, num_features)
        y_train : numpy array of shape (number of training examples, num_classes)
        x_test : numpy array of shape (number of training examples, nujm_features)
        y_test : numpy array of shape (number of training examples, num_classes)
        returns : float, float, float, float
        """
        self.num_features = x_train.shape[1]
        self.m = y_train.max() + 1
        y_train = self.make_one_versus_all_labels(y_train, self.m)
        y_test = self.make_one_versus_all_labels(y_test, self.m)
        self.w = np.zeros([self.num_features, self.m])

        train_losses = []
        train_accs = []
        test_losses = []
        test_accs = []

        for iteration in range(self.niter):
            # Train one pass through the training set
            for x, y in self.minibatch(x_train, y_train, size=self.batch_size):
                grad = self.compute_gradient(x, y)
                self.w -= self.eta * grad

            # Measure loss and accuracy on training set
            train_loss = self.compute_loss(x_train, y_train)
            y_inferred = self.infer(x_train)
            train_accuracy = self.compute_accuracy(y_inferred, y_train)

            # Measure loss and accuracy on test set
            test_loss = self.compute_loss(x_test, y_test)
            y_inferred = self.infer(x_test)
            test_accuracy = self.compute_accuracy(y_inferred, y_test)

            if self.verbose:
                print(f"Iteration {iteration} | Train loss {train_loss:.04f} | Train acc {train_accuracy:.04f} |"
                      f" Test loss {test_loss:.04f} | Test acc {test_accuracy:.04f}")

            # Record losses, accs
            train_losses.append(train_loss)
            train_accs.append(train_accuracy)
            test_losses.append(test_loss)
            test_accs.append(test_accuracy)

        return train_losses, train_accs, test_losses, test_accs


# DO NOT MODIFY THIS FUNCTION
# Data should be downloaded from the below url, and the
# unzipped folder should be placed in the same directory
# as your solution file:.
# https://drive.google.com/file/d/0Bz9_0VdXvv9bX0MzUEhVdmpCc3c/view?usp=sharing&resourcekey=0-BirYbvtYO-hSEt09wpEBRw
def load_data():
    # Load the data files
    print("Loading data...")
    data_path = "Smartphone Sensor Data/train/"
    x = np.genfromtxt(data_path + "X_train.txt")
    y = np.genfromtxt(data_path + "y_train.txt", dtype=np.int64) - 1
    
    # Create the train/test split
    x_train = np.concatenate([x[0::5], x[1::5], x[2::5], x[3::5]], axis=0)
    x_test = x[4::5]
    y_train = np.concatenate([y[0::5], y[1::5], y[2::5], y[3::5]], axis=0)
    y_test = y[4::5]

    # normalize the data
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    # add implicit bias in the feature
    x_train = np.concatenate([x_train, np.ones((x_train.shape[0], 1))], axis=1)
    x_test = np.concatenate([x_test, np.ones((x_test.shape[0], 1))], axis=1)

    return x_train, y_train, x_test, y_test


if __name__ == "__main__":

    x_train, y_train, x_test, y_test = load_data()

    print("Fitting the model...")
    svm = SVM(eta=0.0001, C=2, niter=200, batch_size=100, verbose=False)
    train_losses, train_accs, test_losses, test_accs = svm.fit(x_train, y_train, x_test, y_test)

    # # to infer after training, do the following:
    # y_inferred = svm.infer(x_test)

    ## to compute the gradient or loss before training, do the following:
    # y_train_ova = svm.make_one_versus_all_labels(y_train, 6) # one-versus-all labels
    # svm.w = np.zeros([x_train.shape[1], 6])
    # grad = svm.compute_gradient(x_train, y_train_ova)
    # loss = svm.compute_loss(x_train, y_train_ova)
