import numpy as np

######## DO NOT MODIFY THIS FUNCTION ########
def draw_rand_label(x, label_list):
    seed = abs(np.sum(x))
    while seed < 1:
        seed = 10 * seed
    seed = int(1000000 * seed)
    np.random.seed(seed)
    return np.random.choice(label_list)
#############################################


class Q1:

    def feature_means(self, banknote):
        return np.mean(banknote[:,:-1], axis=0)

    def covariance_matrix(self, banknote):
        return np.cov(banknote[:,:-1], rowvar=False)

    def feature_means_class_1(self, banknote):
        return np.mean(banknote[banknote[:, -1] == 1][:,:-1], axis=0)

    def covariance_matrix_class_1(self, banknote):
        return np.cov(banknote[banknote[:, -1] == 1][:,:-1], rowvar=False)


class HardParzen:
    def __init__(self, h):
        self.h = h

    def train(self, train_inputs, train_labels):
        self.train_inputs = train_inputs
        self.train_labels = [int(item) for item in train_labels]
        self.label_list = np.unique(train_labels)
        self.n_class = len(np.unique(train_labels))

    def compute_predictions(self, test_data):
        num_test = test_data.shape[0]
        counts = np.ones((num_test, self.n_class))
        classes_pred = np.zeros(num_test)
        for (i, ex) in enumerate(test_data):
            distances = (np.sum((np.abs(ex - self.train_inputs)) ** 2, axis=1)) ** (1.0 / 2)
            indices_nn=np.array([k for k in range(len(distances)) if distances[k]<=self.h])
            if len(indices_nn) == 0 :
                classes_pred[i] = draw_rand_label(ex, self.label_list)
            else:
                for j in indices_nn:
                    counts[i,self.train_labels[j]]+=1
                classes_pred[i]=np.argmax(counts[i,:])
        return classes_pred


class SoftRBFParzen:
    def __init__(self, sigma):
        self.sigma  = sigma

    def train(self, train_inputs, train_labels):
        self.train_inputs = train_inputs
        self.train_labels = [int(item) for item in train_labels]
        self.label_list = np.unique(train_labels)
        self.n_class = len(np.unique(train_labels))
        self.d = train_inputs.shape[1] - 1

    def compute_predictions(self, test_data):
        num_test = test_data.shape[0]
        counts = np.ones((num_test, self.n_class))
        classes_pred = np.zeros(num_test)
        for (i, ex) in enumerate(test_data):
            for(j, ex2) in enumerate(self.train_inputs):
                distance =  (np.abs(ex2 - ex)**2).sum()**(1.0/2)
                k = 1/((2*np.pi)**(self.d/2) * self.sigma**self.d) * np.exp(-distance**2/(2*self.sigma**2))
                counts[i,self.train_labels[j]]+=k
            classes_pred[i]=np.argmax(counts[i,:])
        return classes_pred


def split_dataset(banknote):
    train_index = []
    validation_index = []
    test_index = []
    for i in range(banknote.shape[0]):
        if i%5 == 3:
            validation_index.append(i)
        elif i%5 == 4:
            test_index.append(i)
        else:
            train_index.append(i)

    return (banknote[train_index],banknote[validation_index],banknote[test_index])


class ErrorRate:
    def __init__(self, x_train, y_train, x_val, y_val):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val

    def hard_parzen(self, h):
        hardparzen = HardParzen(h)
        hardparzen.train(self.x_train, self.y_train)
        y_pred = hardparzen.compute_predictions(self.x_val)
        return np.sum(y_pred != self.y_val) / len(self.y_val)

    def soft_parzen(self, sigma):
        softparzen = SoftRBFParzen(sigma)
        softparzen.train(self.x_train, self.y_train)
        y_pred = softparzen.compute_predictions(self.x_val)
        return np.sum(y_pred != self.y_val) / len(self.y_val)


def get_test_errors(banknote):
    train, validation, test = split_dataset(banknote)
    x_train = train[:,:-1]
    y_train = train[:,-1]
    x_val = validation[:,:-1]
    y_val = validation[:,-1]
    x_test = test[:,:-1]
    y_test = test[:,-1]

    error_rate = ErrorRate(x_train, y_train, x_val, y_val)
    h = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 3.0, 10.0, 20.0]
    sigma = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 3.0, 10.0, 20.0]

    hard_parzen_error = []
    soft_parzen_error = []
    for i in h:
        hard_parzen_error.append(error_rate.hard_parzen(i))
    for i in sigma:
        soft_parzen_error.append(error_rate.soft_parzen(i))

    h_star = h[len(hard_parzen_error) - 1 - hard_parzen_error[::-1].index(min(hard_parzen_error))]
    sigma_star = sigma[len(soft_parzen_error) - 1 - soft_parzen_error[::-1].index(min(soft_parzen_error))]

    hardparzen = HardParzen(h_star)
    hardparzen.train(x_train, y_train)

    softRBFparzen = SoftRBFParzen(sigma_star)
    softRBFparzen.train(x_train, y_train)

    y_pred_HardParzen = hardparzen.compute_predictions(x_test)
    y_pred_SoftRBFParzen = softRBFparzen.compute_predictions(x_test)

    return (np.sum(y_pred_HardParzen != y_test) / len(y_test), np.sum(y_pred_SoftRBFParzen != y_test) / len(y_test))


def random_projections(X, A):
    return np.dot(1/2**(1./2)*X, A)
