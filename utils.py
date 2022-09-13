"""
   Contains all the utility functions that would be needed
   1. _normalized
   2. _split
   3._batchify
   4. get_batches
"""

"""
[보충 설명]
Data_utility를 통해서 나온 train, valid, test의 shape은 하나의 batch마다 window_size와 n_features에 맞게 넣어준 것
이후 modeling을 할 때, get_batches를 통해 지정한 args.batch_size별로 ipnut 및 backprop을 진행한다.
""" 

import torch
import numpy as np;
from torch.autograd import Variable


def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.) / (len(x)))


class Data_utility(object):
    # train and valid is the ratio of training set and validation set. test = 1 - train - valid
    def __init__(self, file_name, train, valid, cuda, horizon, window, normalize=2):
        self.cuda = cuda;
        self.window_length = window;
        self.horizon = horizon
        fin = open(file_name);
        self.original_data = np.loadtxt(fin, delimiter=','); # (7588, 8)
        self.normalized_data = np.zeros(self.original_data.shape);
        # self_original_rows : 7588
        # self_original_columns : 8
        self.original_rows, self.original_columns = self.normalized_data.shape;
        self.normalize = 2 # 정규화를 진행하는 게 디폴트
        self.scale = np.ones(self.original_columns); # 변수 개수만큼 1 생성 [1,1,1,1,1,1,1,1]
        self._normalized(normalize);

        # train, valid, test를 각 비율에 맞게 나누기
            # train은 train data의 비율, valid는 valid data의 비율을 의미
        self._split(int(train * self.original_rows), int((train + valid) * self.original_rows), self.original_rows);

        self.scale = torch.from_numpy(self.scale).float();# 변수 개수만큼 생성한 1을 float 형의 텐서로 전환
        # _split() 함수에 있는 self.test
        tmp = self.test[1] * self.scale.expand(self.test[1].size(0), self.original_columns);

        if self.cuda:
            self.scale = self.scale.cuda()
        self.scale = Variable(self.scale)

        #rse and rae must be some sort of errors for now, will come back to them later
        self.rse = normal_std(tmp);
        self.rae = torch.mean(torch.abs(tmp - torch.mean(tmp)));

    def _normalized(self, normalize):
        # normalized by the maximum value of entire matrix.
        if (normalize == 0):
            self.normalized_data = self.original_data
        if (normalize == 1):
            self.normalized_data = self.original_data / np.max(self.original_data);

        # normalized by the maximum value of each row(sensor).
        if (normalize == 2):
            for i in range(self.original_columns):
                self.scale[i] = np.max(np.abs(self.original_data[:, i]));
                self.normalized_data[:, i] = self.original_data[:, i] / np.max(np.abs(self.original_data[:, i]));

    def _split(self, train, valid, test):

        train_set = range(self.window_length + self.horizon - 1, train);
        valid_set = range(train, valid);
        test_set = range(valid, self.original_rows);
        self.train = self._batchify(train_set, self.horizon);
        self.valid = self._batchify(valid_set, self.horizon);
        self.test = self._batchify(test_set, self.horizon);
    """최종적인 batch 형태 [batch_size, window_size, n_features]로 만들어주는 게 아님
    각 batch 별로 window size만큼의 데이터가 들어갈 수 있게 해주는 것"""
    def _batchify(self, idx_set, horizon):

        n = len(idx_set); # idx_set은 데이터의 개수를 의미 -> trainset의 개수 or validset의 개수
        """8개의 변수에 대해 168의 window size로 4378개의 batch가 만들어짐
        이후 32개 혹은 128개 등의 batch_size를 적용하기 위해서는 get_batches가 필요"""
        X = torch.zeros((n, self.window_length, self.original_columns)); # self_window_length=168, # self_original_columns = 8
        Y = torch.zeros((n, self.original_columns));

        for i in range(n):
            end = idx_set[i] - self.horizon + 1;
            start = end - self.window_length;
            X[i, :, :] = torch.from_numpy(self.normalized_data[start:end, :]);
            Y[i, :] = torch.from_numpy(self.normalized_data[idx_set[i], :]);

        """
            Here matrix X is 3d matrix where each of it's 2d matrix is the separate window which has to be sent in for training.
            Y is validation.           
        """
        return [X, Y];

    def get_batches(self, inputs, targets, batch_size, shuffle=True):
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt];
            Y = targets[excerpt];
            if (self.cuda):
                X = X.cuda();
                Y = Y.cuda();
            yield Variable(X), Variable(Y);
            start_idx += batch_size