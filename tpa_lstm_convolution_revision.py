import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Model(nn.Module):
    def __init__(self, args, data):
        super(Model, self).__init__()
        self.use_cuda = args.cuda
        self.window_length = args.window;  # window, read about it after understanding the flow of the code...What is window size? --- temporal window size (default 24 hours * 7)
        self.original_columns = data.original_columns  # the number of columns or features
        self.hidR = args.hidRNN;
        self.hidden_state_features = args.hidden_state_features
        self.hidC = args.hidCNN;
        self.hidS = args.hidSkip;
        self.Ck = args.CNN_kernel;  # the kernel size of the CNN layers
        self.skip = args.skip;
        self.pt = (self.window_length - self.Ck) // self.skip
        self.hw = args.highway_window
        self.num_layers_lstm = args.num_layers_lstm
        self.hidden_state_features_uni_lstm = args.hidden_state_features_uni_lstm
        self.attention_size_uni_lstm = args.attention_size_uni_lstm
        self.num_layers_uni_lstm = args.num_layers_uni_lstm
        self.lstm = nn.LSTM(input_size=self.original_columns, hidden_size=self.hidden_state_features,
                            num_layers=self.num_layers_lstm,
                            bidirectional=False);
        self.uni_lstm = nn.LSTM(input_size=1, hidden_size=args.hidden_state_features_uni_lstm,
                            num_layers=args.num_layers_uni_lstm,
                            bidirectional=False);
        self.compute_convolution = nn.Conv2d(1, self.hidC, kernel_size=(
            self.Ck, self.hidden_state_features))  # hidC are the num of filters, default value of Ck is one
        # attention을 적용하기 위한 파라미터 (수정 전의 기존의 코드는 convolution이 window 별로 이루어졌다.)
        self.attention_matrix = nn.Parameter(
            torch.ones(args.batch_size, self.hidC, self.hidden_state_features, requires_grad=True)).cuda() #, device='cuda'
        self.context_vector_matrix = nn.Parameter(
            torch.ones(args.batch_size, self.hidden_state_features, self.hidC, requires_grad=True)).cuda() #, device='cuda'
        self.final_state_matrix = nn.Parameter(
            torch.ones(args.batch_size, self.hidden_state_features, self.hidden_state_features, requires_grad=True)).cuda() #, device='cuda'
        self.final_matrix = nn.Parameter(
            torch.ones(args.batch_size, self.original_columns, self.hidden_state_features, requires_grad=True)).cuda() #, device='cuda'

        self.attention_matrix_uni_lstm = nn.Parameter(
            torch.ones(args.batch_size, self.hidden_state_features_uni_lstm, self.hidden_state_features_uni_lstm, self.original_columns, requires_grad=True)).cuda()
        self.context_vector_matrix_uni_lstm = nn.Parameter(
            torch.ones(args.batch_size, self.hidden_state_features_uni_lstm, self.hidden_state_features_uni_lstm, self.original_columns,
                       requires_grad=True)).cuda()
        self.final_hidden_uni_matrix = nn.Parameter(
            torch.ones(args.batch_size, self.hidden_state_features_uni_lstm, self.hidden_state_features_uni_lstm,
                       self.original_columns,
                       requires_grad=True)).cuda()
        self.final_uni_matrix = nn.Parameter(
            torch.ones(args.batch_size, self.hidden_state_features, self.hidden_state_features_uni_lstm,
                       self.original_columns,
                       requires_grad=True)).cuda()


        self.bridge_matrix = nn.Parameter(
            torch.ones(args.batch_size, self.hidden_state_features, self.hidden_state_features,
                       requires_grad=True)).cuda()


        torch.nn.init.xavier_uniform(self.attention_matrix)
        torch.nn.init.xavier_uniform(self.context_vector_matrix)
        torch.nn.init.xavier_uniform(self.final_state_matrix)
        torch.nn.init.xavier_uniform(self.final_matrix)
        torch.nn.init.xavier_uniform(self.attention_matrix_uni_lstm)
        torch.nn.init.xavier_uniform(self.context_vector_matrix_uni_lstm)
        torch.nn.init.xavier_uniform(self.final_hidden_uni_matrix)
        torch.nn.init.xavier_uniform(self.final_state_matrix)
        torch.nn.init.xavier_uniform(self.bridge_matrix)


        self.conv1 = nn.Conv2d(1, self.hidC, kernel_size=(self.Ck, self.original_columns));  # kernel size is size for the filters
        self.GRU1 = nn.GRU(self.hidC, self.hidR);
        self.dropout = nn.Dropout(p=args.dropout);
        if (self.skip > 0):
            self.GRUskip = nn.GRU(self.hidC, self.hidS);
            self.linear1 = nn.Linear(self.hidR + self.skip * self.hidS, self.original_columns);
        else:
            self.linear1 = nn.Linear(self.hidR, self.original_columns);
        if (self.hw > 0):
            self.highway = nn.Linear(self.hw, 1);
        self.output = None;
        if (args.output_fun == 'sigmoid'):
            self.output = F.sigmoid;
        if (args.output_fun == 'tanh'):
            self.output = F.tanh;

    def forward(self, x):
        batch_size = x.size(0);
        x = x.cuda()

        """
           Step 1. First step is to feed this information to LSTM and find out the hidden states

            General info about LSTM:

            Inputs: input, (h_0, c_0)
        - **input** of shape `(seq_len, batch, input_size)`: tensor containing the features
          of the input sequence.
          The input can also be a packed variable length sequence.
          See :func:`torch.nn.utils.rnn.pack_padded_sequence` or
          :func:`torch.nn.utils.rnn.pack_sequence` for details.
        - **h_0** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
          containing the initial hidden state for each element in the batch.
          If the RNN is bidirectional, num_directions should be 2, else it should be 1.
        - **c_0** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
          containing the initial cell state for each element in the batch.

          If `(h_0, c_0)` is not provided, both **h_0** and **c_0** default to zero.


    Outputs: output, (h_n, c_n)
        - **output** of shape `(seq_len, batch, num_directions * hidden_size)`: tensor
          containing the output features `(h_t)` from the last layer of the LSTM,
          for each t. If a :class:`torch.nn.utils.rnn.PackedSequence` has been
          given as the input, the output will also be a packed sequence.

          For the unpacked case, the directions can be separated
          using ``output.view(seq_len, batch, num_directions, hidden_size)``,
          with forward and backward being direction `0` and `1` respectively.
          Similarly, the directions can be separated in the packed case.
        - **h_n** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
          containing the hidden state for `t = seq_len`.

          Like *output*, the layers can be separated using
          ``h_n.view(num_layers, num_directions, batch, hidden_size)`` and similarly for *c_n*.
        - **c_n** (num_layers * num_directions, batch, hidden_size): tensor
          containing the cell state for `t = seq_len`
        """

        ##Incase in future bidirectional lstms are to be used, size of hn would needed to be modified a little (as output is of size (num_layers * num_directions, batch, hidden_size))
        """LSTM에 데이터를 input 시켜서 hidden state를 뽑아낸다."""
        # 모델의 self.lstm이 batch_first=True로 되어있지 않기에 shape을 [window_size, batch_size, n_features]로 바꿔준다.
        input_to_lstm = X.permute(1, 0, 2).contiguous() # .contiguous()를 사용하여 tensor의 차원을 바꿨을 때 메모리를 새롭게 할당
        lstm_hidden_states, (h_all, c_all) = self.lstm(input_to_lstm) # original_feature를 hidden_state_features 만큼 늘린다.
        # h_all[-1]을 통해서 [1, 128, 12] -> [128, 12]로 바꿔준다.
        # h_all.size(1) = batch_size를 의미
        # h_all.size(2) = lstm의 hidden_state_features를 의미
        hn = h_all[-1].view(1, h_all.size(1), h_all.size(2))

        """
            Step 2. Apply convolution on these hidden states. As in the paper TPA-LSTM, these filters are applied on the rows of the hidden state
        """
        """LSTM의 batch별+window별 hidden state를 가져와서 차원 변경"""
        ## hidden state들은 행 방향으로 convolution이 적용됨
        output_realigned = lstm_hidden_states.permute(1, 0, 2).contiguous()
        hn = hn.permute(1, 0, 2).contiguous()
        # cn = cn.permute(1, 0, 2).contiguous()
        # convolution에 들어가기 위해서는 channel이 필요함 [batch_size, channels, window_size, n_features]
        # tpa-lstm은 row가 변수, column이 window_length가 되어서 lstm으로부터 나온 각 hidden feature의 window_length 별로 convolution이 이루어져야 한다.
        input_to_convolution_layer = output_realigned.permute(0,2,1).unsqueeze(1) # [batch_size, channels, n_features, window_size]

        # nn.Conv1d(1, )에서 1은 in_channels이다.
        # self.hidC = convolution filter의 개수
        # self.Ck = cnn의 kernel size
        convolution_output = F.relu(self.compute_convolution(input_to_convolution_layer)); # 10개의 filter를 적용하여서 1개인 기존의 channel을 10개로 늘림
        convolution_output = self.dropout(convolution_output);


        """
            Step 3. Apply attention on this convolution_output
            - LSTM의 final hidden state와 CNN을 거치 hidden state 값들의 matrix multiplication 부분 -> scoring function
            - LSTM의 final hidden states : final_hn
            - CNN을 거친 후의 hidden states : final_convolution_output

            * 수정 전 : [10, 168]로 168개의 window들에 대한 10개 filter의 output 값이 나옴 
            * 수정 후(convolution 적용 부분 수정) : [10, 12]로 12개의 hidden_features들에 대한 10개 filter의 output 값이 나옴
            
        """
        convolution_output = convolution_output.squeeze(3)


        """Matrix Multiplication을 할 때, padding을 진행하기 위한 부분"""
        ## final_hn : LSTM의 final_hidden_state를 scoring function과 곱하기 위해서 padding을 진행하는 부분
        ## final_convolution_output : LSTM의 hidden state matrix에 convolution을 진행한 후 나온 값으로 이후 weight를 곱하여 (scoring function을 진행하여) atttention weight가 반영된 Vt를 만들어낸다.
        final_hn = torch.zeros(self.attention_matrix.size(0), 1, self.hidden_state_features)
        input = torch.zeros(self.attention_matrix.size(0), x.size(1), x.size(2))
        final_convolution_output = torch.zeros(self.attention_matrix.size(0), self.hidC, self.window_length)
        diff = 0
        ## 첫 부분은 batch_size별로 묶었을 때의 마지막 batch를 위한 부분 128씩 묶었을 때, 데이터가 부족하여 21개만 묶일 수 있다.
        if (hn.size(0) < self_attention_matrix.size(0)): # hn의 0차원 값 (batch_size)이 self_attention_matrix의 0차원 값(batch_size) 보다 작다면
            # final_hn과 final_convolution_output은 모두 128의 크기로 되어있다.
            # 마지막 batch 묶음이 128개가 되지 않는다면 나머지는 0으로 padding 처리
            final_hn[:hn.size(0), :, :] = hn
            final_convolution_output[:convolution_output.size(0), :, :] = convolution_output
            input[:x.size(0), :, :] = x
            diff = self.attention_matrix.size(0) - hn.size(0)
        ## 128개의 batch가 모두 온전하게 들어있는 경우 그대로 설정
        else:
            final_hn = hn
            final_convolution_output = convolution_output
            input = x.cuda()

        """
           final_hn, final_convolution_output are the matrices to be used from here on
        """
        convolution_output_for_scoring = final_convolution_output.permute(0, 2, 1).contiguous()
        final_hn_realigned = final_hn.permute(0, 2, 1).contiguous()
        convolution_output_for_scoring = convolution_output_for_scoring.cuda()
        final_hn_realigned = final_hn_realigned.cuda()
        mat1 = torch.bmm(convolution_output_for_scoring, self.attention_matrix).cuda()
        scoring_function = torch.bmm(mat1, final_hn_realigned).cuda()
        ## 위에서 곱한 mat1과 padding을 적용한 LSTM의 마지막 layer를 곱한 값
        ## 이후 sigmoid를 취하면 -> 최종 attention weight가 구해진다.
        alpha = torch.sigmoid(scoring_function)
        context_vector = alpha * convolution_output_for_scoring
        context_vector = torch.sum(context_vector, dim=1).cuda()

        """
           Step 4. Compute the output based upon final_hn_realigned, context_vector
        """
        context_vector = context_vector.view(-1, self.hidC, 1)

        # lstm의 final hideen_state와 weight_1을 곱하고, attention을 적용한 context vector(Vt)와 weight_2를 곱하여 h't를 만들어 낸다.
        h_intermediate = torch.bmm(self.final_state_matrix, final_hn_realigned) + torch.bmm(self.context_vector_matrix, context_vector)

        result = torch.bmm(self.final_matrix, h_intermediate.cuda())
        result = result.permute(0, 2, 1).contiguous()
        result = result.squeeze()

        """ padding을 한 부분 제외 """
            # 마지막 batch에 해당할 듯 (batch_size에 맞지 않는 batch에 해당)
        final_result = result[:result.size(0) - diff]

        """
        Adding highway network to it
        """
        if (self.hw > 0):
            z = x[:, -self.hw:, :];
            z = z.permute(0, 2, 1).contiguous().view(-1, self.hw);
            z = self.highway(z);
            z = z.view(-1, self.original_columns);
            res = final_result + z;

        return torch.sigmoid(res)

