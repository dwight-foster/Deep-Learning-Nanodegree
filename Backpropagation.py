def forward_back_prop(rnn, optimizer, criterion, inp, target, hidden):
    """
    Forward and backward propagation on the neural network
    :param decoder: The PyTorch Module that holds the neural network
    :param decoder_optimizer: The PyTorch optimizer for the neural network
    :param criterion: The PyTorch loss function
    :param inp: A batch of input to the neural network
    :param target: The target output for the batch of input
    :return: The loss and the latest hidden state Tensor
    """
    clip = 5
    # TODO: Implement Function
    
    # move data to GPU, if available
    if train_on_gpu: 
        inp, target = inp.cuda(), target.cuda()
    # perform backpropagation and optimization
    h = tuple([each.data for each in hidden])
    rnn.zero_grad()
    output, hidden = rnn(inp, h)
    print(output.squeeze())
   
    loss = criterion(output.squeeze(), target.float())
    loss.backward()
    nn.utils.clip_grad_norm_(rnn.parameters(),clip)
    optimizer.step()
    # return the loss over a batch and the hidden state produced by our model
    return loss, hidden

# Note that these tests aren't completely extensive.
# they are here to act as general checks on the expected outputs of your functions
"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_forward_back_prop(RNN, forward_back_prop, train_on_gpu)





Full Error here: 
RuntimeError                              Traceback (most recent call last)
<ipython-input-32-bf2baee81e3a> in <module>()
     33 DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
     34 """
---> 35 tests.test_forward_back_prop(RNN, forward_back_prop, train_on_gpu)

/home/workspace/problem_unittests.py in test_forward_back_prop(RNN, forward_back_prop, train_on_gpu)
    217         hidden = rnn.init_hidden(batch_size)
    218 
--> 219         loss, hidden_out = forward_back_prop(mock_decoder, mock_decoder_optimizer, mock_criterion, inp, target, hidden)
    220 
    221     assert (hidden_out[0][0]==hidden[0][0]).sum()==batch_size*hidden_dim, 'Returned hidden state is the incorrect size.'

<ipython-input-32-bf2baee81e3a> in forward_back_prop(rnn, optimizer, criterion, inp, target, hidden)
     21     print(output.squeeze())
     22 
---> 23     loss = criterion(output.squeeze(), target.float())
     24     loss.backward()
     25     nn.utils.clip_grad_norm_(rnn.parameters(),clip)

/opt/conda/lib/python3.6/unittest/mock.py in __call__(_mock_self, *args, **kwargs)
    937         # in the signature
    938         _mock_self._mock_check_sig(*args, **kwargs)
--> 939         return _mock_self._mock_call(*args, **kwargs)
    940 
    941 

/opt/conda/lib/python3.6/unittest/mock.py in _mock_call(_mock_self, *args, **kwargs)
   1007         if (self._mock_wraps is not None and
   1008              self._mock_return_value is DEFAULT):
-> 1009             return self._mock_wraps(*args, **kwargs)
   1010         if ret_val is DEFAULT:
   1011             ret_val = self.return_value

/opt/conda/lib/python3.6/site-packages/torch/nn/modules/module.py in __call__(self, *input, **kwargs)
    489             result = self._slow_forward(*input, **kwargs)
    490         else:
--> 491             result = self.forward(*input, **kwargs)
    492         for hook in self._forward_hooks.values():
    493             hook_result = hook(self, input, result)

/opt/conda/lib/python3.6/site-packages/torch/nn/modules/loss.py in forward(self, input, target)
    757         _assert_no_grad(target)
    758         return F.cross_entropy(input, target, self.weight, self.size_average,
--> 759                                self.ignore_index, self.reduce)
    760 
    761 

/opt/conda/lib/python3.6/site-packages/torch/nn/functional.py in cross_entropy(input, target, weight, size_average, ignore_index, reduce)
   1440         >>> loss.backward()
   1441     """
-> 1442     return nll_loss(log_softmax(input, 1), target, weight, size_average, ignore_index, reduce)
   1443 
   1444 

/opt/conda/lib/python3.6/site-packages/torch/nn/functional.py in nll_loss(input, target, weight, size_average, ignore_index, reduce)
   1330                          .format(input.size(0), target.size(0)))
   1331     if dim == 2:
-> 1332         return torch._C._nn.nll_loss(input, target, weight, size_average, ignore_index, reduce)
   1333     elif dim == 4:
   1334         return torch._C._nn.nll_loss2d(input, target, weight, size_average, ignore_index, reduce)

RuntimeError: Expected object of type torch.LongTensor but found type torch.FloatTensor for argument #2 'target'
