# TruncatedBackpropagationThroughTime
 Implementation of TBPTT in Pytorch
 This is an implementation of TBPTT for n-layer LSTM network 
 
![tbptt](https://github.com/hansungj/TruncatedBackpropagationThroughTime/blob/master/tbtpp.jpg)
TBPTT with k1=2, k2=5. 

This is a special case of TBPTT where you backpropagate k1 losses at every k1 steps back for k2 steps. In the method described in Sutskever's Thesis, you only back propagate from t down to k2 whenever t divides by k1. Here we backpropagate t, t-1,t-2,t-3...,t-k1. TBPTT is useful when training sequences are too long for LSTM to handle. 

Based on the code from https://discuss.pytorch.org/t/implementing-truncated-backpropagation-through-time/15500 

References:
1. https://www.cs.utoronto.ca/~ilya/pubs/ilya_sutskever_phd_thesis.pdf (see page 23)
