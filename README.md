# TruncatedBackpropagationThroughTime
 Implementation of TBPTT in Pytorch
 This is an implementation of TBPTT for n-layer LSTM network 
 
![tbptt](https://github.com/hansungj/TruncatedBackpropagationThroughTime/blob/master/tbtpp.jpg)
TBPTT with k1=2, k2=5. 

This is a special case of TBPTT where you backpropagate k1 losses at every k1 steps back for k2 steps. TBPTT is useful when it is needed to train long sequences. 

Based on the code from https://discuss.pytorch.org/t/implementing-truncated-backpropagation-through-time/15500 

References:
1. https://www.cs.utoronto.ca/~ilya/pubs/ilya_sutskever_phd_thesis.pdf (see page 23)
