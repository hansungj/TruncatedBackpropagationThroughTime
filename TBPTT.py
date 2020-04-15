import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy
import time

class TruncatedBPPT():

    def __init__(self, one_step_module, loss_module, k1, k2, optimizer, num_layers, N , batch_size, device, max_norm = 5):
        self.one_step_module = one_step_module
        self.loss_module = loss_module
        self.k1 = k1
        self.k2 = k2
        self.retain_graph = k1 < k2
        self.optimizer = optimizer
        self.num_layers = num_layers
        self.N = N
        self.batch_size = batch_size
        self.device = device
        self.max_norm = max_norm

        self.loss_history = []
        self.loss = 0

    def train(self, input_sequence, epoch):
        
        self.one_step_module.train()
        init_state = self.one_step_module.init_hidden(self.batch_size, self.device)
        states = [(None, init_state)] # init_state is a tuple of (h,c)
        outputs = []
        targets = []

        count = 0
        loss_track = 0
        pgr = 0

        start = time.time()

        for i, (num, emb, tgt, flag) in enumerate(input_sequence):
            #unpack and repack
            state = []
            for l in range(self.num_layers):
                state_h = states[-1][1][l][0].detach()
                state_c = states[-1][1][l][1].detach() 
                state_h.requires_grad = True
                state_c.requires_grad = True
                state.append((state_h,state_c))

            output, new_state = self.one_step_module(num, emb, state)
            outputs.append(output)
            targets.append(tgt)
            
            while len(outputs) > self.k1:
                # Delete stuff that is too old
                del outputs[0]
                del targets[0]

            #([tuple], [tuple])
            states.append((state, new_state)) 

            # Delete stuff that is too old
            while len(states) > self.k2:
                del states[0]

            if ((i+1) % self.k1 == 0 ) | flag:

                optimizer.zero_grad()
                # backprop 
                for j in range(self.k2-1):

                    if j < self.k1:
                        loss = self.loss_module(outputs[-j-1], targets[-j-1])
                        loss.backward(retain_graph=True)
                        loss_track += float(loss)

                    
                    if states[-j-2][0] is None:
                        break

                    #back propagate from the top layer 
                    for l in range(self.num_layers-1,-1,-1):
                        curr_grad_h = states[-j-1][0][l][0].grad
                        curr_grad_c = states[-j-1][0][l][1].grad

                        # retain so that we can backpropagate for the next k1 elements
                        states[-j-2][1][l][0].backward(curr_grad_h, retain_graph=self.retain_graph)
                        states[-j-2][1][l][1].backward(curr_grad_c, retain_graph=self.retain_graph)


                #apply gradient clipping
                for name, p in self.one_step_module.named_parameters():
                    _ = nn.utils.clip_grad_norm_(p, max_norm = self.max_norm )
                
                optimizer.step()
            count += 1

            if flag:
                pgr += 1
                progress(pgr/self.N, "epoch: [%2d] [%4d/%4d] loss: %2.6f" % (epoch, pgr, self.N, loss_track/count))

                #next batch start over
                init_state = self.one_step_module.init_hidden(self.batch_size, self.device)
                states = [(None, init_state)] # init_state is a tuple of (h,c)

                outputs = []
                targets = []
                self.loss += loss_track/count
                loss_track = 0
                count = 0
                
        
        self.loss_history.append(self.loss/self.N)
        print("Training loss {}: It has taken {} seconds".format(self.loss/self.N, (time.time()-start)/60))
        self.loss = 0

    def evaluate(self, data, test_i):
        self.one_step_module.eval()

        predictions = []
        targets = []
        j = 0

        pred = []
        hid = model.init_hidden(1, self.device)
        tgt = []
        for i, (num, emb, target, flag) in enumerate(data):
            
            

            p, hid = model(num, emb, hid)
            pred.append(p.detach().cpu().numpy())
            tgt.append(target.detach().cpu().numpy().squeeze())

            if flag:
                pred = np.array(pred[-test_i[j]:])
                tgt = np.array(tgt[-test_i[j]:])

                predictions.append(pred)
                targets.append(tgt)

                hid = model.init_hidden(1, self.device)
                tgt = []
                pred = []
                j += 1

        return np.concatenate(predictions).squeeze(), np.concatenate(targets) 

