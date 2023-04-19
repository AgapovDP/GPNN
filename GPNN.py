from torch import nn
import numpy as np

def outputLaeyrs(otput, lenOfFirstLayer = 700,lenOfSecondLayer = 70, nonlinearity = nn.ReLU, mod = 'classifier'):
  seq = nn.Sequential(nn.Linear(lenOfFirstLayer, lenOfSecondLayer, bias=False),
                      nn.BatchNorm1d(lenOfSecondLayer),
                      nonlinearity(),
                      nn.Linear(lenOfSecondLayer, otput))
  if mod == 'classifier':
    seq.append(nn.Sigmoid())
  return seq
  
embSeq_base = nn.Sequential(
     nn.Linear(5, 200, bias=False),
     nn.BatchNorm1d(200),
     nn.ReLU(),
     nn.Linear(200, 2000, bias=False),
     nn.BatchNorm1d(2000),
     nn.ReLU(),
     nn.Linear(2000, 4000, bias=False),
     nn.BatchNorm1d(4000),
     nn.ReLU(),
     nn.Linear(4000, 700, bias=False),
     nn.BatchNorm1d(700),
     nn.ReLU()
)


regrSeq_base = nn.Sequential(
     nn.Linear(700, 1500, bias=False),
     nn.BatchNorm1d(1500),
     nn.ReLU(),
     nn.Linear(1500, 700, bias=False),
     nn.BatchNorm1d(700),
     nn.ReLU()
)

class GPNN_v1(nn.Module):
  def __init__(self, SeqEmbedding = embSeq_base, SeqRegression = regrSeq_base, init_form = "normal"):
    super().__init__()
    self.embedding_stack = SeqEmbedding
    self.regression_stack = SeqRegression

    self.classifierLAA = outputLaeyrs(1); self.classifierLPA = outputLaeyrs(1)
    self.classifierCAA = outputLaeyrs(1); self.classifierCPA = outputLaeyrs(1)
    
    #self.regressionT = outputLaeyrs(1, mod = 0)
    self.regressionP = outputLaeyrs(1, mod = 0)
    self.regressionTheta = outputLaeyrs(1, mod = 0) 
    self.regressionDelta = outputLaeyrs(1, mod = 0)
    self.regressionPhi = outputLaeyrs(1, mod = 0)
    self.regressionCAA = outputLaeyrs(1, mod = 0)
    self.regressionCPA = outputLaeyrs(1, mod = 0)

    self.init_form = init_form
    if self.init_form is not None:
      self.init()
            

  def forward(self, x, mod = 'NoTrain', classVectors = np.nan):
    emb = self.embedding_stack(x)
    if mod == 'classifier':
      return self.classifierLAA(emb),self.classifierLPA(emb),\
            self.classifierCAA(emb),self.classifierCPA(emb)

    if mod == 'regression':
      outReg = self.regression_stack(emb)
      return self.regressionP(outReg),self.regressionTheta(outReg),self.regressionDelta(outReg),\
              self.regressionPhi(outReg), self.regressionCAA(outReg), self.regressionCPA(outReg)

    if mod == 'NoTrain':
      outReg = self.regression_stack(emb)
      return self.classifierLAA(emb),self.classifierLPA(emb), self.classifierCAA(emb),\
        self.classifierCPA(emb), self.regressionP(outReg),\
          self.regressionTheta(outReg), self.regressionDelta(outReg), self.regressionPhi(outReg),\
            self.regressionCAA(outReg), self.regressionCPA(outReg)


  
  def init(self):
    relu_gain = nn.init.calculate_gain("relu")
    for child in self.embedding_stack.children():
      if isinstance(child, nn.Linear):
        if self.init_form == "normal":
          nn.init.kaiming_normal_(child.weight, nonlinearity='relu')
          if child.bias is not None: nn.init.zeros_(child.bias)
        elif self.init_form == "uniform":
          nn.init.kaiming_uniform_(child.weight,nonlinearity='relu')
          if child.bias is not None:nn.init.zeros_(child.bias)
        else:
          raise NotImplementedError()

    for child in self.regression_stack.children():
      if isinstance(child, nn.Linear):
        if self.init_form == "normal":
          nn.init.kaiming_normal_(child.weight, nonlinearity='relu')
          if child.bias is not None: nn.init.zeros_(child.bias)
        elif self.init_form == "uniform":
          nn.init.kaiming_uniform_(child.weight,nonlinearity='relu')
          if child.bias is not None:nn.init.zeros_(child.bias)
        else:
          raise NotImplementedError()
