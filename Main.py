import random
import math

#Cleanup
import os

# This is just a simple NN that i plan to use to easily conceptualise the ideas, therefore doesnt have lots of optimizations, effectively use matrixes, etc.. 

class Model:
  def __init__(self,data):
    self.initWB()
    self.learningRate = 0.1
    self.data = data
  def initWB(self): # add a value from the normal distribution: mean 0,std 1
    #print(random.normalvariate(0,1))
    self.w1,self.w2,self.w3,self.w4,self.b1,self.b2,self.b3 = [random.normalvariate(0,1) for _ in range(7)]
  def readWB(self,file): # file name to read the (trained) weights and biases from a file
    with open(file,'r') as f:
      read_data = f.read()
      print(read_data)
      pass
  def writeWB(self,file): # file name to save the (trained) weights and biases to a file
    with open(file,'w') as f:
      f.write(f'{self.w1}\n{self.w2}\n{self.w3}\n{self.w4}\n{self.b1}\n{self.b2}\n{self.b3}\n\n')
      pass
  def displayWB(self,choice=1):
    if choice == 0:
      print(f'w1:{self.w1}\n\nw2:{self.w2}\n\nw3:{self.w3}\n\nw4:{self.w4}\n\nb1:{self.b1}\n\nb2:{self.b2}\n\nb3:{self.b3}\n')
      with open('WeightsBiases.txt','a') as f:
        f.write(f'w1:{self.w1}\nw2:{self.w2}\nw3:{self.w3}\nw4:{self.w4}\nb1:{self.b1}\nb2:{self.b2}\nb3:{self.b3}\n\n')
    if choice == 1:
      print(f'w1:{self.w1}\n\nw2:{self.w2}\n\nw3:{self.w3}\n\nw4:{self.w4}\n\nb1:{self.b1}\n\nb2:{self.b2}\n\nb3:{self.b3}\n')
    else:
      with open('WeightsBiases.txt','a') as f:
        f.write(f'w1:{self.w1}\nw2:{self.w2}\nw3:{self.w3}\nw4:{self.w4}\nb1:{self.b1}\nb2:{self.b2}\nb3:{self.b3}\n\n')
    
  def activationFunc(self,x):
    return (1+math.exp(-x))**-1 #Sigmoid
    return (math.log((1+math.exp(x)),math.e)) #SoftPlus
    return max(0,x) #ReLu
    pass
  def dactivationFunc(self,x):
    return (math.exp(-x)/(1+math.exp(-x)**2)) #Sigmoid
    return (math.exp(x)/(1+math.exp(x))) #SoftPlus
    if x >= 0:    # ReLu
      return 1
    else:
      return 0
    pass
  def predict(self,x):# single for this architecture
    self.upperx = (x*self.w1+self.b1)
    self.lowerx = (x*self.w2+self.b2)
    upper = self.activationFunc(self.upperx)*self.w3
    lower = self.activationFunc(self.lowerx)*self.w4
    Result = upper + lower + self.b3
    return Result
  def backpropagation(self):
    # print(self.data[0])
    # print(self.data[1])
    #dw1: -2(actual-predicted)*w3 * d(Act) * input
    dw1 = dw2 = db1 = db2 = dw3 = dw4 = db3 = 0
    for datapoint in range(len(self.data[1])): # len(self.data[1]) : must use index of 1 or 0 as we want the length of the inner array
      # Repeated calculations 
      dSP = -2*(float(self.data[1][datapoint]) - self.predict(float(self.data[0][datapoint]))) # derivative of ssr in relation to Predicted
      dUAF = self.dactivationFunc(self.upperx) # derivative of the activation function evaluated by upper branch
      dLAF = self.dactivationFunc(self.lowerx) # derivative of the activation function evaluated by lower branch

      dw1 += dSP * self.w3 *  dUAF * float(self.data[0][datapoint])
      # print(w1)
      dw2 += dSP * self.w4 *  dLAF * float(self.data[0][datapoint])
      db1 += dSP * self.w3 *  dUAF 
      db2 += dSP * self.w4 *  dLAF
      dw3 += dSP * self.activationFunc(self.upperx)
      dw4 += dSP * self.activationFunc(self.lowerx)
      db3 += dSP
    #change the old value by gradient descent: Old - (Derivative * Learning Rate) 
    # possible to add criteria for when to stop
    self.w1 -= dw1 * self.learningRate
    self.w2 -= dw2 * self.learningRate
    self.w3 -= dw2 * self.learningRate
    self.w4 -= dw4 * self.learningRate
    self.b1 -= db1 * self.learningRate
    self.b2 -= db2 * self.learningRate
    self.b3 -= db3 * self.learningRate
    #Repeat

    pass 



## Cleanup
try:
  os.remove('WeightsBiases.txt')
except:
  pass

def getdata():
  f = open('data.txt','r')
  R_data = f.read().strip().split(',')
  f.close()
  datapoints = int(R_data[0])
  data =[[],[]]
  for point in range(1,datapoints+1):
    data[0].append(R_data[point])
    data[1].append(R_data[point+datapoints])
  return data

data = getdata()

Network = Model(data) 
Network.initWB()
# Network.displayWB()



# for x in range(5):
#   Network.backpropagation()
#   # Network.displayWB()

# def ppp(Model,no):
#   print(f'\nInput:{no}\nPrediction:{Model.predict(no)}')

# ppp(Network,0.75)
# ppp(Network,0.7)
# ppp(Network,0.55)
# ppp(Network,0.01)
# ppp(Network,0)

# for x in range(0,100):
#   no = x/100
#   ppp(Network,no)
# print(Network.activationFunc(10))
# Network.predict(0)
# Network.displayWB()
# print(Network.activationFunc(-8978443))
# print((1+math.exp(-1))**-1,math.e)
