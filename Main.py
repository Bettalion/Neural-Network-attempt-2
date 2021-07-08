import random
import math

class Model:
  def __init__(self):
    self.initWB()
  def initWB(self): # add a value from the normal distribution: mean 0,std 1
    #print(random.normalvariate(0,1))
    self.w1,self.w2,self.w3,self.w4,self.b1,self.b2,self.b3 = [random.normalvariate(0,1) for _ in range(7)]
  def displayWB(self):
    print(f'w1:{self.w1}\n\nw2:{self.w2}\n\nw3:{self.w3}\n\nw4:{self.w4}\n\nb1:{self.b1}\n\nb2:{self.b2}\n\nb3:{self.b3}')
  def activationFunc(self,x):
    return max(0,x) #ReLu
    return (1+math.exp(-x))**-1 #Sigmoid
    return (math.log((1+math.exp(x)),math.e)) #SoftPlus
    pass
  def predict(self,x):# single for this architecture
    upper = self.activationFunc(x*self.w1+self.b1)*self.w3
    lower = self.activationFunc(x*self.w2+self.b2)*self.w4
    Result = upper + lower + b3

def getdata():
  f = open('data.txt','r')
  R_data = f.read().strip().split(',')
  datapoints = int(R_data[0])
  data =[[],[]]
  for point in range(1,datapoints+1):
    data[0].append(R_data[point])
    data[1].append(R_data[point+datapoints])
  return data

data = getdata()

Network = Model() 
Network.initWB()

# Network.displayWB()
# print(Network.activationFunc(-8978443))
# print((1+math.exp(-1))**-1,math.e)
