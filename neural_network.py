import numpy as np


import neurons


class Network:
    thetas:list
    deltas:list
    lamb:float=0.0001
    layers:list#contain each neurons
    SYMM_BREAK=1#[-SYM,STM]
    num_input:int
    num_layer:int
    feature:list

    def __init__(self,feature:list) -> None:
        self.feature=feature

        self.num_input=feature[0]
        self.num_layer=len(feature)
        
        self.initinal_thetas(feature)
        self.connect_neurons(feature)
        self.initinal_deltas(feature)


    def initinal_thetas(self,feature:list):#[number input,num neuron in layer2,....,number output]
        arr=[]
        for i in range(1,len(feature)):
            thetas=np.float32(np.random.randint(2,size=(feature[i],feature[i-1]+1)))# +1 is the bias unit
            thetas=thetas*2*self.SYMM_BREAK-self.SYMM_BREAK
            arr.append(thetas)

        
        self.thetas=arr
        #print(*self.thetas)
    
    def initinal_deltas(self,feature:list):
        arr=[]
        for i in range(1,len(feature)):
            deltas=np.zeros((feature[i],feature[i-1]+1))# +1 is the bias unit
            
            arr.append(deltas)

        
        self.deltas=arr
        #print(self.deltas)
        
    def back_propagaton(self,result,expect):# calaulate diiference of >=2 layers' neutons
        for i in range(len(result)):# set the last layer's difference
            self.layers[-1][i].diffference=result[i]-expect[i]
        for neu in self.layers[1][:-1]:
            neu.calculate_difference()

        
        
    def increase_deltas(self):
        for layer_i in range(self.num_layer-1):
            for neu_higher in range(0,len(self.layers[layer_i+1])-1):#-1 : don't care the bias unit
                for neu_lower in range(0,len(self.layers[layer_i])-1):
                    self.deltas[layer_i][neu_higher,neu_lower]+=self.layers[layer_i][neu_lower].a*self.layers[layer_i+1][neu_higher].diffference
        
        


    def changeTheta(self,a,m):#m : number of sample a:learning rate
        #self.gradient_check([2,3],[1])
        derivative=[self.lamb*self.thetas[i]+self.deltas[i]/m for i in range(self.num_layer-1)]
        #print(derivative)
        for i in range(self.num_layer-1):
            #print(self.num_layer,i)
            self.thetas[i]-=a*derivative[i]
        self.initinal_deltas(self.feature)

        

    def pridict(self,data,train=False):#forward
        for i in range(self.num_input):# set the input 
            self.layers[0][i].a=data[i]
        output=[neu.calculate_a() for neu in self.layers[self.num_layer-1]]
        if not train:
            self.set_a_Zero()
        return output[:-1]
        

    def connect_neurons(self,feature:list):
        #print(self.thetas)
        self.layers=[]
        for layer_i in range(self.num_layer):
            arr=[]
            for neu_i in range(feature[layer_i]):# set subNeurons and declear neurons and thetas
                neuron=neurons.Neuron(layer_i,neu_i,0)
                if self.layers:
                    neuron.subNeurons=self.layers[-1]
                    neuron.thetas_forward=self.thetas[layer_i-1][neu_i]
                arr.append(neuron)
            arr.append(neurons.Neuron(layer_i,neu_i+1,1))


            if self.layers:# set postNeurons 
                for next_neu_i in range(len(self.layers[-1])-1):
                    self.layers[-1][next_neu_i].postNeurons=arr
                    
                    self.layers[-1][next_neu_i].thetas_backward=self.thetas[layer_i-1][:,next_neu_i]

            self.layers.append(arr)

    def set_a_Zero(self):#set all neu's a None
        for layer in self.layers:
            for neu in layer[:-1]:
                neu.a=None

    def set_diff_Zero(self):#set all neu's difference None
        for layer in self.layers:
            for neu in layer[:-1]:
                neu.diffference=None
    
    def cost_function(self,predict,expect):
        cost=sum([(expect[i])*np.log(predict[i])+(expect[i]-1)*np.log(1-predict[i]) for i in range(len(predict))])
        
        return cost

    def train(self,samples,expects,a=3,loop:int=1):#a learning rate
        sample_len=len(samples)
        for turn in range(loop):
            for m in (range(sample_len)):
                pridict=self.pridict(samples[m],True)
                #print(pridict)
                self.back_propagaton(pridict,expects[m])
                self.increase_deltas()
                self.changeTheta(a,sample_len)
                self.set_a_Zero()
                self.set_diff_Zero()
                print()
                print(f"第{turn}轮,第{m+1}次梯度下降")
                print(f"预估值为:{pridict},实际值为:{expects[m]}")


    def gradient_check(self,value,expect):#check whether the delta is true
        print(self.deltas)
        self.set_a_Zero()
        self.thetas[0][0,1]-=0.0000001
        #print(self.thetas)
        #print(self.display_neu())
        p1=self.pridict(value)
        cost1=self.cost_function(p1,expect)
        self.thetas[0][0,1]+=0.0000002
        p2=self.pridict(value)
        cost2=self.cost_function(p2,expect)
        print(cost1,cost2,(cost1-cost2)/0.0000002)

    def display_neu(self):# used to show neurons
        for layer in self.layers:
            for neu in layer:
                print(neu)
            print()
            

    

if __name__=="__main__":
    n=Network([3,5,5,4])
    n.display_neu()
    
    n.train([[2,3,4],[4,3,2]],[[1,0,0,0],[0,0,1,0]],loop=20)#梯度下降
    #[[2,3,4],[4,3,2]]是两组样本，[[1,0,0,0],[0,0,1,0]]两组是实际值，result是预估值
    
    