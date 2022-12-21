import numpy as np


class Neuron:
    a:float=None
    z:float
    layer:int
    index:int
    subNeurons:list=None
    postNeurons:list=None
    diffference:float=None
    thetas_forward=None
    thetas_backward=None
    TYPE:str



    def __init__(self,layer,neu_i,typ:bool=0) -> None:
        self.layer=layer
        self.index=neu_i 
        if typ:
            self.TYPE="bias unit" 
            self.a=1
        else :
            self.TYPE="normal unit"

    def calculate_a(self):
        #print(self,str(self.thetas)!="None")
        if str(self.a)=="None":
            z=(self.calculate_z())
            a=1 / ( 1+np.e**(-z) )
            
            self.z,self.a=z,a
            #print()
            return a
        else:
            
            return self.a

        #self.z=self.calculate_z(data,theta)
        

    def calculate_z(self):
        theta=np.matrix(self.thetas_forward)
        value=[neu.calculate_a() for neu in self.subNeurons]
        #print(value)
        value=np.matrix(np.resize(value,(len(self.subNeurons),1)))
        #print(self,theta*value)
        return (theta*value)[0,0]

        #return theta*data

    def calculate_difference(self):# diff=(g'(z)*(a*theta+a*theta....))
        
        if str(self.diffference)=="None":
            theta=np.matrix(self.thetas_backward)
            value=[neu.calculate_difference() for neu in self.postNeurons[:-1]]
            value=np.matrix(np.resize(value,(len(self.postNeurons)-1,1)))
            
            difference=(theta*value)[0,0]*(self.a)*(1-self.a)
            self.diffference=difference
            return self.diffference
        else:
            return self.diffference

    def __repr__(self) -> str:
        return f"neuron in layer:{self.layer} ,index:{self.index},theta:{self.thetas_forward},backward_theta:{self.thetas_backward},type:{self.TYPE},a:{self.a},difference:{self.diffference}"

    