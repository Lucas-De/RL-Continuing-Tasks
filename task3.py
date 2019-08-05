
num_action=3
state_size=10*10

import random
import numpy as np
import copy
import time
import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import readchar


class task3():
    def __init__(self,render=False):
        self.UI=render
        self.size=10
        self.p=self.rain(self.size)
        self.position=int(self.size/2)

        img=copy.deepcopy(self.p.screen)
        img[0][self.position]=0.7
        img.reverse()
        self.state=np.array(img)*255
        self.steps=0
        self.avg_reward=0
    
    def step(self,action):
        self.steps+=1
        if action==0:
            reward=self.go_left()
        elif action==1:
            reward=self.nothing()
        elif action==2:
            reward=self.go_right()
            
        img=copy.deepcopy(self.p.screen)
        img[0][self.position]=0.7
        img.reverse()
        img=np.array(img)*255
        self.state=img
        self.avg_reward+=(1/self.steps)*(reward-self.avg_reward)
        if self.UI:
            self.render(self.steps==1)

        return img, reward
        
    def go_left(self): 
        if self.position>0:
            if self.p.screen[1][self.position-1]==1:
                reward=1
            else:
                reward=0
                
            self.p.forward()
            self.position-=1
            
        else:
            reward=self.nothing()
        return reward
    
    def go_right(self): 
        if self.position<(self.size-1):
            if self.p.screen[1][self.position+1]==1:
                reward=1
            else:
                reward=0
                
            self.p.forward()
            self.position+=1
            
        else:
            reward=self.nothing()

            
        return reward
            
    def nothing(self):
        if self.p.screen[1][self.position]==1:
            reward=1
        else:
            reward=0
        self.p.forward()
        return reward
    
    def render(self,first):
        cvals  = [-1,1]
        colors = ["black","white"]
        norm=plt.Normalize(min(cvals),max(cvals))
        tuples = list(zip(map(norm,cvals), colors))
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)

        a=self.zoom()
        plt.clf()
        plt.imshow(a,cmap=cmap)
        
        prop = fm.FontProperties(fname='PressStart2P.ttf')
        plt.title('average reward = %.3f' % self.avg_reward, fontproperties=prop, size=10, pad=7)
        plt.axis('off')
        if first:
            plt.show(block=False)
        else:
            self.mypause(0.1)

    def mypause(self,interval):
        backend = plt.rcParams['backend']
        if backend in matplotlib.rcsetup.interactive_bk:
            figManager = matplotlib._pylab_helpers.Gcf.get_active()
            if figManager is not None:
                canvas = figManager.canvas
                if canvas.figure.stale:
                    canvas.draw()
                canvas.start_event_loop(interval)
                return

    
    def zoom(self):
        img=self.state
        x=6
        IMG=[]
        for i in img:
            for k in range(x):
                b=[]
                for j in i:
                    b.extend([j]*x)
                IMG.append(b)
        IMG=np.array(IMG)
        return IMG
    
    class rain():
        def __init__(self, size):
            self.size=size
            self.screen= self.init()

        def init(self):
            rain=[self.get_row() for i in range(self.size-1)]+[[0]*self.size]
            return rain

        def forward(self):
            self.screen=self.screen[1:]
            row=self.get_row()
            self.screen.append(row)

        def get_row(self):
            row=[0]*self.size
            if random.random()>0.5:
                row[random.randint(0,self.size-1)]=1
            return row

Env=task3(True)

def main():
    Env.step(0)
    for i in range(1000):
        a=readchar.readchar()
        if a == 'c':
            Env.step(0)
        if a == 'v':
            Env.step(1)
        if a == 'b':
            Env.step(2)


if __name__ == "__main__" :
    main()

