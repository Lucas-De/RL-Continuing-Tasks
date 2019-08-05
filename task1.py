import random
import numpy as np
import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

servers=10
free_proba=0.06
queue_size=10

state_size=servers+1
num_action=2


class task1:
    
    def __init__(self,render=False):
        self.UI=render
        self.customer_types=[1,2,4,8]
        self.server_capacity=servers
        self.occupied=int(self.server_capacity/2)
        self.server={i:(random.choice(self.customer_types) if i<self.occupied else 0) for i in range(10)}
        self.free_proba=free_proba
        self.queue=random.choices(self.customer_types,k=queue_size)
        self.state=self.queue+[self.occupied]
        self.steps=0
        self.avg_reward=0
 
    def free_servers(self):
        number_freed= sum([self.free_proba > random.random() for i in range(self.occupied)])
        self.occupied-=number_freed
        terminated=0
        for s in random.sample(self.server.keys(),len(self.server)):
            if self.server[s]!=0 and terminated<number_freed:
                self.server[s]=0
                terminated+=1

        
    def customer_joins_queue(self):
        self.queue=[random.choice(self.customer_types)]+self.queue

    def step(self,accept):
        self.free_servers()
        self.steps+=1
        
        if accept==1 and self.occupied<self.server_capacity:
            self.occupied+=1
            reward=self.queue.pop(-1)
            for s in list(self.server.keys()):
                if self.server[s]==0:
                    self.server[s]=reward
                    break
        elif accept==0:
            self.queue.pop(-1)
            reward=0
        else:
            self.queue.pop(-1)
            reward=-1
        
        self.customer_joins_queue()
        self.state=self.queue+[self.occupied]

        if self.UI:
            self.render(self.steps==1)

        self.avg_reward+=(1/self.steps)*(reward-self.avg_reward)
        
        return self.state, reward

    def render(self,first):

        cvals  = [0,250,253,255]
        colors = ["black","white",'green','red']
        norm=plt.Normalize(min(cvals),max(cvals))
        tuples = list(zip(map(norm,cvals), colors))
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)

        plt.clf()
        a=self.get_img()
        plt.imshow(a,cmap,vmin=0, vmax=255)
    
        prop = fm.FontProperties(fname='PressStart2P.ttf')
        pos=12
        for i in range(1,10):
            plt.text(pos,157,str(i),fontproperties=prop, size=8,color='white')
            pos+=19
        plt.text(pos-2,157,10,fontproperties=prop, size=8,color='white')
        plt.title('average reward: %.3f'% self.avg_reward,fontproperties=prop, size=10,pad=7)
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

    def get_img(self):
        s=self.state
        rows=[]
        occupied=s[10]

        d={0:0,1:100,2:150,4:200,8:250}   

        block=10
        gap=int(block/2)
        total=200

        white=int((total-block)/2)

        for i in range(10):
            for k in range(block):
                rows.append([0]*white+[d[s[i]]]*block+[0]*white )
            for k in range(gap):
                rows.append([0]*total)

        rows.append([250]*total)
        for i in range(7): rows.append([0]*total)
        rows.append([250]*total)
        for i in range(5): rows.append([0]*total)


        server_bar=self.server.values()
        for i in range(block):
            row=[0]*10
            for k in server_bar: 
                row+=[d[k]]*block+[0]*9

            right_pad=total-len(row)
            row+=[0]*right_pad
            rows.append(row)

        
        for i in range(5): rows.append([0]*total)
    
        return rows