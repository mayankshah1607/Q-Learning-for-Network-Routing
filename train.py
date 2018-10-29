import numpy as np
import matplotlib.pyplot as plt

def get_next_state(state,action):
    
    global row,col,new_row,new_col
    
    if state is not None:
    
        for i in range(len(network_map)):
            if state in network_map[i]:
                row = i
                col = network_map[i].index(state)
        if action==0:
            new_row = row-1
            new_col = col
        if action==1:
            new_row = row+1
            new_col = col
        if action==2:
            new_row = row
            new_col = col -1
        if action==3:
            new_row = row 
            new_col = col+1


        if new_row<0 or new_row>3 or new_col<0 or new_col>3 or new_col is None or new_row is None:
            return (state,state[4],state[4])

        else :
            return (network_map[new_row][new_col],network_map[new_row][new_col][4],network_map[new_row][new_col][4])


def get_data_loss(q_table):
    s = 0
    for i in q_table :
        s += max(i)
    return s


# 16 x 4 matrix for storing optimal Q values
q_learning_table = [ [0 for i in range(4)] for j in range (16) ]

n_episodes = 100000
n_iterations = 50000

discount = 0.98 
alpha = 0.01

#We use this array to keep track of total data packets sent on every episode
total_packets_sent = []

#(n,t,h,s,r)
# n = state number
# t = traffic [1-10]
# h = probablity of attacker [0-1]
# s = network speed at that node [0-10]
# r = reward [0,1]


for i in range(n_episodes):
    
    #On every episode, some parameters keep changing as it is a stochastic environment.
    
    network_map = [
    [
        (1,np.random.randint(1,9),np.random.uniform(0,0.1),np.random.randint(5,6),0),
        (2,np.random.randint(3,10),np.random.uniform(0,0.1),np.random.randint(6,9),0),
        (3,np.random.randint(1,9),np.random.uniform(0,0.1),np.random.randint(3,9),0),
        (4,np.random.randint(9,10),np.random.uniform(0,1),np.random.randint(1,6),0)
    ],
    [
        (5,np.random.randint(1,5),np.random.uniform(0,0.1),np.random.randint(1,3),0),
        (6,np.random.randint(4,10),np.random.uniform(0,0.1),np.random.randint(6,10),0),
        (7,np.random.randint(3,9),np.random.uniform(0,0.1),np.random.randint(2,9),0),
        (8,np.random.randint(1,11),np.random.uniform(0,0.1),np.random.randint(1,6),0)
    ],
    [
        (9,np.random.randint(1,7),np.random.uniform(0,0.1),np.random.randint(3,6),0),
        (10,np.random.randint(5,10),np.random.uniform(0,0.1),np.random.randint(1,4),0),
        (11,np.random.randint(3,7),np.random.uniform(0,0.1),6,np.random.randint(4,8),0),
        (12,np.random.randint(1,9),np.random.uniform(0,0.1),np.random.randint(1,10),0)
    ],
    [
        (13,np.random.randint(1,5),np.random.uniform(0,0.1),np.random.randint(4,9),0),
        (14,np.random.randint(6,10),np.random.uniform(0,0.1),np.random.randint(1,9),0),
        (15,np.random.randint(3,7),np.random.uniform(0,0.1),np.random.randint(6,9),0),
        (16,np.random.randint(4,11),np.random.uniform(0,0.1),np.random.randint(6,10),1)
    ]
]

    
    
    state = network_map[np.random.randint(0,4)][np.random.randint(0,4)]
    state_number = state[0]-1
    
    for i in range(n_iterations):
        
        action = np.random.randint(0,4)
        state_new,reward,done = get_next_state(state,action)
        
        q_learning_table[state_number][action] = q_learning_table[state_number][action] + alpha * (reward + max(q_learning_table[state_new[0]-1]) - q_learning_table[state_number][action])  * (state[1]/100) * (state[2]) * ( (10 - state[3])/100)
        
        state = state_new

        if done: break
    total_packets_sent.append(get_data_loss(q_learning_table))
    
        
plt.plot([i for i in range(1000)],total_packets_sent[:1000])
plt.xlabel('Number of iterations')
plt.ylabel('Data packets sent')