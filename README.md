# REPER on DQN

SI151 Optimization and Machine Learning project: 

**Recent Emphasized Prioritized Experience Replay on Deep Q-Network**

-----------------------------------------------------------------------------------------------------------------

Operation manual for our code

1. To see a quick result curve on Jupyter:  
First, log into http://10.15.89.41:30303/notebooks/REPER-on-DQN/REPER-Pri-Random.ipynb via password 123. (If you fail to connect to the server, please contact chenzhuo@shanghaitech.edu.cn and we will open the port for Jupyter, because we cannot guarantee that the server is always in service.)  
Second, to compare the results with random sampling, prioritized sampling and our REPER model you might want to adjust sampling_method to be one of the three choices.  
Then, click "Cell" -> "Run All", and the curve will show up at the end.  
![image](https://github.com/Hanson-Liuhx/REPER-on-DQN/blob/master/image/Operation%20manual%20for%20code.jpg)  

2. To train 2 players playing Pong Game:  

First, enter the folder REPER-on-DQN/framework.  
Then, to train an agent based on random sampling, run:  
```
    python main.py --sample_method=random
```
To train an agent based on simple prioritized sampling, run:  
```
    python main.py --sample_method=PER
```  
To train an agent based on REPER model, run:  
```
    python main.py --sample_method=REPER
```  
Note that the training process may take about 4 hours on a machine with CUDA. 

3. Prerequisites:  

| Name | Version | Build | Channel |
| ---- | ---- | ---- | --- |
| gym | 0.17.2 | pypi | pypi |
| gym-wrappers | 0.1.0 | pypi_0 | pypi |
| pytorch | 1.4.0 | py3.6_cuda9.2.148_cudnn7.6.3_0 | pytorch |
| torchvision | 0.5.0 | py36_cu92 | pytorch |
| numpy | 1.15.4 | pypi_0 | pypi |
| opencv-python | 4.2.0.34 | pypi_0 | pypi |

------------------------------------------------------------------------------------

Result comparision between two models:  
Win:  
![image](https://github.com/Hanson-Liuhx/REPER-on-DQN/blob/master/image/win.gif)

Lose:  
![image](https://github.com/Hanson-Liuhx/REPER-on-DQN/blob/master/image/lose.gif)
