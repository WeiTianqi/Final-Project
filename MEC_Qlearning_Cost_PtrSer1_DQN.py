

import numpy as np
import math
import random
from sklearn import preprocessing
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import copy


'''
函数名称：Build_scene_vars()
函数功能：构建场景参数，定义全局变量，并进行初始化

'''
def Build_scene_vars():
    #任务到达率
    global Rate_MissionArrival
    Rate_MissionArrival = 10
    
    #宏基站计算能力
    global f_MBS
    f_MBS = 6e10

    #小基站计算能力
    global f_SBS
    f_SBS = 4e10
    
    #Iot计算能力，本地完成任务所需的 CPU 周期
    global f_Iot  
    f_Iot = 1.2e9
    
    #有限状态马尔科夫信道模型
    global SNR_FMSC 
    SNR_FMSC = [8, 7, 5, 3, 4, 6, 9, 12, 11, 10]
    
    #每个任务输入的数据量
    global Circle_CPU 
    Circle_CPU = 1e8
    
    
    #本地计算消耗的能量
    global Energy_local 
    Energy_local = 1e-9
    
    #表示IoT设备单个任务的原始数据量
    global Ita 
    Ita = 1
    
    #为IoT设备空闲时的功率
    global Po 
    Po = 1
    
    #每个时隙最大卸载数量 能力
    global Q_max
    Q_max = 10
    
    #每个时隙最小卸载数量 能力
    global Q_min
    Q_min = 1
    
    #基站个数
    global Num_BS
    Num_BS = 4
    
    #各个基站分配给iot带宽组成的向量
    global Bi_ser 
    Bi_ser = [30, 15, 14, 15.5]
    Bi_ser = np.array(Bi_ser)/4
        
    #初始化表示IoT设备向基站传输数据的功率组成的向量，序号依次为宏基站，小基站i
    global Ptr_ser
    Ptr_ser = [0.3, 1, 1.1, 1.4]
    
    #时延权重 
    global lambda_t
    lambda_t = 0.5
    
    #能耗权重
    global lambda_e
    lambda_e = 1-lambda_t
    
    #Q-learning 参数
    global gamma
    gamma = 0.9
    
    global alpha
    alpha = 0.6
        
    #初始化各个基站的SNR组成的向量，序号依次为宏基站，小基站i
    global SNR_ser
    SNR_ser = [8,6,5,10]
    
    #初始化信道SNR的右转概率矩阵，序号依次为宏基站，小基站i
    global P_right_ser
    P_right_ser = [1, 1, 0, 1] 
        
    #移动用户传输功率
    global P_Iot
    P_Iot = 1
    
    #初始化当前排队数量
    global Q_current
    Q_current = 0
    
    #队列中最大任务数，多余将被溢出丢弃
    global Q_queue_max
    Q_queue_max = 10
    
    
'''
函数名称：Poisson_shift()
函数功能：根据指针，输出泊松序列的数字
输入参数: Num: 输出的泊松到达数
返回参数：pointer: 指针位置

'''
def Poisson_shift(pointer): 
    Ser = [8, 5, 10, 15, 8, 7, 10, 10, 6, 5, 10, 8, 6, 8, 13, 15, 11, 11, 15, 16]
    if isinstance(pointer,(list)):
        pointer=pointer[0]
        
    Num = Ser[(pointer%len(Ser))]
    return Num


'''
函数名称：SNR_transion()
函数功能：根据有限状态马尔科夫信道模型，按概率生成下时刻SNR
输入参数: SNR_current: 前时刻SNR；P_right： 右转概率
返回参数：SNR_next: 下时刻SNR

'''
def SNR_transion(SNR_current, P_right): 
    global SNR_FMSC #有限状态马尔科夫信道模型
    d = SNR_FMSC.index(SNR_current)
    if random.random() <  P_right:
        if d+1 != len(SNR_FMSC):
            SNR_next=SNR_FMSC[d+1]
        else:
            SNR_next=SNR_FMSC[0]
    else:
        if d != 0:
            SNR_next=SNR_FMSC[d-1]
        else:
            SNR_next=SNR_FMSC[len(SNR_FMSC)-1]
    return  SNR_next
            
            
'''
函数名称：SNRstate_shift()
函数功能：根据指针，输出当前各基站的状态向量
输入参数: pointer: 指针位置
返回参数：SNRstate_shift: 输出当前各基站的状态SNR向量

'''
def SNRstate_shift(pointer): 
    global SNR_ser
    global P_right_ser
    global SNR_FMSC
    
    C = copy.deepcopy(SNR_ser)
    for i in range(0,(len(SNR_ser))):
        C[i] = SNR_transion(C[i], P_right_ser[i])
    SNR_ser = C
    return SNR_ser
    




'''
函数名称：Cost_slot_compute()
函数功能：计算当前时隙处理成本
输入参数:  Action： 动作，即选择谁卸载，Action=0,1...i 分别代表由本地处理、宏基站处理、小基站i处理
          SNR_ser： 各个基站的SNR组成的向量，序号依次为宏基站，小基站i
          Bi_ser: 各个基站分配给iot带宽组成的向量，序号依次为宏基站，小基站i
          Ptr_ser: 表示IoT设备向基站传输数据的功率组成的向量，序号依次为宏基站，小基站i
          lambda_t： 时延权重
          lambda_e： 能耗权重
          Q： 本时隙卸载的任务数
返回参数： Cost_slot： 当前时隙处理成本
'''

def Cost_slot_compute(Action, SNR_ser, Bi_ser, Ptr_ser, lambda_t, lambda_e, Q):
    global f_MBS #宏基站计算能力
    global f_SBS #小基站计算能力
    global f_Iot #Iot计算能力，本地完成任务所需的 CPU 周期
    global Circle_CPU  #每个任务输入的数据量
    global Energy_local #本地计算消耗的能量
    global Ita #表示IoT设备单个任务的原始数据量
    global Po #为IoT设备空闲时的功率
    
    #本地卸载
    if np.size(SNR_ser[0])==4:
        SNR_ser= SNR_ser[0]
        
    if isinstance(Action,(list)):
        Action=Action[0]
    if Action == 0:
        Cost_slot = lambda_t*Q*Circle_CPU/f_Iot+lambda_e*Q*Circle_CPU*Energy_local
    else:
        #大基站卸载
        if Action == 1:
#            print(2222)
#            print(Bi_ser[0])
#            print(Ptr_ser[0])
#            print(SNR_ser[0])
#            print(SNR_ser)
            
            r = Bi_ser[0]*math.log2(1+Ptr_ser[0]*SNR_ser[0])
            Cost_slot = lambda_t*(Q*Ita/r+Q*Circle_CPU/f_MBS)+lambda_e*(Ptr_ser[0]*Q*Ita/r+Po*Circle_CPU/f_MBS)
        else:
            i = int(copy.deepcopy(Action))
#            print(333)
#            print(i)
#            print(Bi_ser)
#            print(Ptr_ser)
#            print(SNR_ser)

            r = Bi_ser[i-1]*math.log2(1+Ptr_ser[i-1]*SNR_ser[i-1])
            #r = Bi_ser[i]*math.log2(1+Ptr_ser[i]*SNR_ser[i])
            Cost_slot = lambda_t*(Q*Ita/r+Q*Circle_CPU/f_SBS)+lambda_e*(Ptr_ser[i-1]*Q*Ita/r+Po*Circle_CPU/f_SBS)
    return Cost_slot
     

'''
函数名称：Reward_Qlearn()
函数功能：根据当前时隙处理的成本，计算奖惩值
输入参数:  Action： 动作，即选择谁卸载，Action=0,1...i 分别代表由本地处理、宏基站处理、小基站i处理
          SNR_ser： 各个基站的SNR组成的向量，序号依次为宏基站，小基站i
          Bi_ser: 各个基站分配给iot带宽组成的向量，序号依次为宏基站，小基站i
          Ptr_ser: 表示IoT设备向基站传输数据的功率组成的向量，序号依次为宏基站，小基站i
          lambda_t： 时延权重
          lambda_e： 能耗权重
          Q： 本时隙卸载的任务数
返回参数：Reward：奖惩值

'''
def Reward_Qlearn(Action, SNR_ser, Bi_ser, Ptr_ser, lambda_t, lambda_e, Q): 
    ddd=Cost_slot_compute(Action, SNR_ser,Bi_ser,Ptr_ser, lambda_t, lambda_e, Q)
    if ddd<0.06:
       Reward=0.8
    else:
       if ddd<0.07:
          Reward=0.3;
       else:
           if ddd<0.08:
              Reward=-0.3;
           else:
              Reward=-0.8;
    return Reward
    

'''
函数名称：Smooth()
函数功能：对数据进行平滑
输入参数:  a:  输入数据
        WSZ:  窗口尺寸
返回参数：output：平滑后输出数据

'''
def Smooth(a,WSZ):
    out0 = np.convolve(a,np.ones(WSZ,dtype=int),'valid')/WSZ
    r = np.arange(1,WSZ-1,2)
    start = np.cumsum(a[:WSZ-1])[::2]/r
    stop = (np.cumsum(a[:-WSZ:-1])[::2]/r)[::-1]
    return np.concatenate(( start , out0, stop ))








def Tcegreedy(Ts, rolestate, QNet, psx, pst):
    global Q_max  # 每个时隙最大卸载数量 能力
    global Q_min  # 每个时隙最小卸载数量 能力
    global Num_BS  # 基站个数

    P_e=random.random()
#    print(111111111111111111111111111111111)
#    print(rolestate)
    
    if P_e<0.05:
        #进行探索,随机卸载
        temp = random.sample(range(0, Num_BS + 1),1)
        if isinstance(temp, (list)):
            temp = temp[0]
        num = temp
        
        temp = random.sample(range(1, min(rolestate[0],Q_max)+1),1)
        if isinstance(temp, (list)):
            temp = temp[0]
        Q = temp
        temp1 = copy.deepcopy(rolestate)
        temp1.append(num)
        temp1.append(Q)
        temp1 = np.array(temp1)
        temp1 = temp1.reshape(-1,1)
#        temp1 = rolestate.append(num)
#        temp2 = temp1.append(Q)
#        print(666666666666666666)
#        print(temp1)
#        print(temp1.shape)
#        print(type(temp1))
        x2 = psx.fit_transform(temp1)
        
        Qmax = QNet.predict(x2.T)  # 模型预测    
        action = num 
        Q_now = Qmax 
    else:
        #输出最大的Q值动作
        L_temp = len(rolestate)
#        print(type(L_temp))
        Input = np.zeros((len(rolestate)+2,len(range(Q_min, min(rolestate[0],Q_max)+1))))
        Qmax = -50
        action = 0
        Q = 1
        for i in range(0, Num_BS+1):
            
            for j in range(max(Q_min, 1), min(rolestate[0],Q_max)+1):
                temp1 = copy.deepcopy(rolestate)
                temp1.append(i)
                temp1.append(j)
                temp1 = np.array(temp1)
                temp1 = temp1.reshape(-1,1)
#                print(666666666666666666)
#                print(temp1)
#                print(temp1.shape)
#                print(type(temp1))
                x2 = psx.fit_transform(temp1)
                Q0 = QNet.predict(x2.T)  # 模型预测 
#                print(666666666666666666)
#                print(Qmax)
#                print(type(Qmax))
#                print(Q0)
#                print(type(Q0))
                if Qmax <= Q0:
                    Qmax = Q0
                    action = i
                    Q = j
      
    Q_now=Qmax
    return action, Q, Q_now



'''
函数名称：PointerMove()
函数功能：返回指针位置
输入参数:  Memopointer: 指针当前位置
                S_memo: 存储池尺寸
返回参数：  Newpointer：更新后的指针位置

'''
def PointerMove(Memopointer, S_memo):
    # S_memo已设置为常数
    if Memopointer < S_memo-1:
        Newpointer=Memopointer+1
    else:
        Newpointer=1
    return Newpointer



'''
函数名称：CalculationQtarget()
函数功能：计算Q_target
输入参数:       Input： 输入的训练数据
                QNet： 当前的网络
            Cost_ser:  成本序列
               Q_ser:  Q序列
                 psx:  训练数据的归一化函数类
                 pst:  预测数据的归一化函数类
返回参数：  Q_target : 目标的Q

'''
def CalculationQtarget(Input, QNet, Cost_ser, Q_ser, psx, pst):
    global gamma
    global Q_max  #每个时隙最大卸载数量 能力
    global Q_min  #每个时隙最小卸载数量 能力
    global Num_BS  #基站个数
    global Bi_ser # 各个基站分配给iot带宽组成的向量，序号依次为宏基站，小基站i
    global Ptr_ser # 表示IoT设备向基站传输数据的功率组成的向量，序号依次为宏基站，小基站i
    global lambda_t #时延权重
    global lambda_e #能耗权重

    #Input为5*N_Batch型矩阵
    S_in = Input.shape
    H_in = S_in[0] #求总共的行数
    L_in = S_in[1] #求总共的列数
    
    #先定义最大Q_next 向量
    Q_next = -10 * np.ones((1,L_in))
    
    #定义reward 向量
    reward = np.zeros((1,L_in)) 
    for i in range(0, L_in):
        #对于每个Q_next，都是通过遍历计算得到
#        print(111111111111111111)
#        print(H_in)
#        print(i)
        Mat = Input[0 : int(H_in/2)-1, i]
        Act = 0 
        Q = 1 
        for m in range(0, Num_BS+1):
            temp3 = min(Mat[0],Q_max)
#            print(22222222222222222222222)
#            print(temp3)
#            print(type(temp3))
            for n in range(Q_min, (1+int(temp3))):
                temp1 = copy.deepcopy(Mat)
                temp = list(temp1)
#                print(44444444444444444444444)
#                print(temp)
#                print(type(temp))
                temp.append(m)
                temp.append(n)
                Mat_total = copy.deepcopy(temp)
                
                temp4 = np.array(Mat_total)
                temp4 = temp4.reshape(1,-1)
#                print(55555555555555555555555)
#                print(temp4)
#                print(type(temp4))
                
                x2 = psx.fit_transform(temp4)
#                print(66666666666666666)
#                print(x2)
#                print(type(x2))
#                print(x2.shape)
                
                Q0 = QNet.predict(x2)  # 模型预测 
#                print(777777777)
#                print(Q0)
#                print(type(Q0))
                temp = list(Q0)
                Q0 = temp[0]
#                print(88888888888)
#                print(type(Q_next))
#                print(Q_next)
##                print(Q_next.shape)
#                print(i)
                if Q_next[0, i] <= Q0:
                    Q_next[0, i] = copy.deepcopy(Q0)
                    Act = copy.deepcopy(m)
                    Q = copy.deepcopy(n)
                
        l = len(Q_ser)
        a = Cost_slot_compute(Act, Mat[1:len(Mat)], Bi_ser, Ptr_ser, lambda_t, lambda_e, Q)/Q
        b = sum(Cost_ser) / len(Q_ser)
    
        if Cost_slot_compute(Act, Mat[1:len(Mat)], Bi_ser, Ptr_ser, lambda_t, lambda_e, Q) / Q < sum(Cost_ser) / len(Cost_ser):
#            print(999999999999999)
#            print(reward)
#            print(type(reward))
#            print(reward.shape)
            
            reward[0, i] = 0.1
        else:
            reward[0, i] = -0.1
#    print(111111111111111111)
#    print(Q_next)
#    print(type(Q_next))
#    print(Q_next.shape)
#    temp = Q_next.reshape(-1, 1)
#    print(temp.shape)
    temp = Q_next[0, :]
#    print(temp)
#    print(type(temp))
#    print(temp.shape)
#    CC = pst.inverse_transform(temp)
    Q_target = reward + gamma * Q_next
    return Q_target










'''
函数名称：MEC_Qlearning_Unload()
函数功能：采用Q学习算法进行卸载决策
输入参数: 各全局变量
返回参数：仿真结果图

'''


def MEC_Qlearning_Unload(x):   
    Build_scene_vars()
    global Ptr_ser  
    Ptr_ser[0] = x
    #设置起始步骤、观察期、训练期
    
    step = 1
    T_obs = 120
    T_episode = 4000
    
    #Q_learning参数初始化
    
    #初始化当前到达任务数状态指针
    pointer_arrive = 0
    
    #初始化当前信噪比状态指针
    pointer_SNRstate = 0
    
    #总共状态个数
    Num_state = (Num_BS+1)*len(range(0,Q_queue_max+1))*(len(SNR_FMSC))
    
    #初始化Q表，为二维矩阵,以左侧行为例，序号依次为 上一时刻的action，当前任务队列数和状态序号组成的三位数，
    #行号计算=上一时刻的action*110+当前任务队列数*10+状态序号
    Q_table = np.zeros((Num_state, Num_state))
    
    #临时文件存放序号
    temp = 0
    
    #累计器
    cum = 0
    
    #选择的卸载个数
    Q_down_choose = 1
    
    #选择哪个进行卸载
    Act_choose = 0
    
    #当前队列数
    Q_queen = 1
    
    #初始化状态,依次为 上一时刻action，当前队列任务数，当前各基站snr
    #    startstate = [random.sample(range(0,5),1),min([Poisson_shift(pointer_arrive),Q_queue_max]),SNRstate_shift(pointer_SNRstate)]
    startstate = random.sample(range(0,5),1)+[min([Poisson_shift(pointer_arrive),Q_queue_max])]+SNRstate_shift(pointer_SNRstate)
    Cost_ser = []
    Q_choose_ser = []
    Act_ser = []
    
    rolestate = copy.deepcopy(startstate)
    #开始迭代
    for Ts in range(1,T_obs+1):
        state_ori = copy.deepcopy(rolestate)
        for Tm in range(1,T_episode+1):
            #当前SNR状态指针        
            Current_state_pointer = random.sample(range(1,len(SNR_FMSC)+1),1)
            #当前状态指针
            Current_arrive_pointer = random.sample(range(1,21),1)
            #当前队列数  
            if isinstance(Current_arrive_pointer,(list)):
                Current_arrive_pointer = Current_arrive_pointer[0]
            temp1 =  Poisson_shift(Current_arrive_pointer-1)
            Current_Num = min([temp1, Q_queue_max]) 
            #上一时刻的action
            Act_last = random.sample(range(0,5),1)
            #行号
            if isinstance(Act_last,(list)):
                Act_last = Act_last[0]
            if isinstance(Current_Num,(list)):
                Current_Num = Current_Num[0]
            if isinstance(Current_state_pointer,(list)):
                Current_state_pointer = Current_state_pointer[0]
            row = Act_last*110+Current_Num*10+Current_state_pointer
            Action = random.sample(range(0,5),1)
            
            #实际卸载数
            Q_arrival = random.sample(range(Q_min,Q_max+1),1)
            if isinstance(Q_arrival,(list)):
                Q_arrival = Q_arrival[0]
            Q = min(Q_arrival, Current_Num)
            SNRstate = SNRstate_shift(Current_state_pointer)
            
            Reward = Reward_Qlearn(Action, SNRstate,Bi_ser,Ptr_ser, lambda_t, lambda_e, Q)
            
            # 下一时刻的 
            #下一时刻的队列数，都是以当前时刻新进入后的累积计算
            if isinstance(Current_arrive_pointer,(list)):
                Current_arrive_pointer = Current_arrive_pointer[0]
            Next_Num  = Current_Num-Q+Poisson_shift(Current_arrive_pointer+1)
            if Next_Num > Q_queue_max:
                Next_Num = copy.deepcopy(Q_queue_max)
            Next_state_pointer = Current_state_pointer+1
            if Next_state_pointer > 10:
                Next_state_pointer = 1
            if isinstance(Action,(list)):
                Action = Action[0]
            row_next = Action*110+Next_Num*10+Next_state_pointer
            
            # 下下时刻的
            #下下时刻的状态指针    
            Nextnext_state_pointer = Next_state_pointer+1
            if Nextnext_state_pointer > 10:
                Nextnext_state_pointer = 1
            Qmax = -50
            for act_ in range(0,5):
                for i in range(1,Next_Num+1):
                    if min(Next_Num-i+Poisson_shift(Current_arrive_pointer+2),Q_queue_max) >= Q_queue_max:
                        Q_next=copy.deepcopy(Q_queue_max)
                    else:
                        Q_next = Next_Num-i+Poisson_shift(Current_arrive_pointer+2)
                    if Qmax <= Q_table[row-1,act_*110+i*10+(pointer_SNRstate%10)]:
                        Qmax = Q_table[row-1,act_*110+i*10+(pointer_SNRstate%10)]
                        #选择的卸载个数
                        Q_down_choose = copy.deepcopy(i)
                        #选择哪个进行卸载
                        Act_choose = copy.deepcopy(act_)
                        #下一时刻队列数
                        Q_queen = copy.deepcopy(Q_next)
            # 利用Bellman's equation更新Q值表 
            Q_table[row-1,row_next-1] = (1-alpha)*Q_table[row-1,row_next-1] + alpha*(Reward + gamma*Qmax)   
            if sum(sum(abs(Q_table-temp))) < 0.000001:
                cum = cum+1
                if cum > 1000:
                    break
            temp = copy.deepcopy(Q_table)
        #当前时刻行号
        row = rolestate[0]*110+rolestate[1]*10+pointer_SNRstate%10+1
        #下时刻信干比序号
        pointer_SNRstate = pointer_SNRstate+1
        #下时刻到达任务数序号
        pointer_arrive = pointer_arrive+1
        #下时刻到达任务数
        Q_arrive = min([Poisson_shift(pointer_arrive),Q_queue_max])
        
        Qmax = -50
        for act_ in range(0,5):
            for i in range(1,(rolestate[1]+1)):
                if rolestate[1]-i+Q_arrive >= Q_queue_max:
                    Q_next = Q_queue_max
                else:
                    #Q_next是可以卸载的个数
                    Q_next = rolestate[1]-i+Q_arrive
                   
                    
                if Qmax <= Q_table[row-1,act_*110+Q_next*10+pointer_SNRstate%10]:
                    print(4444)
                    print(np.size(Q_table))
                    print(row)
                    print(act_*110+Q_next*10+pointer_SNRstate%10+1)
                    Qmax = Q_table[row-1,act_*110+Q_next*10+pointer_SNRstate%10]
                    #选择的卸载个数
                    Q_down_choose = copy.deepcopy(i)
                    #选择哪个进行卸载
                    Act_choose = copy.deepcopy(act_)
                    #下一时刻队列数
                    Q_queen = copy.deepcopy(Q_next)
        bb = Cost_slot_compute(Act_choose, rolestate[2:6],Bi_ser,Ptr_ser, lambda_t, lambda_e, Q_down_choose)/Q_down_choose
        Cost_ser.append(bb)            
    #    Cost_ser=[Cost_ser, Cost_slot_compute(Act_choose, rolestate[2:6],Bi_ser,Ptr_ser, lambda_t, lambda_e, Q_down_choose)/Q_down_choose]
        Q_choose_ser.append(Q_down_choose)
    #    Q_choose_ser=[Q_choose_ser,Q_down_choose]
        Act_ser.append(Act_choose)
    #    Act_ser=[Act_ser,Act_choose]
        rolestate = [Act_choose, Q_queen, SNRstate_shift(pointer_SNRstate)]
        
    fig1 = plt.figure(figsize=(10,10))
    #plt.subplot(2,1,1)
    D=Smooth(Cost_ser,49)
    plt.plot(range(0,len(D)), D, color='0.05')
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude')
    plt.title('Mixed 1st Sound')
    plt.show()
    return Cost_ser








'''
函数名称：MEC_Random_Unload()
函数功能：采用卸载决策
输入参数: 各全局变量
返回参数：仿真结果图

'''

def MEC_Random_Unload(x):   
    Build_scene_vars()
    global Ptr_ser  
    Ptr_ser[0] = x
    #设置起始步骤、观察期、训练期
    step = 1
    T_obs = 120
    T_episode = 4000
    
    #Q_learning参数初始化
    
    #初始化当前到达任务数状态指针
    pointer_arrive = 0
    
    #初始化当前信噪比状态指针
    pointer_SNRstate = 0
    
    #总共状态个数
    Num_state = (Num_BS+1)*len(range(0,Q_queue_max+1))*(len(SNR_FMSC))
    
    #初始化Q表，为二维矩阵,以左侧行为例，序号依次为 上一时刻的action，当前任务队列数和状态序号组成的三位数，
    #行号计算=上一时刻的action*110+当前任务队列数*10+状态序号
    Q_table = np.zeros((Num_state, Num_state))
    
    #临时文件存放序号
    temp = 0
    
    #累计器
    cum = 0
    
    #选择的卸载个数
    Q_down_choose = 1
    
    #选择哪个进行卸载
    Act_choose = 0
    
    #当前队列数
    Q_queen = 1
    
    #初始化状态,依次为 上一时刻action，当前队列任务数，当前各基站snr
    #    startstate = [random.sample(range(0,5),1),min([Poisson_shift(pointer_arrive),Q_queue_max]),SNRstate_shift(pointer_SNRstate)]
    startstate = random.sample(range(0,5),1)+[min([Poisson_shift(pointer_arrive),Q_queue_max])]+SNRstate_shift(pointer_SNRstate)
    Cost_ser = []
    Q_choose_ser = []
    Act_ser = []
    
    rolestate = copy.deepcopy(startstate)
    #开始迭代
    for Ts in range(1,T_obs+1):
        state_ori = copy.deepcopy(rolestate)
        for Tm in range(1,T_episode+1):
            #当前SNR状态指针        
            Current_state_pointer = random.sample(range(1,len(SNR_FMSC)+1),1)
            #当前状态指针
            Current_arrive_pointer = random.sample(range(1,21),1)
            #当前队列数  
            if isinstance(Current_arrive_pointer,(list)):
                Current_arrive_pointer = Current_arrive_pointer[0]
            temp1 =  Poisson_shift(Current_arrive_pointer-1)
            Current_Num = min([temp1, Q_queue_max]) 
            #上一时刻的action
            Act_last = random.sample(range(0,5),1)
            #行号
            if isinstance(Act_last,(list)):
                Act_last = Act_last[0]
            if isinstance(Current_Num,(list)):
                Current_Num = Current_Num[0]
            if isinstance(Current_state_pointer,(list)):
                Current_state_pointer = Current_state_pointer[0]
            row = Act_last*110+Current_Num*10+Current_state_pointer
            Action = random.sample(range(0,5),1)
            
            #实际卸载数
            Q_arrival = random.sample(range(Q_min,Q_max+1),1)
            if isinstance(Q_arrival,(list)):
                Q_arrival = Q_arrival[0]
            Q = min(Q_arrival, Current_Num)
            SNRstate = SNRstate_shift(Current_state_pointer)
            
            Reward = Reward_Qlearn(Action, SNRstate,Bi_ser,Ptr_ser, lambda_t, lambda_e, Q)
            
            # 下一时刻的 
            #下一时刻的队列数，都是以当前时刻新进入后的累积计算
            if isinstance(Current_arrive_pointer,(list)):
                Current_arrive_pointer = Current_arrive_pointer[0]
            Next_Num  = Current_Num-Q+Poisson_shift(Current_arrive_pointer+1)
            if Next_Num > Q_queue_max:
                Next_Num = copy.deepcopy(Q_queue_max)
            Next_state_pointer = Current_state_pointer+1
            if Next_state_pointer > 10:
                Next_state_pointer = 1
            if isinstance(Action,(list)):
                Action = Action[0]
            row_next = Action*110+Next_Num*10+Next_state_pointer
            
            # 下下时刻的
            #下下时刻的状态指针    
            Nextnext_state_pointer = Next_state_pointer+1
            if Nextnext_state_pointer > 10:
                Nextnext_state_pointer = 1
            Qmax = -50
            for act_ in range(0,5):
                for i in range(1,Next_Num+1):
                    if min(Next_Num-i+Poisson_shift(Current_arrive_pointer+2),Q_queue_max) >= Q_queue_max:
                        Q_next=copy.deepcopy(Q_queue_max)
                    else:
                        Q_next = Next_Num-i+Poisson_shift(Current_arrive_pointer+2)
                    if Qmax <= Q_table[row-1,act_*110+i*10+(pointer_SNRstate%10)]:
                        Qmax = Q_table[row-1,act_*110+i*10+(pointer_SNRstate%10)]
                        #选择的卸载个数
                        Q_down_choose = copy.deepcopy(i)
                        #选择哪个进行卸载
                        Act_choose = copy.deepcopy(act_)
                        #下一时刻队列数
                        Q_queen = copy.deepcopy(Q_next)
            # 利用Bellman's equation更新Q值表 
            Q_table[row-1,row_next-1] = (1-alpha)*Q_table[row-1,row_next-1] + alpha*(Reward + gamma*Qmax)   
            if sum(sum(abs(Q_table-temp))) < 0.000001:
                cum = cum+1
                if cum > 1000:
                    break
            temp = copy.deepcopy(Q_table)
        #当前时刻行号
        row = rolestate[0]*110+rolestate[1]*10+pointer_SNRstate%10+1
        #下时刻信干比序号
        pointer_SNRstate = pointer_SNRstate+1
        #下时刻到达任务数序号
        pointer_arrive = pointer_arrive+1
        #下时刻到达任务数
        Q_arrive = min([Poisson_shift(pointer_arrive),Q_queue_max])
        
        Qmax = -50
        for act_ in range(0,5):
            for i in range(1,(rolestate[1]+1)):
                if rolestate[1]-i+Q_arrive >= Q_queue_max:
                    Q_next = Q_queue_max
                else:
                    #Q_next是可以卸载的个数
                    Q_next = rolestate[1]-i+Q_arrive
                   
                    
                if Qmax <= Q_table[row-1,act_*110+Q_next*10+pointer_SNRstate%10]:
#                    print(4444)
#                    print(np.size(Q_table))
#                    print(row)
#                    print(act_*110+Q_next*10+pointer_SNRstate%10+1)
                    Qmax = Q_table[row-1,act_*110+Q_next*10+pointer_SNRstate%10]
                    #选择的卸载个数
                    Q_down_choose = copy.deepcopy(i)
                    #选择哪个进行卸载
                    Act_choose = copy.deepcopy(act_)
                    #下一时刻队列数
                    Q_queen = copy.deepcopy(Q_next)
        act_ =  random.sample(range(0,5),1)
        if isinstance(act_,(list)):
            act_ = act_[0]
        Act_choose = copy.deepcopy(act_)
        ccc = random.sample(range(1,Q+1),1)
        if isinstance(ccc,(list)):
            ccc = ccc[0]
#        if isinstance(rolestate,(list)):
#            rolestate = rolestate[0]
        Q_down_choose = copy.deepcopy(ccc)
        print(5555)
        print(Q)
        print(Act_choose)
        print(Q_down_choose)
        print(rolestate)
        bb = Cost_slot_compute(Act_choose, rolestate[2:6],Bi_ser,Ptr_ser, lambda_t, lambda_e, Q_down_choose)/Q_down_choose
        Cost_ser.append(bb)            
    #    Cost_ser=[Cost_ser, Cost_slot_compute(Act_choose, rolestate[2:6],Bi_ser,Ptr_ser, lambda_t, lambda_e, Q_down_choose)/Q_down_choose]
        Q_choose_ser.append(Q_down_choose)
    #    Q_choose_ser=[Q_choose_ser,Q_down_choose]
        Act_ser.append(Act_choose)
    #    Act_ser=[Act_ser,Act_choose]
        rolestate = [Act_choose, Q_queen, SNRstate_shift(pointer_SNRstate)]
        
    fig1 = plt.figure(figsize=(10,10))
    #plt.subplot(2,1,1)
    D=Smooth(Cost_ser,9)
    plt.plot(range(0,len(D)), D, color='0.5')
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude')
    plt.title('Mixed 1st Sound')
    plt.show()
    return Cost_ser








'''
函数名称：MEC_Local_Unload()
函数功能：采用卸载决策
输入参数: 各全局变量
返回参数：仿真结果图

'''

def MEC_Local_Unload(x):   
    Build_scene_vars()
    global Ptr_ser  
    Ptr_ser[0] = x
    
    #设置起始步骤、观察期、训练期
    step = 1
    T_obs = 120
    T_episode = 4000
    
    #Q_learning参数初始化
    
    #初始化当前到达任务数状态指针
    pointer_arrive = 0
    
    #初始化当前信噪比状态指针
    pointer_SNRstate = 0
    
    #总共状态个数
    Num_state = (Num_BS+1)*len(range(0,Q_queue_max+1))*(len(SNR_FMSC))
    
    #初始化Q表，为二维矩阵,以左侧行为例，序号依次为 上一时刻的action，当前任务队列数和状态序号组成的三位数，
    #行号计算=上一时刻的action*110+当前任务队列数*10+状态序号
    Q_table = np.zeros((Num_state, Num_state))
    
    #临时文件存放序号
    temp = 0
    
    #累计器
    cum = 0
    
    #选择的卸载个数
    Q_down_choose = 1
    
    #选择哪个进行卸载
    Act_choose = 0
    
    #当前队列数
    Q_queen = 1
    
    #初始化状态,依次为 上一时刻action，当前队列任务数，当前各基站snr
    #    startstate = [random.sample(range(0,5),1),min([Poisson_shift(pointer_arrive),Q_queue_max]),SNRstate_shift(pointer_SNRstate)]
    startstate = random.sample(range(0,5),1)+[min([Poisson_shift(pointer_arrive),Q_queue_max])]+SNRstate_shift(pointer_SNRstate)
    Cost_ser = []
    Q_choose_ser = []
    Act_ser = []
    
    rolestate = copy.deepcopy(startstate)
    #开始迭代
    for Ts in range(1,T_obs+1):
        state_ori = copy.deepcopy(rolestate)
        for Tm in range(1,T_episode+1):
            #当前SNR状态指针        
            Current_state_pointer = random.sample(range(1,len(SNR_FMSC)+1),1)
            #当前状态指针
            Current_arrive_pointer = random.sample(range(1,21),1)
            #当前队列数  
            if isinstance(Current_arrive_pointer,(list)):
                Current_arrive_pointer = Current_arrive_pointer[0]
            temp1 =  Poisson_shift(Current_arrive_pointer-1)
            Current_Num = min([temp1, Q_queue_max]) 
            #上一时刻的action
            Act_last = random.sample(range(0,5),1)
            #行号
            if isinstance(Act_last,(list)):
                Act_last = Act_last[0]
            if isinstance(Current_Num,(list)):
                Current_Num = Current_Num[0]
            if isinstance(Current_state_pointer,(list)):
                Current_state_pointer = Current_state_pointer[0]
            row = Act_last*110+Current_Num*10+Current_state_pointer
            Action = random.sample(range(0,5),1)
            
            #实际卸载数
            Q_arrival = random.sample(range(Q_min,Q_max+1),1)
            if isinstance(Q_arrival,(list)):
                Q_arrival = Q_arrival[0]
            Q = min(Q_arrival, Current_Num)
            SNRstate = SNRstate_shift(Current_state_pointer)
            
            Reward = Reward_Qlearn(Action, SNRstate,Bi_ser,Ptr_ser, lambda_t, lambda_e, Q)
            
            # 下一时刻的 
            #下一时刻的队列数，都是以当前时刻新进入后的累积计算
            if isinstance(Current_arrive_pointer,(list)):
                Current_arrive_pointer = Current_arrive_pointer[0]
            Next_Num  = Current_Num-Q+Poisson_shift(Current_arrive_pointer+1)
            if Next_Num > Q_queue_max:
                Next_Num = copy.deepcopy(Q_queue_max)
            Next_state_pointer = Current_state_pointer+1
            if Next_state_pointer > 10:
                Next_state_pointer = 1
            if isinstance(Action,(list)):
                Action = Action[0]
            row_next = Action*110+Next_Num*10+Next_state_pointer
            
            # 下下时刻的
            #下下时刻的状态指针    
            Nextnext_state_pointer = Next_state_pointer+1
            if Nextnext_state_pointer > 10:
                Nextnext_state_pointer = 1
            Qmax = -50
            for act_ in range(0,5):
                for i in range(1,Next_Num+1):
                    if min(Next_Num-i+Poisson_shift(Current_arrive_pointer+2),Q_queue_max) >= Q_queue_max:
                        Q_next=copy.deepcopy(Q_queue_max)
                    else:
                        Q_next = Next_Num-i+Poisson_shift(Current_arrive_pointer+2)
                    if Qmax <= Q_table[row-1,act_*110+i*10+(pointer_SNRstate%10)]:
                        Qmax = Q_table[row-1,act_*110+i*10+(pointer_SNRstate%10)]
                        #选择的卸载个数
                        Q_down_choose = copy.deepcopy(i)
                        #选择哪个进行卸载
                        Act_choose = copy.deepcopy(act_)
                        #下一时刻队列数
                        Q_queen = copy.deepcopy(Q_next)
            # 利用Bellman's equation更新Q值表 
            Q_table[row-1,row_next-1] = (1-alpha)*Q_table[row-1,row_next-1] + alpha*(Reward + gamma*Qmax)   
            if sum(sum(abs(Q_table-temp))) < 0.000001:
                cum = cum+1
                if cum > 1000:
                    break
            temp = copy.deepcopy(Q_table)
        #当前时刻行号
        row = rolestate[0]*110+rolestate[1]*10+pointer_SNRstate%10+1
        #下时刻信干比序号
        pointer_SNRstate = pointer_SNRstate+1
        #下时刻到达任务数序号
        pointer_arrive = pointer_arrive+1
        #下时刻到达任务数
        Q_arrive = min([Poisson_shift(pointer_arrive),Q_queue_max])
        
        Qmax = -50
        for act_ in range(0,5):
            for i in range(1,(rolestate[1]+1)):
                if rolestate[1]-i+Q_arrive >= Q_queue_max:
                    Q_next = Q_queue_max
                else:
                    #Q_next是可以卸载的个数
                    Q_next = rolestate[1]-i+Q_arrive
                   
                    
                if Qmax <= Q_table[row-1,act_*110+Q_next*10+pointer_SNRstate%10]:
#                    print(4444)
#                    print(np.size(Q_table))
#                    print(row)
#                    print(act_*110+Q_next*10+pointer_SNRstate%10+1)
                    Qmax = Q_table[row-1,act_*110+Q_next*10+pointer_SNRstate%10]
                    #选择的卸载个数
                    Q_down_choose = copy.deepcopy(i)
                    #选择哪个进行卸载
                    Act_choose = copy.deepcopy(act_)
                    #下一时刻队列数
                    Q_queen = copy.deepcopy(Q_next)
        act_ =  0
        if isinstance(act_,(list)):
            act_ = act_[0]
        Act_choose = copy.deepcopy(act_)
        ccc = random.sample(range(1,Q+1),1)
        if isinstance(ccc,(list)):
            ccc = ccc[0]
#        if isinstance(rolestate,(list)):
#            rolestate = rolestate[0]
        Q_down_choose = copy.deepcopy(ccc)
#        print(5555)
#        print(Q)
#        print(Act_choose)
#        print(Q_down_choose)
#        print(rolestate)
        bb = Cost_slot_compute(Act_choose, rolestate[2:6],Bi_ser,Ptr_ser, lambda_t, lambda_e, Q_down_choose)/Q_down_choose
        Cost_ser.append(bb)            
    #    Cost_ser=[Cost_ser, Cost_slot_compute(Act_choose, rolestate[2:6],Bi_ser,Ptr_ser, lambda_t, lambda_e, Q_down_choose)/Q_down_choose]
        Q_choose_ser.append(Q_down_choose)
    #    Q_choose_ser=[Q_choose_ser,Q_down_choose]
        Act_ser.append(Act_choose)
    #    Act_ser=[Act_ser,Act_choose]
        rolestate = [Act_choose, Q_queen, SNRstate_shift(pointer_SNRstate)]
        
    fig1 = plt.figure(figsize=(10,10))
    #plt.subplot(2,1,1)
    D=Smooth(Cost_ser,9)
    plt.plot(range(0,len(D)), D, color='0.3')
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude')
    plt.title('Mixed 1st Sound')
    plt.show()
    return Cost_ser








'''
函数名称：MEC_DQN_Unload()
函数功能：采用DQN算法进行卸载决策
输入参数: 各全局变量
返回参数：仿真结果图

'''

def MEC_DQN_Unload(x):   
    Build_scene_vars()
    global Ptr_ser  
    Ptr_ser[0] = x
    #设置起始步骤、观察期、训练期
    T_obs=30                     
    T_train=90
    T_episode=40
    
    #DQN评估网络初始化 
    QNet_eval = MLPRegressor(hidden_layer_sizes=(40, 40), random_state=10, activation='relu',
                             learning_rate_init=0.1, batch_size='auto', solver='lbfgs')  # BP神经网络回归模型
    
    #状态主要由以下部分组成[当前Iot设备排队任务数， 大基站当前SNR，小基站1当前SNR，小基站2当前SNR，...小基站i当前SNR，...]
    #前7行为输入，依次为当前状态，选择的Action和卸载个数Q；最后1行为目标输出，神经网络输出的Q值 目前就是将平均成本取负作为输出
    Iniset = np.zeros((1+Num_BS+2+1,400))
    for i in range(0, 400):
        
        #当前任务排队数
        temp = random.sample(range(0, Q_queue_max+1),1)
        if isinstance(temp, (list)):
            temp = temp[0]
        Iniset[0, i] = temp  
        
        #SNR向量
        Iniset[1:Num_BS+1, i] =  random.sample(range(3, 13), len(SNR_ser))
        
        #选择的Action
        temp = random.sample(range(0, Num_BS+1),1)
        if isinstance(temp, (list)):
            temp = temp[0]
        Iniset[1+Num_BS,i] = temp  
        
        #卸载个数Q
        temp = random.sample(range(0, int(min(Iniset[0, i], Q_max))+1),1)
        if isinstance(temp, (list)):
            temp = temp[0]
        if temp < 1:
            temp = 1
        Iniset[2+Num_BS,i] = temp
        
        #调用Cost_slot_compute(Action, SNR_ser, Bi_ser, Ptr_ser, lambda_t, lambda_e, Q)
        ddd = Cost_slot_compute(Iniset[1+Num_BS,i], Iniset[1:1+Num_BS,i], Bi_ser, Ptr_ser, lambda_t, lambda_e, Iniset[2+Num_BS, i]) / (Iniset[2+Num_BS ,i])
        if ddd<0.07:
            Iniset[3+Num_BS,i] = 0.8
        else:
            if ddd < 0.08:
                Iniset[3+Num_BS,i] = 0.3
            else:
                if ddd < 0.09:
                    Iniset[3+Num_BS,i] = -0.3
                else:
                    Iniset[3+Num_BS,i] = -0.8
#        print(Iniset[3+Num_BS,i])
#    print(Iniset.shape)
                    
    #归一化处理
    psx = MinMaxScaler(feature_range=(0, 1))  
    x1 = psx.fit_transform(Iniset[0:3+Num_BS,:])
#    print(1212121212121212123333333333333333333)
#    print(type(Iniset[0:3+Num_BS,:]))
#    print(Iniset[0:3+Num_BS,:])
#    print(Iniset[0:3+Num_BS,:].shape)
#    print(7777777777777777777777777777)
#    print(Iniset[0:3+Num_BS,:].shape)
#    print(type(Iniset[0:3+Num_BS,:]))
    
    pst = MinMaxScaler(feature_range=(0, 1)) 
    temp = copy.deepcopy(Iniset[3+Num_BS,:])
    temp1 = np.array(temp)
    temp1 = temp1.reshape(1,-1)
    t1 = pst.fit_transform(temp1)
#    print(121212121212121212)
#    print(temp1.shape)
#    print(t1.shape)
#    print(22222222222222222)
#    print(type(psx))
#    print(x1)
#    print(t1)
#    print(x1.shape)
#    print(t1.shape)
#    print(111111111111111111111111)
#    print(QNet_eval)
#    print((x1.T).shape)
#    print((t1.T).shape)
    QNet_eval.fit(x1.T, t1.T)  

    #将评估网络赋予目标网络
    QNet_target = copy.deepcopy(QNet_eval)
    
    #Replaymemory初始化：
    S_memo = 400
    
    #1+Num_BS，当前状态； 2+Num_BS：3+Num_BS， action和卸载个数Q； 3+Num_BS+1：3+Num_BS+1+Num_BS：下一时隙状态
    Rmemo=np.zeros((2+2*Num_BS+2, S_memo))
    
    #Replay memory的写入指针
    Memopointer=1 
    
    # 学习参数初始化
    gamma = 0.9 
    nBatch = 5
    T_gap = 5   #20,25,30
    T_renew = 3 * T_gap
        
    #记录列表初始化
    #成本记录
    Cost_ser=[]
    #Ｑ值记录
    Q_ser=[]
    #动作记录
    Act_ser=[]
    
    #观察期：此期间不更新Qnet,仅记录replay memory
    #初始化状态,依次为 上一时刻action，当前队列任务数，当前各基站snr
    temp = random.sample(range(3, 12),1)
    if isinstance(temp, (list)):
        temp = temp[0]
    startstate = copy.deepcopy(SNR_ser)
    startstate.insert(0, temp)
#    startstate = SNR_ser.insert(0, poi_l[0])
#    print(555555555555555555)
#    print(startstate)
#    startstate = poi_l.append(SNR_ser)
#    startstate = poi_l+SNR_ser
#    print(type(poi_l))
#    print(type(SNR_ser))
#    print(poi_l)
#    print(SNR_ser)
    step = 1
    rolestate = copy.deepcopy(startstate)
#    print(4444444444444444444)
#    print(startstate)
#    print(rolestate)
   
    for Ts in range(0, T_obs):
        state_ori = copy.deepcopy(rolestate)
        for Tm in range(1, T_episode):
            
            #根据Tcegreedy策略执行动作
#            print(222222222222222)
#            print(rolestate)
#            print(nextstate)
#            print(rolestate)
            [action, Q, Q_now] = Tcegreedy(Ts, rolestate, QNet_eval, psx, pst)
            
            #下时刻队列任务数
            temp = random.sample(range(3, 12),1)
            if isinstance(temp, (list)):
                temp = temp[0]
#            poi = np.random.poisson(Rate_MissionArrival, 1)
#            poi_l = poi.tolist()
            Q_next = rolestate[0] - Q + temp
            if Q_next > Q_queue_max:
                Q_next = copy.deepcopy(Q_queue_max)
            
            #下时刻SNR状态值
            SNR_next = np.zeros((1, len(SNR_ser)))
            for i in range(0, len(SNR_next)):
                SNR_next[i] = SNR_transion(rolestate[i], P_right_ser[i-1])
            
            #下时刻状态值
#            print(33333333333333333)
#            print(Q_next)
#            print(type(Q_next))
#            print(SNR_next)
#            print(type(SNR_next))
#            startstate.insert(0, poi_l[0])
#            nextstate = Q_next.append(SNR_next)
            temp = copy.deepcopy(SNR_next)
            temp = temp.tolist()
            temp = temp[0]
#            print(444444444444444444444444)
#            print(Q_next)
#            print(temp)
            temp.insert(0, Q_next)
            nextstate = copy.deepcopy(temp)
            #Replaymemory记录+指针更新
            temp1 = copy.deepcopy(rolestate)
            temp1.append(action)
#            print(3333333333333)
#            print(Q)
#            print(type(Q))
            
            temp1.append(Q)
#            temp1.append(nextstate)
            temp1 = temp1 + nextstate
#            print(33333333333333333)
#            print(type(action))
#            print(action)
#            print(type(Q))
#            print(Q)
#            print(type(nextstate))
#            print(nextstate)
#            print(temp1)
#            print(type(temp1))
#            print(temp1.shape)
#            print(Rmemo.shape)
#            print(Memopointer)
            Rmemo[:, Memopointer] = list(map(int,temp1))
            Memopointer = PointerMove(Memopointer, S_memo)
            
            #更新位置
            step = step+1
            rolestate = copy.deepcopy(nextstate)
            
            #判断是否跳出本episode
            if len(Cost_ser) > 10:
#                if Cost_slot_compute(action, state_ori[1 : len(state_ori)],Bi_ser,Ptr_ser, lambda_t, lambda_e, Q)/Q < 0.98*sum(Cost_ser)/len(Cost_ser):
                if Cost_slot_compute(action, state_ori[1 : len(state_ori)],Bi_ser,Ptr_ser, lambda_t, lambda_e, Q)/Q < 0.98*sum(Cost_ser)/len(Cost_ser):
                    break
        
        temp = Cost_slot_compute(action, state_ori[1 : len(state_ori)],Bi_ser,Ptr_ser, lambda_t, lambda_e, Q)/Q
#        temp = Cost_slot_compute(action, state_ori[1 : len(state_ori)],Bi_ser,Ptr_ser, lambda_t, lambda_e, Q)/Q
#        print(444444444444444444444444444444)
#        print(temp)
        Cost_ser.append(temp)
        Q_ser.append(Q)
        Act_ser.append(action)

#    print(Cost_ser)
#    print(Q_ser)
#    print(Act_ser)
#    print(len(Cost_ser))
#    print(len(Q_ser))
#    print(len(Act_ser))
    
    # 探索期 探索期开始更新神经网络参数
    Tnode1 = 1 + T_obs
    Tnode2 = T_obs + T_train
    
    #网络训练参数更新
    rolestate = copy.deepcopy(startstate)
    for Ts in range(Tnode1, Tnode2+1):
        state_ori = copy.deepcopy(rolestate)
        for Tm in range(1, T_episode+1):
            
            #根据tcegreedy策略执行动作
            action, Q, Q_now = Tcegreedy(Ts, rolestate, QNet_eval, psx, pst)
            
            #下时刻队列任务数
            temp = random.sample(range(3, 12),1)
            if isinstance(temp, (list)):
                temp = temp[0]
            Q_next = rolestate[0] - Q + temp
            
            #如果任务超出了，就溢出，保留最大排队数
            if Q_next > Q_queue_max:
                Q_next = copy.deepcopy(Q_queue_max)
            
            #下时刻SNR状态值
            SNR_next = np.zeros((1, len(SNR_ser)))
            for i in range(0, len(SNR_next)):
                SNR_next[i]=SNR_transion(rolestate[i], P_right_ser[i-1])
                
            #下时刻状态值
            temp = copy.deepcopy(SNR_next)
            temp = temp.tolist()
            temp = temp[0]
            temp.insert(0, Q_next)
            nextstate = copy.deepcopy(temp)
            
            #Replaymemory记录+指针更新
            temp1 = copy.deepcopy(rolestate)
            temp1.append(action)
            temp1.append(Q)
            temp1 = temp1 + nextstate
            Rmemo[:, Memopointer] = list(map(int,temp1))
            Memopointer = PointerMove(Memopointer, S_memo)
            
            #更新位置
            step = step + 1
            rolestate = copy.deepcopy(nextstate)
            
#            if Cost_slot_compute(action,state_ori(2:end),Bi_ser,Ptr_ser, lambda_t, lambda_e, Q)/Q<0.98*sum(Cost_ser)/length(Cost_ser)
            if len(Cost_ser) > 10:
#                print(22222222222222222222)
#                print(action)
#                print(state_ori[1 : len(state_ori)])
#                print(Q)
#                print(Cost_slot_compute(action, state_ori[1 : len(state_ori)],Bi_ser,Ptr_ser, lambda_t, lambda_e, Q))
#                print(333333333333333333333)
#                print(0.98*sum(Cost_ser)/len(Cost_ser))
#                print(Cost_ser)
#                print(len(Cost_ser))
                if Cost_slot_compute(action, state_ori[1 : len(state_ori)],Bi_ser,Ptr_ser, lambda_t, lambda_e, Q)/Q < 0.98*sum(Cost_ser)/len(Cost_ser):
                    break
            
            #按照T-renew间隔更新估计Q_target的目标神经网络QNet_target
            if step % T_renew == 0:
                QNet_target = copy.deepcopy(QNet_eval)
            
            #按照T_gap的间隔训练估计Q_eval的评估神经网络QNet_eval
            if step % T_gap == 0:
                
                #1. 利用Rmemo生成训练数据级
                #前面行与replaymemory一致，后一行为利用QNet_target计算得到的Q_target
                Trainset = np.zeros((2+2+2*Num_BS+1,nBatch))
                cc = 1
                while cc < nBatch:
                    
                    #随机抽取ReplayMemory中的数据
                    temp = random.sample(range(0, S_memo),1)
                    if isinstance(temp, (list)):
                        temp = temp[0]
                    num1 = copy.deepcopy(temp)
                    if Rmemo[0, num1-1] > 0:
                        Trainset[0:(2 + 2 + 2 * Num_BS), cc] = Rmemo[:, num1-1]
                        cc = cc + 1
                
                #2. 计算Q_target
                Trainset[2+2+2*Num_BS, :] = CalculationQtarget(Trainset[0 : (2+2+2*Num_BS),:], QNet_target, Cost_ser, Q_ser, psx, pst)
                
                #3. 训练QNet_eval
                x2 = psx.fit_transform(Trainset[0:1+Num_BS+2,:])
#                x2 = mapminmax('apply',Trainset(1:1+Num_BS+2,:),psx);
#                t2 = pst.fit_transform(Trainset[2+2+2*Num_BS, :])
                t2 = Trainset[1+Num_BS+2,:]
#                t2 = Trainset(end,:)
                QNet_eval.fit(x2.T, t2.T) 
        temp = Cost_slot_compute(action, state_ori[1 : len(state_ori)],Bi_ser,Ptr_ser, lambda_t, lambda_e, Q)/ Q
        Cost_ser.append(temp)
#        print(55555555555555555)
#        print(temp)
        Q_ser.append(Q)
        Act_ser.append(action)

#    print(Cost_ser)
#    print(Q_ser)
#    print(Act_ser)
#    print(len(Cost_ser))
#    print(len(Q_ser))
#    print(len(Act_ser))
    return Cost_ser    









#主函数入口

def main():
    Ptr_ser1=[0.03, 0.1, 0.3, 1, 3]
    Cost_total=[[0, 0, 0, 0, 0],[0, 0, 0, 0, 0],[0, 0, 0, 0, 0],[0, 0, 0, 0, 0]]
    for kkk in range(0, len(Ptr_ser1)):
        A1=MEC_Qlearning_Unload(Ptr_ser1[kkk])
        A1=Smooth(A1,49)
        Cost_total[0][kkk]=sum(A1)
        A2=MEC_Random_Unload(Ptr_ser1[kkk])
        A2=Smooth(A2,49)
        Cost_total[1][kkk]=sum(A2)
        A3=MEC_Local_Unload(Ptr_ser1[kkk])
        A3=Smooth(A3,49)
        Cost_total[2][kkk]=sum(A3)
        
        A4=MEC_DQN_Unload(Ptr_ser1[kkk])
        A4=Smooth(A4,49)
        Cost_total[3][kkk]=sum(A4)
        
    fig5 = plt.figure(figsize=(10,10))
    plt.plot(range(0,5), Cost_total[0], label='Q-learning',color='r')
    plt.plot(range(0,5), Cost_total[1], label='Random',color='g')
    plt.plot(range(0,5), Cost_total[2], label='Local',color='b')
    plt.plot(range(0,5), Cost_total[3], label='Local',color='k')
    plt.xlabel('Transmit power to macro base station')
    plt.ylabel('Cost')
#    plt.title('3种卸载方法对比')
    plt.legend()
    plt.show()
    path='C:/Users/Administrator/Desktop/11/88/'
    fig5.savefig(path+'fig5.png',format='png',dpi=100)  
    
    
    
    
if __name__ == "__main__":
    main()    

    





























