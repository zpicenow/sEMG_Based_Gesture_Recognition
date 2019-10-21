#!/usr/bin/env python3
# coding: utf-8

# In[1]:


TrainingData_PATH = './training_data/'
TestData_PATH = './test_data/'
gesture1 = 'doubletap_'
gesture2 = 'fist_'
gesture3 = 'spread_'
gesture4 = 'static_'
gesture5 = 'wavein_'
gesture6 = 'waveout_'
txt_format = '.txt'


# In[2]:


def rescaling_max_min(input_list):
    #https://www.zhihu.com/question/20467170
    max_value = max(input_list)
    min_value = min(input_list)
    rescaled_list = [(x-min_value)/(max_value-min_value) for x in input_list]
    return rescaled_list


# In[3]:


def mean_normalization(input_list):
    #https://www.zhihu.com/question/20467170
    import numpy as np
    max_value = max(input_list)
    min_value = min(input_list)
    mean_value = np.mean(input_list)
    mean_normalized_list = [(x-mean_value)/(max_value-min_value) for x in input_list]
    return mean_normalized_list


# In[4]:


def quantum_rescaling(input_list):
    #Quantum Mechanics Wavefunction
    import numpy as np
    squared_list = [x*x for x in input_list]
    squared_sum = np.sum(squared_list)
    quantum_rescaled_list = [x/squared_sum for x in input_list]
    return quantum_rescaled_list


# In[5]:


def rescaled_2D(input_list):
    max_value = max(max(input_list))
    min_value = min(min(input_list))
    rescaled_list = [[(row_iter-min_value)/(max_value-min_value) for row_iter in column_iter] for column_iter in input_list]
    return rescaled_list


# In[6]:


def MAV_cal(input_file):
    #Mean Absolute Value
    import math
    
    with open(input_file) as file_var:
        for line in file_var:
            channel_numbers =len(line.split())#actually 8 as already known
            break
    #print("Channel number:", channel_numbers)
    number_of_lines = 0
    
    with open(input_file) as file_var:
        for line in file_var:
            number_of_lines += 1
    #print("Line number:", number_of_lines)
    
    MAV_list = [0]*channel_numbers
    with open(input_file) as file_var:
        for line in file_var:
            line_iter = line.split()
            for channel_iter in range(channel_numbers):
                value_line_channel = line_iter[channel_iter]
                MAV_list[channel_iter] += math.fabs(float(value_line_channel))/number_of_lines
    
    return MAV_list


# In[7]:


def RMS_cal(input_file):
    #Root Mean Square
    import math
    
    with open(input_file) as file_var:
        for line in file_var:
            channel_numbers =len(line.split())#actually 8 as already known
            break
    #print("Channel number:", channel_numbers)
    number_of_lines = 0
    
    with open(input_file) as file_var:
        for line in file_var:
            number_of_lines += 1
    #print("Line number:", number_of_lines)
    
    RMS_list = []
    square_sum = [0]*channel_numbers
    with open(input_file) as file_var:
        for line in file_var:
            line_iter = line.split()
            for channel_iter in range(channel_numbers):
                value_line_channel = line_iter[channel_iter]
                square_sum[channel_iter] += float(value_line_channel)**2
        
    for item in square_sum:
        RMS_list.append(math.sqrt(item/number_of_lines))
    
    return RMS_list


# In[8]:


def ZC_cal(input_file):
    #Zero Crossings
    import math
    
    with open(input_file) as file_var:
        for line in file_var:
            channel_numbers =len(line.split())#actually 8 as already known
            break
    #print("Channel number:", channel_numbers)
    number_of_lines = 0
    
    with open(input_file) as file_var:
        for line in file_var:
            number_of_lines += 1
    #print("Line number:", number_of_lines)
    
    ZC_list = [0]*channel_numbers
    value_previous_line = [0]*channel_numbers
    
    with open(input_file) as file_var:
        
        first_line = file_var.readline().split()
        for channel_iter in range(channel_numbers):
            value_previous_line[channel_iter] = float(first_line[channel_iter])
        
        for line in file_var:
            line_iter = line.split()
            for channel_iter in range(channel_numbers):
                value_line_channel = line_iter[channel_iter]
                value_line_channel = float(value_line_channel)
                times_tmp = -value_line_channel*value_previous_line[channel_iter]
                if times_tmp>0:
                    ZC_list[channel_iter] += 1
                value_previous_line[channel_iter] = value_line_channel
                
    return ZC_list


# In[9]:


def SSC_cal(input_file):
    #Slope Sign Changes
    import math
    
    with open(input_file) as file_var:
        for line in file_var:
            channel_numbers =len(line.split())#actually 8 as already known
            break
    #print("Channel number:", channel_numbers)
    number_of_lines = 0
    
    with open(input_file) as file_var:
        for line in file_var:
            number_of_lines += 1
    #print("Line number:", number_of_lines)
    
    SSC_list = [0]*channel_numbers
    value_previous_previous_line = [0]*channel_numbers
    value_previous_line = [0]*channel_numbers
    
    with open(input_file) as file_var:
        
        first_line = file_var.readline().split()
        second_line = file_var.readline().split()
        for channel_iter in range(channel_numbers):
            value_previous_previous_line[channel_iter] = float(first_line[channel_iter])
            value_previous_line[channel_iter] = float(second_line[channel_iter])
        
        for line in file_var:
            line_iter = line.split()
            for channel_iter in range(channel_numbers):
                value_line_channel = line_iter[channel_iter]
                value_line_channel = float(value_line_channel)
                slope_times_tmp = (value_previous_line[channel_iter]-value_previous_previous_line[channel_iter])*(value_previous_line[channel_iter]-value_line_channel)
                if slope_times_tmp>0:
                    SSC_list[channel_iter] += 1
                value_previous_previous_line[channel_iter] = value_previous_line[channel_iter]
                value_previous_line[channel_iter] = value_line_channel
                
    return SSC_list


# In[10]:


def WL_cal(input_file):
    #Waveform Length
    import math
    
    with open(input_file) as file_var:
        for line in file_var:
            channel_numbers =len(line.split())#actually 8 as already known
            break
    #print("Channel number:", channel_numbers)
    number_of_lines = 0
    
    with open(input_file) as file_var:
        for line in file_var:
            number_of_lines += 1
    #print("Line number:", number_of_lines)
    
    WL_list = [0]*channel_numbers
    value_previous_line = [0]*channel_numbers
    
    with open(input_file) as file_var:
        
        first_line = file_var.readline().split()
        for channel_iter in range(channel_numbers):
            value_previous_line[channel_iter] =float(first_line[channel_iter])
        
        for line in file_var:
            line_iter = line.split()
            for channel_iter in range(channel_numbers):
                value_line_channel = line_iter[channel_iter]
                value_line_channel = float(value_line_channel)
                WL_list[channel_iter] += math.fabs(value_line_channel-value_previous_line[channel_iter])
                value_previous_line[channel_iter] = value_line_channel
    
    return WL_list


# In[11]:


def WA_cal(epsilon_var, input_file):
    #Willison Amplitude
    #epsilon_var set as 2.5 originally
    import math
    
    with open(input_file) as file_var:
        for line in file_var:
            channel_numbers =len(line.split())#actually 8 as already known
            break
    #print("Channel number:", channel_numbers)
    number_of_lines = 0
    
    with open(input_file) as file_var:
        for line in file_var:
            number_of_lines += 1
    #print("Line number:", number_of_lines)
    
    WA_list = [0]*channel_numbers
    value_previous_line = [0]*channel_numbers
    
    with open(input_file) as file_var:
        
        first_line = file_var.readline().split()
        for channel_iter in range(channel_numbers):
            value_previous_line[channel_iter] =float(first_line[channel_iter])
        
        for line in file_var:
            line_iter = line.split()
            for channel_iter in range(channel_numbers):
                value_line_channel = line_iter[channel_iter]
                value_line_channel = float(value_line_channel)
                fabs_tmp = math.fabs(value_line_channel-value_previous_line[channel_iter])
                if fabs_tmp>epsilon_var:
                    WA_list[channel_iter] += 1
                value_previous_line[channel_iter] = value_line_channel
    
    return WA_list


# In[12]:


def VAR_cal(input_file):
    #Variance
    import math
    
    with open(input_file) as file_var:
        for line in file_var:
            channel_numbers =len(line.split())#actually 8 as already known
            break
    #print("Channel number:", channel_numbers)
    number_of_lines = 0
    
    with open(input_file) as file_var:
        for line in file_var:
            number_of_lines += 1
    #print("Line number:", number_of_lines)
    
    VAR_list = []
    square_sum = [0]*channel_numbers
    with open(input_file) as file_var:
        for line in file_var:
            line_iter = line.split()
            for channel_iter in range(channel_numbers):
                value_line_channel = line_iter[channel_iter]
                square_sum[channel_iter] += float(value_line_channel)**2
        
    for item in square_sum:
        VAR_list.append(item/(number_of_lines-1))
    
    return VAR_list


# In[13]:


def LogD_cal(input_file):
    #Log Detector
    resolution_size = 0.01
    import math
    
    with open(input_file) as file_var:
        for line in file_var:
            channel_numbers =len(line.split())#actually 8 as already known
            break
    #print("Channel number:", channel_numbers)
    number_of_lines = 0
    
    with open(input_file) as file_var:
        for line in file_var:
            number_of_lines += 1
    #print("Line number:", number_of_lines)
    
    LogD_list = [0]*channel_numbers
    with open(input_file) as file_var:
        for line in file_var:
            line_iter = line.split()
            for channel_iter in range(channel_numbers):
                value_line_channel = line_iter[channel_iter]
                fabs_tmp = math.fabs(float(value_line_channel))
                if fabs_tmp < resolution_size:
                    fabs_tmp = resolution_size
                LogD_list[channel_iter] += math.log(fabs_tmp)/number_of_lines
                
    LogD_list = [math.exp(x) for x in LogD_list]
    
    return LogD_list


# In[14]:


def ARC_cal(input_file):
    #Auto Regression Coefficient
    #https://machinelearningmastery.com/autoregression-models-time-series-forecasting-python/
    import pandas as pd
    from pandas import Series
    from matplotlib import pyplot
    from statsmodels.tsa.ar_model import AR #conda install statsmodels
    from sklearn.metrics import mean_squared_error #pip install -U scikit-learn scipy matplotlib #conda installe scikit-learn
    import numpy as np
    
    with open(input_file) as file_var:
        for line in file_var:
            channel_numbers =len(line.split())#actually 8 as already known
            break
    #print("Channel number:", channel_numbers)
    '''
    number_of_lines = 0    
    
    with open(input_file) as file_var:
        for line in file_var:
            number_of_lines += 1
    #print("Line number:", number_of_lines)
    '''
    
    ARC_list = []
    ARC_list_LagNumber = []
    
    '''
    for column_n in range(channel_numbers):
        ARC_list.append([])
    '''
    
    input_series = pd.read_csv(input_file, header=None)
    input_tmp_file = input_series.values
    number_of_lines = len(input_tmp_file)
    input_file_array = [ [ float(ele) for ele in input_tmp_file[line_n][0].split() ] for line_n in range(number_of_lines)]
    input_file_array = np.array(input_file_array)
    for column_n in range(channel_numbers):
        model_fit = AR(input_file_array[:,column_n]).fit()
        ARC_list_LagNumber.append(model_fit.k_ar)
        ARC_list.append(model_fit.params.tolist())

    return ARC_list_LagNumber, ARC_list


# In[15]:


def ARC_results_plot(Input_LagList, Input_ARC_list, gesture_name, i_label):
    
    from mpl_toolkits.mplot3d import Axes3D  
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    import numpy as np
    
    X_data = np.arange(min(Input_LagList)+1)
    Y_data = np.arange(len(Input_ARC_list))
    X_data, Y_data = np.meshgrid(X_data, Y_data)
    Z_data = np.array(Input_ARC_list)
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Plot the surface.
    surf = ax.plot_surface(X_data, Y_data, Z_data, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    plt.title(gesture_name+"ARC"+i_label)
    plt.show()
    return fig


# In[16]:


def Ceps_cal(input_file):
    #Cepstrum coefficients
    LagList,ARC_List = ARC_cal(input_file)
    CepsList = []
    for channel_iter in range(len(ARC_List)):
        CepsList.append([-ARC_List[channel_iter][0]])
        
    for channel_iter in range(len(ARC_List)):
        item_size = LagList[channel_iter]+1
        for lag_iter in range(1, item_size):
            c_tmp = -ARC_List[channel_iter][lag_iter]
            for j_iter in range(0,lag_iter):
                c_tmp -= (lag_iter-j_iter)/(lag_iter+1)*CepsList[channel_iter][j_iter]*ARC_List[channel_iter][lag_iter-j_iter-1]
            CepsList[channel_iter].append(c_tmp)
    
    return CepsList


# In[17]:


def Ceps_results_plot(Input_CepsList, gesture_name, i_label):
    
    from mpl_toolkits.mplot3d import Axes3D  
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    import numpy as np
    
    X_data = np.arange(len(Input_CepsList[0]))
    Y_data = np.arange(len(Input_CepsList))
    X_data, Y_data = np.meshgrid(X_data, Y_data)
    Z_data = np.array(Input_CepsList)
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Plot the surface.
    surf = ax.plot_surface(X_data, Y_data, Z_data, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

    # Customize the z axis.
    #ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    plt.title(gesture_name+"Ceps"+i_label)
    plt.show()
    return fig


# In[18]:


def fMD_cal(input_file):
    #Median frequency
    return 0


# In[19]:


def fME_cal(input_file):
    #Mean frequency
    return 0


# In[20]:


def emgHist(input_file):
    #sEMG Histogram
    return 0


# In[21]:


def list_plot_mean_normalized(input_list, label_name, fig, ax):
    #https://matplotlib.org/3.1.0/gallery/lines_bars_and_markers/simple_plot.html#sphx-glr-gallery-lines-bars-and-markers-simple-plot-py
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    x_channel = np.arange(len(input_list))
    y_signal_feature = mean_normalization(input_list)
    ax.plot(x_channel, y_signal_feature, label=label_name)
    ax.legend()
    
    return fig


# In[22]:


def list_plot_rescaled_max_min(input_list, label_name, fig, ax):
    #https://matplotlib.org/3.1.0/gallery/lines_bars_and_markers/simple_plot.html#sphx-glr-gallery-lines-bars-and-markers-simple-plot-py
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    x_channel = np.arange(len(input_list))
    y_signal_feature = rescaling_max_min(input_list)
    ax.plot(x_channel, y_signal_feature, label=label_name)
    ax.legend()
    
    return fig


# In[31]:


#import matplotlib.pyplot as plt#Step1

number_of_gestures = 6
NumberPerGesture = 42

training_images = []
training_labels = []

import numpy as np
training_ARC_images = np.empty((0,8,8), int)
training_ARC_labels = []

training_Ceps_images = np.empty((0,8,8), int)
training_Ceps_labels = []

for gesture_iter in range(number_of_gestures):
    
    gesture = "gesture"+str(gesture_iter+1)
    gesture = locals()[gesture]

    #fig, ax = plt.subplots()Step2
    
    for i in range(NumberPerGesture):

        import matplotlib.pyplot as plt
        feature_list = []
        ARC_list_inverted = []
        Ceps_list_inverted = []

        number_of_lines = 0
        i_label = i+1
        input_file = TrainingData_PATH+gesture+str(i_label)+txt_format

        feature_list.append(rescaling_max_min(MAV_cal(input_file)))#Line0
        #fig_MAV = list_plot_rescaled_max_min(MAV_cal(input_file),gesture+"MAV"+str(i_label), fig, ax)Step3
        feature_list.append(rescaling_max_min(RMS_cal(input_file)))#Line1
        feature_list.append(rescaling_max_min(ZC_cal(input_file)))#Line2
        feature_list.append(rescaling_max_min(SSC_cal(input_file)))#Line3
        feature_list.append(rescaling_max_min(WL_cal(input_file)))#Line4
        feature_list.append(rescaling_max_min(WA_cal(2.5,input_file)))#Line5
        feature_list.append(rescaling_max_min(VAR_cal(input_file)))#Line6
        feature_list.append(rescaling_max_min(LogD_cal(input_file)))#Line7
        
        training_images.append(feature_list)
        training_labels.append(gesture_iter+1)
        
        import matplotlib.pyplot as plt
        plt.imshow(feature_list)
        #print(feature_list)
        print(gesture+str(i_label)+" Row: %s, Column: %s" % (len(feature_list),len(feature_list[0])))


        LagList,ARC_List = ARC_cal(input_file)
        for item in [[row[i] for row in ARC_List] for i in range(len(ARC_List[0]))]:
            ARC_list_inverted.append(item)#Line8-19
        fig_tmp = ARC_results_plot(LagList, ARC_List, gesture, str(i_label))
        #fig_tmp.savefig(gesture+str(i_label)+"ARC"+".jpg")
        
        training_ARC_images = np.concatenate((training_ARC_images, [rescaled_2D(ARC_list_inverted)[:8]]), axis = 0)
        training_ARC_labels.append(gesture_iter+1)
        plt.imshow(rescaled_2D(ARC_list_inverted))

        
        Ceps_List = Ceps_cal(input_file)
        for item in [[row[i] for row in Ceps_List] for i in range(len(Ceps_List[0]))]:
            Ceps_list_inverted.append(item)#Line20-31
        fig_tmp = Ceps_results_plot(Ceps_list_inverted, gesture, str(i_label))
        #fig_tmp.savefig(gesture+str(i_label)+"Ceps"+".jpg")
        
        training_Ceps_images = np.concatenate((training_Ceps_images, [rescaled_2D(Ceps_list_inverted)[:8]]), axis = 0)
        training_Ceps_labels.append(gesture_iter+1)
        plt.imshow(rescaled_2D(Ceps_list_inverted))

    #plt.show()#Step4
    #fig_MAV.savefig(gesture+"MAV_Rescaled"+".jpg")#Step5#END


# #import matplotlib.pyplot as plt#Step1
# 
# number_of_gestures = 3
# NumberPerGesture = 7
# 
# test_images = []
# test_labels = []
# 
# import numpy as np
# test_ARC_images = np.empty((0,10,8), int)
# test_ARC_labels = []
# 
# test_Ceps_images = np.empty((0,10,8), int)
# test_Ceps_labels = []
# 
# for gesture_iter in range(number_of_gestures):
#     
#     gesture = "gesture"+str(gesture_iter+1)
#     gesture = locals()[gesture]
# 
#     #fig, ax = plt.subplots()Step2
#     
#     for i in range(NumberPerGesture):
# 
#         import matplotlib.pyplot as plt
#         feature_list = []
#         ARC_list_inverted = []
#         Ceps_list_inverted = []
# 
#         number_of_lines = 0
#         i_label = i+1
#         input_file = TestData_PATH+gesture+str(i_label)+txt_format
# 
#         feature_list.append(rescaling_max_min(MAV_cal(input_file)))#Line0
#         #fig_MAV = list_plot_rescaled_max_min(MAV_cal(input_file),gesture+"MAV"+str(i_label), fig, ax)Step3
#         feature_list.append(rescaling_max_min(RMS_cal(input_file)))#Line1
#         feature_list.append(rescaling_max_min(ZC_cal(input_file)))#Line2
#         feature_list.append(rescaling_max_min(SSC_cal(input_file)))#Line3
#         feature_list.append(rescaling_max_min(WL_cal(input_file)))#Line4
#         feature_list.append(rescaling_max_min(WA_cal(2.5,input_file)))#Line5
#         feature_list.append(rescaling_max_min(VAR_cal(input_file)))#Line6
#         feature_list.append(rescaling_max_min(LogD_cal(input_file)))#Line7
#         
#         test_images.append(feature_list)
#         test_labels.append(gesture_iter+1)
#         
#         import matplotlib.pyplot as plt
#         plt.imshow(feature_list)
#         #print(feature_list)
#         print(gesture+str(i_label)+" Row: %s, Column: %s" % (len(feature_list),len(feature_list[0])))
# 
# 
#         LagList,ARC_List = ARC_cal(input_file)
#         for item in [[row[i] for row in ARC_List] for i in range(len(ARC_List[0]))]:
#             ARC_list_inverted.append(item)#Line8-19
#         fig_tmp = ARC_results_plot(LagList, ARC_List, gesture, str(i_label))
#         #fig_tmp.savefig(gesture+str(i_label)+"ARC"+".jpg")
#         
#         test_ARC_images = np.concatenate((test_ARC_images, [rescaled_2D(ARC_list_inverted)[:10]]), axis = 0)
#         test_ARC_labels.append(gesture_iter+1)
#         plt.imshow(rescaled_2D(ARC_list_inverted))
# 
#         
#         Ceps_List = Ceps_cal(input_file)
#         for item in [[row[i] for row in Ceps_List] for i in range(len(Ceps_List[0]))]:
#             Ceps_list_inverted.append(item)#Line20-31
#         fig_tmp = Ceps_results_plot(Ceps_list_inverted, gesture, str(i_label))
#         #fig_tmp.savefig(gesture+str(i_label)+"Ceps"+".jpg")
#         
#         test_Ceps_images = np.concatenate((test_Ceps_images, [rescaled_2D(Ceps_list_inverted)[:10]]), axis = 0)
#         test_Ceps_labels.append(gesture_iter+1)
#         plt.imshow(rescaled_2D(Ceps_list_inverted))
# 
#     #plt.show()#Step4
#     #fig_MAV.savefig(gesture+"MAV_Rescaled"+".jpg")#Step5#END

# In[ ]:





# In[32]:


import numpy as np

training_images = np.array(training_images)
training_labels = np.array(training_labels)
#test_images = np.array(test_images)
#test_labels = np.array(test_labels)

training_ARC_images = np.array(training_ARC_images)
training_ARC_labels = np.array(training_ARC_labels)
#test_ARC_images = np.array(test_ARC_images)
#test_ARC_labels = np.array(test_ARC_labels)

training_Ceps_images = np.array(training_Ceps_images)
training_Ceps_labels = np.array(training_Ceps_labels)
#test_Ceps_images = np.array(test_Ceps_images)
#test_Ceps_labels = np.array(test_Ceps_labels)


# In[ ]:





# In[33]:


import random
index_ran = [i for i in range(len(training_images))]
random.shuffle(index_ran)


# In[34]:


training_images = training_images[index_ran]
training_labels = training_labels[index_ran]
training_ARC_images = training_ARC_images[index_ran]
training_ARC_labels = training_ARC_labels[index_ran]
training_Ceps_images = training_Ceps_images[index_ran]
training_Ceps_labels = training_Ceps_labels[index_ran]


# In[35]:


test_number = 24
test_images = training_images[-test_number:]
test_labels = training_labels[-test_number:]
training_images = training_images[:-test_number]
training_labels = training_labels[:-test_number]
test_ARC_images = training_ARC_images[-test_number:]
test_ARC_labels = training_ARC_labels[-test_number:]
training_ARC_images = training_ARC_images[:-test_number]
training_ARC_labels = training_ARC_labels[:-test_number]
test_Ceps_images = training_Ceps_images[-test_number:]
test_Ceps_labels = training_Ceps_labels[-test_number:]
training_Ceps_images = training_Ceps_images[:-test_number]
training_Ceps_labels = training_Ceps_labels[:-test_number]


# In[36]:


training_images = np.array(training_images)
training_labels = np.array(training_labels)
test_images = np.array(test_images)
test_labels = np.array(test_labels)

training_ARC_images = np.array(training_ARC_images)
training_ARC_labels = np.array(training_ARC_labels)
test_ARC_images = np.array(test_ARC_images)
test_ARC_labels = np.array(test_ARC_labels)

training_Ceps_images = np.array(training_Ceps_images)
training_Ceps_labels = np.array(training_Ceps_labels)
test_Ceps_images = np.array(test_Ceps_images)
test_Ceps_labels = np.array(test_Ceps_labels)


# In[37]:


import tensorflow as tf
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(8,8)),
    keras.layers.Dense(10240, activation = tf.nn.relu),
    keras.layers.Dense(7, activation = tf.nn.softmax)
])

model.compile(optimizer = tf.train.AdamOptimizer(), loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(training_images, training_labels, epochs = 500)


# In[38]:


model.evaluate(test_images, test_labels)


# In[ ]:





# In[ ]:





# ARC&Ceps Methods below: not good

# import tensorflow as tf
# from tensorflow import keras
# 
# model = keras.Sequential([
#     keras.layers.Flatten(input_shape=(8,8)),
#     keras.layers.Dense(10240, activation = tf.nn.relu),
#     keras.layers.Dense(7, activation = tf.nn.softmax)
# ])
# 
# model.compile(optimizer = tf.train.AdamOptimizer(), loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
# 
# model.fit(training_ARC_images, training_ARC_labels, epochs = 500)

# model.evaluate(test_ARC_images, test_ARC_labels)

# 

# import tensorflow as tf
# from tensorflow import keras
# 
# model = keras.Sequential([
#     keras.layers.Flatten(input_shape=(8,8)),
#     keras.layers.Dense(10240, activation = tf.nn.relu),
#     keras.layers.Dense(7, activation = tf.nn.softmax)
# ])
# 
# model.compile(optimizer = tf.train.AdamOptimizer(), loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
# 
# model.fit(training_Ceps_images, training_Ceps_labels, epochs = 500)

# model.evaluate(test_Ceps_images, test_Ceps_labels)

# In[ ]:





# In[ ]:




