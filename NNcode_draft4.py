# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 20:05:29 2024

@author: xuyun
"""

#%% libraries

import numpy as np
from scipy import optimize
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import TruncatedNormal
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import time
from scipy.optimize import minimize
from datetime import datetime
from pytz import timezone
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
from datetime import timedelta
import holidays

#%% global parameters

T=10
sigma=1

K = 217

File_object = open(r"D:/a.NTU/Y3 summer/Time-consistent planning/Meeting13/217_8_nopti","r")

lines = File_object.readlines()

quantize_grid = np.zeros((9,K))
for i in range(K):
    temp = lines[i].split()
    for j in range(9):
        quantize_grid[j][i] = temp[j] # 0 is probability weight, 1~9 are values
        if j !=0:
            quantize_grid[j] = quantize_grid[j] * sigma


#%% training data set: all state variables

numTrain = 10000

c_min = 5000
c_max = 40000
c_train = np.random.uniform(c_min, c_max, numTrain) # (M_train, ) array


y_min = 1
y_max = 500
y_train = np.random.uniform(y_min, y_max, numTrain) # (M_train, ) array


gamma_min = 0.35 
gamma_max = 0.85
gamma_train = np.random.uniform(gamma_min, gamma_max, numTrain) # (M_train, ) array


v_min = 0.70
v_max = 0.99
v_train = np.random.uniform(v_min, v_max, numTrain)

S_min = 4
S_max = 400
S01_train,S02_train = np.random.uniform(S_min, S_max, numTrain), np.random.uniform(S_min, S_max, numTrain)
S11_train,S12_train = np.random.uniform(S_min, S_max, numTrain), np.random.uniform(S_min, S_max, numTrain)
S21_train,S22_train = np.random.uniform(S_min, S_max, numTrain), np.random.uniform(S_min, S_max, numTrain)
S31_train,S32_train = np.random.uniform(S_min, S_max, numTrain), np.random.uniform(S_min, S_max, numTrain)

para_min = -2
para_max = 2
a01_train, a02_train, b01_train = np.random.uniform(para_min, para_max, numTrain), np.random.uniform(para_min, para_max, numTrain), np.random.uniform(para_min, para_max, numTrain)
a11_train, a12_train, b11_train = np.random.uniform(para_min, para_max, numTrain), np.random.uniform(para_min, para_max, numTrain), np.random.uniform(para_min, para_max, numTrain)
a21_train, a22_train, b21_train = np.random.uniform(para_min, para_max, numTrain), np.random.uniform(para_min, para_max, numTrain), np.random.uniform(para_min, para_max, numTrain)
a31_train, a32_train, b31_train = np.random.uniform(para_min, para_max, numTrain), np.random.uniform(para_min, para_max, numTrain), np.random.uniform(para_min, para_max, numTrain)

mu0_train, mu1_train, mu2_train, mu3_train = np.random.uniform(4, 400, numTrain), np.random.uniform(4, 400, numTrain), np.random.uniform(4, 400, numTrain), np.random.uniform(4, 400, numTrain)

#%% terminal value function set up

def numerat(a1, a2,mu, S1, S2):
    numerator = a1*S1 + a2*S2 + mu
    return numerator

#def denomin(a1, a2, mu, S2, S3):
#    denominator = a1*S2 + a2*S3 + mu
#    return denominator

def ValueFunctionTerminal(gamma, c):
    return 1/gamma * np.sign(c) * (np.abs(c)) ** gamma

#r1, ..., r4 are ARMA(2,2) vectors of the length of quantizer
def StateActionValueFunctionTerminal(c1,y1,gamma,v, u , 
                                     a01, a02, b01, mu0, a11, a12, b11, mu1, 
                                     a21, a22, b21, mu2, a31, a32, b31, mu3,
                                     S01,S02,S11,S12,S21,S22,S31,S32, quantizer):
    
    r0 = np.divide(numerat(a01, a02, mu0, S01, S02) + b01*quantizer[1],
                   S01) 
    r1 = np.divide(numerat(a11, a12, mu1, S11, S12) + b11*quantizer[2],
                   S11) 
    r2 = np.divide(numerat(a21, a22, mu2, S21, S22) + b21*quantizer[3],
                   S21) 
    r3 = np.divide(numerat(a31, a32, mu3, S31, S32) + b31*quantizer[4],
                   S31) 
    
    temp_c = (u[0]*r0 + u[1]*r1 + u[2]*r2 + u[3]*r3)*(c1+y1-u[4]*c1)
    vec = 1/gamma * np.sign(u[4]*c1) * (np.abs(u[4]*c1)) ** gamma + v*ValueFunctionTerminal(gamma, temp_c)
    return np.sum(vec*quantizer[0])


#%% NN Predictor set up

def Predictor(c1,y1,gamma,v, 
              a01, a02, b01, mu0, a11, a12, b11, mu1, 
              a21, a22, b21, mu2, a31, a32, b31, mu3,
              S01,S02,S11,S12,S21,S22,S31,S32, 
              nnweights, inputscaler, outputscaler, scaleOutput = 1):
    
    inputdata = np.concatenate((
                                c1.reshape(-1,1),
                                y1.reshape(-1,1),
                                gamma.reshape(-1,1), 
                                v.reshape(-1,1),
                                
                                a01.reshape(-1,1),
                                a02.reshape(-1,1),
                                a11.reshape(-1,1),
                                a12.reshape(-1,1),
                                a21.reshape(-1,1),
                                a22.reshape(-1,1),
                                a31.reshape(-1,1),
                                a32.reshape(-1,1),
                                
                                b01.reshape(-1,1),
                                b11.reshape(-1,1),
                                b21.reshape(-1,1),
                                b31.reshape(-1,1),
                                
                                mu0.reshape(-1,1),
                                mu1.reshape(-1,1),
                                mu2.reshape(-1,1),
                                mu3.reshape(-1,1),
                                
                                S01.reshape(-1,1),
                                S02.reshape(-1,1),
                                S11.reshape(-1,1),
                                S12.reshape(-1,1),
                                S21.reshape(-1,1),
                                S22.reshape(-1,1),
                                S31.reshape(-1,1),
                                S32.reshape(-1,1)), axis = 1)
    
    inputdata = inputscaler.transform(inputdata)
    
    layer1out = np.dot(inputdata, nnweights[0]) + nnweights[1]
    
    layer1out = tf.keras.activations.elu(layer1out).numpy()
    
    layer2out = np.dot(layer1out, nnweights[2]) + nnweights[3]
    
    layer2out = tf.keras.activations.elu(layer2out).numpy()
    
    layer3out = np.dot(layer2out, nnweights[4]) + nnweights[5]
    
    layer3out = tf.keras.activations.elu(layer3out).numpy()
    
    layer4out = np.dot(layer3out, nnweights[6]) + nnweights[7]
    
    if scaleOutput == 0:   # for policy apply sigmoid (proportion)
        output = tf.keras.activations.sigmoid(layer4out).numpy() 
    if scaleOutput == 1:   # for value function apply output scaler
        output = outputscaler.inverse_transform(layer4out)
    
    return output

#%% Surrogate set up

def StateActionValueFunction(c1,y1,gamma,v, u , 
                             a01, a02, b01, mu0, a11, a12, b11, mu1, 
                             a21, a22, b21, mu2, a31, a32, b31, mu3,
                             S01,S02,S11,S12,S21,S22,S31,S32, 
                             nnweights, inputscaler, outputscaler, quantizer):
    
    numWeights = len(quantizer[0])
    
    r0 = np.divide(numerat(a01, a02, mu0, S01, S02) + b01*quantizer[1],
                   S01) 
    r1 = np.divide(numerat(a11, a12, mu1, S11, S12) + b11*quantizer[2],
                   S11) 
    r2 = np.divide(numerat(a21, a22, mu2, S21, S22) + b21*quantizer[3],
                   S21) 
    r3 = np.divide(numerat(a31, a32, mu3, S31, S32) + b31*quantizer[4],
                   S31) 
    
    temp = 1/gamma * np.sign(u[4]*c1) * (np.abs(u[4]*c1)) ** gamma \
            + v* Predictor(np.ones(numWeights) * (u[0]*r0+u[1]*r1+u[2]*r2+u[3]*r3)*(c1+y1-u[4]*c1),
                           np.ones(numWeights) * y1,
                           np.ones(numWeights) * gamma,
                           np.ones(numWeights) * v,
                           
                           np.ones(numWeights) * a01,np.ones(numWeights) * a02,
                           np.ones(numWeights) * a11,np.ones(numWeights) * a12,
                           np.ones(numWeights) * a21,np.ones(numWeights) * a22,
                           np.ones(numWeights) * a31,np.ones(numWeights) * a32,
                           
                           np.ones(numWeights) * b01,
                           np.ones(numWeights) * b11,
                           np.ones(numWeights) * b21,
                           np.ones(numWeights) * b31,
                           
                           np.ones(numWeights) * mu0,
                           np.ones(numWeights) * mu1,
                           np.ones(numWeights) * mu2,
                           np.ones(numWeights) * mu3,
                           
                           np.ones(numWeights) * S01,
                           np.ones(numWeights) * S02,
                           np.ones(numWeights) * S11,
                           np.ones(numWeights) * S12,
                           np.ones(numWeights) * S21,
                           np.ones(numWeights) * S22,
                           np.ones(numWeights) * S31,
                           np.ones(numWeights) * S32,
                           nnweights, inputscaler, outputscaler)
            
    v_m_n = np.sum(temp.flatten() * quantizer[0])
    
    return v_m_n


#%% training set up
#minmize may have problem, but the idea is there                             

def BuildAndTrainModel(c1_train, y1_train, gamma_train, v_train, 
                       a01_train, a02_train, b01_train, mu0_train, 
                       a11_train, a12_train, b11_train, mu1_train, 
                       a21_train, a22_train, b21_train, mu2_train, 
                       a31_train, a32_train, b31_train, mu3_train,
                       S01_train, S02_train,
                       S11_train, S12_train,
                       S21_train, S22_train,
                       S31_train, S32_train,
                       quantizer, nn_dim = 28, node_num = 20, batch_num = 64, epoch_num = 3000,
                       initializer = TruncatedNormal(mean = 0.0, stddev = 0.05, seed = 0) ):
        
    # Create training input and rescale
    numTrain = len(c1_train)
    
    input_train = np.concatenate((
                                    c1_train.reshape(-1,1),
                                    y1_train.reshape(-1,1),
                                    gamma_train.reshape(-1,1), 
                                    v_train.reshape(-1,1),
                                    
                                    a01_train.reshape(-1,1),a02_train.reshape(-1,1),
                                    a11_train.reshape(-1,1),a12_train.reshape(-1,1),
                                    a21_train.reshape(-1,1),a22_train.reshape(-1,1),
                                    a31_train.reshape(-1,1),a32_train.reshape(-1,1),
                                    
                                    b01_train.reshape(-1,1),b11_train.reshape(-1,1),
                                    b21_train.reshape(-1,1),b31_train.reshape(-1,1),
                                    
                                    mu0_train.reshape(-1,1),mu1_train.reshape(-1,1),
                                    mu2_train.reshape(-1,1),mu3_train.reshape(-1,1),
                                    
                                    S01_train.reshape(-1,1),S02_train.reshape(-1,1),
                                    S11_train.reshape(-1,1),S12_train.reshape(-1,1),
                                    S21_train.reshape(-1,1),S22_train.reshape(-1,1),
                                    S31_train.reshape(-1,1),S32_train.reshape(-1,1)
                                    
                                    ), axis = 1) 
    
    input_scaler = MinMaxScaler(feature_range = (0,1))
    input_scaler.fit(input_train)
    input_train_scaled = input_scaler.transform(input_train)
    
    valuefun_train = np.zeros((T+1, numTrain))
    policy_train = np.zeros((5, T+1, numTrain))
    
    # Create objects to save all NN solvers and scalers     
    output_scaler_valuefun = np.empty(T+1, dtype = object)
    nnsolver_valuefun = np.empty(T+1, dtype = object)
    nnsolver_policy = np.empty((5, T+1), dtype = object)
    
    start = time.perf_counter() 
    
    
    # Run through all time steps backwards 
    for j in range(T-1, 0, -1): # j is equivalent to t
        
        start_i = time.perf_counter()
        print("Time step " + str(j))
        
        # Create training output for value function and policy
        for i in range(numTrain):
            
            if j < (T-1):
                output_scaler = output_scaler_valuefun[j+1]
                
                f_i = lambda u: -1*StateActionValueFunction(
                                                        c1_train[i], y1_train[i], 
                                                        gamma_train[i], v_train[i], u, 
                                                        a01_train[i], a02_train[i], b01_train[i], mu0_train[i], 
                                                        a11_train[i], a12_train[i], b11_train[i], mu1_train[i], 
                                                        a21_train[i], a22_train[i], b21_train[i], mu2_train[i], 
                                                        a31_train[i], a32_train[i], b31_train[i], mu3_train[i],
                                                        S01_train[i], S02_train[i], 
                                                        S11_train[i], S12_train[i], 
                                                        S21_train[i], S22_train[i], 
                                                        S31_train[i], S32_train[i], 
                                                        nnsolver_valuefun[j+1].get_weights(),
                                                        input_scaler, output_scaler, quantizer)
# this output scaler valufun is where DP is incorporated, every previous period optimizaton takes the numeric 
# value of the last value function

            else:
                f_i = lambda u: -1*StateActionValueFunctionTerminal(
                                                        c1_train[i], y1_train[i], 
                                                        gamma_train[i], v_train[i], u,
                                                        a01_train[i], a02_train[i], b01_train[i], mu0_train[i], 
                                                        a11_train[i], a12_train[i], b11_train[i], mu1_train[i], 
                                                        a21_train[i], a22_train[i], b21_train[i], mu2_train[i], 
                                                        a31_train[i], a32_train[i], b31_train[i], mu3_train[i],
                                                        S01_train[i], S02_train[i], 
                                                        S11_train[i], S12_train[i],
                                                        S21_train[i], S22_train[i], 
                                                        S31_train[i], S32_train[i], 
                                                          quantizer)
                
            
            cons = ({'type': 'eq', 'fun': lambda u:  u[0] + u[1] + u[2] + u[3] - 1},
                    {'type': 'ineq', 'fun': lambda u:  0.5 - u[4]}) 

            solv_i = minimize(f_i, [0.25, 0.25, 0.25, 0.25, 0.1], 
                              bounds = [(-1, 1), (-1, 1), (-1, 1), (-1, 1), (0, 0.5)],
                              method='SLSQP', constraints=cons, options={'maxiter': 10000})
        
            policy_train[0][j][i] = solv_i.x[0]
            policy_train[1][j][i] = solv_i.x[1]
            policy_train[2][j][i] = solv_i.x[2]
            policy_train[3][j][i] = solv_i.x[3]
            policy_train[4][j][i] = solv_i.x[4]
            valuefun_train[j][i] = -1*solv_i.fun 
        
        end_i = time.perf_counter()
        print("     optimizations done: " + str(round((end_i-start_i)/60,2)) + " min.")
        

        start_i = time.perf_counter()
                
            
        # Build and train NN model for value function 
        output_scaler_valuefun[j] = MinMaxScaler(feature_range = (0,1))
        output_scaler_valuefun[j].fit(valuefun_train[j].reshape(-1, 1))
        valuefun_train_scaled = output_scaler_valuefun[j].transform(valuefun_train[j].reshape(-1,1))     
        nnsolver_valuefun[j] = Sequential()    
        nnsolver_valuefun[j].add(Dense(node_num, input_shape = (nn_dim,), activation = 'elu',
                                    kernel_initializer = initializer, bias_initializer = initializer))            
        nnsolver_valuefun[j].add(Dense(node_num, activation = 'elu',
                                    kernel_initializer = initializer, bias_initializer = initializer))
        nnsolver_valuefun[j].add(Dense(node_num, activation = 'elu',
                                    kernel_initializer = initializer, bias_initializer = initializer))
        nnsolver_valuefun[j].add(Dense(1, activation = None,
                                    kernel_initializer = initializer, bias_initializer = initializer))
        nnsolver_valuefun[j].compile(optimizer = 'adam', loss = 'mean_squared_error')
        nnsolver_valuefun[j].fit(input_train_scaled, valuefun_train_scaled,
                              epochs = epoch_num, batch_size = batch_num, verbose = 0)
        end_i = time.perf_counter()
        print("     train value function done: " + str(round((end_i-start_i)/60,2)) + " min.")       
        
        start_i = time.perf_counter()
        # Build and train NN model for policy
        
        for k in range(5):
            nnsolver_policy[k][j] = Sequential()
            nnsolver_policy[k][j].add(Dense(node_num, input_shape = (nn_dim,), activation = 'elu',
                                      kernel_initializer = initializer, bias_initializer = initializer))            
            nnsolver_policy[k][j].add(Dense(node_num, activation = 'elu',
                                      kernel_initializer = initializer, bias_initializer = initializer))           
            nnsolver_policy[k][j].add(Dense(node_num, activation = 'elu',
                                      kernel_initializer = initializer, bias_initializer = initializer))            
            nnsolver_policy[k][j].add(Dense(1, activation = 'sigmoid',
                                      kernel_initializer = initializer, bias_initializer = initializer))                 
            nnsolver_policy[k][j].compile(optimizer = 'adam', loss = 'mean_squared_error')            
            nnsolver_policy[k][j].fit(input_train_scaled, policy_train[k][j].reshape(-1, 1),
                            epochs = epoch_num, batch_size = batch_num, verbose = 0)
        
        end_i = time.perf_counter()
        print("     train policy done: " + str(round((end_i-start_i)/60,2)) + " min.")
        
    
    end = time.perf_counter()
    duration = (end-start)/60

    print("Duration: " + str(duration) + " min.")
    
    return nnsolver_policy,nnsolver_valuefun, input_scaler, output_scaler_valuefun

#%% Train

nnsolver_policy, nnsolver_valuefun, in_scaler, out_scaler_valuefun \
= BuildAndTrainModel(c_train, y_train, gamma_train, v_train, 
                       a01_train, a02_train, b01_train, mu0_train, 
                       a11_train, a12_train, b11_train, mu1_train, 
                       a21_train, a22_train, b21_train, mu2_train, 
                       a31_train, a32_train, b31_train, mu3_train,
                       S01_train, S02_train, 
                       S11_train, S12_train, 
                       S21_train, S22_train, 
                       S31_train, S32_train, 
                     quantize_grid)


#%% back testing prep: actual prices and ARIMA parameters

def predict_stock_prices(start_date, stocks, x):
    # Calculate end_date assuming 5 trading days per week and a buffer for holidays
    # 11 trading days + x days for ARIMA + buffer (~2 weeks for safety)
    buffer_days = 10
    total_days_needed = 11 + x + buffer_days
    end_date = pd.to_datetime(start_date) + pd.Timedelta(days=total_days_needed * 7/5)
    
    # Results containers
    actual_prices = {}
    arima_params = {}
    
    for stock in stocks:
        # Fetch historical stock data
        data = yf.download(stock, start=start_date, end=end_date.strftime('%Y-%m-%d'))
        
        # If not enough data, skip this stock
        if len(data) < 11 + x:
            print(f"Not enough data for {stock}. Skipping.")
            continue
        
        # Store actual prices for the 11 days with indices as keys
        actual_prices[stock] = data['Close'][-11:].reset_index(drop=True)
        
        # Initialize container for ARIMA parameters for each day
        arima_params[stock] = []
        
        # Train ARIMA model and save parameters for each day
        for t in range(11):
            train_data = data['Close'][t:t+x]
            model = ARIMA(train_data, order=(2,0,1))
            fitted_model = model.fit()
            
            # Store the parameters of the model
            arima_params[stock].append({
                'coef': fitted_model.params
            })
    
    return actual_prices, arima_params

# Example usage:
start_date = '2022-01-01'
stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
x = 5
actual, params = predict_stock_prices(start_date, stocks, x)



#%% back testing set

def TestModel(c0, gamma, v, y, 
              nnsolver_policy, 
              input_scaler, stocks, T, actual, params):
    
    samples = np.ones((1, 6, T+2))   # 3 = DNN solver, linear, VWAP; 
                                               # 5 = 5 elements u0, ..., u4, c        
    samples[:,5,0:T+2] = c0

    S0, S1, S2, S3 = stocks[0], stocks[1], stocks[2], stocks[3]  
    
    # Run through all parameter configurations            
    # Run through all time steps
    for t in range(T):
                
                # NN feedback policy, inventory, deviation, value/cost
        if t < (T-1):
                # assume t+1 is the each of the trading time t because of the index problem
                for i in range(5):
                    samples[0][i][t+1] = \
                                    Predictor(samples[0][5][t+1], y, gamma, v, 
                                              params[S0][t+1]['coef'][1], params[S0][t+1]['coef'][2], params[S0][t+1]['coef'][3], params[S0][t+1]['coef'][0], 
                                              params[S1][t+1]['coef'][1], params[S1][t+1]['coef'][2], params[S1][t+1]['coef'][3], params[S1][t+1]['coef'][0], 
                                              params[S2][t+1]['coef'][1], params[S2][t+1]['coef'][2], params[S2][t+1]['coef'][3], params[S2][t+1]['coef'][0], 
                                              params[S3][t+1]['coef'][1], params[S3][t+1]['coef'][2], params[S3][t+1]['coef'][3], params[S3][t+1]['coef'][0], 
                                              actual[S0][t+1], actual[S0][t], #here t+1 is present time, assume it is when we start to trade
                                              actual[S1][t+1], actual[S1][t], #hence we need 11 time points
                                              actual[S2][t+1], actual[S2][t],
                                              actual[S3][t+1], actual[S3][t],
                                              nnsolver_policy[i][t+1].get_weights(), input_scaler, None, 0)[0][0]
                            

                samples[0][5][t+2] = \
                (samples[0][0][t+1]*actual[S0][t+2]/actual[S0][t+1] +
                 samples[0][1][t+1]*actual[S1][t+2]/actual[S1][t+1] +
                 samples[0][2][t+1]*actual[S2][t+2]/actual[S2][t+1] +
                 samples[0][3][t+1]*actual[S3][t+2]/actual[S3][t+1]) * \
                    (samples[0][5][t+1] + y - samples[0][5][t+1]*samples[0][4][t+1] )
            
        else:
            samples[0][0][t+1] = samples[0][0][T-1]
            samples[0][1][t+1] = samples[0][1][T-1]
            samples[0][2][t+1] = samples[0][2][T-1]
            samples[0][3][t+1] = samples[0][3][T-1]                
            samples[0][4][t+1] = samples[0][4][T-1]
                          
    return samples


def RunTests(test_dates, c0, gamma, v, y, nnsolver_policy, input_scaler, stocks, T, x):
    results = {}  # Dictionary to store results for each test date

    for start_date in test_dates:
        actual, params = predict_stock_prices(start_date, stocks, x)
        
        # Run TestModel for each start_date
        samples = TestModel(c0, gamma, v, y, nnsolver_policy, input_scaler, stocks, T, actual, params)
        
        # Store the results
        results[start_date] = samples
    
    return results


#%% Run test

test_dates = ['2023-06-15', '2023-06-01', '2023-04-15', '2023-04-01', '2023-02-15', '2023-02-01', 
              '2022-11-15', '2022-11-01', '2022-09-15', '2022-09-01','2022-07-15', '2022-07-01']

stocks = ['AAPL', 'MSFT', 'JPM', 'AMZN']

results = RunTests(test_dates, 10000, np.array(0.5), np.array(0.9), np.array(100), 
                   nnsolver_policy, in_scaler, stocks, 10, 5)

for date in results:
    print(f"{date}: {results[date][0][5][10]}")




























