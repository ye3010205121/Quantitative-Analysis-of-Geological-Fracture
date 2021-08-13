# -*- coding: utf-8 -*-
# @author: Chen jinqiao
# @date: 2020-01
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import minimize_scalar
from scipy.optimize import fmin,fminbound

#输入点坐标文件
data=pd.read_csv("data2-6.csv")
ydata=[]
xdata=[]
xdata=data.loc[:,"x"]
ydata=data.loc[:,"y"]
#确定周期
T=len(xdata)
b=2*np.pi/T

def Fun(x,a,c,d):                   # 定义拟合函数形式
    return a*np.sin(b*x+c)+d
    
def error (p,x,y): # 拟合残差
    return (Fun(p,x)-y)
def main(): 
    x = np.linspace(18,159,157)       # 创建时间序列
    a,c,d = [10,10,10]              # 原始数据的参数
    para,pcov=curve_fit(Fun,xdata,ydata)
    y_fitted = Fun(x,para[0],para[1],para[2]) # 画出拟合后的曲线
   
    def f(x):
        return para[0]*np.sin(b*x +para[1])+para[2]
    res = minimize_scalar(f, bounds=(18, 160), method='bounded')
    print ("min=",res.x)
    print (Fun(res.x,para[0],para[1],para[2]))
    
    def m(x):
        return (-para[0]*np.sin(b*x +para[1])-para[2])
    res2 = minimize_scalar(m, bounds=(18, 160), method='bounded')
    print ("max=",res2.x)
    
    
    #print (Fun(res2.x,para[0],para[1],para[2]))
    
    def circle_2d(dx=0.001,plot=True):
        dx = dx # 变化率
        x = np.arange(20,160, dx)
        y = f(x)

    # print(len(t))
        area_list = [] # 存储每一微小步长的曲线长度
   
        for i in range(1,len(x)):
        # 计算每一微小步长的曲线长度，dx = x_{i}-x{i-1}，索引从1开始
            dl_i = np.sqrt( (x[i]-x[i-1])**2 + (y[i]-y[i-1])**2 ) 
        # 将计算结果存储起来
            area_list.append(dl_i)

        area = sum(area_list)# 求和计算曲线在t:[0,2*pi]的长度

        print("length：{:.4f}".format(area))
 
    plt.figure #画图
    plt.plot(xdata,ydata,'r', label = 'Original curve')
    plt.plot(x,y_fitted,'-b', label ='Fitted curve')
   
    
    plt.legend()
    plt.show()
    print('y=%0.3f*sin(%0.5f*x+%0.2f)+%0.2f'%(para[0],b,para[1],para[2]))#输出表达式
    circle_2d(dx=0.001,plot=True)
    #print (para)
 
if __name__=='__main__':
   main()