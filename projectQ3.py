# -*- coding: utf-8 -*-
"""
CHE223 - Project Q3

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as matplot
import pandas as pd

from scipy import stats


#matplot.rcParams['text.usetex'] = True
#matplot.rcParams.update({'font.size': 22})
import csv

with open('Statistics_Project_Dataset(2021).csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    #Needed for question 2
    #years = []
    #Mg = []

    #Needed for question 3
    distance = []
    Cl = []
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
        elif row[5] == "NA":
            line_count += 1
        # Uncomment this part for removing the outliers based on visual inspection---------------------------------------------------
        #----------------------------------------------------------------------------------------------------------------------------
        #elif float(row[7])>400:
            #line_count += 1
        else:
            distance = distance + [float(row[7])]
            Cl = Cl + [float(row[5])]
            line_count += 1

x=np.array(distance)
y=np.array(Cl)

print(x,y)
n=len(x) # count of data
if n != len(y):
    print('Size of x does not match size of y')


#%% SLR function
def SLR(x,y,alpha=0.05,make_plot=False):
    # Simple linear regression    
    # inputs x, y - x and y data for the linear regression
    # alpha (default = 0.05) - significance level
    # make_plot (default = False) - if true then plot data, fit, CI, PI and
        # residual plot
    temp=['n','b0','b1','yhat','e','SSE','SSR','SST','s2','R2',
          'b0_CI_lo','b0_CI_hi','b0_t','b0_pval',
          'b1_CI_lo','b1_CI_hi','b1_t','b1_f','b1_pval',
          'mu0_lo','mu0_hi','y0_lo','y0_hi',
          'talpha2','falpha']
    lr=pd.Series('object',index=temp)
    lr.n=len(x)
    if lr.n!=len(y):
        print('x and y must have the same dimension')
        return    
   
    xbar = np.average(x)
    ybar = np.average(y)
    Sxx = np.sum((x-xbar)**2)
    Syy = np.sum((y-ybar)**2)
    Sxy = np.sum((x-xbar)*(y-ybar))
    Sx2 = np.sum(x**2)

    lr.b1 = Sxy/Sxx
    lr.b0 = ybar - lr.b1*xbar
    lr.yhat = lr.b0+lr.b1*x
    lr.e=y-lr.yhat
    lr.SSE = np.sum(lr.e**2)
    lr.SSR = np.sum((lr.yhat-ybar)**2)
    lr.SST = np.sum((y-ybar)**2)
    lr.s2 = lr.SSE/(lr.n-2)
    s=np.sqrt(lr.s2)
    lr.R2 = 1-lr.SSE/lr.SST
    
    t_alpha2 = stats.t.ppf(1-alpha/2,df = lr.n-2)
    se_b0 = s*np.sqrt(Sx2/(lr.n*Sxx))
    lr.b0_CI_lo = lr.b0 - t_alpha2*se_b0
    lr.b0_CI_hi = lr.b0 + t_alpha2*se_b0
    lr.b0_t = lr.b0/se_b0
    lr.b0_pval = 2*(1-stats.t.cdf(lr.b0_t,lr.n-2))

    se_b1 = s/np.sqrt(Sxx)
    lr.b1_CI_lo = lr.b1 - t_alpha2*se_b1
    lr.b1_CI_hi = lr.b1 + t_alpha2*se_b1
    lr.b1_t = lr.b1/se_b1
    lr.b1_pval = 2*(1-stats.t.cdf(lr.b1_t,lr.n-2))

    se_mu = s*np.sqrt(1/lr.n+(x-xbar)**2/Sxx)
    lr.mu0_lo = lr.yhat - t_alpha2*se_mu
    lr.mu0_hi = lr.yhat + t_alpha2*se_mu

    se_y0 = s*np.sqrt(1+1/n+(x-xbar)**2/Sxx)
    lr.y0_lo = lr.yhat-t_alpha2*se_y0
    lr.y0_hi = lr.yhat+t_alpha2*se_y0

    lr.talpha2 = t_alpha2
    lr.falpha = stats.f.ppf(1-alpha,1,lr.n-2)
    lr.b1_f = lr.SSR/(s**2)
    
    if make_plot:
        plt.figure()
        plt.plot(x,y,'x')
        plt.xlabel('Minimum Distance to Treatment Plant (km)')
        plt.ylabel('Cl Concentration')
        plt.title("Cl Concentration vs Minimum Distance to Treatment Plant")
        #fign=plt.gcf().number
        plt.plot(x,lr.yhat)
        #confidence interval
        plt.plot(x,lr.mu0_lo,"b-")
        plt.plot(x,lr.mu0_hi,"b-",label="_nolegend_")
        #prediction interval
        plt.plot(x,lr.y0_lo,"g-")
        plt.plot(x,lr.y0_hi,"g-", label = "_nolegend_")

        plt.legend(("Data","Regression Line","Confidence Interval", "Prediction Interval"))

        plt.figure()
        plt.plot(lr.yhat,lr.e,"x")
        plt.axhline(y=0,color="k")
        plt.xlabel("Regressed Cl Concentration")
        plt.ylabel("Residual")
        plt.title("Cl Concentration vs Minimum Distance to Treatment Plant Residual Plot")
    
    return lr


#%% Call SLR function
temp=SLR(x,y,alpha = 0.05, make_plot=True)
print(temp)
plt.show()
