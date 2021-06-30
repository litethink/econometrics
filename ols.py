
import numpy as np
import statsmodels.api as sm
#收入
x = [1300,1700,800,2100,1200,2300,1300,1500,1800,2200,3000,2500,1900,2200,900,1000,1500,600,1200,700]
#消费
y=[1000,2000,1200,2000,800,2400,1000,1500,1800,2000,2300,2500,2000,2100,1000,900,1200,800,1000,600]

x = np.array(x)
y = np.array(y)

from matplotlib import pyplot as plt
plt.scatter(x,y)
plt.show()

sm.add_constant(x)
c_x = sm.add_constant(x)
result = (sm.OLS(y,c_x)).fit()
result.summary()

                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.850
Model:                            OLS   Adj. R-squared:                  0.841
Method:                 Least Squares   F-statistic:                     101.8
Date:                Wed, 30 Jun 2021   Prob (F-statistic):           7.75e-09
Time:                        10:16:57   Log-Likelihood:                -137.40
No. Observations:                  20   AIC:                             278.8
Df Residuals:                      18   BIC:                             280.8
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const        131.8368    146.729      0.899      0.381    -176.430     440.103
x1             0.8663      0.086     10.092      0.000       0.686       1.047
==============================================================================
Omnibus:                        1.000   Durbin-Watson:                   2.234
Prob(Omnibus):                  0.607   Jarque-Bera (JB):                0.772
Skew:                          -0.103   Prob(JB):                        0.680
Kurtosis:                       2.060   Cond. No.                     4.57e+03
==============================================================================

阿尔法截距 131.8368
贝塔斜率 0.8663 
