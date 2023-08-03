import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize as opt
import pickle

def calib_func(c, A0, A12, A1):
    return A0 + A12*np.sqrt(c) + A1*c

calib_data = np.loadtxt('calibration.csv', delimiter='\t')
print(calib_data)
optimizedResult = opt.curve_fit(calib_func, calib_data[:,0], calib_data[:,1])
print(optimizedResult)


observed_data = np.loadtxt('observed_data.csv', delimiter='\t')
observed_time = []
for iBin in range(observed_data.shape[0]):
    for iCount in range(int(observed_data[iBin,1])):
        observed_time.append(calib_func(observed_data[iBin,0], *optimizedResult[0]))

observed_time = np.array(observed_time)
print(observed_time.shape)
np.random.shuffle(observed_time)
with open('muon_lifetime.pkl', 'wb') as f:
    pickle.dump(observed_time, f)
                             
x = np.linspace(0,2000,1000)
y = calib_func(x, *optimizedResult[0])
plt.scatter(calib_data[:,0], calib_data[:,1])
plt.plot(x,y,'r-')
plt.show()


