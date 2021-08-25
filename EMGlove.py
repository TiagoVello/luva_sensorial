import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numbers
import warnings
from vpython import *
import time
import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations
import argparse


# Defining Constants

names = ['Pressure Sensor 0', 'Pressure Sensor 1', 'Accel X 1', 'Accel Y 1', 'Accel Z 1', 'Gyro X 1', 'Gyro Y 1', 'Gyro Z 1', 'Accel X 2', 'Accel Y 2', 'Accel Z 2', 
         'Gyro X 2', 'Gyro Y 2', 'Gyro Z 2', 'Accel X 3', 'Accel Y 3', 'Accel Z 3', 'Gyro X 3', 'Gyro Y 3', 'Gyro Z 3', 'Accel X 4', 'Accel Y 4', 
         'Accel Z 4', 'Gyro X 4', 'Gyro Y 4', 'Gyro Z 4', 'Accel X 5', 'Accel Y 5', 'Accel Z 5', 'Gyro X 5', 'Gyro Y 5', 'Gyro Z 5','Accel X 6',
         'Accel Y 6', 'Accel Z 6', 'Gyro X 6', 'Gyro Y 6', 'Gyro Z 6', 'Accel X 7', 'Accel Y 7', 'Accel Z 7', 'Gyro X 7', 'Gyro Y 7', 'Gyro Z 7', 
         'Timestamp']
names_no_wrist = ['Pressure Sensor 0', 'Pressure Sensor 1', 'Accel X 1', 'Accel Y 1', 'Accel Z 1', 'Gyro X 1', 'Gyro Y 1', 'Gyro Z 1', 'Accel X 2', 'Accel Y 2', 'Accel Z 2', 
         'Gyro X 2', 'Gyro Y 2', 'Gyro Z 2', 'Accel X 3', 'Accel Y 3', 'Accel Z 3', 'Gyro X 3', 'Gyro Y 3', 'Gyro Z 3', 'Accel X 4', 'Accel Y 4', 
         'Accel Z 4', 'Gyro X 4', 'Gyro Y 4', 'Gyro Z 4', 'Accel X 5', 'Accel Y 5', 'Accel Z 5', 'Gyro X 5', 'Gyro Y 5', 'Gyro Z 5','Accel X 6',
         'Accel Y 6', 'Accel Z 6', 'Gyro X 6', 'Gyro Y 6', 'Gyro Z 6', 'Timestamp']

#trial 27 is not emged
trial_number = 27

openbci_path = 'Documents\OpenBCI_GUI\Recordings'


# Defining Class
class Quaternion:
    
    """
    A simple class implementing basic quaternion arithmetic.
    """
    def __init__(self, w_or_q, x=None, y=None, z=None):
        """
        Initializes a Quaternion object
        :param w_or_q: A scalar representing the real part of the quaternion, another Quaternion object or a
                    four-element array containing the quaternion values
        :param x: The first imaginary part if w_or_q is a scalar
        :param y: The second imaginary part if w_or_q is a scalar
        :param z: The third imaginary part if w_or_q is a scalar
        """
        self._q = np.array([1, 0, 0, 0])

        if x is not None and y is not None and z is not None:
            w = w_or_q
            q = np.array([w, x, y, z])
        elif isinstance(w_or_q, Quaternion):
            q = np.array(w_or_q.q)
        else:
            q = np.array(w_or_q)
            if len(q) != 4:
                raise ValueError("Expecting a 4-element array or w x y z as parameters")

        self._set_q(q)

    # Quaternion specific interfaces

    def conj(self):
        """
        Returns the conjugate of the quaternion
        :rtype : Quaternion
        :return: the conjugate of the quaternion
        """
        return Quaternion(self._q[0], -self._q[1], -self._q[2], -self._q[3])

    def to_angle_axis(self):
        """
        Returns the quaternion's rotation represented by an Euler angle and axis.
        If the quaternion is the identity quaternion (1, 0, 0, 0), a rotation along the x axis with angle 0 is returned.
        :return: rad, x, y, z
        """
        if self[0] == 1 and self[1] == 0 and self[2] == 0 and self[3] == 0:
            return 0, 1, 0, 0
        rad = np.arccos(self[0]) * 2
        imaginary_factor = np.sin(rad / 2)
        if abs(imaginary_factor) < 1e-8:
            return 0, 1, 0, 0
        x = self._q[1] / imaginary_factor
        y = self._q[2] / imaginary_factor
        z = self._q[3] / imaginary_factor
        return rad, x, y, z

    @staticmethod
    def from_angle_axis(rad, x, y, z):
        s = np.sin(rad / 2)
        return Quaternion(np.cos(rad / 2), x*s, y*s, z*s)

    def to_euler_angles(self):
        pitch = np.arcsin(2 * self[1] * self[2] + 2 * self[0] * self[3])
        if np.abs(self[1] * self[2] + self[3] * self[0] - 0.5) < 1e-8:
            roll = 0
            yaw = 2 * np.arctan2(self[1], self[0])
        elif np.abs(self[1] * self[2] + self[3] * self[0] + 0.5) < 1e-8:
            roll = -2 * np.arctan2(self[1], self[0])
            yaw = 0
        else:
            roll = np.arctan2(2 * self[0] * self[1] - 2 * self[2] * self[3], 1 - 2 * self[1] ** 2 - 2 * self[3] ** 2)
            yaw = np.arctan2(2 * self[0] * self[2] - 2 * self[1] * self[3], 1 - 2 * self[2] ** 2 - 2 * self[3] ** 2)
        return roll, pitch, yaw

    def to_euler123(self):
        roll = np.arctan2(-2*(self[2]*self[3] - self[0]*self[1]), self[0]**2 - self[1]**2 - self[2]**2 + self[3]**2)
        pitch = np.arcsin(2*(self[1]*self[3] + self[0]*self[1]))
        yaw = np.arctan2(-2*(self[1]*self[2] - self[0]*self[3]), self[0]**2 + self[1]**2 - self[2]**2 - self[3]**2)
        return roll, pitch, yaw

    def __mul__(self, other):
        """
        multiply the given quaternion with another quaternion or a scalar
        :param other: a Quaternion object or a number
        :return:
        """
        if isinstance(other, Quaternion):
            w = self._q[0]*other._q[0] - self._q[1]*other._q[1] - self._q[2]*other._q[2] - self._q[3]*other._q[3]
            x = self._q[0]*other._q[1] + self._q[1]*other._q[0] + self._q[2]*other._q[3] - self._q[3]*other._q[2]
            y = self._q[0]*other._q[2] - self._q[1]*other._q[3] + self._q[2]*other._q[0] + self._q[3]*other._q[1]
            z = self._q[0]*other._q[3] + self._q[1]*other._q[2] - self._q[2]*other._q[1] + self._q[3]*other._q[0]

            return Quaternion(w, x, y, z)
        elif isinstance(other, numbers.Number):
            q = self._q * other
            return Quaternion(q)

    def __add__(self, other):
        """
        add two quaternions element-wise or add a scalar to each element of the quaternion
        :param other:
        :return:
        """
        if not isinstance(other, Quaternion):
            if len(other) != 4:
                raise TypeError("Quaternions must be added to other quaternions or a 4-element array")
            q = self.q + other
        else:
            q = self.q + other.q

        return Quaternion(q)

    # Implementing other interfaces to ease working with the class

    def _set_q(self, q):
        self._q = q

    def _get_q(self):
        return self._q

    q = property(_get_q, _set_q)

    def __getitem__(self, item):
        return self._q[item]

    def __array__(self):
        return self._q

class Madgwick:
# https://github.com/morgil/madgwick_py/blob/master/quaternion.py
    quaternion = Quaternion(1, 0, 0, 0)
    beta = 1

    def __init__(self, quaternion=None, beta=None):
        """
        Initialize the class with the given parameters.
        :param sampleperiod: The sample period
        :param quaternion: Initial quaternion
        :param beta: Algorithm gain beta
        :return:
        """
        if quaternion is not None:
            self.quaternion = quaternion
        if beta is not None:
            self.beta = beta
        
    def update_imu(self, gyroscope, accelerometer, delta_t):
        """
        Perform one update step with data from a IMU sensor array
        :param gyroscope: A three-element array containing the gyroscope data in radians per second.
        :param accelerometer: A three-element array containing the accelerometer data. Can be any unit since a normalized value is used.
        """
        q = self.quaternion
        
        gyroscope = np.array(gyroscope, dtype=float).flatten()
        accelerometer = np.array(accelerometer, dtype=float).flatten()

        # Normalise accelerometer measurement
        
        if np.linalg.norm(accelerometer) == 0:
            warnings.warn("accelerometer is zero")
            return
        accelerometer /= np.linalg.norm(accelerometer)

        # Gradient descent algorithm corrective step
        f = np.array([
            2*(q[1]*q[3] - q[0]*q[2]) - accelerometer[0],
            2*(q[0]*q[1] + q[2]*q[3]) - accelerometer[1],
            2*(0.5 - q[1]**2 - q[2]**2) - accelerometer[2]
        ])
        j = np.array([
            [-2*q[2], 2*q[3], -2*q[0], 2*q[1]],
            [2*q[1], 2*q[0], 2*q[3], 2*q[2]],
            [0, -4*q[1], -4*q[2], 0]
        ])
        step = j.T.dot(f)
        step /= np.linalg.norm(step)  # normalise step magnitude

        # Compute rate of change of quaternion
        qdot = (q * Quaternion(0, gyroscope[0], gyroscope[1], gyroscope[2])) * 0.5 - self.beta * step.T

        # Integrate to yield quaternion
        q += qdot * delta_t
        self.quaternion = Quaternion(q / np.linalg.norm(q))  # normalise quaternion
            
# Defining funcions

def cook_data (raw,names):    
    errors = 0
    if 'Accel X 7' in names:
        sensor_range = range(1,8)
    else:
        sensor_range = range(1,7)
    for sensor_number in sensor_range:
        for axis in ['X','Y','Z']:
            for sensor in ['Accel', 'Gyro']:
                key = '{} {} {}'.format(sensor, axis, str(sensor_number))
                for row in range(len(raw)):
                    # Correcting Errors
                    if raw.loc[row,key] == 'X':
                        errors += 1
                        raw.loc[row,key] = raw.loc[row-1,key]
                    raw.loc[row,key] = int(raw.loc[row,key]) 
                if sensor == 'Accel':
                    raw[key] = raw[key]/16384.0
                else:
                    raw[key] = (raw[key]*np.pi)/(131.0*360)
                    
    return raw
 
def calibrate (cal_cooked_data, cooked_data, valid_range):
    if 'Accel X 7' in cal_cooked_data.columns:
        calibration_coef={'Accel X 1':0 ,'Accel Y 1':0, 'Accel Z 1':0, 'Gyro X 1':0, 'Gyro Y 1':0, 'Gyro Z 1':0, 'Accel X 2':0, 'Accel Y 2':0, 'Accel Z 2':0, 
         'Gyro X 2':0, 'Gyro Y 2':0, 'Gyro Z 2':0, 'Accel X 3':0, 'Accel Y 3':0, 'Accel Z 3':0, 'Gyro X 3':0, 'Gyro Y 3':0, 'Gyro Z 3':0, 'Accel X 4':0, 'Accel Y 4':0, 
         'Accel Z 4':0, 'Gyro X 4':0, 'Gyro Y 4':0, 'Gyro Z 4':0, 'Accel X 5':0, 'Accel Y 5':0, 'Accel Z 5':0, 'Gyro X 5':0, 'Gyro Y 5':0, 'Gyro Z 5':0,'Accel X 6':0,
         'Accel Y 6':0, 'Accel Z 6':0, 'Gyro X 6':0, 'Gyro Y 6':0, 'Gyro Z 6':0, 'Accel X 7':0, 'Accel Y 7':0, 'Accel Z 7':0, 'Gyro X 7':0, 'Gyro Y 7':0, 'Gyro Z 7':0}
    else:
        calibration_coef={'Accel X 1':0 ,'Accel Y 1':0, 'Accel Z 1':0, 'Gyro X 1':0, 'Gyro Y 1':0, 'Gyro Z 1':0, 'Accel X 2':0, 'Accel Y 2':0, 'Accel Z 2':0, 
         'Gyro X 2':0, 'Gyro Y 2':0, 'Gyro Z 2':0, 'Accel X 3':0, 'Accel Y 3':0, 'Accel Z 3':0, 'Gyro X 3':0, 'Gyro Y 3':0, 'Gyro Z 3':0, 'Accel X 4':0, 'Accel Y 4':0, 
         'Accel Z 4':0, 'Gyro X 4':0, 'Gyro Y 4':0, 'Gyro Z 4':0, 'Accel X 5':0, 'Accel Y 5':0, 'Accel Z 5':0, 'Gyro X 5':0, 'Gyro Y 5':0, 'Gyro Z 5':0,'Accel X 6':0,
         'Accel Y 6':0, 'Accel Z 6':0, 'Gyro X 6':0, 'Gyro Y 6':0, 'Gyro Z 6':0}
    for key in calibration_coef.keys():
        calibration_coef[key] = cal_cooked_data.loc[valid_range[0]:valid_range[1],key].mean()
        
    for key in calibration_coef.keys():
        if 'Gyro' in key:
            cooked_data[key] = cooked_data[key] - calibration_coef[key]
    return cooked_data

def plot_cooked_data (cooked_data, calibration = False):
    if calibration:
        cal = 'Calibration '
    else:
        cal = ''
    if 'Accel X 7' in cooked_data:
        sensor_names = ['Little', 'Ring', 'Middle', 'Index', 'Back of Hand', 'Thumb', 'Wrist']
    else:
        sensor_names = ['Little', 'Ring', 'Middle', 'Index', 'Back of Hand', 'Thumb',]
    for n, sensor_name in enumerate(sensor_names):
        for sensor in ['Accel ', 'Gyro ']:
            plt.title(cal+sensor+sensor_name)
            plt.plot(cooked_data[sensor+'X '+str(n+1)],color='r')
            plt.plot(cooked_data[sensor+'Y '+str(n+1)],color='g')
            plt.plot(cooked_data[sensor+'Z '+str(n+1)],color='b')
            plt.show()
    
def data_to_euler_angles(data,beta):
    if 'Accel X 7' in data.columns:
        columns = ['Pressure Sensor 0', 'Pressure Sensor 1', 'Roll 1', 'Pitch 1', 'Yaw 1', 'Roll 2', 'Pitch 2', 'Yaw 2', 'Roll 3', 'Pitch 3', 'Yaw 3', 'Roll 4', 'Pitch 4', 
             'Yaw 4', 'Roll 5', 'Pitch 5', 'Yaw 5', 'Roll 6', 'Pitch 6', 'Yaw 6', 'Roll 7', 'Pitch 7', 'Yaw 7', 'Delta t']
        sensor_numbers = ['1','2','3','4','5','6','7']
    else:
        columns = ['Pressure Sensor 0', 'Pressure Sensor 1', 'Roll 1', 'Pitch 1', 'Yaw 1', 'Roll 2', 'Pitch 2', 'Yaw 2', 'Roll 3', 'Pitch 3', 'Yaw 3', 'Roll 4', 'Pitch 4', 
             'Yaw 4', 'Roll 5', 'Pitch 5', 'Yaw 5', 'Roll 6', 'Pitch 6', 'Yaw 6', 'Delta t']
        sensor_numbers = ['1','2','3','4','5','6']
    euler_data = pd.DataFrame(columns=columns)
    mad = []
    for i in range(7):
        mad.append(Madgwick(beta = beta))
        
    euler_data.loc[0,'Pressure Sensor 0'] = data.loc[0,'Pressure Sensor 0'] 
    euler_data.loc[0,'Pressure Sensor 1'] = data.loc[0,'Pressure Sensor 1'] 
    euler_data.loc[0,'Delta t'] = 0.0   
    
    for n,sensor_number in enumerate(sensor_numbers):
        angles = mad[n].quaternion.to_euler_angles()
        euler_data.loc[0,('Roll '+sensor_number)] = angles[0]
        euler_data.loc[0,('Pitch '+sensor_number)] = angles[1]
        euler_data.loc[0,('Yaw '+sensor_number)] = angles[2]
        
    for row in range(1,len(data)):
        euler_data.loc[row,'Pressure Sensor 0'] = data.loc[row,'Pressure Sensor 0'] 
        euler_data.loc[row,'Pressure Sensor 1'] = data.loc[row,'Pressure Sensor 1'] 
        euler_data.loc[row,'Delta t'] = data.loc[row,'Timestamp']-data.loc[row-1,'Timestamp']
        
        for n,sensor_number in enumerate(sensor_numbers):
            g = [data.loc[row,('Gyro X '+sensor_number)],data.loc[row,('Gyro Y '+sensor_number)],data.loc[row,('Gyro Z '+sensor_number)]]
            a = [data.loc[row,('Accel X '+sensor_number)],data.loc[row,('Accel Y '+sensor_number)],data.loc[row,('Accel Z '+sensor_number)]]
            t = euler_data.loc[row,'Delta t']
            
            mad[n].update_imu(gyroscope=g,accelerometer=a,delta_t=t)
            angles = mad[n].quaternion.to_euler_angles()
            euler_data.loc[row,('Roll '+sensor_number)] = angles[0]
            euler_data.loc[row,('Pitch '+sensor_number)] = angles[1]
            euler_data.loc[row,('Yaw '+sensor_number)] = angles[2]
    
    return euler_data, mad
  
def plot_euler_data(euler_data, calibration = False):
    if calibration:
        cal = 'Calibration '
    else:
        cal = ''
    if 'Roll 7' in euler_data.columns:
        sensors = ['Little', 'Ring', 'Middle', 'Index', 'Back of Hand', 'Thumb', 'Wrist']
    else:
        sensors = ['Little', 'Ring', 'Middle', 'Index', 'Back of Hand', 'Thumb']
    for n, sensor in enumerate(sensors):
        plt.title(cal+'Sensor : '+sensor)
        plt.plot(euler_data['Roll '+str(n+1)],color='r')
        plt.plot(euler_data['Pitch '+str(n+1)],color='g')
        plt.plot(euler_data['Yaw '+str(n+1)],color='b')
        plt.show()    
    
def clear_scene():
    for obj in scene.objects: 
        obj.visible = False

def obj_orientation(bias, data, row, sensor_number, obj_name, forward, position):
    roll = data.loc[row,'Roll '+str(sensor_number)] + bias[int(sensor_number-1)][0]
    pitch = data.loc[row,'Pitch '+str(sensor_number)] + bias[int(sensor_number-1)][1]
    yaw = data.loc[row,'Yaw '+str(sensor_number)] + bias[int(sensor_number-1)][2]
    k=vector(cos(yaw)*cos(pitch), sin(pitch),sin(yaw)*cos(pitch))
    y=vector(0,1,0)
    s=cross(k,y)
    v=cross(s,k)
    vrot=v*cos(roll)+cross(k,v)*sin(roll)
    obj_name.axis=k*3
    obj_name.up=vrot
    if position == 0:
        obj_name.pos=vector(-3,0,0)
    else:
        obj_name.pos=position+forward*k
    return k, cross(k,vrot), v, vrot

def animation(data, mad):
    
    bias=[]
    for n,sensor_number in enumerate(['1','2','3','4','5','6']):
        if mad != None:
            bias.append( mad[n].quaternion.to_euler_angles() )
        else:
            bias.append ([0,0,0])
    
    scene.range=5
#    scene.forward=vector(-1,-1,-1)
 
    scene.width=1900
    scene.height=950
    
    clear_scene()
    xarrow=arrow(lenght=8, shaftwidth=.1,pos=vector(0,0,0,),opacity=1, color=color.red,axis=vector(1,0,0))
    yarrow=arrow(lenght=8, shaftwidth=.1,pos=vector(0,0,0,),opacity=1, color=color.green,axis=vector(0,1,0))
    zarrow=arrow(lenght=8, shaftwidth=.1, pos=vector(0,0,0,),opacity=1, color=color.blue,axis=vector(0,0,1))
    cylinder(pos=vector(0,-8,0), radius=100, axis=vector(0,1,0))
    
    
    back=box(length=3,width=2,height=.4,opacity=1,pos=vector(0,0,0,))
    middle=box(length=3,width=2,height=.4,opacity=1,pos=vector(0,0,0,))
    ring=box(length=3,width=2,height=.4,opacity=1,pos=vector(0,0,0,))
    little=box(length=3,width=2,height=.4,opacity=1,pos=vector(0,0,0,))
    index=box(length=3,width=2,height=.4,opacity=1,pos=vector(0,0,0,))
    thumb=box(length=3,width=2,height=.4,opacity=1,pos=vector(0,0,0,))
    if 'Roll 7' in data.columns:
        wrist=box(length=3,width=2,height=.4,opacity=1,pos=vector(10,0,4,))
    
    for i in range(len(data)):
        
        k5, s5, v5, vrot5 = obj_orientation(bias, data, i, 5, back, 6, vector(0,0,0))
        
        # middle finger
        obj_orientation(bias, data, i, 3, middle, 4, 4.5*k5 +3*s5 +6*k5)
        
        # ring finger
        obj_orientation(bias, data, i, 2, ring, 3.5, k5*4 +k5*6)
        
        # little finger
        obj_orientation(bias, data, i, 1, little, 3, k5*3.5 -3*s5 +k5*6)
        
        # index finger
        obj_orientation(bias, data, i, 4, index, 3.5, k5*4.5 +5.5*s5 +k5*6)
        
        # thumb
        obj_orientation(bias, data, i, 6, thumb, 6, k5*6 +6*s5 -1*k5 -2*vrot5)
        
        
        
        # wrist
        if 'Roll 7' in data.columns:
            obj_orientation(bias, data, i, 6, wrist, 0, 0)
        
        time.sleep(data.loc[i,'Delta t'])
    

    

cal_raw = pd.read_csv ('Documents\Python\Data\Calibration_{}'.format(trial_number),names=names_no_wrist)
cal_cooked_data = cook_data(cal_raw, names=names_no_wrist)
cal_cooked_data, calibration_coef = calibrate(cal_cooked_data, cal_cooked_data, (3,999))
plot_cooked_data(cal_cooked_data,calibration=True)
cal_euler_data, mad = data_to_euler_angles(cal_cooked_data,0.05)
plot_euler_data(cal_euler_data,calibration=True)

raw = pd.read_csv ('Documents\Python\Data\Trial_{}'.format(trial_number),names=names_no_wrist)
cooked_data = cook_data(raw, names_no_wrist)
cooked_data = calibrate(cal_cooked_data, cooked_data, (3,999))
plot_cooked_data(cooked_data)
euler_data = data_to_euler_angles(cooked_data,0.05)
plot_euler_data(euler_data)



animation(cal_euler_data.loc[0:1200,:].reset_index(), None)




# Draft   

#full_path_1 = openbci_path+'\OpenBCISession_2021-08-20_17-14-51\OpenBCI-RAW-2021-08-20_17-15-09.txt'
#emg_raw = pd.read_csv (full_path_1,skiprows=(4))


data = cook_data(raw,names)
euler_data = data_to_euler_angles(data)
 
data = cook_data(raw,names)
mad = Madgwick(beta = 0.05)
    
g = [warm.loc[1,'Gyro X 5'],warm.loc[1,'Gyro Y 5'],warm.loc[1,'Gyro Z 5']]
a = [warm.loc[0,'Accel X 5'],warm.loc[0,'Accel Y 5'],warm.loc[0,'Accel Z 5']]
t = warm.loc[1,'Timestamp']-warm.loc[0,'Timestamp']
mad.update_imu(gyroscope=g,accelerometer=a,delta_t=t)
mad.quaternion.to_euler_angles()

def q0 (ax0,ay0,az0):
    V0 = np.array([-ay0,ax0,0.0]) 
    teta0 = np.arccos(-az0)
    dot = np.dot(np.sin(teta0/2),(V0/np.linalg.norm(V0)))
    return np.quaternion(np.cos(teta0/2),dot[0],dot[1],dot[2])
    
def new_q (old_q,w,delta_t):
    
    
    gamma = old_q*(np.dot(delta_t,w))*np.conjugate(old_q)
    fi = np.quaternion(1, gamma[0]/2, gamma[1]/2, gamma[2]/2)
    return fi*old_q
    
def new_q_b (old_q,w,delta_t):
    norm_w = np.linalg.norm(w)
    delta_q = np.quaternion(delta_t*norm_w)
    

def correct_q_by_gravity (q, a):
    A = q*a*np.conjugate(q)
    psi = np.quaternion[1,A[]]
    
#raw_cal = cook_data(raw_cal,names)
warm = cook_data(raw,names)
q = q0(warm.loc[0,'Accel X 5'],warm.loc[0,'Accel Y 5'],warm.loc[0,'Accel Z 5'])
w = np.array([warm.loc[1,'Gyro X 5'],warm.loc[1,'Gyro Y 5'],warm.loc[1,'Gyro Z 5']])
old_q = q
delta_t = warm.loc[1,'Timestamp']-warm.loc[0,'Timestamp']
n_q = new_q(old_q,w,delta_t)

q = np.quaternion(1,2,3,4)
b = q*np.array([1,2,3])*q.conjugate()

np.conjugate(np.quaternion(1,2,3,4))
  

for sensor in names:
    print(raw.loc[0,sensor])
    
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(cooked_data['Accel Z 5'])                