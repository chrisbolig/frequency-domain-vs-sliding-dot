"""
Created on Thu Feb 24 23:14:05 2021

@author: cbolig
"""
import sounddevice as sd
from scipy.io.wavfile import write
from scipy.io.wavfile import read
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib import animation
from scipy.fft import fft, ifft, fftfreq


class domain_FvsT():

    def __init__(self, fs, recording_time, downsample_factor, yourpath):
        """



        """
        self.fs = fs
        self.time = recording_time
        self.scale = downsample_factor
        self.mypath = yourpath

    def record(self):
        """

        file_name: str, file name to save wav recording
        fs: int, sample rate
        time: int, audio length in time(s)
        returns: None, saves the sound array

         """

        sd.default.samplerate = self.fs
        print(str(self.time)+'(s) recording has started!!!')
        myrecording = sd.rec(int(self.time * fs),
                             samplerate=self.fs, channels=1)
        sd.wait()
        write(self.mypath+'myrecording.wav', self.fs, myrecording)

    def play(self, s):
        """

        s: array, sound data.
        fs: int, sample rate
        returns: None, plays the sound array
        """

        s = s/max(s)
        sd.play(s, samplerate=(self.fs/self.scale), blocking=True)

    def formdata(self, wavpath):
        """

        wavpath: str, file name
        time: int, audio length in time(s), default is 5
        N: int, down samples by every subsequent Nth item
        forms low res audio with low res time

        """

        a = read(wavpath)
        try:
            m, n = a[1].shape
            soundDS = a[1].T[0][::self.scale]
            a = a[1].T[0]
        except:
            soundDS = a[1][::self.scale]
            a = a[1]
        self.a = soundDS
        self.time_data = np.linspace(0, time, len(soundDS))
        self.dt = self.time_data[1]

    def Convolution_value(self, X, tvar):
        """

        helper function for Convolution

        """

        movingX = X(tvar)
        return (movingX@self.a)*self.dt

    def Convolution(self, func):
        """

        func: function, function that will slide with time
        time_data: array, array of time data
        returns: Convolution array, approx integral method using dot
        product, uses tqdm to show load time 

        """

        return np.array([self.Convolution_value(func, j) for j in tqdm(self.time_data)])

    def FFT(self, signal):
        """

        signal: array, signal under test
        fs: int, sample rate
        returns: (discrete fourier, freq values), uses fft algo for forier values

        """

        N = len(self.a)
        T = 1/self.fs
        yf = fft(signal)
        xf = fftfreq(N, T)[:N//2]
        plt.plot(xf, np.abs(yf[0:N//2]))
        plt.grid()
        plt.show()
        return yf, xf

    def FFT_Convolv(self, yfa, yfX):
        """

        yfa: array, fft of voice audio
        yfX: array, mixing audio
        returns: array, convolution of audio using fft .
        warning: mixing audio decays. For most accurate fft, signal is centerd in middle
        of time array. The inv. fft. convolution audio needs to be shifted since
        the time delay is captured and effects results

        """

        FFTconvol = yfa*yfX
        FFTconvolinv = ifft(FFTconvol).real
        self.fftconvolv = np.roll(FFTconvolinv, -(len(FFTconvolinv))//2)
        return self.fftconvolv

    def delta3(self, t_varialbe=0):
        """

        x: array, time array
        t_varialbe: float, root of function
        func_Tau: boolean, treat function moving with time condition
        Hz: float, freq if function needs it
        returns: array centered at t_varialbe value 

        """

        ind, = np.where(self.time_data >= t_varialbe)
        y = np.zeros_like(self.time_data)
        y[ind[0]] = 1/self.dt
        y[ind[0]-100] = -1/self.dt
        y[ind[0]-200] = 1/self.dt
        return y

    def diracfunc(self, t_varialbe=0):

        ind, = np.where(self.time_data >= t_varialbe)
        y = np.zeros_like(self.time_data)
        y[ind[0]] = 1/self.dt
        return y

    def sinc_abs(self, t_varialbe=0, Hz=66):

        x = np.copy(t_varialbe-self.time_data)
        ind, = np.where(x == 0)
        x[ind] = 10e-6
        y = abs((np.sin(Hz*2*np.pi*x)/(Hz*2*np.pi*x)))
        return y/np.max(y)

    def gausswave(self, t_varialbe=0, Hz=66):
        sig = .1
        mu = 0
        t = np.copy(t_varialbe-self.time_data)
        y = 1/(sig*np.sqrt(np.pi*2))*np.exp(-.5 *
                                            ((t-mu)/sig)**2)*np.cos(Hz * 2*np.pi*t)
        return abs(y)/np.max(y)

    def gausswave3(self, t_varialbe=0, Hz=33):
        t = np.copy(t_varialbe-self.time_data)
        y = np.cos(66*2*np.pi*t)*(np.exp(-1000*(t)**2) +
                                  np.exp(-1000*(t-.2)**2)+np.exp(-1000*(t+.2)**2))
        return abs(y)/np.max(y)

    def funct(self, X, tvar):
        """

        X: function that will slide with time
        returns: array centered at tvar value 

        """
        return (X(tvar))


# %%

fs = 44100
time = 5
scale = 2
mypath = '/Users/cbolig/Documents/spyder/fun math projects/soundfiles/'

C = domain_FvsT(fs, time, scale, mypath)


# %%

C.record()


# %%

C.formdata(mypath+'myrecording.wav')
recording = C.a
C.play(recording)


# %%

# fut: function under test

# fut = C.delta3
# fut = C.diracfunc
# fut = C.gaussWave
# fut = C.gausswave3
fut = C.sinc_abs

convolution_array = C.Convolution(fut)


# %%

plotfunc = C.funct(fut, time/2)
plt.plot(plotfunc)


# %%

C.play(convolution_array)


# %%


fftconvolv = C.FFT_Convolv(C.FFT(recording)[0], C.FFT(C.funct(fut, time/2))[0])


# %%

C.play(convolution_array)


# %%


def plotresults(time_data, invfft, convolution_array, a, X):
    '''
    time_data: array, array of time data
    convolution_array: array, Convolution array returned from Convolution func
    a: array, orginal audio data
    X: function, function used in convolution
    returns: None, plots three resulting graphs 
    '''
    plt.figure(0)
    plt.plot(time_data, invfft)
    plt.title('fourier convolution')
    plt.grid()
    plt.show()
    plt.figure(1)
    plt.plot(time_data, convolution_array)
    plt.title('integral convolution')
    plt.grid()
    plt.show()
    plt.figure(2)
    plt.plot(time_data, a)
    plt.title('original audio')
    plt.grid()
    plt.show()
    plt.figure(3)
    plt.plot(time_data, X)
    plt.title('mixing audio')
    plt.grid()
    plt.show()


plotresults(C.time_data, fftconvolv, convolution_array, recording, plotfunc)

# %%

time_data = C.time_data
dt = time_data[1]
speed_factor = 200
fig, (ax, ax2) = plt.subplots(figsize=(16, 7), ncols=2)
line, = ax.plot([], [], lw=2, color='k', alpha=.5)
line2, = ax2.plot([], [], lw=2, color='crimson', alpha=.5)
convolution_array = convolution_array/np.max(convolution_array)
ax.plot(time_data, recording/np.max(recording), lw=2, alpha=.5)
ax2.set_xlim(time_data[0], time_data[-1])
ax.set_ylim((-1.2, 1.2))
ax2.set_ylim((-1.2, 1.2))


def init():
    return (line, line2)


def animate(i):
    print(i)
    inc = + i*speed_factor
    yslide = C.funct(fut, inc*dt)
    line.set_data(time_data, yslide)
    line2.set_data(time_data[:int(inc)], convolution_array[:int(inc)])
    return (line, line2)


frames = int(len(recording)/speed_factor)
anim = animation.FuncAnimation(
    fig, animate, frames=frames, init_func=init, blit=True, interval=1/3, repeat=False)
