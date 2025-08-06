"""
@ Correlation analysis for large-scale turbulence
@ author ZimoJupiter
@ w.zimo@outlook.com
@ date 10 Mar 2025
@ license MIT License
"""
import numpy as np
import pandas as pd
from numpy import pi, exp, sqrt
import scipy
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from statsmodels.tsa.stattools import acf, ccf
plt.rcParams['font.weight'] = 'normal'
plt.rcParams["figure.figsize"] = (3.2, 3.2*3/4)
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 600
plt.rcParams['font.family'] = 'Times New Roman'

def ReadData():
    EData = np.array(pd.read_csv('Data/eugene_processed.csv', header=0)) # datetime wspd wdir wgst
    AData = np.array(pd.read_csv('Data/ameranda_pass_processed.csv', header=0)) # datetime wl wspd wdir wgst
    GData = np.array(pd.read_csv('Data/grand_isle_processed.csv', header=0)) # datetime wl wspd wdir wgst
    CData = np.array(pd.read_csv('Data/USGS_CaillouBay_processed.csv', header=0)) #datetime gage_height temp sp_cond wspd wdir sal

    return EData[:, [0, 1]], AData[:, [0, 2]], GData[:, [0, 2]], CData[:, [0, 4]]

def PlotInitialData(EData, AData, GData, CData):
    fig, ax = plt.subplots(4, 1, figsize=(3.2, 3.2*3/4*2))
    dates = [datetime.strptime(date, '%m/%d/%Y %H:%M') for date in EData[:, 0]]
    ax[0].plot(dates, EData[:, 1], 'b', linewidth=0.1)
    ax[0].set_title('Eugene Island')
    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('Wind Speed (m/s)')
    ax[0].xaxis.set_major_locator(mdates.YearLocator(base=1))
    ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    dates = [datetime.strptime(date, '%m/%d/%Y %H:%M') for date in AData[:, 0]]
    ax[1].plot(dates, AData[:, 1], 'b', linewidth=0.1)
    ax[1].set_title('Amerada Pass')
    ax[1].set_xlabel('Time')
    ax[1].set_ylabel('Wind Speed (m/s)')
    ax[1].xaxis.set_major_locator(mdates.YearLocator(base=2))
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    dates = [datetime.strptime(date, '%m/%d/%Y %H:%M') for date in GData[:, 0]]
    ax[2].plot(dates, GData[:, 1], 'b', linewidth=0.1)
    ax[2].set_title('Grand Isle')
    ax[2].set_xlabel('Time')
    ax[2].set_ylabel('Wind Speed (m/s)')
    ax[2].xaxis.set_major_locator(mdates.YearLocator(base=4))
    ax[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    dates = [datetime.strptime(date, '%m/%d/%Y %H:%M') for date in CData[:, 0]]
    ax[3].plot(dates, CData[:, 1], 'b', linewidth=0.1)
    ax[3].set_title('Caillou Bay')
    ax[3].set_xlabel('Time')
    ax[3].set_ylabel('Wind Speed (m/s)')
    ax[3].xaxis.set_major_locator(mdates.YearLocator(base=1))
    ax[3].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.tight_layout()
    plt.savefig('Figures/InitialData.png')

def Detrend(EData, AData, GData, CData):
    EData_detrended = np.array(EData[:, 1], dtype=np.float64)
    mask = ~np.isnan(EData_detrended)
    EData_detrended[mask] = scipy.signal.detrend(EData_detrended[mask])
 
    AData_detrended = np.array(AData[:, 1], dtype=np.float64)
    mask = ~np.isnan(AData_detrended)
    AData_detrended[mask] = scipy.signal.detrend(AData_detrended[mask])
    
    GData_detrended = np.array(GData[:, 1], dtype=np.float64)
    mask = ~np.isnan(GData_detrended)
    GData_detrended[mask] = scipy.signal.detrend(GData_detrended[mask])

    CData_detrended = np.array(CData[:, 1], dtype=np.float64)
    mask = ~np.isnan(CData_detrended)
    CData_detrended[mask] = scipy.signal.detrend(CData_detrended[mask])

    fig, ax = plt.subplots(4, 1, figsize=(3.2, 3.2*3/4*2))
    dates = [datetime.strptime(date, '%m/%d/%Y %H:%M') for date in EData[:, 0]]
    ax[0].plot(dates, EData_detrended, 'b', linewidth=0.1)
    ax[0].set_title('Eugene Island')
    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('Wind Speed (m/s)')
    ax[0].xaxis.set_major_locator(mdates.YearLocator(base=1))
    ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    dates = [datetime.strptime(date, '%m/%d/%Y %H:%M') for date in AData[:, 0]]
    ax[1].plot(dates, AData_detrended, 'b', linewidth=0.1)
    ax[1].set_title('Amerada Pass')
    ax[1].set_xlabel('Time')
    ax[1].set_ylabel('Wind Speed (m/s)')
    ax[1].xaxis.set_major_locator(mdates.YearLocator(base=2))
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    dates = [datetime.strptime(date, '%m/%d/%Y %H:%M') for date in GData[:, 0]]
    ax[2].plot(dates, GData_detrended, 'b', linewidth=0.1)
    ax[2].set_title('Grand Isle')
    ax[2].set_xlabel('Time')
    ax[2].set_ylabel('Wind Speed (m/s)')
    ax[2].xaxis.set_major_locator(mdates.YearLocator(base=4))
    ax[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    dates = [datetime.strptime(date, '%m/%d/%Y %H:%M') for date in CData[:, 0]]
    ax[3].plot(dates, CData_detrended, 'b', linewidth=0.1)
    ax[3].set_title('Caillou Bay')
    ax[3].set_xlabel('Time')
    ax[3].set_ylabel('Wind Speed (m/s)')
    ax[3].xaxis.set_major_locator(mdates.YearLocator(base=1))
    ax[3].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.tight_layout()
    plt.savefig('Figures/InitialData_Detrended.png')

    return EData_detrended, AData_detrended, GData_detrended, CData_detrended

def FFT(EData_detrended, AData_detrended, GData_detrended, CData_detrended):
    EData_detrended_FFT = EData_detrended[~np.isnan(EData_detrended)]
    AData_detrended_FFT = AData_detrended[~np.isnan(AData_detrended)]
    GData_detrended_FFT = GData_detrended[~np.isnan(GData_detrended)]
    CData_detrended_FFT = CData_detrended[~np.isnan(CData_detrended)]

    FFT_E = np.fft.fft(EData_detrended_FFT)
    FFT_E = 2/FFT_E.shape[0]*np.abs(FFT_E)
    FFT_E -= np.mean(FFT_E)
    f_E = np.fft.fftfreq(EData_detrended_FFT.shape[0], 1)
    FFT_A = np.fft.fft(AData_detrended_FFT)
    FFT_A = 2/FFT_A.shape[0]*np.abs(FFT_A)
    FFT_A -= np.mean(FFT_A)
    f_A = np.fft.fftfreq(AData_detrended_FFT.shape[0], 1)
    FFT_G = np.fft.fft(GData_detrended_FFT)
    FFT_G = 2/FFT_G.shape[0]*np.abs(FFT_G)
    FFT_G -= np.mean(FFT_G)
    f_G = np.fft.fftfreq(GData_detrended_FFT.shape[0], 1)
    FFT_C = np.fft.fft(CData_detrended_FFT)
    FFT_C = 2/FFT_C.shape[0]*np.abs(FFT_C)
    FFT_C -= np.mean(FFT_C)
    f_C = np.fft.fftfreq(CData_detrended_FFT.shape[0], 1)

    fig, ax = plt.subplots(2, 2, figsize=(3.2*2, 3.2*3/4*2))
    lag = np.arange(10*24)
    ax[0,0].plot(24*f_E[:FFT_E.shape[0]//2], FFT_E[:FFT_E.shape[0]//2], 'b')
    ax[0,0].set_title('Eugene Island')
    ax[0,0].set_xlabel('Frequency (1/day)')
    ax[0,0].set_ylabel('Amplitude')
    ax[0,0].set_xticks(np.arange(0, 6))
    ax[0,0].set_xlim(-0.5,5.5)
    ax[0,0].grid(True)
    ax[0,1].plot(24*f_A[:FFT_A.shape[0]//2], FFT_A[:FFT_A.shape[0]//2], 'b')
    ax[0,1].set_title('Amerada Pass')
    ax[0,1].set_xlabel('Frequency (1/day)')
    ax[0,1].set_ylabel('Amplitude')
    ax[0,1].set_xticks(np.arange(0, 6))
    ax[0,1].set_xlim(-0.5,5.5)
    ax[0,1].grid(True)
    ax[1,0].plot(24*f_G[:FFT_G.shape[0]//2], FFT_G[:FFT_G.shape[0]//2], 'b') 
    ax[1,0].set_title('Grand Isle')
    ax[1,0].set_xlabel('Frequency (1/day)')
    ax[1,0].set_ylabel('Amplitude')
    ax[1,0].set_xticks(np.arange(0, 6))
    ax[1,0].set_xlim(-0.5,5.5)
    ax[1,0].grid(True)
    ax[1,1].plot(24*f_C[:FFT_C.shape[0]//2], FFT_C[:FFT_C.shape[0]//2], 'b')
    ax[1,1].set_title('Caillou Bay')
    ax[1,1].set_xlabel('Frequency (1/day)')
    ax[1,1].set_ylabel('Amplitude')
    ax[1,1].set_xticks(np.arange(0, 6))
    ax[1,1].set_xlim(-0.5,5.5)
    ax[1,1].grid(True)
    plt.tight_layout()
    plt.savefig('Figures/FFT.png')

    fig, ax = plt.subplots(2, 2, figsize=(3.2*2, 3.2*3/4*2))
    lag = np.arange(10*24)
    ax[0,0].plot(24*360*f_E[:FFT_E.shape[0]//2], FFT_E[:FFT_E.shape[0]//2], 'b')
    ax[0,0].set_title('Eugene Island')
    ax[0,0].set_xlabel('Frequency (1/year)')
    ax[0,0].set_ylabel('Amplitude')
    ax[0,0].set_xticks(np.arange(0, 6))
    ax[0,0].set_xlim(-0.5,5.5)
    ax[0,0].grid(True)
    ax[0,1].plot(24*360*f_A[:FFT_A.shape[0]//2], FFT_A[:FFT_A.shape[0]//2], 'b')
    ax[0,1].set_title('Amerada Pass')
    ax[0,1].set_xlabel('Frequency (1/year)')
    ax[0,1].set_ylabel('Amplitude')
    ax[0,1].set_xticks(np.arange(0, 6))
    ax[0,1].set_xlim(-0.5,5.5)
    ax[0,1].grid(True)
    ax[1,0].plot(24*360*f_G[:FFT_G.shape[0]//2], FFT_G[:FFT_G.shape[0]//2], 'b') 
    ax[1,0].set_title('Grand Isle')
    ax[1,0].set_xlabel('Frequency (1/year)')
    ax[1,0].set_ylabel('Amplitude')
    ax[1,0].set_xticks(np.arange(0, 6))
    ax[1,0].set_xlim(-0.5,5.5)
    ax[1,0].grid(True)
    ax[1,1].plot(24*360*f_C[:FFT_C.shape[0]//2], FFT_C[:FFT_C.shape[0]//2], 'b')
    ax[1,1].set_title('Caillou Bay')
    ax[1,1].set_xlabel('Frequency (1/year)')
    ax[1,1].set_ylabel('Amplitude')
    ax[1,1].set_xticks(np.arange(0, 6))
    ax[1,1].set_xlim(-0.5,5.5)
    ax[1,1].grid(True)
    plt.tight_layout()
    plt.savefig('Figures/FFT_year.png')

def AC(EData_detrended, AData_detrended, GData_detrended, CData_detrended):
    acf_E = acf(EData_detrended, nlags=360*24, missing='conservative')
    acf_A = acf(AData_detrended, nlags=360*24, missing='conservative')
    acf_G = acf(GData_detrended, nlags=360*24, missing='conservative')
    acf_C = acf(CData_detrended, nlags=360*24, missing='conservative')

    fig, ax = plt.subplots(2, 2, figsize=(3.2*2, 3.2*3/4*2))
    lag = np.arange(10*24)
    ax[0,0].plot(lag/24, acf_E[:10*24], 'b')
    ax[0,0].set_title('Eugene Island')
    ax[0,0].set_xlabel('Lag (days)')
    ax[0,0].set_ylabel('Auto-correlation')
    ax[0,0].set_xticks(np.arange(lag[0]//24, lag[-1]//24+2))
    ax[0,0].set_yticks(np.arange(-0.5, 1.5, 0.5))
    ax[0,0].set_ylim(-0.1,1.1)
    ax[0,0].grid(True)
    ax[0,1].plot(lag/24, acf_A[:10*24], 'b')
    ax[0,1].set_title('Amerada Pass')
    ax[0,1].set_xlabel('Lag (days)')
    ax[0,1].set_ylabel('Auto-correlation')
    ax[0,1].set_xticks(np.arange(lag[0]//24, lag[-1]//24+2))
    ax[0,1].set_yticks(np.arange(-0.5, 1.5, 0.5))
    ax[0,1].set_ylim(-0.1,1.1)
    ax[0,1].grid(True)
    ax[1,0].plot(lag/24, acf_G[:10*24], 'b') 
    ax[1,0].set_title('Grand Isle')
    ax[1,0].set_xlabel('Lag (days)')
    ax[1,0].set_ylabel('Auto-correlation')
    ax[1,0].set_xticks(np.arange(lag[0]//24, lag[-1]//24+2))
    ax[1,0].set_yticks(np.arange(-0.5, 1.5, 0.5))
    ax[1,0].set_ylim(-0.1,1.1)
    ax[1,0].grid(True)
    ax[1,1].plot(lag/24, acf_C[:10*24], 'b')
    ax[1,1].set_title('Caillou Bay')
    ax[1,1].set_xlabel('Lag (days)')
    ax[1,1].set_ylabel('Auto-correlation')
    ax[1,1].set_xticks(np.arange(lag[0]//24, lag[-1]//24+2))
    ax[1,1].set_yticks(np.arange(-0.5, 1.5, 0.5))
    ax[1,1].set_ylim(-0.1,1.1)
    ax[1,1].grid(True)
    plt.tight_layout()
    plt.savefig('Figures/AutoCorrelation_nowindows.png')

    TL_E_mean = np.zeros((7*24))
    TL_A_mean = np.zeros((7*24))
    TL_G_mean = np.zeros((7*24))
    TL_C_mean = np.zeros((7*24))

    for j in range(7*24):
        win = 7*24+j
        TL_E = np.zeros((EData_detrended.shape[0]//win-1))
        first_window = 0
        for i in range(EData_detrended.shape[0]//win-1):
            if np.isnan(EData_detrended[i*win : (i+1)*win+1]).any():
                continue
            else:
                acf_values = acf(EData_detrended[i*win : (i+1)*win+1], nlags=10000, missing='conservative')
                tau_0 = np.where(acf_values < 0)[0][0]
                TL_E[i] = np.sum(acf_values[:tau_0])
                if win == 7*24 and first_window == 0:
                    first_window = 1
                    acf_E_7 = acf_values
                if win == 7*24 + 7*24 - 1 and first_window == 0:
                    first_window = 1
                    acf_E_14 = acf_values
        TL_E_mean[j] = np.mean(TL_E[TL_E!=0])

    for j in range(7*24):
        win = 7*24+j
        TL_A = np.zeros((AData_detrended.shape[0]//win-1))
        first_window = 0
        for i in range(AData_detrended.shape[0]//win-1):
            if np.isnan(AData_detrended[i*win : (i+1)*win+1]).any():
                continue
            else:
                acf_values = acf(AData_detrended[i*win : (i+1)*win+1], nlags=10000, missing='conservative')
                tau_0 = np.where(acf_values < 0)[0][0]
                TL_A[i] = np.sum(acf_values[:tau_0])
                if win == 7*24 and first_window == 0:
                    first_window = 1
                    acf_A_7 = acf_values
                if win == 7*24 + 7*24 - 1 and first_window == 0:
                    first_window = 1
                    acf_A_14 = acf_values
        TL_A_mean[j] = np.mean(TL_A[TL_A!=0])

    for j in range(7*24):
        win = 7*24+j
        TL_G = np.zeros((GData_detrended.shape[0]//win-1))
        first_window = 0
        for i in range(GData_detrended.shape[0]//win-1):
            if np.isnan(GData_detrended[i*win : (i+1)*win+1]).any():
                continue
            else:
                acf_values = acf(GData_detrended[i*win : (i+1)*win+1], nlags=10000, missing='conservative')
                tau_0 = np.where(acf_values < 0)[0][0]
                TL_G[i] = np.sum(acf_values[:tau_0])
                if win == 7*24 and first_window == 0:
                    first_window = 1
                    acf_G_7 = acf_values
                if win == 7*24 + 7*24 - 1 and first_window == 0:
                    first_window = 1
                    acf_G_14 = acf_values
        TL_G_mean[j] = np.mean(TL_G[TL_G!=0])

    for j in range(7*24):
        win = 7*24+j
        TL_C = np.zeros((CData_detrended.shape[0]//win-1))
        first_window = 0
        for i in range(CData_detrended.shape[0]//win-1):
            if np.isnan(CData_detrended[i*win : (i+1)*win+1]).any():
                continue
            else:
                acf_values = acf(CData_detrended[i*win : (i+1)*win+1], nlags=10000, missing='conservative')
                tau_0 = np.where(acf_values < 0)[0][0]
                TL_C[i] = np.sum(acf_values[:tau_0])
                if win == 7*24 and first_window == 0:
                    first_window = 1
                    acf_C_7 = acf_values
                if win == 7*24 + 7*24 - 1 and first_window == 0:
                    first_window = 1
                    acf_C_14 = acf_values
        TL_C_mean[j] = np.mean(TL_C[TL_C!=0])

    fig, ax = plt.subplots(2, 2, figsize=(3.2*2, 3.2*3/4*2))
    lag7 = np.arange(7*24)
    lag14 = np.arange(2*7*24)
    ax[0,0].plot(lag7/24, acf_E_7[:7*24], 'r', label='Window lenghth = 7 days')
    ax[0,0].plot(lag14/24, acf_E_14[:2*7*24], 'b', label='Window lenghth = 14 days')
    ax[0,0].set_title('Eugene Island')
    ax[0,0].set_xlabel('Lag (days)')
    ax[0,0].set_ylabel('Auto-correlation')
    ax[0,0].set_xticks(np.arange(lag14[0]//24, lag14[-1]//24+2))
    ax[0,0].set_yticks(np.arange(-0.5, 1.5, 0.5))
    ax[0,0].set_ylim(-0.6,1.1)
    ax[0,0].grid(True)
    ax[0,0].legend(facecolor='white', edgecolor='white')
    ax[0,1].plot(lag7/24, acf_A_7[:7*24], 'r', label='Window lenghth = 7 days')
    ax[0,1].plot(lag14/24, acf_A_14[:2*7*24], 'b', label='Window lenghth = 14 days')
    ax[0,1].set_title('Amerada Pass')
    ax[0,1].set_xlabel('Lag (days)')
    ax[0,1].set_ylabel('Auto-correlation')
    ax[0,1].set_xticks(np.arange(lag14[0]//24, lag14[-1]//24+2))
    ax[0,1].set_yticks(np.arange(-0.5, 1.5, 0.5))
    ax[0,1].set_ylim(-0.6,1.1)
    ax[0,1].grid(True)
    ax[0,1].legend(facecolor='white', edgecolor='white')
    ax[1,0].plot(lag7/24, acf_G_7[:7*24], 'r', label='Window lenghth = 7 days')
    ax[1,0].plot(lag14/24, acf_G_14[:2*7*24], 'b', label='Window lenghth = 14 days')   
    ax[1,0].set_title('Grand Isle')
    ax[1,0].set_xlabel('Lag (days)')
    ax[1,0].set_ylabel('Auto-correlation')
    ax[1,0].set_xticks(np.arange(lag14[0]//24, lag14[-1]//24+2))
    ax[1,0].set_yticks(np.arange(-0.5, 1.5, 0.5))
    ax[1,0].set_ylim(-0.6,1.1)
    ax[1,0].grid(True)
    ax[1,0].legend(facecolor='white', edgecolor='white')
    ax[1,1].plot(lag7/24, acf_C_7[:7*24], 'r', label='Window lenghth = 7 days')
    ax[1,1].plot(lag14/24, acf_C_14[:2*7*24], 'b', label='Window lenghth = 14 days')
    ax[1,1].set_title('Caillou Bay')
    ax[1,1].set_xlabel('Lag (days)')
    ax[1,1].set_ylabel('Auto-correlation')
    ax[1,1].set_xticks(np.arange(lag14[0]//24, lag14[-1]//24+2))
    ax[1,1].set_yticks(np.arange(-0.5, 1.5, 0.5))
    ax[1,1].set_ylim(-0.6,1.1)
    ax[1,1].grid(True)
    ax[1,1].legend(facecolor='white', edgecolor='white')
    plt.tight_layout()
    plt.savefig('Figures/AutoCorrelation.png')

    windows = np.arange(7*24, 2*7*24)
    plt.figure()
    plt.plot(windows/24, TL_E_mean, 'r', linewidth=0.5, label='Eugene Island')
    plt.plot(windows/24, TL_A_mean, 'b', linewidth=0.5, label='Amerada Pass')
    plt.plot(windows/24, TL_G_mean, 'g', linewidth=0.5, label='Grand Isle')
    plt.plot(windows/24, TL_C_mean, 'orange', linewidth=0.5, label='Caillou Bay')
    plt.xlabel('Window length (days)')
    plt.ylabel('Averaged integral time scale')
    plt.legend(loc='upper left',facecolor='white', edgecolor='white', ncol=2, handletextpad=0.2, borderpad=0, columnspacing=0.5)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('Figures/Integral_time_scale.png')

def CC(EData_detrended, AData_detrended, GData_detrended, CData_detrended):
    Slice_A = AData_detrended[int(np.where(AData=='01/01/2019 00:00')[0]):int(np.where(AData=='01/01/2022 00:00')[0])]
    Slice_G = GData_detrended[int(np.where(GData=='01/01/2019 00:00')[0]):int(np.where(GData=='01/01/2022 00:00')[0])]
    Slice_C = CData_detrended[int(np.where(CData=='01/01/2019 0:00')[0]):int(np.where(CData=='01/01/2022 0:00')[0])]

    Slice_AC_A = Slice_A.copy()
    Slice_AC_C = Slice_C.copy()
    Slice_GC_G = Slice_G.copy()
    Slice_GC_C = Slice_C.copy()

    Slice_AC_A[np.isnan(Slice_C)] = np.nan
    Slice_AC_C[np.isnan(Slice_A)] = np.nan
    Slice_GC_G[np.isnan(Slice_C)] = np.nan
    Slice_GC_C[np.isnan(Slice_G)] = np.nan

    fig, ax = plt.subplots(1, 2, figsize=(3.2*2, 3.2*3/4))
    dates = [datetime.strptime(date, '%m/%d/%Y %H:%M') for date in \
             AData[int(np.where(AData=='01/01/2019 00:00')[0]):int(np.where(AData=='01/01/2022 00:00')[0]), 0]]
    ax[0].plot(dates, Slice_AC_A, 'b', linewidth=0.1, alpha=0.8, label='Amerada Pass')
    ax[0].plot(dates, Slice_AC_C, 'r', linewidth=0.1, alpha=0.4, label='Caillou Bay')
    ax[0].set_title('Amerada Pass and Caillou Bay')
    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('Wind Speed (m/s)')
    ax[0].xaxis.set_major_locator(mdates.YearLocator(base=1))
    ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax[0].legend(facecolor='white', edgecolor='white')
    dates = [datetime.strptime(date, '%m/%d/%Y %H:%M') for date in \
             GData[int(np.where(GData=='01/01/2019 00:00')[0]):int(np.where(GData=='01/01/2022 00:00')[0]), 0]]
    ax[1].plot(dates, Slice_GC_G, 'b', linewidth=0.1, alpha=0.8, label='Grand Isle')
    ax[1].plot(dates, Slice_GC_C, 'r', linewidth=0.1, alpha=0.4, label='Caillou Bay')
    ax[1].set_title('Grand Isle and Caillou Bay')
    ax[1].set_xlabel('Time')
    ax[1].set_ylabel('Wind Speed (m/s)')
    ax[1].xaxis.set_major_locator(mdates.YearLocator(base=1))
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax[1].legend(facecolor='white', edgecolor='white')
    plt.tight_layout()
    plt.savefig('Figures/CrossCorrelation_Data.png')

    Slice_AC_A = Slice_AC_A[~np.isnan(Slice_AC_A)]
    Slice_AC_C = Slice_AC_C[~np.isnan(Slice_AC_C)]
    Slice_GC_G = Slice_GC_G[~np.isnan(Slice_GC_G)]
    Slice_GC_C = Slice_GC_C[~np.isnan(Slice_GC_C)]

    CCF_AC = ccf(Slice_AC_A, Slice_AC_C, unbiased=False)
    CCF_GC = ccf(Slice_GC_G, Slice_GC_C, unbiased=False)

    fig, ax = plt.subplots(1, 2, figsize=(3.2*2, 3.2*3/4))
    lag = np.arange(10*24)
    ax[0].plot(lag/24, CCF_AC[:10*24], 'b')
    ax[0].set_title('Amerada Pass and Caillou Bay')
    ax[0].set_xlabel('Lag (days)')
    ax[0].set_ylabel('Auto-correlation')
    ax[0].set_xticks(np.arange(lag[0]//24, lag[-1]//24+2))
    ax[0].set_yticks(np.arange(-0.5, 1.5, 0.5))
    ax[0].set_ylim(-0.1,1.1)
    ax[0].grid(True)
    ax[1].plot(lag/24, CCF_GC[:10*24], 'b')
    ax[1].set_title('Gand Isle and Caillou Bay')
    ax[1].set_xlabel('Lag (days)')
    ax[1].set_ylabel('Auto-correlation')
    ax[1].set_xticks(np.arange(lag[0]//24, lag[-1]//24+2))
    ax[1].set_yticks(np.arange(-0.5, 1.5, 0.5))
    ax[1].set_ylim(-0.1,1.1)
    ax[1].grid(True)
    plt.tight_layout()
    plt.savefig('Figures/CrossCorrelation.png')

    TL_AC_mean = np.zeros((7*24))
    TL_GC_mean = np.zeros((7*24))

    for j in range(7*24):
        win = 7*24+j
        TL_AC = np.zeros((Slice_AC_A.shape[0]//win-1))
        first_window = 0
        for i in range(Slice_AC_A.shape[0]//win-1):
            if np.isnan(Slice_AC_A[i*win : (i+1)*win+1]).any():
                continue
            else:
                ccf_values = ccf(Slice_AC_A[i*win : (i+1)*win+1], Slice_AC_C[i*win : (i+1)*win+1], unbiased=False)
                tau_0 = np.where(ccf_values < 0)[0][0]
                TL_AC[i] = np.sum(ccf_values[:tau_0])
                if win == 7*24 and first_window == 0:
                    first_window = 1
                    ccf_AC_7 = ccf_values
                if win == 7*24 + 7*24 - 1 and first_window == 0:
                    first_window = 1
                    ccf_AC_14 = ccf_values
        TL_AC_mean[j] = np.mean(TL_AC[TL_AC!=0])

    for j in range(7*24):
        win = 7*24+j
        TL_GC = np.zeros((Slice_GC_G.shape[0]//win-1))
        first_window = 0
        for i in range(Slice_GC_G.shape[0]//win-1):
            if np.isnan(Slice_GC_G[i*win : (i+1)*win+1]).any():
                continue
            else:
                ccf_values = ccf(Slice_GC_G[i*win : (i+1)*win+1], Slice_GC_C[i*win : (i+1)*win+1], unbiased=False)
                tau_0 = np.where(ccf_values < 0)[0][0]
                TL_GC[i] = np.sum(ccf_values[:tau_0])
                if win == 7*24 and first_window == 0:
                    first_window = 1
                    ccf_GC_7 = ccf_values
                if win == 7*24 + 7*24 - 1 and first_window == 0:
                    first_window = 1
                    ccf_GC_14 = ccf_values
        TL_GC_mean[j] = np.mean(TL_GC[TL_GC!=0])

    fig, ax = plt.subplots(1, 2, figsize=(3.2*2, 3.2*3/4))
    lag7 = np.arange(7*24)
    lag14 = np.arange(2*7*24)
    ax[0].plot(lag7/24, ccf_AC_7[:7*24], 'r', label='Window lenghth = 7 days')
    ax[0].plot(lag14/24, ccf_AC_14[:2*7*24], 'b', label='Window lenghth = 14 days')
    ax[0].set_title('Amerada Pass and Caillou Bay')
    ax[0].set_xlabel('Lag (days)')
    ax[0].set_ylabel('Cross-correlation')
    ax[0].set_xticks(np.arange(lag14[0]//24, lag14[-1]//24+2))
    ax[0].set_yticks(np.arange(-0.5, 1.5, 0.5))
    ax[0].set_ylim(-0.6,1.1)
    ax[0].grid(True)
    ax[0].legend(facecolor='white', edgecolor='white')
    ax[1].plot(lag7/24, ccf_GC_7[:7*24], 'r', label='Window lenghth = 7 days')
    ax[1].plot(lag14/24, ccf_GC_14[:2*7*24], 'b', label='Window lenghth = 14 days')
    ax[1].set_title('Gand Isle and Caillou Bay')
    ax[1].set_xlabel('Lag (days)')
    ax[1].set_ylabel('Cross-correlation')
    ax[1].set_xticks(np.arange(lag14[0]//24, lag14[-1]//24+2))
    ax[1].set_yticks(np.arange(-0.5, 1.5, 0.5))
    ax[1].set_ylim(-0.6,1.1)
    ax[1].grid(True)
    ax[1].legend(facecolor='white', edgecolor='white')
    plt.tight_layout()
    plt.savefig('Figures/CrossCorrelation_win.png')

    windows = np.arange(7*24, 2*7*24)
    plt.figure()
    plt.plot(windows/24, TL_AC_mean, 'r', linewidth=0.5, label='Amerada Pass and Caillou Bay')
    plt.plot(windows/24, TL_GC_mean, 'b', linewidth=0.5, label='Grand Isle and Caillou Bay')
    plt.xlabel('Window length (days)')
    plt.ylabel('Averaged integral time scale')
    plt.legend(loc='upper left',facecolor='white', edgecolor='white', ncol=1, handletextpad=0.2, borderpad=0, columnspacing=0.5)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('Figures/Integral_time_scale_CC.png')

if __name__ == '__main__':
    EData, AData, GData, CData = ReadData()
    PlotInitialData(EData, AData, GData, CData)
    EData_detrended, AData_detrended, GData_detrended, CData_detrended \
        = Detrend(EData, AData, GData, CData)
    FFT(EData_detrended, AData_detrended, GData_detrended, CData_detrended)
    AC(EData_detrended, AData_detrended, GData_detrended, CData_detrended)
    CC(EData_detrended, AData_detrended, GData_detrended, CData_detrended)

    breakpoint


