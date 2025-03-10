# Auto-correlation-integral-time-scale-for-large-scale-turbulence
Correlation analysis for wind speed fluctuation in four stations.

## Correlation analysis

In this project, wind speed data from four stations are analyzed to calculate the auto-correlation function and integral time scale. The data is first detrended using the `Scipy` library, where detrending involves subtracting the least squares fit polynomial. For data gaps, segments with missing values are ignored, and the remaining data is detrended over the time scale. The specific operations are given in the `Detrend()` function in the `main.py` script.

One of the most common methods for analyzing periodic changes in a set of signals is the Fourier transform:

$$F(\omega) = \int_{-\infty}^{\infty} f(t) e^{-i\omega t} \text{d} t,$$

This method allows direct observation of signal amplitudes ($F$) at different frequencies ($\omega$). Mastering the law of periodic changes facilitates the analysis of the changes in the auto-correlation function. The conventional spectrum analysis process also includes operations such as removing DC components and normalizing. For the specific implementation algorithm, please refer to the \textit{FFT()} function in the appendix; it will not be described in detail here.

The cross-correlation function is a statistical measure that quantifies the similarity between two signals as a function of a time lag. It is commonly used in signal processing and time series analysis to identify patterns and relationships between signals. For example, given two signals $u(t)$ and $v(t)$, the cross-correlation function $R_{xy}(\tau)$ is defined as:

$$R_{xy}(\tau) = \int_{-\infty}^{\infty} u(t)v(t+\tau) \text{d} t.$$

On the other hand, auto-correlation ($R_{ii}$) measures how a signal relates to a delayed version of itself over different time integral  and is defined as:

$$R_{ii}(\tau) = \int_{-\infty}^{\infty} u(t)u(t+\tau) \text{d} t.$$

The integral time scale is used to evaluate the characteristic time over which a signal, such as a turbulent flow, remains correlated with itself:

$$T_L = \int_{0}^{\tau_{cut}} R_{ii}(\tau) \text{d} \tau,$$

where $\tau_{cut}$ is the cut-off lag. The cut-off lag chosen in this project is the lag at which the auto-correlation function first returns to zero.

The signal is divided into multiple time windows; consecutive time windows are chosen, with the auto-correlation function computed for each window to determine the integral time scale. This scale reflects the temporal correlation of the signal, indicating the characteristic duration of its features. The mean of the integral time scales across all windows is then calculated to represent the signal's overall correlation scale.

## Rusults

### Data detrending

A set of wind speed data from Eugene Island, Amerada Pass, Caillou Bay, and Grand Isle are used as an example. The initial data and detrended data for the four stations are shown respectively:

![image](https://github.com/ZimoJupiter/Auto-correlation-integral-time-scale-for-large-scale-turbulence/blob/main/Figures/InitialData.png)

![image](https://github.com/ZimoJupiter/Auto-correlation-integral-time-scale-for-large-scale-turbulence/blob/main/Figures/InitialData_Detrended.png)

### Frequency domain analysis

The frequency domain signal calculated by the Fourier transform is illustrated below (note that applying window functions could lead to better results).

![image](https://github.com/ZimoJupiter/Auto-correlation-integral-time-scale-for-large-scale-turbulence/blob/main/Figures/FFT.png)

![image](https://github.com/ZimoJupiter/Auto-correlation-integral-time-scale-for-large-scale-turbulence/blob/main/Figures/FFT_year.png)

### Auto-correlation analysis

Auto-correlation analysis is conducted on the four sets of signals individually:

![image](https://github.com/ZimoJupiter/Auto-correlation-integral-time-scale-for-large-scale-turbulence/blob/main/Figures/AutoCorrelation_nowindows.png)

When studying the signals by windowing, the auto-correlation functions of the first effective windows of the four sets of signals are:

![image](https://github.com/ZimoJupiter/Auto-correlation-integral-time-scale-for-large-scale-turbulence/blob/main/Figures/AutoCorrelation.png)

The averaged time integral scale reflects the degree of auto-correlation. The larger the time integration scale is, the stronger the autocorrelation that the signal maintains over a longer period of time, and the current value will have a longer-lasting impact on the future. 

![image](https://github.com/ZimoJupiter/Auto-correlation-integral-time-scale-for-large-scale-turbulence/blob/main/Figures/Integral_time_scale.png)

### Cross-correlation analysis

 Intersection of signals at Amerada Pass and Caullou Bay, Grand Isle and Caullou Bay, respectively:

![image](https://github.com/ZimoJupiter/Auto-correlation-integral-time-scale-for-large-scale-turbulence/blob/main/Figures/CrossCorrelation_Data.png)

Similar to the pattern observed in autocorrelation, the cross-correlation functions of these two pairs of signals also exhibit a trend of decay towards zero as the lag increases:

![image](https://github.com/ZimoJupiter/Auto-correlation-integral-time-scale-for-large-scale-turbulence/blob/main/Figures/CrossCorrelation.png)

 The consecutive time windows method is also used in cross-correlation analysis:

![image](https://github.com/ZimoJupiter/Auto-correlation-integral-time-scale-for-large-scale-turbulence/blob/main/Figures/CrossCorrelation_win.png)

Following the same methodology as in autocorrelation analysis, the overall cross-correlation of the signals is evaluated by calculating the integral time scale within each window and then averaging these values:

![image](https://github.com/ZimoJupiter/Auto-correlation-integral-time-scale-for-large-scale-turbulence/blob/main/Figures/Integral_time_scale_CC.png)
 
