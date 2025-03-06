import numpy as np

#Function t-y^2 (for #1)
def f(t, y):
    return t - y**2

#Euler Method
def euler_method(f, t0, y0, t_end, steps):
    h = (t_end - t0) / steps  
    t = t0
    y = y0
    for _ in range(steps):
        y += h * f(t, y)
        t += h
    return y

#Runge-Kutta Method
def runge_kutta_method(f, t0, y0, t_end, steps):
    h = (t_end - t0) / steps  
    t = t0
    y = y0
    for _ in range(steps):
        k1 = h * f(t, y)
        k2 = h * f(t + h/2, y + k1/2)
        k3 = h * f(t + h/2, y + k2/2)
        k4 = h * f(t + h, y + k3)
        y += (k1 + 2*k2 + 2*k3 + k4) / 6
        t += h
    return y

#Parameters
t0, y0 = 0, 1  
t_end = 2 #range    
steps = 10 #number of iterations

#Solutions
euler_result = euler_method(f, t0, y0, t_end, steps)
rk4_result = runge_kutta_method(f, t0, y0, t_end, steps)

#Output
print(f"{euler_result}\n")
print(f"{rk4_result}")
