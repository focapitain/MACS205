import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# =============================================================================
# CONDITIONS INITIALES + FONCTIONS (copie du solver)
# =============================================================================
Atmosphere_Initial=750; CarbonateRock_Initial=100000000; DeepOcean_Initial=38000
FossilFuel_Initial=7500; Plant_Initial=560; Soil_Initial=1500
SurfaceOcean_Initial=890; VegLandArea_percent_Initial=100
x0=np.array([750,100000000,38000,7500,560,1500,890,100],dtype=float)
Alk=2.222446077610055; Kao=0.278; SurfOcVol=0.0362; Deforestation=0

def AtmCO2(A):      return A*(280/Atmosphere_Initial)
def GlobalTemp(c):  return 15+((c-280)*0.01)
def CO2Effect(c):   return 1.5*(c-40)/(c+80)
def WaterTemp(g):   return 273+g
def TempEffect(g):  return ((60-g)*(g+15))/(((60+15)/2)**2)/0.96
def SurfCConc(S):   return (S/12000)/SurfOcVol
def Kcarb(w):       return 0.000575+(0.000006*(w-278))
def KCO2(w):        return 0.035+(0.0019*(w-278))
def HCO3(k,s):      return (s-np.sqrt(s**2-Alk*(2*s-Alk)*(1-4*k)))/(1-4*k)
def CO3(h):         return (Alk-h)/2
def pCO2Oc(k,h,c):  return 280*k*(h**2/c)

FossFuelData=np.array([[1850.,0.],[1875.,.30],[1900.,.60],[1925.,1.35],[1950.,2.85],
    [1975.,4.95],[2000.,7.20],[2025.,10.05],[2050.,14.85],[2075.,20.70],[2100.,30.]])

def FFC(t):
    if t>=FossFuelData[-1,0]: return FossFuelData[-1,1]
    i=0
    while i+1<len(FossFuelData) and t>=FossFuelData[i,0]: i+=1
    if i==0: return FossFuelData[0,1]
    return FossFuelData[i-1,1]+(t-FossFuelData[i-1,0])/(FossFuelData[i,0]-FossFuelData[i-1,0])*(FossFuelData[i,1]-FossFuelData[i-1,1])

def derivative(x,t):
    Atmosphere,CarbonateRock,DeepOcean,FossilFuelCarbon,Plants,Soils,SurfaceOcean,Veg=x
    PlantResp=Litterfall=(Plants*(55/Plant_Initial))+Deforestation/2
    SoilResp=Soils*(55/Soil_Initial); Volcanoes=0.1
    atm=AtmCO2(Atmosphere); gt=GlobalTemp(atm); wt=WaterTemp(gt)
    Photo=110*CO2Effect(atm)*(Veg/100)*TempEffect(gt)
    h=HCO3(Kcarb(wt),SurfCConc(SurfaceOcean))
    AtmOc=Kao*(atm-pCO2Oc(KCO2(wt),h,CO3(h)))
    fc=FFC(t) if FossilFuelCarbon>0 else 0
    Sed=DeepOcean*(0.1/DeepOcean_Initial)
    Down=SurfaceOcean*(90.1/SurfaceOcean_Initial)
    Up=DeepOcean*(90/DeepOcean_Initial)
    Dev=(Deforestation/Plant_Initial*0.2)*100
    return np.array([PlantResp+SoilResp+Volcanoes+fc-Photo-AtmOc,
        Sed-Volcanoes, Down-Sed-Up, -fc,
        Photo-PlantResp-Litterfall, Litterfall-SoilResp,
        Up+AtmOc-Down, -Dev])

def euler(x0,t0,t_end,n):
    dt=(t_end-t0)/n; times=np.linspace(t0,t_end,n+1)
    X=np.zeros((n+1,len(x0))); X[0]=x0
    for i in range(n):
        X[i+1]=X[i]+dt*derivative(X[i],times[i])
    return times,X

def rk4(x0,t0,t_end,n):
    dt=(t_end-t0)/n; times=np.linspace(t0,t_end,n+1)
    X=np.zeros((n+1,len(x0))); X[0]=x0
    for i in range(n):
        t=times[i]
        k1=derivative(X[i],t)
        k2=derivative(X[i]+dt/2*k1,t+dt/2)
        k3=derivative(X[i]+dt/2*k2,t+dt/2)
        k4=derivative(X[i]+dt*k3,t+dt)
        X[i+1]=X[i]+(dt/6)*(k1+2*k2+2*k3+k4)
    return times,X

# =============================================================================
# SIMULATION 500 ANS (1850–2350)
# =============================================================================
t_start, t_end = 1850, 2350
N = 5000  # dt = 0.1 an

print("Simulation Euler 500 ans...")
t_e, X_e = euler(x0, t_start, t_end, N)
print("Simulation RK4 500 ans...")
t_r, X_r = rk4(x0,   t_start, t_end, N)

# Noms et couleurs des 8 variables (même ordre que le screenshot)
vars_info = [
    (0, 'Atmosphere',       'tab:red'),
    (1, 'Carbonate Rock',   'tab:cyan'),
    (2, 'Deep Ocean',       'tab:blue'),
    (3, 'Fossil Fuel Carbon','tab:orange'),
    (4, 'Plants',           'tab:purple'),
    (5, 'Soil',             'tab:brown'),
    (6, 'Surface Ocean',    'tab:pink'),
    (7, 'Veg Land Area %',  'yellowgreen'),
]

def make_plot(t, X, title):
    fig, ax1 = plt.subplots(figsize=(15, 6))
    fig.subplots_adjust(right=0.55)  # espace pour les axes droits

    # Axe principal : Atmosphere (rouge, à gauche)
    idx0, label0, color0 = vars_info[0]
    ax1.plot(t, X[:, idx0], color=color0, linewidth=1.5)
    ax1.set_ylabel(label0, color=color0, fontsize=10)
    ax1.tick_params(axis='y', labelcolor=color0)
    ax1.set_xlabel('Year')
    ax1.set_title(title)
    ax1.grid(True, alpha=0.4)
    ax1.set_xlim(t_start, t_end)

    # 7 axes twinx empilés vers la droite
    axes_right = []
    offsets = [1.0, 1.12, 1.24, 1.36, 1.48, 1.60, 1.72]

    for k, (idx, label, color) in enumerate(vars_info[1:]):
        ax = ax1.twinx()
        ax.spines['right'].set_position(('axes', offsets[k]))
        ax.spines['right'].set_visible(True)
        ax.plot(t, X[:, idx], color=color, linewidth=1.5)
        ax.set_ylabel(label, color=color, fontsize=9, rotation=90, labelpad=5)
        ax.tick_params(axis='y', labelcolor=color, labelsize=7)
        ax.yaxis.label.set_rotation(90)
        axes_right.append(ax)

    return fig

print("Génération des figures...")
fig_euler = make_plot(t_e, X_e, 'Carbon Cycle Model - Euler Method')
plt.show()
plt.close()

fig_rk4 = make_plot(t_r, X_r, 'Carbon Cycle Model - Runge-Kutta Method')
plt.show()
plt.close()

print("Done.")
