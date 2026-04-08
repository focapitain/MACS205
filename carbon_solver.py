import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# =============================================================================
# CONDITIONS INITIALES
# =============================================================================
Atmosphere_Initial   = 750
CarbonateRock_Initial = 100000000
DeepOcean_Initial    = 38000
FossilFuel_Initial   = 7500
Plant_Initial        = 560
Soil_Initial         = 1500
SurfaceOcean_Initial = 890
VegLandArea_percent_Initial = 100

x0 = np.array([
    Atmosphere_Initial, CarbonateRock_Initial, DeepOcean_Initial,
    FossilFuel_Initial, Plant_Initial, Soil_Initial,
    SurfaceOcean_Initial, VegLandArea_percent_Initial
], dtype=float)

Alk = 2.222446077610055
Kao = 0.278
SurfOcVol = 0.0362
Deforestation = 0

# =============================================================================
# FONCTIONS AUXILIAIRES
# =============================================================================
def AtmCO2(Atmosphere):       return Atmosphere * (280 / Atmosphere_Initial)
def GlobalTemp(atm_co2):      return 15 + ((atm_co2 - 280) * 0.01)
def CO2Effect(atm_co2):       return 1.5 * (atm_co2 - 40) / (atm_co2 + 80)
def WaterTemp(global_temp):   return 273 + global_temp
def TempEffect(global_temp):  return ((60 - global_temp) * (global_temp + 15)) / (((60 + 15) / 2) ** 2) / 0.96
def SurfCConc(SurfaceOcean):  return (SurfaceOcean / 12000) / SurfOcVol
def Kcarb(water_temp):        return 0.000575 + (0.000006 * (water_temp - 278))
def KCO2(water_temp):         return 0.035 + (0.0019 * (water_temp - 278))
def HCO3(kcarb, surf_c_conc):
    return (surf_c_conc - np.sqrt(surf_c_conc**2 - Alk * (2*surf_c_conc - Alk) * (1-4*kcarb))) / (1-4*kcarb)
def CO3(hco3):                return (Alk - hco3) / 2
def pCO2Oc(kco2, hco3, co3): return 280 * kco2 * (hco3**2 / co3)

FossFuelData = np.array([
    [1850.,0.],[1875.,.30],[1900.,.60],[1925.,1.35],[1950.,2.85],
    [1975.,4.95],[2000.,7.20],[2025.,10.05],[2050.,14.85],[2075.,20.70],[2100.,30.]
])

def FossilFuelsCombustion(t):
    if t >= FossFuelData[-1, 0]: return FossFuelData[-1, 1]
    i = 0
    while i + 1 < len(FossFuelData) and t >= FossFuelData[i, 0]: i += 1
    if i == 0: return FossFuelData[0, 1]
    return FossFuelData[i-1,1] + (t-FossFuelData[i-1,0])/(FossFuelData[i,0]-FossFuelData[i-1,0])*(FossFuelData[i,1]-FossFuelData[i-1,1])

def derivative(x, t):
    Atmosphere,CarbonateRock,DeepOcean,FossilFuelCarbon,Plants,Soils,SurfaceOcean,VegLandArea_percent = x
    PlantResp   = (Plants*(55/Plant_Initial)) + Deforestation/2
    Litterfall  = (Plants*(55/Plant_Initial)) + Deforestation/2
    SoilResp    = Soils*(55/Soil_Initial)
    Volcanoes   = 0.1
    atm_co2     = AtmCO2(Atmosphere)
    global_temp = GlobalTemp(atm_co2)
    water_temp  = WaterTemp(global_temp)
    Photosynthesis = 110 * CO2Effect(atm_co2) * (VegLandArea_percent/100) * TempEffect(global_temp)
    hco3_       = HCO3(Kcarb(water_temp), SurfCConc(SurfaceOcean))
    pCO2Oc_     = pCO2Oc(KCO2(water_temp), hco3_, CO3(hco3_))
    AtmOcExch   = Kao * (atm_co2 - pCO2Oc_)
    FFC         = FossilFuelsCombustion(t) if FossilFuelCarbon > 0 else 0
    Sedimentation = DeepOcean * (0.1/DeepOcean_Initial)
    Downwelling   = SurfaceOcean * (90.1/SurfaceOcean_Initial)
    Upwelling     = DeepOcean * (90/DeepOcean_Initial)
    Development   = (Deforestation/Plant_Initial*0.2)*100
    return np.array([
        PlantResp+SoilResp+Volcanoes+FFC-Photosynthesis-AtmOcExch,
        Sedimentation-Volcanoes,
        Downwelling-Sedimentation-Upwelling,
        -FFC,
        Photosynthesis-PlantResp-Litterfall,
        Litterfall-SoilResp,
        Upwelling+AtmOcExch-Downwelling,
        -Development
    ])

def derivative_scipy(t, x): return derivative(x, t)

# =============================================================================
# SOLVEURS
# =============================================================================
def euler(x0, t0, t_end, n_steps):
    dt = (t_end - t0) / n_steps
    times = np.linspace(t0, t_end, n_steps + 1)
    X = np.zeros((n_steps + 1, len(x0)))
    X[0] = x0
    for i in range(n_steps):
        X[i+1] = X[i] + dt * derivative(X[i], times[i])
    return times, X

def rk4(x0, t0, t_end, n_steps):
    dt = (t_end - t0) / n_steps
    times = np.linspace(t0, t_end, n_steps + 1)
    X = np.zeros((n_steps + 1, len(x0)))
    X[0] = x0
    for i in range(n_steps):
        t = times[i]
        k1 = derivative(X[i],           t)
        k2 = derivative(X[i]+dt/2*k1,   t+dt/2)
        k3 = derivative(X[i]+dt/2*k2,   t+dt/2)
        k4 = derivative(X[i]+dt*k3,     t+dt)
        X[i+1] = X[i] + (dt/6)*(k1+2*k2+2*k3+k4)
    return times, X

# =============================================================================
# REFERENCE + SIMULATION PRINCIPALE
# =============================================================================
t_start, t_end = 1850, 2100
N_main = 250  # 250 pas = dt=1 an

print("Calcul référence scipy RK45...")
sol_ref = solve_ivp(derivative_scipy, [t_start, t_end], x0,
                    method='RK45', rtol=1e-10, atol=1e-12, dense_output=True)

print("Simulation Euler et RK4 (N=250 pas)...")
t_e, X_e = euler(x0, t_start, t_end, N_main)
t_r, X_r = rk4(x0,   t_start, t_end, N_main)

temp_e   = GlobalTemp(AtmCO2(X_e[:, 0]))
temp_r   = GlobalTemp(AtmCO2(X_r[:, 0]))
temp_ref = GlobalTemp(AtmCO2(sol_ref.sol(t_e)[0]))
co2_e    = AtmCO2(X_e[:, 0])
co2_r    = AtmCO2(X_r[:, 0])
co2_ref  = AtmCO2(sol_ref.sol(t_e)[0])

# =============================================================================
# FIGURE 1 — Euler vs RK4 vs Exact  (style notebook : figsize 14×6, b-/g-/r--)
# =============================================================================
fig1, axes1 = plt.subplots(2, 2, figsize=(14, 8))
fig1.suptitle('Carbon Cycle Model — Euler vs Runge-Kutta vs Exact', fontsize=14)

labels = ['Atmosphère (GtC)', 'Plantes (GtC)', 'Sols (GtC)', 'Océan de surface (GtC)']
idxs   = [0, 4, 5, 6]

for ax, idx, label in zip(axes1.flat, idxs, labels):
    ref_vals = sol_ref.sol(t_e)[idx]
    ax.plot(t_e, X_e[:, idx], 'b-',  linewidth=2, label='Euler')
    ax.plot(t_r, X_r[:, idx], 'g-',  linewidth=2, label='Runge-Kutta')
    ax.plot(t_e, ref_vals,    'r--', linewidth=2, label='Exact')
    ax.set_title(label, fontsize=12)
    ax.set_xlabel('Year')
    ax.set_ylabel(label)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='upper left')

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/fig1_solutions.png', dpi=150, bbox_inches='tight')
plt.close()
print("fig1 done")

# =============================================================================
# FIGURE 2 — Température globale (style twinx du notebook)
# =============================================================================
fig2, ax1 = plt.subplots(figsize=(15, 6))
ax1.set_ylabel('Global Temp (°C)', color='tab:red')
ax1.plot(t_e, temp_e,   color='tab:blue',   linewidth=2, label='Euler',       linestyle='-')
ax1.plot(t_r, temp_r,   color='tab:green',  linewidth=2, label='Runge-Kutta', linestyle='-')
ax1.plot(t_e, temp_ref, color='tab:red',    linewidth=2, label='Exact',       linestyle='--')
ax1.tick_params(axis='y', labelcolor='tab:red')
ax1.set_xlabel('Year')
ax1.set_title('Global Temperature (°C) — Carbon Cycle Model')
ax1.grid()
ax1.legend()

ax2 = ax1.twinx()
ax2.set_ylabel('CO₂ atmosphérique (ppm)', color='tab:orange')
ax2.plot(t_e, co2_e,   color='tab:blue',   linewidth=1.5, linestyle=':', alpha=0.6)
ax2.plot(t_r, co2_r,   color='tab:green',  linewidth=1.5, linestyle=':', alpha=0.6)
ax2.plot(t_e, co2_ref, color='tab:orange', linewidth=1.5, linestyle=':',  label='CO₂ Exact')
ax2.tick_params(axis='y', labelcolor='tab:orange')
ax2.legend(loc='upper left')

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/fig2_temp_co2.png', dpi=150, bbox_inches='tight')
plt.close()
print("fig2 done")

# =============================================================================
# FIGURE 3 — Max Consistency Error vs Number of Steps (style notebook log-log)
# =============================================================================
print("Analyse de convergence...")
n_steps_list = [250, 500, 1000, 2000, 5000]
max_err_e = []
max_err_r = []

for n in n_steps_list:
    _, Xe = euler(x0, t_start, t_end, n)
    _, Xr = rk4(x0,   t_start, t_end, n)
    t_test = np.linspace(t_start, t_end, n+1)
    ref_atm = sol_ref.sol(t_test)[0]
    max_err_e.append(np.max(np.abs(Xe[:, 0] - ref_atm)))
    max_err_r.append(np.max(np.abs(Xr[:, 0] - ref_atm)))

n_steps_arr = np.array(n_steps_list, dtype=float)
max_err_e   = np.array(max_err_e)
max_err_r   = np.array(max_err_r)

# Pentes log-log
slope_e = np.polyfit(np.log10(n_steps_arr), np.log10(max_err_e), 1)
slope_r = np.polyfit(np.log10(n_steps_arr), np.log10(max_err_r), 1)

# Lignes de référence (2 points)
x0_e, x1_e = n_steps_arr[0],  n_steps_arr[-1]
y0_e = 10**np.polyval(slope_e, np.log10(x0_e))
y1_e = 10**np.polyval(slope_e, np.log10(x1_e))

x0_r, x1_r = n_steps_arr[0],  n_steps_arr[-1]
y0_r = 10**np.polyval(slope_r, np.log10(x0_r))
y1_r = 10**np.polyval(slope_r, np.log10(x1_r))

fig3, (ax_e, ax_r) = plt.subplots(1, 2, figsize=(14, 6))
fig3.suptitle('Max Consistency Error vs Number of Steps', fontsize=14)

# Euler
ax_e.loglog(n_steps_arr, max_err_e, 'b-o', linewidth=2, label='Données')
ax_e.loglog([x0_e, x1_e], [y0_e, y1_e], 'r--', linewidth=2, label='Ligne de référence')
ax_e.text(0.05, 0.15,
          f'Pente: {slope_e[0]:.1f}\nIntercept (log10(C)) = {slope_e[1]:.2f}',
          transform=ax_e.transAxes, bbox=dict(facecolor='white', alpha=0.8), fontsize=10)
ax_e.set_xlabel('Number of Steps')
ax_e.set_ylabel('Max Consistency Error (GtC)')
ax_e.set_title(f'Euler — ordre empirique ≈ {abs(slope_e[0]):.1f} (théorique : 1)')
ax_e.grid(True)
ax_e.legend()

# RK4
ax_r.loglog(n_steps_arr, max_err_r, 'g-o', linewidth=2, label='Données')
ax_r.loglog([x0_r, x1_r], [y0_r, y1_r], 'r--', linewidth=2, label='Ligne de référence')
ax_r.text(0.05, 0.15,
          f'Pente: {slope_r[0]:.0f}\nIntercept (log10(C)) = {slope_r[1]:.2f}',
          transform=ax_r.transAxes, bbox=dict(facecolor='white', alpha=0.8), fontsize=10)
ax_r.set_xlabel('Number of Steps')
ax_r.set_ylabel('Max Consistency Error (GtC)')
ax_r.set_title(f'Runge-Kutta 4 — ordre empirique ≈ {abs(slope_r[0]):.1f} (théorique : 4)')
ax_r.grid(True)
ax_r.legend()

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/fig3_convergence.png', dpi=150, bbox_inches='tight')
plt.close()
print("fig3 done")

# =============================================================================
# FIGURE 4 — Comparaison des erreurs absolues au cours du temps  (style subplot 1×2)
# =============================================================================
ref_atm_main = sol_ref.sol(t_e)[0]
err_e_time   = np.abs(X_e[:, 0] - ref_atm_main)
err_r_time   = np.abs(X_r[:, 0] - ref_atm_main)

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(t_e, X_e[:, 0], 'b-',  linewidth=2, label='Euler')
plt.plot(t_r, X_r[:, 0], 'g-',  linewidth=2, label='Runge-Kutta')
plt.plot(t_e, ref_atm_main, 'r--', linewidth=2, label='Exact')
plt.title('Euler vs Runge-Kutta vs Exact', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Atmosphère (GtC)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10, loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(t_e, err_e_time, 'b:',  linewidth=2, label='Erreur Euler')
plt.plot(t_r, err_r_time, 'g-.', linewidth=2, label='Erreur RK4')
plt.title('Comparaison des erreurs absolues', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Erreur (GtC)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10, loc='upper left')

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/fig4_erreur_temps.png', dpi=150, bbox_inches='tight')
plt.close()
print("fig4 done")

# =============================================================================
# RÉSUMÉ
# =============================================================================
print(f"\n{'='*55}")
print("RÉSUMÉ")
print(f"{'='*55}")
print(f"Température 1850 : {temp_ref[0]:.3f} °C")
print(f"Température 2100 — Euler: {temp_e[-1]:.3f} | RK4: {temp_r[-1]:.3f} | Exact: {temp_ref[-1]:.3f} °C")
print(f"CO₂ 2100        — Euler: {co2_e[-1]:.1f} | RK4: {co2_r[-1]:.1f} | Exact: {co2_ref[-1]:.1f} ppm")
print(f"Ordre empirique — Euler: {abs(slope_e[0]):.2f} (théorique 1) | RK4: {abs(slope_r[0]):.2f} (théorique 4)")
