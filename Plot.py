import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.optimize import brentq, minimize_scalar

# --------------
#Parameters
#---------------
m = 1.0
k = 10.0
alpha = 0.1

# Initial conditions
r0 = 1
vr0 = 3
theta0 = 0.1
theta0dot = 2  # Initial angular velocity
vtheta0 = r0*theta0dot

vx0 = vr0 * np.cos(theta0) - vtheta0 * np.sin(theta0)
vy0 = vr0 * np.sin(theta0) + vtheta0 * np.cos(theta0)


# Angular momentum
L = m * r0**2 * theta0dot

# -----------------------------
# Effective potential
# -----------------------------
def V_eff(r):
    return L**2 /(2*m* r**2) + alpha * r**2 - k / r

def dV_dr(r):
    return 2*alpha*r + k / r**2


# -----------------------------
# Mnimum potential
# -----------------------------
res = minimize_scalar(V_eff, bounds=(0.01, 5*r0), method='bounded')
r_min_pot = res.x
V_min = res.fun

# -----------------------------
# Differential equations
# -----------------------------
def f(t, y):
    r, vr, theta = y
    drdt = vr
    dvrdt = L**2/(m**2 * r**3) - (1 / m) * dV_dr(r)
    dthetadt = L/(m * r**2)
    return [drdt, dvrdt, dthetadt]

# Initial conditions
y0 = [r0, vr0, theta0]

# Integration
t_span = (0, 100)
sol = solve_ivp(f, t_span, y0, max_step=0.01, rtol=1e-9, atol=1e-14)

# -----------------------------
# Extract solutions
# -----------------------------
r = sol.y[0]
vr = sol.y[1]
theta = sol.y[2]
x = r * np.cos(theta)
y = r * np.sin(theta)
E = 0.5*m*vr**2 + L**2/(2*m*r**2) - k/r + alpha * r**2
E_total = E[0]
print("Total energy:", np.round(E_total,6))

# -----------------------------
# Radios max/min
# -----------------------------
r_max_tray = np.max(r)
r_min_tray = np.min(r)
print(f"Trayectory maximum radius: {r_max_tray:.6f}")
print(f"Trayectory minimum radius: {r_min_tray:.6f}")

def f_r(r):
    return V_eff(r) - E_total

r_min_eff = brentq(f_r, 0.01, r0)
r_max_eff = brentq(f_r, r0, 5*r_max_tray)
print(f"V_eff minimum radius: {r_min_eff:.6f}")
print(f"V_eff maximum radius: {r_max_eff:.6f}")
print(f"Initial position (x, y): ({x[0]:.6f}, {y[0]:.6f})")

# -----------------------------
# PLot
# -----------------------------

# --- Trayectory ---
plt.figure(figsize=(6,5))
colors = {'min':'green','max':'orange'}
plt.plot(x, y, label='Trajectory', color='blue',linewidth=0.25)
circle_min = plt.Circle((0,0), r_min_tray, color=colors['min'], fill=False, linestyle=':', linewidth=2, label=fr'$r_{{\mathrm{{min}}}} = {r_min_tray:.3f}$')
circle_max = plt.Circle((0,0), r_max_tray, color=colors['max'], fill=False, linestyle='--', linewidth=2, label=fr'$r_{{\mathrm{{max}}}} = {r_max_tray:.3f}$')
plt.gca().add_patch(circle_min)
plt.gca().add_patch(circle_max)
plt.scatter([x[0]], [y[0]], color='black')
plt.quiver(x[0], y[0], vx0, vy0, angles='xy', scale_units='xy', scale=5, color='black', width=0.005, headlength=5, headwidth=4)
plt.xlabel('x (a.u.)')
plt.ylabel('y (a. u.)')
plt.title('Bound trajectory in a central effective potential')
plt.axis('equal')
plt.text(-0.2, 0.5, 'Initial velocity', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()
#plt.savefig("trayectoria-mecanica1.pdf")
#plt.savefig("trayectoria-mecanica1.png")



# --- Effective potential ---
plt.figure(figsize=(6, 5))
r_vals = np.linspace(0.01, 5, 500)
plt.plot(r_vals, V_eff(r_vals), label='Effective Potential', color='blue')
plt.axhline(E_total, color='red', linestyle='--', label=f'Energy level')

# Confined zone
plt.fill_between(r_vals, V_eff(r_vals), E_total, where=(r_vals>=r_min_eff)&(r_vals<=r_max_eff), color='lightskyblue', alpha=0.4, label='Confined region')

# Vertical lines
plt.axvline(r_min_tray, color='darkgray', linestyle=':', linewidth=2)
plt.axvline(r_max_tray, color='dimgray', linestyle=':', linewidth=2)

# marker points
plt.scatter(r_min_pot, V_min, color='black', s=50, zorder=5)
plt.scatter(r_min_tray, E_total, color='black', s=50, zorder=5)
plt.scatter(r_max_tray, E_total, color='black', s=50, zorder=5)

plt.xlabel('r (a. u.)')
plt.ylabel('Veff (energy units)')
plt.title('Bound motion in an effective potential')
plt.text(0.5, -13, 'Stable equilibrium', fontsize=11)
plt.text(0.32, -3, r'$r_{\mathrm{min}}$', fontsize=14)
plt.text(2, -3, r'$r_{\mathrm{max}}$', fontsize=14)
plt.text(-0.5, E_total,f'{E_total:.3f}' , fontsize=9)
plt.text(r_min_tray-0.1, -14.5,f'{r_min_tray:.3f}' , fontsize=9)
plt.text(r_max_tray-0.1, -14.5,f'{r_max_tray:.3f}' , fontsize=9)
plt.text(0.3, -4.5,'Oscillation region', fontsize=11)
plt.legend(fontsize=11)
plt.xlim(0, 5)
plt.ylim(-14,0)
plt.grid(True)
