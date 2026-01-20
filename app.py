import streamlit as st
import numpy as np
from scipy.integrate import odeint
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Interactive Quadrupole Mass Filter", layout="wide")

st.title("🔬 Interactive Quadrupole Mass Spectrometer Simulator")
st.markdown("""
This simulator demonstrates the operating principle of a **Quadrupole Mass Filter (QMF)**. 
Adjust the voltages and frequency to see how they create a "stable tunnel" for specific ions 
while causing others to crash into the rods.
""")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("1. Instrument Settings")

# Physical Constants & Dimensions
r0 = 0.005  # Radius of the field (meters) -> 5mm
length = 0.2 # Length of rods (meters) -> 20cm
f_mhz = st.sidebar.slider("RF Frequency (MHz)", 0.5, 3.0, 1.0, 0.1)
omega = 2 * np.pi * (f_mhz * 1e6)

# Voltage Controls
# We set a target mass to auto-tune the voltages initially, but allow manual override
target_mz_guess = 100.0
# Approximate "Apex" of stability diagram values (a=0.237, q=0.706) roughly
# This helps the user start with a working simulation
u_init = (0.237 * target_mz_guess * 1.66e-27 * omega**2 * r0**2) / (8 * 1.602e-19)
v_init = (0.706 * target_mz_guess * 1.66e-27 * omega**2 * r0**2) / (4 * 1.602e-19)

U = st.sidebar.number_input("DC Voltage (U) [Volts]", value=float(f"{u_init:.2f}"), step=0.5)
V = st.sidebar.number_input("RF Voltage (V) [Volts]", value=float(f"{v_init:.2f}"), step=1.0)

st.sidebar.header("2. Ion Parameters")
v_z = st.sidebar.slider("Initial Axial Velocity (m/s)", 1000, 5000, 2500, 100)

st.sidebar.subheader("Define 3 Ions (m/z)")
m1 = st.sidebar.number_input("Ion 1 Mass (amu) - (Blue)", value=90.0, step=1.0)
m2 = st.sidebar.number_input("Ion 2 Mass (amu) - (Green)", value=100.0, step=1.0)
m3 = st.sidebar.number_input("Ion 3 Mass (amu) - (Red)", value=110.0, step=1.0)

# --- PHYSICS ENGINE ---

def mathieu_derivs(state, t, e_over_m, U, V, omega, r0):
    """
    Computes derivatives for the equations of motion in a quadrupole field.
    State vector: [x, vx, y, vy]
    Equations derived from Mathieu equation forms:
    d2x/dt2 + (e/m * r0^2) * (U + V*cos(wt)) * x = 0
    d2y/dt2 - (e/m * r0^2) * (U + V*cos(wt)) * y = 0
    """
    x, vx, y, vy = state
    
    # Potential factor Phi_0 = U + V cos(omega * t)
    # Force F = qE = -q * gradient(Phi)
    # acceleration a = F/m
    
    # The coefficient commonly used is k = 2 * e / (m * r0^2)
    # However, strictly: d2x/dt2 + (2e/mr0^2)(U + Vcos(wt))x = 0 is a common approximation
    
    k = (2 * 1.60217663e-19) / (e_over_m * r0**2)
    phi = U + V * np.cos(omega * t)
    
    ax = -k * phi * x
    ay = k * phi * y  # Note the sign change for the quadrupole field in Y
    
    return [vx, ax, vy, ay]

def simulate_ion(mass_amu, U, V, omega, r0, length, v_z):
    mass_kg = mass_amu * 1.66053907e-27
    
    # Time to traverse the rods
    total_time = length / v_z
    t_eval = np.linspace(0, total_time, 1000)
    
    # Initial conditions: small random offset from center to simulate real beam width
    # x0, vx0, y0, vy0
    init_state = [0.0005, 0, 0.0005, 0] 
    
    sol = odeint(mathieu_derivs, init_state, t_eval, args=(mass_kg, U, V, omega, r0))
    
    x = sol[:, 0]
    y = sol[:, 2]
    z = v_z * t_eval
    
    # Check if ion crashed (distance from center > r0)
    radial_dist = np.sqrt(x**2 + y**2)
    crashed_indices = np.where(radial_dist > r0)[0]
    
    status = "Transmitted"
    if len(crashed_indices) > 0:
        crash_idx = crashed_indices[0]
        # Truncate arrays at crash point
        x = x[:crash_idx]
        y = y[:crash_idx]
        z = z[:crash_idx]
        status = "Crashed"
        
    return x, y, z, status

# --- SIMULATION LOOP ---
ions = [
    {"mass": m1, "color": "blue", "name": f"Ion A ({m1} amu)"},
    {"mass": m2, "color": "#00CC96", "name": f"Ion B ({m2} amu)"},
    {"mass": m3, "color": "red", "name": f"Ion C ({m3} amu)"},
]

# Run simulation for each ion
for ion in ions:
    x, y, z, status = simulate_ion(ion["mass"], U, V, omega, r0, length, v_z)
    ion["x"] = x
    ion["y"] = y
    ion["z"] = z
    ion["status"] = status

# --- VISUALIZATION ---

# Create 3D Plot
fig = go.Figure()

# Add Quadrupole Rods (Visual representation only - Wireframe)
# Simplified as 4 lines for performance
rod_len = np.linspace(0, length, 10)
rod_offset = r0 * 1.1 # slightly larger than field radius
for ang in [0, np.pi/2, np.pi, 3*np.pi/2]:
    rx = rod_offset * np.cos(ang)
    ry = rod_offset * np.sin(ang)
    fig.add_trace(go.Scatter3d(
        x=[rx]*10, y=[ry]*10, z=rod_len,
        mode='lines', line=dict(color='gray', width=5),
        name='Rod', showlegend=False, hoverinfo='skip'
    ))

# Add Ion Trajectories
for ion in ions:
    # Trajectory
    fig.add_trace(go.Scatter3d(
        x=ion["x"], y=ion["y"], z=ion["z"],
        mode='lines',
        line=dict(color=ion["color"], width=4),
        name=f"{ion['name']} - {ion['status']}"
    ))
    
    # End Point Marker
    fig.add_trace(go.Scatter3d(
        x=[ion["x"][-1]], y=[ion["y"][-1]], z=[ion["z"][-1]],
        mode='markers',
        marker=dict(size=5, color=ion["color"]),
        showlegend=False
    ))

fig.update_layout(
    title="3D Ion Trajectories",
    scene=dict(
        xaxis_title="X Position (m)",
        yaxis_title="Y Position (m)",
        zaxis_title="Z Position (m) - Along Rods",
        aspectratio=dict(x=1, y=1, z=3), # Elongate Z axis for better view
        xaxis=dict(range=[-r0*2, r0*2]),
        yaxis=dict(range=[-r0*2, r0*2]),
    ),
    margin=dict(l=0, r=0, b=0, t=40),
    height=600
)

# --- 2D Cross Section View ---
fig_2d = go.Figure()

# Draw the limit circle
theta = np.linspace(0, 2*np.pi, 100)
fig_2d.add_trace(go.Scatter(
    x=r0*np.cos(theta), y=r0*np.sin(theta),
    mode='lines', line=dict(color='black', dash='dash'), name='Rod Boundary'
))

for ion in ions:
    fig_2d.add_trace(go.Scatter(
        x=ion["x"], y=ion["y"],
        mode='lines', line=dict(color=ion["color"]),
        name=ion['name']
    ))

fig_2d.update_layout(
    title="2D Cross-Section (X-Y Plane)",
    xaxis_title="X Position (m)",
    yaxis_title="Y Position (m)",
    height=500,
    xaxis=dict(range=[-r0*1.5, r0*1.5]),
    yaxis=dict(range=[-r0*1.5, r0*1.5]),
)

# --- LAYOUT RENDERING ---
col1, col2 = st.columns([2, 1])

with col1:
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.plotly_chart(fig_2d, use_container_width=True)
    st.markdown("### Simulation Status")
    for ion in ions:
        status_icon = "✅" if ion["status"] == "Transmitted" else "❌"
        st.write(f"{status_icon} **{ion['name']}**: {ion['status']}")

st.markdown("---")
st.subheader("💡 The Physics Behind It")
st.latex(r"""
\frac{d^2u}{dt^2} \pm \frac{2e}{mr_0^2} (U + V \cos(\omega t)) u = 0
""")
st.markdown("""
The simulation solves the **Mathieu Equations** (shown above) for $x$ and $y$ motion. 
* **$U$ (DC Voltage):** Tries to destabilize heavy ions in one direction and light ions in the other.
* **$V$ (RF Voltage):** Creates the oscillation that stabilizes specific masses.
* **Filter Action:** Only ions with a specific Mass-to-Charge ($m/z$) ratio have a stable trajectory (bounded oscillation) and reach the detector. All others oscillate too wildly and hit the rods (represented by the dashed circle).
""")
