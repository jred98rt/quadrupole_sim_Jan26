import streamlit as st
import numpy as np
from scipy.integrate import odeint
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Interactive Quadrupole Mass Filter", layout="wide")

st.title("🔬 Interactive Quadrupole Mass Spectrometer Simulator")
st.markdown("""
This simulator demonstrates the operating principle of a **Quadrupole Mass Filter (QMF)**.
Use the controls to scan voltages or jump to specific masses to see how the "Stability Tunnel" changes.
""")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("1. Instrument Configuration")

# Physical Constants & Dimensions
r0 = 0.005  # Radius of the field (meters) -> 5mm
length = 0.2 # Length of rods (meters) -> 20cm
f_mhz = st.sidebar.slider("RF Frequency (MHz)", 0.5, 3.0, 1.0, 0.1)
omega = 2 * np.pi * (f_mhz * 1e6)

# Constants for Mathieu Equation Stability (Approx Apex of Stability Diagram)
# These define the tip of the stability triangle where resolution is highest.
TARGET_A = 0.23699  # Mathieu parameter a
TARGET_Q = 0.70600  # Mathieu parameter q
# Mass of proton/neutron in kg
AMU_KG = 1.66053907e-27
CHARGE = 1.60217663e-19

# Helper: Calculate Voltages from Mass
def calculate_voltages(target_mass_amu):
    mass_kg = target_mass_amu * AMU_KG
    # V = (q * m * r0^2 * omega^2) / (4 * e)
    V_calc = (TARGET_Q * mass_kg * (r0**2) * (omega**2)) / (4 * CHARGE)
    # U = (a * m * r0^2 * omega^2) / (8 * e)
    U_calc = (TARGET_A * mass_kg * (r0**2) * (omega**2)) / (8 * CHARGE)
    return U_calc, V_calc

# Helper: Calculate Mass from RF Voltage (V) assuming apex q
def calculate_mass_from_V(V_input):
    # m = (4 * e * V) / (q * r0^2 * omega^2)
    m_kg = (4 * CHARGE * V_input) / (TARGET_Q * (r0**2) * (omega**2))
    return m_kg / AMU_KG

# --- CONTROL MODES ---
st.sidebar.header("2. Voltage Control Mode")
control_mode = st.sidebar.radio(
    "Select Mode:", 
    ["Manual Independent", "Linked Scan (Ratio)", "Auto-Tune (Target m/z)"]
)

# Initialize variables to avoid scope errors
U, V = 0.0, 0.0

if control_mode == "Manual Independent":
    st.sidebar.markdown("**Manual Mode:** Adjust DC and RF independently.")
    U = st.sidebar.number_input("DC Voltage (U) [Volts]", value=30.0, step=0.5)
    V = st.sidebar.number_input("RF Voltage (V) [Volts]", value=180.0, step=1.0)

elif control_mode == "Linked Scan (Ratio)":
    st.sidebar.markdown("**Scan Mode:** Adjusting RF automatically sets DC to maintain constant resolution.")
    
    # Calculate the ratio needed for the apex
    scan_ratio = 0.95 * TARGET_A / (2.0 * TARGET_Q) # Approx 0.1678 Reduced ratio by mult 0.9
    
    # Slider for V (RF)
    V = st.sidebar.slider("RF Voltage (V)", min_value=0.0, max_value=1500.0, value=180.0, step=1.0)
    # Calculate U based on fixed ratio
    U = V * scan_ratio
    
    # Display the derived value
    st.sidebar.info(f"**Linked DC Voltage (U):** {U:.2f} V")
    
    # Show what mass this corresponds to
    current_mass = calculate_mass_from_V(V)
    st.sidebar.caption(f"Currently tuned for approx: **{current_mass:.1f} amu**")

elif control_mode == "Auto-Tune (Target m/z)":
    st.sidebar.markdown("**Auto-Tune:** Enter a mass, and the simulator calculates the required voltages.")
    
    target_mz = st.sidebar.number_input("Target m/z (amu)", value=100.0, step=1.0)
    
    # Calculate required U and V
    U, V = calculate_voltages(target_mz)
    
    st.sidebar.success(f"**Calculated Settings:**\n\nDC (U): {U:.2f} V\n\nRF (V): {V:.2f} V")

# --- ION PARAMETERS ---
st.sidebar.header("3. Ion Parameters")
v_z = st.sidebar.slider("Initial Axial Velocity (m/s)", 1000, 5000, 2500, 100)

st.sidebar.subheader("Define 3 Ions (m/z)")
m1 = st.sidebar.number_input("Ion 1 Mass (amu) - (Blue)", value=90.0, step=1.0)
m2 = st.sidebar.number_input("Ion 2 Mass (amu) - (Green)", value=100.0, step=1.0)
m3 = st.sidebar.number_input("Ion 3 Mass (amu) - (Red)", value=110.0, step=1.0)

# --- PHYSICS ENGINE (Unchanged) ---
def mathieu_derivs(state, t, e_over_m, U, V, omega, r0):
    x, vx, y, vy = state
    k = (2 * CHARGE) / (e_over_m * r0**2)
    phi = U + V * np.cos(omega * t)
    ax = -k * phi * x
    ay = k * phi * y 
    return [vx, ax, vy, ay]

def simulate_ion(mass_amu, U, V, omega, r0, length, v_z):
    mass_kg = mass_amu * AMU_KG
    total_time = length / v_z
    t_eval = np.linspace(0, total_time, 1000)
    
    # Initial conditions: small random offset
    init_state = [0.0005, 0, 0.0005, 0] 
    
    sol = odeint(mathieu_derivs, init_state, t_eval, args=(mass_kg, U, V, omega, r0))
    
    x, y = sol[:, 0], sol[:, 2]
    z = v_z * t_eval
    
    # Check crash
    radial_dist = np.sqrt(x**2 + y**2)
    crashed_indices = np.where(radial_dist > r0)[0]
    
    status = "Transmitted"
    if len(crashed_indices) > 0:
        crash_idx = crashed_indices[0]
        x, y, z = x[:crash_idx], y[:crash_idx], z[:crash_idx]
        status = "Crashed"
        
    return x, y, z, status

# --- SIMULATION LOOP ---
ions = [
    {"mass": m1, "color": "blue", "name": f"Ion A ({m1} amu)"},
    {"mass": m2, "color": "#00CC96", "name": f"Ion B ({m2} amu)"},
    {"mass": m3, "color": "red", "name": f"Ion C ({m3} amu)"},
]

for ion in ions:
    x, y, z, status = simulate_ion(ion["mass"], U, V, omega, r0, length, v_z)
    ion["x"], ion["y"], ion["z"], ion["status"] = x, y, z, status

# --- VISUALIZATION ---
fig = go.Figure()

# Rods
rod_len = np.linspace(0, length, 10)
rod_offset = r0 * 1.1
for ang in [0, np.pi/2, np.pi, 3*np.pi/2]:
    rx, ry = rod_offset * np.cos(ang), rod_offset * np.sin(ang)
    fig.add_trace(go.Scatter3d(
        x=[rx]*10, y=[ry]*10, z=rod_len,
        mode='lines', line=dict(color='gray', width=5),
        name='Rod', showlegend=False, hoverinfo='skip'
    ))

# Ions
for ion in ions:
    fig.add_trace(go.Scatter3d(
        x=ion["x"], y=ion["y"], z=ion["z"],
        mode='lines', line=dict(color=ion["color"], width=4),
        name=f"{ion['name']} - {ion['status']}"
    ))
    fig.add_trace(go.Scatter3d(
        x=[ion["x"][-1]], y=[ion["y"][-1]], z=[ion["z"][-1]],
        mode='markers', marker=dict(size=5, color=ion["color"]), showlegend=False
    ))

fig.update_layout(
    title=f"3D Ion Trajectories (U={U:.1f}V, V={V:.1f}V)",
    scene=dict(
        xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)",
        aspectratio=dict(x=1, y=1, z=3),
        xaxis=dict(range=[-r0*2, r0*2]), yaxis=dict(range=[-r0*2, r0*2]),
    ),
    margin=dict(l=0, r=0, b=0, t=40), height=600
)

# 2D Plot
fig_2d = go.Figure()
theta = np.linspace(0, 2*np.pi, 100)
fig_2d.add_trace(go.Scatter(
    x=r0*np.cos(theta), y=r0*np.sin(theta),
    mode='lines', line=dict(color='black', dash='dash'), name='Rod Limit'
))
for ion in ions:
    fig_2d.add_trace(go.Scatter(
        x=ion["x"], y=ion["y"],
        mode='lines', line=dict(color=ion["color"]), name=ion['name']
    ))
fig_2d.update_layout(
    title="2D Cross-Section (X-Y)",
    xaxis=dict(range=[-r0*1.5, r0*1.5]), yaxis=dict(range=[-r0*1.5, r0*1.5]),
    height=500
)

# Render
col1, col2 = st.columns([2, 1])
with col1: st.plotly_chart(fig, use_container_width=True)
with col2:
    st.plotly_chart(fig_2d, use_container_width=True)
    st.markdown("### Status")
    for ion in ions:
        st.write(f"{'✅' if ion['status'] == 'Transmitted' else '❌'} **{ion['name']}**: {ion['status']}")
