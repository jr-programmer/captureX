#Acceleration of CO2 Solubility Trapping Mechanism for Enhanced Storage Capacity Utilizing Artificial Intelligence 






#Value Creation in Sustainable Energy Transition Using Reinforcement Learning


import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

# Placeholder function to simulate environment
def simulate_investment(oil_gas_investment, renewables_investment, ccs_investment):
    # Simplified random simulation results
    production = oil_gas_investment * 0.5 + renewables_investment * 0.4
    co2_reduction = ccs_investment * 0.6
    return production, co2_reduction

# Q-learning variables
Q_table = defaultdict(lambda: np.zeros(3))  # Example state-action space
alpha = 0.1  # Learning rate
gamma = 0.95  # Discount factor
epsilon = 0.1  # Exploration rate

# Simulate a single step of Q-learning (simplified)
def q_learning_step(state, action, reward, next_state):
    current_q_value = Q_table[state][action]
    next_max = np.max(Q_table[next_state])
    Q_table[state][action] = current_q_value + alpha * (reward + gamma * next_max - current_q_value)

# Streamlit App
st.title("Investment Decision Simulation")

# File uploader for input
uploaded_file = st.file_uploader("Upload your initial investment strategy (CSV)", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Investment Strategy", df)

# Inputs for investments
st.sidebar.title("Adjust Investment Levels")
oil_gas_investment = st.sidebar.slider("Oil & Gas Investment", 0, 100, 50)
renewables_investment = st.sidebar.slider("Renewables Investment", 0, 100, 50)
ccs_investment = st.sidebar.slider("CCS Investment", 0, 100, 50)

# Simulate and display results
if st.sidebar.button("Simulate"):
    production, co2_reduction = simulate_investment(oil_gas_investment, renewables_investment, ccs_investment)
    st.write(f"Simulated Production: {production}")
    st.write(f"CO2 Reduction: {co2_reduction}")

    # Update Q-learning (simplified)
    state = (oil_gas_investment, renewables_investment, ccs_investment)
    action = np.random.choice([0, 1, 2])  # Random action for now
    reward = production - co2_reduction  # Simple reward metric
    next_state = state  # No transition in this step
    q_learning_step(state, action, reward, next_state)

    # Display updated Q-table
    st.write("Updated Q-table:", Q_table)

# Placeholder for deep Q-learning section
st.write("Deep Q-Network (DQN) implementation coming soon!")

# Visualization section
fig, ax = plt.subplots()
ax.bar(["Oil & Gas", "Renewables", "CCS"], [oil_gas_investment, renewables_investment, ccs_investment])
ax.set_ylabel("Investment Amount")
st.pyplot(fig)
