# Indoor-Localization-System
The objective of this work is to develop an indoor localization system based on RF signal technologies which is adaptive and implementable in different circumstances, and applicable to different indoor scenarios. This system will focus on the signals of appropriate wireless communication technologies as the source of information to tackle the localization problem in an indoor environment.
To overcome the problem of the nonlinear uncertainty of the sensors in modelling the environment, we should explore and take advantage of using machine learning algorithms to handle the uncertainty and the nonlinearity.
Existing systems for tracking and positioning, use mathematical complex approaches for calculating the distance and the angles, so it takes more time to get a decision to find a path and navigate the user to the targeted point. In our project we will try to reduce the latency of computations using fuzzy systems.
---------------------------
To create an indoor localization system based on RF signal technologies that adapts to different environments, we can leverage several machine learning approaches. We can use various wireless communication signals like Wi-Fi, Bluetooth, or Zigbee to estimate the position of a device within an indoor environment. Fuzzy logic systems can help address the non-linearity and uncertainty of sensor data, and machine learning can be used to improve accuracy and reduce computation latency.
Key Objectives:

    Signal Data: We collect data from wireless communication technologies (Wi-Fi, Bluetooth, etc.).
    Fuzzy Logic System: This is used to model the uncertainty in sensor data and reduce computational latency.
    Machine Learning: Used to improve the estimation of positions and adapt the system to different environments.

Steps:

    Data Collection: Collect RF signals (RSSI values) from multiple access points (APs) in the indoor environment.
    Fuzzy Logic: Implement a fuzzy inference system to process the collected data and calculate distances, reducing computation time and uncertainty.
    Machine Learning: Use supervised learning (e.g., Regression, Random Forest, or KNN) to predict positions based on the signal data.

Components:

    Data Collection: RSSI values from Wi-Fi/Bluetooth beacons.
    Fuzzy System: A fuzzy logic controller to estimate distances and handle uncertainty.
    Machine Learning: Train a machine learning model on labeled data (positions corresponding to certain signal strengths).

Below is a Python code example that implements the general structure of this system.
Python Code Implementation
1. Data Collection (Simulated RSSI Values from Multiple Beacons)

In this part, we simulate RSSI data from multiple access points.

import numpy as np
import pandas as pd

# Simulating RSSI values from 3 beacons
# RSSI values are typically negative values, representing signal strength
def generate_rssi_data():
    # Simulating 5 samples for each beacon
    data = {
        'Beacon1': np.random.normal(loc=-50, scale=10, size=5),  # RSSI from Beacon 1
        'Beacon2': np.random.normal(loc=-60, scale=15, size=5),  # RSSI from Beacon 2
        'Beacon3': np.random.normal(loc=-55, scale=12, size=5)   # RSSI from Beacon 3
    }
    return pd.DataFrame(data)

# Generate sample data
rssi_data = generate_rssi_data()
print("Simulated RSSI Data:\n", rssi_data)

2. Fuzzy Logic System

Fuzzy logic can be used to map the RSSI values to distances. We use scikit-fuzzy to create fuzzy rules for distance estimation.

Install scikit-fuzzy if necessary:

pip install scikit-fuzzy

import skfuzzy as fuzz
import numpy as np
import matplotlib.pyplot as plt

# Fuzzy logic system for distance estimation based on RSSI
def fuzzy_distance(rssi):
    # Define fuzzy sets for RSSI
    rssi_range = np.arange(-100, 0, 1)
    distance_range = np.arange(0, 30, 1)

    # Membership functions for RSSI (Low, Medium, High)
    rssi_low = fuzz.trimf(rssi_range, [-100, -100, -50])
    rssi_medium = fuzz.trimf(rssi_range, [-100, -50, 0])
    rssi_high = fuzz.trimf(rssi_range, [-50, 0, 0])

    # Membership functions for Distance (Near, Mid, Far)
    distance_near = fuzz.trimf(distance_range, [0, 0, 15])
    distance_mid = fuzz.trimf(distance_range, [0, 15, 30])
    distance_far = fuzz.trimf(distance_range, [15, 30, 30])

    # Fuzzy rules
    rule1 = np.fmin(rssi_low, distance_far)
    rule2 = np.fmin(rssi_medium, distance_mid)
    rule3 = np.fmin(rssi_high, distance_near)

    # Aggregate the rules
    aggregated = np.fmax(rule1, np.fmax(rule2, rule3))

    # Defuzzify to get the final distance
    distance = fuzz.defuzz(distance_range, aggregated, 'centroid')
    return distance

# Example: Calculate the distance based on an RSSI value
rssi_value = -55
estimated_distance = fuzzy_distance(rssi_value)
print(f"Estimated distance based on RSSI {rssi_value} is {estimated_distance:.2f} meters.")

3. Machine Learning for Position Estimation

Next, we apply machine learning to estimate the position based on multiple RSSI readings. We'll use a simple regression model as an example.

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Simulate position data and corresponding RSSI values
def simulate_position_data():
    positions = np.random.uniform(low=0, high=30, size=(100, 2))  # 2D position (x, y)
    rssi_values = np.random.normal(loc=-60, scale=10, size=(100, 3))  # RSSI from 3 beacons

    data = pd.DataFrame(rssi_values, columns=['Beacon1', 'Beacon2', 'Beacon3'])
    data['x_position'] = positions[:, 0]
    data['y_position'] = positions[:, 1]
    return data

# Generate simulated dataset
dataset = simulate_position_data()

# Feature columns (RSSI values)
X = dataset[['Beacon1', 'Beacon2', 'Beacon3']]
y = dataset[['x_position', 'y_position']]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression model for position estimation
model = LinearRegression()
model.fit(X_train, y_train)

# Predict the position for a new set of RSSI values
new_rssi = np.array([[-50, -55, -60]])  # Example RSSI values
predicted_position = model.predict(new_rssi)
print(f"Predicted Position (x, y): {predicted_position[0]}")

Explanation:

    Data Collection: We simulate RSSI data from three beacons. In a real scenario, you would collect actual RSSI values from Wi-Fi, Bluetooth, or Zigbee devices.
    Fuzzy Logic: A fuzzy system maps the RSSI values to distances, helping to reduce uncertainty and model the environment more flexibly. The fuzzy logic helps to infer the distance from the RSSI signal.
    Machine Learning: We use a linear regression model to predict the position (x, y) of a device based on its RSSI readings from multiple beacons. The model is trained using labeled data, where RSSI values correspond to known positions.

Potential Improvements:

    Enhance Fuzzy Logic: More complex fuzzy systems or multi-input fuzzy systems could be implemented to handle multi-dimensional uncertainty.
    Advanced ML Models: Instead of linear regression, more complex machine learning models like Random Forest, SVM, or deep learning models could be used to handle non-linearity more effectively.
    Real-Time Data: For real-time systems, you would need to integrate with actual RF signal data streams and continuously predict the position of the user.
    Latency Reduction: If latency is a key concern, the fuzzy logic system will play a crucial role in reducing the complexity of distance estimation and improving the real-time responsiveness of the system.

This system can be adapted for different indoor scenarios and offer an adaptive, responsive localization solution using RF signals.
