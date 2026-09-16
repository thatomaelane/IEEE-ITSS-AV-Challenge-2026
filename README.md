# IEEE ITSS AV Challenge 2026

> An autonomous vehicle trajectory prediction system developed for the IEEE ITSS AV Challenge 2026, achieving a **Top 15 ranking worldwide**, using an Enhanced LSTM architecture, attention mechanisms, behavioral vehicle features, ensemble learning, and physics-inspired trajectory modeling to predict the future trajectory of a Following Automated Vehicle (FAV).

![Python](https://img.shields.io/badge/Python-3.x-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange)
![LSTM](https://img.shields.io/badge/Model-LSTM-green)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-green)
![IEEE](https://img.shields.io/badge/IEEE-ITSS-red)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

---

# Project

## IEEE ITSS AV Challenge 2026

**Participant:**  
**Thato Maelane**

**Final Competition Result:**  
**Top 15 Worldwide**

This project was developed as part of the **IEEE ITSS AV Challenge 2026**, organized through the IEEE Intelligent Transportation Systems Society (ITSS) Emerging Transportation Technology Testing (ET3) Technical Committee.

The challenge focused on developing a behavioral model capable of predicting the future trajectory of a Following Automated Vehicle (FAV).

---

# Competition

The IEEE ITSS AV Challenge required participants to develop a behavioral model using the provided training data and predict the future trajectory of the FAV for each test segment.

For each trajectory segment:

- The training dataset provides the complete 30-second trajectories of the Lead Vehicle (LV) and Following Automated Vehicle (FAV).
- The test dataset provides the complete 30-second LV trajectory.
- The first 10 seconds of the FAV trajectory are provided.
- The model predicts the remaining 20 seconds of the FAV trajectory.
- The prediction horizon consists of 200 time steps.
- The simulation time step is 0.1 seconds.

---

# Project Objective

The main objective was to develop a robust time-series behavioral model capable of predicting realistic future FAV trajectories while considering:

- Vehicle-following behavior
- Lead vehicle dynamics
- Relative velocity
- Relative acceleration
- Spatial headway
- Time-to-collision
- Vehicle characteristics
- Acceleration
- Jerk and trajectory smoothness

The model was designed to balance trajectory accuracy, safety and comfort.

---

# Final Result

## Top 15 Worldwide

The submitted solution achieved a **Top 15 ranking worldwide** on the IEEE ITSS AV Challenge 2026 leaderboard.

### Official Leaderboard

https://ieee-et3-challenge.com/leaderboard/

---

# Model Overview

The final solution combines deep learning with behavioral and physics-inspired modeling.

The architecture consists of:

- Historical trajectory encoder
- Future Lead Vehicle encoder
- Bidirectional LSTM
- Attention mechanism
- Autoregressive LSTM decoder
- IDM-inspired behavioral features
- Acceleration-based trajectory generation
- Ensemble learning
- Low-pass acceleration filtering
- Kinematic trajectory integration

---

# Model Architecture

```text
Historical FAV/LV Trajectory
            │
            ▼
     Feature Engineering
            │
            ▼
      LSTM Encoder
            │
            │
            ├───────────────┐
            │               │
            ▼               ▼
     Behavioral State   Future LV Data
                            │
                            ▼
                  Bidirectional LSTM
                            │
                            ▼
                       Attention
                            │
                            ▼
                  Autoregressive Decoder
                            │
                            ▼
                  Predicted Acceleration
                            │
                            ▼
                  Ensemble Averaging
                            │
                            ▼
                   Low-Pass Filtering
                            │
                            ▼
                  Kinematic Integration
                            │
                            ▼
                  Predicted FAV Position
