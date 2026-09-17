# IEEE ITSS AV Challenge 2026

> An autonomous vehicle trajectory prediction system developed for the IEEE ITSS AV Challenge 2026, achieving a **Top 15 ranking worldwide**, using an Enhanced LSTM architecture, attention mechanisms, behavioral vehicle features, ensemble learning, and physics-inspired trajectory modeling to predict the future trajectory of a Following Automated Vehicle (FAV).

![Python](https://img.shields.io/badge/Python-3.x-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange)
![LSTM](https://img.shields.io/badge/Model-LSTM-green)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-green)
![IEEE](https://img.shields.io/badge/IEEE-ITSS-red)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

---

## Project

### IEEE ITSS AV Challenge 2026

**Participant:** Thato Maelane

**Final Competition Result:** 🏆 **Top 15 Worldwide**

This project was developed as part of the **IEEE ITSS AV Challenge 2026**, organized through the IEEE Intelligent Transportation Systems Society (ITSS) Emerging Transportation Technology Testing (ET3) Technical Committee.

The challenge focused on developing a behavioral model capable of predicting the future trajectory of a Following Automated Vehicle (FAV).

---

## Competition

The IEEE ITSS AV Challenge required participants to develop a behavioral model using the provided training data and predict the future trajectory of the FAV for each test segment.

For each trajectory segment:

- The training dataset provides the complete 30-second trajectories of the Lead Vehicle (LV) and Following Automated Vehicle (FAV).
- The test dataset provides the complete 30-second LV trajectory.
- The first 10 seconds of the FAV trajectory are provided.
- The model predicts the remaining 20 seconds of the FAV trajectory.
- The prediction horizon consists of 200 time steps.
- The simulation time step is 0.1 seconds.

---

## Project Objective

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

## Final Result

### 🏆 Top 15 Worldwide

The submitted solution achieved a **Top 15 ranking worldwide** on the IEEE ITSS AV Challenge 2026 leaderboard.

**Official Leaderboard:** https://ieee-et3-challenge.com/leaderboard/

---

## Model Overview

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

## Model Architecture

```
Historical FAV/LV Trajectory
            │
            ▼
     Feature Engineering
            │
            ▼
      LSTM Encoder
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
```

---

## Historical Feature Engineering

The model uses historical information from both the FAV and LV to capture vehicle-following behavior.

The feature set includes:

- FAV speed
- LV speed
- Spatial gap
- Relative velocity
- Relative acceleration
- LV acceleration
- LV vehicle type
- LV vehicle ID
- Spatial headway
- Time-to-collision (TTC)
- IDM-inspired acceleration

The spatial gap is calculated using:

```
Gap = Pos_LV - Pos_FAV - 4.5
```

Relative velocity is calculated as:

```
Relative Velocity = Speed_LV - Speed_FAV
```

---

## IDM-Inspired Behavioral Modeling

An Intelligent Driver Model (IDM)-inspired acceleration feature was incorporated into the historical feature set.

The IDM component provides an additional behavioral representation of vehicle-following dynamics based on:

- Current FAV velocity
- LV velocity
- Vehicle gap
- Desired headway
- Maximum acceleration
- Comfortable deceleration

The IDM acceleration is used as an additional input feature rather than as the final prediction model.

---

## Future Lead Vehicle Encoder

The challenge provides the complete future LV trajectory.

The model uses future LV information including:

- LV speed
- LV acceleration
- LV position

A bidirectional LSTM is used to encode the future LV trajectory. This allows the model to capture temporal relationships within the lead vehicle's future movement.

---

## Attention Mechanism

An attention mechanism connects the decoder with the encoded future LV trajectory.

The decoder generates a query based on its current hidden state, while the future LV encoder provides keys and values. This allows the model to dynamically determine which parts of the future LV trajectory are most relevant during each prediction step.

The attention dimension used in the model is: **64**

---

## Autoregressive Prediction

The FAV trajectory is generated sequentially over the 20-second prediction horizon.

At each timestep, the decoder considers:

- Current FAV velocity
- Future LV velocity
- Current vehicle gap
- Relative velocity
- LV acceleration
- Current FAV acceleration
- Attention context

The decoder predicts an acceleration adjustment, which is then smoothed and integrated to obtain the next FAV position.

---

## Acceleration-Based Trajectory Generation

Instead of directly predicting position independently at every timestep, the model predicts acceleration and generates the trajectory through kinematic integration.

Position is calculated using:

```
x(t+1) = x(t) + v(t)Δt + 0.5a(t)Δt²
```

Velocity is updated using:

```
v(t+1) = max(0, v(t) + a(t)Δt)
```

This approach maintains temporal consistency between predicted position, velocity and acceleration.

---

## Acceleration Smoothing

The predicted acceleration is smoothed using:

```
a(t) = 0.25 × predicted_acceleration + 0.75 × previous_acceleration
```

A low-pass Butterworth filter is subsequently applied to the ensemble acceleration predictions.

The selected smoothing cutoff was: **1.0 Hz**

This reduces high-frequency fluctuations in the predicted acceleration before trajectory integration.

---

## Ensemble Learning

The final prediction system uses an ensemble of five independently initialized models. Each model is trained using a different random seed.

The acceleration predictions from the five models are averaged:

```
Ensemble Acceleration = (A1 + A2 + A3 + A4 + A5) / 5
```

The ensemble prediction is then filtered and integrated to generate the final FAV trajectory.

---

## Training

The model was trained using the provided training trajectory data.

Training included:

- Segment-based train/validation split
- Feature normalization
- Teacher forcing
- Smooth L1 loss
- Velocity loss
- Headway loss
- Acceleration loss
- Jerk regularization
- Gradient clipping
- Adam optimization
- Cosine annealing learning-rate scheduling
- Five-model ensemble training

A fixed random seed of **42** was used for the validation split.

### Training Configuration

| Parameter | Value |
|---|---:|
| Time step | 0.1 seconds |
| History length | 100 |
| Prediction horizon | 200 |
| Hidden dimension | 384 |
| Attention dimension | 64 |
| Dropout | 0.15 |
| Batch size | 256 |
| Epochs | 150 |
| Ensemble size | 5 |
| Learning rate | 0.001 |
| Minimum learning rate | 0.000001 |
| Smoothing cutoff | 1.0 Hz |

---

## Training Objective

The training loss combines several trajectory objectives:

- Position Loss
- Velocity Loss
- Headway Loss
- Acceleration Loss
- Jerk Regularization

The loss function was designed to encourage accurate trajectory prediction while promoting appropriate vehicle-following behavior and smooth motion.

The implemented loss weighting is:

```
Loss = 0.05 × Position Loss
     + 0.20 × Velocity Loss
     + 0.60 × Headway Loss
     + 0.15 × Acceleration Loss
     + 0.05 × Jerk Loss
```

---

## Teacher Forcing

Teacher forcing is gradually reduced during training. The training process transitions from:

```
Ground-truth trajectory feedback
              ↓
Reduced teacher forcing
              ↓
Autoregressive prediction
```

This allows the model to progressively learn to operate using its own predicted trajectory.

---

## Evaluation

The project includes a local evaluation implementation based on the challenge scoring structure.

The evaluation considers three main components:

**Accuracy** — evaluates:
- Velocity RMSE
- Headway RMSE
- Acceleration RMSE

**Safety** — evaluates:
- Spatial gap
- Time-to-collision (TTC)

**Comfort** — evaluates:
- Predicted jerk

The overall score combines:

```
Total Score = 0.5 × Accuracy + 0.3 × Safety + 0.2 × Comfort
```

---

## Validation Strategy

The training data was divided by trajectory segment rather than randomly splitting individual timesteps.

A fixed random seed of **42** was used. Approximately 3% of the available trajectory segments were reserved for validation.

The validation set was used to evaluate model performance and select the best-performing model checkpoints.

---

## Submission

The final submission contains:

- `Segment_ID`
- `Time_Index`
- `Pos_FAV`

Each trajectory segment contains **200 predicted timesteps**, representing the remaining 20 seconds of the trajectory.

The final submission file is: `submission.csv`

The submission pipeline also performs validation checks to confirm that:

- Required columns are present.
- Each segment contains exactly 200 predictions.
- The prediction horizon is correctly generated.

---

## Technologies Used

**Programming**
- Python

**Deep Learning**
- PyTorch
- LSTM
- Bidirectional LSTM
- Attention Mechanism

**Data Processing**
- NumPy
- Pandas

**Machine Learning**
- Time-Series Modeling
- Ensemble Learning
- Behavioral Modeling
- Physics-Inspired Machine Learning

**Signal Processing**
- SciPy
- Butterworth Low-Pass Filtering

**Python Libraries**
- numpy
- pandas
- torch
- scipy

---

## Repository Structure

```
IEEE-ITSS-AV-Challenge-2026/
│
├── README.md
│
├── model.py
├── evaluate.py
│
├── train.csv
├── test.csv
│
├── submission.csv
│
├── requirements.txt
│
├── Models/
│   ├── ensemble_kaggle_0.pt
│   ├── ensemble_kaggle_1.pt
│   ├── ensemble_kaggle_2.pt
│   ├── ensemble_kaggle_3.pt
│   └── ensemble_kaggle_4.pt
│
└── Images/
    ├── model_architecture.png
    ├── trajectory_prediction.png
    └── results.png
```

> **Note:** The challenge dataset should only be redistributed if permitted by the challenge organizers. If redistribution is restricted, keep `train.csv` and `test.csv` out of the public repository and provide instructions for obtaining them from the official challenge website.

---

## Project Files

**`model.py`**

Contains the main trajectory prediction pipeline, including:

- Data preparation
- Feature engineering
- Enhanced LSTM
- Attention mechanism
- IDM-inspired features
- Autoregressive prediction
- Ensemble inference
- Acceleration smoothing
- Trajectory integration
- Submission generation

**`evaluate.py`**

Contains the local implementation of the trajectory evaluation metrics, including:

- Accuracy
- Safety
- Comfort
- Total score

**`submission.csv`**

Contains the final FAV trajectory predictions submitted to the competition.

---

## Skills Demonstrated

- Autonomous Vehicle Modeling
- Intelligent Transportation Systems
- Time-Series Forecasting
- Deep Learning
- LSTM Networks
- Attention Mechanisms
- Ensemble Learning
- Behavioral Vehicle Modeling
- Feature Engineering
- Physics-Inspired Modeling
- Trajectory Prediction
- Vehicle-Following Dynamics
- Safety-Aware Prediction
- Signal Processing
- Model Evaluation
- Machine Learning Pipeline Development
- Reproducible Machine Learning

---

## Key Learning Outcomes

This project provided practical experience in developing machine-learning models for autonomous vehicle applications.

Key areas included:

- Modeling vehicle-following behavior
- Predicting long-horizon trajectories
- Combining machine learning with behavioral models
- Designing time-series neural networks
- Incorporating attention into sequential prediction
- Using ensemble methods for trajectory prediction
- Evaluating trajectory accuracy, safety and comfort
- Applying signal processing to machine-learning outputs
- Converting acceleration predictions into physically consistent trajectories

---

## Challenge Timeline

| Event | Date |
|---|---|
| Challenge Start | June 1, 2026 |
| Model Development | June – August 2026 |
| Submission Deadline | August 31, 2026 |
| Winners Announcement | October 1, 2026 |

---

## Official Challenge Resources

- Challenge Website — https://ieee-et3-challenge.com/
- Challenge Overview — https://ieee-et3-challenge.com/overview/
- Dataset — https://ieee-et3-challenge.com/data/
- Evaluation — https://ieee-et3-challenge.com/evaluation/
- Submission — https://ieee-et3-challenge.com/submission/
- Leaderboard — https://ieee-et3-challenge.com/leaderboard/

---

## Final Result

### 🏆 Top 15 Worldwide — IEEE ITSS AV Challenge 2026

My solution achieved a Top 15 ranking worldwide on the final IEEE ITSS AV Challenge 2026 leaderboard.

The project demonstrates the application of deep learning, behavioral modeling, ensemble learning and intelligent transportation concepts to autonomous vehicle trajectory prediction.

---

## Author

**Thato Maelane**

Electrical Engineering | Data Science | Artificial Intelligence | Intelligent Transportation Systems

GitHub: https://github.com/thatomaelane

---

## Acknowledgement

I would like to acknowledge the IEEE Intelligent Transportation Systems Society (IEEE ITSS) and the Emerging Transportation Technology Testing (ET3) Technical Committee for organizing the IEEE ITSS AV Challenge 2026.

The challenge provided an opportunity to apply machine learning and behavioral modeling techniques to an autonomous vehicle trajectory prediction problem.

---

⭐ If you found this project useful, consider giving the repository a star!
