# 🍔 Food Delivery Time Predictor

An end-to-end **Machine Learning web application** that predicts food delivery time in minutes using **Random Forest Regression**.

The application takes delivery-related information such as distance, weather, traffic, time of day, vehicle type, preparation time, and courier experience, and provides an estimated delivery time through an interactive **Streamlit dashboard**.

---

## 🚀 Project Overview

Food delivery time depends on multiple factors, including delivery distance, traffic conditions, weather, food preparation time, and courier experience.

This project uses a **Random Forest Regression model** to learn relationships between these factors and historical delivery times.

The trained model is integrated into a **Streamlit web application**, allowing users to enter delivery information and receive an estimated delivery time instantly.

### 🎯 Objective

> Build a practical machine learning application that can estimate food delivery time from real-world delivery conditions.

---

## ✨ Features

* 🤖 Random Forest Regression for delivery-time prediction
* 📊 Interactive Streamlit web interface
* 🌦️ Weather condition input
* 🚦 Traffic-level input
* 🕐 Time-of-day selection
* 🛵 Vehicle type selection
* 📍 Delivery distance input
* 🍳 Food preparation time input
* 👨‍💼 Courier experience input
* 🔐 Saved ML model and Label Encoders
* 📈 Model feature importance visualization
* 🧠 Model information and parameters
* 📋 Order summary after prediction
* ⚡ Fast real-time predictions
* 🎨 Modern dark-themed dashboard

---

## 🧠 Machine Learning Workflow

The project follows a complete machine learning workflow:

```text
Dataset
   ↓
Data Cleaning
   ↓
Missing Value Handling
   ↓
Feature Selection
   ↓
Categorical Encoding
   ↓
Train-Test Split
   ↓
Decision Tree Baseline
   ↓
Random Forest Regression
   ↓
Hyperparameter Tuning
   ↓
Best Model Selection
   ↓
Model Serialization
   ↓
Streamlit Deployment
```

---

## 📊 Dataset

The model was trained using a food delivery dataset containing **1,000 records and 9 columns**.

### Dataset Features

| Feature                  | Description                                |
| ------------------------ | ------------------------------------------ |
| `Order_ID`               | Unique identifier for each order           |
| `Distance_km`            | Delivery distance in kilometers            |
| `Weather`                | Weather condition during delivery          |
| `Traffic_Level`          | Traffic condition                          |
| `Time_of_Day`            | Time period of the delivery                |
| `Vehicle_Type`           | Vehicle used by the courier                |
| `Preparation_Time_min`   | Food preparation time                      |
| `Courier_Experience_yrs` | Courier's experience in years              |
| `Delivery_Time_min`      | Target variable — delivery time in minutes |

### Target Variable

```text
Delivery_Time_min
```

The model predicts the expected delivery time in minutes.

---

## 🧹 Data Preprocessing

Before training the model, the dataset was cleaned and prepared.

### Missing Values

Categorical missing values were replaced using their respective **mode**:

* Weather
* Traffic Level
* Time of Day

The missing values in `Courier_Experience_yrs` were handled using the **median** and converted to integer values.

### Feature Removal

`Order_ID` was removed because it is an identifier and does not provide useful predictive information.

### Categorical Encoding

Categorical variables were converted into numerical values using separate `LabelEncoder` objects.

The trained encoders are stored in:

```text
label_encoders.pkl
```

This ensures that the Streamlit application applies the same encoding used during model training.

---

## 🌲 Random Forest Regression

The final prediction model is a:

```text
RandomForestRegressor
```

Random Forest combines multiple decision trees and averages their predictions to produce a more robust regression result.

### Model Configuration

The saved model uses:

```text
n_estimators = 100
max_depth = None
min_samples_split = 10
min_samples_leaf = 4
random_state = 42
bootstrap = True
criterion = squared_error
```

Hyperparameters were optimized using **GridSearchCV with 5-fold cross-validation**.

---

## 📌 Model Features

The trained model uses the following 7 features:

```text
Distance_km
Weather
Traffic_Level
Time_of_Day
Vehicle_Type
Preparation_Time_min
Courier_Experience_yrs
```

### Feature Importance

The trained Random Forest model identified the following relative importance:

| Feature                | Importance |
| ---------------------- | ---------: |
| Distance_km            |     0.7635 |
| Preparation_Time_min   |     0.1484 |
| Traffic_Level          |     0.0221 |
| Courier_Experience_yrs |     0.0289 |
| Weather                |     0.0199 |
| Vehicle_Type           |     0.0097 |
| Time_of_Day            |     0.0075 |

The results indicate that **delivery distance is the most influential feature** in the trained model, followed by food preparation time.

---

## 🖥️ Streamlit Application

The project includes an interactive Streamlit application.

Users can enter:

```text
📍 Distance
🌦️ Weather
🚦 Traffic Level
🕐 Time of Day
🛵 Vehicle Type
🍳 Preparation Time
👨‍💼 Courier Experience
```

The application then generates:

```text
Estimated Delivery Time
```

For example:

```text
Estimated Delivery Time
        54 minutes
```

The application also categorizes the estimated delivery time as:

| Prediction  | Category          |
| ----------- | ----------------- |
| `< 30 min`  | Fast Delivery     |
| `30–60 min` | Standard Delivery |
| `> 60 min`  | Longer Delivery   |

---

## 📁 Project Structure

```text
food_delivery_predictor/
│
├── app.py
├── best_rf_model.pkl
├── label_encoders.pkl
├── requirements.txt
├── README.md
└── Food_Delivery_Times.csv
```

> The dataset is not required by the Streamlit application at prediction time because the trained model and encoders are already serialized.

---

## ⚙️ Technologies Used

### Programming Language

* Python

### Machine Learning

* Scikit-learn
* Random Forest Regression
* Decision Tree Regression
* GridSearchCV
* Label Encoding

### Data Processing

* Pandas
* NumPy

### Visualization

* Matplotlib

### Web Application

* Streamlit

### Model Serialization

* Pickle

---

## 🔧 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/food-delivery-time-predictor.git
```

Navigate into the project:

```bash
cd food-delivery-time-predictor
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser.

Usually, Streamlit runs at:

```text
http://localhost:8501
```

---

## 🧪 Example Prediction

Example input:

```text
Distance: 10.0 km
Weather: Clear
Traffic Level: Medium
Time of Day: Evening
Vehicle Type: Bike
Preparation Time: 17 minutes
Courier Experience: 5 years
```

Example model output:

```text
Predicted Delivery Time: ~54 minutes
```

The exact prediction is generated by the trained Random Forest model.

---

## 🔍 Application Sections

### 🏠 Prediction

The main prediction dashboard where users enter delivery information and receive the estimated delivery time.

### 📊 Model Information

Displays:

* Model type
* Model parameters
* Input features
* Feature importance
* Machine learning information

### 📚 About Project

Provides information about:

* Dataset
* Data preprocessing
* Machine learning workflow
* Technologies used
* Project objective

---

## 📦 Model Files

### `best_rf_model.pkl`

Contains the trained Random Forest Regression model.

### `label_encoders.pkl`

Contains the LabelEncoder objects used for categorical variables.

These files allow the application to make predictions without retraining the model every time it starts.

---

## 💡 Key Learning Outcomes

Through this project, I worked with:

* Data preprocessing
* Missing-value handling
* Exploratory data preparation
* Categorical encoding
* Train-test splitting
* Decision Tree Regression
* Random Forest Regression
* Ensemble learning
* Hyperparameter tuning
* GridSearchCV
* Cross-validation
* Feature importance analysis
* Model serialization using Pickle
* Streamlit application development
* Machine Learning model integration

---

## 👨‍💻 Author

**Vishal Bhatti**

B.Tech — AI & Robotics Engineering

Interested in:

* 🤖 Artificial Intelligence
* 🧠 Machine Learning
* 📊 Data Science
* 👁️ Computer Vision
* 🐍 Python
* 🚀 AI Engineering

---

## ⭐ If You Like This Project

If you found this project useful or interesting:

⭐ Star the repository
🍴 Fork the project
📢 Share it with others
💡 Feel free to contribute or suggest improvements

---

## 📄 License

This project is created for **educational and portfolio purposes**.

This version is designed to look good on GitHub while still showing recruiters that you understand the **full ML pipeline**, not just the Streamlit UI.
