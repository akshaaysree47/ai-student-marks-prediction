import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(
    page_title="AI Student Marks Prediction",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 AI Student Marks Prediction")
st.subheader("Mathematics Behind Artificial Intelligence")
st.write(
    "This project demonstrates how mathematics, statistics and "
    "Artificial Intelligence can be used to predict student marks."
)
st.divider()

# Demonstration dataset
data = {
    "Study_Hours": [
        2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5,
        7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 3.0, 4.0, 5.0,
        6.0, 7.0, 8.0, 9.0, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5,
        8.5, 9.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0
    ],
    "Attendance": [
        65, 70, 72, 75, 78, 80, 82, 85, 87, 88,
        90, 91, 92, 94, 95, 96, 97, 68, 74, 79,
        84, 89, 93, 95, 66, 73, 77, 81, 86, 90,
        92, 96, 60, 71, 76, 83, 86, 91, 94, 97
    ],
    "Previous_Marks": [
        45, 48, 50, 52, 55, 57, 60, 62, 65, 67,
        70, 72, 75, 77, 80, 82, 85, 46, 51, 58,
        63, 68, 73, 78, 44, 49, 54, 59, 64, 69,
        74, 81, 40, 47, 53, 61, 66, 71, 76, 83
    ],
    "Final_Marks": [
        48, 51, 55, 58, 62, 64, 68, 71, 74, 77,
        80, 82, 85, 88, 91, 93, 95, 52, 59, 66,
        73, 79, 84, 90, 49, 56, 62, 68, 75, 81,
        86, 93, 43, 53, 60, 69, 75, 81, 87, 93
    ]
}

df = pd.DataFrame(data)

# Train model
X = df[["Study_Hours", "Attendance", "Previous_Marks"]]
y = df["Final_Marks"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Equation
intercept = model.intercept_
study_coef = model.coef_[0]
attendance_coef = model.coef_[1]
previous_coef = model.coef_[2]

# Sidebar
st.sidebar.header("📝 Enter Student Details")

student_name = st.sidebar.text_input("Student Name", "Rahul")

study_hours = st.sidebar.slider(
    "Study Hours Per Day", 1.0, 12.0, 7.0, 0.5
)

attendance = st.sidebar.slider(
    "Attendance (%)", 40, 100, 90
)

previous_marks = st.sidebar.slider(
    "Previous Exam Marks", 0, 100, 70
)

predict_button = st.sidebar.button("🔮 PREDICT MARKS")

if predict_button:
    new_student = pd.DataFrame({
        "Study_Hours": [study_hours],
        "Attendance": [attendance],
        "Previous_Marks": [previous_marks]
    })

    prediction = model.predict(new_student)[0]
    prediction = max(0, min(100, prediction))

    if prediction >= 90:
        grade = "A+"
    elif prediction >= 80:
        grade = "A"
    elif prediction >= 70:
        grade = "B+"
    elif prediction >= 60:
        grade = "B"
    elif prediction >= 50:
        grade = "C"
    elif prediction >= 40:
        grade = "D"
    else:
        grade = "F"

    st.success(f"Prediction completed for {student_name}!")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("🎯 Predicted Marks", f"{prediction:.2f} / 100")

    with col2:
        st.metric("🏆 Grade", grade)

    with col3:
        st.metric("📊 R² Score", f"{r2:.2%}")

    st.divider()

    st.subheader("👨‍🎓 Student Information")

    student_table = pd.DataFrame({
        "Parameter": [
            "Student Name",
            "Study Hours",
            "Attendance",
            "Previous Marks",
            "Predicted Marks",
            "Grade"
        ],
        "Value": [
            student_name,
            f"{study_hours} hours/day",
            f"{attendance}%",
            previous_marks,
            f"{prediction:.2f}",
            grade
        ]
    })

    st.table(student_table)

    st.subheader("🧮 Mathematics Behind the Prediction")

    st.latex(r"y = b_0 + b_1x_1 + b_2x_2 + b_3x_3")

    st.write("**y** = Predicted Marks")
    st.write("**x₁** = Study Hours")
    st.write("**x₂** = Attendance")
    st.write("**x₃** = Previous Marks")

    st.write(
        f"**Learned Equation:** Final Marks = {intercept:.2f} "
        f"+ ({study_coef:.2f} × Study Hours) "
        f"+ ({attendance_coef:.2f} × Attendance) "
        f"+ ({previous_coef:.2f} × Previous Marks)"
    )

    st.subheader("📈 Study Hours vs Final Marks")

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.scatter(
        df["Study_Hours"],
        df["Final_Marks"],
        label="Student Data"
    )

    hours = np.linspace(
        df["Study_Hours"].min(),
        df["Study_Hours"].max(),
        100
    )

    average_attendance = df["Attendance"].mean()
    average_previous = df["Previous_Marks"].mean()

    graph_data = pd.DataFrame({
        "Study_Hours": hours,
        "Attendance": average_attendance,
        "Previous_Marks": average_previous
    })

    graph_prediction = model.predict(graph_data)

    ax.plot(
        hours,
        graph_prediction,
        label="AI Prediction Line"
    )

    ax.set_xlabel("Study Hours")
    ax.set_ylabel("Final Marks")
    ax.set_title("Study Hours vs Final Marks")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)

    st.subheader("📊 Model Performance")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("MAE", f"{mae:.2f} marks")

    with col2:
        st.metric("RMSE", f"{rmse:.2f} marks")

    with col3:
        st.metric("R² Score", f"{r2:.2%}")

else:
    st.info(
        "👈 Enter student details and click **PREDICT MARKS** "
        "to see the result."
    )

st.divider()

st.subheader("💡 How Does This Project Work?")

st.write(
    """
    **Step 1:** Student data is collected.

    **Step 2:** Mathematics and statistics are used to analyze the data.

    **Step 3:** Multiple Linear Regression learns the relationship
    between the input variables.

    **Step 4:** The trained AI model predicts marks for a new student.

    **Step 5:** The prediction is displayed as marks and grade.

    **Input → Mathematical Model → AI → Prediction**
    """
)

st.caption(
    "Note: The dataset used in this project is synthetic data "
    "created for educational demonstration."
)
