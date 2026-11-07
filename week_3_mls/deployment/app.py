import os
import streamlit as st
import pandas as pd
import joblib

# Load the model committed by the pipeline (sits next to this file)
model_path = os.path.join(os.path.dirname(__file__), "best_machine_failure_model_v1.joblib")
model = joblib.load(model_path)
# You don't create this file manually in Colab.
# When your GitHub Actions pipeline runs (specifically the Model Training job), 
#it executes the train.py script.
# The train.py script trains the model and saves it as best_machine_failure_model_v1.joblib 
#in the week_3_mls/deployment/ folder.
# Finally, the pipeline automatically commits and pushes this newly created .joblib file straight 
# to your GitHub repository.
#Because of this, when Streamlit runs app.py from your GitHub repo,
# os.path.dirname(__file__) simply tells it to look in its current folder, 
# where it will successfully find the model sitting right beside it.

st.title("Machine Failure Prediction App")
st.write("""
This application predicts the likelihood of a machine failing based on its operational parameters.
Enter the sensor and configuration data below to get a prediction.
""")

Type         = st.selectbox("Machine Type", ["H", "L", "M"])
air_temp     = st.number_input("Air Temperature (K)", 250.0, 400.0, 298.0, 0.1)
process_temp = st.number_input("Process Temperature (K)", 250.0, 500.0, 324.0, 0.1)
rot_speed    = st.number_input("Rotational Speed (RPM)", 0, 3000, 1400)
torque       = st.number_input("Torque (Nm)", 0.0, 100.0, 40.0, 0.1)
tool_wear    = st.number_input("Tool Wear (min)", 0, 300, 10)

input_data = pd.DataFrame([{
    "Air temperature": air_temp,
    "Process temperature": process_temp,
    "Rotational speed": rot_speed,
    "Torque": torque,
    "Tool wear": tool_wear,
    "Type": Type,
}])

if st.button("Predict Failure"):
    prediction = model.predict(input_data)[0]
    result = "Machine Failure" if prediction == 1 else "No Failure"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
