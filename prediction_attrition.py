import datetime
import pandas as pd
import streamlit as st
import joblib
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import os
from typing import Dict, Any, Tuple, List

# --- Constants ---
DATA_FILE = 'employee_data_cleaned.csv'
MODEL_FILE = 'best_model_gb.joblib'
REQUIRED_COLUMNS = [ # Define the exact order expected by the model after preprocessing
    'Age', 'BusinessTravel', 'DailyRate', 'Department', 'DistanceFromHome',
    'Education', 'EducationField', 'EnvironmentSatisfaction', 'Gender',
    'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction',
    'MaritalStatus', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked',
    'OverTime', 'PercentSalaryHike', 'PerformanceRating',
    'RelationshipSatisfaction', 'StandardHours', 'StockOptionLevel',
    'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',
    'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
    'YearsWithCurrManager'
]

# --- Mappings for Ordinal Features ---
EDUCATION_MAP = {'Below College': 1, 'College': 2, 'Bachelor': 3, 'Master': 4, 'Doctor': 5}
SATISFACTION_MAP = {'Low': 1, 'Medium': 2, 'High': 3, 'Very High': 4}
JOB_INVOLVEMENT_MAP = {'Low': 1, 'Medium': 2, 'High': 3, 'Very High': 4}
PERFORMANCE_RATING_MAP = {'Low': 1, 'Good': 2, 'Excellent': 3, 'Outstanding': 4}
WORK_LIFE_BALANCE_MAP = {'Bad': 1, 'Good': 2, 'Better': 3, 'Best': 4}

# Helper list of columns that use explicit mapping
EXPLICITLY_MAPPED_COLS = [
    'Education', 'EnvironmentSatisfaction', 'JobInvolvement', 'JobSatisfaction',
    'PerformanceRating', 'RelationshipSatisfaction', 'WorkLifeBalance'
]
MAP_DICTIONARIES = {
    'Education': EDUCATION_MAP,
    'EnvironmentSatisfaction': SATISFACTION_MAP,
    'JobInvolvement': JOB_INVOLVEMENT_MAP,
    'JobSatisfaction': SATISFACTION_MAP,
    'PerformanceRating': PERFORMANCE_RATING_MAP,
    'RelationshipSatisfaction': SATISFACTION_MAP,
    'WorkLifeBalance': WORK_LIFE_BALANCE_MAP
}

# --- Caching Functions ---

@st.cache_data
def load_base_data(file_path: str) -> pd.DataFrame:
    """Loads the cleaned base dataset."""
    if not os.path.exists(file_path):
        st.error(f"Error: Dataset file '{file_path}' not found.")
        st.stop()
    try:
        df = pd.read_csv(file_path)
        # Use REQUIRED_COLUMNS to select and order, handle missing ones
        present_cols = [col for col in REQUIRED_COLUMNS if col in df.columns]
        if len(present_cols) != len(REQUIRED_COLUMNS):
             missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
             st.warning(f"Warning: Missing expected columns in {file_path}: {missing}. Fitting preprocessors only with available columns: {present_cols}")
        # Return a copy with only present required columns in the correct order
        return df[present_cols].copy()
    except Exception as e:
        st.error(f"Error loading data from {file_path}: {e}")
        st.stop()

@st.cache_resource
def load_model(file_path: str):
    """Loads the pre-trained machine learning model."""
    if not os.path.exists(file_path):
        st.error(f"Error: Model file '{file_path}' not found.")
        st.stop()
    try:
        model = joblib.load(file_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

@st.cache_resource
def get_preprocessors(df_train: pd.DataFrame) -> Tuple[Dict[str, LabelEncoder], MinMaxScaler, List[str], List[str]]:
    """
    Fits and returns label encoders (for nominal) and a min-max scaler
    (for all numerical including mapped ordinal).
    """
    df_train_processed = df_train.copy() # Work on a copy

    # --- Identify Column Types ---
    potential_object_cols = df_train_processed.select_dtypes(include='object').columns.tolist()
    original_numerical_cols = df_train_processed.select_dtypes(exclude='object').columns.tolist()

    # 1. Nominal Categorical Columns (Object type AND NOT explicitly mapped)
    nominal_categorical_cols = [
        col for col in potential_object_cols
        if col in df_train_processed.columns and col not in EXPLICITLY_MAPPED_COLS
    ]

    # 2. Apply Mappings to Ordinal Columns IN THE COPY used for fitting scaler
    for col, mapping in MAP_DICTIONARIES.items():
        if col in df_train_processed.columns:
            # Map string categories to numbers based on the defined maps
            # Handle potential missing keys in map or values in data gracefully if needed
            df_train_processed[col] = df_train_processed[col].map(mapping).fillna(-1) # Fill unknowns with -1 or handle appropriately
            # Ensure dtype is numeric after mapping
            df_train_processed[col] = pd.to_numeric(df_train_processed[col], errors='coerce')

    # 3. Ensure JobLevel and StockOptionLevel are numeric IN THE COPY
    for col in ['JobLevel', 'StockOptionLevel']:
        if col in df_train_processed.columns:
            df_train_processed[col] = pd.to_numeric(df_train_processed[col], errors='coerce')
            # Remove from original_numerical_cols if present to avoid adding twice later
            if col in original_numerical_cols:
                 original_numerical_cols.remove(col)

    # 4. Identify ALL Columns to be Scaled
    # Combine original numerical + explicitly mapped (now numeric) + JobLevel/StockOptionLevel
    columns_to_scale = list(set(original_numerical_cols + EXPLICITLY_MAPPED_COLS + ['JobLevel', 'StockOptionLevel']))
    # Filter out any columns that might not actually exist in the dataframe
    columns_to_scale = [col for col in columns_to_scale if col in df_train_processed.columns]

    # Check for NaNs introduced by mapping/coercion *before* fitting scaler
    if df_train_processed[columns_to_scale].isnull().any().any():
        st.error(f"NaN values found in columns intended for scaling after mapping/conversion. Check base data and mappings. Problem columns: {df_train_processed[columns_to_scale].isnull().sum()[df_train_processed[columns_to_scale].isnull().sum() > 0].index.tolist()}")
        st.stop()

    # --- Fit Preprocessors ---

    # Fit Label Encoders ONLY on nominal categoricals using ORIGINAL training data
    label_encoders = {}
    print("Fitting Label Encoders for columns:", nominal_categorical_cols) # Debug
    for col in nominal_categorical_cols:
        if col in df_train.columns: # Use original df_train here
            le = LabelEncoder()
            # Fit on unique non-null string values
            unique_values = df_train[col].astype(str).dropna().unique()
            if len(unique_values) > 0:
                le.fit(unique_values)
                label_encoders[col] = le
            else:
                st.warning(f"Column '{col}' has no unique values to fit LabelEncoder.")
        else:
             st.warning(f"Nominal categorical column '{col}' defined but not found in training data.")


    # Fit Scaler ONLY on the final set of numerical columns using the PROCESSED training data
    scaler = MinMaxScaler()
    print("Fitting Scaler for columns:", columns_to_scale) # Debug
    if not columns_to_scale:
        st.warning("No numerical columns identified for scaling.")
        scaler = None # Or an identity scaler if needed downstream
    else:
        try:
            scaler.fit(df_train_processed[columns_to_scale])
        except ValueError as e:
            st.error(f"Error fitting scaler. Check data in columns {columns_to_scale}: {e}")
            st.stop()
        except Exception as e:
            st.error(f"Unexpected error fitting scaler: {e}")
            st.stop()

    # For debugging - print the identified lists
    st.session_state['debug_nominal_cols'] = nominal_categorical_cols
    st.session_state['debug_scale_cols'] = columns_to_scale

    # Return fitted encoders, scaler, and the lists of columns they apply to
    return label_encoders, scaler, nominal_categorical_cols, columns_to_scale

# --- Processing and Prediction ---

def preprocess_input_data(input_data: Dict[str, Any], label_encoders: Dict[str, LabelEncoder], scaler: MinMaxScaler, nominal_categorical_cols: List[str], columns_to_scale: List[str]) -> pd.DataFrame:
    """Preprocesses user input using fitted preprocessors."""
    # Start with user input
    df_input = pd.DataFrame([input_data])
    # Create df to hold processed numerical values before scaling
    df_numerical_processed = pd.DataFrame(index=df_input.index)
    # Create df to hold final processed output
    df_final_processed = pd.DataFrame(index=df_input.index)

    # 1. Apply explicit mappings
    for col, mapping in MAP_DICTIONARIES.items():
        if col in df_input.columns:
            # Map the input value
            mapped_value = df_input[col].map(mapping).iloc[0]
            # Add the mapped numerical value to the numerical df
            df_numerical_processed[col] = [mapped_value]
        else:
            st.warning(f"Mapped column '{col}' not in user input dict.")

    # 2. Apply Label Encoders to nominal categoricals
    for col in nominal_categorical_cols:
        if col in df_input.columns:
            le = label_encoders.get(col)
            if le:
                value_to_transform = df_input[col].iloc[0]
                try:
                    if value_to_transform in le.classes_:
                        encoded_value = le.transform([value_to_transform])[0]
                        # Add encoded value directly to the final output df
                        df_final_processed[col] = [encoded_value]
                    else:
                        st.error(f"Error: Category '{value_to_transform}' in column '{col}' was not seen during training.")
                        st.stop()
                except Exception as e:
                    st.error(f"Error encoding value '{value_to_transform}' in column '{col}': {e}")
                    st.stop()
            else:
                 st.warning(f"Label encoder for nominal column '{col}' not found.")
                 # Potentially add raw value if needed? Or error out.
                 # df_final_processed[col] = df_input[col] # Risky
        else:
             st.warning(f"Nominal categorical column '{col}' expected but not found in input.")

    # 3. Add other numerical columns (original num + JobLevel/Stock) to the numerical df
    other_numerical_cols = [
        col for col in columns_to_scale
        if col not in EXPLICITLY_MAPPED_COLS # Avoid adding mapped ones again
    ]
    for col in other_numerical_cols:
         if col in df_input.columns:
             # Ensure numeric type
             df_numerical_processed[col] = pd.to_numeric(df_input[col], errors='coerce')
         else:
             st.warning(f"Numerical column '{col}' expected for scaling not in user input dict.")


    # Check for NaNs in the combined numerical data BEFORE scaling
    if df_numerical_processed[columns_to_scale].isnull().any().any():
        st.error(f"Input data contains non-numeric values or failed mappings in numerical fields before scaling: {df_numerical_processed[columns_to_scale].isnull().sum()}")
        st.stop()

    # 4. Scale the combined numerical columns
    if scaler and columns_to_scale:
        try:
            # Ensure columns are in the correct order for the scaler
            scaled_data = scaler.transform(df_numerical_processed[columns_to_scale])
            df_scaled = pd.DataFrame(scaled_data, columns=columns_to_scale, index=df_numerical_processed.index)
            # Add scaled data to the final processed df
            for col in columns_to_scale:
                df_final_processed[col] = df_scaled[col]
        except ValueError as e:
            st.error(f"Error scaling input. Columns expected by scaler: {scaler.feature_names_in_}. Columns received: {df_numerical_processed[columns_to_scale].columns.tolist()}. Error: {e}")
            st.stop()
        except Exception as e:
             st.error(f"An unexpected error occurred during scaling: {e}")
             st.stop()
    elif columns_to_scale:
         st.warning("Scaler not available. Numerical features were not scaled.")
         # Add unscaled numerical data if scaler missing
         for col in columns_to_scale:
              df_final_processed[col] = df_numerical_processed[col]


    # 5. Ensure Final Column Order matches REQUIRED_COLUMNS
    try:
        df_output = df_final_processed.reindex(columns=REQUIRED_COLUMNS)
        if df_output.isnull().any().any():
             missing_cols = df_output.columns[df_output.isnull().any()].tolist()
             st.error(f"The following columns were missing or null after processing and reindexing: {missing_cols}. Check processing logic and input data.")
             # You might want to show df_final_processed.columns here for debugging
             st.write("Columns before reindexing:", df_final_processed.columns.tolist())
             st.stop()
    except Exception as e:
         st.error(f"Error reordering final columns: {e}")
         st.stop()

    return df_output

# --- make_prediction function remains the same ---
def make_prediction(model, processed_data: pd.DataFrame) -> int:
    """Makes a prediction using the loaded model."""
    try:
        prediction = model.predict(processed_data.to_numpy())
        return int(prediction[0]) # Return single prediction result
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        # Log the shape and columns of data passed to predict
        st.write("Data shape passed to model:", processed_data.shape)
        st.write("Data columns passed to model:", processed_data.columns.tolist())
        st.stop()


# --- build_ui and display_footer remain largely the same ---
# (Ensure UI options match the keys in the MAP dictionaries exactly)
def build_ui() -> Dict[str, Any]:
    """Builds the Streamlit user interface and collects input."""
    # st.set_page_config() MOVED TO MAIN()

    st.title('⚖️🏹 Employee Attrition Prediction')
    st.markdown("Provide employee details below to predict the likelihood of attrition.")

    input_data = {}

    # --- Personal Information ---
    with st.expander("👤 Personal Information", expanded=True):
        col_pers1, col_pers2, col_pers3 = st.columns(3)
        with col_pers1:
            input_data['Gender'] = st.radio('Gender', options=['Male', 'Female'], horizontal=True)
        with col_pers2:
            input_data['Age'] = st.number_input('Age', min_value=18, max_value=60, value=30, step=1)
        with col_pers3:
            input_data['MaritalStatus'] = st.selectbox('Marital Status', ('Single', 'Married', 'Divorced')) # Correct: Nominal

    # --- Education & Background ---
    with st.expander("🎓 Education & Background"):
        col_edu1, col_edu2 = st.columns(2)
        with col_edu1:
            # Correct: Use keys from map for display (Ordinal - Mapped)
            input_data['Education'] = st.selectbox('Education Level', options=list(EDUCATION_MAP.keys()), index=2)
        with col_edu2:
            # Correct: Nominal
            input_data['EducationField'] = st.selectbox('Education Field', ('Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 'Human Resources', 'Other'))

    # --- Work Logistics ---
    with st.expander("🏢 Work Logistics"):
        col_log1, col_log2 = st.columns(2)
        with col_log1:
            # Correct: Numerical
            input_data['DistanceFromHome'] = st.number_input('Distance From Home (km)', min_value=1, max_value=30, value=5, step=1)
        with col_log2:
            # Correct: Nominal
            input_data['BusinessTravel'] = st.selectbox('Business Travel Frequency', ('Non-Travel', 'Travel_Rarely', 'Travel_Frequently'), index=1)

    # --- Job Details ---
    with st.expander("💼 Job Details"):
        col_job1, col_job2, col_job3 = st.columns([2, 3, 1])
        with col_job1:
            # Correct: Nominal
            input_data['Department'] = st.selectbox('Department', ('Research & Development', 'Sales', 'Human Resources'), index=0)
        with col_job2:
             # Correct: Nominal
            input_data['JobRole'] = st.selectbox('Job Role', (
                'Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Manufacturing Director',
                'Healthcare Representative', 'Manager', 'Sales Representative', 'Research Director', 'Human Resources'
            ), index=1)
        with col_job3:
            # Correct: Numerical (Treated as numerical for scaling)
            input_data['JobLevel'] = st.selectbox('Job Level', (1, 2, 3, 4, 5), index=1)

    # --- Compensation ---
    with st.expander("💰 Compensation"):
        # All Correct: Numerical
        col_comp1, col_comp2, col_comp3 = st.columns(3)
        with col_comp1:
            input_data['HourlyRate'] = st.number_input('Hourly Rate ($)', min_value=30, max_value=100, value=65, step=1)
        with col_comp2:
            input_data['DailyRate'] = st.number_input('Daily Rate ($)', min_value=100, max_value=1500, value=800, step=10)
        with col_comp3:
            input_data['MonthlyRate'] = st.number_input('Monthly Rate ($)', min_value=2000, max_value=27000, value=14000, step=100)

        col_comp4, col_comp5 = st.columns(2)
        with col_comp4:
            input_data['MonthlyIncome'] = st.number_input('Monthly Income ($)', min_value=1000, max_value=20000, value=5000, step=100)
        with col_comp5:
            input_data['PercentSalaryHike'] = st.number_input('Percent Salary Hike (%)', min_value=10, max_value=25, value=15, step=1)

        col_comp6, col_comp7 = st.columns(2)
        with col_comp6:
            input_data['StandardHours'] = st.number_input('Standard Hours', value=80, disabled=True)
        with col_comp7:
            # Correct: Nominal ('Yes'/'No')
            input_data['OverTime'] = 'Yes' if st.checkbox('Works Overtime?', value=False) else 'No'

    # --- Satisfaction & Engagement ---
    with st.expander("😊 Satisfaction & Engagement"):
         # All Correct: Use keys from maps (Ordinal - Mapped)
        col_sat1, col_sat2, col_sat3 = st.columns(3)
        with col_sat1:
            input_data['JobSatisfaction'] = st.select_slider('Job Satisfaction', options=list(SATISFACTION_MAP.keys()), value='High')
        with col_sat2:
            input_data['EnvironmentSatisfaction'] = st.select_slider('Environment Satisfaction', options=list(SATISFACTION_MAP.keys()), value='High')
        with col_sat3:
            input_data['RelationshipSatisfaction'] = st.select_slider('Relationship Satisfaction', options=list(SATISFACTION_MAP.keys()), value='High')

        col_eng1, col_eng2, col_eng3 = st.columns(3)
        with col_eng1:
            input_data['JobInvolvement'] = st.select_slider('Job Involvement', options=list(JOB_INVOLVEMENT_MAP.keys()), value='High')
        with col_eng2:
            input_data['PerformanceRating'] = st.select_slider('Performance Rating', options=list(PERFORMANCE_RATING_MAP.keys()), value='Excellent')
        with col_eng3:
            input_data['WorkLifeBalance'] = st.select_slider('Work Life Balance', options=list(WORK_LIFE_BALANCE_MAP.keys()), value='Better')


    # --- Career & Tenure ---
    with st.expander("📈 Career & Tenure"):
        # All Correct: Numerical or treated as numerical for scaling
        col_car1, col_car2, col_car3 = st.columns(3)
        with col_car1:
            input_data['StockOptionLevel'] = st.selectbox('Stock Option Level', (0, 1, 2, 3), index=1)
        with col_car2:
            input_data['NumCompaniesWorked'] = st.number_input('Number of Previous Companies', min_value=0, max_value=10, value=1, step=1)
        with col_car3:
            input_data['TrainingTimesLastYear'] = st.number_input('Training Sessions Last Year', min_value=0, max_value=6, value=2, step=1)

        col_ten1, col_ten2, col_ten3 = st.columns(3)
        with col_ten1:
            input_data['TotalWorkingYears'] = st.number_input('Total Working Years', min_value=0, max_value=40, value=10, step=1)
        with col_ten2:
            input_data['YearsAtCompany'] = st.number_input('Years at Current Company', min_value=0, max_value=40, value=5, step=1)
        with col_ten3:
            input_data['YearsInCurrentRole'] = st.number_input('Years in Current Role', min_value=0, max_value=18, value=3, step=1)

        col_ten4, col_ten5 = st.columns(2)
        with col_ten4:
            input_data['YearsSinceLastPromotion'] = st.number_input('Years Since Last Promotion', min_value=0, max_value=15, value=1, step=1)
        with col_ten5:
            input_data['YearsWithCurrManager'] = st.number_input('Years with Current Manager', min_value=0, max_value=17, value=3, step=1)

    return input_data

def display_footer():
    """Displays the application footer."""
    st.markdown("---")
    year_now = datetime.date.today().year
    copyright_year = "2025" # if year_now <= 2024 else f"2025 - {year_now}"
    st.caption(f"© {copyright_year} | Employee Attrition Prediction | Developed by [Frans Gabriel Sianturi](http://linkedin.com/in/fransgs)")


# --- Main Application Logic ---
def main():
    """Runs the Streamlit application."""
    st.set_page_config(page_title="Employee Attrition Predictor", layout="wide", initial_sidebar_state="collapsed")

    # Load resources (cached)
    try:
        base_df = load_base_data(DATA_FILE)
        model = load_model(MODEL_FILE)
        # Pass the potentially filtered base_df
        label_encoders, scaler, nominal_categorical_cols, columns_to_scale = get_preprocessors(base_df)

        # Store column lists in session state if needed elsewhere, or just use them directly
        st.session_state['nominal_cols'] = nominal_categorical_cols
        st.session_state['scale_cols'] = columns_to_scale

    except Exception as e:
        st.error(f"Failed to initialize resources: {e}")
        st.stop()

    # Display debug info from session state (optional)
    # if 'debug_nominal_cols' in st.session_state and 'debug_scale_cols' in st.session_state:
    #    with st.expander("Debug Info: Preprocessor Column Identification"):
    #        st.write("Nominal Columns for Label Encoding:", st.session_state['debug_nominal_cols'])
    #        st.write("Numerical Columns for Scaling:", st.session_state['debug_scale_cols'])


    user_input = build_ui()

    st.markdown("---")
    predict_button = st.button('✨ Predict Attrition Likelihood', use_container_width=True, type="primary")
    st.markdown("---") # Separator before results

    result_placeholder = st.empty()

    if predict_button:
        if user_input:
            try:
                # Use the correct column lists returned by get_preprocessors
                processed_input_df = preprocess_input_data(
                    user_input,
                    label_encoders,
                    scaler,
                    st.session_state['nominal_cols'], # Use list from session state or pass directly
                    st.session_state['scale_cols'] # Use list from session state or pass directly
                )

                prediction_result = make_prediction(model, processed_input_df)

                with result_placeholder.container():
                    st.subheader("Prediction Result:")
                    if prediction_result == 1:
                        st.error('🚨 Prediction: **Employee is Likely to Leave** (Attrition: Yes)')
                    else:
                        st.success('✅ Prediction: **Employee is Likely to Stay** (Attrition: No)')

            except Exception as e:
                 # Catch errors from preprocessing/prediction if they didn't st.stop()
                 result_placeholder.error(f"An error occurred: {e}")
                 # Optionally add more detailed traceback or logging here
                 # import traceback
                 # result_placeholder.text(traceback.format_exc())

        else:
            result_placeholder.warning("Input data dictionary is empty.")

    display_footer()

if __name__ == '__main__':
    main()