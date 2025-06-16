import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import joblib
import io
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.set_page_config(page_title="Student DropOff Predictor", layout="wide")
st.title("📊 Student DropOff & Engagement Prediction App")

uploaded_file = st.file_uploader("Upload your cleaned CSV dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Data Loaded Successfully!")
    st.dataframe(df.head())

    # EDA
    st.subheader("📈 Interactive EDA")
    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.pie(df, names="Status Description", title="Status Distribution", hole=0.4)
        st.plotly_chart(fig1)

        dropoff_gender = df.groupby("Gender")["DropOff"].mean().reset_index()
        fig2 = px.bar(dropoff_gender, x="Gender", y="DropOff", title="DropOff Rate by Gender")
        st.plotly_chart(fig2)

    with col2:
        fig3 = px.histogram(df, x="Age", nbins=10, title="Age Distribution", marginal="box")
        st.plotly_chart(fig3)

        fig4 = px.scatter(df, x="Apply Delay (days)", y="Start Delay (days)", color="DropOff",
                          title="Apply vs Start Delay by DropOff")
        st.plotly_chart(fig4)

    # Model training
    st.subheader("⚙️ Model Configuration + Training")

    # Feature prep
    cat_cols = ['Gender', 'Country', 'Institution Name', 'Current/Intended Major', 'Status Description']
    drop_cols = ['Opportunity Name', 'Opportunity Category', 'Opportunity Id', 'Opportunity End Date',
                 'Learner SignUp DateTime', 'Entry created at', 'Apply Date', 'Opportunity Start Date', 'Date of Birth']
    num_cols = ['Age', 'Apply Delay (days)', 'Start Delay (days)']

    df_encoded = df.copy()
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        encoders[col] = le

    df_encoded = df_encoded.drop(columns=drop_cols)
    scaler = StandardScaler()
    df_encoded[num_cols] = scaler.fit_transform(df_encoded[num_cols])

    X = df_encoded.drop(columns=['DropOff'])
    y = df_encoded['DropOff']

    test_size = st.slider("Test size (%)", 10, 50, 20) / 100
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    model_choice = st.selectbox("Choose model", ["Random Forest", "Logistic Regression"])

    if model_choice == "Random Forest":
        n_estimators = st.slider("Number of trees", 50, 500, 100, 50)
        max_depth = st.slider("Max depth", 2, 20, 5)
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    else:
        c_value = st.number_input("Inverse Regularization Strength (C)", 0.01, 10.0, 1.0)
        model = LogisticRegression(C=c_value, max_iter=1000, random_state=42)

    if st.button("🚀 Train Model"):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.subheader("📊 Evaluation Metrics")
        st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.2f}")
        st.text("Classification Report")
        st.text(classification_report(y_test, y_pred))

        cm = confusion_matrix(y_test, y_pred)
        fig_cm = ff.create_annotated_heatmap(
            z=cm, x=["Pred 0", "Pred 1"], y=["True 0", "True 1"], colorscale="Viridis"
        )
        st.plotly_chart(fig_cm)

        if model_choice == "Random Forest":
            st.subheader("🔑 Feature Importance")
            fi_df = pd.DataFrame({
                "Feature": X.columns,
                "Importance": model.feature_importances_
            }).sort_values(by="Importance", ascending=False)
            fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation="h", title="Feature Importance")
            st.plotly_chart(fig_fi)

        # Save the whole package
        model_package = {
            "model": model,
            "encoders": encoders,
            "scaler": scaler,
            "num_cols": num_cols,
            "cat_cols": cat_cols,
            "drop_cols": drop_cols
        }
        joblib.dump(model_package, "trained_model_with_preproc.pkl")
        st.download_button("💾 Download Trained Model Package", open("trained_model_with_preproc.pkl", "rb").read(),
                           file_name="trained_model_with_preproc.pkl")

# Prediction with saved model
st.subheader("🌟 Predict with Saved Model")
model_upload = st.file_uploader("Upload saved model (.pkl)", type=["pkl"])
if model_upload:
    model_package = joblib.load(io.BytesIO(model_upload.read()))
    st.success("✅ Model loaded! Now upload a new CSV file for prediction.")

    test_file = st.file_uploader("Upload new data CSV for prediction", type=["csv"])
    if test_file:
        test_df = pd.read_csv(test_file)

        try:
            # Drop columns if they exist
            drop_cols_present = [col for col in model_package["drop_cols"] if col in test_df.columns]
            test_df = test_df.drop(columns=drop_cols_present)

            # Check for missing required columns
            required_cols = model_package["cat_cols"] + model_package["num_cols"]
            missing_cols = [col for col in required_cols if col not in test_df.columns]
            if missing_cols:
                st.error(f"⚠️ Your test data is missing required columns: {missing_cols}")
            else:
                # Encode categorical columns
                for col in model_package["cat_cols"]:
                    le = model_package["encoders"][col]
                    # Handle unseen categories by mapping to the first class
                    test_df[col] = test_df[col].apply(
                        lambda x: x if x in le.classes_ else le.classes_[0]
                    )
                    test_df[col] = le.transform(test_df[col].astype(str))

                # Scale numeric columns
                test_df[model_package["num_cols"]] = model_package["scaler"].transform(
                    test_df[model_package["num_cols"]]
                )

                # Predict
                preds = model_package["model"].predict(test_df)
                proba = model_package["model"].predict_proba(test_df)

                # Append predictions + probability of DropOff = 1
                test_df["Predicted DropOff"] = preds
                test_df["DropOff Probability"] = proba[:, 1]

                # Show table
                st.dataframe(test_df)

                # Pie chart of prediction distribution
                pred_dist = test_df["Predicted DropOff"].value_counts().reset_index()
                pred_dist.columns = ["DropOff", "Count"]
                pred_dist["DropOff"] = pred_dist["DropOff"].map({0: "No DropOff", 1: "DropOff"})

                fig_pie = px.pie(pred_dist, names="DropOff", values="Count", title="Predicted DropOff Distribution", hole=0.4)
                st.plotly_chart(fig_pie)

                # Allow download
                st.download_button(
                    "📥 Download Predictions CSV",
                    test_df.to_csv(index=False).encode("utf-8"),
                    file_name="predictions.csv"
                )

        except Exception as e:
            st.error(f"⚠️ Preprocessing failed: {e}. Ensure your test data matches the required format.")
