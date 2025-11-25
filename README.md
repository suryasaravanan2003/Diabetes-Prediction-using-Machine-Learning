📝 PROJECT DESCRIPTION

Title: Diabetes Prediction Using Machine Learning (Day 5 Project)

This project aims to build a predictive machine learning model that determines whether a person is at risk of diabetes based on several medical features. The dataset includes parameters such as glucose level, blood pressure, BMI, insulin level, age, and more. Using this data, the system applies machine learning algorithms to classify an individual as Diabetic (1) or Non-Diabetic (0).

The project uses Logistic Regression, a widely used classification algorithm, due to its simplicity and high performance on medical datasets. The data is preprocessed by scaling numerical features using StandardScaler, which helps improve model accuracy. The dataset is then split into training and testing sets to validate the model's performance.

To visualize insights, the project includes a correlation heatmap and confusion matrix, helping understand feature relationships and prediction accuracy. Finally, the system accepts user input in CMD, processes the values, and instantly predicts whether the person is at a high or low risk of diabetes.

This project demonstrates data preprocessing, model training, evaluation, and user-interactive prediction — making it a complete practical machine learning application.

🔍 Technologies & Libraries Used
Purpose	Library
Data loading & cleaning	Pandas
Numerical operations	NumPy
Data visualization	Matplotlib, Seaborn
Scaling features	StandardScaler (sklearn)
Splitting dataset	train_test_split (sklearn)
Machine Learning model	LogisticRegression (sklearn)
Model Evaluation	Accuracy Score, Confusion Matrix, Classification Report
⭐ Key Features

Predicts diabetes risk using medical data

User-friendly CMD input

Data visualization (heatmap + confusion matrix)

Scaled and preprocessed dataset

80/20 train-test split

Easy to run and understand

📈 Output Provided

Dataset preview

Heatmap correlation

Confusion matrix

Accuracy score

Final prediction:

HIGH RISK of Diabetes ❗

LOW RISK of Diabetes ✅
