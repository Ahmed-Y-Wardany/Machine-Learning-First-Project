# Student Score Prediction System

This project implements a machine learning system to predict student scores in Math, Reading, and Writing based on various demographic and educational factors. The system uses Linear Regression models and provides a user-friendly web interface built with Streamlit.

## Features

- Predicts scores for three subjects: Math, Reading, and Writing
- Takes into account 10 different input features about the student
- Provides a clean, intuitive web interface
- Displays both input summary and prediction results

## Input Features

The system uses the following input features to make predictions:

1. **Gender**: Male or Female
2. **Ethnic Group**: Group A, B, C, D, or E
3. **Parent's Education Level**: Various education levels from "some high school" to "master's degree"
4. **Lunch Type**: Standard or free/reduced
5. **Test Preparation**: Completed or none
6. **Parent Marital Status**: Married, single, or divorced
7. **Practice Sport**: Never, sometimes, or regularly
8. **Is First Child**: Yes or no
9. **Transportation Means**: Private or school bus
10. **Weekly Study Hours**: < 5, 5-10, or > 10 hours

## Installation

1. Clone this repository
2. Install the required dependencies:
3. Ensure your trained model files are in the same directory:

- `linearRegressionForMath.pkl`
- `linearRegressionForReading.pkl`
- `linearRegressionForWriting.pkl`

## Usage

1. Run the Streamlit application:
2. Open your web browser and navigate to the local URL provided (typically http://localhost:8501)
3. Fill out the form with the student's information
4. Click the "Predict Scores" button to see the predicted scores

## Model Details

- **Algorithm**: Linear Regression
- **Target Variables**: MathScore, ReadingScore, WritingScore
- **Preprocessing**: Categorical variables are encoded using the same scheme as during training

## Project Structure

student-score-predictor/

- │
- ├── app.py # Main Streamlit application
- ├── requirements.txt # Python dependencies
- ├── linearRegressionForMath.pkl # Trained model for Math
- ├── linearRegressionForReading.pkl # Trained model for Reading
- ├── linearRegressionForWriting.pkl # Trained model for Writing
- └── README.md # Project documentation

## Dependencies

The project requires the following Python libraries:

- streamlit
- pandas
- scikit-learn
- numpy
- matplotlib
- seaborn

See `requirements.txt` for specific versions.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For support or questions about this project, please open an issue in the GitHub repository.
