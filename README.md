# End-to-End House Price Prediction ML App with Streamlit

## Introduction
This project aims to create a Machine Learning App that can predict housing prices in Indonesia, specifically in Jakarta. The data was collected from the 99.co website, which provides excellent information such as the number of bedrooms and bathrooms, as well as the land and floor area of the listed properties. Streamlit is used to develop the final ML App, providing an interactive and user-friendly interface.

## Architecture

![Pipeline Flow](/images/project_architecture.png "Project Architecture")

The goal of this project is to leverage data-driven techniques to help users estimate the value of properties based on key features. By using machine learning models, we can provide accurate predictions and valuable insights into the real estate market.

## Project Flow

### A. Data Scraping
1. Install the Instant Data Scraper Extension for Chrome or any browser
2. Open the targeted web for data scraping
3. Launch the Scraper Extension and locate the next button of the website
4. Once the 'Next' button is located, the Scraping can be start by clicking the 'Start' button
5. Make sure to set the minimum delay above 2 seconds in order to prevent the Scraper to crash

    ![Data Scraping](/images/scraping.gif "Scraping Proccess")

### B. Data Preparation
1. Load All Raw Data
    - The initial data is loaded from three separate CSV files into DataFrames

        ```python
        - part_0 = pd.read_csv('/content/99.csv')
        - part_1 = pd.read_csv('/content/99 (1).csv')
        - part_2 = pd.read_csv('/content/99 (2).csv')
        ```

2. Drop Unnecessary Columns
    - Columns that are not needed for the analysis are removed from each DataFrame to simplify the dataset

        ```python
        # Drop unnecessary columns
        part_1.drop(
            columns=['cardSecondary__media src',
                    'cardSecondary__info-wrapper_detail-basic-content href',
                    'cardSecondary__info-agent href',
                    'cardSecondary__media-count',
                    'agent-detail_name',
                    'agent-detail_name 2',
                    'text',
                    'text 2',
                    'Svg href 5',
                    'avatar src',
                    'price'
                    ], axis = 1, inplace = True)
        ```

3. Rename Columns
    - The columns are renamed to more meaningful names for ease of understanding and usability

        ```python
        # Rename columns
        part_1.rename(
            columns={'updated-date 2': 'date',
                    'price__tag': 'price',
                    'cardSecondary__info-wrapper_detail-basic-content': 'description',
                    'cardSecondary__info-wrapper_detail-basic-content 2': 'location',
                    'left': 'bedrooms',
                    'left 2': 'bathrooms',
                    'right': 'land',
                    'right 3': 'floor'}, inplace=True)
        ```

4. Extract the Last Two Segments from Location
    - The location column is transformed to retain only the last two segments, providing consistent and relevant location information

        ```python
        # Extract the last two segments from location
        part_1['location'] = part_1['location'].apply(lambda x: ', '.join(x.split(', ')[-2:]))
        ```

5. Combine DataFrames
    - All the cleaned DataFrames are combined into a single list

        ```python
        # List all the file
        house_data = [part_0, part_1, part_2]
        ```

6. Merge All DataFrames into One
    - The individual DataFrames in the list are concatenated into a single DataFrame

        ```python
        # Merge all the DataFrames into one
        merged_data = pd.concat(house_data, axis=0, ignore_index=True)
        ```

7. Save the Merged Data into One CSV File
    - The merged DataFrame is saved into a CSV file for further analysis

        ```python
        merged_data.to_csv('house_data.csv', header = True)
        ```

### C. Data Processing
1. Data Transformation
    - Load the Dataset:
        - The dataset is loaded from a CSV file, and unnecessary columns such as Unnamed: 0 are dropped

    - Split Columns:
        - The location column is split into district and city to make these features usable for analysis

        ```python
        # Split the 'location' column into 'district' and 'city'
        df[['district', 'city']] = df['location'].str.split(', ', expand=True)

        df.drop(columns=['location'], axis = 1, inplace = True)
        ```

    - Handle Missing Values:
        - Null values in critical columns such as city are dropped to ensure data integrity

        ```python
        # Convert to string, remove non-numeric characters, handle NaN, and convert to int
        df['land'] = df['land'].astype(str).str.replace(r'\D', '', regex=True)
        df['land'] = pd.to_numeric(df['land'], errors='coerce').fillna(0).astype(int)

        df['floor'] = df['floor'].astype(str).str.replace(r'\D', '', regex=True)
        df['floor'] = pd.to_numeric(df['floor'], errors='coerce').fillna(0).astype(int)

        # Drop any null values in the 'city' column
        df.dropna(subset = 'city', inplace = True)
        ```

    - Convert Data Types:
        - Columns like land are cleaned to remove non-numeric characters and converted into numerical types

        ```python
        # Function to convert the currency values
        def convert_currency(value):
            # Remove 'Rp' prefix
            value = value.replace('Rp', '').strip()

            # Define magnitude multipliers
            multipliers = {
                'Juta': 1e6,
                'Miliar': 1e9
            }

            # Extract the number and magnitude
            match = re.match(r'([\d,.]+)\s*(Juta|Miliar)?', value)
            if match:
                num_str, magnitude = match.groups()
                # Replace comma with dot for decimal points
                num_str = num_str.replace(',', '.')
                number = float(num_str)
                if magnitude in multipliers:
                    number *= multipliers[magnitude]
                return number
            else:
                raise ValueError(f"Unrecognized format: {value}")
        ```

    - Currency Conversion:
        - A function is applied to convert currency values into a consistent numerical format for analysis

        ```python
        # Apply conversion function
        df['price'] = df['price'].apply(convert_currency)
        ```

2. Data Analysis
    - Univariate Analysis:
        - Visualized categorical features like district and city using bar plots to identify the distribution of house listings

            ![](/images/barchart.png "")

            ![](/images/barchart2.png "")

            *Note: The Barchart above shows that 'Jakarta Utara' has the most house listing according to the website, specifically in the 'Kelapa Gading' district*

        - Summarized numerical features like price, bedrooms, and bathrooms to understand the range and central tendencies

    - Bivariate Analysis
        - Scatterplots:
            - Explored relationships between features like bedrooms vs. price and land vs. price

            ![](/images/scatterplot1.png "")

            ![](/images/scatterplot2.png "")

            ![](/images/scatterplot3.png "")

            ![](/images/scatterplot4.png "")

           *Note: The scatterplot above shows an extremely high and extremely low data point relative to the nearest data point and the rest of the neighboring co-existing values in many of the numerical columns*

        - Correlation Analysis:
            - Computed correlation coefficients to identify relationships among features

            ![](/images/heatmap1.png "")

           *Note: The heatmap above indicates a weak positive relationship between the price and land size and also floors and price. Larger land sizes or floors are associated with higher prices, but this factor alone does not strongly influence the price*           

            ![](/images/heatmap2.png "")

           *Note: The heatmap above indicates a very weak positive relationship between the price and number of bedrooms and bathrooms and. This means that the number of bedrooms and bathrooms has a minimal impact on the price*            

3. Outlier Handling
    - Floor and Price Outliers:
        - Removed extreme values in floor and price that were identified as statistical outliers

        ```python
        # Drop floor > 1500, since the gap is getting wider gradually
        df = df[df['floor'] <= 1500]

        # Drop outlier in the 'price' columns
        df = df[df['price'] <= 50000000000]
        ```

    - Land Outliers:
        - Applied similar methods to handle extreme values in the land column

        ```python
        # Drop 'land' above 1500
        df = df[df['land'] <= 1500]
        ```

    - Statistical Techniques:
        - Used Z-scores and interquartile ranges (IQR) to detect and handle outliers

        ```python
        # Calculate Q1, Q3, and IQR
        Q1 = df['price'].quantile(0.25)
        Q3 = df['price'].quantile(0.75)
        IQR = Q3 - Q1

        # Define lower and upper bounds for non-outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Filter the DataFrame to remove outliers
        df_filtered = df[(df['price'] >= lower_bound) & (df['price'] <= upper_bound)]
        ```
        *Note: Please check the notebook for detailed Outlier Handling*        

    - Handling Result:
        - After handling some of the outlier, we can see an increase in correlation between the price and the floor as well as land

            ![](/images/scatterplot5.png "")

            ![](/images/heatmap3.png "")

            *Note: A correlation of 0.737 indicates a strong positive relationship. This means that as the size of the land increases, the price tends to increase significantly. Also, a correlation of 0.722 indicates a strong positive relationship. This suggests that properties with more floors tend to have higher prices* 

### D. Model Building
1. Feature Preprocessing
    - Define the target variable
        - Define the feature variables (X) and the target variable (y)

        ```python
        X = df.drop(columns=['price'])
        y = df['price']
        ```
    - Numerical and Categorical features:
        - Identify numerical and categorical features available in the dataset

        ```python
        numeric_features = ['bedrooms', 'bathrooms', 'floor', 'land']
        categorical_features = ['district', 'city']
        ```
    - Imputation: 
        - Simple Imputer was utilized to handle missing values in both numerical and categorical data by imputing with the mean for numerical features and the most frequent value for categorical features

    - Scaling: 
        - Standard Scaler was employed for standardizing numerical features (`bedrooms`, `bathrooms`, `floor`, `land`) by removing the mean and scaling to unit variance, ensuring that the model treats each feature equally

    - One-Hot Encoding: 
        - Applied to convert categorical variables (`district`, `city`) into a binary format, allowing them to be included in the model

        ```python
        # Define the preprocessing steps for numerical features
        numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())])

        # Define the preprocessing steps for categorical features
        categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),('onehot', OneHotEncoder(handle_unknown='ignore'))])

        # Combine preprocessing steps
        preprocessor = ColumnTransformer(
        transformers=[('num', numeric_transformer, numeric_features), ('cat', categorical_transformer, categorical_features)])
        ```
2. Model Selection
    - Gradient Boosting Regressor
        - The Gradient Boosting Regressor is used for this project due to its superior performance, achieving a higher R² score compared to the Linear Regression model

        ```python
        # Create the pipeline with preprocessing and the model
        model = Pipeline(steps=[('preprocessor', preprocessor),
                                ('regressor', GradientBoostingRegressor(random_state=42))])
        ```
3. Model Training and Evaluation
    - Data Split
        - The processed dataset is split into training and testing sets

        ```python
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        ```
    - Training
        - The Gradient Boosting Regressor is trained on the training set, and its performance is evaluated using metrics such as R² and Mean Squared Error (MSE)

        ```python
        # Train the model
        model.fit(X_train, y_train)
        ```
    - Result
        - The Gradient Boosting Regressor was able to achieve the R2 score of 74.326 which is acceptable for this project

        ```python
        y_pred = model.predict(X_test)

        # Evaluate the model
        r2 = r2_score(y_test, y_pred)
        print(f'R2 score: {r2*100}')

        R2 score: 74.32682219412816
        ```
### E. Machine Learning App
1. Saving the ML Model
    - The model then saved as a pickle.dump(), to makes it easy to integrate with applications such Streamlit

        ```python
        pk.dump(model, open('house_model.pkl', 'wb'))
        ```

2. Stremlit Integration
    - Streamlit is later used to develop the final ML App, providing an interactive and user-friendly interface

        ![](/images/ml_app.gif "")

## Notes
1. **Assumptions**:
    - The scraped data is assumed to represent accurate and up-to-date real estate listings

    - All null or missing values in critical fields like city, price, or land have been handled to ensure data integrity

2. **Limitations**:
    - Dataset Bias:
        - The dataset may contain inherent biases from the original source website, potentially affecting the generalizability of insights

    - Outlier Handling:
        - Outlier removal and imputation methods could have removed some legitimate but unusual data points, impacting the accuracy of the model

    - Missing Features:
        - Key factors such as whether the property is fully furnished, semi-furnished, or non-furnished are not included in the dataset due to data availability. These variables can significantly influence house prices but are not accounted for in the current model
        
    - Location-Specific Impact:
        - Variables like city and district have been used in the model, but the analysis may not fully capture their heavy influence on prices in prestigious areas such as Cilandak or Menteng. These areas may demand premium prices due to reputation, amenities, or exclusivity, which the model does not explicitly consider

3. **Model Performance**:
    - The Gradient Boosting Regressor achieved a higher R² score, which validates its selection over simpler models like Linear Regression

## Prerequisites
1. **Environment Setup**:
    - Python 3.8 or above

    - Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn, xgboost

2. **Required Tools**:
    - Instant Data Scraper: A browser extension for web scraping

    - Jupyter Notebook (optional): For running the .ipynb notebook

## Conclusion
This project demonstrates the implementation of an end-to-end machine learning application for predicting housing prices in Jakarta, Indonesia. By leveraging web scraping, data cleaning, and machine learning techniques, we built a model to provide price predictions. The final application, developed with Streamlit, offers an interactive interface for users to input property features and obtain price estimates
