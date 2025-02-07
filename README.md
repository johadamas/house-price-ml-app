# End-to-End House Price Prediction ML App with Streamlit

## Table of Contents
1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Project Flow](#project-flow)
4. [Limitations](#limitations)
5. [Conclusion](#conclusion)
6. [Future Enhancements](#future-enhancements)

## Introduction
1. **Problem Statement**

    The real estate market in Jakarta is dynamic and complex, making it challenging for buyers and sellers to determine accurate property prices. Location, property size, and amenities significantly influence prices, but their relationships are often non-linear and challenging to quantify without advanced tools

2. **Project Objectives**

    This project aims to develop a **Streamlit-based Machine Learning (ML) application** to predict housing prices in Jakarta using data from 99.co. The architecture and key components are as follows:

    ![](/images/project_architecture.png "")

    - `Data Collection`: Instant Data Scraper to extract housing data from 99.co.

    - `Model Development`: Process data and train an ML model using Jupyter Notebook.
    
    - `Interactive App`: Deploy the trained model in a user-friendly Streamlit app for real-time price predictions.

    By leveraging machine learning, this app provides buyers and sellers with accurate price estimates and market insights, enabling informed decision-making

## Prerequisites
1. **Environment Setup**:
    - Python 3.8 or above
    - Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn, xgboost, Streamlit.

2. **Required Tools**:
    - Instant Data Scraper (browser extension)
    - Jupyter Notebook (optional) for running the `.ipynb` notebook

## Project Flow:
1. **Data Collection**
    - The property listing data is scraped from **99.co** using the **Instant Data Scraper** browser extension

        ![](/images/scraping.gif "")

    - The data includes property listings with **key features** such as:
        - Number of bedrooms and bathrooms
        - Land and floor area
        - Location (district and city)
        - Price

2. **Data Preparation**
    - Transformations:
        - Drop unnecessary columns

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
        - Renamed columns
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

        - Combine DataFrames
            ```python
            # List all the file
            house_data = [part_0, part_1, part_2]

            # Merge all the DataFrames into one
            merged_data = pd.concat(house_data, axis=0, ignore_index=True)
            ```

        - Split location column
            ```python
            # Split the 'location' column into 'district' and 'city'
            df[['district', 'city']] = df['location'].str.split(', ', expand=True)

            df.drop(columns=['location'], axis = 1, inplace = True)
            ```
        - Handle `land` and `floor` columns
            ```python
            # Convert to string, remove non-numeric characters, handle NaN, and convert to int
            df['land'] = df['land'].astype(str).str.replace(r'\D', '', regex=True)
            df['land'] = pd.to_numeric(df['land'], errors='coerce').fillna(0).astype(int)

            df['floor'] = df['floor'].astype(str).str.replace(r'\D', '', regex=True)
            df['floor'] = pd.to_numeric(df['floor'], errors='coerce').fillna(0).astype(int)
            ```

        - Convert currency data types
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

            # Apply conversion function
            df['price'] = df['price'].apply(convert_currency)
            ```

    - Data Cleaning:
        - Handle missing values

3. **Data Analysis**
    - Univariate Analysis:
        - Visualized categorical features (`district` and `city`) using bar charts

            ![](/images/barchart.png "Count of City, ordered by the largest count")

            ![](/images/barchart2.png "Count of District, ordered by the largest count")

            *`Insight`: **Jakarta Utara** has the highest number of listings, especially in **Kelapa Gading***

        - Summarized numerical features (`price`, `bedrooms`, `bathrooms`) to analyze their range and distribution

    - Bivariate Analysis
        - Explored feature relationships using scatterplots:

            ![](/images/scatterplot1.png "Price vs Bedrooms")

            ![](/images/scatterplot2.png "Price vs Bathrooms")

            ![](/images/scatterplot3.png "Price vs Floors")

            ![](/images/scatterplot4.png "Price vs Land")

           *`Insight`: Scatterplots reveal extreme outliers, later handled in preprocessing*

        - Conducted correlation analysis:

            ![](/images/heatmap1.png "Correlation Heatmap between Price, Floor, Land")         

            ![](/images/heatmap2.png "Correlation Heatmap between Price, Bathrooms, Bedrooms")

           *`Insight`: Land size and floors have a weak positive correlation with price, while bedrooms and bathrooms have minimal impact*            

4. **Outlier Handling**
    - Identified outliers from Data Analysis and removed them using Z-scores and IQR:

        - **Z-Scores**: We calculated the Z-scores for the price column to identify outliers. Data points with absolute Z-scores greater than 3 were considered outliers and removed

            ```python
            from scipy import stats
            import numpy as np

            df = df[(np.abs(stats.zscore(df['price'])) < 3)]
            ```

        - **Interquartile Range (IQR)**: We calculated the first quartile (Q1) and the third quartile (Q3) for the price column and determined the IQR. Outliers were identified as data points falling below Q1 - 1.5 * IQR or above Q3 + 1.5 * IQR

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

            # Optional: To confirm the outliers are removed
            df_filtered.shape  # This shows the new size of the DataFrame
            ```

    - **Impact of Outlier Removal**:
        - Correlation between price and land size **increased to 0.737**

        - Correlation between price and floors **increased to 0.722**

            ![](/images/scatterplot5.png "")

            ![](/images/heatmap3.png "")

            *`Insight`: Larger land sizes and more floors strongly correlate with higher property prices*

5. **Model Building**
    - One-hot encoded categorical features (`district`, `city`) and numerical features (`Bedrooms`, `Bathrooms`, `Floor`, `Land`)

    - Trained a **Gradient Boosting Regressor** (outperforming Linear Regression)

    - Used 80% training / 20% test split

    - Evaluated using **R²** score and **Mean Squared Error (MSE)**

6. **Model Performance**
    - Achieved an **R² score of 74.33%**, indicating strong predictive performance

7. **ML App Building**
    - Integrated the trained model into a Streamlit app for an interactive user experience

    - Users input property details (e.g., bedrooms, bathrooms, land area) and receive instant price predictions

        ![](/images/ml_app.gif "")

---

## Limitations:
1. **Dataset Bias:**
    - The dataset may contain biases from its source, affecting generalizability

2. **Outlier Handling:**
    - Removing outliers may have excluded legitimate but unusual data points, impacting accuracy

3. **Missing Features:**
    - Some important factors, like furnishing status, were initially available but had a high percentage of missing values. To maintain data quality and model reliability, they were excluded, which may limit accuracy

4. **Location-Specific Impact:**
    - The model considers city and district but may not fully capture premium pricing in prestigious areas like Cilandak or Menteng.

5. **Model Accuracy in Real Estate Pricing:**
    - The R² score of 74.33% is strong but may be affected by market trends, economic conditions, and buyer sentiment, which are not included

## Conclusion
This project demonstrates the implementation of an end-to-end machine learning application by leveraging web scraping, data cleaning, and machine learning techniques for predicting housing prices in Jakarta.

## Future Enhancements
1. **Incorporate Additional Features**
    - Include features like property furnishing status (fully furnished, semi-furnished, etc.) to improve model accuracy

2. **Expand to Other Cities**
    - Extend the model to predict housing prices in other major cities in Indonesia

3. **Improve Outlier Handling**
    - Develop more robust methods for outlier detection and handling

4. **Enhance Model Performance**
    - Compare performance with advanced models like XGBoost to improve prediction accuracy

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.