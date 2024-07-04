## Title : 
Electric Vehicle Population Analysis

## Project Description : 
This project analyzes the adoption and growth of electric vehicles (EVs) over the years using a dataset of electric vehicle populations. The analysis includes visualizations of EV growth by year, market share among brands, and the distribution of EVs across cities. Additionally, it predicts the future growth of the EV market.

## Installation : 
To run this project, you need to have Python and the following libraries installed:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn


## Usage : 
To run this project, follow these steps:

1. **Clone this repository or download the script.**

   - **Clone the repository**: If you have a GitHub repository for this project, you can clone it using the following command:
     ```bash
     git clone <repository-url>
     ```
     Replace `<repository-url>` with the URL of your GitHub repository.

   - **Download the script**: If you don't have a GitHub repository, ensure that the `ev_population_analysis.py` script is downloaded to your local machine.

2. **Ensure you have the required dataset (`Electric_Vehicle_Population_Data.csv`) in the same directory as the script.**

   - Make sure that the dataset file `Electric_Vehicle_Population_Data.csv` is in the same directory where you saved the `ev_population_analysis.py` script. The script reads this file to perform the analysis.

3. **Run the script:**

   Open a terminal or command prompt, navigate to the directory containing the script and the dataset, and run the following command:
   ```bash
   python ev_population_analysis.py

## Project Objectives :
1. Analyze the growth of electric vehicles by year
2. Determine the market share of electric vehicles among different brands
3. Discover the distribution of electric vehicles across various cities
4. Discover the electric range distribution of the electric vehicle
5. Predict the future growth of the electric vehicle market

## Data

### Dataset Overview

The dataset used for this project is `Electric_Vehicle_Population_Data.csv`. It includes various attributes related to electric vehicles, such as their model year, make, city, and electric range. This data helps analyze the growth, distribution, and market share of electric vehicles.

### Data Attributes

- **Model Year**: The year the electric vehicle model was released.
- **Make**: The manufacturer of the electric vehicle.
- **Model**: The specific model of the electric vehicle.
- **Electric Range**: The range of the vehicle on a full charge, measured in miles.
- **City**: The city where the vehicle is registered.

### Obtaining the Dataset

Ensure you have the `Electric_Vehicle_Population_Data.csv` file in the same directory as the script. If you don't have the dataset, you can download it from the [link](https://github.com/ngating/electric_vehicle_data_analysis/blob/5dee4b51b4bb790a4f146259e5f8e8f59bc5822a/dataset/Electric_Vehicle_Population_Data.csv) or check if it's included in the project repository.

#### Example
1. Download the dataset from the [link](https://github.com/ngating/electric_vehicle_data_analysis/blob/5dee4b51b4bb790a4f146259e5f8e8f59bc5822a/dataset/Electric_Vehicle_Population_Data.csv) and save it to your project directory.
2. Verify that the file name is `Electric_Vehicle_Population_Data.csv`.
3. Run the script as described in the "Usage" section to perform the analysis.

## Results

### Summary of Key Findings

- **Growth of Electric Vehicles by Year**: There has been a steady increase in the adoption of electric vehicles over the past decade.
- **Market Share by Brand**: Tesla dominates the electric vehicle market, followed by other major brands like Nissan and Chevrolet.
- **Distribution by City**: Seattle has the highest number of registered electric vehicles among cities.
- **Electric Range Distribution**: Most electric vehicles have an electric range between 100 and 300 miles, with the mean range being approximately 200 miles.
- **Future Predictions**: The trend indicates a continued growth in the adoption of electric vehicles over the next five years.

### Visualizations and Explanations

#### Growth of Electric Vehicles by Year

[EV Growth by Year](ev_data_analysis/images/ev_growth_by_year.png)
*Figure: The number of electric vehicles has been increasing steadily year over year.*

#### Market Share by Brand

[EV Market Share by Brand](ev_data_analysis/images/ev_market_share_by_brand.png)
*Figure: Tesla has the largest market share among electric vehicle brands.*

#### Distribution by City

[EV Distribution by City](ev_data_analysis/images/ev_growth_by_city.png)
*Figure: Seattle has the highest number of registered electric vehicles among cities.*

#### Electric Range Distribution

[EV Range Distribution](ev_data_analysis/images/ev_electric_range_distribution.png)
*Figure: Distribution of electric vehicle ranges, with most vehicles having a range between 100 and 300 miles.*

#### Future Predictions

[EV Future Predictions](ev_data_analysis/images/ev_growth_prediction.png)
*Figure: Predicted growth in the number of electric vehicles over the next five years.*


This README provides a clear overview of your project and its results without the need for a license.

