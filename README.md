
# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Project 5: California Wildfires
#### Carly Sharma, Jake Parker, & Adam Zucker

### Problem Statement
Given California's history of terrible wildfires, we've set out to predict the potential severity of future fires based on past fire data, county-specific weather patterns, and global atmospheric carbon data. We'll use a variety of models, including Logistic Regression, K-Nearest Neighbors, a Random Forest Classifier, a Support Vector Classifier, and a Convolutional Neural Network. With these predictions, we hope to recommend mitigating factors and aid the California Department of Forestry and Fire Protection (Cal Fire) in stopping the spread of wildfires in their earliest stages.


---
### Background Research
As climate change becomes an ever-increasing concern for the health and longevity of the Earth, wildfires, and their severity, have proportionally been on the rise. Arid climates, such as those found in many parts of California, are at particularly high risk - 2019 and 2020 have seen some of the most destructive fires in the state's history ([Cal Fire](https://www.fire.ca.gov/)). Cal Fire has implemented measures in attempt to mitigate the spread of wildfires, but we would like to investigate if there are more specific actions to be taken to help prevent the start, and most certainly the spread of these fires.

---
### Contents
* **Notebook 1:** Data imports, cleaning, and merging into a dataframe for modeling
* **Notebook 2:** A notebook devoted to EDA, feature selection and engineering,  and further data cleaning
* **Notebook 3:** Visualization notebook, containing graphs and additional EDA
* **Notebook 4:** Modeling notebook containing all classification models we ran

---
### Datasets Used

* `true_df.csv`: Our engineered dataset based on the below:
  * California Wildfire Incidents dataset (from [Kaggle](https://www.kaggle.com/ananthu017/california-wildfire-incidents-20132020?select=California_Fire_Incidents.csv))
  * Meteorological data (from [NOAA](https://www.ncei.noaa.gov/metadata/geoportal/rest/metadata/item/gov.noaa.ncdc:C00946/html))
  * Atmospheric carbon level data (from [NOAA/Mauna Loa Observatory](https://www.esrl.noaa.gov/gmd/dv/data/index.php?parameter_name=Carbon%2BDioxide&search=mauna+loa))
* `dummy_df.csv`: A dummified version of our `true_df`

---

### Data Dictionary
|Feature|Type|Dataset|Description|
|---|---|---|---|
|**acres_burned**|*float*|`true_df.csv`|Total acres burned in a given fire|
|**admin_unit**|*object*|`true_df.csv`|Responding fire unit|
|**avg_monthly_temp**|*float*|`true_df.csv`|Average monthly temperature in degrees Fahrenheit|
|**avg_wind_speed**|*float*|`true_df.csv`|Average monthly wind speed in MPH|
|**co2_measured_mole_fraction**|*float*|`true_df.csv`|Measured mole fraction of carbon dioxide in atmosphere, collected from Mauna Loa Observatory, HI|
|**cooling_degree_days**|*float*|`true_df.csv`|Cooling degree days, computed as the mean daily temperature minus 65 degrees Fahrenheit. Each day is summed to produce a monthly total.|
|**county**|*object*|`true_df.csv`|County where a given fire started|
|**date**|*object*|`true_df.csv`|The date corresponding to the month during which a given fire started|
|**dp10**|*float*|`true_df.csv`|Number of days in a month with at least 0.01 inches of rainfall|
|**dt00**|*float*|`true_df.csv`|Number of days in a month with a minimum temperature less than or equal to 0 degrees Fahrenheit|
|**dt32**|*float*|`true_df.csv`|Number of days in a month with a minimum temperature less than or equal to 32 degrees Fahrenheit|
|**dx32**|*float*|`true_df.csv`|Number of days in a month with a maximum temperature less than or equal to 32 degrees Fahrenheit|
|**dx70**|*float*|`true_df.csv`|Number of days in a month with a maximum temperature less than or equal to 70 degrees Fahrenheit|
|**dsnd**|*float*|`true_df.csv`|Number of days in a month with snowfall greater than an inch|
|**duration**|*float*|`true_df.csv`|The duration of a given fire in days|
|**elevation**|*float*|`true_df.csv`|The elevation, given in feet, at which the fire started|
|**extinguished**|*object*|`true_df.csv`|Date the fire was extinguished|
|**extreme_max_temp**|*float*|`true_df.csv`|Highest daily temperature from the month|
|**extreme_min_temp**|*float*|`true_df.csv`|Lowest minimum temperature from the month|
|**fire_bins**|*integer*|`true_df.csv`|Bins representing acres burned in a given fire|
|**fire_name**|*object*|`true_df.csv`|The name of a given fire|
|**highest_daily_snowfall**|*float*|`true_df.csv`|Highest snowfall on a given day in a month|
|**latitude**|*float*|`true_df.csv`|Latitude, given in decimal degrees|
|**longitude**|*float*|`true_df.csv`|Longitude, given in decimal degrees|
|**major_incident**|*boolean*|`true_df.csv`|Whether or not the fire was classified as a major incident|
|**month**|*integer*|`true_df.csv`|The month a given fire started|
|**qc_flag**|*boolean*|`true_df.csv`|Whether or not an atmospheric carbon dioxide measurement was considered viable|
|**season**|*object*|`true_df.csv`|Season of the year|
|**started**|*object*|`true_df.csv`|Date the fire started|
|**temp_range**|*float*|`true_df.csv`|The difference between the maximum and minimum temperature in a given month|
|**total_monthly_precipitation**|*float*|`true_df.csv`|Total precipitation in a given month, given in inches|
|**total_monthly_snowfall**|*float*|`true_df.csv`|Total snowfall in a month, given in inches|

---
### Analysis Summary
* We began with 1 dataset on California Fire Incidents from Kaggle, 12 California-specific meteorological datasets from NOAA spanning from 2017 to 2019, and 1 dataset from the Mauna Loa Observatory in Hawaii measuring approximate global atmospheric carbon dioxide levels. In Notebook 1, we began by merging all 12 NOAA dataset, then cleaning the resulting dataset based on null values, odd or unexpected entries, and general formatting. We then cleaned our California Fire Incidents and Mauna Loa carbon dioxide datasets in preparation for merging all our data into a single dataframe. In this process, we imputed values for nulls based on monthly averages, and converted longitude and latitude data to county names, to enable proper merging of our data.
* In Notebook 2, we went on to explore the data, select our desired features, account for multicollinearity, and engineer potentially useful features (such as season and month of the fire). We plotted various Seaborn heatmaps, and generated a number of correlation matrices to examine features we wanted to include in our final models.
* We visualized our data in Notebook 3, looking at graphs to compare acres burned by county, month, season, and year; total acres burned in California between 2017 and 2019; duration of fire by county; and average duration of fire based on severity.
* We ultimately tested 5 classification models: Logistic Regression, K-Nearest Neighbors, a Random Forest Classifier, a Support Vector Classifier, and a Convolutional Neural Network. Our target feature was the total acres burned, classified into bins we defined as:
  * **Bin 1:** 50 acres or below
  * **Bin 2:** 100 acres or below, and greater than 50 acres
  * **Bin 3:** 250 acres or below, and greater than 100 acres
  * **Bin 4:** 500 acres or below, and greater than 250 acres
  * **Bin 5:** 1000 acres or below, and greater than 500 acres
  * **Bin 6:** Greater than 1000 acres


---

### Conclusions & Recommendations
* From our EDA, we found that counties in higher elevations tend to have more and larger fires. Expectedly, precipitation levels and maximum temperatures also contribute heavily to the likelihood of wildfires - low rainfall and snowfall equates to a higher chance of a large fire, as does an extreme monthly temperature of above 90 degrees Fahrenheit.
* Wildfires are most prevalent in the Summer and Fall seasons. Specifically in July, we noticed an especially high occurrence of small and large wildfires.
* The most frequently occurring fires are small in terms of acres burned, but many small fires can add up to a large amount of damage.
* We would recommend controlled burn-offs of dry foliage leading up to fire season (Summer into Fall), especially in arid and high-elevation areas.
* We would also recommend reallocation of resources and firefighters based on environmental and weather conditions, such that more units can respond as quickly as possible to a fire threat, thus stopping the spread of the fire.



#### Further Research
* Given the time, we'd like to look at wind speeds at the time the fire started, and not just the monthly average, as this could inform how quickly a fire spreads.
* We'd also like to compare rural versus urban areas, in order to try to get a feel for where and how the most severe fires start and spread.
* Looking at a longer history of fires in California, and globally, would be an interesting and informative metric if we had access to longer-term data.
* We could research and gather more data on the concrete effects of climate change on already fire-prone environments.


---

### Sources Cited:
* [Kaggle](https://www.kaggle.com/ananthu017/california-wildfire-incidents-20132020?select=California_Fire_Incidents.csv) California wildfire data
* [NOAA](https://www.ncei.noaa.gov/metadata/geoportal/rest/metadata/item/gov.noaa.ncdc:C00946/html) meteorological data
* [NOAA/Mauna Loa Observatory](https://www.esrl.noaa.gov/gmd/dv/data/index.php?parameter_name=Carbon%2BDioxide&search=mauna+loa) atmospheric carbon data
* [Cal Fire](https://www.fire.ca.gov/)
