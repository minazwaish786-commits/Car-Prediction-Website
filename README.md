# AutoWorth — Car Price Prediction (HTML, CSS & JavaScript)

A standalone browser-based website based on the supplied **Project 8: Car Price Prediction** notebook and `car_data_2.csv` dataset.

## Files
- `index.html` — website structure and form
- `style.css` — responsive design
- `script.js` — CSV parsing, feature encoding, regression training and prediction
- `car_data_2.csv` — supplied dataset (301 rows, 9 columns)

## Run the website
1. Extract `Car_Price_Prediction_Website.zip`.
2. Open `index.html` in Chrome, Edge or Firefox.
3. Click the CSV upload control and select the included `car_data_2.csv` file.
4. Wait for the “Dataset loaded and model trained” message.
5. Enter car details and click **Predict selling price**.

No Python, Flask, Node.js, installation or server is required. All processing happens locally in the browser; the CSV is not uploaded to a server.

## Notebook alignment
The supplied notebook uses `Selling_Price` as the target and removes `Car_Name` from the input features. It encodes:
- Fuel Type: Petrol=0, Diesel=1, CNG=2
- Seller Type: Dealer=0, Individual=1
- Transmission: Manual=0, Automatic=1

The website uses the same seven input features: `Year`, `Present_Price`, `Kms_Driven`, `Fuel_Type`, `Seller_Type`, `Transmission`, and `Owner`.

## Important model note
The notebook trains both Linear Regression and Lasso Regression after a 90/10 train-test split (`random_state=2`). This browser-only version fits a Linear Regression-style model using all valid rows in the uploaded CSV. It does not reproduce the notebook's train/test evaluation or its separate Lasso model. Predictions are approximate estimates, not guaranteed resale offers.
