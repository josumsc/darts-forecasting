import numpy as np
import matplotlib.pyplot as plt
from darts.utils.statistics import check_seasonality


def plot_time_series(series_dict, figsize=(16, 8)):
    """Plots the data inside the dict
    
    :param series_dict: Dictionary(label: TimeSeries) with the information
    :param figsize: Size of the Figure
    :return None:
    """
    fig, ax = plt.subplots(figsize=(16, 8))
    for label, timeseries in series_dict.items():
        timeseries.plot(label=label)
    plt.show()

    return None


def is_data_seasonal(series, max_period=12):
    """Checks if the data set is seasonal on the periods observed
    
    :param series: Series of data to evaluate
    :param max_period: Max number of seasonal degrees to check
    :return: None
    """
    for i in range(2, max_period+1):
        is_seasonal, s = check_seasonality(series, m=i)
        if is_seasonal:
            print(f"Seasonality of degree {s} found!")
            
    return None


def get_best_model(models, series, metric, stride=6, num_periods=12):
    """Retrieves the best model according to the 

    :param models: List of models to evaluate
    :param series: Series of data to evaluate with
    :param metric: Metric to measure performance
    :param stride: Number of steps to jump between forecasts
    :param num_periods: Number of periods to forecast
    :return: Darts model with best scoring
    """
    best_model = None
    initial_error = np.inf

    for model in models:
        backtest_errors = model.backtest(
            series, start=0.5, stride=6, forecast_horizon=num_periods, metric=metric
        )
        model_error = np.mean(backtest_errors)
        print(model, ':', model_error)

        if model_error < initial_error:
            best_model = model
            
    return best_model
