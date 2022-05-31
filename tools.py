import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

""" r/w/e dataframes """
def opendf(ver:str) -> pd.Series:
	with open(f"dataframes/v{ver}", 'rb') as file:
		return pickle.load(file)

def savedf(df:pd.Series, ver:str) -> None:
	with open(f"dataframes/v{ver}", 'wb') as file:
		pickle.dump(df, file)
	return

def concat_csvs(names:list) -> pd.Series:
	dfs = []
	for name in names:
		tmp = pd.read_csv(f"data/{name}.csv")
		tmp[f"firm"] = name
		dfs.append(tmp)
	df = pd.concat(dfs, axis=0)
	return df

def normalizer(df:pd.Series, with_firm:bool) -> pd.Series:
	# convert price -> hrn, mpg -> kmpl, mileage -> km
	df["price"] = df["price"] * 37.07
	df["mpg"] = df["mpg"] * 0.425144
	df["mileage"] = df["mileage"] * 1.60934
	df = df.rename(columns={"mpg":"kmpl"})

	# throw away tax, model
	df = df.drop(columns=["tax", "model"])

	# create dummies for transmission, fuel type, mufacturer
	if with_firm:
		df = pd.concat([pd.get_dummies(df["firm"]),
						df,
						pd.get_dummies(df["transmission"]),
						pd.get_dummies(df["fuelType"])], axis=1)
	else:
		df = pd.concat([df,
						pd.get_dummies(df["transmission"]),
						pd.get_dummies(df["fuelType"])], axis=1)
	df = df.drop(columns=["firm", "transmission", "fuelType"])

	# shuffle dataframe entries
	df = shuffle(df)
	return df

""" plot utilities """
def heatmapHelper(df:pd.Series) -> None:
	corr = df.corr()
	sns.heatmap(corr,
				xticklabels=corr.columns,
				yticklabels=corr.columns)
	plt.show()
	return