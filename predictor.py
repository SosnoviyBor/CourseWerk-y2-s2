from logging import exception
import pickle
import pandas as pd
import openpyxl as excel

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

def __create_dummies (data:dict, cell:str, dummies:list) -> dict:
	if cell in data.keys():
		flag = False
		for dummie in dummies:
			if data[cell].lower() == f"{dummie}".lower():
				data.update({f"{dummie}":1})
				flag = True
			else: 
				data.update({f"{dummie}":0})
		if not flag:
			exception(f"This {dummie} is not supported currently. Or you misspelled it")
		data.pop(cell)
		return data
	else:
		exception(f"Bro, you forgor {cell}")

def predict(data:dict|pd.Series, model_name:str, to_excel:bool=False) -> list|pd.Series:
	"""
	If data is dictionary, it should look like this (any order):\n
	data = {
		"firm": str,
		"year": int,
		"price": int,
		"mileage": int,
		"fuelType": str,
		"transmission": str,
		"kmpl": int|float,
		"engineSize": int|float
	}
	"""
	# generate init dataframe if data is dictionary
	if type(data) == type({}):
		# create all dummies
		data = __create_dummies(data, "firm", [
			"audi", "bmw", "ford", "hyundai", "mercedes",
			"skoda", "toyota", "vauxhall", "volkswagen"
		])
		data = __create_dummies(data, "transmission", ['Manual', 'Automatic', 'Semi-Auto', 'Other'])
		data = __create_dummies(data, "fuelType", ['Petrol', 'Diesel', 'Hybrid', 'Other', 'Electric'])
		# insert all values in lists so that data could be transformed to dataframe
		for key in data.keys():
			data[key] = [data[key]]
		# moment that we all were waiting for!
		df = pd.DataFrame.from_dict(data)
		# rearrange dataframe columns order
		df = df[["audi","bmw","ford","hyundai","mercedes",			# firms 1
				 "skoda","toyota","vauxhall","volkswagen",			# firms 2
				 "year","mileage","kmpl","engineSize",				# default data
				 "Automatic","Manual","Other","Semi-Auto",			# transmission
				 "Diesel","Electric","Hybrid","Other","Petrol"]]	# fuel type
	else:
		df = data.drop("price", axis=1)

	# get needed or predetermined model
	with open(f"models/{model_name}", "rb") as file:
		model = pickle.load(file)
		# do magic bullshittery
		rawResult = model.predict(df)
		# format result to same format
		result = []
		for cost in rawResult:
			if type(cost) == type("<class 'numpy.ndarray'>"):
				cost = cost[0]
			if cost < 0:
				cost = abs(cost)
			cost = int(cost)
			result.append(cost)

		if not to_excel:
			return result
		else:
			# write result to excel file
			wb = excel.load_workbook("test results.xlsx")
			ws = wb.create_sheet(model_name)
			ws["A1"] = model_name
			ws["A2"] = "actual"
			ws["B2"] = "predicted"
			ws["C2"] = "diffference"
			ws["D2"] = "diffference%"

			for i in range(len(result)):
				actual = int(data.iloc[i]["price"])
				predicted = int(result[i])
				
				if predicted < 10000000:
					ws[f"A{i+3}"] = actual
					ws[f"B{i+3}"] = predicted
					ws[f"C{i+3}"] = f"=ABS(A{i+3}-B{i+3})"
					ws[f"D{i+3}"] = f"=(C{i+3}/A{i+3}*100)"
				else: ws[f"A{i+3}"] = "DELETE THIS ROW"
			
			ws["F2"] = "Avg diff%"
			ws["F3"] = f"=AVERAGE(D3:D{3+len(result)})"
			ws["G2"] = "Max diff%"
			ws["G3"] = f"=MAX(D3:D{3+len(result)})"
			ws["H2"] = "Min diff%"
			ws["H3"] = f"=MIN(D3:D{3+len(result)})"
			ws["I2"] = "Avg diff"
			ws["I3"] = f"=AVERAGE(C3:C{3+len(result)})"
			
			wb.save(filename="test results.xlsx")
			print(f"Model {model_name} finished calculating!")
			return

def train(df:pd.Series, regression:str, length:int, degree:str|int=""):
	df = df.copy(deep=True)
	trainingData = df.drop("price", axis=1)
	
	match regression:
		case "multi":
			pipa = LinearRegression.fit(LinearRegression(),X=trainingData[:length], y=df["price"][:length])
			
			filepath = f"models/{length} {regression}"
			if "bmw" not in trainingData.columns:
				filepath += " nof"
			with open(filepath, "wb") as file:
				pickle.dump(pipa, file)
		case "poly":
			blueprint = [('scale',StandardScaler()),('polynomial',PolynomialFeatures(degree=degree)),('mode',LinearRegression())]
			pipa = Pipeline(blueprint)
			pipa.fit(trainingData[:length], df[["price"]][:length])
			
			filepath = f"models/{length} {regression} {degree}"
			if "bmw" not in trainingData.columns:
				filepath += " nof"
			with open(filepath, "wb") as file:
				pickle.dump(pipa, file)
		case _:
			exception("Expected regression to be 'multi' or 'poly'")
	
	is_nof = "" if "bmw" in df.columns else " nof"
	print(f"Model '{length} {regression} {degree}{is_nof}' successfully trained!")
	return

def train_and_predict (instructions:list, regression:str, do_tests:bool, to_excel:bool=False):
	"""
	instructions = [
		[dataframe, length, degree (if regression == "poly")],\n
		...
	]
	"""
	match regression:
		case "multi":
			for args in instructions:
				if not "bmw" in args[0].columns:
					filepath = f"{args[1]} multi nof"
				else:
					filepath = f"{args[1]} multi"
				train(args[0][:args[1]], "multi", args[1])
				if do_tests:
					return predict(args[0][args[1]:], filepath, to_excel)
		case "poly":
			for args in instructions:
				if not "bmw" in args[0].columns:
					filepath = f"{args[1]} poly {args[2]} nof"
				else:
					filepath = f"{args[1]} poly {args[2]}"
				train(args[0][:args[1]], "poly", args[1], degree=args[2])
				if do_tests:
					return predict(args[0][args[1]:], filepath, to_excel)