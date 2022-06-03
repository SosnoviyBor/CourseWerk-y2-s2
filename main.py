import tools
import predictor
import pandas as pd

inputData = {
	"firm": "BMW",
	"year": 6,
	"transmission": "automatic",
	"mileage": 100000,
	"fuelType": "Petrol",
	"kmpl": 10,
	"engineSize": 2.0
}
#df0 = tools.opendf("1.0")
#df1 = tools.opendf("1.1")
#df2 = tools.opendf("1.1 nof")

# THERE IS 99187 ENTRIES IN v1.1 DATAFRAME
# 75k is approximately 3/4

#tools.heatmapHelper(df1)
#predictor.train(df1, "poly", 75000, 3)
#predictor.predict(df1, "75000 poly 3", True)

# bulk things
"""
tmp = [
	[df1, 75000, 1],
	[df1, 75000, 2],
	[df1, 75000, 3],
	[df2, 75000, 1],
	[df2, 75000, 2],
	[df2, 75000, 3],
]
predictor.train_and_predict(tmp,"poly",True,True,False)

tmp = [
	[df1, 75000],
	[df2, 75000],
]
predictor.train_and_predict(tmp,"multi",True,True,False)
"""
print("done!")