from collections import deque
from sklearn import preprocessing
import pandas as pd
import random
import numpy as np

SEQ_LEN = 60
FUTURE_PERIOD_PREDICT = 3
RATIO_TO_PREDICT = "LTC-USD"

def classify(current, future):
    if float(future) > float(current):
        return 1 # 1 means we should buy it.
    else:
        return 0  

def preprocess_df(df):
    df = df.drop('future', 1)
    
    for col in df.columns:
        if col != "target":
            df[col] = df[col].pct_change()
            df.dropna(inplace = True)
            df[col] = preprocessing.scale(df[col].values)
    
    df.dropna(inplace = True)
    
    sequential_data = []
    prev_days = deque(maxlen = SEQ_LEN) # Wait until predates has sixty values and then from there just keep populating it 
    
    for i in df.values:
        prev_days.append([n for n in i[:-1]]) # Right here we are not taking the "target" 
        if len(prev_days) == SEQ_LEN:
            sequential_data.append([np.array(prev_days), i[-1]])       
    random.shuffle(sequential_data) # Sequences 
                           
main_df = pd.DataFrame()
ratios = ["BTC-USD", "LTC-USD", "ETH-USD", "BCH-USD"]

for ratio in ratios:
    dataset = f"crypto_data/{ratio}.csv"
    df = pd.read_csv(dataset, names = ["time", "low", "high", "open", "close", "volume"])
    df.rename(columns = {"close": f"{ratio}_close", "volume": f"{ratio}_volume"}, inplace = True)
    df.set_index("time", inplace = True)
    
    df = df[[f"{ratio}_close", f"{ratio}_volume"]]
    
    if len(main_df) == 0:
        main_df = df
    else:
        main_df = main_df.join(df)
        
main_df['future'] = main_df[f"{RATIO_TO_PREDICT}_close"].shift(-FUTURE_PERIOD_PREDICT)

main_df['target'] = list(map(classify, main_df[f"{RATIO_TO_PREDICT}_close"], main_df["future"]))
#print(main_df[[f"{RATIO_TO_PREDICT}_close", "future", "target"]].head())
 
times = sorted(main_df.index.values)
last_5pct = times[-int(0.05 * len(times))]

validation_main_df = main_df[(main_df.index >= last_5pct)]
main_df = main_df[(main_df.index < last_5pct)]

preprocess_df(main_df)
#train_x, train_y = preprocess_df(main_df)
#validation_x, validation_y = preprocess_df(validation_main_df)