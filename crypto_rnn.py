import tensorflow as tf
import pandas as pd
import random
import numpy as np
import time
from collections import deque
from sklearn import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

SEQ_LEN = 60 # How long of a preceeding sequence to collect for RNN
FUTURE_PERIOD_PREDICT = 3 # How far into the future are we trying to predict?
RATIO_TO_PREDICT = "LTC-USD"
EPOCHS = 10 # how many passes through our data
BATCH_SIZE = 64 # how many batches? Try smaller batch if you're getting OOM (out of memory) errors
NAME = f"{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED-{int(time.time())}"

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

def classify(current, future):
    if float(future) > float(current): # if the future price is higher than the current, that's a buy, or a 1
        return 1 
    else: # otherwise... it's a 0!
        return 0

def preprocess_df(df):
    df = df.drop('future', 1)

    for col in df.columns:
        if col != "target":
            df[col] = df[col].pct_change()
            df.dropna(inplace=True)
            df[col] = preprocessing.scale(df[col].values)

    df.dropna(inplace=True)

    sequential_data = []  # this is a list that will CONTAIN the sequences
    # Wait until predates has sixty values and then from there just keep populating it
    prev_days = deque(maxlen=SEQ_LEN)

    for i in df.values:
        # Right here we are not taking the "target"
        prev_days.append([n for n in i[:-1]])
        if len(prev_days) == SEQ_LEN:
            sequential_data.append([np.array(prev_days), i[-1]])

    random.shuffle(sequential_data)   
    buys = []
    sells = []

    # Balancing sequence data
    for seq, target in sequential_data:
        if target == 0:
            sells.append([seq, target])
        elif target == 1:
            buys.append([seq, target])

    random.shuffle(buys)
    random.shuffle(sells)
    
    lower = min(len(buys), len(sells))
    # Buy of sell up to the lower
    buys = buys[:lower]
    sells = sells[:lower]
    
    sequential_data = buys + sells
    random.shuffle(sequential_data)

    X = []
    Y = []
    
    for seq, target in sequential_data:
        X.append(seq)
        Y.append(target)
    return np.array(X), Y

main_df = pd.DataFrame()
ratios = ["BTC-USD", "LTC-USD", "ETH-USD", "BCH-USD"] # the 4 ratios we want to consider

for ratio in ratios:
    dataset = f"crypto_data/{ratio}.csv"
    df = pd.read_csv(
        dataset, names=["time", "low", "high", "open", "close", "volume"])
    df.rename(columns={"close": f"{ratio}_close",
              "volume": f"{ratio}_volume"}, inplace=True)
    df.set_index("time", inplace=True)

    df = df[[f"{ratio}_close", f"{ratio}_volume"]]

    if len(main_df) == 0:
        main_df = df
    else:
        main_df = main_df.join(df)

main_df['future'] = main_df[f"{RATIO_TO_PREDICT}_close"].shift(-FUTURE_PERIOD_PREDICT)
main_df['target'] = list(map(classify, main_df[f"{RATIO_TO_PREDICT}_close"], main_df["future"]))

times = sorted(main_df.index.values)
last_5pct = times[-int(0.05 * len(times))]

validation_main_df = main_df[(main_df.index >= last_5pct)]
main_df = main_df[(main_df.index < last_5pct)]

train_x, train_y = preprocess_df(main_df)
validation_x, validation_y = preprocess_df(validation_main_df)

print(f"Train data: {len(train_x)} validation: {len(validation_x)}")
print(f"Dont buys: {train_y.count(0)}, buys: {train_y.count(1)}")
print(f"VALIDATION Dont buys: {validation_y.count(0)}, buys: {validation_y.count(1)}")

# -------------- Model ----------------
model = Sequential()

model.add(LSTM(128, input_shape = (train_x.shape[1:]), return_sequences = True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(LSTM(128, input_shape = (train_x.shape[1:]), return_sequences = True))
model.add(Dropout(0.1))
model.add(BatchNormalization())

model.add(LSTM(128, input_shape = (train_x.shape[1:])))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(32, activation = "relu"))
model.add(Dropout(0.2))

model.add(Dense(2, activation = "softmax")) # output layer 

opt = tf.keras.optimizers.Adam(lr = 0.001, decay = 1e-6)

# Compile Model
model.compile(loss = 'sparse_categorical_crossentropy',
              optimizer = opt,
              metrics = ['accuracy'])

tensorboard = TensorBoard(log_dir = f'logs/{NAME}')

filepath = "RNN_Final-{epoch:02d}-{val_accuracy:3f}"  # unique file name that will include the epoch and the validation acc for that epoch
checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')) # saves only the best ones

# Train model
train_x = np.asarray(train_x)
train_y = np.asarray(train_y)
validation_x = np.asarray(validation_x)
validation_y = np.asarray(validation_y)

history = model.fit(
    train_x, train_y,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(validation_x, validation_y),
    callbacks=[tensorboard, checkpoint],
)

# Score model
score = model.evaluate(validation_x, validation_y, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Save model
model.save("models/{}".format(NAME))