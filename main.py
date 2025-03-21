import matplotlib.pyplot as plt
import tensorflow as tf
from keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN, LSTM, GRU, Dense

# Load the IMDB dataset
max_features = 10000  # Number of words to consider as features
maxlen = 500  # Cut off reviews after this number of words
batch_size = 32

print('Loading data...')
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=max_features)
print(len(train_data), 'train sequences')
print(len(test_data), 'test sequences')

# Pad sequences to a fixed length
print('Pad sequences (samples x time)')
train_data =pad_sequences(train_data, maxlen=maxlen)
test_data =pad_sequences(test_data, maxlen=maxlen)
print('Train data shape:', train_data.shape)
print('Test data shape:', test_data.shape)

# Define a function to create and train a model
def create_and_train_model(model_type):
    model = Sequential()

    # Add an Embedding layer
    model.add(Embedding(max_features, 32))

    # Choose the RNN layer based on the provided model type
    if model_type == 'SimpleRNN':
        model.add(SimpleRNN(32))
    elif model_type == 'LSTM':
        model.add(LSTM(32))
    elif model_type == 'GRU':
        model.add(GRU(32))
    else:
        raise ValueError("Invalid model type. Use 'SimpleRNN', 'LSTM', or 'GRU'.")

    # Add a Dense layer
    model.add(Dense(1, activation='sigmoid'))
    # Compile the model
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    # Train the model
    history = model.fit(train_data, train_labels, epochs=5, batch_size=batch_size, validation_split=0.2, verbose=0)

    return model, history

# Create and train models for SimpleRNN, LSTM, and GRU
model_rnn, history_rnn = create_and_train_model('SimpleRNN')
model_lstm, history_lstm = create_and_train_model('LSTM')
model_gru, history_gru = create_and_train_model('GRU')

# Evaluate models on the test set
results_rnn = model_rnn.evaluate(test_data, test_labels, verbose=0)
results_lstm = model_lstm.evaluate(test_data, test_labels, verbose=0)
results_gru = model_gru.evaluate(test_data, test_labels, verbose=0)

# Print test accuracy
print(f'Test accuracy (SimpleRNN): {results_rnn[1]}')
print(f'Test accuracy (LSTM): {results_lstm[1]}')
print(f'Test accuracy (GRU): {results_gru[1]}')

# Plot validation accuracy
plt.figure(figsize=(8, 5))
plt.plot(history_rnn.history['val_accuracy'], label='SimpleRNN')
plt.plot(history_lstm.history['val_accuracy'], label='LSTM')
plt.plot(history_gru.history['val_accuracy'], label='GRU')
plt.title('Validation Accuracy Comparison')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#predictions
predictions_rnn = model_rnn.predict(test_data)
predictions_lstm = model_lstm.predict(test_data)
predictions_gru = model_gru.predict(test_data)
