import tensorflow as tf
from keras import layers
import flwr as fl
from dataset import load_data
from keras.metrics import Precision, Recall, AUC
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau

# Load the dataset
(x_train, y_train), (x_test, y_test) = load_data()

# Define the model
model = tf.keras.Sequential(
    [
        tf.keras.Input(shape=(8,)),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ]
)

# Compile the model with a reduced learning rate
model.compile(
    optimizer=Adam(learning_rate=0.0001),  # Reduced learning rate
    loss="binary_crossentropy",
    metrics=["accuracy", Precision(), Recall(), AUC()],
)

# Define the callback
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',   # Monitor validation loss
    factor=0.1,           # Factor by which the learning rate will be reduced
    patience=3,           # Number of epochs with no improvement after which learning rate will be reduced
    min_lr=1e-6           # Lower bound on the learning rate
)

# Define the Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(x_train, y_train, epochs=20, validation_split=0.2, callbacks=[reduce_lr])
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        # Evaluate the model and extract each metric
        loss, accuracy, precision, recall, auc = model.evaluate(x_test, y_test)

        # Return the evaluation results in a valid format
        return loss, len(x_test), {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "auc": auc,
        }

# Start the Flower client
fl.client.start_client(
    server_address="127.0.0.1:8080", client=FlowerClient().to_client()
)
