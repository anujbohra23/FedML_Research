import tensorflow as tf
import tensorflow as tf
import keras
from keras import layers
import flwr as fl
from dataset import load_data

# Load data
(x_train, y_train), (x_test, y_test) = load_data()


# Define the model architecture
def create_model(learning_rate=0.001):
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(8,)),  # Input layer with 8 features
            tf.keras.layers.Dense(64, activation="relu"),  # First hidden layer
            tf.keras.layers.Dense(32, activation="relu"),  # Second hidden layer
            tf.keras.layers.Dense(1, activation="sigmoid"),  # Output layer
        ]
    )

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# Create the model with a specified learning rate
model = create_model(learning_rate=0.001)


# Define the Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=2)  # Batch size
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test)
        return loss, len(x_test), {"accuracy": accuracy}


# Start Flower client
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=FlowerClient())
