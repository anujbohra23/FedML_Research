# import flwr as fl

# # Start the server with specified number of communication rounds
# fl.server.start_server(
#     server_address="0.0.0.0:8080", config=fl.server.ServerConfig(num_rounds=30)
# )



import flwr as fl
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# List to store accuracy after each round
acc = []

class AggregateMetricStrategy(fl.server.strategy.FedAvg):
    def aggregate_evaluate(
        self,
        server_round,
        results,
        failures,
    ):
        """Aggregate evaluation accuracy using weighted average."""

        if not results:
            return None, {}

        # Aggregate loss and metrics using the parent class method
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)

        # Calculate the aggregated accuracy
        accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]
        
        aggregated_accuracy = sum(accuracies) / sum(examples)
        
        # Append the aggregated accuracy to the list
        acc.append(aggregated_accuracy)
        
        print(f"Round {server_round} accuracy aggregated from client results: {aggregated_accuracy}")

        return aggregated_loss, {"accuracy": aggregated_accuracy}

# Start the federated learning server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=15),
    strategy=AggregateMetricStrategy()
)

# Create a DataFrame from the accuracy list
acc_df = pd.DataFrame(acc, columns=['accuracy'])

# Print the DataFrame to verify the data
print(acc_df)

# Set the style for the seaborn plot
sns.set_style("whitegrid")

# Create a new figure for plotting
plt.figure(figsize=(10, 6))

# Plot the accuracy over rounds
plot = sns.lineplot(data=acc_df, palette="tab10", linewidth=2.5)

# Set plot titles and labels
plot.set_title("Accuracy Over Rounds", fontsize=16, fontweight='bold')
plot.set_xlabel("Rounds", fontsize=14)
plot.set_ylabel("Accuracy", fontsize=14)

# Set the font size for ticks
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Add legend to the plot
plt.legend(title="Client", fontsize=12, title_fontsize=12)

# Show the plot
plt.show()
