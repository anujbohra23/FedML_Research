import matplotlib.pyplot as plt

rounds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20]
accuracy = [0.775, 0.778, 0.780, 0.785, 0.788, 0.790, 0.795, 0.797, 0.800, 0.795]

plt.figure(figsize=(8, 6))
plt.plot(rounds, accuracy, marker='o', label='Client accuracy')
plt.xlabel('Rounds')
plt.ylabel('Accuracy')
plt.title('Accuracy Over Rounds')
plt.legend()
plt.grid(True)
plt.show()
