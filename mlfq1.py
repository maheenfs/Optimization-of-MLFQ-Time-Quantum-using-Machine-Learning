import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



# Load and preprocess the data
data = pd.read_csv('artificial_dataset.csv')
filename='optimized_artificial_dataset.csv'

#data = pd.read_csv('logfile_maheen[45].csv')
#filename='optimized_logfile_maheen.csv'

#data = pd.read_csv('logfile_chathurya_2[10].CSV')
#filename='optimized_logfile_chathurya_2.csv'



# Function to convert time string to seconds
def time_to_seconds(time_str):
    time_parts = time_str.split(':')
    if len(time_parts) == 2:
        m, s = time_parts
        return int(m) * 60 + float(s)
    elif len(time_parts) == 3:
        h, m, s = time_parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    else:
        raise ValueError("Invalid time format")

# Create a label encoder for the 'Process Name' column
process_name_encoder = LabelEncoder()

# Create an imputer for filling missing values with the mean of the respective column
imputer = SimpleImputer(strategy='mean')

# Encode the 'Process Name' column
data['Process Name'] = process_name_encoder.fit_transform(data['Process Name'])

# Convert the 'Time of Day' column to seconds
data['Time of Day'] = data['Time of Day'].apply(time_to_seconds)

# Encode the 'Operation' column
data = pd.get_dummies(data, columns=['Operation'])
# Handle missing values
data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
# Train a machine learning model to predict time quantum values
X = data.drop(columns=['Total Time (In Milliseconds)'])
y = data['Total Time (In Milliseconds)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Add the predicted time quantum values to the data
data['Predicted Time Quantum'] = np.nan
data.loc[X_test.index, 'Predicted Time Quantum'] = y_pred


# Filter the data to remove rows with NaN values in the 'Predicted Time Quantum' column
data_filtered = data.loc[data['Predicted Time Quantum'].notna()].reset_index(drop=True)



# Implement the MLFQ algorithm



def mlfq_algorithm(data):
    num_queues = 4
    queues = [[] for _ in range(num_queues)]

    # Assign processes to queues using pd.qcut() and the 'Burst Time (Randomly Generated)' column
    data['Queue Index'] = pd.qcut(data['Burst Time (Randomly Generated)'], num_queues, labels=False, duplicates='drop')

    for _, row in data.iterrows():
        queue_index = int(row['Queue Index'])  # Convert the Queue Index to an integer
        queues[queue_index].append(row)

    optimized_data = []
    for queue in queues:
        optimized_data.extend(queue)

    return pd.DataFrame(optimized_data, columns=data.columns)

# Optimize the dataset using the MLFQ algorithm
optimized_data = mlfq_algorithm(data_filtered)


# Evaluate the performance of the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("R2 Score:", r2)

# Save the optimization results in a CSV file
optimized_data.to_csv(filename, index=False)


# Visualize the optimized data using a Gantt chart
def plot_gantt_chart(optimized_data, num_queues=4):
    fig, ax = plt.subplots(figsize=(12, 6))
    y_ticks = []
    colors = plt.get_cmap('tab10', num_queues)

    # Calculate the thresholds for the priority queues
    time_quantum_values = optimized_data['Predicted Time Quantum'].values
    thresholds = [np.percentile(time_quantum_values, i * 100 / num_queues) for i in range(1, num_queues)]

    for index, row in optimized_data.iterrows():
        # Calculate the queue index based on the 'Predicted Time Quantum' value and the thresholds
        time_quantum = row['Predicted Time Quantum']
        queue_index = sum(time_quantum > threshold for threshold in thresholds)

        ax.broken_barh([(row['Time of Day'], row['Burst Time (Randomly Generated)'])], (index * 10, 5), facecolors=colors(queue_index))
        y_ticks.append(index * 10 + 2.5)

    ax.set_xlabel('Time (ms)')
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(optimized_data['PID'])
    ax.set_ylabel('Processes')
    ax.set_title('Gantt Chart for Optimized Processes')

    # Create a custom legend for the priority queues
    legend_handles = [plt.Line2D([0], [0], color=colors(i), lw=4) for i in range(num_queues)]
    legend_labels = [f'Priority Queue {i}' for i in range(num_queues)]
    ax.legend(legend_handles, legend_labels)
    plt.savefig('gantt_chart.png', bbox_inches='tight')
    plt.show()


def compute_performance_metrics(optimized_data):
    n_processes = optimized_data.shape[0]
    waiting_times = []
    turnaround_times = []
    response_times = []
    completion_times = []

    current_time = 0
    for index, row in optimized_data.iterrows():
        arrival_time = row['Total Time (In Milliseconds)']
        burst_time = row['Burst Time (Randomly Generated)']

        waiting_time = max(0, current_time - arrival_time)

        waiting_times.append(waiting_time)
        turnaround_time = waiting_time + burst_time
        turnaround_times.append(turnaround_time)
        response_time = waiting_time
        response_times.append(response_time)
        completion_time = current_time + burst_time
        completion_times.append(completion_time)

        current_time = completion_time

    optimized_data['Waiting Time'] = waiting_times  # Add waiting times to the DataFrame
    optimized_data['Completion Time'] = completion_times  # Add completion times to the DataFrame

    avg_waiting_time = sum(waiting_times) / n_processes
    avg_turnaround_time = sum(turnaround_times) / n_processes
    avg_response_time = sum(response_times) / n_processes
    avg_completion_time = sum(completion_times) / n_processes

    print("Average Waiting Time:", avg_waiting_time)
    print("Average Turnaround Time:", avg_turnaround_time)
    print("Average Response Time:", avg_response_time)
    print("Average Completion Time:", avg_completion_time)

    # Save performance metrics to a CSV file
    performance_metrics = pd.DataFrame({
        'Metric': ['Average Waiting Time', 'Average Turnaround Time', 'Average Response Time',
                   'Average Completion Time', 'Mean Absolute Error', 'Mean Squared Error', 'R2 Score'],
        'Value': [avg_waiting_time, avg_turnaround_time, avg_response_time, avg_completion_time, mae, mse, r2]
    })

    performance_metrics.to_csv('performance_metrics.csv', index=False)


    return response_times



def analyze_queues(data):
    num_queues = 4
    queues = [0] * num_queues
    time_quantum_values = data['Predicted Time Quantum'].values
    thresholds = [np.percentile(time_quantum_values, i * 100 / num_queues) for i in range(1, num_queues)]

    queue_indices = []

    for _, row in data.iterrows():
        time_quantum = row['Predicted Time Quantum']

        # Find the queue index based on the calculated thresholds
        queue_index = sum(time_quantum > threshold for threshold in thresholds)
        queues[queue_index] += 1
        queue_indices.append(queue_index)

    print("Number of processes in each priority queue:", queues)

    # Create a DataFrame for the histogram plot
    hist_data = pd.DataFrame({"Priority Queue Index": queue_indices})

    # Assign different colors to each priority queue using a color palette
    palette = sns.color_palette("tab10", num_queues)
    sns.histplot(data=hist_data, x="Priority Queue Index", bins=num_queues, kde=False, discrete=True, hue="Priority Queue Index", palette=palette)
    plt.ylabel('Frequency')
    plt.title('Histogram of Processes in Priority Queues')
    plt.savefig('priority_queue_histogram.png', bbox_inches='tight')  # Save the plot as a PNG file

    plt.show()

# Histogram of Burst Time Distribution
def plot_burst_time_distribution(optimized_data):
    plt.figure(figsize=(12, 6))
    sns.histplot(data=optimized_data, x='Burst Time (Randomly Generated)', bins=20, kde=True)
    plt.xlabel('Burst Time (ms)')
    plt.ylabel('Frequency')
    plt.title('Burst Time Distribution')
    plt.savefig('burst_time_distribution.png', bbox_inches='tight')  # Save the plot as a PNG file
    plt.show()

# Call the function with the optimized_data DataFrame



# Boxplot of Waiting Time by Priority Queue
def plot_waiting_time_by_priority_queue(optimized_data, num_queues=4):
    optimized_data['Queue Index'] = pd.qcut(optimized_data['Waiting Time'], num_queues, labels=False, duplicates='drop')
    sns.boxplot(data=optimized_data, x='Queue Index', y='Waiting Time')
    plt.xlabel('Priority Queue')
    plt.ylabel('Waiting Time (ms)')
    plt.title('Boxplot of Waiting Time by Priority Queue')
    plt.savefig('waiting_time_by_priority_queue.png', bbox_inches='tight')
    plt.show()

# Scatterplot of Arrival Time vs. Completion Time
def plot_arrival_vs_completion_time(optimized_data, num_queues=4):
    plt.scatter(optimized_data['Time of Day'], optimized_data['Completion Time'], c=optimized_data['Queue Index'], cmap='tab10')
    plt.xlabel('Arrival Time (ms)')
    plt.ylabel('Completion Time (ms)')
    plt.title('Scatterplot of Arrival Time vs. Completion Time')
    plt.colorbar(label='Priority Queue')
    plt.savefig('arrival_vs_completion_time.png', bbox_inches='tight')  # Save the plot as a PNG file
    plt.show()

def plot_turnaround_time_by_priority_queue(optimized_data, num_queues=4 ):
    optimized_data['Queue Index'] = pd.qcut(optimized_data['Turnaround Time'], num_queues, labels=False,
                                            duplicates='drop')
    plt.figure()
    sns.boxplot(data=optimized_data, x='Queue Index', y='Turnaround Time')
    plt.title("Turnaround Time by Priority Queue")
    plt.savefig('turnaround_time_by_priority_queue.png', bbox_inches='tight')
    plt.show()



def plot_response_time_by_priority_queue(optimized_data, num_queues=4):
    optimized_data['Queue Index'] = pd.qcut(optimized_data['Response Time'], num_queues, labels=False,
                                            duplicates='drop')
    plt.figure()
    sns.boxplot(data=optimized_data, x='Queue Index', y='Response Time')
    plt.title("Response Time by Priority Queue")
    plt.savefig('response_time_by_priority_queue.png', bbox_inches='tight')  # Save the plot as a PNG file
    plt.show()

# Update the optimized_data DataFrame to include the priority queue index
num_queues = 4
optimized_data['Queue Index'] = pd.qcut(optimized_data['Predicted Time Quantum'], num_queues, labels=False, duplicates='drop')

# Save the distribution of time quantum values for each priority queue to a CSV file
def save_time_quantum_distribution_by_queues(optimized_data):
    time_quantum_distribution = optimized_data.groupby('Queue Index')['Predicted Time Quantum'].value_counts().reset_index(name='Frequency')
    time_quantum_distribution.to_csv('time_quantum_distribution_by_queues.csv', index=False)



# Call the functions with the optimized_data DataFrame
save_time_quantum_distribution_by_queues(optimized_data)





# Call the function with the optimized_data DataFrame
#compute_performance_metrics(optimized_data)
response_times = compute_performance_metrics(optimized_data)
optimized_data['Response Time'] = response_times
optimized_data['Turnaround Time'] = optimized_data['Waiting Time'] + optimized_data['Burst Time (Randomly Generated)']




# Call the visualization functions

analyze_queues(optimized_data)
plot_gantt_chart(optimized_data)
plot_burst_time_distribution(optimized_data)
plot_waiting_time_by_priority_queue(optimized_data)
plot_arrival_vs_completion_time(optimized_data)

plot_turnaround_time_by_priority_queue(optimized_data)
plot_response_time_by_priority_queue(optimized_data)




