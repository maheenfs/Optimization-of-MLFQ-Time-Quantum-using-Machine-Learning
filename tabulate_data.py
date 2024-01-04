import pandas as pd
from tabulate import tabulate

def display_summary_table(file1, file2, file3):
    # Read CSV files
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    df3 = pd.read_csv(file3)

    # Calculate turnaround time
    turnaround_time1 = df1['Total Time (In Milliseconds)'].sum()
    turnaround_time2 = df2['Total Time (In Milliseconds)'].sum()
    turnaround_time3 = df3['Total Time (In Milliseconds)'].sum()

    # Get the top 10 predicted quantum numbers
    top_quantum1 = df1['Predicted Time Quantum'].nlargest(10).tolist()
    top_quantum2 = df2['Predicted Time Quantum'].nlargest(10).tolist()
    top_quantum3 = df3['Predicted Time Quantum'].nlargest(10).tolist()

    # Create a summary table
    summary_table = [
        ['Dataset', 'Total Turnaround Time', 'Top 10 Predicted Quantum Numbers'],
        ['Artificial Dataset', turnaround_time1, top_quantum1],
        ['Maheen_Dataset', turnaround_time2, top_quantum2],
        ['Chathurya_Dataset', turnaround_time3, top_quantum3],
    ]

    # Display the table
    print(tabulate(summary_table, headers='firstrow', tablefmt='grid'))

    # Save the table to a CSV file
    summary_df = pd.DataFrame(summary_table[1:], columns=summary_table[0])
    summary_df.to_csv('summary_table.csv', index=False)


# Example usage
display_summary_table('optimized_artificial_dataset.csv', 'optimized_logfile_maheen.csv', 'optimized_logfile_chathurya_2.csv')
