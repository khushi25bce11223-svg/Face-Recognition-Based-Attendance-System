import csv
from datetime import datetime

# Function to save the attendance
def save_presence_to_csv(file_name):
  # Get the current date and time
  now = datetime.now()
  time_now = now.strftime("%H:%M:%S")
  date_now = now.strftime("%d/%m/%Y")

# Opening the file in 'append' mode so it doesn't delete old data
# I am opening and closing it every time to be safe
file = open(file_name, 'a', newline='')
writer = csv.writer(file)

#Writing a simple row: [Status, Date, Time]
writer.writerow(["Student Present", date_now, time_now])

file.close()
print(f"Log updated at {time_now}")
