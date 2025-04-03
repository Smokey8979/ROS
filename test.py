# This is a program that asks for the user's name and age, then calculates the year when they'll turn 100

# Ask for user's name
name = input("What is your name? ")

# Ask for user's age
age = int(input("How old are you? "))

# Get the current year
import datetime
current_year = datetime.datetime.now().year

# Calculate the year when the user will turn 100
year_when_100 = current_year + (100 - age)

# Print out the result
print(f"Hello, {name}! You will turn 100 years old in the year {year_when_100}.")
