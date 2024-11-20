import pandas as pd
from enum import Enum


class Religions(Enum):
    CATHOLIC = 0
    OTHER_CHRISTIAN = 1
    MUSLIM = 2
    BUDDHIST = 3
    HINDU = 4
    ETHNIC = 5
    MARXIST = 6
    OTHERS = 7

# Load the flag data
data = pd.read_csv('flag.csv')

data.columns = ["Country", "Landmass", "Zone", "Area", "Population", 
                "Language", "Religion", "NumVBars", "NumStripes", "NumColours", 
                "Red", "Green", "Blue", "Gold", "White", "Black", "Orange", 
                "Mainhue", "Circles", "Crosses", "Saltires", "Quarters", "Sunstars", 
                "Crescent", "Triangle", "Icon", "Animate", "Text", "Topright", "Botleft"]

valid_colours = ["Red", "Green", "Blue", "Gold", "White", "Black", "Orange"]
valid_symbols = ["Sunstars", "Crescent", "Circles", "Crosses", "Saltires", "Quarters", "Triangle", "Icon", "Animate", "Text"]
valid_religions = ["CATHOLIC", "OTHER_CHRISTIAN", "MUSLIM", "BUDDHIST",
                   "HINDU", "ETHNIC", "MARXIST", "OTHERS"]

TP_COUNT = 0
TN_COUNT = 0
FP_COUNT = 0
FN_COUNT = 0

def outputResults(TP, TN, FP, FN ):
    print("True positive: ", TP)
    print("True negative: ", TN)
    print("False positive: ", FP)
    print("False negative: ", FN)
    
    accuracy = (TP + FN) / (TP + TN + FP + FN) 
    accuracy = accuracy * 100

    print(f"Accuracy: {accuracy:.2f}%") 

def crescent_muslim_accuracy_check():
    # Initialize counters
    TP_COUNT = 0
    TN_COUNT = 0
    FP_COUNT = 0
    FN_COUNT = 0
    
    # Loop through rows of data
    for index, row in data.iterrows():
        # print(index)
        # print(row)
        if (row["Crescent"] == 1):
            if (row["Religion"] == 2):
                TP_COUNT += 1
            else:
                TN_COUNT += 1
        else:
            if (row["Religion"] == 2):
                FP_COUNT += 1
            else:
                FN_COUNT += 1
    outputResults(TP_COUNT, TN_COUNT, FP_COUNT, FN_COUNT)

         
def crosses_saltires_christian_accuracy_check():

    TP_COUNT = 0
    TN_COUNT = 0
    FP_COUNT = 0
    FN_COUNT = 0

    for index, row in data.iterrows():
        if (row["Crosses"] == 1) or (row["Saltires"] == 1):
            if (row["Religion"] == 1):
                TP_COUNT += 1
            else:
                TN_COUNT += 1
        else:
            if (row["Religion"] == 1):
                FP_COUNT += 1
            else:
                FN_COUNT += 1
    outputResults(TP_COUNT, TN_COUNT, FP_COUNT, FN_COUNT)

def icon_animate_budha_accuracy_check():
    for index, row in data.iterrows():
        if (row["Icon"] == 1) or (row["Animate"] == 1):
            if (row["Religion"] == 1):
                TP_COUNT += 1
            else:
                TN_COUNT += 1
        else:
            if (row["Religion"] == 1):
                FP_COUNT += 1
            else:
                FN_COUNT += 1

def icon_animate_hindu_accuracy_check():
    for index, row in data.iterrows():
        if (row.columns["Icon"] == 1) or (row.columns["Animate"] == 1):
            if (row.valid_religions["HINDU"] == 1):
                TP_COUNT += 1
            else:
                TN_COUNT += 1
        else:
            if (row.valid_religions["HINDU"] == 1):
                FP_COUNT += 1
            else:
                FN_COUNT += 1

def infer_countries(data, userFilters):
    filtered_data = data
    for feature, value in userFilters.items():
        if value is not None:
            if feature == 'Population':
                if value == '>20':
                    filtered_data = filtered_data[filtered_data['Population'] >= 20]
                elif value == '<20':
                    filtered_data = filtered_data[filtered_data['Population'] < 20]
            else:
                filtered_data = filtered_data[filtered_data[feature] == value]
    return filtered_data[["Country", "Religion", "Population"]]


# Main expert system loop
userFilters = {}

print("Welcome to the Flag Religion Expert System!")
print("We'll help you identify countries based on their flag features and religious demographics.")

# Continue asking questions until the number of matching countries is minimal - so 4:
while True:
    matching_countries = infer_countries(data, userFilters)
    
    # Informs the user about current matches meeting with their prompts 
    print(f"\nCurrently matching {len(matching_countries)} countries.")
    
    if len(matching_countries) <= 0:
        print("No valid flags meet your criteria")
        exit()
    elif len(matching_countries) <= 4:
         break  # Exit the loop if the number of matches is small enough



    # Display selected criteria
    print("Criteria so far:")
    for key, value in userFilters.items():
        print(f" - {key}: {'Present' if value == 1 else 'Absent'}")

    # Ask about colour features
    while True:
        feature = input("Does the flag contain any of these colours? (e.g. Red, Green, Blue, Gold, White, Orange or press enter to skip this step): ").capitalize()
        if feature in valid_colours:
            userFilters[feature] = 1 
            break
        elif feature == '':
            print("Skipping colour input")
            break
        else:
            print("Not a valid colour. Please try again.")

    # Ask about symbols on the flag
    while True:
        symbol_feature = input("Does the flag contain any symbols? (e.g. Sunstars, Crescent, Circles, Crosses, Saltires, Quarters, Triangle, Icon or press enter to skip this step): ").capitalize()
        if symbol_feature in valid_symbols:
            userFilters[symbol_feature] = 1 
            if symbol_feature == "Crescent":
                print("crescent")
                crescent_muslim_accuracy_check()
            elif symbol_feature == "Crosses" or symbol_feature == "Saltires" :
                crosses_saltires_christian_accuracy_check()
            else:
                print("doing nothing")
                break
            break
        elif symbol_feature == '':
            print("Skipping colour input")
            break
        else:
            print("Not a valid symbol. Please try again.")
    while True:
        population_input = input("Is population above 20 million (y/n) or press enter to skip: ")
        if population_input == 'y':
            userFilters['Population'] = '>20'
            break
        elif population_input == 'n':
            userFilters['Population'] = '<20'
            break
        elif population_input == '':
            print("Skipping population filter")
            break
        else:
            print("Invalid input. Please enter 'y' for above 20 million, 'n' for below, or press enter to skip.")
            

# Display the final matching countries
print("Here are the main countries that match your criteria:")
print(matching_countries)


#   1. name:	Name of the country concerned
#    2. landmass:	1=N.America, 2=S.America, 3=Europe, 4=Africa, 4=Asia, 6=Oceania
#    3. zone:	Geographic quadrant, based on Greenwich and the Equator; 1=NE, 2=SE, 3=SW, 4=NW
#    4. area:	in thousands of square km
#    5. population:	in round millions
#    6. language: 1=English, 2=Spanish, 3=French, 4=German, 5=Slavic, 6=Other Indo-European, 7=Chinese, 8=Arabic, 9=Japanese/Turkish/Finnish/Magyar, 10=Others
#    7. religion: 0=Catholic, 1=Other Christian, 2=Muslim, 3=Buddhist, 4=Hindu, 5=Ethnic, 6=Marxist, 7=Others
#    8. bars:     Number of vertical bars in the flag
#    9. stripes:  Number of horizontal stripes in the flag
#   10. colours:  Number of different colours in the flag
#   11. red:      0 if red absent, 1 if red present in the flag
#   12. green:    same for green
#   13. blue:     same for blue
#   14. gold:     same for gold (also yellow)
#   15. white:    same for white
#   16. black:    same for black
#   17. orange:   same for orange (also brown)
#   18. mainhue:  predominant feature in the flag (tie-breaks decided by taking the topmost hue, if that fails then the most central hue, and if that fails the leftmost hue)
#   19. circles:  Number of circles in the flag
#   20. crosses:  Number of (upright) crosses
#   21. saltires: Number of diagonal crosses
#   22. quarters: Number of quartered sections
#   23. sunstars: Number of sun or star symbols
#   24. crescent: 1 if a crescent moon symbol present, else 0
#   25. triangle: 1 if any triangles present, 0 otherwise
#   26. icon:     1 if an inanimate image present (e.g., a boat), otherwise 0
#   27. animate:  1 if an animate image (e.g., an eagle, a tree, a human hand) present, 0 otherwise
#   28. text:     1 if any letters or writing on the flag (e.g., a motto or slogan), 0 otherwise
#   29. topleft:  feature in the top-left corner (moving right to decide tie-breaks)
#   30. botright: feature in the bottom-left corner (moving left to decide tie-breaks)
