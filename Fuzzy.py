import numpy as np
import skfuzzy as fuzz
import scipy
from skfuzzy import control as ctrl

# (Dubetcky, 2024)
# Dubetcky, O. (2024) 'Mastering Fuzzy Logic in Python', Medium, -04-03T13:24:10.510. Available at: https://oleg-dubetcky.medium.com/mastering-fuzzy-logic-in-python-c90463bf1135 (Accessed: Nov 13, 2024).

# Defining fuzzy variables for features that might influence religion
# (Start, Stop, Step) either will be true or false below

red = ctrl.Antecedent(np.arange(0, 2, 1), 'Red')
green = ctrl.Antecedent(np.arange(0, 2, 1), 'Green')
blue = ctrl.Antecedent(np.arange(0, 2, 1), 'Blue')
gold = ctrl.Antecedent(np.arange(0, 2, 1), 'Gold')
crescent = ctrl.Antecedent(np.arange(0, 2, 1), 'Crescent')
crosses = ctrl.Antecedent(np.arange(0, 2, 1), 'Crosses')

# with Triangular Membership Function and uses universe of discourse for fuzzy sets
# [left end point, peak point, right end point]
green['Included'] = fuzz.trimf(green.universe, [0, 1, 1]) 
green['Absent'] = fuzz.trimf(green.universe, [0, 0, 1])
red['Included'] = fuzz.trimf(red.universe, [0, 1, 1])
red['Absent'] = fuzz.trimf(red.universe, [0, 0, 1])
crescent['Included'] = fuzz.trimf(crescent.universe, [0, 1, 1])
crescent['Absent'] = fuzz.trimf(crescent.universe, [0, 0, 1])
crosses['Included'] = fuzz.trimf(crescent.universe, [0, 1, 1])
crosses['Absent'] = fuzz.trimf(crescent.universe, [0, 0, 1])

# Finding the Liklihood its muslim 
muslim = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'Muslim')
muslim['High'] = fuzz.trimf(muslim.universe, [0.6, 0.9, 1.0])
muslim['Medium'] = fuzz.trimf(muslim.universe, [0.3, 0.5, 0.7])
muslim['Low'] = fuzz.trimf(muslim.universe, [0.0, 0.2, 0.4])
# Finding the Liklihood its christian
christian = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'Christian')
christian['High'] = fuzz.trimf(christian.universe, [0.6, 0.9, 1.0])
christian['Medium'] = fuzz.trimf(christian.universe, [0.3, 0.5, 0.7])
christian['Low'] = fuzz.trimf(christian.universe, [0.0, 0.2, 0.4])

# Defining the fuzzy rules for muslim
Mus_rule1 = ctrl.Rule(green['Included'] & crescent['Included'], muslim['High'])
Mus_rule2 = ctrl.Rule(green['Included'] & crescent['Absent'], muslim['Medium'])
Mus_rule3 = ctrl.Rule(green['Absent'] & crescent['Included'], muslim['Medium'])
Mus_rule4 = ctrl.Rule(green['Absent'] & crescent['Absent'], muslim['Low'])
Mus_rule5 = ctrl.Rule(red['Included'], muslim['Low'])
Mus_rule6 = ctrl.Rule(crosses['Included'], muslim['Low'])

# Defining the fuzzy rules for christianity
Chr_rule1 = ctrl.Rule(red['Included'] & crosses['Included'], christian['High'])
Chr_rule2 = ctrl.Rule(red['Included'] & crosses['Absent'], christian['Medium'])
Chr_rule3 = ctrl.Rule(red['Absent'] & crosses['Included'], christian['Medium'])
Chr_rule4 = ctrl.Rule(red['Absent'] & crosses['Absent'], christian['Low'])
Chr_rule5 = ctrl.Rule(green['Included'], christian['Low'])
Chr_rule6 = ctrl.Rule(crescent['Included'], christian['Low'])

# gathers the fuzzy rules and conditions above for muslim
is_muslim_ctrl = ctrl.ControlSystem([Mus_rule1, Mus_rule2, Mus_rule3, Mus_rule4, Mus_rule5, Mus_rule6])
is_muslim_ctrl_simulate = ctrl.ControlSystemSimulation(is_muslim_ctrl)

# gathers the fuzzy rules and conditions above for christian
is_christian_ctrl = ctrl.ControlSystem([Chr_rule1, Chr_rule2, Chr_rule3, Chr_rule4, Chr_rule5, Chr_rule6])
is_christian_ctrl_simulate = ctrl.ControlSystemSimulation(is_christian_ctrl)

# prompts the user to enter
green_input = int(input("Enter 1 if the flag has Green, otherwise 0: "))
crescent_input = int(input("Enter 1 if the flag has a Crescent, otherwise 0: "))
crosses_input = int(input("Enter 1 if the flag has a Cross, otherwise 0: "))
red_input = int(input("Enter 1 if the flag has Red, otherwise 0: "))

# 1 meaning 'included' 0 meaning 'absent' 
is_muslim_ctrl_simulate.input['Green'] = green_input # setting green to true for one of the features in the flag
is_muslim_ctrl_simulate.input['Crescent'] = crescent_input 
is_muslim_ctrl_simulate.input['Red'] = red_input
is_muslim_ctrl_simulate.input['Crosses'] = crosses_input



is_christian_ctrl_simulate.input['Green'] = green_input # setting green to true for one of the features in the flag
is_christian_ctrl_simulate.input['Crosses'] = crosses_input
is_christian_ctrl_simulate.input['Crescent'] = crescent_input 
is_christian_ctrl_simulate.input['Red'] = red_input

is_muslim_ctrl_simulate.compute() # Computes the muslim probability

muslim_probability = is_muslim_ctrl_simulate.output['Muslim']
print(f"Muslim probability number: {muslim_probability:.2f}")

# outputs the probability number and chance its a muslim flag
if is_muslim_ctrl_simulate.output['Muslim'] >= 0.7:
    print("Its a very high chance its muslim")
elif is_muslim_ctrl_simulate.output['Muslim'] >= 0.4:
    print ("Theres a medium chance its muslim")
else:
    print ("There is a very low chance your flag is muslim")

# outputs the probability number and chance its a christian flag
is_christian_ctrl_simulate.compute() # Computes the christian probability

christian_probability = is_christian_ctrl_simulate.output['Christian']
print(f"Christian probability number: {christian_probability:.2f}")


if is_christian_ctrl_simulate.output['Christian'] >= 0.7:
    print("Its a very high chance its christian")
elif is_christian_ctrl_simulate.output['Christian'] >= 0.4:
    print ("Theres a medium chance its christian")
else:
    print ("There is a very low chance your flag is christian")