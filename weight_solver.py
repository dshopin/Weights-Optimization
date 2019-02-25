# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 17:20:40 2019

MIP optimization for weights and thresholds for the health score

@author: e6on6gv
"""

from ortools.linear_solver import pywraplp as lp
import csv


with open("C:\\Users\\e6on6gv\\Documents\\Print Attrition\\three_scores.csv") as f:
    scores = {}
    scores['retention'] = []
    scores['ar'] = []
    scores['trend'] = []
    reader = csv.reader(f)
    excluded = 0
    for row in reader:
        try:
            (float(r) for r in row)
            scores['retention'].append(float(row[0]))
            scores['ar'].append(float(row[1]))
            scores['trend'].append(float(row[2]))
        except ValueError:
            excluded += 1
            pass
    row_num = reader.line_num- excluded

targets = {}
for k in ['retention', 'ar', 'trend', 'total']:
    targets[k] = {}
    for c in ['green', 'red']:
        targets[k][c] = 0.2 * row_num


solver = lp.Solver("HealthScore", lp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
solver.set_time_limit(100000000)
objective = solver.Objective()

###### Decision Variables ######################

# decision variables for colors (isGreen and isRed)
color_vars = {}
for k in ['retention', 'ar', 'trend', 'total']:
    color_vars[k] = {}
    for c in ['green', 'red']:
        color_vars[k][c] = []
        for i in range(row_num):
            color_vars[k][c].append(solver.IntVar(0, 1, k + '_' + c + '_' + str(i)))

        
# decision variables for weights
weight_vars = {}
for k in ['retention', 'ar', 'trend']:
    weight_vars[k] = solver.NumVar(0.1, 1, k)

# decision variables for thresholds
thresh_vars = {}
for k in ['retention', 'ar', 'trend', 'total']:
    thresh_vars[k] = {}
    for c in ['green', 'red']:
        thresh_vars[k][c] = solver.NumVar(0,100, k + '_' + c)

########## Constraints #############################
#constraint - sum of weights = 1
sw = solver.Constraint(1,1, 'sum of weights')
for w in weight_vars.values():
    sw.SetCoefficient(w, 1)
   

# GreenThreshold - RedThreshold >= 0
red_lt_green_constr = {}
for k in ['retention', 'ar', 'trend', 'total']:
    red_lt_green_constr[k] = solver.Constraint(0,100)
    red_lt_green_constr[k].SetCoefficient(thresh_vars[k]['green'], 1)
    red_lt_green_constr[k].SetCoefficient(thresh_vars[k]['red'], -1)


# Three individual scores colors
# We compare score S with green threshold T and create binary D: if S >= green threshold then D=1 otherwise 0.
# We compare score S with red threshold T and create binary D: if S <= red threshold then D=1 otherwise 0.
#    
#         Green Threshold:
#         S >= T - 101*(1-D)
#         S < T + 101*D
#            
#         Red Threshold:
#         S >= T - 101*D
#         S < T + 101*(1-D)
#
#Regrouping:
#         Green Threshold:    
#         S < (T + 101 * D) <= S + 101
#
#         Red Threshold:
#         S - 101 < (T - 101 * D) <= S
#
# For Total Score we need to include weigths, because TotalS = W1*S1 + W2*S2 +W3*S3:
#
#         Green Threshold:    
#         0 < (T + 101 * D) - (W1*S1 + W2*S2 +W3*S3) <= 101
#
#         Red Threshold:
#         - 101 < (T - 101 * D) - (W1*S1 + W2*S2 +W3*S3) <= 0
              
thresh_constr = {}
for k in ['retention', 'ar', 'trend', 'total']:
    thresh_constr[k] = {'green':[], 'red':[]}
    for i in range(row_num):
        if k != 'total':
            thresh_constr[k]['green'].append(solver.Constraint(scores[k][i] + 0.01,scores[k][i] + 101)) #+0.01 because lower bound is enclusive
            thresh_constr[k]['green'][-1].SetCoefficient(thresh_vars[k]['green'], 1)
            thresh_constr[k]['green'][-1].SetCoefficient(color_vars[k]['green'][i], 101)
            
            thresh_constr[k]['red'].append(solver.Constraint(scores[k][i] - 101 + 0.01, scores[k][i]))
            thresh_constr[k]['red'][-1].SetCoefficient(thresh_vars[k]['red'], 1)
            thresh_constr[k]['red'][-1].SetCoefficient(color_vars[k]['red'][i], -101)
        else:
            thresh_constr[k]['green'].append(solver.Constraint(0 + 0.01,101))
            thresh_constr[k]['green'][-1].SetCoefficient(thresh_vars[k]['green'], 1)
            thresh_constr[k]['green'][-1].SetCoefficient(color_vars[k]['green'][i], 101)
            thresh_constr[k]['green'][-1].SetCoefficient(weight_vars['retention'], -scores['retention'][i])
            thresh_constr[k]['green'][-1].SetCoefficient(weight_vars['ar'], -scores['ar'][i])
            thresh_constr[k]['green'][-1].SetCoefficient(weight_vars['trend'], -scores['trend'][i])
            
            thresh_constr[k]['red'].append(solver.Constraint(-101 + 0.01,0))
            thresh_constr[k]['red'][-1].SetCoefficient(thresh_vars[k]['red'], 1)
            thresh_constr[k]['red'][-1].SetCoefficient(color_vars[k]['red'][i], -101)
            thresh_constr[k]['red'][-1].SetCoefficient(weight_vars['retention'], -scores['retention'][i])
            thresh_constr[k]['red'][-1].SetCoefficient(weight_vars['ar'], -scores['ar'][i])
            thresh_constr[k]['red'][-1].SetCoefficient(weight_vars['trend'], -scores['trend'][i])
        
# Logic rules of combining colors
# GMR = Greens Minus Reds.
#If GMR for 3 scores >= 2 then Total Score isGreen=1 otherwise isGreen=0
#If GMR for 3 scores < 0 then Total Score isRed=1 otherwise isRed = 0
#		
#    TotGreenRule:
#       GMR >= 2 - M + M*d	M=5
#    	GMR < 2 + M*w	
#    	TotGreen == w	
#    
#    		
#    TotRedRule:
#       GMR >= 0 - M*y 
#       GMR < 0 + M - M*y	  M=4
#    	TotRed == y	
#
#       OR
#
#    TotGreenRule:
#       RetGreen+ARGreen+TrendGreen - (RetRed+ARRed+TrendRed) - 5*w >= -3
#    	RetGreen+ARGreen+TrendGreen - (RetRed+ARRed+TrendRed) - 5*w < 2	
#    	TotGreen == w	
#    
#    		
#    TotRedRule:
#       RetGreen+ARGreen+TrendGreen - (RetRed+ARRed+TrendRed) + 4*y >= 0
#       RetGreen+ARGreen+TrendGreen - (RetRed+ARRed+TrendRed) + 4*y < 4
#    	TotRed == y                   
                    
gmr_bin_vars = {}
for k in ['green', 'red']:
    gmr_bin_vars[k] = []
    for i in range(row_num):
        gmr_bin_vars[k].append(solver.IntVar(0,1,k + '_' + str(i)))
                   
# Create N sets of constraints (1 per customer) for Total Score colors
#                             1 customer
#                       /                   \ 
#                      /                     \
#                     /                       \
#                    /                         \
#                isGreen                      isRed
#                /    \                      /    \
#    For_bin_var    Color_logic    For_bin_var  Color_logic
#    

        
gmr_constr = {}
for c in ['green', 'red']:
    if c == 'green':
        lb = -3
        ub = 2 - 0.01
        M = -5
    else:
        lb = 0
        ub = 4 - 0.01
        M = 4
    gmr_constr[c] = {'bin_var':[], 'color_logic':[]}
    for i in range(row_num):
        gmr_constr[c]['bin_var'].append(solver.Constraint(lb, ub))
        
        gmr_constr[c]['bin_var'][-1].SetCoefficient(color_vars['retention']['green'][i],1)
        gmr_constr[c]['bin_var'][-1].SetCoefficient(color_vars['ar']['green'][i],1)
        gmr_constr[c]['bin_var'][-1].SetCoefficient(color_vars['trend']['green'][i],1)
        gmr_constr[c]['bin_var'][-1].SetCoefficient(color_vars['retention']['red'][i],-1)
        gmr_constr[c]['bin_var'][-1].SetCoefficient(color_vars['ar']['red'][i],-1)
        gmr_constr[c]['bin_var'][-1].SetCoefficient(color_vars['trend']['red'][i],-1)
        gmr_constr[c]['bin_var'][-1].SetCoefficient(gmr_bin_vars[c][i],M)
        
        gmr_constr[c]['color_logic'].append(solver.Constraint(0,0))
        gmr_constr[c]['color_logic'][-1].SetCoefficient(color_vars['total'][c][i],1)
        gmr_constr[c]['color_logic'][-1].SetCoefficient(gmr_bin_vars[c][i],-1)
        
    
    
 
## Objective: have greens/reds below but as close as possible to targets
        
# constraints for greens/reds to be below targets but above 0.5*targets
targets_constr = {}
for k in ['retention', 'ar', 'trend', 'total']:
#for k in ['total']:
    targets_constr[k] = {}
    for c in ['green', 'red']:
        targets_constr[k][c] = solver.Constraint(0, targets[k][c])
        for i in range(row_num):
            targets_constr[k][c].SetCoefficient(color_vars[k][c][i], 1)
            objective.SetCoefficient(color_vars[k][c][i], 1)
        

                
# maximize, since we want to minimize (Target - Greens/Reds)
objective.SetMaximization()

status = solver.Solve()
# 0   OPTIMAL,        // optimal.
# 1   FEASIBLE,       // feasible, or stopped by limit.
# 2   INFEASIBLE,     // proven infeasible.
# 3   UNBOUNDED,      // proven unbounded.
# 4   ABNORMAL,       // abnormal, i.e., error of some kind.
# 5   MODEL_INVALID,  // the model is trivially invalid (NaN coefficients, etc).
# 6   NOT_SOLVED = 6  // not been solved yet.


# Solution

if status == 0:
    
    # weights
    for k,v in weight_vars.items():
        print('Weight for ', k, ':', v.solution_value())
        
    # thresholds
    for k,v in thresh_vars.items():
        for c,t in v.items():
            print(c,'threshold for ', k, ':', t.solution_value())



    # percentage of each color for each score
    for k,v in color_vars.items():
        for c,t in v.items():
            scores = [d.solution_value() for d in t]
            print('Percent of', c, 'for', k, ':', sum(scores)/len(scores))





