# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 17:20:40 2019

MIP optimization for weights and thresholds for the health score

@author: e6on6gv
"""

from ortools.linear_solver import pywraplp as lp
import csv
from collections import Counter


def color_rules(solver, variables):
    '''
    Creates constraints, enforcing rules of combining original scores' colors
    into total score colors.
    
    solver - ortools.Solver object
    variables - tuple of color variables in order (g1, g2, g3, G, r1, r2, r3, R)

    
    Returns a list of constraints restricting possible colors of Total based on colors of
    original scores.
    '''
        
    def add_color_rule(variables, coefs, lbs):
        ''' helper function , returns a list of constraints for a certain rule
            
            variables - list of color variables in order (g1, g2, g3, G, r1, r2, r3, R)
            coefs - tuple of tuples of coefficients in the same order. Each
                    nested tuple - one constraint
            lbs - tuple of numbers, corresponding RHS
            
            e. g. "g2 + g3 + r2 + r3 - g1 - G  ≥ -1" entered as:
                coefs = (-1,1,1,-1,0,1,1,0)
                lbs = -1
        '''
        constrs = []
        for i,lb in enumerate(lbs):
            constr = solver.Constraint(lb, solver.infinity())
            for j,v in enumerate(variables):
                constr.SetCoefficient(v, coefs[i][j])
            constrs.append(constr)
        return(constrs)
        
        
    constr_list = []

    # Rules
    lhs = [((-1,-1,-1,1,0,0,0,0),),
           ((-1,-1,1,1,0,0,1,0),(-1,1,-1,1,0,1,0,0),(1,-1,-1,1,1,0,0,0)),
           ((-1,1,1,-1,0,1,1,0),(-1,1,1,0,0,1,1,-1),(1,-1,1,-1,1,0,1,0),(1,-1,1,0,1,0,1,-1),(1,1,-1,-1,1,1,0,0),(1,1,-1,0,1,1,0,-1)),
           ((1,1,1,-1,1,1,1,0),(1,1,1,0,1,1,1,-1)),
           ((-1,-1,0,1,0,0,-1,-1),(-1,0,-1,1,0,-1,0,-1),(0,-1,-1,1,-1,0,0,-1)),
           ((-1,0,1,-1,0,-1,1,1),(0,-1,1,-1,-1,0,1,1),(1,-1,0,-1,1,0,-1,1),(1,0,-1,-1,1,-1,0,1),(-1,1,0,-1,0,1,-1,1),(0,1,-1,-1,-1,1,0,1)),
           ((0,0,-1,-1,-1,-1,0,1),(0,-1,0,-1,-1,0,-1,1),(-1,0,0,-1,0,-1,-1,1)),
           ((0,1,1,-1,-1,1,1,0),(0,1,1,0,-1,1,1,-1),(1,0,1,-1,1,-1,1,0),(1,0,1,0,1,-1,1,-1),(1,1,0,-1,1,1,-1,0),(1,1,0,0,1,1,-1,-1)),
           ((0,0,1,0,-1,-1,1,1),(0,1,0,0,-1,1,-1,1),(1,0,0,0,1,-1,-1,1)),
           ((0,0,0,0,-1,-1,-1,1),)
            ]
    rhs = [(-2,),
           (-1,-1,-1),
           (-1,-1,-1,-1,-1,-1),
           (0,0),
           (-3,-3,-3),
           (-2,-2,-2,-2,-2,-2),
           (-3,-3,-3),
           (-1,-1,-1,-1,-1,-1),
           (-1,-1,-1),
           (-2,)
            ]
    for rule in range(10):
        constr_list += add_color_rule(variables, lhs[rule], rhs[rule])

    return(constr_list)
        

with open("C:\\Users\\e6on6gv\\Documents\\Print Attrition\\Weights Optimization\\three_scores.csv") as f:
    counts = []
    scores = {}
    scores['retention'] = []
    scores['ar'] = []
    scores['trend'] = []
    reader = csv.reader(f)
    excluded = 0
    for row in reader:
        try:
            (print(float(r)) for r in row)
            scores['retention'].append(float(row[0]))
            scores['ar'].append(float(row[1]))
            scores['trend'].append(float(row[2]))
            counts.append(int(row[3]))
        except ValueError:
            excluded += 1
            pass
    row_num = reader.line_num- excluded


solver = lp.Solver("HealthScore", lp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
solver.set_time_limit(35*3600*1000)
objective = solver.Objective()



# decision variables for weights
weight_vars = {}
for k in ['retention', 'ar', 'trend']:
    weight_vars[k] = solver.NumVar(0.1, 1, k)
    #constraint - sum of weights = 1
sw = solver.Constraint(1,1, 'sum of weights')

for w in weight_vars.values():
    sw.SetCoefficient(w, 1)


# decision variables for thresholds
thresh_vars = {}
for k in ['retention', 'ar', 'trend', 'total']:
    thresh_vars[k] = {}
    for c in ['green', 'red']:
        thresh_vars[k][c] = solver.NumVar(0,100, k + '_' + c)

# Constraint that Green Threshold above Red Threshold
red_lt_green_constr = {}
for k in ['retention', 'ar', 'trend', 'total']:
    red_lt_green_constr[k] = solver.Constraint(1,100)
    red_lt_green_constr[k].SetCoefficient(thresh_vars[k]['green'], 1)
    red_lt_green_constr[k].SetCoefficient(thresh_vars[k]['red'], -1)

# binary variables for colors (isGreen and isRed)
color_vars = {}
for k in ['retention', 'ar', 'trend', 'total']:
    color_vars[k] = {}
    for c in ['green', 'red']:
        color_vars[k][c] = []
        for i in range(row_num):
            color_vars[k][c].append(solver.IntVar(0, 1, k + '_' + c + '_' + str(i)))

# Constraints for binary variables for colors; G - green threshold, R - red one
#         Green Threshold:
#   G + (100 + ε) * δ ≥ S + ε
#   G + 100 * δ ≤ 100 + S
#
#         Red Threshold:
#   R - (100 + ε) * δ ≤ S - ε
#   R - 100 * δ ≥ S - 100
#
#       For the total score
#
#         Green Threshold:
#   G + (100 + ε) * δ  - (W1*S1 + W2*S2 +W3*S3) ≥  + ε
#   G + 100 * δ - (W1*S1 + W2*S2 +W3*S3) ≤ 100
#
#         Red Threshold:
#   R - (100 + ε) * δ - (W1*S1 + W2*S2 +W3*S3) ≤  - ε
#   R - 100 * δ - (W1*S1 + W2*S2 +W3*S3) ≥ -100

epsilon = 0.01
thresh_constr = {}
for k in ['retention', 'ar', 'trend', 'total']:
    thresh_constr[k] = {}
    for c in ['green', 'red']:
        thresh_constr[k][c] = []
        for i in range(row_num):
            if k != 'total':
                if c == 'green':
                    constr_lb = solver.Constraint(scores[k][i] + epsilon, solver.infinity())
                    constr_lb.SetCoefficient(thresh_vars[k]['green'], 1)
                    constr_lb.SetCoefficient(color_vars[k]['green'][i], 100 + epsilon)
                    
                    constr_ub = solver.Constraint(-solver.infinity(), 100 + scores[k][i])
                    constr_ub.SetCoefficient(thresh_vars[k]['green'], 1)
                    constr_ub.SetCoefficient(color_vars[k]['green'][i], 100)
                else:
                    constr_lb = solver.Constraint(scores[k][i] - 100, solver.infinity())
                    constr_lb.SetCoefficient(thresh_vars[k]['red'], 1)
                    constr_lb.SetCoefficient(color_vars[k]['red'][i], -100)
                    
                    constr_ub = solver.Constraint(-solver.infinity(), scores[k][i] - epsilon)
                    constr_ub.SetCoefficient(thresh_vars[k]['red'], 1)
                    constr_ub.SetCoefficient(color_vars[k]['red'][i], -(100 + epsilon))
            else:
                if c == 'green':
                    constr_lb = solver.Constraint(epsilon, solver.infinity())
                    constr_lb.SetCoefficient(thresh_vars[k]['green'], 1)
                    constr_lb.SetCoefficient(color_vars[k]['green'][i], 100 + epsilon)
                    constr_lb.SetCoefficient(weight_vars['retention'], -scores['retention'][i])
                    constr_lb.SetCoefficient(weight_vars['ar'], -scores['ar'][i])
                    constr_lb.SetCoefficient(weight_vars['trend'], -scores['trend'][i])
                    
                    constr_ub = solver.Constraint(-solver.infinity(), 100)
                    constr_ub.SetCoefficient(thresh_vars[k]['green'], 1)
                    constr_ub.SetCoefficient(color_vars[k]['green'][i], 100)
                    constr_ub.SetCoefficient(weight_vars['retention'], -scores['retention'][i])
                    constr_ub.SetCoefficient(weight_vars['ar'], -scores['ar'][i])
                    constr_ub.SetCoefficient(weight_vars['trend'], -scores['trend'][i])
                else:
                    constr_lb = solver.Constraint(-100, solver.infinity())
                    constr_lb.SetCoefficient(thresh_vars[k]['red'], 1)
                    constr_lb.SetCoefficient(color_vars[k]['red'][i], -100)
                    constr_lb.SetCoefficient(weight_vars['retention'], -scores['retention'][i])
                    constr_lb.SetCoefficient(weight_vars['ar'], -scores['ar'][i])
                    constr_lb.SetCoefficient(weight_vars['trend'], -scores['trend'][i])
                               
                    constr_ub = solver.Constraint(-solver.infinity(), -epsilon)
                    constr_ub.SetCoefficient(thresh_vars[k]['red'], 1)
                    constr_ub.SetCoefficient(color_vars[k]['red'][i], -(100 + epsilon))
                    constr_ub.SetCoefficient(weight_vars['retention'], -scores['retention'][i])
                    constr_ub.SetCoefficient(weight_vars['ar'], -scores['ar'][i])
                    constr_ub.SetCoefficient(weight_vars['trend'], -scores['trend'][i])
                    
            thresh_constr[k][c].append((constr_lb, constr_ub))



# Constraints with colors' combining logic
color_logic = []
for i in range(row_num):
    variables = []
    for c in ['green', 'red']:
        for k in ['retention', 'ar', 'trend', 'total']:
            variables.append(color_vars[k][c][i])
    color_logic.append(color_rules(solver, variables))        


      
# Objective function - minimize deviation from the target

#target numbers of greens and reds
targets = {'retention':{'green':0.2 * sum(counts), 'red':0.2 * sum(counts)},
           'ar':{'green':0.2 * sum(counts), 'red':0.2 * sum(counts)},
           'trend':{'green':0.2 * sum(counts), 'red':0.2 * sum(counts)},
           'total':{'green':0.2 * sum(counts), 'red':0.2 * sum(counts)}
           }

dev_vars = {}
dev_constrs = {}
for k in ['retention', 'ar', 'trend', 'total']:
    dev_vars[k] = {}
    dev_constrs[k] = {}
    for c in ['green', 'red']:
        u = solver.NumVar(0,solver.infinity(),k+c+'_u')
        v = solver.NumVar(0,solver.infinity(),k+c+'_v')
        dev_vars[k][c] = (u, v)
        
        constr = solver.Constraint(targets[k][c], targets[k][c])
        for i in range(row_num):
            constr.SetCoefficient(color_vars[k][c][i], counts[i])
        constr.SetCoefficient(u, 1)
        constr.SetCoefficient(v, -1)
        dev_constrs[k][c] = constr
        
        #lower  penalties for original scores, higher - for total
        coef = 1
        if k == 'total':
            coef = 10
        objective.SetCoefficient(u, coef)
        objective.SetCoefficient(v, coef)
    
objective.SetMinimization()

status = solver.Solve()
# 0   OPTIMAL,        // optimal.
# 1   FEASIBLE,       // feasible, or stopped by limit.
# 2   INFEASIBLE,     // proven infeasible.
# 3   UNBOUNDED,      // proven unbounded.
# 4   ABNORMAL,       // abnormal, i.e., error of some kind.
# 5   MODEL_INVALID,  // the model is trivially invalid (NaN coefficients, etc).
# 6   NOT_SOLVED = 6  // not been solved yet.


# Solution

if status in [0, 1]:
    print("Status", status)
    
    # weights
    for k,v in weight_vars.items():
        print('Weight for ', k, ':', round(v.solution_value(),6))
        
    # thresholds
    for k,v in thresh_vars.items():
        for c,t in v.items():
            print(c,'threshold for ', k, ':', round(t.solution_value(),6))

    # percentage of each color for each score
    for k,v in color_vars.items():
        for c,t in v.items():
            number = 0
            for i in range(row_num):
                number += counts[i] * t[i].solution_value()
            print('Percent of', c, 'for', k, ':', number/sum(counts))

    #check color coding
    for k in ['retention', 'ar', 'trend', 'total']:
        print(k)
        greens = []
        yellows = []
        reds = []
        errors = []
        for i in range(row_num):
            isGreen = color_vars[k]['green'][i].solution_value()
            isRed = color_vars[k]['red'][i].solution_value()
            w1 = weight_vars['retention'].solution_value()
            w2 = weight_vars['ar'].solution_value()
            w3 = weight_vars['trend'].solution_value()
            if k == 'total':
                score = w1*scores['retention'][i] + w2*scores['ar'][i] + w3*scores['trend'][i]
            else:
                score = scores[k][i]
            if  isGreen and not isRed:
                greens += [score]*counts[i]
            elif not isGreen and isRed:
                reds += [score]*counts[i]
            elif not isGreen and not isRed:
                yellows += [score]*counts[i]
            else:
                errors.append(i)
        print('Greens: max score=', max(greens, default = None), ' min score=', min(greens, default = None), ' count=', len(greens))
        print('Yellows: max score=', max(yellows, default = None), ' min score=', min(yellows, default = None), ' count=', len(yellows))
        print('Reds: max score=', max(reds, default = None), ' min score=', min(reds, default = None), ' count=', len(reds))
        print('Number of errors=', len(errors))
        
    # check colors combination rules
    combos = []
    for i in range(row_num):
        greens = 0
        reds = 0
        for k in ['retention', 'ar', 'trend']:
           greens += color_vars[k]['green'][i].solution_value()
           reds += color_vars[k]['red'][i].solution_value()
        if color_vars['total']['green'][i].solution_value():
            result_color = 'Green'
        elif color_vars['total']['red'][i].solution_value():
            result_color = 'Red'
        else:
            result_color = 'Yellow'
        combos += [(greens, reds, result_color)] * counts[i]
    color_groups = Counter(combos).most_common()
    for g in color_groups:
        print(int(g[0][0]), 'Greens +', int(3-(g[0][0]+g[0][1])),
              'Yellows +', int(g[0][1]),'Reds ==>', g[0][2],':',g[1])
    
    


### Problem complexity

print(solver.NumVariables(), 'decision variables')
print(solver.NumConstraints(), 'constraints')






    
