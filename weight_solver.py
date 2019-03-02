# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 17:20:40 2019

MIP optimization for weights and thresholds for the health score

@author: e6on6gv
"""

from ortools.linear_solver import pywraplp as lp
import csv
from collections import Counter


def color_rules(solver, color_vars):
    '''
    Creates constraints, enforcing rules of combining original scores' colors
    into total score colors.
    
    solver - ortools.Solver object
    color_vars - dict with original scores' color variables
                    with a structure:
                    color_vars = {'retention':{'green':[ortools.Variable,...],
                                               'red':[ortools.Variable,...]},
                                  'ar':{'green':[ortools.Variable,...],
                                        'red':[ortools.Variable,...]},
                                  'trend':{'green':[ortools.Variable,...],
                                           'red':[ortools.Variable,...]},
                                  'total':{'green':[ortools.Variable,...],
                                           'red':[ortools.Variable,...]}}
    
    Returns a list with a number of elements equal to the number of customers.
    Each element is a dict containing:
        'deltas': list of 12 delta-variables
        'delta_constr': list of 12 delta-constraints
        'color_constr': list of 23 color rules' constraints
    '''
    
    def add_delta_constr(delta, idx, color, coef):
        ''' helper function to add constraint for a specific delta
            
            delta - delta variable
            idx - customer index
            coef - coefficient for delta
        '''
        constr = solver.Constraint(0, 3)
        for k in ['retention', 'ar', 'trend']:
            constr.SetCoefficient(color_vars[k][color][idx], 1)
        constr.SetCoefficient(delta, coef)
        return(constr)
        
    def add_color_constr(delta_list, constr_list, idx):
        ''' helper function to add a color constraint
            
            delta_list - list of included deltas
            constr_list - list of tuples; each tuple is (lower_bound, coef for G, coef for R)
            idx - customer index
        '''
        constrs = []
        for c in constr_list:
            constr = solver.Constraint(c[0], solver.infinity())
            constr.SetCoefficient(color_vars['total']['green'][idx], c[1])
            constr.SetCoefficient(color_vars['total']['red'][idx], c[2])
            for d in delta_list:
                constr.SetCoefficient(d, 1)
            constrs.append(constr)
        return(constrs)
        
        
    row_num = len(color_vars['total']['green'])
    color_logic = [{}] * row_num
    for r in range(row_num):
        # create deltas - indicator variables and their constraints
        delta = [0] * 12
        delta_constr = [0] * 12
        
        delta[0] = solver.IntVar(0,1,'delta1_' + str(r)) 
        delta_constr[0] = add_delta_constr(delta[0],r,'green',1)       
        delta[1] = solver.IntVar(0,1,'delta2_' + str(r)) 
        delta_constr[1] = add_delta_constr(delta[1],r,'green',2)       
        delta[2] = solver.IntVar(0,1,'delta3_' + str(r)) 
        delta_constr[2] = add_delta_constr(delta[2],r,'green',3)       
        delta[3] = solver.IntVar(0,1,'delta4_' + str(r)) 
        delta_constr[3] = add_delta_constr(delta[3],r,'green',-1)     
        delta[4] = solver.IntVar(0,1,'delta5_' + str(r)) 
        delta_constr[4] = add_delta_constr(delta[4],r,'green',-2)      
        delta[5] = solver.IntVar(0,1,'delta6_' + str(r)) 
        delta_constr[5] = add_delta_constr(delta[5],r,'green',-3)    
        delta[6] = solver.IntVar(0,1,'delta7_' + str(r)) 
        delta_constr[6] = add_delta_constr(delta[6],r,'red',1)      
        delta[7] = solver.IntVar(0,1,'delta8_' + str(r)) 
        delta_constr[7] = add_delta_constr(delta[7],r,'red',2)      
        delta[8] = solver.IntVar(0,1,'delta9_' + str(r)) 
        delta_constr[8] = add_delta_constr(delta[8],r,'red',3)      
        delta[9] = solver.IntVar(0,1,'delta10_' + str(r)) 
        delta_constr[9] = add_delta_constr(delta[9],r,'red',-1)     
        delta[10] = solver.IntVar(0,1,'delta11_' + str(r)) 
        delta_constr[10] = add_delta_constr(delta[10],r,'red',-2)      
        delta[11] = solver.IntVar(0,1,'delta12_' + str(r)) 
        delta_constr[11] = add_delta_constr(delta[11],r,'red',-3)
        
        color_logic[r]['deltas'] = delta
        color_logic[r]['delta_constr'] = delta_constr

        # create color rules' constraints
        color_constr = []
        
        # Rule 1
        color_constr += add_color_constr([delta[0]], [(1, 1, 0), (0, 0, -1)], r)
        
        # Rule 2
        color_constr += add_color_constr([delta[1], delta[5], delta[9]], [(1, 1, 0), (0, 0, -1)], r)
       
        # Rule 3
        color_constr += add_color_constr([delta[2], delta[4], delta[9]], [(0, -1, 0), (0, 0, -1)], r)
        
        # Rule 4
        color_constr += add_color_constr([delta[3], delta[9]], [(0, -1, 0), (0, 0, -1)], r)
        
        # Rule 5
        color_constr += add_color_constr([delta[1], delta[5], delta[8], delta[10]],
                                         [(0, 1, -1), (-1, -1, -1), (0, 0, -1)], r)
        
        # Rule 6
        color_constr += add_color_constr([delta[2], delta[4], delta[8], delta[10]],
                                         [(0, -1, 0), (0, -1, 1), (-1, -1, -1)], r)
        
        # Rule 7
        color_constr += add_color_constr([delta[2], delta[4], delta[7], delta[11]],
                                         [(0, -1, 0), (0, -1, 1), (-1, -1, -1)], r)
        
        # Rule 8
        color_constr += add_color_constr([delta[3], delta[8], delta[10]],
                                         [(0, -1, 0), (0, 0, -1)], r)
        
        # Rule 9
        color_constr += add_color_constr([delta[3], delta[7], delta[11]],
                                         [(0, -1, 0), (1, 0, 1)], r)
        
        # Rule 10
        color_constr += add_color_constr([delta[6]],[(0, -1, 0), (1, 0, 1)], r)
        
        
        color_logic[r]['color_constr'] = color_constr
    return(color_logic)
        


with open("C:\\Users\\e6on6gv\\Documents\\Print Attrition\\Weights Optimization\\three_scores.csv") as f:
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

#target number of greens or reds
target = 0.2 * row_num


solver = lp.Solver("HealthScore", lp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
solver.set_time_limit(100000000)
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
for k in ['retention', 'ar', 'trend']:
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
color_logic = color_rules(solver, color_vars)         


      
# Adding constraints and objective for greens/reds to be below target
# but as close to it as possible
target_green_constr = solver.Constraint(0.2 * target, target) 
target_red_constr = solver.Constraint(0.2 * target, target)            
for i in range(row_num):
    target_green_constr.SetCoefficient(color_vars['total']['green'][i], 1)
    target_red_constr.SetCoefficient(color_vars['total']['red'][i], 1)
    objective.SetCoefficient(color_vars['total']['green'][i], 1)
    objective.SetCoefficient(color_vars['total']['red'][i], 1) 
    
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
            colors = [d.solution_value() for d in t]
            print('Percent of', c, 'for', k, ':', sum(colors)/len(colors))

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
                greens.append(score)
            elif not isGreen and isRed:
                reds.append(score)
            elif not isGreen and not isRed:
                yellows.append(score)
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
        combos.append((greens, reds, result_color))
    color_groups = Counter(combos).most_common()
    for g in color_groups:
        print(int(g[0][0]), 'Greens +', int(3-(g[0][0]+g[0][1])),
              'Yellows +', int(g[0][1]),'Reds ==>', g[0][2],':',g[1])
    
    
    
    
    
#    ggr2r = []
#    for i in range(row_num):
#        greens = 0
#        reds = 0
#        for k in ['retention', 'ar', 'trend']:
#           greens += color_vars[k]['green'][i].solution_value()
#           reds += color_vars[k]['red'][i].solution_value()
#        if greens == 2 and reds == 1:
#            ggr2r.append(i)
#    
#    
#    for k in ['retention', 'ar', 'trend', 'total']:
#        for c in ['green', 'red']:
#            print(k, c, color_vars[k][c][0].solution_value())
#    
#    
#    if color_vars[k][c][0].solution_value():
#        print(1)
    
    
    
    
    
    
    
    
    
    
    
    
    