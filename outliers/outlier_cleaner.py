#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []
    ### your code goes here
    errors = predictions - net_worths
    res = sorted(enumerate(abs(errors)), key=lambda item:item[1])
    for x in res[:80]:
        cleaned_data.append(tuple([ages[x[0]], net_worths[x[0]], errors[x[0]]]))

    return cleaned_data

