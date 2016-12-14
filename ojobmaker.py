import itertools 
import numpy as np
import EDGE as edge
from astropy.io import ascii

'''
ojobmaker.py

Script that uses job_optthin_create to produce a file with all the parameters in the grid and create optically thin dust job files.

HOW TO USE THIS SCRIPT:
    1) Change the gridpath to be where you want the jobfiles to be placed. NOTE: This should also be the location of the jobsample file.
    
    2) Change the paramfiletag to an insightful name that will help you identify the grid of models you ran three months from now.
    
    3) Change the parameters in the brackets below to whatever set of parameters you want to run.
       The parameters should be surrounded by []'s and be separated by a commas (A list)
    
    4) Change the labelname to the name of the object, e.g. 'gmauriga'
    
    5) Run the script


NOTES:
    Currently set up with a zero padding of 3 (e.g. job001 instead of job0001)

MODIFICATION HISTORY
Written by Connor Robinson 6/20/16

'''

#Where you want the parameter file and the jobfiles to be placed
#Also must be the location of the sample job file
gridpath = '/Users/Connor/Desktop/Research/amanda/'

#Tag that you can add to make the parameter file identifiable for a given run
#Can leave it blank if you don't care. 
paramfiletag = 'test'

#What number to start counting from, must be an integer 
jobnumstart = 1

#Define parameters to feed into file, must be filled with at least 1 value
#Want to check that the values for amaxs and epsilon are possible in the sample job file

amax      = [1.0]
tstar     = [5035]
rstar     = [1.38]
dist      = [86]
mui       = [.82]
rout      = [170.0]
rin       = [80]
tau       = [0.035]
power     = [0]
fudgeorg  = [1]
fudgetroi = [0.000001]
fracsil   = [0.9]
fracent   = [0.000001]
fracforst = [0.000001]
fracamc   = [0.001]

#No need to add an underscore/jobnumber, the script will do that for you.
labelend = 'MPMus'

#***********************************************
#Unlikly you need to change anything below here.
#***********************************************

#Open up a file and print the parameter names
f = open(gridpath+paramfiletag+'optthin_job_params.txt', 'w') 
f.writelines('Job Number, amax, tstar, rstar, dist, mui, rout, rin, tau, power, fudgeorg, fudgetroi, fracsil, fracent, fracforst, fracamc \n')

#Write each iteration as a row in the table
for ind, values in enumerate(itertools.product(amax, tstar, rstar, dist, mui, rout, rin, tau, power, fudgeorg, fudgetroi, fracsil, fracent, fracforst, fracamc)):
    f.writelines(str(ind+jobnumstart)+', '+ str(values)[1:-1]+ '\n')
f.close()

#Open up the table
table = ascii.read(gridpath+paramfiletag+'optthin_job_params.txt') 

#Create the jobfiles using edge.job_file_create
for i in range(len(table)):
    label = labelend+'_'+edge.numCheck(i+jobnumstart)
    
    edge.job_optthin_create(i+jobnumstart, gridpath, \
    amax      = table['amax'][i], \
    tstar     = table['tstar'][i], \
    rstar     = table['rstar'][i], \
    dist      = table['dist'][i], \
    mui       = table['mui'][i], \
    rout     = table['rout'][i], \
    rin       = table['rin'][i], \
    tau       = table['tau'][i], \
    power     = table['power'][i], \
    fudgeorg  = table['fudgeorg'][i], \
    fudgetroi = table['fudgetroi'][i], \
    fracsil   = table['fracsil'][i], \
    fracent   = table['fracent'][i], \
    fracforst = table['fracforst'][i], \
    fracamc   = table['fracamc'][i], \
    labelend  = label)
    
