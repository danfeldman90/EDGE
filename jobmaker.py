import itertools 
import numpy as np
import EDGE as edge
from astropy.io import ascii

'''
jobmaker.py

Script that uses job_file_create to produce a file with all the parameters in the grid and creates the job files
themselves.

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
Written by Connor Robinson 4/29/16

'''

#Where you want the parameter file and the jobfiles to be placed
#Also must be the location of the sample job file
gridpath = '/Users/Connor/Desktop/Research/diad/test/'

#Tag that you can add to make the parameter file identifiable for a given run
#Can leave it blank if you don't care. 
paramfiletag = 'testgrid'

#What number to start counting from, must be an integer 
jobnumstart = 1

#Define parameters to feed into file, must be filled with at least 1 value
#Want to check that the values for amaxs and epsilon are possible in the sample job file

#Stellar parameters taken from Espaillat 2011 + McJunkin 2014
mstar   = [1.3]  # Espaillat 2011 NEW 
tstar   = [4730]  # Espaillat NEW
rstar   = [1.6] # Espaillat 2011 NEW
dist    = [140] # Espaillat 2011 NEW
mdot    = [3.3e-9] # Espaillat 2011 NEW

mdotstar= [1e-9] #VERY IMPORTANT! CODE BELOW MODIFED NOT TO USE THIS VALUE AND ASSUMES MDOT = MDOTSTAR

amaxs   = [0.25, 10]
epsilon = [0.1, 1.0, 0.001]
alpha   = [1e-2]
rdisk   = [30]

temp    = [1400]
altinh  = [1]

lamaxb  = ['1mm']
mui     = [0.5]
tshock  = [8000]

fracolive = [.65]
fracpyrox = [.3]
fracforst = [0]
fracent   = [0.05]

#No need to add an underscore/jobnumber, the script will do that for you.
labelend = 'test'

#***********************************************
#Unlikly you need to change anything below here.
#***********************************************


#Open up a file and print the parameter names
f = open(gridpath+paramfiletag+'job_params.txt', 'w') 
f.writelines('Job Number, amaxs, epsilon, mstar, tstar, rstar, dist, mdot, mdotstar, tshock, alpha, mui, rdisk, temp, altinh, fracolive, fracpyrox, fracforst, fracent, lamaxb \n') 

#Write each iteration as a row in the table


#NOTE: CURRENTLY MODIFIED TO WRITE THE SAME VALUE FOR MDOTSTAR AS MDOT!!!!!!!!


for ind, values in enumerate(itertools.product(amaxs, epsilon, mstar, tstar, rstar, dist, mdot, mdot, tshock, alpha, mui, rdisk, temp, altinh, fracolive, fracpyrox, fracforst, fracent, lamaxb)):
    f.writelines(str(ind+jobnumstart)+', '+ str(values)[1:-1]+ '\n')
f.close()

#Open up the table
table = ascii.read(gridpath+paramfiletag+'job_params.txt') 

#Create the jobfiles using edge.job_file_create
for i in range(len(table)):
    label = labelend+'_'+str(i+jobnumstart).zfill(3)
    
    mdotstarval = table['mdot'][i]
    
    edge.job_file_create(i+jobnumstart, gridpath, \
    amaxs     = table['amaxs'][i],\
    epsilon   = table['epsilon'][i],\
    mstar     = table['mstar'][i], \
    tstar     = table['tstar'][i], \
    rstar     = table['rstar'][i], \
    dist      = table['dist'][i], \
    mdot      = table['mdot'][i], \
    mdotstar  = table['mdotstar'][i],\
    tshock    = table['tshock'][i], \
    alpha     = table['alpha'][i], \
    mui       = table['mui'][i], \
    rdisk     = table['rdisk'][i], \
    temp      = table['temp'][i], \
    altinh    = table['altinh'][i],\
    fracolive = table['fracolive'][i], \
    fracpyrox = table['fracpyrox'][i], \
    fracforst = table['fracforst'][i], \
    fracent = table['fracent'][i], \
    lamaxb    = table['lamaxb'][i].split("'")[1],\
    labelend  = label)
    
