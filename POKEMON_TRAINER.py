#
#
# 10/29/19
#
#
import numpy as np
import pandas as pd
import time
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches


#timer for tracking length of runs
program_starts = time.time()

#local data file for reading
datafile = './pokemon_trainer_application_data.csv'

#grab certain columns, not whole file
df = pd.read_csv(datafile, na_values = 'NaN', usecols=['PokemonTrainerClass','PokemonWorldRegion','hired','TotalYearsOfExp','UndergradSchoolGPA','RecentTrainingExperience1Pokemon','HighSchoolGPA','InternalEmployee','ReferenceRelationship1'])

#replace nulls with 'NaN'
df['hired'].fillna("NaN", inplace = True)
df['UndergradSchoolGPA'].fillna("NaN", inplace = True)

#lists for trainerclass and regions. Used for permutation through data set 
trainerclass = ['Curmudgeon','Doctor','Dragon Tamer','Engineer','Nurse','Pokemon Ranger','Scientist','Skier']
worldregion = ['Kanto','Johto','Hoenn','Sinnoh','Unova','Kalos','Alola','Sevii Islands']


#function to step through data collecting and counting hires vs. non-hires per each trainer-region combinations.
#This function is brute force and takes about 260 seconds but gets all combos (64 total).
def count_teams(trr,wrr):
    cntr = 0
    cnt_hire = 0 
    cnt_gpa = 0
    gpa = []
    for i in range(len(df)):
        if df['hired'][i] != 'NaN':
            if df['PokemonTrainerClass'][i] == trr and df['PokemonWorldRegion'][i] == wrr:
                cntr = cntr + 1
                if int(df['hired'][i]) == 1:
                    cnt_hire = cnt_hire + 1
                if str(df['UndergradSchoolGPA'][i]) != 'NaN':
                    cnt_gpa = cnt_gpa + 1
                    gpa.append(float(df['UndergradSchoolGPA'][i]))

    if cnt_gpa == 0:
        return cntr,cnt_hire,trr,wrr,0,0
    else:
        return cntr,cnt_hire,trr,wrr,cnt_gpa,np.mean(gpa)

# key to turn function on or off (key=5 --> on)
# When on, the above function is looped based on the permutation of trainer-region
# and the results are written to a log file called './most_competitive.csv'. This
# log file contains the columns: TrainerClass, WorldRegion, Total (applicants), Hired, NotHired, %HiredRate'
key=6
if key == 5:
    cnter = 0
    m = open('most_competitive.csv','w')
    m.write('TrainerClass,WorldRegion,Total (applicants),Hired,NotHired,%HiredRate,CountGPA,AverageUndergradGPA'+'\n')
    for trr in trainerclass:
        for wrr in worldregion:
            f,cnt_hired,tr,wr,cn_gpa,g_pa = count_teams(trr,wrr)
            if int(f) == 0:
                print(tr,wr,f,cnt_hired,f-cnt_hired,'{:0.2f}'.format(0),cn_gpa,'{:0.2f}'.format(g_pa))
                m.write(tr+','+wr+','+str(f)+','+str(cnt_hired)+','+str(f-cnt_hired)+',0.00'+','+str(cn_gpa)+','+str('{:0.2f}'.format(g_pa))+'\n')
            else:
                print(tr,wr,f,cnt_hired,f-cnt_hired,'{:0.2f}'.format((float(cnt_hired)/float(f))*100.0),cn_gpa,'{:0.2f}'.format(g_pa))
                m.write(tr+','+wr+','+str(f)+','+str(cnt_hired)+','+str(f-cnt_hired)+','+str('{:0.2f}'.format((float(cnt_hired)/float(f))*100.0))+','+str(cn_gpa)+','+str('{:0.2f}'.format(g_pa))+'\n')


    m.close()


#read the newly created log file
dff = pd.read_csv('./most_competitive.csv')
print('')

#print out some results
print('Looking at the hired rate % (total hired / total applied) with all \'NAN\'s weeded out, we see that the lowest hired rate is 0.42% described below:' )
print(dff[dff['%HiredRate'] > 0.0].sort_values(by=['%HiredRate']).head())
print('')
print('If we are looking at the the hired rate % for those trainers that had applicants but did not hire any then we have a 0% hired rate for 155 applicants making Skier Alola the most competitive combination of trainer and region to get hired in to:')
print(dff[dff['Hired'] == 0].sort_values(by=['Total (applicants)'],ascending=False).head())


print('')
#Drop all rows with one NaN keeping a full set for Naive Bayes model below
dfff = df.dropna()
dfff = dfff[dfff['hired'] != 'NaN']
dnn = dfff[dfff['UndergradSchoolGPA'] != 'NaN']

#3D plot with labels
key2 = 4
if key2 == 3:
    colors1 = {'Curmudgeon':'rd','Doctor':'gr','Dragon Tamer':'bl','Engineer':'bk',
          'Nurse':'or','Pokemon Ranger':'yw','Scientist':'cyan','Skier':'sil'}
    classes = ['Curmudgeon','Doctor','Dragon Tamer','Engineer',
          'Nurse','Pokemon Ranger','Scientist','Skier']

    class_colours = ['red','green','blue','black','orange','yellow','cyan','silver']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    lo = ax.scatter(dff['AverageUndergradGPA'],dff['%HiredRate'],c=dff['TrainerClass'].apply(lambda x: colors[x]),label=colors1)
    ax.set_xlabel('AverageUnderGradGPA')
    ax.set_ylabel('%HiredRate')
    recs = []
    for i in range(0,len(class_colours)):
        recs.append(mpatches.Circle((0,0),1,fc=class_colours[i]))
    plt.legend(recs,classes,loc=2)
    plt.show()


#naive Bayes model------------------------------------------------------------------------
#X = dnn[['UndergradSchoolGPA','HighSchoolGPA','TotalYearsOfExp']].copy()
m = dfff[dfff['hired'] != 'NaN']
n = m[dfff['UndergradSchoolGPA'] != 'NaN']
X = n[['UndergradSchoolGPA','HighSchoolGPA','TotalYearsOfExp']].copy()
y = n['hired'].copy()


#change the object type over to ints for target data
y = y.astype('float')

#create a dataframe for predicting
dat = {'UndergradSchoolGPA':[2.5], 'HighSchoolGPA':[1.5], 'TotalYearsOfExp':[1.0]}
X_predict = pd.DataFrame(data=dat)


#create the train-test sets using 75-25
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)

#print(X_predict.info())
X_test = X_test.astype('float')
X_train = X_train.astype('float')


#Standardise and scale from 0-1
scaler = StandardScaler()
train_scaled = scaler.fit_transform(X_train)
test_scaled = scaler.transform(X_test)



#Fit the model
model = GaussianNB(priors=None)
model.fit(train_scaled, y_train)


predicted = model.predict(X_predict)

#print the accuracy of the model
print('')
print('Naive Bayes Results:')
print('Accuracy of training data set: ',accuracy_score(y_train, model.predict(train_scaled)))
print('Accuracy of test data set: ', accuracy_score(y_test, model.predict(test_scaled)))
print('Prediction for: ',dat, predicted)

#time
program_ends = time.time()
print('It took: ', '{:0.2f}'.format(program_ends-program_starts), ' seconds')
print('Finished...')
