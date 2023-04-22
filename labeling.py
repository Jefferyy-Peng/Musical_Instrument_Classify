import glob, os

path = '/Users/amadeus/Downloads/archive/Train_submission/Train_submission'

for infile in sorted(glob.glob(os.path.join(path, f'*.wav'))):
    print("Current File Being Processed is: " + infile)
print("a")