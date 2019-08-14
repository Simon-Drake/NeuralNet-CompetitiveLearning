# ------------------------------------
# Author: Simon Drake
# Date finalised: 26/03/2019
# All of this code was written by myself except the *initial*
# normalisation of the weights the random selection of the 
# input and which was taken from lab5.py.
# ------------------------------------

import numpy as np
import math, os, numpy.matlib, pickle, sys, getopt
import matplotlib.pyplot as plt
from random import randint

# CommandLine class is used to provide initial conditions.
class CommandLine:
    def __init__(self):
            opts, args = getopt.getopt(sys.argv[1:], "-w-c-h")
            opts = dict(opts)

            # Defaults used if no arguments are provided
            if len(args) == 0 and "-h" not in opts:
                print("** no arg - defaults used ***")
                args.append(0.05)
                args.append(20000)

            # Help message
            if "-h" in opts:
                print("If you do not want to use extensions, please call: ")
                print("python3 competitiveL.py learning_rate max_iterations", end="\n\n")
                print("If you want to use extension 1.1, please call: ")
                print("python3 competitiveL.py -w learning_rate max_iterations min_iterations step_check threshold", end="\n\n")
                print("If you want to use extension 1.2, please call: ")
                print("python3 competitiveL.py -c learning_rate max_iterations min_iterations step_check threshold", end="\n\n")
                print("For more information please read README.txt file")
                sys.exit()

            # Display the parameters used
            if "-w" in opts or "-c" in opts:
                print("Learning rate: {}".format(args[0]))
                print("Maximum iterations: {}".format(args[1]))
                print("Minimum iterations before check: {}".format(args[2]))
                print("Step between each check: {}".format(args[3]))
                print("Threshold: {}".format(args[4]))
            else:
                print("Learning rate: {}".format(args[0]))
                print("Maximum iterations: {}".format(args[1]))


            self.opts = opts
            self.args = [float(i) for i in args]

# Data class, used to pickle data as reading in from csv files takes time
class Data:
    def __init__(self, outputs):

        # Read in files
        train = np.genfromtxt ('letters.csv', delimiter=",")
        self.trainlabels = np.genfromtxt ('letterslabels.csv', delimiter=",")

        # Number of pixels and number of training instances (7744, 7000)
        [n,m]  = np.shape(train) 
        self.m = m

        # Normalise the training data 
        pix = int(math.sqrt(n))
        normT = np.sqrt(np.diag(train.dot(train.T)))
        normT = normT.reshape(pix*pix,-1)         
        self.train = np.divide(train, normT, out=np.zeros_like(train), where=normT!=0)

        # Normalise using repmat or broadcasting (one is commented out)
        W = np.random.rand(outputs,n)
        normW = np.sqrt(np.diag(W.dot(W.T)))
        normW = normW.reshape(outputs,-1)   
        self.W = W / np.matlib.repmat(normW.T,n,1).T   

# Returns a matrix of the magnitudes of the difference in weights for all weights
def get_distances(W):

    # For all weights: 
    # Get the magnitude of the distance between weights
    distances = list()
    for i, item in enumerate(W):
        distances.append(list())
        for j in W:
            distance = np.linalg.norm(item - j)
            distances[i].append(distance)

    X = np.stack((distances), axis = 0)
    return X

# Function used to print out prototypes ~ Visualisation aid 
def printprototypes(W, max_it , learn_rate, step, threshold):
    f, axarr = plt.subplots(2, 5, figsize=(10,10))
    plt.tight_layout()
    count = 0
    for i in range(0, 2):
        for j in range(0, 5):
            axarr[i][j].imshow(W[count,:].reshape((88,88), order = 'F'),interpolation = 'nearest', cmap='inferno')
            count += 1
    plt.savefig("{} prototypes {}__ {}.png".format(max_it + learn_rate, step, threshold))



if __name__ == "__main__":

    # Initialise command line containing parameters
    # and number of output digits. 
    cl = CommandLine()
    outputs =  10

    # If the pickle file has already been created read in the object otherwise instantiate it. 
    # IMPORTANT: if the number of outputs have been changed data.pkl needs to be deleted and reinitialised. 
    if "data.pkl" in os.listdir():
        with open("data.pkl", 'rb') as f:
            dt = pickle.load(f)
    else:
        dt = Data(outputs)
        with open("data.pkl", 'wb') as f:
            pickle.dump(dt, f)


    # Initialise max iteration and learn rate
    tmax = int(cl.args[1])
    learn_rate = cl.args[0] 

    # If extension used initialise min iteration, step check and threshold
    # For details please read paper or README.txt
    if "-w" in cl.opts or "-c" in cl.opts:
        tmin = int(cl.args[2])
        step = int(cl.args[3])
        threshold = cl.args[4]  

    # Initialise other 
    too_close = True
    t = 0
    counter = np.zeros((1,outputs))   

    # Initialise the dictionary containing the length of change in weight vectors. 
    len_dw = dict()
    for i in range(1, outputs+1):
        len_dw[i] = list()

    while t < tmax and too_close:

        # Get a randomly generated index in the input range
        i = math.ceil(dt.m * np.random.rand())-1
        x = dt.train[:,i]

        # Get output of all neurons, and find the winner's index
        h = dt.W.dot(x)                   
        h = h.reshape(h.shape[0],-1)
        output = np.max(h)                     
        k = np.argmax(h)             

        # Calculate the change in weight for the winner and update it
        dw = learn_rate * (x.T - dt.W[k,:])
        dt.W[k,:] = dt.W[k,:] + dw  

        # Update current iteration, the counter for the winner and add the change in weights vector to our dict (for evaluation purposes).
        t+=1
        counter[0,k] += 1
        len_dw[k+1].append(np.linalg.norm(dw))

        # # Code needed for Appendix E
        # if t == 2000:
        #     printprototypes(dt.W, cl.args[0], cl.args[1], 0, 1)

        if "-w" in cl.opts or "-c" in cl.opts:

            if t > tmin and t % step == 0:
                changed_weights = set()
                if "-w" in cl.opts:
                    X = get_distances(dt.W) 
                else:
                    X = np.corrcoef(dt.W)*-1
                for i, item in enumerate(X):
                    for j, item_j in enumerate(item):
                        # If we are comparing the same weights, or the weight has already been changed, move on. 
                        if i == j or (j in changed_weights) or (i in changed_weights):
                            continue
                        else:
                            # If the distance is lower than the distance threshold required:
                            if item_j < threshold:
                                # Create a new weight vecotr and normalise it
                                new_weight = np.random.rand(1, np.shape(x)[0])
                                new_weight = new_weight/np.linalg.norm(new_weight)

                                # Uncomment for Appendix G
                                # printprototypes(dt.W, cl.args[0], cl.args[1], i, j)

                                # Replace the old weight vector
                                dt.W[j] = new_weight
                                changed_weights.add(j)

                                # Uncomment for Appendix G
                                # printprototypes(dt.W, cl.args[0], cl.args[1], i, j)

                # If no weights have been changes, distances are good and we can exit the while loop
                if len(changed_weights) == 0 :
                    print("Exiting at {} iterations".format(t), end="\n\n")
                    too_close = False

            if t == tmax-1 :
                print("Exiting at max iterations", end="\n\n")
                

    # Print final prototype
    if "-w" in cl.opts or "-c" in cl.opts:
        printprototypes(dt.W, cl.args[0], cl.args[1], step, threshold)
    else:
        printprototypes(dt.W, cl.args[0], cl.args[1], 0, 0)


    ################# Plot the magnitudes of the difference in weight vectors #################

    # f, axarr = plt.subplots(5, 2, figsize=(10,10))
    # plt.tight_layout()
    # count = 1
    # for i in range(0, 5):
    #     for j in range(0, 2):
    #         axarr[i][j].plot(range(0, len(len_dw[count])), len_dw[count])
    #         count += 1
    # plt.savefig("len_dw{}.png".format(cl.args[0] + cl.args[1]))

    ################# Write results for Table 1 and Table 2 #################
    
    # results = list()
    # with open("results.txt", "r") as f:
    #     results = f.readlines()

    # s = "\nIterations: {}\nLearning rate: {}\nCounter : ".format(int(cl.args[1]), cl.args[0])
    # for i in counter[0]:
    #     s += " " + str(int(i))
    # variance = str(int(np.var(counter)))
    # results.append(s + "\nVariance:" + variance+ "\n")

    # with open("results.txt", "w") as f:
    #     for i in results:
    #         f.write(i)
