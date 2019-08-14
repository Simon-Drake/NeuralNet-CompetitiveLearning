-------------------------
Use of competitiveL.py and replication of results
-------------------------

**First you need to unpack letter_data.zip IN the current directory (not a subdirectory)**

COMMANDS:

To run core algorithm run:
python3 competitiveL.py learning_rate max_iterations

To run extension 1.1 run:
python3 competitiveL.py -w learning_rate max_iterations min_iterations_before_check step_check threshold

To run extension 1.2 run:
python3 competitiveL.py -c learning_rate max_iterations min_iterations_before_check step_check -threshold

**Example : python3 competitiveL.py 0.008 50000 30000 10000 -0.91**

**IMPORTANT** The threshold for the Pearson product-moment correlation coefficient needs to be given as a negative value in order to
use the same logical operation and for other efficiency reasons it was better to work with the correlations in the negative. 

If no arguments are provided defaults are used equivalent to:
python3 competitiveL.py 0.05 20000

To print the help message run: 
python3 competitiveL.py -h

REQUIREMENTS:

The program uses numpy and matplotlib, in order for it to function correctly you need to run it on a system with these libraries installed. 
Please see imports for more information on use of external libraries. 

IMAGES AND PICKLE FILES:

The program uses pickle to serialise the training data in order for consecutive runs to finish quickly. On the first run the program will
make a "data.pkl" file in the current directory. 

**IMPORTANT** If the number of outputs are changed (e.g. to replicate appendix F) "data.pkl" NEEDS to be deleted. The range of printprototypes
needs to be changed (e.g. from 2,5 to 10,5)

Images are saved in the current directory in the format:

"max_iterations+learning_rate prototypes step threshold.png" OR
"max_iterations+learning_rate prototypes 0 0.png" for the core algorithm

If you want to print out the final prototypes uncomment the lines 220 - 225

To see how the dws change and plot them you need to uncomment the right code at the bottom of the file. 

TABLES AND REPLICATION OF RESULTS:

To replicate the results for tables 1 and 2 you need to uncomment the right lines at the bottom of the file and the results will be printed to results.txt in the current directory. 

The results for tables 3 and 4 are written in the terminal.

To make things easier if you are running on Linux I have provided a bash shell script to run the commands for tables 1-4, namely "compL.sh"


