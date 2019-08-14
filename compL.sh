
# # For the Table1
# arr=(5000 10000 20000 40000 80000)

# for i in "${arr[@]}"
# do
#     python3 competitiveL.py 0.05 $i
# done

# # For Table 2
# arr=(0.5 0.05 0.005 0.0005 0.0005)

# for i in "${arr[@]}"
# do
#     python3 competitiveL.py $i 20000
# done

# # For Table 3
# arr=(-0.81 -0.83 -0.85 -0.87 -0.89)
# for i in "${arr[@]}"
# do 
#     python3 competitiveL.py -c 0.008 60000 20000 2000 $i
# done

# For Table 4
arr=(0.11 0.12 0.13 0.14 0.15)
for i in "${arr[@]}"
do 
    python3 competitiveL.py -w 0.008 60000 20000 2000 $i
done