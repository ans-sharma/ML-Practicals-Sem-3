# 2022-11-28 10:38:06

word_count=0
file="exe0/test.txt"
with open(file,'r') as file:
    for line in file:
        # print(line)
        word_count=(word_count + len(line.split())) 
print("Number of words:",word_count)