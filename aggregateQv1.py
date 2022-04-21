import numpy as np
from scipy.interpolate import lagrange

# every calculation is based in mod
mod = 3017
inverseModArr = [1, 6, 4, 3, 9, 2, 8, 7, 5, 10]
#note:
# y = pow(3, -1, 11)
# y is the inverse of 3 mod 11

# functions---------------------------------

# plug in an x to calculate polynomials in terms of mod n in vector 
def computePolyVec(polyVec, x, size):
    rarr = [0 for i in range (size)]
    for i, val in enumerate(polyVec):
        rarr[i] = polyVec[i](x) % mod
    return rarr

# input x value to batch index of queries in terms of mod n
def computeBatchInput(batch, x):
    rarr = [[0 for col in range(14)] for row in range(4)]
    for i, val in enumerate(batch):
        for j, val2 in enumerate(batch[i]):
            rarr[i][j] = batch[i][j](x) % mod
    return rarr

# converts vector to mod n
def convertMod(resArr):
    rarr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i, val in enumerate(resArr):
        rarr[i] = resArr[i] % mod
    return rarr

# converts result vector to mod n
def convertMod2(resArr):
    rarr = [0, 0, 0, 0, 0, 0, 0]
    for i, val in enumerate(resArr):
        rarr[i] = resArr[i] % mod
    return rarr

# user functions:

# 1 degree lagrange interpolating polynomial
def create1DPoly(sbv1, sbv2, xes):
    res=[0, 0, 0, 0]
    for i in range(4):
        mul = [0, 0]
        mul[0] = pow((xes[0] - xes[1])%mod, -1, mod)
        mul[1] = pow((xes[1] - xes[0])%mod, -1, mod) 
        tempPoly1 = [(sbv1[i] * mul[0])%mod, (-1*(sbv1[i] *((xes[1] * mul[0])%mod))%mod)%mod]
        tempPoly2 = [(sbv2[i] * mul[1])%mod, (-1*(sbv2[i] *((xes[0] * mul[1])%mod))%mod)%mod]
        res[i] = np.poly1d([(tempPoly1[0] + tempPoly2[0])%mod, (tempPoly1[1] + tempPoly2[1])%mod])
    return res

# 2 degree lagrange interpolating polynomial
# this is to create batch index of queries
def create2DPoly(ioq1, ioq2, ioq3, xCoor):
    res = [[0 for col in range(14)] for row in range(4)]
    for j in range(0,4):
        for i in range(0,14):
            mul=[0, 0, 0]
            mul[0] = pow((((xCoor[0]-xCoor[1])%mod)*((xCoor[0]-xCoor[2])%mod))% mod, -1, mod)
            mul[1] = pow((((xCoor[1]-xCoor[0])%mod)*((xCoor[1]-xCoor[2])%mod))% mod, -1, mod)
            mul[2] = pow((((xCoor[2]-xCoor[0])%mod)*((xCoor[2]-xCoor[1])%mod))% mod, -1, mod)
            tempPoly1 = [(ioq1[j][i]*mul[0])% mod, (ioq1[j][i] * ((mul[0] * ((-1 * (xCoor[2]+xCoor[1]))%mod))%mod))%mod, (ioq1[j][i] * ((mul[0] * ((xCoor[1]*xCoor[2])%mod))%mod))%mod]
            tempPoly2 = [(ioq2[j][i]*mul[1])% mod, (ioq2[j][i] * ((mul[1] * ((-1 * (xCoor[2]+xCoor[0]))%mod))%mod))%mod, (ioq2[j][i] * ((mul[1] * ((xCoor[0]*xCoor[2])%mod))%mod))%mod]
            tempPoly3 = [(ioq3[j][i]*mul[2])% mod, (ioq3[j][i] * ((mul[2] * ((-1 * (xCoor[1]+xCoor[0]))%mod))%mod))%mod, (ioq3[j][i] * ((mul[2] * ((xCoor[0]*xCoor[1])%mod))%mod))%mod]
            res[j][i] = np.poly1d([(((tempPoly1[0]+tempPoly2[0])%mod) + tempPoly3[0])%mod, (((tempPoly1[1]+tempPoly2[1])%mod) + tempPoly3[1])%mod, (((tempPoly1[2]+tempPoly2[2])%mod) + tempPoly3[2])%mod])
    return res

# 3 degree lagrange interpolating polynomial
def create3DPoly(s1, s2, s3, s4, x):
    res=[0, 0, 0, 0, 0, 0, 0]
    for i in range(7):
        mul = [0, 0, 0, 0]
        mul[0] = pow(((((x[0] - x[1])%mod * ((x[0] - x[2])%mod))%mod) * ((x[0] - x[3])%mod))% mod, -1, mod)
        mul[1] = pow(((((x[1] - x[0])%mod * ((x[1] - x[2])%mod))%mod) * ((x[1] - x[3])%mod))% mod, -1, mod)
        mul[2] = pow(((((x[2] - x[0])%mod * ((x[2] - x[1])%mod))%mod) * ((x[2] - x[3])%mod))% mod, -1, mod)
        mul[3] = pow(((((x[3] - x[0])%mod * ((x[3] - x[1])%mod))%mod) * ((x[3] - x[2])%mod))% mod, -1, mod)
        tempPoly1 = [(s1[i] * mul[0])%mod, (s1[i] * ((mul[0] * (((-1 * x[3])-x[2]-x[1])%mod))%mod))%mod, (s1[i] * ((mul[0] * ((((x[2]*x[3])%mod + (x[1]*x[3])%mod + (x[1]*x[2])%mod)%mod))%mod)))%mod, ((-1*(s1[i] * ((mul[0] * (x[1]*x[2]*x[3])%mod))%mod))%mod)%mod]
        tempPoly2 = [(s2[i] * mul[1])%mod, (s2[i] * ((mul[1] * (((-1 * x[3])-x[2]-x[0])%mod))%mod))%mod, (s2[i] * ((mul[1] * ((((x[2]*x[3])%mod + (x[0]*x[3])%mod + (x[0]*x[2])%mod)%mod))%mod)))%mod, ((-1*(s2[i] * ((mul[1] * (x[0]*x[2]*x[3])%mod))%mod))%mod)%mod]
        tempPoly3 = [(s3[i] * mul[2])%mod, (s3[i] * ((mul[2] * (((-1 * x[3])-x[1]-x[0])%mod))%mod))%mod, (s3[i] * ((mul[2] * ((((x[1]*x[3])%mod + (x[0]*x[3])%mod + (x[0]*x[1])%mod)%mod))%mod)))%mod, ((-1*(s3[i] * ((mul[2] * (x[0]*x[1]*x[3])%mod))%mod))%mod)%mod]
        tempPoly4 = [(s4[i] * mul[3])%mod, (s4[i] * ((mul[3] * (((-1 * x[2])-x[1]-x[0])%mod))%mod))%mod, (s4[i] * ((mul[3] * ((((x[1]*x[2])%mod + (x[0]*x[2])%mod + (x[0]*x[1])%mod)%mod))%mod)))%mod, ((-1*(s4[i] * ((mul[3] * (x[0]*x[1]*x[2])%mod))%mod))%mod)%mod]
        res[i] = np.poly1d([(tempPoly1[0]+tempPoly2[0]+tempPoly3[0]+tempPoly4[0])%mod, (tempPoly1[1]+tempPoly2[1]+tempPoly3[1]+tempPoly4[1])%mod, (tempPoly1[2]+tempPoly2[2]+tempPoly3[2]+tempPoly4[2])%mod, (tempPoly1[3]+tempPoly2[3]+tempPoly3[3]+tempPoly4[3])%mod])
    return res

# end functions-----------------------------

# What each server contains-----------------
# database
sData = [[2010, 830, 126, 446, 285, 669, 1],
        [2009, 109, 876, 234, 294, 345, 2],
        [2010, 928, 135, 458, 288, 572, 3],
        [2013, 311, 281, 391, 286, 570, 4],
        [2007, 116,	732, 445, 972, 110, 3],
        [2009, 105, 111, 448, 287, 522, 6],
        [2010, 830, 123, 450, 245, 563, 2],
        [2002, 311, 281, 494, 288, 511, 5],
        [2009, 600, 332, 20, 222, 883, 1],
        [2007, 178, 778, 655, 100, 901, 6],
        [2007, 65, 853, 733, 888, 300, 1],
        [2009, 9, 654, 554, 123, 44, 3],
        [2010, 732, 771, 555, 432, 12, 6],
        [2007, 900, 900, 221, 555, 309, 2]]

# index of queries
covid = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0]]

hypertension = [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]]

diabetes = [[0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]]

# index of queries labels
ioq_labels = [0, 1, 2]

# batch index of queries for covid, hypertension, and diabetes respecitely
batchIndServer = create2DPoly(covid, hypertension, diabetes, ioq_labels)

# end of server contains--------------------

# Start of protocol-------------------------
# printing database
print("Database:")
for i in range(0,14):
    print(sData[i])
print("\n")


# userside
# Alice wants the top 2 years of covid admissions and hypertension admissions of yolo count
# covid
#alice = [0, 1, 0, 0]
alice = [334, 821, 105, 234]
# hypertension
alice2 = [0, 1, 0, 0]

# labels for vectors above
xCoor = [0,1]
print("Alice's Vectors:")
print(alice)
print(xCoor[0])
print(alice2)
print(xCoor[1])
print("\n")

# generate polynomial vector out of alice's vector and random point vector
# this is where you use approx lagrange mod
polyVec = create1DPoly(alice, alice2, xCoor)

print("Alice's generate polynomials:")
print(polyVec)
print("\n")

# plug x=3 and x=4 into Alice's polynomial vector to send to the 2 servers
aliceQ1 = computePolyVec(polyVec, 4, 4)
aliceQ2 = computePolyVec(polyVec, 5, 4)
aliceQ3 = computePolyVec(polyVec, 6, 4)
aliceQ4 = computePolyVec(polyVec, 7, 4)

print("Query1 sent to Server 1 with x = 4")
print(aliceQ1)
print("Query2 sent to Server 2 with x = 5")
print(aliceQ2)
print("Query3 sent to Server 3 with x = 6")
print(aliceQ3)
print("Query4 sent to Server 4 with x = 7")
print(aliceQ3)
print("\n")

# severside
print("Covid index of queries:")
for i in range(0,4):
    print(covid[i])
print("\n")

print("Hypertension index of queries:")
for i in range(0,4):
    print(hypertension[i])
print("\n")

print("Diabetes index of queries:")
for i in range(0,4):
    print(diabetes[i])
print("\n")

print("Batch Index of queries:")
for i in range(0,4):
    print(batchIndServer[i])
print("\n")

# server plugs in x to batch index of queries
# server 1
s1_idxq = computeBatchInput(batchIndServer, 4)
# server 2
s2_idxq = computeBatchInput(batchIndServer, 5)
# server 3
s3_idxq = computeBatchInput(batchIndServer, 6)
# server 3
s4_idxq = computeBatchInput(batchIndServer, 7)

print("Server 1 batch index of queries x = 4")
for i in range(0,4):
    print(s1_idxq[i])
print("\n")
print("Server 2 batch index of quereis x = 5")
for i in range(0,4):
    print(s2_idxq[i])
print("\n")
print("Server 3 batch index of quereis x = 6")
for i in range(0,4):
    print(s3_idxq[i])
print("\n")
print("Server 4 batch index of quereis x = 7")
for i in range(0,4):
    print(s4_idxq[i])
print("\n")
print("\n")

# Matrix multiply Alice's respective server query and the batch index of queries calculated then convert to mod n
# server 1
s1_Q1 = np.dot( aliceQ1, s1_idxq)
s2_Q2 = np.dot( aliceQ2, s2_idxq)
s3_Q3 = np.dot( aliceQ3, s3_idxq)
s4_Q4 = np.dot( aliceQ4, s4_idxq)
s1_Q1 = convertMod(s1_Q1)
s2_Q2 = convertMod(s2_Q2)
s3_Q3 = convertMod(s3_Q3)
s4_Q4 = convertMod(s4_Q4)

print("Matrix Multiply of Alice's Query 1 and server1")
print(s1_Q1)
print("Matrix Multiply of Alice's Query 2 and server2")
print(s2_Q2)
print("Matrix Multiply of Alice's Query 3 and server3")
print(s3_Q3)
print("Matrix Multiply of Alice's Query 4 and server4")
print(s4_Q4)
print("\n")

# resulting query(x)*indexofqueries(x) matrix multiply to database and converted to mod n
# server1
s1_Q1_data = np.dot(s1_Q1, sData)
# server2
s2_Q2_data = np.dot(s2_Q2, sData)
# server3
s3_Q3_data = np.dot(s3_Q3, sData)
# server3
s4_Q4_data = np.dot(s4_Q4, sData)

s1_Q1_data = convertMod2(s1_Q1_data)
s2_Q2_data = convertMod2(s2_Q2_data)
s3_Q3_data = convertMod2(s3_Q3_data)
s4_Q4_data = convertMod2(s4_Q4_data)

print("Matrix Multiply of Alice's Query 1, server1, and database")
print(s1_Q1_data)
print("Matrix Multiply of Alice's Query 2, server2, and database")
print(s2_Q2_data)
print("Matrix Multiply of Alice's Query 3, server3, and database")
print(s3_Q3_data)
print("Matrix Multiply of Alice's Query 4, server4, and database")
print(s4_Q4_data)
print("\n")

# Alice reconstructs vectors to polynomial vectors
# this is where you use approximate lagrange mod
serverLab = [4, 5, 6, 7]
resPoly = create3DPoly(s1_Q1_data, s2_Q2_data, s3_Q3_data, s4_Q4_data, serverLab)
print("Reconstructed polynomials from server results:")
print(resPoly)
print("\n")

# Alice inputs x to resultant polynomial vectors to get data
data = computePolyVec(resPoly, 0, 7)
print("Final Result Query 1:")
print(data)
print("Actual Result Query 1:")
print(np.dot(np.dot(alice,covid), sData))

data2 = computePolyVec(resPoly, 1, 7)
print("Final Result: Query 2:")
print(data2)
print("Actual Result Query 2:")
print(np.dot(np.dot(alice2, hypertension), sData))
print("\n")

# Total number of 2 years of the highest admissions who have hypertension at Yolo County
print("Total number of 2 years of the highest admissions who have hypertension at Yolo County:")
print(data2[4])
print("\n")

# Total number of 2 years of the highest admissions who have covid at Yolo County
print("Total number of 2 years of the highest admissions who have covid at Yolo County:")
print(data[2])
print("\n")