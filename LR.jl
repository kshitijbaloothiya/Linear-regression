using CSV
using Statistics
using LinearAlgebra
file = CSV.read("C:/Users/91704/Documents/IIT/SEM-2/Machine learning/Assignment1/housingPriceData.csv")
X1=file.bedrooms
X2=file.bathrooms
X3=file.sqft_living
Y=file.price
X1=(X1.-mean(X1))
X1=(X1./std(X1))
X2=(X2.-mean(X2))
X2=(X2./std(X2))
X3=(X3.-mean(X3))
X3=(X3./std(X3))
#Y=(Y.-mean(Y))
#Y=(Y./std(Y))
m = length(X1)
X0 = ones(m)
X=cat(X0,X1,X2,X3,dims=2)
function costFunction(X, Y, B)
    m = length(Y)
    cost = sum(((X * B) - Y).^2)/(2*m)
    return cost
end
B = zeros(4, 1)
intialCost = costFunction(X, Y, B)
function gradientDescent(X, Y, B, learningRate, numIterations)
    costHistory = zeros(numIterations)
    m = length(Y)
    for iteration in 1:numIterations
        loss = (X * B) - Y
        gradient = (X' * loss)/m
        B = B - learningRate * gradient
        cost = costFunction(X, Y, B)
        costHistory[iteration] = cost
    end
    return B, costHistory
end
learningRate = 0.01
newB, costHistory = gradientDescent(X, Y, B, learningRate, 1000)
YPred = X * newB
R=0
for i=1:length(Y)
    R=R+(Y[i]-YPred[i])^2
end
R=R/length(Y)
R=sqrt(R)
R1=0
R2=0
for j=1:length(Y)
    R1=R1+(YPred[j]-Y[j])^2
    R2=R2+(Y[j]-mean(Y))^2
end
Rsquare=1-(R1/R2)
