using Plots
using CSV
using DataFrames

dataset = CSV.read("./data/housingPriceData.csv")

function mean(X)
    n = length(X)
    mn = sum(X)/n
    return mn
end

function sd(X)
    m = length(X)
    sumSq = sum(X.^2)/m
    ans = sqrt(sumSq .- (mean(X)^2))
    return ans
end

function normalise(X)
    mn = mean(X)
    s = sd(X)
    norm = (X .- mn)./s
    return norm
end

function RMSE(Y, YPred)
	n = length(Y)
	err = sum((YPred-Y).^2)/n
	err = sqrt(err)
	return err
end

function R2_score(Y, YPred)
	n = length(Y)
	mean = sum(Y)/n
	err = (sum((YPred-Y).^2)/n)/(sum((Y.-mean).^2)/n)
	R2 = 1-err
	return R2
end

n = length(dataset[1,:])
normdata = dataset
for i in 3:5
    normdata[!,i] = normalise(dataset[!,i])
end

m = length(dataset[:,1])
# println(m)
train_data = normdata[1:floor(Int, 0.8*m),:]
test_data = normdata[floor(Int, 0.8*m)+1:end,:]

sqft_living = train_data.sqft_living
bedrooms = train_data.bedrooms
bathrooms = train_data.bathrooms
price = train_data.price

m = length(bathrooms)
x0 = ones(m)
x1 = sqft_living
x2 = bedrooms
x3 = sqft_living.^2
x4 = bedrooms.^2
x5 = bedrooms.*sqft_living

X = cat(x0, x1, x2, x3, x4, x5, dims=2)
Y = price

x1 = ones(length(test_data.price))
X_test = cat(x1, test_data.sqft_living, test_data.bedrooms, test_data.sqft_living.^2, test_data.bedrooms.^2, test_data.bedrooms.*test_data.sqft_living, dims=2)
Y_test = test_data.price

function costFunction(X, Y, B)
    m = length(Y)
    cost = sum(((X * B) - Y).^2)/(2*m)
    return cost
end

B = zeros(6, 1)

intialCost = costFunction(X, Y, B)

function gradientDescent(X, Y, B, learningRate, numIterations)
    costHistory = zeros(numIterations)
    B = zeros(6, 1)
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

learningRate = 0.001
# println("started")
newB, costHistory = gradientDescent(X, Y, B, learningRate, 20000)

# plot(costHistory)

Y_pred = X_test * newB
rms = RMSE(Y_test, Y_pred)
r2_score = R2_score(Y_test, Y_pred)
# println("RMSE : ", rms)

# println("R2 score : ",R2_score(Y_test, Y_pred))

CSV.write("./data/1b.csv", DataFrame(Y_pred), writeheader=false)
