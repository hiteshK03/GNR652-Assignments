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
train_data = normdata[1:floor(Int, 0.6*m),:]
val_data = normdata[floor(Int, 0.6*m)+1:floor(Int, 0.8*m),:]
test_data = normdata[floor(Int, 0.8*m)+1:end,:]

sqft_living = train_data.sqft_living
bedrooms = train_data.bedrooms
bathrooms = train_data.bathrooms
price = train_data.price
# rooms = bedrooms4 .+ bathrooms4

m = length(bathrooms)
x0 = ones(m)

X = cat(x0, bathrooms, bedrooms, sqft_living, dims=2)
Y = price

x1 = ones(length(val_data.price))
X_val = cat(x1, val_data.bathrooms, val_data.bedrooms, val_data.sqft_living, dims=2)
Y_val = val_data.price

x2 = ones(length(test_data.price))
X_test = cat(x2, test_data.bathrooms, test_data.bedrooms, test_data.sqft_living, dims=2)
Y_test = test_data.price

function costFunction(X, Y, B)
    m = length(Y)
    cost = sum(((X * B) - Y).^2)
    return cost
end

function lasso(X, Y, B, parameter)
    cost = costFunction(X, Y, B)
    m = length(B)
    reg = parameter * sum(broadcast(abs, B))
    regCost = cost .+ reg
    return regCost
end

B = ones(4, 1).*0.3
parameter = 17
learningRate = 0.001
numIterations = 10000

intialCost = lasso(X, Y, B, parameter)

function gradientDescent(X, Y, B, parameter, learningRate, numIterations)
    costHistory = zeros(numIterations)
    m = length(Y)
    for iteration in 1:numIterations
        loss = (X * B) - Y
        gradient = (X' * loss)/m + 2*parameter*(broadcast(<, 0, B)-broadcast(>, 0, B))
        B = B - learningRate * gradient
        cost = lasso(X, Y, B, parameter)
        costHistory[iteration] = cost
    end
    return B, costHistory
end

function validate(newB)
    Y_predVal = X_val * newB
    rms = RMSE(Y_predVal, Y_val)
    r2 = R2_score(Y_val, Y_predVal)
    # println("RMSE value : ",rms)
    # println("R2 score : ", r2)
    return rms
end

# numEpoch = 20

newB, costHistory = gradientDescent(X, Y, B, parameter, learningRate, numIterations)
# for i in 1:numEpoch
#     println("Epoch : ",i)
#     parameter = i*2
#     println("Current parameter : ",parameter)
#     newB, costHistory = gradientDescent(X, Y, B, parameter*i*10, learningRate, 100000)
#     println("Validation")
#     rms = validate(newB)
#     RMSEHistory[i] = rms
# end

# plot(costHistory)

Y_pred = X_test * newB
rms = RMSE(Y_test, Y_pred)
r2_score = R2_score(Y_test, Y_pred)

CSV.write("./data/2b.csv", DataFrame(Y_pred), writeheader=false)
