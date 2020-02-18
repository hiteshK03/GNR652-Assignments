using CSV
using Plots

dataset = CSV.read("housingPriceData.csv")
# dataset = CSV.read("scores.csv")
sqft_living = dataset.sqft_living
bedrooms = dataset.bedrooms
bathrooms = dataset.bathrooms
price = dataset.price

rooms = bedrooms .+ bathrooms

p1 = scatter(rooms, sqft_living)
p2 = scatter(bedrooms, sqft_living)
# removed a bedroom outlier
data = dataset[dataset[:bedrooms].<20,:]

sqft_living1 = data.sqft_living
bedrooms1 = data.bedrooms
bathrooms1 = data.bathrooms
price1 = data.price
rooms1 = bedrooms1 .+ bathrooms1

p3 = scatter(sqft_living1, price1)
# removed a sqft_living outlier
data = data[(sqft_living1.>12000) .& (price1.<4000000),:]

sqft_living2 = data.sqft_living
bedrooms2 = data.bedrooms
bathrooms2 = data.bathrooms
price2 = data.price
rooms2 = bedrooms2 .+ bathrooms2

p4 = scatter3d(bathrooms2, bedrooms2, sqft_living2)
p5 = scatter(bathrooms2, sqft_living2)
findall((sqft_living2.<5000) .& (bathrooms2.>6.5))
# removed a bathroom outlier
data = data[(sqft_living2.>5000) .| (bathrooms2.<6.5),:]

sqft_living3 = data.sqft_living
bedrooms3 = data.bedrooms
bathrooms3 = data.bathrooms
price3 = data.price
rooms3 = bedrooms3 .+ bathrooms3

p3 = scatter3d(rooms3, sqft_living3, price3)
p4 = scatter(rooms3, sqft_living3)
p4 = scatter(rooms3, price3)
sqft_living3[findall(rooms3.>15)]
rooms3[findall(sqft_living3.<4600)]
# Therefore it seems to be an outlier with rooms>15 & sqft_living3 < 4600
findall((sqft_living3.<4600) .& (rooms3.>15))
# removed a room outlier
data = data[(sqft_living3.>4600) .| (rooms3.<15),:]

sqft_living4 = data.sqft_living
bedrooms4 = data.bedrooms
bathrooms4 = data.bathrooms
price4 = data.price
rooms4 = bedrooms4 .+ bathrooms4

# Better results than the gaussian normalisation
function normalise(X)
    m = length(X)
    max = maximum(X)
    min = minimum(X)
    mid = (max+min)/2
    norm = ((X .- mid).*2)./(max-min)
    return norm
end

sqft_living_norm = normalise(sqft_living4)
bedrooms_norm = normalise(bedrooms4)
bathrooms_norm = normalise(bathrooms4)
price_norm = normalise(price4)

m = length(sqft_living_norm)
x0 = ones(m)

X = cat(x0, bathrooms_norm, bedrooms_norm, sqft_living_norm, dims=2)
Y = price_norm

function costFunction(X, Y, B)
    m = length(Y)
    cost = sum(((X * B) - Y).^2)
    return cost
end

function lasso(X, Y, B, parameter)
    cost = costFunction(X, Y, B)
    m = length(B)
    reg = parameter * sum(B.^2)
    regCost = cost .+ reg
    return regCost
end

B = zeros(4, 1)

intialCost = lasso(X, Y, B, parameter)

function gradientDescent(X, Y, B, parameter, learningRate, numIterations)
    costHistory = zeros(numIterations)
    m = length(Y)
    for iteration in 1:numIterations
        loss = (X * B) - Y
        gradient = (X' * loss)/m + 2*parameter*B
        B = B - learningRate * gradient
        cost = lasso(X, Y, B, parameter)
        costHistory[iteration] = cost
    end
    return B, costHistory
end

parameter = 0.001
learningRate = 0.0001
newB, costHistory = gradientDescent(X, Y, B, learningRate, 25000)

# for modified data
YPred = X * newB

# for original data
X_orig = cat(x0, bathrooms, bedrooms, sqft_living, dims=2)
Y_orig = price
YPred_orig = X_orig * newB

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

RMSE(Y, YPred)
R2_score(Y, YPred)

plot(Y[1:10])
plot!(YPred[1:10])
plot(costHistory)