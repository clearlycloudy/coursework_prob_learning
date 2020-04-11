# import Automatic Differentiation
# You may use Neural Network Framework, but only for building MLPs
# i.e. no fancy probabilistic implementations
using Flux
using MLDatasets
using Statistics
using Logging
using Test
using Random
using StatsFuns: log1pexp
Random.seed!(412414);

#### Probability Stuff
# Make sure you test these against a standard implementation!

# log-pdf of x under Factorized or Diagonal Gaussian N(x|μ,σI)
function factorized_gaussian_log_density(mu, logsig,xs)
  """
  mu and logsig either same size as x in batch or same as whole batch
  returns a 1 x batchsize array of likelihoods
  """
  σ = exp.(logsig)
  return sum((-1/2)*log.(2π*σ.^2) .+ -1/2 * ((xs .- mu).^2)./(σ.^2),dims=1)
end

# log-pdf of x under Bernoulli
function bernoulli_log_density(logit_means,x)
  """Numerically stable log_likelihood under bernoulli by accepting μ/(1-μ)"""
  b = x .* 2 .- 1 # {0,1} -> {-1,1}
  return - log1pexp.(-b .* logit_means)
end
## This is really bernoulli
@testset "test stable bernoulli" begin
  using Distributions
  x = rand(10,100) .> 0.5
  μ = rand(10)
  logit_μ = log.(μ./(1 .- μ))
  @test logpdf.(Bernoulli.(μ),x) ≈ bernoulli_log_density(logit_μ,x)
  # over i.i.d. batch
  @test sum(logpdf.(Bernoulli.(μ),x),dims=1) ≈ sum(bernoulli_log_density(logit_μ,x),dims=1)
end

# sample from Diagonal Gaussian x~N(μ,σI) (hint: use reparameterization trick here)
sample_diag_gaussian(μ,logσ) = (ϵ = randn(size(μ)); μ .+ exp.(logσ).*ϵ)
# sample from Bernoulli (this can just be supplied by library)
sample_bernoulli(θ) = rand.(Bernoulli.(θ))

# Load MNIST data, binarise it, split into train and test sets (10000 each) and partition train into mini-batches of M=100.
# You may use the utilities from A2, or dataloaders provided by a framework
function load_binarized_mnist(train_size=10000, test_size=10000)
  train_x, train_label = MNIST.traindata(1:train_size);
  test_x, test_label = MNIST.testdata(1:test_size);
  @info "Loaded MNIST digits with dimensionality $(size(train_x))"
  train_x = reshape(train_x, 28*28,:)
  test_x = reshape(test_x, 28*28,:)
  @info "Reshaped MNIST digits to vectors, dimensionality $(size(train_x))"
  train_x = train_x .> 0.5; #binarize
  test_x = test_x .> 0.5; #binarize
  @info "Binarized the pixels"
  return (train_x, train_label), (test_x, test_label)
end

function batch_data((x,label)::Tuple, batch_size=100)
  """
  Shuffle both data and image and put into batches
  """
  N = size(x)[end] # number of examples in set
  rand_idx = shuffle(1:N) # randomly shuffle batch elements
  batch_idx = Iterators.partition(rand_idx,batch_size) # split into batches
  batch_x = [x[:,i] for i in batch_idx]
  batch_label = [label[i] for i in batch_idx]
  return zip(batch_x, batch_label)
end
# if you only want to batch xs
batch_x(x::AbstractArray, batch_size=100) = first.(batch_data((x,zeros(size(x)[end])),batch_size))


### Implementing the model

## Load the Data
train_data, test_data = load_binarized_mnist(10000, 10000);
train_x, train_label = train_data;
test_x, test_label = test_data;

## Test the dimensions of loaded data
@testset "correct dimensions" begin
@test size(train_x) == (784,1000)
@test size(train_label) == (1000,)
@test size(test_x) == (784,1000)
@test size(test_label) == (1000,)
end

## Model Dimensionality
# #### Set up model according to Appendix C (using Bernoulli decoder for Binarized MNIST)
# Set latent dimensionality=2 and number of hidden units=500.
Dz, Dh = 2, 500
Ddata = 28^2

# ## Generative Model
# This will require implementing a simple MLP neural network
# See example_flux_model.jl for inspiration
# Further, you should read the Basics section of the Flux.jl documentation
# https://fluxml.ai/Flux.jl/stable/models/basics/
# that goes over the simple functions you will use.
# You will see that there's nothing magical going on inside these neural network libraries
# and when you implemented a neural network in previous assignments you did most of the work.
# If you want more information about how to use the functions from Flux, you can always reference
# the internal docs for each function by typing `?` into the REPL:
# ? Chain
# ? Dense

## Model Distributions
#compute log of prior for a digit's representation log p(z)
log_prior(z) = factorized_gaussian_log_density(0,0,z)

#TODO
# (z x batch), NN parameters -> hidden x batch -> Ddata
latent_z_test = randn((Dz,10))

decoder = Chain(
  Dense(Dz, Dh, tanh),
  Dense(Dh, Ddata))

decoder(latent_z_test)

function log_likelihood(x,z)
  """ Compute log likelihood log_p(x|z)"""
  logit_mean = decoder(z)
  # return likelihood for each element in batch
  return sum(bernoulli_log_density(logit_mean,x),dims=1)
end

#TODO
joint_log_density(x,z) = log_prior(z) .+ log_likelihood(x,z)

## Amortized Inference
function unpack_gaussian_params(θ)
  μ, logσ = θ[1:2,:], θ[3:end,:]
  return  μ, logσ
end

 #TODO
 #x, recognition params -> mean,log-sd of latent multivariate gaussian
# Hint: last "layer" in Chain can be 'unpack_gaussian_params'
encoder = Chain(
  Dense(Ddata, Dh, tanh),
  Dense(Dh, Dz*2),
  unpack_gaussian_params)

#TODO: write log likelihood under variational distribution.
log_q(q_μ, q_logσ, z) =  factorized_gaussian_log_density(q_μ, q_logσ, z)

function elbo(x)
  # # version 1
  # (q_μ, q_logσ) = encoder(x)
  # # sample from variational distribution
  # z = sample_diag_gaussian(q_μ, q_logσ)
  # # joint likelihood of z and x under model
  # joint_ll = joint_log_density(x,z)
  # # likelihood of z under variational distribution
  # log_q_z = log_q(q_μ, q_logσ, z))
  # # Scalar value, mean variational evidence lower bound over batch
  # elbo_estimate = mean(joint_ll - log_q_z, dims=2)
  # return elbo_estimate[1]

  # version 2
  (q_μ, q_logσ) = encoder(x)
  # sample from variational distribution
  z = sample_diag_gaussian(q_μ, q_logσ)
  likelihoods = log_likelihood(x,z)
  negative_kl = 1/2 * sum(1 .+ 2 .* q_logσ .- q_μ .* q_μ .- exp.(2 .* q_logσ), dims=1)
  return mean(negative_kl + likelihoods, dims=2)[1]
end

function loss(x)
  return -elbo(x)
end

# Training with gradient optimization:
# See example_flux_model.jl for inspiration

function train_model_params!(loss, encoder, decoder, train_x, test_x; nepochs=10)
  # model params
  # parameters to update with gradient descent
  ps = Flux.params(encoder, decoder)
  # ADAM optimizer with default parameters
  opt = ADAM(0.0025)
  # over batches of the data
  for i in 1:nepochs
    for d in batch_x(train_x,1000)
      # compute gradients with respect to variational loss over batch
      gs  = Flux.gradient(()->loss(d), ps)
      # update the paramters with gradients
      Flux.Optimise.update!(opt,ps,gs)
    end
    if i%1 == 0 # change 1 to higher number to compute and print less frequently
      @info "Test loss at epoch $i: $(loss(batch_x(test_x)[1]))"
    end
  end
  @info "Parameters of encoder and decoder trained!"
end


## Train the model
train_model_params!(loss,encoder,decoder,train_x,test_x, nepochs=100)

### Save the trained model!
using BSON:@save
cd(@__DIR__)
@info "Changed directory to $(@__DIR__)"
save_dir = "trained_models"
if !(isdir(save_dir))
  mkdir(save_dir)
  @info "Created save directory $save_dir"
end
@save joinpath(save_dir,"encoder_params.bson") encoder
@save joinpath(save_dir,"decoder_params.bson") decoder
@info "Saved model params in $save_dir"



## Load the trained model!
using BSON:@load
cd(@__DIR__)
@info "Changed directory to $(@__DIR__)"
load_dir = "trained_models"
@load joinpath(load_dir,"encoder_params.bson") encoder
@load joinpath(load_dir,"decoder_params.bson") decoder
@info "Load model params from $load_dir"




# Visualization
using Images
using Plots
# make vector of digits into images, works on batches also
mnist_img(x) = ndims(x)==2 ? Gray.(reshape(x,28,28,:))' : Gray.(reshape(x,28,28))'

function helper_reconstruct(x)
  temp = encoder(x)
  logit_mean = decoder(temp[1])
  reconstruct = 1.0 ./(1.0 .+ exp.(-logit_mean))
  reconstruct[:]
end

#test only
#plot(mnist_img(train_x[:,499]))
#plot(mnist_img(helper_reconstruct(train_x[:,499])))

# 3a
number_of_samples = 10

sample_z = randn((2,number_of_samples))
logit_mean = decoder(sample_z)
bernoulli_mean = 1.0 ./ (1.0 .+ exp.(-logit_mean))
sample_images = sample_bernoulli(bernoulli_mean)

plots_ber = []
plots_bin = []
for i in 1:number_of_samples
     push!(plots_ber, plot(mnist_img(bernoulli_mean[:,i])))
     push!(plots_bin, plot(mnist_img(sample_images[:,i])))
end

plots = [ plots_ber; plots_bin ]
rows = Int(number_of_samples*2/10)
display(plot(plots..., layout=grid(rows,10), size =(150*10, 150*rows), axis=nothing))

savefig(joinpath("plots","3a.png"))

# 3b
mean_vector_x = [[] for i = 1:10]
mean_vector_y = [[] for i = 1:10]
mean_label = ["0" "1" "2" "3" "4" "5" "6" "7" "8" "9"]
for i in 1:size(train_x)[2]
  (mu, logsig) = encoder(train_x[:,i])
  push!(mean_vector_x[train_label[i]+1], mu[1])
  push!(mean_vector_y[train_label[i]+1], mu[2])
end

plot(mean_vector_x,
     mean_vector_y,
     seriestype = :scatter,
     xlabel="z1 mean",
     ylabel="z2 mean",
     label=mean_label)

savefig(joinpath("plots","3b.png"))

# 3c
function interp(a, b, alpha)
  alpha .* a + (1-alpha) .* b
end

#test only
#plot(mnist_img(train_x[:,9981]))
#plot(mnist_img(helper_reconstruct(train_x[:,9981])))

image_samples = [ (train_x[:,3889],train_x[:,9000]), #,0 9
                  (train_x[:,499],train_x[:,600]), # 6, 7
                  (train_x[:,9137],train_x[:,99]), #1, 3
                ]

image_samples_encode = map(x -> (encoder(x[1]), encoder(x[2])),
                          image_samples)

plots_interp = []

for i in 1:3
  (enc_a, enc_b) = image_samples_encode[i]
  mu_a = enc_a[1]
  mu_b = enc_b[1]
  sig_a = exp.(enc_a[2])
  sig_b = exp.(enc_b[2])
  for j in 0:10
    alpha = j/10.0
    mu_alpha = interp(mu_a, mu_b, alpha)
    sig_alpha = interp(sig_a, sig_b, alpha)
    logit_mean = decoder(mu_alpha)
    reconstruct = exp.(logit_mean) ./ (1 .+ exp.(logit_mean))
    push!(plots_interp, plot(mnist_img(reconstruct[:])))
  end
end

display(plot(plots_interp..., layout=grid(3,11), size =(1000, 200), axis=nothing))
savefig(joinpath("plots","3c.png"))

# 4a
function image_top(x)
  #assume x is flattened per image
  xx = reshape(x,28,28,:)
  xxx = permutedims(xx,[2,1,3])
  xxx[1:14,1:28,:]
end

function image_top_single(x)
  @assert size(x)[2]==1
  #assume x is flattened per image
  xx = reshape(x,28,28)
  xxx = permutedims(xx,[2,1])
  xxx[1:14,1:28]
end

function image_bottom(x)
  #assume x is flattened per image
  xx = reshape(x,28,28,:)
  xxx = permutedims(xx,[2,1,3])
  xxx[15:28,1:28,:]
end

function log_likelihood_image_top(x,z)
  #assume x is the top portion of the original image
  logit_mean = decoder(z)

  logit_mean_cropped = reshape(image_top_single(logit_mean), 14*28,:)
  # println(size(logit_mean_cropped))
  # println(size(x))
  @assert size(x)==size(logit_mean_cropped)

  sum(bernoulli_log_density(logit_mean_cropped, x), dims=1)
end

function log_joint_top(x,z)
  log_prior(z) .+ log_likelihood_image_top(x,z)
end

#4 b: stochastic variational inference
encoder_2_in_dim = Int(Ddata/2)
encoder_2 = Chain(
  Dense(encoder_2_in_dim, Dh, tanh),
  Dense(Dh, Dz*2),
  unpack_gaussian_params)

function elbo_2(x, K=10)

  temp = image_top(x)
  # @info size(temp)
  xx = reshape(temp, 14*28,:)
  @assert size(xx)[1] == encoder_2_in_dim

  #@info size(z_mean)
  (z_mean, z_log_sig) = encoder_2(xx)
  samples = size(z_mean)[2]
  samples_elbo = []

  avg = 0.0

  for i in range(1,samples;step=1)
    accum = 0
    for h in range(1,K,;step=1)
      a = z_mean[:,i:i]
      b = z_log_sig[:,i:i]
      #@info a
      @assert size(a)==(2,1)
      @assert size(b)==(2,1)
      z_sample = sample_diag_gaussian(a, b)
      @assert size(z_sample)==(2,1)
      #@info size(xx[:,i:i])
      xxx = xx[:,i:i]
      joint = log_joint_top(xxx, z_sample)
      #@info size(joint)
      posterior = log_q(a, b, z_sample)
      #@info size(posterior)
      accum += joint[1] - posterior[1]
    end
    avg += accum / samples
  end
  avg
end

function loss_2(x)
  return -elbo_2(x)
end

function train_model_params_2!(loss, enc, dec, trainx, testx; nepochs=10)
  # parameters to update with gradient descent
  ps = Flux.params(enc)
  # ADAM optimizer with default parameters
  opt = ADAM(0.0025)
  # over batches of the data
  for i in 1:nepochs
    for d in batch_x(trainx,1000)
      # compute gradients with respect to variational loss over batch
      gs  = Flux.gradient(()->loss(d), ps)
      # update the paramters with gradients
      Flux.Optimise.update!(opt,ps,gs)
    end
    if i%1 == 0 # change 1 to higher number to compute and print less frequently
      @info "Test loss at epoch $i: $(loss(batch_x(testx)[1]))"
    end
  end
  @info "Parameters of encoder and decoder trained!"
end

function crop_top(xs)
  yyy = image_top(xs)
  empty = zeros(14,28,size(yyy)[3])
  tt = vcat(yyy,empty)
  ss = permutedims(reshape(tt,28,28,:),[2,1,3])
  reshape(ss,28*28,:)
end

idx_train = train_label .== 5
digits_train = train_x[:,idx_train]

#sanity test
# plot(Gray.(reshape(digits_train,28,28,:)[:,:,569])')
# plot(Gray.(reshape(digits_cropped_train,28,28,:)[:,:,569])')

idx_test = test_label .== 5
digits_test = test_x[:,idx_test]

train_model_params_2!(loss_2,encoder_2,decoder,digits_train, digits_test, nepochs=100)

### Save the trained model!
using BSON:@save
cd(@__DIR__)
@info "Changed directory to $(@__DIR__)"
save_dir = "trained_models"
if !(isdir(save_dir))
  mkdir(save_dir)
  @info "Created save directory $save_dir"
end
@save joinpath(save_dir,"encoder_2_params.bson") encoder_2
@info "Saved model params in $save_dir"

## Load the trained model!
using BSON:@load
cd(@__DIR__)
@info "Changed directory to $(@__DIR__)"
load_dir = "trained_models"
@load joinpath(load_dir,"encoder_2_params.bson") encoder_2
@info "Load model params from $load_dir"

digits_cropped_train = crop_top(digits_train)
#plot(Gray.(reshape(digits_train,28,28,:)[:,:,5])')
#plot(Gray.(reshape(digits_cropped_train,28,28,:)[:,:,5])')

temp = image_top(digits_cropped_train)
# @info size(temp)
xx = reshape(temp, 14*28,:)
temp = encoder_2(xx)
#z_sample = sample_diag_gaussian(temp[1], temp[2])
logit_mean = decoder(temp[1])
reconstruct = 1.0 ./(1.0 .+ exp.(-logit_mean))

stitched = vcat(image_top(digits_cropped_train), image_bottom(reconstruct))
ss = reshape(permutedims(reshape(stitched,28,28,:),[2,1,3]),28*28,:)

#plot(Gray.(reshape(ss,28,28,:)[:,:,5])')

idxs = shuffle(1:size(ss)[2])
samples = ss[:,idxs][:,1:50]

samples_original = digits_train[:,idxs][:,1:50]
#test
#plot(Gray.(reshape(samples,28,28,:)[:,:,2])')

plot_recon_stitch = []
plot_sample_original = []

for i in range(1,size(samples)[2],step=1)
  push!(plot_recon_stitch, plot(Gray.(reshape(samples[:,i],28,28))'))
  push!(plot_sample_original, plot(Gray.(reshape(samples_original[:,i],28,28))'))
end

display(plot(plot_sample_original..., layout=grid(5,10), size =(1500, 750), axis=nothing))
savefig(joinpath("plots","4b_e_1.png"))

display(plot(plot_recon_stitch..., layout=grid(5,10), size =(1500, 750), axis=nothing))
savefig(joinpath("plots","4b_e_2.png"))

# contour plot

# select a sample
x = reshape(image_top(digits_cropped_train), 14*28,:)[:,31:31]
plot(Gray.(reshape(x,14,28)))
savefig(joinpath("plots","4b_d_sample.png"))

z1 = -4:0.07:4.0
z2 = -4:0.07:4.0

f(z1,z2) = begin
  z = zeros(2,1)
  z[1,1] = z1
  z[2,1] = z2
  v = log_joint_top(x,z)
  @assert size(v)==(1,1)
  v[1]
end

p1 = contour(z1, z2, f,
             fill=true,
             title="p(z,top half of x)")

(z_mean, z_log_sig) = encoder_2(x)

a = z_mean[:,1:1]
b = z_log_sig[:,1:1]

f2(z1,z2) = begin
  z = zeros(2,1)
  z[1,1] = z1
  z[2,1] = z2
  v = log_q(a, b, z)
  @assert size(v)==(1,1)
  v[1]
end

p2 = contour(z1, z2, f2,
             fill=true,
             title="approximate posterior q(z| top half of x)")

plot(p1,p2, size=(1000,500))
savefig(joinpath("plots","4b_d.png"))
