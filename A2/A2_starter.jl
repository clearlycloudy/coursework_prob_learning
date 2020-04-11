using Revise # lets you change A2funcs without restarting julia!
include("A2_src.jl")
using Plots
using Statistics: mean
using Zygote
using Test
using Logging
using .A2funcs: log1pexp # log(1 + exp(x)) stable
using .A2funcs: factorized_gaussian_log_density
using .A2funcs: skillcontour!
using .A2funcs: plot_line_equal_skill!

function log_prior(zs)
  # factorized_gaussian_log_density(0,0,zs)
  N = size(zs)[1]
  -1/2 * sum(zs.*zs,dims=1) .+ N*log(1/sqrt(2*pi))
end

function logp_a_beats_b(za,zb)
  -log1pexp.(zb.-za)
end


function all_games_log_likelihood(zs,games)
  zs_a = zs[games[:,1],:]
  zs_b =  zs[games[:,2],:]
  likelihoods =  logp_a_beats_b(zs_a,zs_b)
  sum(likelihoods,dims=1)
end

function joint_log_density(zs,games)
  all_games_log_likelihood(zs,games) .+ log_prior(zs)
end

@testset "Test shapes of batches for likelihoods" begin
  B = 15 # number of elements in batch
  N = 4 # Total Number of Players
  test_zs = randn(4,15)
  test_games = [1 2; 3 1; 4 2] # 1 beat 2, 3 beat 1, 4 beat 2
  @test size(test_zs) == (N,B)
  #batch of priors
  @test size(log_prior(test_zs)) == (1,B)
  # loglikelihood of p1 beat p2 for first sample in batch
  @test size(logp_a_beats_b(test_zs[1,1],test_zs[2,1])) == ()
  # loglikelihood of p1 beat p2 broadcasted over whole batch
  @test size(logp_a_beats_b.(test_zs[1,:],test_zs[2,:])) == (B,)
  # batch loglikelihood for evidence
  @test size(all_games_log_likelihood(test_zs,test_games)) == (1,B)
  # batch loglikelihood under joint of evidence and prior
  @test size(joint_log_density(test_zs,test_games)) == (1,B)
end

# Convenience function for producing toy games between two players.
two_player_toy_games(p1_wins, p2_wins) = vcat([repeat([1,2]',p1_wins), repeat([2,1]',p2_wins)]...)

# Example for how to use contour plotting code
plot(title="Example Gaussian Contour Plot",
    xlabel = "Player 1 Skill",
    ylabel = "Player 2 Skill"
   )
example_gaussian(zs) = exp(factorized_gaussian_log_density([-1.,2.],[0.,0.5],zs))
skillcontour!(example_gaussian)
plot_line_equal_skill!()
savefig(joinpath("plots","example_gaussian.png"))

# TODO: plot prior contours
plot(title="Prior Contour Plot",
    xlabel = "Player 1 Skill",
    ylabel = "Player 2 Skill")
skillcontour!(zs -> exp(log_prior(zs)))
plot_line_equal_skill!()
savefig(joinpath("plots","prior_contour.png"))

# TODO: plot likelihood contours
plot(title="Likelihood Contour Plot",
    xlabel = "Player 1 Skill",
    ylabel = "Player 2 Skill")
skillcontour!(zs -> exp.(logp_a_beats_b(zs[1,:],zs[2,:])))
plot_line_equal_skill!()
savefig(joinpath("plots","likelihood_contour.png"))

# TODO: plot joint contours with player A winning 1 game
games = two_player_toy_games(1,0)

plot(title="Joint Contour Plot, A Winning 1 Game",
    xlabel = "Player A Skill",
    ylabel = "Player B Skill")
skillcontour!(zs -> exp(joint_log_density(zs,games)))
plot_line_equal_skill!()
savefig(joinpath("plots","joint_contour_A1_B0.png"))

# TODO: plot joint contours with player A winning 10 games
games = two_player_toy_games(10,0)

plot(title="Joint Contour Plot, A Winning 10 Games",
    xlabel = "Player A Skill",
    ylabel = "Player B Skill")
skillcontour!(zs -> exp(joint_log_density(zs,games)))
plot_line_equal_skill!()
savefig(joinpath("plots","joint_contour_A10_B0.png"))

#TODO: plot joint contours with player A winning 10 games and player B winning 10 games
games = two_player_toy_games(10,10)

plot(title="Joint Contour Plot, A Winning 10 Games, B Winning 10 Games",
    xlabel = "Player A Skill",
    ylabel = "Player B Skill")
skillcontour!(zs -> exp(joint_log_density(zs,games)))
plot_line_equal_skill!()
savefig(joinpath("plots","joint_contour_A10_B10.png"))

function elbo(params,logp,num_samples)

  lsigma = params[2]
  sigma = exp.(params[2])
  m = params[1]

  N = length(params[1]) #number of players
  B = num_samples #batch size of samples

  s = randn(Float64, (N, B))
  samples = sigma .* s .+ m

  logp_estimate = logp(samples)
  logq_estimate = factorized_gaussian_log_density(m,lsigma,samples)

  mean((logp_estimate .- logq_estimate),dims=2)
end

# Conveinence function for taking gradients
function neg_toy_elbo(params; games = two_player_toy_games(1,0), num_samples = 100)
  # TODO: Write a function that takes parameters for q,
  # evidence as an array of game outcomes,
  # and returns the -elbo estimate with num_samples many samples from q
  logp(zs) = joint_log_density(zs,games)
  return -elbo(params,logp, num_samples)
end


using Distributions

# Toy game
num_players_toy = 2
toy_mu = randn(2)
toy_ls = rand(Uniform(0,1), 2)
toy_params_init = (toy_mu, toy_ls)

function fit_toy_variational_dist(init_params, toy_evidence; num_itrs=200, lr= 1e-2, num_q_samples = 30)
  params_cur = init_params
  for i in 1:num_itrs

    grad_params = gradient(in -> neg_toy_elbo(in, games=toy_evidence;num_samples=num_q_samples)[1], params_cur)[1]
    params_cur =  params_cur .- lr .* grad_params;

    if i % 25 == 0 || i == num_itrs
      neg_elbo = neg_toy_elbo(params_cur, games=toy_evidence;num_samples=num_q_samples)[1]
      @info "neg_elbo: " neg_elbo
      p = plot(title="True Posterior and Variational Plot",
          xlabel = "Player 1 Skill",
          ylabel = "Player 2 Skill")
      skillcontour!(zs -> exp(factorized_gaussian_log_density(params_cur[1], params_cur[2], zs)), colour=:blue)
      skillcontour!(zs -> exp(joint_log_density(zs, toy_evidence)), colour=:red)
      display(p);
    end
  end
  return params_cur
end

#TODO: fit q with SVI observing player A winning 1 game
#TODO: save final posterior plots
evidence = two_player_toy_games(1,0)
params_learned = fit_toy_variational_dist(toy_params_init, evidence, num_itrs=800)

p = plot(title="Variational Contour, A winning 1 Game",
    xlabel = "Player A Skill",
    ylabel = "Player B Skill")
skillcontour!(zs -> exp(factorized_gaussian_log_density(params_learned[1],params_learned[2],zs)),colour=:blue)
skillcontour!(zs -> exp(joint_log_density(zs, evidence)),colour=:red)
display(p);
savefig(joinpath("plots","variational_fit_A1_B0.png"))

#TODO: fit q with SVI observing player A winning 10 games
#TODO: save final posterior plots
evidence = two_player_toy_games(10,0)
params_learned = fit_toy_variational_dist(toy_params_init, evidence, num_itrs=700)

p = plot(title="Variational Contour, A winning 10 Games",
    xlabel = "Player A Skill",
    ylabel = "Player B Skill")
skillcontour!(zs -> exp(factorized_gaussian_log_density(params_learned[1],params_learned[2],zs)),colour=:blue)
skillcontour!(zs -> exp(joint_log_density(zs, evidence)),colour=:red)
display(p);
savefig(joinpath("plots","variational_fit_A10_B0.png"))

#TODO: fit q with SVI observing player A winning 10 games and player B winning 10 games
#TODO: save final posterior plots
evidence = two_player_toy_games(10,10)
params_learned = fit_toy_variational_dist(toy_params_init, evidence, num_itrs=700)

p = plot(title="Variational Contour, A winning 10 Games, B wining 10 Games ",
    xlabel = "Player A Skill",
    ylabel = "Player B Skill")
skillcontour!(zs -> exp(factorized_gaussian_log_density(params_learned[1],params_learned[2],zs)),colour=:blue)
skillcontour!(zs -> exp(joint_log_density(zs, evidence)),colour=:red)
display(p);
savefig(joinpath("plots","variational_fit_A10_B10.png"))

## Question 4
# Load the Data
using MAT
vars = matread("tennis_data.mat")
player_names = vars["W"]
tennis_games = Int.(vars["G"])
num_players = length(player_names)
print("Loaded data for $num_players players")


function fit_variational_dist(init_params,
                              tennis_games;
                              num_itrs=200,
                              lr= 1e-2,
                              num_q_samples = 10)
  params_cur = init_params
  for i in 1:num_itrs

    grad_params = gradient(in -> neg_toy_elbo(in,
                                              games=tennis_games;
                                              num_samples=num_q_samples)[1],
                          params_cur)[1]
    params_cur =  params_cur .- lr .* grad_params;

    if i % 25 == 0 || i == num_itrs
      neg_elbo = neg_toy_elbo(params_cur,
                              games=tennis_games;
                              num_samples=num_q_samples)[1]
      @info "neg_elbo: " neg_elbo
      # p = plot(title="True Posterior and Variational Plot",
      #     xlabel = "Player 1 Skill",
      #     ylabel = "Player 2 Skill")
      # skillcontour!(zs -> exp(factorized_gaussian_log_density(params_cur[1], params_cur[2], zs)), colour=:blue)
      # skillcontour!(zs -> exp(joint_log_density(zs, toy_evidence)), colour=:red)
      # display(p);
    end
  end
  neg_elbo = neg_toy_elbo(params_cur,
                          games=tennis_games;
                          num_samples=num_q_samples)[1]
  @info "neg_elbo: " neg_elbo
  return params_cur
end

# TODO: Initialize variational family
init_mu = randn(num_players)
init_log_sigma = rand(Uniform(0,1), num_players)
init_params = (init_mu, init_log_sigma)

# Train variational distribution
trained_params = fit_variational_dist(init_params,
                                      tennis_games,
                                      num_itrs=700,
                                      lr= 5e-3,
                                      num_q_samples = 30)

perm = sortperm(trained_params[1])
plot(trained_params[1][perm], yerror=exp.(trained_params[2][perm]),
    title="Player Skills and Uncertainty(S.D.)",
    xlabel = "Player Enumeration",
    ylabel = "Player Skill",
    legend=false)
savefig(joinpath("plots","4c_all_player_skills.png"))

#TODO: 10 players with highest mean skill under variational model
#hint: use sortperm
idx_ordered = sortperm(trained_params[1])
idx_top = reverse(idx_ordered[end-9:end])
top_players = player_names[idx_top]

top_players_skills = trained_params[1][idx_top]
top = collect(zip(top_players,top_players_skills))
print("top players: ", top)

#TODO: joint posterior over "Roger-Federer" and ""Rafael-Nadal""
#hint: findall function to find the index of these players in player_names
idx_1 = findall(x-> x =="Roger-Federer", player_names)
idx_2 = findall(x-> x == "Rafael-Nadal", player_names)
idx = [idx_1;idx_2]
m = trained_params[1][idx]
ls = trained_params[2][idx]

p = plot(title="Joint Posterior Plot",
    xlabel = "Roger-Federer Skill",
    ylabel = "Rafael-Nadal Skill")
skillcontour!(zs -> exp(factorized_gaussian_log_density(m,ls,zs)))
display(p);
savefig(joinpath("plots","joint_posterior_federer_nadal.png"))

#exact prob of skill_federer > skill_nadal from learned posterior
idx_nadal = findall(x-> x == "Rafael-Nadal", player_names)[1]
idx_federer = findall(x-> x =="Roger-Federer", player_names)[1]
m_federer = trained_params[1][idx_federer]
ls_federer = trained_params[2][idx_federer]

m_nadal = trained_params[1][idx_nadal]
ls_nadal = trained_params[2][idx_nadal]

z_score = (0-(m_federer-m_nadal))/exp(ls_federer)
p_exact = 1-cdf(Normal(0, 1), z_score)

#monte carlo for federer vs nadal
n=10000
sample_federer = randn(1,1000) .* exp(ls_federer) .+ m_federer
sample_last = randn(1,1000) .* exp(ls_nadal) .+ m_nadal
indicator = sample_federer .> sample_last
mean(indicator)

#exact prob of skill_federer > skill_last_ranked from learned posterior
idx_ordered = sortperm(trained_params[1])
idx_last = idx_ordered[1]
last_player = player_names[idx_last]
m_last = trained_params[1][idx_last]
ls_last = trained_params[2][idx_last]
idx_federer = findall(x-> x =="Roger-Federer", player_names)[1]
m_federer = trained_params[1][idx_federer]
ls_federer = trained_params[2][idx_federer]
z_score = (0-(m_federer-m_last))/exp(ls_federer)
p_exact = 1-cdf(Normal(0, 1), z_score)
#monte carlo for federer vs last ranked player
n=10000
sample_federer = randn(1,1000) .* exp(ls_federer) .+ m_federer
sample_last = randn(1,1000) .* exp(ls_last) .+ m_last
indicator = sample_federer .> sample_last
mean(indicator)
