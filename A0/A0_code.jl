using Test
using Distributions

function gaussian_pdf(x; mean=0., variance=0.01)
    1. / ((2 * pi * variance)^0.5) * exp(-(x - mean)^2 / (2 * variance))
end;

x = randn()
@test gaussian_pdf(x) ≈ pdf.(Normal(0.,sqrt(0.01)),x)
@test isapprox(gaussian_pdf(x,mean=10., variance=1) ,
    pdf.(Normal(10., sqrt(1)),x))

function sample_gaussian(n; mean=0., variance=0.01)
    # from Normal(0,1):
    x = randn(n)
    sqrt(variance).* x .+ mean
    # #alternatively using central limit theorem:
    # #N(0,1) = 1/m ∑(x_i-μ)/sqrt(m σ^2)
    # ys = zeros(n)
    # m = 100
    # #X_i from uniform(0,1):
    # #uniform_variance = 1/12.*(1-0)^2=1/12.
    # #uniform_mean = 0.5
    # #N(0,1) = sqrt(12/m)∑(x_i) - sqrt(12m)(0.5)
    # for i=1:n
    #     xs = rand(Float64,m)
    #     ys[i] = sqrt(12. / m) * sum(xs) - sqrt(12. * m)*0.5
    # end;
    # sqrt(variance).*ys.+mean
end;

using Statistics: mean, var
@testset "Numerically testing Gaussian Sample Statistics" begin
    samples = sample_gaussian(100000, mean=5., variance=3.5);
    m = mean(samples)
    v = var(samples)
    @test isapprox(m, 5, atol=1e-2)
    @test isapprox(v, 3.5, atol=1e-2)
end;

samples = sample_gaussian(100000, mean=10., variance=2.);

using Plots

histogram(samples,normalize=true,
    label="100000 outputs from sample_gaussian()",
    legend = :bottomright)

dist = x -> pdf(Normal(10.,sqrt(2.)),x)
plot!(dist, 2, 18,
    label="pdf of normal distribution",
    title="Gaussian Sampler Distribution Check (mu=10,sigma^2=2)",
    xlabel="x",
    ylabel="pdf(x), histogram fraction",
    linewidth = 2,
    linecolor = :red)

savefig("sample_gaussian_pdf.png")

using LinearAlgebra;
# Choose dimensions of toy data
m = 8
n = 11

x = rand(m,)
y = rand(m,)
A = rand(m,n)
B = rand(m,m)

# Make sure your toy data is the size you expect!
@testset "Sizes of Toy Data" begin
    @test (m,) == size(x)
    @test (m,) == size(y)
    @test (m,n) == size(A)
    @test (m,m) == size(B)
end;

using Zygote;

@testset "derivative checks" begin
    g1 = gradient((x, y) -> x' * y, x, y)[1];
    g2 = gradient(x-> x' * x, x)[1];
    g3 = zeros(n,m);
    for i=1:size(A)[2] #1 to n
        g3[i,:] = gradient((x, A) -> (x' * A)[i], x, A)[1]
    end;
    g4 = gradient((x, B) -> x' * B * x, x, B)[1];
    @test g1 ≈ y
    @test g2 ≈ 2 * x
    @test g3 ≈ A'
    @test g4 ≈ B*x + B'*x
end;
