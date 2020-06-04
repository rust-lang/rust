using Enzyme
using Test
# using ReverseDiff

@testset "Internal tests" begin
    f(x) = 1.0 + x
    thunk = Enzyme.Thunk(f, Float64, (Float64,))
end

@testset "Simple tests" begin
    f1(x) = 1.0 + x
    f2(x) = x*x
    @test autodiff(f1, 1.0) ≈ 1.0
    @test autodiff(f2, 1.0) ≈ 2.0
end

@testset "Taylor series tests" begin

# Taylor series for `-log(1-x)`
# eval at -log(1-1/2) = -log(1/2)
function euroad(f::T) where T
    g = zero(T)
    for i in 1:10^7
        g += f^i / i
    end
    return g
end
euroad′(x) = autodiff(euroad, x)

@test euroad(0.5) ≈ -log(0.5) # -log(1-x)
@show euroad′(0.5)
@test euroad′(0.5) ≈ 2.0 # d/dx -log(1-x) = 1/(1-x)

end
