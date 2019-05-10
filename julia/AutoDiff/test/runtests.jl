using AutoDiff

function jl_f2(f::Float64)
    f + 1.0
end
@show autodiff(jl_f2, Float64, 1.0)
