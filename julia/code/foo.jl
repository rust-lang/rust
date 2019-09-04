using InteractiveUtils


function loss(X :: Float64)
    return X
end

const closs = @cfunction(loss, Float64, (Float64,) )
@code_llvm loss(1.0)
res = ccall( "llvm.autodiff", llvmcall, Float64, (Ptr{Cvoid}, Float64), closs, 3.2 )

@show(res)

