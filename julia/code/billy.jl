using LLVM
using LLVM.Interop
using LLVM.Core

Base.@ccallable Cvoid function jl_f()
    println("Hello")
end

Base.@ccallable Cdouble function jl_f2(i::Float64)
    println("Hello $i")
    i
end

@generated function autodiff(f::F, ::Type{T}, args...) where {F, T}
    fname = String(nameof(f))[2:end] # weird pound in front of name
    rettype = convert(LLVMType, T)
    argtypes = LLVMType[convert(LLVMType, T) for T in args]

    ctx = JuliaContext()
    mod = LLVM.Module("autodiff", ctx)

    ft  = LLVM.FunctionType(rettype, argtypes)
    ccf = LLVM.Function(mod, fname, ft)

    llvmf = LLVM.Function(mod, "", ft)
    push!(function_attributes(llvmf), EnumAttribute("alwaysinline", 0, ctx))
    linkage!(llvmf, LLVM.API.LLVMPrivateLinkage)

    pt = LLVM.PointerType(LLVM.Int8Type(ctx))
    ftd  = LLVM.FunctionType(rettype, LLVMType[pt], true)
    llvmdif = LLVM.Function(mod, "llvm.autodiff.p0i8", ftd)

    Builder(ctx) do builder
        entry = BasicBlock(llvmf, "entry", ctx)
        position!(builder, entry)

        #a0 = collect{Value}(parameters(llvmf))
        a0 = Value[]
        for p in parameters(llvmf)
            push!(a0, p)
        end
        tc = bitcast!(builder, ccf, pt)
        pushfirst!(a0, tc)

        val = call!(builder, llvmdif, a0)
        if T === Nothing
            ret!(builder)
        else
            ret!(builder, val)
        end
    end

    _args = (:(args[$i]) for i in 1:length(args))
    call_function(llvmf, T, Tuple{args...}, Expr(:tuple, _args...))
end

# autodiff(jl_f, Cvoid)
autodiff(jl_f2, Cdouble, 1.0)

@show(res)
