using LLVM
using LLVM.Interop

const globalUnique = Ref{Int64}(0)

function lookup_sig(@nospecialize(func), @nospecialize(tt))
    isa(func, Core.Builtin) && error("function is not a generic function")
    return Base.signature_type(func, tt)::Type
end

function genMod(@nospecialize(sig))
    world = typemax(UInt)
    m = ccall(:jl_gf_invoke_lookup, Any, (Any, UInt), sig, world)
    meth = m.func::Method

    (ti, env) = ccall(:jl_type_intersection_with_env, Any,
                      (Any, Any), sig, meth.sig)::Core.SimpleVector
    if VERSION >= v"1.2.0-DEV.320"
        meth = Base.func_for_method_checked(meth, ti, env)
    else
        meth = Base.func_for_method_checked(meth, ti)
    end

    linfo = ccall(:jl_specializations_get_linfo, Ref{Core.MethodInstance},
                  (Any, Any, Any, UInt), meth, ti, env, world)

    dependencies = Vector{LLVM.Module}()
    function hook_module_activation(ref::Ptr{Cvoid})
        ref = convert(LLVM.API.LLVMModuleRef, ref)
        push!(dependencies, LLVM.Module(ref))
    end

    params = Base.CodegenParams(cached            = false,
                                prefer_specsig    = true,
                                static_alloc      = false,
                                module_activation = hook_module_activation,
                                )

    # get the code
    mod = let
        ref = ccall(:jl_get_llvmf_defn, LLVM.API.LLVMValueRef,
                    (Any, UInt, Bool, Bool, Base.CodegenParams),
                    linfo, world, #=wrapper=#false, #=optimize=#false, params)
        if ref == C_NULL
            throw(CompilerError(ctx, "the Julia compiler could not generate LLVM IR"))
        end

        llvmf = LLVM.Function(ref)
        LLVM.parent(llvmf)
    end

    return mod, dependencies
end

function irgen(@nospecialize(func), @nospecialize(tt))
    sig = lookup_sig(func, tt)
    irgen(sig)
end

function irgen(@nospecialize(sig))
    mod, dependencies = genMod(sig)

    # the main module should contain a single jfptr_ function definition,
    # e.g. jfptr_kernel_vadd_62977
    definitions = LLVM.Function[]

    for llvmf in functions(mod)
        if !isdeclaration(llvmf)
            push!(definitions, llvmf)
        end
    end

    wrapper = nothing
    for llvmf in definitions
        if startswith(LLVM.name(llvmf), "jfptr_")
            @assert wrapper == nothing
            wrapper = llvmf
        end
    end
    @assert wrapper != nothing


    # the jfptr wrapper function should point us to the actual entry-point,
    # e.g. julia_kernel_vadd_62984
    # FIXME: Julia's globalUnique starting with `-` is probably a bug.
    entry_tag = let
        m = match(r"^jfptr_(.+)_[-\d]+$", LLVM.name(wrapper))
        if m == nothing
            error(LLVM.name(wrapper))
        end
        m.captures[1]
    end
    unsafe_delete!(mod, wrapper)
    entry = let
        re = Regex("^julia_$(entry_tag)_[-\\d]+\$")
        entrypoints = LLVM.Function[]
        for llvmf in definitions
            if llvmf != wrapper
                llvmfn = LLVM.name(llvmf)
                if occursin(re, llvmfn)
                    push!(entrypoints, llvmf)
                end
            end
        end
        if length(entrypoints) != 1
            @warn ":cry:" functions=Tuple(LLVM.name.(definitions)) tag=entry_tag entrypoints=Tuple(LLVM.name.(entrypoints))
        end
        entrypoints[1]
    end

    # link in dependent modules
    for dep in dependencies
        link!(mod, dep)
    end

    # remove other fptr wrappers
    for llvmf in functions(mod)
        if !isdeclaration(llvmf) && startswith(LLVM.name(llvmf), "jfptr_")
            unsafe_delete!(mod, llvmf)
        end
    end

    # rename the entry point
    llvmfn = replace(LLVM.name(entry), r"_\d+$"=>"")

    ## append a global unique counter
    globalUnique[] += 1
    llvmfn *= "_$(globalUnique[])"
    LLVM.name!(entry, llvmfn)

    return mod, entry
end
@generated function testgen(f, args...)
    # Obtain the function and all it's dependencies in one handy module
    diffetypes = []
    autodifftypes = Type[f, args[1]]
    mod, ccf = irgen(Tuple{autodifftypes...}) 
end

function sqcube(x::Array{Float64,1})
    sum(x .* x)
end

ar = [1.0, 2.0]
arp = [0.0, 0.0]
@show sqcube(ar)
mod, ccf = irgen(Tuple{typeof(sqcube), Array{Float64,1}})
@show mod
#@show testgen(sqcube, ar, arp)
