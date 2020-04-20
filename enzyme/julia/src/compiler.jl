module Compiler

using GPUCompiler
using LLVM
using LLVM.Interop

import GPUCompiler: FunctionSpec, codegen

import Libdl
llvmver = LLVM.version().major
if haskey(ENV, "ENZYME_PATH")
    enzyme_path = ENV["ENZYME_PATH"]
else
    error("Please set the environment variable ENZYME_PATH")
end
const libenzyme = abspath(joinpath(enzyme_path, "LLVMEnzyme-$(llvmver).$(Libdl.dlext)"))

if !isfile(libenzyme)
    error("$(libenzyme) does not exist, Please specify a correct path in ENZYME_PATH, and restart Julia.")
end

if Libdl.dlopen_e(libenzyme) in (C_NULL, nothing)
    error("$(libenzyme) cannot be opened, Please specify a correct path in ENZYME_PATH, and restart Julia.")
end

function __init__()
    Libdl.dlopen(libenzyme, Libdl.RTLD_GLOBAL)
    LLVM.clopts("-enzyme_preopt=0")
end

# Define EnzymeTarget & EnzymeJob
using LLVM: triple, Target, TargetMachine
import GPUCompiler: llvm_triple

Base.@kwdef struct EnzymeTarget <: AbstractCompilerTarget
end

GPUCompiler.isintrinsic(::EnzymeTarget, fn::String) = true
GPUCompiler.can_throw(::EnzymeTarget) = true

llvm_triple(::EnzymeTarget) = triple()

# GPUCompiler.llvm_datalayout(::EnzymeTarget) =  nothing

function GPUCompiler.llvm_machine(target::EnzymeTarget)
    t = Target(llvm_triple(target))
    tm = TargetMachine(t, llvm_triple(target))
    LLVM.asm_verbosity!(tm, true)

    return tm
end

module Runtime
    # the runtime library
    signal_exception() = return
    malloc(sz) =  return
    report_oom(sz) = return
    report_exception(ex) = return
    report_exception_name(ex) = return
    report_exception_frame(idx, func, file, line) = return
end

GPUCompiler.runtime_module(target::EnzymeTarget) = Runtime

## job

export EnzymeJob

Base.@kwdef struct EnzymeJob <: AbstractCompilerJob
    target::EnzymeTarget
    source::FunctionSpec
end

import GPUCompiler: target, source
target(job::EnzymeJob) = job.target
source(job::EnzymeJob) = job.source

Base.similar(job::EnzymeJob, source::FunctionSpec) =
    EnzymeJob(target=job.target, source=source)

function Base.show(io::IO, job::EnzymeJob)
    print(io, "Enzyme CompilerJob of ", GPUCompiler.source(job))
end

# TODO: encode debug build or not in the compiler job
#       https://github.com/JuliaGPU/CUDAnative.jl/issues/368
GPUCompiler.runtime_slug(job::EnzymeJob) = "enzyme" 

include("compiler/optimize.jl")

end