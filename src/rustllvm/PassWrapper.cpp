// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#include "rustllvm.h"

using namespace llvm;

// Pass conversion fns
typedef struct LLVMOpaquePass *LLVMPassRef;

inline Pass *unwrap(LLVMPassRef P) {
    return reinterpret_cast<Pass*>(P);
}

inline LLVMPassRef wrap(const Pass *P) {
    return reinterpret_cast<LLVMPassRef>(const_cast<Pass*>(P));
}

template<typename T>
inline T *unwrap(LLVMPassRef P) {
    T *Q = (T*)unwrap(P);
    assert(Q && "Invalid cast!");
    return Q;
}

#define WRAP_PASS(name)                             \
    extern "C" LLVMPassRef LLVMCreate##name##Pass() {      \
        return wrap(llvm::create##name##Pass());    \
    }

extern "C" void LLVMAddPass(LLVMPassManagerRef PM, LLVMPassRef P) {
    PassManagerBase * pm = unwrap(PM);
    Pass * p = unwrap(P);

    pm->add(p);
}

////////////////
// Transforms //
// /////////////

// IPO Passes
WRAP_PASS(StripSymbols)
WRAP_PASS(StripNonDebugSymbols)
WRAP_PASS(StripDebugDeclare)
WRAP_PASS(StripDeadDebugInfo)
WRAP_PASS(ConstantMerge)
WRAP_PASS(GlobalOptimizer)
WRAP_PASS(GlobalDCE)
WRAP_PASS(AlwaysInliner)
WRAP_PASS(PruneEH)
WRAP_PASS(Internalize)
WRAP_PASS(DeadArgElimination)
WRAP_PASS(DeadArgHacking)
WRAP_PASS(ArgumentPromotion)
WRAP_PASS(IPConstantPropagation)
WRAP_PASS(IPSCCP)
WRAP_PASS(LoopExtractor)
WRAP_PASS(SingleLoopExtractor)
WRAP_PASS(BlockExtractor)
WRAP_PASS(StripDeadPrototypes)
WRAP_PASS(FunctionAttrs)
WRAP_PASS(MergeFunctions)
WRAP_PASS(PartialInlining)
WRAP_PASS(MetaRenamer)
WRAP_PASS(BarrierNoop)

extern "C" LLVMPassRef LLVMCreateFunctionInliningPass(int Threshold) {
    return wrap(llvm::createFunctionInliningPass(Threshold));
}

// Instrumentation Passes
WRAP_PASS(EdgeProfiler)
WRAP_PASS(OptimalEdgeProfiler)
WRAP_PASS(PathProfiler)
WRAP_PASS(GCOVProfiler)
WRAP_PASS(BoundsChecking)

// Scalar Passes
WRAP_PASS(ConstantPropagation)
WRAP_PASS(SCCP)
WRAP_PASS(DeadInstElimination)
WRAP_PASS(DeadCodeElimination)
WRAP_PASS(DeadStoreElimination)
WRAP_PASS(AggressiveDCE)
WRAP_PASS(SROA)
WRAP_PASS(ScalarReplAggregates)
WRAP_PASS(IndVarSimplify)
WRAP_PASS(InstructionCombining)
WRAP_PASS(LICM)
WRAP_PASS(LoopStrengthReduce)
WRAP_PASS(GlobalMerge)
WRAP_PASS(LoopUnswitch)
WRAP_PASS(LoopInstSimplify)
WRAP_PASS(LoopUnroll)
WRAP_PASS(LoopRotate)
WRAP_PASS(LoopIdiom)
WRAP_PASS(PromoteMemoryToRegister)
WRAP_PASS(DemoteRegisterToMemory)
WRAP_PASS(Reassociate)
WRAP_PASS(JumpThreading)
WRAP_PASS(CFGSimplification)
WRAP_PASS(BreakCriticalEdges)
WRAP_PASS(LoopSimplify)
WRAP_PASS(TailCallElimination)
WRAP_PASS(LowerSwitch)
WRAP_PASS(LowerInvoke)
WRAP_PASS(BlockPlacement)
WRAP_PASS(LCSSA)
WRAP_PASS(EarlyCSE)
WRAP_PASS(GVN)
WRAP_PASS(MemCpyOpt)
WRAP_PASS(LoopDeletion)
WRAP_PASS(SimplifyLibCalls)
WRAP_PASS(CodeGenPrepare)
WRAP_PASS(InstructionNamer)
WRAP_PASS(Sinking)
WRAP_PASS(LowerAtomic)
WRAP_PASS(CorrelatedValuePropagation)
WRAP_PASS(InstructionSimplifier)
WRAP_PASS(LowerExpectIntrinsic)

// Vectorize Passes
WRAP_PASS(BBVectorize)
WRAP_PASS(LoopVectorize)

//////////////
// Analyses //
//////////////

WRAP_PASS(GlobalsModRef)
WRAP_PASS(AliasAnalysisCounter)
WRAP_PASS(AAEval)
WRAP_PASS(NoAA)
WRAP_PASS(BasicAliasAnalysis)
WRAP_PASS(ScalarEvolutionAliasAnalysis)
WRAP_PASS(TypeBasedAliasAnalysis)
WRAP_PASS(ProfileLoader)
WRAP_PASS(ProfileMetadataLoader)
WRAP_PASS(NoProfileInfo)
WRAP_PASS(ProfileEstimator)
WRAP_PASS(ProfileVerifier)
WRAP_PASS(PathProfileLoader)
WRAP_PASS(NoPathProfileInfo)
WRAP_PASS(PathProfileVerifier)
WRAP_PASS(LazyValueInfo)
WRAP_PASS(DependenceAnalysis)
WRAP_PASS(CostModelAnalysis)
WRAP_PASS(InstCount)
WRAP_PASS(RegionInfo)
WRAP_PASS(ModuleDebugInfoPrinter)
WRAP_PASS(Lint)
WRAP_PASS(Verifier)
