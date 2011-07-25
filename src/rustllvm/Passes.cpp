#include "llvm/Analysis/Passes.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/PassManager.h"
#include "llvm-c/Core.h"
#include <cstdlib>

using namespace llvm;

extern "C" void LLVMAddStandardFunctionPasses(LLVMPassManagerRef PM,
    unsigned int OptimizationLevel) {
  PassManagerBuilder PMBuilder;
  PMBuilder.OptLevel = OptimizationLevel;
  FunctionPassManager *FPM = (FunctionPassManager*) unwrap(PM);
  PMBuilder.populateFunctionPassManager(*FPM);
}

extern "C" void LLVMAddStandardModulePasses(LLVMPassManagerRef PM,
    unsigned int OptimizationLevel, LLVMBool OptimizeSize,
    LLVMBool UnitAtATime, LLVMBool UnrollLoops, LLVMBool SimplifyLibCalls,
    unsigned int InliningThreshold) {

  PassManagerBuilder PMBuilder;
  PMBuilder.OptLevel = OptimizationLevel;
  PMBuilder.SizeLevel = OptimizeSize;
  PMBuilder.DisableUnitAtATime = !UnitAtATime;
  PMBuilder.DisableUnrollLoops = !UnrollLoops;

  PMBuilder.DisableSimplifyLibCalls = !SimplifyLibCalls;

  if (InliningThreshold)
    PMBuilder.Inliner = createFunctionInliningPass(InliningThreshold);

  PassManager *MPM = (PassManager*) unwrap(PM);
  PMBuilder.populateModulePassManager(*MPM);
}

