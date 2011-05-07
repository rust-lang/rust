#include "llvm/Analysis/Passes.h"
#include "llvm/Support/StandardPasses.h"
#include "llvm/PassManager.h"
#include "llvm-c/Core.h"
#include <cstdlib>

using namespace llvm;

extern "C" void LLVMAddStandardFunctionPasses(LLVMPassManagerRef PM,
    unsigned int OptimizationLevel) {
  createStandardFunctionPasses(unwrap(PM), OptimizationLevel);
}

extern "C" void LLVMAddStandardModulePasses(LLVMPassManagerRef PM,
    unsigned int OptimizationLevel, LLVMBool OptimizeSize,
    LLVMBool UnitAtATime, LLVMBool UnrollLoops, LLVMBool SimplifyLibCalls,
    LLVMBool HaveExceptions, unsigned int InliningThreshold) {
  Pass *InliningPass;
  if (InliningThreshold)
    InliningPass = createFunctionInliningPass(InliningThreshold);
  else
    InliningPass = NULL;

  createStandardModulePasses(unwrap(PM), OptimizationLevel, OptimizeSize,
                             UnitAtATime, UnrollLoops, SimplifyLibCalls,
                             HaveExceptions, InliningPass);
}


