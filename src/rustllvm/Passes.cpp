#include "llvm/Analysis/Passes.h"
#include "llvm/PassManager.h"
#include "llvm-c/Core.h"

using namespace llvm;
extern "C" {
  void LLVMAddTypeBasedAliasAnalysisPass(LLVMPassManagerRef PM) {
    unwrap(PM)->add(createTypeBasedAliasAnalysisPass());
  }

  void LLVMAddBasicAliasAnalysisPass(LLVMPassManagerRef PM) {
    unwrap(PM)->add(createBasicAliasAnalysisPass());
  }
}
