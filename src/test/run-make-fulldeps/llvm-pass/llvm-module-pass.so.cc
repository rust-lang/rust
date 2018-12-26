#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "llvm/IR/Module.h"

using namespace llvm;

namespace {

  class TestLLVMPass : public ModulePass {

  public:

    static char ID;
    TestLLVMPass() : ModulePass(ID) { }

    bool runOnModule(Module &M) override;

    StringRef getPassName() const override {
      return "Some LLVM pass";
    }

  };

}

bool TestLLVMPass::runOnModule(Module &M) {
  // A couple examples of operations that previously caused segmentation faults
  // https://github.com/rust-lang/rust/issues/31067

  for (auto F = M.begin(); F != M.end(); ++F) {
    /* code */
  }

  LLVMContext &C = M.getContext();
  IntegerType *Int8Ty  = IntegerType::getInt8Ty(C);
  PointerType::get(Int8Ty, 0);
  return true;
}

char TestLLVMPass::ID = 0;

static RegisterPass<TestLLVMPass> RegisterAFLPass(
  "some-llvm-module-pass", "Some LLVM pass");
