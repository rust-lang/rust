#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "llvm/Pass.h"
#include "llvm/IR/Function.h"

using namespace llvm;

namespace {

  class TestLLVMPass : public FunctionPass {

  public:

    static char ID;
    TestLLVMPass() : FunctionPass(ID) { }

    bool runOnFunction(Function &F) override;

    StringRef getPassName() const override {
      return "Some LLVM pass";
    }

  };

}

bool TestLLVMPass::runOnFunction(Function &F) {
  // A couple examples of operations that previously caused segmentation faults
  // https://github.com/rust-lang/rust/issues/31067

  for (auto N = F.begin(); N != F.end(); ++N) {
    /* code */
  }

  LLVMContext &C = F.getContext();
  IntegerType *Int8Ty  = IntegerType::getInt8Ty(C);
  PointerType::get(Int8Ty, 0);
  return true;
}

char TestLLVMPass::ID = 0;

static RegisterPass<TestLLVMPass> RegisterAFLPass(
  "some-llvm-function-pass", "Some LLVM pass");
