// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

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

#if LLVM_VERSION_MAJOR >= 4
    StringRef
#else
    const char *
#endif
    getPassName() const override {
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
