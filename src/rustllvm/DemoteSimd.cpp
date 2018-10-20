// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#include <vector>
#include <set>

#include "rustllvm.h"

#if LLVM_VERSION_GE(5, 0)

#include "llvm/IR/CallSite.h"
#include "llvm/IR/Module.h"
#include "llvm/ADT/STLExtras.h"

using namespace llvm;

static std::vector<Function*>
GetFunctionsWithSimdArgs(Module *M) {
  std::vector<Function*> Ret;

  for (auto &F : M->functions()) {
    // Skip all intrinsic calls as these are always tightly controlled to "work
    // correctly", so no need to fixup any of these.
    if (F.isIntrinsic())
      continue;

    // We're only interested in rustc-defined functions, not unstably-defined
    // imported SIMD ffi functions.
    if (F.isDeclaration())
      continue;

    // Argument promotion only happens on internal functions, so skip demoting
    // arguments in external functions like FFI shims and such.
    if (!F.hasLocalLinkage())
      continue;

    // If any argument to this function is a by-value vector type, then that's
    // bad! The compiler didn't generate any functions that looked like this,
    // and we try to rely on LLVM to not do this! Argument promotion may,
    // however, promote arguments from behind references. In any case, figure
    // out if we're interested in demoting this argument.
    if (any_of(F.args(), [](Argument &arg) { return arg.getType()->isVectorTy(); }))
      Ret.push_back(&F);
  }

  return Ret;
}

extern "C" void
LLVMRustDemoteSimdArguments(LLVMModuleRef Mod) {
  Module *M = unwrap(Mod);

  auto Functions = GetFunctionsWithSimdArgs(M);

  for (auto F : Functions) {
    // Build up our list of new parameters and new argument attributes.
    // We're only changing those arguments which are vector types.
    SmallVector<Type*, 8> Params;
    SmallVector<AttributeSet, 8> ArgAttrVec;
    auto PAL = F->getAttributes();
    for (auto &Arg : F->args()) {
      auto *Ty = Arg.getType();
      if (Ty->isVectorTy()) {
        Params.push_back(PointerType::get(Ty, 0));
        ArgAttrVec.push_back(AttributeSet());
      } else {
        Params.push_back(Ty);
        ArgAttrVec.push_back(PAL.getParamAttributes(Arg.getArgNo()));
      }
    }

    // Replace `F` with a new function with our new signature. I'm... not really
    // sure how this works, but this is all the steps `ArgumentPromotion` does
    // to replace a signature as well.
    assert(!F->isVarArg()); // ArgumentPromotion should skip these fns
    FunctionType *NFTy = FunctionType::get(F->getReturnType(), Params, false);
    Function *NF = Function::Create(NFTy, F->getLinkage(), F->getName());
    NF->copyAttributesFrom(F);
    NF->setSubprogram(F->getSubprogram());
    F->setSubprogram(nullptr);
    NF->setAttributes(AttributeList::get(F->getContext(),
                                         PAL.getFnAttributes(),
                                         PAL.getRetAttributes(),
                                         ArgAttrVec));
    ArgAttrVec.clear();
    F->getParent()->getFunctionList().insert(F->getIterator(), NF);
    NF->takeName(F);

    // Iterate over all invocations of `F`, updating all `call` instructions to
    // store immediate vector types in a local `alloc` instead of a by-value
    // vector.
    //
    // Like before, much of this is copied from the `ArgumentPromotion` pass in
    // LLVM.
    SmallVector<Value*, 16> Args;
    while (!F->use_empty()) {
      CallSite CS(F->user_back());
      assert(CS.getCalledFunction() == F);
      Instruction *Call = CS.getInstruction();
      const AttributeList &CallPAL = CS.getAttributes();

      // Loop over the operands, inserting an `alloca` and a store for any
      // argument we're demoting to be by reference
      //
      // FIXME: we probably want to figure out an LLVM pass to run and clean up
      // this function and instructions we're generating, we should in theory
      // only generate a maximum number of `alloca` instructions rather than
      // one-per-variable unconditionally.
      CallSite::arg_iterator AI = CS.arg_begin();
      size_t ArgNo = 0;
      for (Function::arg_iterator I = F->arg_begin(), E = F->arg_end(); I != E;
           ++I, ++AI, ++ArgNo) {
        if (I->getType()->isVectorTy()) {
          AllocaInst *AllocA = new AllocaInst(I->getType(), 0, nullptr, "", Call);
          new StoreInst(*AI, AllocA, Call);
          Args.push_back(AllocA);
          ArgAttrVec.push_back(AttributeSet());
        } else {
          Args.push_back(*AI);
          ArgAttrVec.push_back(CallPAL.getParamAttributes(ArgNo));
        }
      }
      assert(AI == CS.arg_end());

      // Create a new call instructions which we'll use to replace the old call
      // instruction, copying over as many attributes and such as possible.
      SmallVector<OperandBundleDef, 1> OpBundles;
      CS.getOperandBundlesAsDefs(OpBundles);

      CallSite NewCS;
      if (InvokeInst *II = dyn_cast<InvokeInst>(Call)) {
        InvokeInst::Create(NF, II->getNormalDest(), II->getUnwindDest(),
                           Args, OpBundles, "", Call);
      } else {
        auto *NewCall = CallInst::Create(NF, Args, OpBundles, "", Call);
        NewCall->setTailCallKind(cast<CallInst>(Call)->getTailCallKind());
        NewCS = NewCall;
      }
      NewCS.setCallingConv(CS.getCallingConv());
      NewCS.setAttributes(
          AttributeList::get(F->getContext(), CallPAL.getFnAttributes(),
                             CallPAL.getRetAttributes(), ArgAttrVec));
      NewCS->setDebugLoc(Call->getDebugLoc());
      Args.clear();
      ArgAttrVec.clear();
      Call->replaceAllUsesWith(NewCS.getInstruction());
      NewCS->takeName(Call);
      Call->eraseFromParent();
    }

    // Splice the body of the old function right into the new function.
    NF->getBasicBlockList().splice(NF->begin(), F->getBasicBlockList());

    // Update our new function to replace all uses of the by-value argument with
    // loads of the pointer argument we've generated.
    //
    // FIXME: we probably want to only generate one load instruction per
    // function? Or maybe run an LLVM pass to clean up this function?
    for (Function::arg_iterator I = F->arg_begin(),
                                E = F->arg_end(),
                                I2 = NF->arg_begin();
         I != E;
         ++I, ++I2) {
      if (I->getType()->isVectorTy()) {
        I->replaceAllUsesWith(new LoadInst(&*I2, "", &NF->begin()->front()));
      } else {
        I->replaceAllUsesWith(&*I2);
      }
      I2->takeName(&*I);
    }

    // Delete all references to the old function, it should be entirely dead
    // now.
    M->getFunctionList().remove(F);
  }
}

#else // LLVM_VERSION_GE(8, 0)
extern "C" void
LLVMRustDemoteSimdArguments(LLVMModuleRef Mod) {
}
#endif // LLVM_VERSION_GE(8, 0)
