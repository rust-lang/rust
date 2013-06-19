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

extern "C" void LLVMInitializePasses() {
  PassRegistry &Registry = *PassRegistry::getPassRegistry();
  initializeCore(Registry);
  initializeCodeGen(Registry);
  initializeScalarOpts(Registry);
  initializeVectorization(Registry);
  initializeIPO(Registry);
  initializeAnalysis(Registry);
  initializeIPA(Registry);
  initializeTransformUtils(Registry);
  initializeInstCombine(Registry);
  initializeInstrumentation(Registry);
  initializeTarget(Registry);
}

extern "C" void LLVMAddPass(LLVMPassManagerRef PM, LLVMPassRef P) {
    PassManagerBase * pm = unwrap(PM);
    Pass * p = unwrap(P);

    pm->add(p);
}

extern "C" LLVMPassRef LLVMCreatePass(const char * PassName) {
    StringRef SR(PassName);
    PassRegistry * PR = PassRegistry::getPassRegistry();

    const PassInfo * PI = PR->getPassInfo(SR);
    if (PI) {
        return wrap(PI->createPass());
    } else {
        return (LLVMPassRef)0;
    }
}

extern "C" void LLVMDestroyPass(LLVMPassRef PassRef) {
    Pass *p = unwrap(PassRef);
    delete p;
}
