// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#include <stdio.h>

#include "rustllvm.h"

#include "llvm/Support/CBindingWrapping.h"
#include "llvm/Target/TargetLibraryInfo.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"

#include "llvm-c/Transforms/PassManagerBuilder.h"

using namespace llvm;

extern cl::opt<bool> EnableARMEHABI;

typedef struct LLVMOpaquePass *LLVMPassRef;
typedef struct LLVMOpaqueTargetMachine *LLVMTargetMachineRef;

DEFINE_STDCXX_CONVERSION_FUNCTIONS(Pass, LLVMPassRef)
DEFINE_STDCXX_CONVERSION_FUNCTIONS(TargetMachine, LLVMTargetMachineRef)
DEFINE_STDCXX_CONVERSION_FUNCTIONS(PassManagerBuilder, LLVMPassManagerBuilderRef)

extern "C" void
LLVMInitializePasses() {
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

extern "C" bool
LLVMRustAddPass(LLVMPassManagerRef PM, const char *PassName) {
    PassManagerBase *pm = unwrap(PM);

    StringRef SR(PassName);
    PassRegistry *PR = PassRegistry::getPassRegistry();

    const PassInfo *PI = PR->getPassInfo(SR);
    if (PI) {
        pm->add(PI->createPass());
        return true;
    }
    return false;
}

extern "C" LLVMTargetMachineRef
LLVMRustCreateTargetMachine(const char *triple,
                            const char *cpu,
                            const char *feature,
                            CodeModel::Model CM,
                            Reloc::Model RM,
                            CodeGenOpt::Level OptLevel,
                            bool EnableSegmentedStacks,
                            bool UseSoftFloat,
                            bool NoFramePointerElim) {
    std::string Error;
    Triple Trip(Triple::normalize(triple));
    const llvm::Target *TheTarget = TargetRegistry::lookupTarget(Trip.getTriple(),
                                                                 Error);
    if (TheTarget == NULL) {
        LLVMRustError = Error.c_str();
        return NULL;
    }

    TargetOptions Options;
    Options.NoFramePointerElim = NoFramePointerElim;
#if LLVM_VERSION_MINOR < 5
    Options.EnableSegmentedStacks = EnableSegmentedStacks;
#endif
    Options.FloatABIType = FloatABI::Default;
    Options.UseSoftFloat = UseSoftFloat;
    if (UseSoftFloat) {
        Options.FloatABIType = FloatABI::Soft;
    }

    TargetMachine *TM = TheTarget->createTargetMachine(Trip.getTriple(),
                                                       cpu,
                                                       feature,
                                                       Options,
                                                       RM,
                                                       CM,
                                                       OptLevel);
    return wrap(TM);
}

extern "C" void
LLVMRustDisposeTargetMachine(LLVMTargetMachineRef TM) {
    delete unwrap(TM);
}

// Unfortunately, LLVM doesn't expose a C API to add the corresponding analysis
// passes for a target to a pass manager. We export that functionality through
// this function.
extern "C" void
LLVMRustAddAnalysisPasses(LLVMTargetMachineRef TM,
                          LLVMPassManagerRef PMR,
                          LLVMModuleRef M) {
    PassManagerBase *PM = unwrap(PMR);
#if LLVM_VERSION_MINOR >= 5
    PM->add(new DataLayoutPass(unwrap(M)));
#else
    PM->add(new DataLayout(unwrap(M)));
#endif
    unwrap(TM)->addAnalysisPasses(*PM);
}

// Unfortunately, the LLVM C API doesn't provide a way to set the `LibraryInfo`
// field of a PassManagerBuilder, we expose our own method of doing so.
extern "C" void
LLVMRustAddBuilderLibraryInfo(LLVMPassManagerBuilderRef PMB, LLVMModuleRef M) {
    Triple TargetTriple(unwrap(M)->getTargetTriple());
    unwrap(PMB)->LibraryInfo = new TargetLibraryInfo(TargetTriple);
}

// Unfortunately, the LLVM C API doesn't provide a way to create the
// TargetLibraryInfo pass, so we use this method to do so.
extern "C" void
LLVMRustAddLibraryInfo(LLVMPassManagerRef PMB, LLVMModuleRef M) {
    Triple TargetTriple(unwrap(M)->getTargetTriple());
    unwrap(PMB)->add(new TargetLibraryInfo(TargetTriple));
}

// Unfortunately, the LLVM C API doesn't provide an easy way of iterating over
// all the functions in a module, so we do that manually here. You'll find
// similar code in clang's BackendUtil.cpp file.
extern "C" void
LLVMRustRunFunctionPassManager(LLVMPassManagerRef PM, LLVMModuleRef M) {
    FunctionPassManager *P = unwrap<FunctionPassManager>(PM);
    P->doInitialization();
    for (Module::iterator I = unwrap(M)->begin(),
         E = unwrap(M)->end(); I != E; ++I)
        if (!I->isDeclaration())
            P->run(*I);
    P->doFinalization();
}

extern "C" void
LLVMRustSetLLVMOptions(int Argc, char **Argv) {
    // Initializing the command-line options more than once is not allowed. So,
    // check if they've already been initialized.  (This could happen if we're
    // being called from rustpkg, for example). If the arguments change, then
    // that's just kinda unfortunate.
    static bool initialized = false;
    if (initialized) return;
    initialized = true;
    cl::ParseCommandLineOptions(Argc, Argv);
}

extern "C" bool
LLVMRustWriteOutputFile(LLVMTargetMachineRef Target,
                        LLVMPassManagerRef PMR,
                        LLVMModuleRef M,
                        const char *path,
                        TargetMachine::CodeGenFileType FileType) {
  PassManager *PM = unwrap<PassManager>(PMR);

  std::string ErrorInfo;
#if LLVM_VERSION_MINOR >= 4
  raw_fd_ostream OS(path, ErrorInfo, sys::fs::F_None);
#else
  raw_fd_ostream OS(path, ErrorInfo, raw_fd_ostream::F_Binary);
#endif
  if (ErrorInfo != "") {
    LLVMRustError = ErrorInfo.c_str();
    return false;
  }
  formatted_raw_ostream FOS(OS);

  unwrap(Target)->addPassesToEmitFile(*PM, FOS, FileType, false);
  PM->run(*unwrap(M));
  return true;
}

extern "C" void
LLVMRustPrintModule(LLVMPassManagerRef PMR,
                    LLVMModuleRef M,
                    const char* path) {
  PassManager *PM = unwrap<PassManager>(PMR);
  std::string ErrorInfo;

#if LLVM_VERSION_MINOR >= 4
  raw_fd_ostream OS(path, ErrorInfo, sys::fs::F_None);
#else
  raw_fd_ostream OS(path, ErrorInfo, raw_fd_ostream::F_Binary);
#endif

  formatted_raw_ostream FOS(OS);

#if LLVM_VERSION_MINOR >= 5
  PM->add(createPrintModulePass(FOS));
#else
  PM->add(createPrintModulePass(&FOS));
#endif

  PM->run(*unwrap(M));
}

extern "C" void
LLVMRustPrintPasses() {
    LLVMInitializePasses();
    struct MyListener : PassRegistrationListener {
        void passEnumerate(const PassInfo *info) {
            if (info->getPassArgument() && *info->getPassArgument()) {
                printf("%15s - %s\n", info->getPassArgument(),
                       info->getPassName());
            }
        }
    } listener;

    PassRegistry *PR = PassRegistry::getPassRegistry();
    PR->enumerateWith(&listener);
}

extern "C" void
LLVMRustAddAlwaysInlinePass(LLVMPassManagerBuilderRef PMB, bool AddLifetimes) {
    unwrap(PMB)->Inliner = createAlwaysInlinerPass(AddLifetimes);
}

extern "C" void
LLVMRustRunRestrictionPass(LLVMModuleRef M, char **symbols, size_t len) {
    PassManager passes;
    ArrayRef<const char*> ref(symbols, len);
    passes.add(llvm::createInternalizePass(ref));
    passes.run(*unwrap(M));
}

extern "C" void
LLVMRustMarkAllFunctionsNounwind(LLVMModuleRef M) {
    for (Module::iterator GV = unwrap(M)->begin(),
         E = unwrap(M)->end(); GV != E; ++GV) {
        GV->setDoesNotThrow();
        Function *F = dyn_cast<Function>(GV);
        if (F == NULL)
            continue;

        for (Function::iterator B = F->begin(), BE = F->end(); B != BE; ++B) {
            for (BasicBlock::iterator I = B->begin(), IE = B->end();
                 I != IE; ++I) {
                if (isa<InvokeInst>(I)) {
                    InvokeInst *CI = cast<InvokeInst>(I);
                    CI->setDoesNotThrow();
                }
            }
        }
    }
}
