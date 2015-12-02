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
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Host.h"
#if LLVM_VERSION_MINOR >= 7
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#else
#include "llvm/Target/TargetLibraryInfo.h"
#endif
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetSubtargetInfo.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"


#include "llvm-c/Transforms/PassManagerBuilder.h"

using namespace llvm;
using namespace llvm::legacy;

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
#if LLVM_VERSION_MINOR <= 7
  initializeIPA(Registry);
#endif
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
                            bool UseSoftFloat,
                            bool PositionIndependentExecutable,
                            bool FunctionSections,
                            bool DataSections) {
    std::string Error;
    Triple Trip(Triple::normalize(triple));
    const llvm::Target *TheTarget = TargetRegistry::lookupTarget(Trip.getTriple(),
                                                                 Error);
    if (TheTarget == NULL) {
        LLVMRustSetLastError(Error.c_str());
        return NULL;
    }

    StringRef real_cpu = cpu;
    if (real_cpu == "native") {
        real_cpu = sys::getHostCPUName();
    }

    TargetOptions Options;
    Options.PositionIndependentExecutable = PositionIndependentExecutable;
    Options.FloatABIType = FloatABI::Default;
    if (UseSoftFloat) {
        Options.FloatABIType = FloatABI::Soft;
    }
    Options.DataSections = DataSections;
    Options.FunctionSections = FunctionSections;

    TargetMachine *TM = TheTarget->createTargetMachine(Trip.getTriple(),
                                                       real_cpu,
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
#if LLVM_VERSION_MINOR >= 7
    PM->add(createTargetTransformInfoWrapperPass(
          unwrap(TM)->getTargetIRAnalysis()));
#else
#if LLVM_VERSION_MINOR == 6
    PM->add(new DataLayoutPass());
#else
    PM->add(new DataLayoutPass(unwrap(M)));
#endif
    unwrap(TM)->addAnalysisPasses(*PM);
#endif
}

extern "C" void
LLVMRustConfigurePassManagerBuilder(LLVMPassManagerBuilderRef PMB,
                                    CodeGenOpt::Level OptLevel,
                                    bool MergeFunctions,
                                    bool SLPVectorize,
                                    bool LoopVectorize) {
#if LLVM_VERSION_MINOR >= 6
    // Ignore mergefunc for now as enabling it causes crashes.
    //unwrap(PMB)->MergeFunctions = MergeFunctions;
#endif
    unwrap(PMB)->SLPVectorize = SLPVectorize;
    unwrap(PMB)->OptLevel = OptLevel;
    unwrap(PMB)->LoopVectorize = LoopVectorize;
}

// Unfortunately, the LLVM C API doesn't provide a way to set the `LibraryInfo`
// field of a PassManagerBuilder, we expose our own method of doing so.
extern "C" void
LLVMRustAddBuilderLibraryInfo(LLVMPassManagerBuilderRef PMB,
                              LLVMModuleRef M,
                              bool DisableSimplifyLibCalls) {
    Triple TargetTriple(unwrap(M)->getTargetTriple());
#if LLVM_VERSION_MINOR >= 7
    TargetLibraryInfoImpl *TLI = new TargetLibraryInfoImpl(TargetTriple);
#else
    TargetLibraryInfo *TLI = new TargetLibraryInfo(TargetTriple);
#endif
    if (DisableSimplifyLibCalls)
      TLI->disableAllFunctions();
    unwrap(PMB)->LibraryInfo = TLI;
}

// Unfortunately, the LLVM C API doesn't provide a way to create the
// TargetLibraryInfo pass, so we use this method to do so.
extern "C" void
LLVMRustAddLibraryInfo(LLVMPassManagerRef PMB,
                       LLVMModuleRef M,
                       bool DisableSimplifyLibCalls) {
    Triple TargetTriple(unwrap(M)->getTargetTriple());
#if LLVM_VERSION_MINOR >= 7
    TargetLibraryInfoImpl TLII(TargetTriple);
    if (DisableSimplifyLibCalls)
      TLII.disableAllFunctions();
    unwrap(PMB)->add(new TargetLibraryInfoWrapperPass(TLII));
#else
    TargetLibraryInfo *TLI = new TargetLibraryInfo(TargetTriple);
    if (DisableSimplifyLibCalls)
      TLI->disableAllFunctions();
    unwrap(PMB)->add(TLI);
#endif
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
                        bool skipCodegen,
                        TargetMachine::CodeGenFileType FileType) {
  PassManager *PM = unwrap<PassManager>(PMR);

  std::string ErrorInfo;
#if LLVM_VERSION_MINOR >= 6
  std::error_code EC;
  raw_fd_ostream OS(path, EC, sys::fs::F_None);
  if (EC)
    ErrorInfo = EC.message();
#else
  raw_fd_ostream OS(path, ErrorInfo, sys::fs::F_None);
#endif
  if (ErrorInfo != "") {
    LLVMRustSetLastError(ErrorInfo.c_str());
    return false;
  }

  // HACK: addPassesToEmitFile() also adds some codegen passes which are
  // MachinePasses that may modify IR in a way that it becomes invalid (we've
  // seen this with stack coloring). So the IR verifier would abort. Therefore,
  // when we want to emit more than one filetype in a single run, we want to
  // run the codegen passes and the verifier only for the first filetype.
  // Telling LLVM to only start adding passes after it has seen a pass that
  // doesn't exist allows us to achieve that.
  char notAPass;
  AnalysisID startBefore = skipCodegen ? (AnalysisID)&notAPass : nullptr;

#if LLVM_VERSION_MINOR >= 7
  unwrap(Target)->addPassesToEmitFile(*PM, OS, FileType, false, startBefore);
#else
  formatted_raw_ostream FOS(OS);
  unwrap(Target)->addPassesToEmitFile(*PM, FOS, FileType, false, startBefore);
#endif
  PM->run(*unwrap(M));

  // Apparently `addPassesToEmitFile` adds a pointer to our on-the-stack output
  // stream (OS), so the only real safe place to delete this is here? Don't we
  // wish this was written in Rust?
  delete PM;
  return true;
}

extern "C" void
LLVMRustPrintModule(LLVMPassManagerRef PMR,
                    LLVMModuleRef M,
                    const char* path) {
  PassManager *PM = unwrap<PassManager>(PMR);
  std::string ErrorInfo;

#if LLVM_VERSION_MINOR >= 6
  std::error_code EC;
  raw_fd_ostream OS(path, EC, sys::fs::F_None);
  if (EC)
    ErrorInfo = EC.message();
#else
  raw_fd_ostream OS(path, ErrorInfo, sys::fs::F_None);
#endif

  formatted_raw_ostream FOS(OS);

  PM->add(createPrintModulePass(FOS));

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

extern "C" void
LLVMRustSetDataLayoutFromTargetMachine(LLVMModuleRef Module,
                                       LLVMTargetMachineRef TMR) {
    TargetMachine *Target = unwrap(TMR);
#if LLVM_VERSION_MINOR >= 7
    unwrap(Module)->setDataLayout(Target->createDataLayout());
#elif LLVM_VERSION_MINOR >= 6
    if (const DataLayout *DL = Target->getSubtargetImpl()->getDataLayout())
        unwrap(Module)->setDataLayout(DL);
#else
    if (const DataLayout *DL = Target->getDataLayout())
        unwrap(Module)->setDataLayout(DL);
#endif
}

extern "C" LLVMTargetDataRef
LLVMRustGetModuleDataLayout(LLVMModuleRef M) {
#if LLVM_VERSION_MINOR >= 7
    return wrap(&unwrap(M)->getDataLayout());
#else
    return wrap(unwrap(M)->getDataLayout());
#endif
}
