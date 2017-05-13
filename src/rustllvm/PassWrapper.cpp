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

#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/AutoUpgrade.h"
#include "llvm/Support/CBindingWrapping.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Host.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetSubtargetInfo.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"

#if LLVM_VERSION_GE(4, 0)
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#endif

#include "llvm-c/Transforms/PassManagerBuilder.h"

using namespace llvm;
using namespace llvm::legacy;

extern cl::opt<bool> EnableARMEHABI;

typedef struct LLVMOpaquePass *LLVMPassRef;
typedef struct LLVMOpaqueTargetMachine *LLVMTargetMachineRef;

DEFINE_STDCXX_CONVERSION_FUNCTIONS(Pass, LLVMPassRef)
DEFINE_STDCXX_CONVERSION_FUNCTIONS(TargetMachine, LLVMTargetMachineRef)
DEFINE_STDCXX_CONVERSION_FUNCTIONS(PassManagerBuilder,
                                   LLVMPassManagerBuilderRef)

extern "C" void LLVMInitializePasses() {
  PassRegistry &Registry = *PassRegistry::getPassRegistry();
  initializeCore(Registry);
  initializeCodeGen(Registry);
  initializeScalarOpts(Registry);
  initializeVectorization(Registry);
  initializeIPO(Registry);
  initializeAnalysis(Registry);
#if LLVM_VERSION_EQ(3, 7)
  initializeIPA(Registry);
#endif
  initializeTransformUtils(Registry);
  initializeInstCombine(Registry);
  initializeInstrumentation(Registry);
  initializeTarget(Registry);
}

enum class LLVMRustPassKind {
  Other,
  Function,
  Module,
};

static LLVMRustPassKind toRust(PassKind Kind) {
  switch (Kind) {
  case PT_Function:
    return LLVMRustPassKind::Function;
  case PT_Module:
    return LLVMRustPassKind::Module;
  default:
    return LLVMRustPassKind::Other;
  }
}

extern "C" LLVMPassRef LLVMRustFindAndCreatePass(const char *PassName) {
  StringRef SR(PassName);
  PassRegistry *PR = PassRegistry::getPassRegistry();

  const PassInfo *PI = PR->getPassInfo(SR);
  if (PI) {
    return wrap(PI->createPass());
  }
  return nullptr;
}

extern "C" LLVMRustPassKind LLVMRustPassKind(LLVMPassRef RustPass) {
  assert(RustPass);
  Pass *Pass = unwrap(RustPass);
  return toRust(Pass->getPassKind());
}

extern "C" void LLVMRustAddPass(LLVMPassManagerRef PMR, LLVMPassRef RustPass) {
  assert(RustPass);
  Pass *Pass = unwrap(RustPass);
  PassManagerBase *PMB = unwrap(PMR);
  PMB->add(Pass);
}

#ifdef LLVM_COMPONENT_X86
#define SUBTARGET_X86 SUBTARGET(X86)
#else
#define SUBTARGET_X86
#endif

#ifdef LLVM_COMPONENT_ARM
#define SUBTARGET_ARM SUBTARGET(ARM)
#else
#define SUBTARGET_ARM
#endif

#ifdef LLVM_COMPONENT_AARCH64
#define SUBTARGET_AARCH64 SUBTARGET(AArch64)
#else
#define SUBTARGET_AARCH64
#endif

#ifdef LLVM_COMPONENT_MIPS
#define SUBTARGET_MIPS SUBTARGET(Mips)
#else
#define SUBTARGET_MIPS
#endif

#ifdef LLVM_COMPONENT_POWERPC
#define SUBTARGET_PPC SUBTARGET(PPC)
#else
#define SUBTARGET_PPC
#endif

#ifdef LLVM_COMPONENT_SYSTEMZ
#define SUBTARGET_SYSTEMZ SUBTARGET(SystemZ)
#else
#define SUBTARGET_SYSTEMZ
#endif

#ifdef LLVM_COMPONENT_MSP430
#define SUBTARGET_MSP430 SUBTARGET(MSP430)
#else
#define SUBTARGET_MSP430
#endif

#ifdef LLVM_COMPONENT_SPARC
#define SUBTARGET_SPARC SUBTARGET(Sparc)
#else
#define SUBTARGET_SPARC
#endif

#define GEN_SUBTARGETS                                                         \
  SUBTARGET_X86                                                                \
  SUBTARGET_ARM                                                                \
  SUBTARGET_AARCH64                                                            \
  SUBTARGET_MIPS                                                               \
  SUBTARGET_PPC                                                                \
  SUBTARGET_SYSTEMZ                                                            \
  SUBTARGET_MSP430                                                             \
  SUBTARGET_SPARC

#define SUBTARGET(x)                                                           \
  namespace llvm {                                                             \
  extern const SubtargetFeatureKV x##FeatureKV[];                              \
  extern const SubtargetFeatureKV x##SubTypeKV[];                              \
  }

GEN_SUBTARGETS
#undef SUBTARGET

extern "C" bool LLVMRustHasFeature(LLVMTargetMachineRef TM,
                                   const char *Feature) {
  TargetMachine *Target = unwrap(TM);
  const MCSubtargetInfo *MCInfo = Target->getMCSubtargetInfo();
  const FeatureBitset &Bits = MCInfo->getFeatureBits();
  const llvm::SubtargetFeatureKV *FeatureEntry;

#define SUBTARGET(x)                                                           \
  if (MCInfo->isCPUStringValid(x##SubTypeKV[0].Key)) {                         \
    FeatureEntry = x##FeatureKV;                                               \
  } else

  GEN_SUBTARGETS { return false; }
#undef SUBTARGET

  while (strcmp(Feature, FeatureEntry->Key) != 0)
    FeatureEntry++;

  return (Bits & FeatureEntry->Value) == FeatureEntry->Value;
}

enum class LLVMRustCodeModel {
  Other,
  Default,
  JITDefault,
  Small,
  Kernel,
  Medium,
  Large,
};

static CodeModel::Model fromRust(LLVMRustCodeModel Model) {
  switch (Model) {
  case LLVMRustCodeModel::Default:
    return CodeModel::Default;
  case LLVMRustCodeModel::JITDefault:
    return CodeModel::JITDefault;
  case LLVMRustCodeModel::Small:
    return CodeModel::Small;
  case LLVMRustCodeModel::Kernel:
    return CodeModel::Kernel;
  case LLVMRustCodeModel::Medium:
    return CodeModel::Medium;
  case LLVMRustCodeModel::Large:
    return CodeModel::Large;
  default:
    llvm_unreachable("Bad CodeModel.");
  }
}

enum class LLVMRustCodeGenOptLevel {
  Other,
  None,
  Less,
  Default,
  Aggressive,
};

static CodeGenOpt::Level fromRust(LLVMRustCodeGenOptLevel Level) {
  switch (Level) {
  case LLVMRustCodeGenOptLevel::None:
    return CodeGenOpt::None;
  case LLVMRustCodeGenOptLevel::Less:
    return CodeGenOpt::Less;
  case LLVMRustCodeGenOptLevel::Default:
    return CodeGenOpt::Default;
  case LLVMRustCodeGenOptLevel::Aggressive:
    return CodeGenOpt::Aggressive;
  default:
    llvm_unreachable("Bad CodeGenOptLevel.");
  }
}

#if LLVM_RUSTLLVM
/// getLongestEntryLength - Return the length of the longest entry in the table.
///
static size_t getLongestEntryLength(ArrayRef<SubtargetFeatureKV> Table) {
  size_t MaxLen = 0;
  for (auto &I : Table)
    MaxLen = std::max(MaxLen, std::strlen(I.Key));
  return MaxLen;
}

extern "C" void LLVMRustPrintTargetCPUs(LLVMTargetMachineRef TM) {
  const TargetMachine *Target = unwrap(TM);
  const MCSubtargetInfo *MCInfo = Target->getMCSubtargetInfo();
  const ArrayRef<SubtargetFeatureKV> CPUTable = MCInfo->getCPUTable();
  unsigned MaxCPULen = getLongestEntryLength(CPUTable);

  printf("Available CPUs for this target:\n");
  for (auto &CPU : CPUTable)
    printf("    %-*s - %s.\n", MaxCPULen, CPU.Key, CPU.Desc);
  printf("\n");
}

extern "C" void LLVMRustPrintTargetFeatures(LLVMTargetMachineRef TM) {
  const TargetMachine *Target = unwrap(TM);
  const MCSubtargetInfo *MCInfo = Target->getMCSubtargetInfo();
  const ArrayRef<SubtargetFeatureKV> FeatTable = MCInfo->getFeatureTable();
  unsigned MaxFeatLen = getLongestEntryLength(FeatTable);

  printf("Available features for this target:\n");
  for (auto &Feature : FeatTable)
    printf("    %-*s - %s.\n", MaxFeatLen, Feature.Key, Feature.Desc);
  printf("\n");

  printf("Use +feature to enable a feature, or -feature to disable it.\n"
         "For example, rustc -C -target-cpu=mycpu -C "
         "target-feature=+feature1,-feature2\n\n");
}

#else

extern "C" void LLVMRustPrintTargetCPUs(LLVMTargetMachineRef) {
  printf("Target CPU help is not supported by this LLVM version.\n\n");
}

extern "C" void LLVMRustPrintTargetFeatures(LLVMTargetMachineRef) {
  printf("Target features help is not supported by this LLVM version.\n\n");
}
#endif

extern "C" LLVMTargetMachineRef LLVMRustCreateTargetMachine(
    const char *TripleStr, const char *CPU, const char *Feature,
    LLVMRustCodeModel RustCM, LLVMRelocMode Reloc,
    LLVMRustCodeGenOptLevel RustOptLevel, bool UseSoftFloat,
    bool PositionIndependentExecutable, bool FunctionSections,
    bool DataSections) {

#if LLVM_VERSION_LE(3, 8)
  Reloc::Model RM;
#else
  Optional<Reloc::Model> RM;
#endif
  auto CM = fromRust(RustCM);
  auto OptLevel = fromRust(RustOptLevel);

  switch (Reloc) {
  case LLVMRelocStatic:
    RM = Reloc::Static;
    break;
  case LLVMRelocPIC:
    RM = Reloc::PIC_;
    break;
  case LLVMRelocDynamicNoPic:
    RM = Reloc::DynamicNoPIC;
    break;
  default:
#if LLVM_VERSION_LE(3, 8)
    RM = Reloc::Default;
#endif
    break;
  }

  std::string Error;
  Triple Trip(Triple::normalize(TripleStr));
  const llvm::Target *TheTarget =
      TargetRegistry::lookupTarget(Trip.getTriple(), Error);
  if (TheTarget == nullptr) {
    LLVMRustSetLastError(Error.c_str());
    return nullptr;
  }

  StringRef RealCPU = CPU;
  if (RealCPU == "native") {
    RealCPU = sys::getHostCPUName();
  }

  TargetOptions Options;
#if LLVM_VERSION_LE(3, 8)
  Options.PositionIndependentExecutable = PositionIndependentExecutable;
#endif

  Options.FloatABIType = FloatABI::Default;
  if (UseSoftFloat) {
    Options.FloatABIType = FloatABI::Soft;
  }
  Options.DataSections = DataSections;
  Options.FunctionSections = FunctionSections;

  TargetMachine *TM = TheTarget->createTargetMachine(
      Trip.getTriple(), RealCPU, Feature, Options, RM, CM, OptLevel);
  return wrap(TM);
}

extern "C" void LLVMRustDisposeTargetMachine(LLVMTargetMachineRef TM) {
  delete unwrap(TM);
}

// Unfortunately, LLVM doesn't expose a C API to add the corresponding analysis
// passes for a target to a pass manager. We export that functionality through
// this function.
extern "C" void LLVMRustAddAnalysisPasses(LLVMTargetMachineRef TM,
                                          LLVMPassManagerRef PMR,
                                          LLVMModuleRef M) {
  PassManagerBase *PM = unwrap(PMR);
  PM->add(
      createTargetTransformInfoWrapperPass(unwrap(TM)->getTargetIRAnalysis()));
}

extern "C" void LLVMRustConfigurePassManagerBuilder(
    LLVMPassManagerBuilderRef PMBR, LLVMRustCodeGenOptLevel OptLevel,
    bool MergeFunctions, bool SLPVectorize, bool LoopVectorize) {
  // Ignore mergefunc for now as enabling it causes crashes.
  // unwrap(PMBR)->MergeFunctions = MergeFunctions;
  unwrap(PMBR)->SLPVectorize = SLPVectorize;
  unwrap(PMBR)->OptLevel = fromRust(OptLevel);
  unwrap(PMBR)->LoopVectorize = LoopVectorize;
}

// Unfortunately, the LLVM C API doesn't provide a way to set the `LibraryInfo`
// field of a PassManagerBuilder, we expose our own method of doing so.
extern "C" void LLVMRustAddBuilderLibraryInfo(LLVMPassManagerBuilderRef PMBR,
                                              LLVMModuleRef M,
                                              bool DisableSimplifyLibCalls) {
  Triple TargetTriple(unwrap(M)->getTargetTriple());
  TargetLibraryInfoImpl *TLI = new TargetLibraryInfoImpl(TargetTriple);
  if (DisableSimplifyLibCalls)
    TLI->disableAllFunctions();
  unwrap(PMBR)->LibraryInfo = TLI;
}

// Unfortunately, the LLVM C API doesn't provide a way to create the
// TargetLibraryInfo pass, so we use this method to do so.
extern "C" void LLVMRustAddLibraryInfo(LLVMPassManagerRef PMR, LLVMModuleRef M,
                                       bool DisableSimplifyLibCalls) {
  Triple TargetTriple(unwrap(M)->getTargetTriple());
  TargetLibraryInfoImpl TLII(TargetTriple);
  if (DisableSimplifyLibCalls)
    TLII.disableAllFunctions();
  unwrap(PMR)->add(new TargetLibraryInfoWrapperPass(TLII));
}

// Unfortunately, the LLVM C API doesn't provide an easy way of iterating over
// all the functions in a module, so we do that manually here. You'll find
// similar code in clang's BackendUtil.cpp file.
extern "C" void LLVMRustRunFunctionPassManager(LLVMPassManagerRef PMR,
                                               LLVMModuleRef M) {
  llvm::legacy::FunctionPassManager *P =
      unwrap<llvm::legacy::FunctionPassManager>(PMR);
  P->doInitialization();

  // Upgrade all calls to old intrinsics first.
  for (Module::iterator I = unwrap(M)->begin(), E = unwrap(M)->end(); I != E;)
    UpgradeCallsToIntrinsic(&*I++); // must be post-increment, as we remove

  for (Module::iterator I = unwrap(M)->begin(), E = unwrap(M)->end(); I != E;
       ++I)
    if (!I->isDeclaration())
      P->run(*I);

  P->doFinalization();
}

extern "C" void LLVMRustSetLLVMOptions(int Argc, char **Argv) {
  // Initializing the command-line options more than once is not allowed. So,
  // check if they've already been initialized.  (This could happen if we're
  // being called from rustpkg, for example). If the arguments change, then
  // that's just kinda unfortunate.
  static bool Initialized = false;
  if (Initialized)
    return;
  Initialized = true;
  cl::ParseCommandLineOptions(Argc, Argv);
}

enum class LLVMRustFileType {
  Other,
  AssemblyFile,
  ObjectFile,
};

static TargetMachine::CodeGenFileType fromRust(LLVMRustFileType Type) {
  switch (Type) {
  case LLVMRustFileType::AssemblyFile:
    return TargetMachine::CGFT_AssemblyFile;
  case LLVMRustFileType::ObjectFile:
    return TargetMachine::CGFT_ObjectFile;
  default:
    llvm_unreachable("Bad FileType.");
  }
}

extern "C" LLVMRustResult
LLVMRustWriteOutputFile(LLVMTargetMachineRef Target, LLVMPassManagerRef PMR,
                        LLVMModuleRef M, const char *Path,
                        LLVMRustFileType RustFileType) {
  llvm::legacy::PassManager *PM = unwrap<llvm::legacy::PassManager>(PMR);
  auto FileType = fromRust(RustFileType);

  std::string ErrorInfo;
  std::error_code EC;
  raw_fd_ostream OS(Path, EC, sys::fs::F_None);
  if (EC)
    ErrorInfo = EC.message();
  if (ErrorInfo != "") {
    LLVMRustSetLastError(ErrorInfo.c_str());
    return LLVMRustResult::Failure;
  }

  unwrap(Target)->addPassesToEmitFile(*PM, OS, FileType, false);
  PM->run(*unwrap(M));

  // Apparently `addPassesToEmitFile` adds a pointer to our on-the-stack output
  // stream (OS), so the only real safe place to delete this is here? Don't we
  // wish this was written in Rust?
  delete PM;
  return LLVMRustResult::Success;
}

extern "C" void LLVMRustPrintModule(LLVMPassManagerRef PMR, LLVMModuleRef M,
                                    const char *Path) {
  llvm::legacy::PassManager *PM = unwrap<llvm::legacy::PassManager>(PMR);
  std::string ErrorInfo;

  std::error_code EC;
  raw_fd_ostream OS(Path, EC, sys::fs::F_None);
  if (EC)
    ErrorInfo = EC.message();

  formatted_raw_ostream FOS(OS);

  PM->add(createPrintModulePass(FOS));

  PM->run(*unwrap(M));
}

extern "C" void LLVMRustPrintPasses() {
  LLVMInitializePasses();
  struct MyListener : PassRegistrationListener {
    void passEnumerate(const PassInfo *Info) {
#if LLVM_VERSION_GE(4, 0)
      StringRef PassArg = Info->getPassArgument();
      StringRef PassName = Info->getPassName();
      if (!PassArg.empty()) {
        // These unsigned->signed casts could theoretically overflow, but
        // realistically never will (and even if, the result is implementation
        // defined rather plain UB).
        printf("%15.*s - %.*s\n", (int)PassArg.size(), PassArg.data(),
               (int)PassName.size(), PassName.data());
      }
#else
      if (Info->getPassArgument() && *Info->getPassArgument()) {
        printf("%15s - %s\n", Info->getPassArgument(), Info->getPassName());
      }
#endif
    }
  } Listener;

  PassRegistry *PR = PassRegistry::getPassRegistry();
  PR->enumerateWith(&Listener);
}

extern "C" void LLVMRustAddAlwaysInlinePass(LLVMPassManagerBuilderRef PMBR,
                                            bool AddLifetimes) {
#if LLVM_VERSION_GE(4, 0)
  unwrap(PMBR)->Inliner = llvm::createAlwaysInlinerLegacyPass(AddLifetimes);
#else
  unwrap(PMBR)->Inliner = createAlwaysInlinerPass(AddLifetimes);
#endif
}

extern "C" void LLVMRustRunRestrictionPass(LLVMModuleRef M, char **Symbols,
                                           size_t Len) {
  llvm::legacy::PassManager passes;

#if LLVM_VERSION_LE(3, 8)
  ArrayRef<const char *> Ref(Symbols, Len);
  passes.add(llvm::createInternalizePass(Ref));
#else
  auto PreserveFunctions = [=](const GlobalValue &GV) {
    for (size_t I = 0; I < Len; I++) {
      if (GV.getName() == Symbols[I]) {
        return true;
      }
    }
    return false;
  };

  passes.add(llvm::createInternalizePass(PreserveFunctions));
#endif

  passes.run(*unwrap(M));
}

extern "C" void LLVMRustMarkAllFunctionsNounwind(LLVMModuleRef M) {
  for (Module::iterator GV = unwrap(M)->begin(), E = unwrap(M)->end(); GV != E;
       ++GV) {
    GV->setDoesNotThrow();
    Function *F = dyn_cast<Function>(GV);
    if (F == nullptr)
      continue;

    for (Function::iterator B = F->begin(), BE = F->end(); B != BE; ++B) {
      for (BasicBlock::iterator I = B->begin(), IE = B->end(); I != IE; ++I) {
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
  unwrap(Module)->setDataLayout(Target->createDataLayout());
}

extern "C" LLVMTargetDataRef LLVMRustGetModuleDataLayout(LLVMModuleRef M) {
  return wrap(&unwrap(M)->getDataLayout());
}

extern "C" void LLVMRustSetModulePIELevel(LLVMModuleRef M) {
#if LLVM_VERSION_GE(3, 9)
  unwrap(M)->setPIELevel(PIELevel::Level::Large);
#endif
}
