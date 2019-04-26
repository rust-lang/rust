#include <stdio.h>

#include <vector>
#include <set>

#include "rustllvm.h"

#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/AutoUpgrade.h"
#include "llvm/IR/AssemblyAnnotationWriter.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Support/CBindingWrapping.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Host.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/IPO/FunctionImport.h"
#include "llvm/Transforms/Utils/FunctionImportUtils.h"
#include "llvm/LTO/LTO.h"

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

extern "C"
void LLVMRustPassManagerBuilderPopulateThinLTOPassManager(
  LLVMPassManagerBuilderRef PMBR,
  LLVMPassManagerRef PMR
) {
  unwrap(PMBR)->populateThinLTOPassManager(*unwrap(PMR));
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

#ifdef LLVM_COMPONENT_RISCV
#define SUBTARGET_RISCV SUBTARGET(RISCV)
#else
#define SUBTARGET_RISCV
#endif

#ifdef LLVM_COMPONENT_SPARC
#define SUBTARGET_SPARC SUBTARGET(Sparc)
#else
#define SUBTARGET_SPARC
#endif

#ifdef LLVM_COMPONENT_HEXAGON
#define SUBTARGET_HEXAGON SUBTARGET(Hexagon)
#else
#define SUBTARGET_HEXAGON
#endif

#define GEN_SUBTARGETS                                                         \
  SUBTARGET_X86                                                                \
  SUBTARGET_ARM                                                                \
  SUBTARGET_AARCH64                                                            \
  SUBTARGET_MIPS                                                               \
  SUBTARGET_PPC                                                                \
  SUBTARGET_SYSTEMZ                                                            \
  SUBTARGET_MSP430                                                             \
  SUBTARGET_SPARC                                                              \
  SUBTARGET_HEXAGON                                                            \
  SUBTARGET_RISCV                                                              \

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
  return MCInfo->checkFeatures(std::string("+") + Feature);
}

enum class LLVMRustCodeModel {
  Other,
  Small,
  Kernel,
  Medium,
  Large,
  None,
};

static CodeModel::Model fromRust(LLVMRustCodeModel Model) {
  switch (Model) {
  case LLVMRustCodeModel::Small:
    return CodeModel::Small;
  case LLVMRustCodeModel::Kernel:
    return CodeModel::Kernel;
  case LLVMRustCodeModel::Medium:
    return CodeModel::Medium;
  case LLVMRustCodeModel::Large:
    return CodeModel::Large;
  default:
    report_fatal_error("Bad CodeModel.");
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
    report_fatal_error("Bad CodeGenOptLevel.");
  }
}

enum class LLVMRustRelocMode {
  Default,
  Static,
  PIC,
  DynamicNoPic,
  ROPI,
  RWPI,
  ROPIRWPI,
};

static Optional<Reloc::Model> fromRust(LLVMRustRelocMode RustReloc) {
  switch (RustReloc) {
  case LLVMRustRelocMode::Default:
    return None;
  case LLVMRustRelocMode::Static:
    return Reloc::Static;
  case LLVMRustRelocMode::PIC:
    return Reloc::PIC_;
  case LLVMRustRelocMode::DynamicNoPic:
    return Reloc::DynamicNoPIC;
  case LLVMRustRelocMode::ROPI:
    return Reloc::ROPI;
  case LLVMRustRelocMode::RWPI:
    return Reloc::RWPI;
  case LLVMRustRelocMode::ROPIRWPI:
    return Reloc::ROPI_RWPI;
  }
  report_fatal_error("Bad RelocModel.");
}

#ifdef LLVM_RUSTLLVM
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
  const Triple::ArchType HostArch = Triple(sys::getProcessTriple()).getArch();
  const Triple::ArchType TargetArch = Target->getTargetTriple().getArch();
  const ArrayRef<SubtargetFeatureKV> CPUTable = MCInfo->getCPUTable();
  unsigned MaxCPULen = getLongestEntryLength(CPUTable);

  printf("Available CPUs for this target:\n");
  if (HostArch == TargetArch) {
    const StringRef HostCPU = sys::getHostCPUName();
    printf("    %-*s - Select the CPU of the current host (currently %.*s).\n",
      MaxCPULen, "native", (int)HostCPU.size(), HostCPU.data());
  }
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

extern "C" const char* LLVMRustGetHostCPUName(size_t *len) {
  StringRef Name = sys::getHostCPUName();
  *len = Name.size();
  return Name.data();
}

extern "C" LLVMTargetMachineRef LLVMRustCreateTargetMachine(
    const char *TripleStr, const char *CPU, const char *Feature,
    LLVMRustCodeModel RustCM, LLVMRustRelocMode RustReloc,
    LLVMRustCodeGenOptLevel RustOptLevel, bool UseSoftFloat,
    bool PositionIndependentExecutable, bool FunctionSections,
    bool DataSections,
    bool TrapUnreachable,
    bool Singlethread,
    bool AsmComments,
    bool EmitStackSizeSection) {

  auto OptLevel = fromRust(RustOptLevel);
  auto RM = fromRust(RustReloc);

  std::string Error;
  Triple Trip(Triple::normalize(TripleStr));
  const llvm::Target *TheTarget =
      TargetRegistry::lookupTarget(Trip.getTriple(), Error);
  if (TheTarget == nullptr) {
    LLVMRustSetLastError(Error.c_str());
    return nullptr;
  }

  TargetOptions Options;

  Options.FloatABIType = FloatABI::Default;
  if (UseSoftFloat) {
    Options.FloatABIType = FloatABI::Soft;
  }
  Options.DataSections = DataSections;
  Options.FunctionSections = FunctionSections;
  Options.MCOptions.AsmVerbose = AsmComments;
  Options.MCOptions.PreserveAsmComments = AsmComments;

  if (TrapUnreachable) {
    // Tell LLVM to codegen `unreachable` into an explicit trap instruction.
    // This limits the extent of possible undefined behavior in some cases, as
    // it prevents control flow from "falling through" into whatever code
    // happens to be laid out next in memory.
    Options.TrapUnreachable = true;
  }

  if (Singlethread) {
    Options.ThreadModel = ThreadModel::Single;
  }

  Options.EmitStackSizeSection = EmitStackSizeSection;

  Optional<CodeModel::Model> CM;
  if (RustCM != LLVMRustCodeModel::None)
    CM = fromRust(RustCM);
  TargetMachine *TM = TheTarget->createTargetMachine(
      Trip.getTriple(), CPU, Feature, Options, RM, CM, OptLevel);
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
    bool MergeFunctions, bool SLPVectorize, bool LoopVectorize, bool PrepareForThinLTO,
    const char* PGOGenPath, const char* PGOUsePath) {
#if LLVM_VERSION_GE(7, 0)
  unwrap(PMBR)->MergeFunctions = MergeFunctions;
#endif
  unwrap(PMBR)->SLPVectorize = SLPVectorize;
  unwrap(PMBR)->OptLevel = fromRust(OptLevel);
  unwrap(PMBR)->LoopVectorize = LoopVectorize;
  unwrap(PMBR)->PrepareForThinLTO = PrepareForThinLTO;

  if (PGOGenPath) {
    assert(!PGOUsePath);
    unwrap(PMBR)->EnablePGOInstrGen = true;
    unwrap(PMBR)->PGOInstrGen = PGOGenPath;
  }
  if (PGOUsePath) {
    assert(!PGOGenPath);
    unwrap(PMBR)->PGOInstrUse = PGOUsePath;
  }
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
    report_fatal_error("Bad FileType.");
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

#if LLVM_VERSION_GE(7, 0)
  buffer_ostream BOS(OS);
  unwrap(Target)->addPassesToEmitFile(*PM, BOS, nullptr, FileType, false);
#else
  unwrap(Target)->addPassesToEmitFile(*PM, OS, FileType, false);
#endif
  PM->run(*unwrap(M));

  // Apparently `addPassesToEmitFile` adds a pointer to our on-the-stack output
  // stream (OS), so the only real safe place to delete this is here? Don't we
  // wish this was written in Rust?
  delete PM;
  return LLVMRustResult::Success;
}


// Callback to demangle function name
// Parameters:
// * name to be demangled
// * name len
// * output buffer
// * output buffer len
// Returns len of demangled string, or 0 if demangle failed.
typedef size_t (*DemangleFn)(const char*, size_t, char*, size_t);


namespace {

class RustAssemblyAnnotationWriter : public AssemblyAnnotationWriter {
  DemangleFn Demangle;
  std::vector<char> Buf;

public:
  RustAssemblyAnnotationWriter(DemangleFn Demangle) : Demangle(Demangle) {}

  // Return empty string if demangle failed
  // or if name does not need to be demangled
  StringRef CallDemangle(StringRef name) {
    if (!Demangle) {
      return StringRef();
    }

    if (Buf.size() < name.size() * 2) {
      // Semangled name usually shorter than mangled,
      // but allocate twice as much memory just in case
      Buf.resize(name.size() * 2);
    }

    auto R = Demangle(name.data(), name.size(), Buf.data(), Buf.size());
    if (!R) {
      // Demangle failed.
      return StringRef();
    }

    auto Demangled = StringRef(Buf.data(), R);
    if (Demangled == name) {
      // Do not print anything if demangled name is equal to mangled.
      return StringRef();
    }

    return Demangled;
  }

  void emitFunctionAnnot(const Function *F,
                         formatted_raw_ostream &OS) override {
    StringRef Demangled = CallDemangle(F->getName());
    if (Demangled.empty()) {
        return;
    }

    OS << "; " << Demangled << "\n";
  }

  void emitInstructionAnnot(const Instruction *I,
                            formatted_raw_ostream &OS) override {
    const char *Name;
    const Value *Value;
    if (const CallInst *CI = dyn_cast<CallInst>(I)) {
      Name = "call";
      Value = CI->getCalledValue();
    } else if (const InvokeInst* II = dyn_cast<InvokeInst>(I)) {
      Name = "invoke";
      Value = II->getCalledValue();
    } else {
      // Could demangle more operations, e. g.
      // `store %place, @function`.
      return;
    }

    if (!Value->hasName()) {
      return;
    }

    StringRef Demangled = CallDemangle(Value->getName());
    if (Demangled.empty()) {
      return;
    }

    OS << "; " << Name << " " << Demangled << "\n";
  }
};

class RustPrintModulePass : public ModulePass {
  raw_ostream* OS;
  DemangleFn Demangle;
public:
  static char ID;
  RustPrintModulePass() : ModulePass(ID), OS(nullptr), Demangle(nullptr) {}
  RustPrintModulePass(raw_ostream &OS, DemangleFn Demangle)
      : ModulePass(ID), OS(&OS), Demangle(Demangle) {}

  bool runOnModule(Module &M) override {
    RustAssemblyAnnotationWriter AW(Demangle);

    M.print(*OS, &AW, false);

    return false;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
  }

  static StringRef name() { return "RustPrintModulePass"; }
};

} // namespace

namespace llvm {
  void initializeRustPrintModulePassPass(PassRegistry&);
}

char RustPrintModulePass::ID = 0;
INITIALIZE_PASS(RustPrintModulePass, "print-rust-module",
                "Print rust module to stderr", false, false)

extern "C" LLVMRustResult
LLVMRustPrintModule(LLVMPassManagerRef PMR, LLVMModuleRef M,
                    const char *Path, DemangleFn Demangle) {
  llvm::legacy::PassManager *PM = unwrap<llvm::legacy::PassManager>(PMR);
  std::string ErrorInfo;

  std::error_code EC;
  raw_fd_ostream OS(Path, EC, sys::fs::F_None);
  if (EC)
    ErrorInfo = EC.message();
  if (ErrorInfo != "") {
    LLVMRustSetLastError(ErrorInfo.c_str());
    return LLVMRustResult::Failure;
  }

  formatted_raw_ostream FOS(OS);

  PM->add(new RustPrintModulePass(FOS, Demangle));

  PM->run(*unwrap(M));

  return LLVMRustResult::Success;
}

extern "C" void LLVMRustPrintPasses() {
  LLVMInitializePasses();
  struct MyListener : PassRegistrationListener {
    void passEnumerate(const PassInfo *Info) {
      StringRef PassArg = Info->getPassArgument();
      StringRef PassName = Info->getPassName();
      if (!PassArg.empty()) {
        // These unsigned->signed casts could theoretically overflow, but
        // realistically never will (and even if, the result is implementation
        // defined rather plain UB).
        printf("%15.*s - %.*s\n", (int)PassArg.size(), PassArg.data(),
               (int)PassName.size(), PassName.data());
      }
    }
  } Listener;

  PassRegistry *PR = PassRegistry::getPassRegistry();
  PR->enumerateWith(&Listener);
}

extern "C" void LLVMRustAddAlwaysInlinePass(LLVMPassManagerBuilderRef PMBR,
                                            bool AddLifetimes) {
  unwrap(PMBR)->Inliner = llvm::createAlwaysInlinerLegacyPass(AddLifetimes);
}

extern "C" void LLVMRustRunRestrictionPass(LLVMModuleRef M, char **Symbols,
                                           size_t Len) {
  llvm::legacy::PassManager passes;

  auto PreserveFunctions = [=](const GlobalValue &GV) {
    for (size_t I = 0; I < Len; I++) {
      if (GV.getName() == Symbols[I]) {
        return true;
      }
    }
    return false;
  };

  passes.add(llvm::createInternalizePass(PreserveFunctions));

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

extern "C" void LLVMRustSetModulePIELevel(LLVMModuleRef M) {
  unwrap(M)->setPIELevel(PIELevel::Level::Large);
}

// Here you'll find an implementation of ThinLTO as used by the Rust compiler
// right now. This ThinLTO support is only enabled on "recent ish" versions of
// LLVM, and otherwise it's just blanket rejected from other compilers.
//
// Most of this implementation is straight copied from LLVM. At the time of
// this writing it wasn't *quite* suitable to reuse more code from upstream
// for our purposes, but we should strive to upstream this support once it's
// ready to go! I figure we may want a bit of testing locally first before
// sending this upstream to LLVM. I hear though they're quite eager to receive
// feedback like this!
//
// If you're reading this code and wondering "what in the world" or you're
// working "good lord by LLVM upgrade is *still* failing due to these bindings"
// then fear not! (ok maybe fear a little). All code here is mostly based
// on `lib/LTO/ThinLTOCodeGenerator.cpp` in LLVM.
//
// You'll find that the general layout here roughly corresponds to the `run`
// method in that file as well as `ProcessThinLTOModule`. Functions are
// specifically commented below as well, but if you're updating this code
// or otherwise trying to understand it, the LLVM source will be useful in
// interpreting the mysteries within.
//
// Otherwise I'll apologize in advance, it probably requires a relatively
// significant investment on your part to "truly understand" what's going on
// here. Not saying I do myself, but it took me awhile staring at LLVM's source
// and various online resources about ThinLTO to make heads or tails of all
// this.

// This is a shared data structure which *must* be threadsafe to share
// read-only amongst threads. This also corresponds basically to the arguments
// of the `ProcessThinLTOModule` function in the LLVM source.
struct LLVMRustThinLTOData {
  // The combined index that is the global analysis over all modules we're
  // performing ThinLTO for. This is mostly managed by LLVM.
  ModuleSummaryIndex Index;

  // All modules we may look at, stored as in-memory serialized versions. This
  // is later used when inlining to ensure we can extract any module to inline
  // from.
  StringMap<MemoryBufferRef> ModuleMap;

  // A set that we manage of everything we *don't* want internalized. Note that
  // this includes all transitive references right now as well, but it may not
  // always!
  DenseSet<GlobalValue::GUID> GUIDPreservedSymbols;

  // Not 100% sure what these are, but they impact what's internalized and
  // what's inlined across modules, I believe.
  StringMap<FunctionImporter::ImportMapTy> ImportLists;
  StringMap<FunctionImporter::ExportSetTy> ExportLists;
  StringMap<GVSummaryMapTy> ModuleToDefinedGVSummaries;

#if LLVM_VERSION_GE(7, 0)
  LLVMRustThinLTOData() : Index(/* HaveGVs = */ false) {}
#endif
};

// Just an argument to the `LLVMRustCreateThinLTOData` function below.
struct LLVMRustThinLTOModule {
  const char *identifier;
  const char *data;
  size_t len;
};

// This is copied from `lib/LTO/ThinLTOCodeGenerator.cpp`, not sure what it
// does.
static const GlobalValueSummary *
getFirstDefinitionForLinker(const GlobalValueSummaryList &GVSummaryList) {
  auto StrongDefForLinker = llvm::find_if(
      GVSummaryList, [](const std::unique_ptr<GlobalValueSummary> &Summary) {
        auto Linkage = Summary->linkage();
        return !GlobalValue::isAvailableExternallyLinkage(Linkage) &&
               !GlobalValue::isWeakForLinker(Linkage);
      });
  if (StrongDefForLinker != GVSummaryList.end())
    return StrongDefForLinker->get();

  auto FirstDefForLinker = llvm::find_if(
      GVSummaryList, [](const std::unique_ptr<GlobalValueSummary> &Summary) {
        auto Linkage = Summary->linkage();
        return !GlobalValue::isAvailableExternallyLinkage(Linkage);
      });
  if (FirstDefForLinker == GVSummaryList.end())
    return nullptr;
  return FirstDefForLinker->get();
}

// The main entry point for creating the global ThinLTO analysis. The structure
// here is basically the same as before threads are spawned in the `run`
// function of `lib/LTO/ThinLTOCodeGenerator.cpp`.
extern "C" LLVMRustThinLTOData*
LLVMRustCreateThinLTOData(LLVMRustThinLTOModule *modules,
                          int num_modules,
                          const char **preserved_symbols,
                          int num_symbols) {
  auto Ret = llvm::make_unique<LLVMRustThinLTOData>();

  // Load each module's summary and merge it into one combined index
  for (int i = 0; i < num_modules; i++) {
    auto module = &modules[i];
    StringRef buffer(module->data, module->len);
    MemoryBufferRef mem_buffer(buffer, module->identifier);

    Ret->ModuleMap[module->identifier] = mem_buffer;

    if (Error Err = readModuleSummaryIndex(mem_buffer, Ret->Index, i)) {
      LLVMRustSetLastError(toString(std::move(Err)).c_str());
      return nullptr;
    }
  }

  // Collect for each module the list of function it defines (GUID -> Summary)
  Ret->Index.collectDefinedGVSummariesPerModule(Ret->ModuleToDefinedGVSummaries);

  // Convert the preserved symbols set from string to GUID, this is then needed
  // for internalization.
  for (int i = 0; i < num_symbols; i++) {
    auto GUID = GlobalValue::getGUID(preserved_symbols[i]);
    Ret->GUIDPreservedSymbols.insert(GUID);
  }

  // Collect the import/export lists for all modules from the call-graph in the
  // combined index
  //
  // This is copied from `lib/LTO/ThinLTOCodeGenerator.cpp`
#if LLVM_VERSION_GE(7, 0)
  auto deadIsPrevailing = [&](GlobalValue::GUID G) {
    return PrevailingType::Unknown;
  };
#if LLVM_VERSION_GE(8, 0)
  // We don't have a complete picture in our use of ThinLTO, just our immediate
  // crate, so we need `ImportEnabled = false` to limit internalization.
  // Otherwise, we sometimes lose `static` values -- see #60184.
  computeDeadSymbolsWithConstProp(Ret->Index, Ret->GUIDPreservedSymbols,
                                  deadIsPrevailing, /* ImportEnabled = */ false);
#else
  computeDeadSymbols(Ret->Index, Ret->GUIDPreservedSymbols, deadIsPrevailing);
#endif
#else
  computeDeadSymbols(Ret->Index, Ret->GUIDPreservedSymbols);
#endif
  ComputeCrossModuleImport(
    Ret->Index,
    Ret->ModuleToDefinedGVSummaries,
    Ret->ImportLists,
    Ret->ExportLists
  );

  // Resolve LinkOnce/Weak symbols, this has to be computed early be cause it
  // impacts the caching.
  //
  // This is copied from `lib/LTO/ThinLTOCodeGenerator.cpp` with some of this
  // being lifted from `lib/LTO/LTO.cpp` as well
  StringMap<std::map<GlobalValue::GUID, GlobalValue::LinkageTypes>> ResolvedODR;
  DenseMap<GlobalValue::GUID, const GlobalValueSummary *> PrevailingCopy;
  for (auto &I : Ret->Index) {
    if (I.second.SummaryList.size() > 1)
      PrevailingCopy[I.first] = getFirstDefinitionForLinker(I.second.SummaryList);
  }
  auto isPrevailing = [&](GlobalValue::GUID GUID, const GlobalValueSummary *S) {
    const auto &Prevailing = PrevailingCopy.find(GUID);
    if (Prevailing == PrevailingCopy.end())
      return true;
    return Prevailing->second == S;
  };
  auto recordNewLinkage = [&](StringRef ModuleIdentifier,
                              GlobalValue::GUID GUID,
                              GlobalValue::LinkageTypes NewLinkage) {
    ResolvedODR[ModuleIdentifier][GUID] = NewLinkage;
  };
#if LLVM_VERSION_GE(8, 0)
  thinLTOResolvePrevailingInIndex(Ret->Index, isPrevailing, recordNewLinkage);
#else
  thinLTOResolveWeakForLinkerInIndex(Ret->Index, isPrevailing, recordNewLinkage);
#endif

  // Here we calculate an `ExportedGUIDs` set for use in the `isExported`
  // callback below. This callback below will dictate the linkage for all
  // summaries in the index, and we basically just only want to ensure that dead
  // symbols are internalized. Otherwise everything that's already external
  // linkage will stay as external, and internal will stay as internal.
  std::set<GlobalValue::GUID> ExportedGUIDs;
  for (auto &List : Ret->Index) {
    for (auto &GVS: List.second.SummaryList) {
      if (GlobalValue::isLocalLinkage(GVS->linkage()))
        continue;
      auto GUID = GVS->getOriginalName();
      if (GVS->flags().Live)
        ExportedGUIDs.insert(GUID);
    }
  }
  auto isExported = [&](StringRef ModuleIdentifier, GlobalValue::GUID GUID) {
    const auto &ExportList = Ret->ExportLists.find(ModuleIdentifier);
    return (ExportList != Ret->ExportLists.end() &&
      ExportList->second.count(GUID)) ||
      ExportedGUIDs.count(GUID);
  };
  thinLTOInternalizeAndPromoteInIndex(Ret->Index, isExported);

  return Ret.release();
}

extern "C" void
LLVMRustFreeThinLTOData(LLVMRustThinLTOData *Data) {
  delete Data;
}

// Below are the various passes that happen *per module* when doing ThinLTO.
//
// In other words, these are the functions that are all run concurrently
// with one another, one per module. The passes here correspond to the analysis
// passes in `lib/LTO/ThinLTOCodeGenerator.cpp`, currently found in the
// `ProcessThinLTOModule` function. Here they're split up into separate steps
// so rustc can save off the intermediate bytecode between each step.

extern "C" bool
LLVMRustPrepareThinLTORename(const LLVMRustThinLTOData *Data, LLVMModuleRef M) {
  Module &Mod = *unwrap(M);
  if (renameModuleForThinLTO(Mod, Data->Index)) {
    LLVMRustSetLastError("renameModuleForThinLTO failed");
    return false;
  }
  return true;
}

extern "C" bool
LLVMRustPrepareThinLTOResolveWeak(const LLVMRustThinLTOData *Data, LLVMModuleRef M) {
  Module &Mod = *unwrap(M);
  const auto &DefinedGlobals = Data->ModuleToDefinedGVSummaries.lookup(Mod.getModuleIdentifier());
#if LLVM_VERSION_GE(8, 0)
  thinLTOResolvePrevailingInModule(Mod, DefinedGlobals);
#else
  thinLTOResolveWeakForLinkerModule(Mod, DefinedGlobals);
#endif
  return true;
}

extern "C" bool
LLVMRustPrepareThinLTOInternalize(const LLVMRustThinLTOData *Data, LLVMModuleRef M) {
  Module &Mod = *unwrap(M);
  const auto &DefinedGlobals = Data->ModuleToDefinedGVSummaries.lookup(Mod.getModuleIdentifier());
  thinLTOInternalizeModule(Mod, DefinedGlobals);
  return true;
}

extern "C" bool
LLVMRustPrepareThinLTOImport(const LLVMRustThinLTOData *Data, LLVMModuleRef M) {
  Module &Mod = *unwrap(M);

  const auto &ImportList = Data->ImportLists.lookup(Mod.getModuleIdentifier());
  auto Loader = [&](StringRef Identifier) {
    const auto &Memory = Data->ModuleMap.lookup(Identifier);
    auto &Context = Mod.getContext();
    auto MOrErr = getLazyBitcodeModule(Memory, Context, true, true);

    if (!MOrErr)
      return MOrErr;

    // The rest of this closure is a workaround for
    // https://bugs.llvm.org/show_bug.cgi?id=38184 where during ThinLTO imports
    // we accidentally import wasm custom sections into different modules,
    // duplicating them by in the final output artifact.
    //
    // The issue is worked around here by manually removing the
    // `wasm.custom_sections` named metadata node from any imported module. This
    // we know isn't used by any optimization pass so there's no need for it to
    // be imported.
    //
    // Note that the metadata is currently lazily loaded, so we materialize it
    // here before looking up if there's metadata inside. The `FunctionImporter`
    // will immediately materialize metadata anyway after an import, so this
    // shouldn't be a perf hit.
    if (Error Err = (*MOrErr)->materializeMetadata()) {
      Expected<std::unique_ptr<Module>> Ret(std::move(Err));
      return Ret;
    }

    auto *WasmCustomSections = (*MOrErr)->getNamedMetadata("wasm.custom_sections");
    if (WasmCustomSections)
      WasmCustomSections->eraseFromParent();

    return MOrErr;
  };
  FunctionImporter Importer(Data->Index, Loader);
  Expected<bool> Result = Importer.importFunctions(Mod, ImportList);
  if (!Result) {
    LLVMRustSetLastError(toString(Result.takeError()).c_str());
    return false;
  }
  return true;
}

extern "C" typedef void (*LLVMRustModuleNameCallback)(void*, // payload
                                                      const char*, // importing module name
                                                      const char*); // imported module name

// Calls `module_name_callback` for each module import done by ThinLTO.
// The callback is provided with regular null-terminated C strings.
extern "C" void
LLVMRustGetThinLTOModuleImports(const LLVMRustThinLTOData *data,
                                LLVMRustModuleNameCallback module_name_callback,
                                void* callback_payload) {
  for (const auto& importing_module : data->ImportLists) {
    const std::string importing_module_id = importing_module.getKey().str();
    const auto& imports = importing_module.getValue();
    for (const auto& imported_module : imports) {
      const std::string imported_module_id = imported_module.getKey().str();
      module_name_callback(callback_payload,
                           importing_module_id.c_str(),
                           imported_module_id.c_str());
    }
  }
}

// This struct and various functions are sort of a hack right now, but the
// problem is that we've got in-memory LLVM modules after we generate and
// optimize all codegen-units for one compilation in rustc. To be compatible
// with the LTO support above we need to serialize the modules plus their
// ThinLTO summary into memory.
//
// This structure is basically an owned version of a serialize module, with
// a ThinLTO summary attached.
struct LLVMRustThinLTOBuffer {
  std::string data;
};

extern "C" LLVMRustThinLTOBuffer*
LLVMRustThinLTOBufferCreate(LLVMModuleRef M) {
  auto Ret = llvm::make_unique<LLVMRustThinLTOBuffer>();
  {
    raw_string_ostream OS(Ret->data);
    {
      legacy::PassManager PM;
      PM.add(createWriteThinLTOBitcodePass(OS));
      PM.run(*unwrap(M));
    }
  }
  return Ret.release();
}

extern "C" void
LLVMRustThinLTOBufferFree(LLVMRustThinLTOBuffer *Buffer) {
  delete Buffer;
}

extern "C" const void*
LLVMRustThinLTOBufferPtr(const LLVMRustThinLTOBuffer *Buffer) {
  return Buffer->data.data();
}

extern "C" size_t
LLVMRustThinLTOBufferLen(const LLVMRustThinLTOBuffer *Buffer) {
  return Buffer->data.length();
}

// This is what we used to parse upstream bitcode for actual ThinLTO
// processing.  We'll call this once per module optimized through ThinLTO, and
// it'll be called concurrently on many threads.
extern "C" LLVMModuleRef
LLVMRustParseBitcodeForLTO(LLVMContextRef Context,
                           const char *data,
                           size_t len,
                           const char *identifier) {
  StringRef Data(data, len);
  MemoryBufferRef Buffer(Data, identifier);
  unwrap(Context)->enableDebugTypeODRUniquing();
  Expected<std::unique_ptr<Module>> SrcOrError =
      parseBitcodeFile(Buffer, *unwrap(Context));
  if (!SrcOrError) {
    LLVMRustSetLastError(toString(SrcOrError.takeError()).c_str());
    return nullptr;
  }
  return wrap(std::move(*SrcOrError).release());
}

// Rewrite all `DICompileUnit` pointers to the `DICompileUnit` specified. See
// the comment in `back/lto.rs` for why this exists.
extern "C" void
LLVMRustThinLTOGetDICompileUnit(LLVMModuleRef Mod,
                                DICompileUnit **A,
                                DICompileUnit **B) {
  Module *M = unwrap(Mod);
  DICompileUnit **Cur = A;
  DICompileUnit **Next = B;
  for (DICompileUnit *CU : M->debug_compile_units()) {
    *Cur = CU;
    Cur = Next;
    Next = nullptr;
    if (Cur == nullptr)
      break;
  }
}

// Rewrite all `DICompileUnit` pointers to the `DICompileUnit` specified. See
// the comment in `back/lto.rs` for why this exists.
extern "C" void
LLVMRustThinLTOPatchDICompileUnit(LLVMModuleRef Mod, DICompileUnit *Unit) {
  Module *M = unwrap(Mod);

  // If the original source module didn't have a `DICompileUnit` then try to
  // merge all the existing compile units. If there aren't actually any though
  // then there's not much for us to do so return.
  if (Unit == nullptr) {
    for (DICompileUnit *CU : M->debug_compile_units()) {
      Unit = CU;
      break;
    }
    if (Unit == nullptr)
      return;
  }

  // Use LLVM's built-in `DebugInfoFinder` to find a bunch of debuginfo and
  // process it recursively. Note that we specifically iterate over instructions
  // to ensure we feed everything into it.
  DebugInfoFinder Finder;
  Finder.processModule(*M);
  for (Function &F : M->functions()) {
    for (auto &FI : F) {
      for (Instruction &BI : FI) {
        if (auto Loc = BI.getDebugLoc())
          Finder.processLocation(*M, Loc);
        if (auto DVI = dyn_cast<DbgValueInst>(&BI))
          Finder.processValue(*M, DVI);
        if (auto DDI = dyn_cast<DbgDeclareInst>(&BI))
          Finder.processDeclare(*M, DDI);
      }
    }
  }

  // After we've found all our debuginfo, rewrite all subprograms to point to
  // the same `DICompileUnit`.
  for (auto &F : Finder.subprograms()) {
    F->replaceUnit(Unit);
  }

  // Erase any other references to other `DICompileUnit` instances, the verifier
  // will later ensure that we don't actually have any other stale references to
  // worry about.
  auto *MD = M->getNamedMetadata("llvm.dbg.cu");
  MD->clearOperands();
  MD->addOperand(Unit);
}
