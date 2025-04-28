#include "LLVMWrapper.h"

#include "llvm-c/Core.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/Lint.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/Bitcode/BitcodeWriterPass.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/IR/AssemblyAnnotationWriter.h"
#include "llvm/IR/AutoUpgrade.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRPrinter/IRPrintingPasses.h"
#include "llvm/LTO/LTO.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Passes/StandardInstrumentations.h"
#include "llvm/Support/CBindingWrapping.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/TimeProfiler.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/Transforms/IPO/FunctionImport.h"
#include "llvm/Transforms/IPO/Internalize.h"
#include "llvm/Transforms/IPO/LowerTypeTests.h"
#include "llvm/Transforms/IPO/ThinLTOBitcodeWriter.h"
#include "llvm/Transforms/Instrumentation/AddressSanitizer.h"
#include "llvm/Transforms/Instrumentation/DataFlowSanitizer.h"
#include "llvm/Transforms/Instrumentation/HWAddressSanitizer.h"
#include "llvm/Transforms/Instrumentation/InstrProfiling.h"
#include "llvm/Transforms/Instrumentation/MemorySanitizer.h"
#include "llvm/Transforms/Instrumentation/ThreadSanitizer.h"
#include "llvm/Transforms/Scalar/AnnotationRemarks.h"
#include "llvm/Transforms/Utils/CanonicalizeAliases.h"
#include "llvm/Transforms/Utils/FunctionImportUtils.h"
#include "llvm/Transforms/Utils/NameAnonGlobals.h"
#include <set>
#include <string>
#include <vector>

// Conditional includes prevent clang-format from fully sorting the list,
// so if any are needed, keep them separate down here.

using namespace llvm;

static codegen::RegisterCodeGenFlags CGF;

typedef struct LLVMOpaquePass *LLVMPassRef;
typedef struct LLVMOpaqueTargetMachine *LLVMTargetMachineRef;

DEFINE_STDCXX_CONVERSION_FUNCTIONS(Pass, LLVMPassRef)
DEFINE_STDCXX_CONVERSION_FUNCTIONS(TargetMachine, LLVMTargetMachineRef)

extern "C" void LLVMRustTimeTraceProfilerInitialize() {
  timeTraceProfilerInitialize(
      /* TimeTraceGranularity */ 0,
      /* ProcName */ "rustc");
}

extern "C" void LLVMRustTimeTraceProfilerFinishThread() {
  timeTraceProfilerFinishThread();
}

extern "C" void LLVMRustTimeTraceProfilerFinish(const char *FileName) {
  auto FN = StringRef(FileName);
  std::error_code EC;
  auto OS = raw_fd_ostream(FN, EC, sys::fs::CD_CreateAlways);

  timeTraceProfilerWrite(OS);
  timeTraceProfilerCleanup();
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

#ifdef LLVM_COMPONENT_AVR
#define SUBTARGET_AVR SUBTARGET(AVR)
#else
#define SUBTARGET_AVR
#endif

#ifdef LLVM_COMPONENT_M68k
#define SUBTARGET_M68K SUBTARGET(M68k)
#else
#define SUBTARGET_M68K
#endif

#ifdef LLVM_COMPONENT_CSKY
#define SUBTARGET_CSKY SUBTARGET(CSKY)
#else
#define SUBTARGET_CSKY
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

#ifdef LLVM_COMPONENT_XTENSA
#define SUBTARGET_XTENSA SUBTARGET(XTENSA)
#else
#define SUBTARGET_XTENSA
#endif

#ifdef LLVM_COMPONENT_HEXAGON
#define SUBTARGET_HEXAGON SUBTARGET(Hexagon)
#else
#define SUBTARGET_HEXAGON
#endif

#ifdef LLVM_COMPONENT_LOONGARCH
#define SUBTARGET_LOONGARCH SUBTARGET(LoongArch)
#else
#define SUBTARGET_LOONGARCH
#endif

#define GEN_SUBTARGETS                                                         \
  SUBTARGET_X86                                                                \
  SUBTARGET_ARM                                                                \
  SUBTARGET_AARCH64                                                            \
  SUBTARGET_AVR                                                                \
  SUBTARGET_M68K                                                               \
  SUBTARGET_CSKY                                                               \
  SUBTARGET_MIPS                                                               \
  SUBTARGET_PPC                                                                \
  SUBTARGET_SYSTEMZ                                                            \
  SUBTARGET_MSP430                                                             \
  SUBTARGET_SPARC                                                              \
  SUBTARGET_HEXAGON                                                            \
  SUBTARGET_XTENSA                                                             \
  SUBTARGET_RISCV                                                              \
  SUBTARGET_LOONGARCH

#define SUBTARGET(x)                                                           \
  namespace llvm {                                                             \
  extern const SubtargetFeatureKV x##FeatureKV[];                              \
  extern const SubtargetFeatureKV x##SubTypeKV[];                              \
  }

GEN_SUBTARGETS
#undef SUBTARGET

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
  std::string thin_link_data;
};

extern "C" bool LLVMRustHasFeature(LLVMTargetMachineRef TM,
                                   const char *Feature) {
  TargetMachine *Target = unwrap(TM);
  const MCSubtargetInfo *MCInfo = Target->getMCSubtargetInfo();
  return MCInfo->checkFeatures(std::string("+") + Feature);
}

enum class LLVMRustCodeModel {
  Tiny,
  Small,
  Kernel,
  Medium,
  Large,
  None,
};

static std::optional<CodeModel::Model> fromRust(LLVMRustCodeModel Model) {
  switch (Model) {
  case LLVMRustCodeModel::Tiny:
    return CodeModel::Tiny;
  case LLVMRustCodeModel::Small:
    return CodeModel::Small;
  case LLVMRustCodeModel::Kernel:
    return CodeModel::Kernel;
  case LLVMRustCodeModel::Medium:
    return CodeModel::Medium;
  case LLVMRustCodeModel::Large:
    return CodeModel::Large;
  case LLVMRustCodeModel::None:
    return std::nullopt;
  default:
    report_fatal_error("Bad CodeModel.");
  }
}

enum class LLVMRustCodeGenOptLevel {
  None,
  Less,
  Default,
  Aggressive,
};

using CodeGenOptLevelEnum = llvm::CodeGenOptLevel;

static CodeGenOptLevelEnum fromRust(LLVMRustCodeGenOptLevel Level) {
  switch (Level) {
  case LLVMRustCodeGenOptLevel::None:
    return CodeGenOptLevelEnum::None;
  case LLVMRustCodeGenOptLevel::Less:
    return CodeGenOptLevelEnum::Less;
  case LLVMRustCodeGenOptLevel::Default:
    return CodeGenOptLevelEnum::Default;
  case LLVMRustCodeGenOptLevel::Aggressive:
    return CodeGenOptLevelEnum::Aggressive;
  default:
    report_fatal_error("Bad CodeGenOptLevel.");
  }
}

enum class LLVMRustPassBuilderOptLevel {
  O0,
  O1,
  O2,
  O3,
  Os,
  Oz,
};

static OptimizationLevel fromRust(LLVMRustPassBuilderOptLevel Level) {
  switch (Level) {
  case LLVMRustPassBuilderOptLevel::O0:
    return OptimizationLevel::O0;
  case LLVMRustPassBuilderOptLevel::O1:
    return OptimizationLevel::O1;
  case LLVMRustPassBuilderOptLevel::O2:
    return OptimizationLevel::O2;
  case LLVMRustPassBuilderOptLevel::O3:
    return OptimizationLevel::O3;
  case LLVMRustPassBuilderOptLevel::Os:
    return OptimizationLevel::Os;
  case LLVMRustPassBuilderOptLevel::Oz:
    return OptimizationLevel::Oz;
  default:
    report_fatal_error("Bad PassBuilderOptLevel.");
  }
}

enum class LLVMRustRelocModel {
  Static,
  PIC,
  DynamicNoPic,
  ROPI,
  RWPI,
  ROPIRWPI,
};

static Reloc::Model fromRust(LLVMRustRelocModel RustReloc) {
  switch (RustReloc) {
  case LLVMRustRelocModel::Static:
    return Reloc::Static;
  case LLVMRustRelocModel::PIC:
    return Reloc::PIC_;
  case LLVMRustRelocModel::DynamicNoPic:
    return Reloc::DynamicNoPIC;
  case LLVMRustRelocModel::ROPI:
    return Reloc::ROPI;
  case LLVMRustRelocModel::RWPI:
    return Reloc::RWPI;
  case LLVMRustRelocModel::ROPIRWPI:
    return Reloc::ROPI_RWPI;
  }
  report_fatal_error("Bad RelocModel.");
}

enum class LLVMRustFloatABI {
  Default,
  Soft,
  Hard,
};

static FloatABI::ABIType fromRust(LLVMRustFloatABI RustFloatAbi) {
  switch (RustFloatAbi) {
  case LLVMRustFloatABI::Default:
    return FloatABI::Default;
  case LLVMRustFloatABI::Soft:
    return FloatABI::Soft;
  case LLVMRustFloatABI::Hard:
    return FloatABI::Hard;
  }
  report_fatal_error("Bad FloatABI.");
}

/// getLongestEntryLength - Return the length of the longest entry in the table.
template <typename KV> static size_t getLongestEntryLength(ArrayRef<KV> Table) {
  size_t MaxLen = 0;
  for (auto &I : Table)
    MaxLen = std::max(MaxLen, std::strlen(I.Key));
  return MaxLen;
}

extern "C" void LLVMRustPrintTargetCPUs(LLVMTargetMachineRef TM,
                                        RustStringRef OutStr) {
  ArrayRef<SubtargetSubTypeKV> CPUTable =
      unwrap(TM)->getMCSubtargetInfo()->getAllProcessorDescriptions();
  auto OS = RawRustStringOstream(OutStr);

  // Just print a bare list of target CPU names, and let Rust-side code handle
  // the full formatting of `--print=target-cpus`.
  for (auto &CPU : CPUTable) {
    OS << CPU.Key << "\n";
  }
}

extern "C" size_t LLVMRustGetTargetFeaturesCount(LLVMTargetMachineRef TM) {
  const TargetMachine *Target = unwrap(TM);
  const MCSubtargetInfo *MCInfo = Target->getMCSubtargetInfo();
  const ArrayRef<SubtargetFeatureKV> FeatTable =
      MCInfo->getAllProcessorFeatures();
  return FeatTable.size();
}

extern "C" void LLVMRustGetTargetFeature(LLVMTargetMachineRef TM, size_t Index,
                                         const char **Feature,
                                         const char **Desc) {
  const TargetMachine *Target = unwrap(TM);
  const MCSubtargetInfo *MCInfo = Target->getMCSubtargetInfo();
  const ArrayRef<SubtargetFeatureKV> FeatTable =
      MCInfo->getAllProcessorFeatures();
  const SubtargetFeatureKV Feat = FeatTable[Index];
  *Feature = Feat.Key;
  *Desc = Feat.Desc;
}

extern "C" const char *LLVMRustGetHostCPUName(size_t *OutLen) {
  StringRef Name = sys::getHostCPUName();
  *OutLen = Name.size();
  return Name.data();
}

extern "C" LLVMTargetMachineRef LLVMRustCreateTargetMachine(
    const char *TripleStr, const char *CPU, const char *Feature,
    const char *ABIStr, LLVMRustCodeModel RustCM, LLVMRustRelocModel RustReloc,
    LLVMRustCodeGenOptLevel RustOptLevel, LLVMRustFloatABI RustFloatABIType,
    bool FunctionSections, bool DataSections, bool UniqueSectionNames,
    bool TrapUnreachable, bool Singlethread, bool VerboseAsm,
    bool EmitStackSizeSection, bool RelaxELFRelocations, bool UseInitArray,
    const char *SplitDwarfFile, const char *OutputObjFile,
    const char *DebugInfoCompression, bool UseEmulatedTls,
    const char *ArgsCstrBuff, size_t ArgsCstrBuffLen) {

  auto OptLevel = fromRust(RustOptLevel);
  auto RM = fromRust(RustReloc);
  auto CM = fromRust(RustCM);
  auto FloatABIType = fromRust(RustFloatABIType);

  std::string Error;
  auto Trip = Triple(Triple::normalize(TripleStr));
  const llvm::Target *TheTarget =
      TargetRegistry::lookupTarget(Trip.getTriple(), Error);
  if (TheTarget == nullptr) {
    LLVMRustSetLastError(Error.c_str());
    return nullptr;
  }

  TargetOptions Options = codegen::InitTargetOptionsFromCodeGenFlags(Trip);

  Options.FloatABIType = FloatABIType;
  Options.DataSections = DataSections;
  Options.FunctionSections = FunctionSections;
  Options.UniqueSectionNames = UniqueSectionNames;
  Options.MCOptions.AsmVerbose = VerboseAsm;
  // Always preserve comments that were written by the user
  Options.MCOptions.PreserveAsmComments = true;
  Options.MCOptions.ABIName = ABIStr;
  if (SplitDwarfFile) {
    Options.MCOptions.SplitDwarfFile = SplitDwarfFile;
  }
  if (OutputObjFile) {
    Options.ObjectFilenameForDebug = OutputObjFile;
  }
  if (!strcmp("zlib", DebugInfoCompression) &&
      llvm::compression::zlib::isAvailable()) {
    Options.MCOptions.CompressDebugSections = DebugCompressionType::Zlib;
  } else if (!strcmp("zstd", DebugInfoCompression) &&
             llvm::compression::zstd::isAvailable()) {
    Options.MCOptions.CompressDebugSections = DebugCompressionType::Zstd;
  } else if (!strcmp("none", DebugInfoCompression)) {
    Options.MCOptions.CompressDebugSections = DebugCompressionType::None;
  }

  Options.MCOptions.X86RelaxRelocations = RelaxELFRelocations;
  Options.UseInitArray = UseInitArray;
  Options.EmulatedTLS = UseEmulatedTls;

  if (TrapUnreachable) {
    // Tell LLVM to codegen `unreachable` into an explicit trap instruction.
    // This limits the extent of possible undefined behavior in some cases, as
    // it prevents control flow from "falling through" into whatever code
    // happens to be laid out next in memory.
    Options.TrapUnreachable = true;
    // But don't emit traps after other traps or no-returns unnecessarily.
    // ...except for when targeting WebAssembly, because the NoTrapAfterNoreturn
    // option causes bugs in the LLVM WebAssembly backend. You should be able to
    // remove this check when Rust's minimum supported LLVM version is >= 18
    // https://github.com/llvm/llvm-project/pull/65876
    if (!Trip.isWasm()) {
      Options.NoTrapAfterNoreturn = true;
    }
  }

  if (Singlethread) {
    Options.ThreadModel = ThreadModel::Single;
  }

  Options.EmitStackSizeSection = EmitStackSizeSection;

  if (ArgsCstrBuff != nullptr) {
#if LLVM_VERSION_GE(20, 0)
    size_t buffer_offset = 0;
    assert(ArgsCstrBuff[ArgsCstrBuffLen - 1] == '\0');
    auto Arg0 = std::string(ArgsCstrBuff);
    buffer_offset = Arg0.size() + 1;

    std::string CommandlineArgs;
    raw_string_ostream OS(CommandlineArgs);
    ListSeparator LS(" ");
    for (StringRef Arg : split(StringRef(ArgsCstrBuff + buffer_offset,
                                         ArgsCstrBuffLen - buffer_offset),
                               '\0')) {
      OS << LS;
      sys::printArg(OS, Arg, /*Quote=*/true);
    }
    OS.flush();
    Options.MCOptions.Argv0 = Arg0;
    Options.MCOptions.CommandlineArgs = CommandlineArgs;
#else
    size_t buffer_offset = 0;
    assert(ArgsCstrBuff[ArgsCstrBuffLen - 1] == '\0');

    const size_t arg0_len = std::strlen(ArgsCstrBuff);
    char *arg0 = new char[arg0_len + 1];
    memcpy(arg0, ArgsCstrBuff, arg0_len);
    arg0[arg0_len] = '\0';
    buffer_offset += arg0_len + 1;

    const size_t num_cmd_arg_strings = std::count(
        &ArgsCstrBuff[buffer_offset], &ArgsCstrBuff[ArgsCstrBuffLen], '\0');

    std::string *cmd_arg_strings = new std::string[num_cmd_arg_strings];
    for (size_t i = 0; i < num_cmd_arg_strings; ++i) {
      assert(buffer_offset < ArgsCstrBuffLen);
      const size_t len = std::strlen(ArgsCstrBuff + buffer_offset);
      cmd_arg_strings[i] = std::string(&ArgsCstrBuff[buffer_offset], len);
      buffer_offset += len + 1;
    }

    assert(buffer_offset == ArgsCstrBuffLen);

    Options.MCOptions.Argv0 = arg0;
    Options.MCOptions.CommandLineArgs =
        llvm::ArrayRef<std::string>(cmd_arg_strings, num_cmd_arg_strings);
#endif
  }

  TargetMachine *TM = TheTarget->createTargetMachine(
      Trip.getTriple(), CPU, Feature, Options, RM, CM, OptLevel);
  return wrap(TM);
}

extern "C" void LLVMRustDisposeTargetMachine(LLVMTargetMachineRef TM) {
#if LLVM_VERSION_LT(20, 0)
  MCTargetOptions &MCOptions = unwrap(TM)->Options.MCOptions;
  delete[] MCOptions.Argv0;
  delete[] MCOptions.CommandLineArgs.data();
#endif

  delete unwrap(TM);
}

// Unfortunately, the LLVM C API doesn't provide a way to create the
// TargetLibraryInfo pass, so we use this method to do so.
extern "C" void LLVMRustAddLibraryInfo(LLVMPassManagerRef PMR, LLVMModuleRef M,
                                       bool DisableSimplifyLibCalls) {
  auto TargetTriple = Triple(unwrap(M)->getTargetTriple());
  auto TLII = TargetLibraryInfoImpl(TargetTriple);
  if (DisableSimplifyLibCalls)
    TLII.disableAllFunctions();
  unwrap(PMR)->add(new TargetLibraryInfoWrapperPass(TLII));
}

extern "C" void LLVMRustSetLLVMOptions(int Argc, char **Argv) {
  // Initializing the command-line options more than once is not allowed. So,
  // check if they've already been initialized. (This could happen if we're
  // being called from rustpkg, for example). If the arguments change, then
  // that's just kinda unfortunate.
  static bool Initialized = false;
  if (Initialized)
    return;
  Initialized = true;
  cl::ParseCommandLineOptions(Argc, Argv);
}

enum class LLVMRustFileType {
  AssemblyFile,
  ObjectFile,
};

static CodeGenFileType fromRust(LLVMRustFileType Type) {
  switch (Type) {
  case LLVMRustFileType::AssemblyFile:
    return CodeGenFileType::AssemblyFile;
  case LLVMRustFileType::ObjectFile:
    return CodeGenFileType::ObjectFile;
  default:
    report_fatal_error("Bad FileType.");
  }
}

extern "C" LLVMRustResult
LLVMRustWriteOutputFile(LLVMTargetMachineRef Target, LLVMPassManagerRef PMR,
                        LLVMModuleRef M, const char *Path, const char *DwoPath,
                        LLVMRustFileType RustFileType, bool VerifyIR) {
  llvm::legacy::PassManager *PM = unwrap<llvm::legacy::PassManager>(PMR);
  auto FileType = fromRust(RustFileType);

  std::string ErrorInfo;
  std::error_code EC;
  auto OS = raw_fd_ostream(Path, EC, sys::fs::OF_None);
  if (EC)
    ErrorInfo = EC.message();
  if (ErrorInfo != "") {
    LLVMRustSetLastError(ErrorInfo.c_str());
    return LLVMRustResult::Failure;
  }

  auto BOS = buffer_ostream(OS);
  if (DwoPath) {
    auto DOS = raw_fd_ostream(DwoPath, EC, sys::fs::OF_None);
    EC.clear();
    if (EC)
      ErrorInfo = EC.message();
    if (ErrorInfo != "") {
      LLVMRustSetLastError(ErrorInfo.c_str());
      return LLVMRustResult::Failure;
    }
    auto DBOS = buffer_ostream(DOS);
    unwrap(Target)->addPassesToEmitFile(*PM, BOS, &DBOS, FileType, !VerifyIR);
    PM->run(*unwrap(M));
  } else {
    unwrap(Target)->addPassesToEmitFile(*PM, BOS, nullptr, FileType, !VerifyIR);
    PM->run(*unwrap(M));
  }

  // Apparently `addPassesToEmitFile` adds a pointer to our on-the-stack output
  // stream (OS), so the only real safe place to delete this is here? Don't we
  // wish this was written in Rust?
  LLVMDisposePassManager(PMR);
  return LLVMRustResult::Success;
}

extern "C" typedef void (*LLVMRustSelfProfileBeforePassCallback)(
    void *,        // LlvmSelfProfiler
    const char *,  // pass name
    const char *); // IR name
extern "C" typedef void (*LLVMRustSelfProfileAfterPassCallback)(
    void *); // LlvmSelfProfiler

std::string LLVMRustwrappedIrGetName(const llvm::Any &WrappedIr) {
  if (const auto *Cast = any_cast<const Module *>(&WrappedIr))
    return (*Cast)->getName().str();
  if (const auto *Cast = any_cast<const Function *>(&WrappedIr))
    return (*Cast)->getName().str();
  if (const auto *Cast = any_cast<const Loop *>(&WrappedIr))
    return (*Cast)->getName().str();
  if (const auto *Cast = any_cast<const LazyCallGraph::SCC *>(&WrappedIr))
    return (*Cast)->getName();
  return "<UNKNOWN>";
}

void LLVMSelfProfileInitializeCallbacks(
    PassInstrumentationCallbacks &PIC, void *LlvmSelfProfiler,
    LLVMRustSelfProfileBeforePassCallback BeforePassCallback,
    LLVMRustSelfProfileAfterPassCallback AfterPassCallback) {
  PIC.registerBeforeNonSkippedPassCallback(
      [LlvmSelfProfiler, BeforePassCallback](StringRef Pass, llvm::Any Ir) {
        std::string PassName = Pass.str();
        std::string IrName = LLVMRustwrappedIrGetName(Ir);
        BeforePassCallback(LlvmSelfProfiler, PassName.c_str(), IrName.c_str());
      });

  PIC.registerAfterPassCallback(
      [LlvmSelfProfiler, AfterPassCallback](
          StringRef Pass, llvm::Any IR, const PreservedAnalyses &Preserved) {
        AfterPassCallback(LlvmSelfProfiler);
      });

  PIC.registerAfterPassInvalidatedCallback(
      [LlvmSelfProfiler,
       AfterPassCallback](StringRef Pass, const PreservedAnalyses &Preserved) {
        AfterPassCallback(LlvmSelfProfiler);
      });

  PIC.registerBeforeAnalysisCallback(
      [LlvmSelfProfiler, BeforePassCallback](StringRef Pass, llvm::Any Ir) {
        std::string PassName = Pass.str();
        std::string IrName = LLVMRustwrappedIrGetName(Ir);
        BeforePassCallback(LlvmSelfProfiler, PassName.c_str(), IrName.c_str());
      });

  PIC.registerAfterAnalysisCallback(
      [LlvmSelfProfiler, AfterPassCallback](StringRef Pass, llvm::Any Ir) {
        AfterPassCallback(LlvmSelfProfiler);
      });
}

enum class LLVMRustOptStage {
  PreLinkNoLTO,
  PreLinkThinLTO,
  PreLinkFatLTO,
  ThinLTO,
  FatLTO,
};

struct LLVMRustSanitizerOptions {
  bool SanitizeAddress;
  bool SanitizeAddressRecover;
  bool SanitizeCFI;
  bool SanitizeDataFlow;
  char **SanitizeDataFlowABIList;
  size_t SanitizeDataFlowABIListLen;
  bool SanitizeKCFI;
  bool SanitizeMemory;
  bool SanitizeMemoryRecover;
  int SanitizeMemoryTrackOrigins;
  bool SanitizeThread;
  bool SanitizeHWAddress;
  bool SanitizeHWAddressRecover;
  bool SanitizeKernelAddress;
  bool SanitizeKernelAddressRecover;
};

// This symbol won't be available or used when Enzyme is not enabled.
// Always set AugmentPassBuilder to true, since it registers optimizations which
// will improve the performance for Enzyme.
#ifdef ENZYME
extern "C" void registerEnzymeAndPassPipeline(llvm::PassBuilder &PB,
                                              /* augmentPassBuilder */ bool);
#endif

extern "C" LLVMRustResult LLVMRustOptimize(
    LLVMModuleRef ModuleRef, LLVMTargetMachineRef TMRef,
    LLVMRustPassBuilderOptLevel OptLevelRust, LLVMRustOptStage OptStage,
    bool IsLinkerPluginLTO, bool NoPrepopulatePasses, bool VerifyIR,
    bool LintIR, LLVMRustThinLTOBuffer **ThinLTOBufferRef, bool EmitThinLTO,
    bool EmitThinLTOSummary, bool MergeFunctions, bool UnrollLoops,
    bool SLPVectorize, bool LoopVectorize, bool DisableSimplifyLibCalls,
    bool EmitLifetimeMarkers, bool RunEnzyme, bool PrintBeforeEnzyme,
    bool PrintAfterEnzyme, bool PrintPasses,
    LLVMRustSanitizerOptions *SanitizerOptions, const char *PGOGenPath,
    const char *PGOUsePath, bool InstrumentCoverage,
    const char *InstrProfileOutput, const char *PGOSampleUsePath,
    bool DebugInfoForProfiling, void *LlvmSelfProfiler,
    LLVMRustSelfProfileBeforePassCallback BeforePassCallback,
    LLVMRustSelfProfileAfterPassCallback AfterPassCallback,
    const char *ExtraPasses, size_t ExtraPassesLen, const char *LLVMPlugins,
    size_t LLVMPluginsLen) {
  Module *TheModule = unwrap(ModuleRef);
  TargetMachine *TM = unwrap(TMRef);
  OptimizationLevel OptLevel = fromRust(OptLevelRust);

  PipelineTuningOptions PTO;
  PTO.LoopUnrolling = UnrollLoops;
  PTO.LoopInterleaving = UnrollLoops;
  PTO.LoopVectorization = LoopVectorize;
  PTO.SLPVectorization = SLPVectorize;
  PTO.MergeFunctions = MergeFunctions;

  PassInstrumentationCallbacks PIC;

  if (LlvmSelfProfiler) {
    LLVMSelfProfileInitializeCallbacks(PIC, LlvmSelfProfiler,
                                       BeforePassCallback, AfterPassCallback);
  }

  std::optional<PGOOptions> PGOOpt;
  auto FS = vfs::getRealFileSystem();
  if (PGOGenPath) {
    assert(!PGOUsePath && !PGOSampleUsePath);
    PGOOpt = PGOOptions(
        PGOGenPath, "", "", "", FS, PGOOptions::IRInstr, PGOOptions::NoCSAction,
        PGOOptions::ColdFuncOpt::Default, DebugInfoForProfiling);
  } else if (PGOUsePath) {
    assert(!PGOSampleUsePath);
    PGOOpt = PGOOptions(
        PGOUsePath, "", "", "", FS, PGOOptions::IRUse, PGOOptions::NoCSAction,
        PGOOptions::ColdFuncOpt::Default, DebugInfoForProfiling);
  } else if (PGOSampleUsePath) {
    PGOOpt =
        PGOOptions(PGOSampleUsePath, "", "", "", FS, PGOOptions::SampleUse,
                   PGOOptions::NoCSAction, PGOOptions::ColdFuncOpt::Default,
                   DebugInfoForProfiling);
  } else if (DebugInfoForProfiling) {
    PGOOpt = PGOOptions(
        "", "", "", "", FS, PGOOptions::NoAction, PGOOptions::NoCSAction,
        PGOOptions::ColdFuncOpt::Default, DebugInfoForProfiling);
  }

  auto PB = PassBuilder(TM, PTO, PGOOpt, &PIC);
  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;

  StandardInstrumentations SI(TheModule->getContext(),
                              /*DebugLogging=*/false);
  SI.registerCallbacks(PIC, &MAM);

  if (LLVMPluginsLen) {
    auto PluginsStr = StringRef(LLVMPlugins, LLVMPluginsLen);
    SmallVector<StringRef> Plugins;
    PluginsStr.split(Plugins, ',', -1, false);
    for (auto PluginPath : Plugins) {
      auto Plugin = PassPlugin::Load(PluginPath.str());
      if (!Plugin) {
        auto Err = Plugin.takeError();
        auto ErrMsg = llvm::toString(std::move(Err));
        LLVMRustSetLastError(ErrMsg.c_str());
        return LLVMRustResult::Failure;
      }
      Plugin->registerPassBuilderCallbacks(PB);
    }
  }

  FAM.registerPass([&] { return PB.buildDefaultAAPipeline(); });

  Triple TargetTriple(TheModule->getTargetTriple());
  std::unique_ptr<TargetLibraryInfoImpl> TLII(
      new TargetLibraryInfoImpl(TargetTriple));
  if (DisableSimplifyLibCalls)
    TLII->disableAllFunctions();
  FAM.registerPass([&] { return TargetLibraryAnalysis(*TLII); });

  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  // We manually collect pipeline callbacks so we can apply them at O0, where
  // the PassBuilder does not create a pipeline.
  std::vector<std::function<void(ModulePassManager &, OptimizationLevel)>>
      PipelineStartEPCallbacks;
#if LLVM_VERSION_GE(20, 0)
  std::vector<std::function<void(ModulePassManager &, OptimizationLevel,
                                 ThinOrFullLTOPhase)>>
      OptimizerLastEPCallbacks;
#else
  std::vector<std::function<void(ModulePassManager &, OptimizationLevel)>>
      OptimizerLastEPCallbacks;
#endif

  if (!IsLinkerPluginLTO && SanitizerOptions && SanitizerOptions->SanitizeCFI &&
      !NoPrepopulatePasses) {
    PipelineStartEPCallbacks.push_back(
        [](ModulePassManager &MPM, OptimizationLevel Level) {
          MPM.addPass(LowerTypeTestsPass(
              /*ExportSummary=*/nullptr,
              /*ImportSummary=*/nullptr));
        });
  }

  if (VerifyIR) {
    PipelineStartEPCallbacks.push_back(
        [VerifyIR](ModulePassManager &MPM, OptimizationLevel Level) {
          MPM.addPass(VerifierPass());
        });
  }

  if (LintIR) {
    PipelineStartEPCallbacks.push_back([](ModulePassManager &MPM,
                                          OptimizationLevel Level) {
#if LLVM_VERSION_GE(21, 0)
      MPM.addPass(
          createModuleToFunctionPassAdaptor(LintPass(/*AbortOnError=*/true)));
#else
      MPM.addPass(createModuleToFunctionPassAdaptor(LintPass()));
#endif
    });
  }

  if (InstrumentCoverage) {
    PipelineStartEPCallbacks.push_back(
        [InstrProfileOutput](ModulePassManager &MPM, OptimizationLevel Level) {
          InstrProfOptions Options;
          if (InstrProfileOutput) {
            Options.InstrProfileOutput = InstrProfileOutput;
          }
          // cargo run tests in multhreading mode by default
          // so use atomics for coverage counters
          Options.Atomic = true;
          MPM.addPass(InstrProfilingLoweringPass(Options, false));
        });
  }

  if (SanitizerOptions) {
    if (SanitizerOptions->SanitizeDataFlow) {
      std::vector<std::string> ABIListFiles(
          SanitizerOptions->SanitizeDataFlowABIList,
          SanitizerOptions->SanitizeDataFlowABIList +
              SanitizerOptions->SanitizeDataFlowABIListLen);
      OptimizerLastEPCallbacks.push_back(
#if LLVM_VERSION_GE(20, 0)
          [ABIListFiles](ModulePassManager &MPM, OptimizationLevel Level,
                         ThinOrFullLTOPhase phase) {
#else
          [ABIListFiles](ModulePassManager &MPM, OptimizationLevel Level) {
#endif
            MPM.addPass(DataFlowSanitizerPass(ABIListFiles));
          });
    }

    if (SanitizerOptions->SanitizeMemory) {
      MemorySanitizerOptions Options(
          SanitizerOptions->SanitizeMemoryTrackOrigins,
          SanitizerOptions->SanitizeMemoryRecover,
          /*CompileKernel=*/false,
          /*EagerChecks=*/true);
      OptimizerLastEPCallbacks.push_back(
#if LLVM_VERSION_GE(20, 0)
          [Options](ModulePassManager &MPM, OptimizationLevel Level,
                    ThinOrFullLTOPhase phase) {
#else
          [Options](ModulePassManager &MPM, OptimizationLevel Level) {
#endif
            MPM.addPass(MemorySanitizerPass(Options));
          });
    }

    if (SanitizerOptions->SanitizeThread) {
      OptimizerLastEPCallbacks.push_back(
#if LLVM_VERSION_GE(20, 0)
          [](ModulePassManager &MPM, OptimizationLevel Level,
             ThinOrFullLTOPhase phase) {
#else
          [](ModulePassManager &MPM, OptimizationLevel Level) {
#endif
            MPM.addPass(ModuleThreadSanitizerPass());
            MPM.addPass(
                createModuleToFunctionPassAdaptor(ThreadSanitizerPass()));
          });
    }

    if (SanitizerOptions->SanitizeAddress ||
        SanitizerOptions->SanitizeKernelAddress) {
      OptimizerLastEPCallbacks.push_back(
#if LLVM_VERSION_GE(20, 0)
          [SanitizerOptions, TM](ModulePassManager &MPM,
                                 OptimizationLevel Level,
                                 ThinOrFullLTOPhase phase) {
#else
          [SanitizerOptions, TM](ModulePassManager &MPM,
                                 OptimizationLevel Level) {
#endif
            auto CompileKernel = SanitizerOptions->SanitizeKernelAddress;
            AddressSanitizerOptions opts = AddressSanitizerOptions{
                CompileKernel,
                SanitizerOptions->SanitizeAddressRecover ||
                    SanitizerOptions->SanitizeKernelAddressRecover,
                /*UseAfterScope=*/true,
                AsanDetectStackUseAfterReturnMode::Runtime,
            };
            MPM.addPass(AddressSanitizerPass(
                opts,
                /*UseGlobalGC*/ true,
                // UseOdrIndicator should be false on windows machines
                // https://reviews.llvm.org/D137227
                !TM->getTargetTriple().isOSWindows()));
          });
    }
    if (SanitizerOptions->SanitizeHWAddress) {
      OptimizerLastEPCallbacks.push_back(
#if LLVM_VERSION_GE(20, 0)
          [SanitizerOptions](ModulePassManager &MPM, OptimizationLevel Level,
                             ThinOrFullLTOPhase phase) {
#else
          [SanitizerOptions](ModulePassManager &MPM, OptimizationLevel Level) {
#endif
            HWAddressSanitizerOptions opts(
                /*CompileKernel=*/false,
                SanitizerOptions->SanitizeHWAddressRecover,
                /*DisableOptimization=*/false);
            MPM.addPass(HWAddressSanitizerPass(opts));
          });
    }
  }

  ModulePassManager MPM;
  bool NeedThinLTOBufferPasses = EmitThinLTO;
  auto ThinLTOBuffer = std::make_unique<LLVMRustThinLTOBuffer>();
  raw_string_ostream ThinLTODataOS(ThinLTOBuffer->data);
  raw_string_ostream ThinLinkDataOS(ThinLTOBuffer->thin_link_data);
  if (!NoPrepopulatePasses) {
    // The pre-link pipelines don't support O0 and require using
    // buildO0DefaultPipeline() instead. At the same time, the LTO pipelines do
    // support O0 and using them is required.
    bool IsLTO = OptStage == LLVMRustOptStage::ThinLTO ||
                 OptStage == LLVMRustOptStage::FatLTO;
    if (OptLevel == OptimizationLevel::O0 && !IsLTO) {
      for (const auto &C : PipelineStartEPCallbacks)
        PB.registerPipelineStartEPCallback(C);
      for (const auto &C : OptimizerLastEPCallbacks)
        PB.registerOptimizerLastEPCallback(C);

      // We manually schedule ThinLTOBufferPasses below, so don't pass the value
      // to enable it here.
      MPM = PB.buildO0DefaultPipeline(OptLevel);
    } else {
      for (const auto &C : PipelineStartEPCallbacks)
        PB.registerPipelineStartEPCallback(C);
      for (const auto &C : OptimizerLastEPCallbacks)
        PB.registerOptimizerLastEPCallback(C);

      switch (OptStage) {
      case LLVMRustOptStage::PreLinkNoLTO:
        if (ThinLTOBufferRef) {
          // This is similar to LLVM's `buildFatLTODefaultPipeline`, where the
          // bitcode for embedding is obtained after performing
          // `ThinLTOPreLinkDefaultPipeline`.
          MPM.addPass(PB.buildThinLTOPreLinkDefaultPipeline(OptLevel));
          if (EmitThinLTO) {
            MPM.addPass(ThinLTOBitcodeWriterPass(
                ThinLTODataOS, EmitThinLTOSummary ? &ThinLinkDataOS : nullptr));
          } else {
            MPM.addPass(BitcodeWriterPass(ThinLTODataOS));
          }
          *ThinLTOBufferRef = ThinLTOBuffer.release();
          MPM.addPass(PB.buildModuleOptimizationPipeline(
              OptLevel, ThinOrFullLTOPhase::None));
          MPM.addPass(
              createModuleToFunctionPassAdaptor(AnnotationRemarksPass()));
        } else {
          MPM = PB.buildPerModuleDefaultPipeline(OptLevel);
        }
        break;
      case LLVMRustOptStage::PreLinkThinLTO:
        MPM = PB.buildThinLTOPreLinkDefaultPipeline(OptLevel);
        NeedThinLTOBufferPasses = false;
        break;
      case LLVMRustOptStage::PreLinkFatLTO:
        MPM = PB.buildLTOPreLinkDefaultPipeline(OptLevel);
        NeedThinLTOBufferPasses = false;
        break;
      case LLVMRustOptStage::ThinLTO:
        // FIXME: Does it make sense to pass the ModuleSummaryIndex?
        // It only seems to be needed for C++ specific optimizations.
        MPM = PB.buildThinLTODefaultPipeline(OptLevel, nullptr);
        break;
      case LLVMRustOptStage::FatLTO:
        MPM = PB.buildLTODefaultPipeline(OptLevel, nullptr);
        break;
      }
    }
  } else {
    // We're not building any of the default pipelines but we still want to
    // add the verifier, instrumentation, etc passes if they were requested
    for (const auto &C : PipelineStartEPCallbacks)
      C(MPM, OptLevel);
    for (const auto &C : OptimizerLastEPCallbacks)
#if LLVM_VERSION_GE(20, 0)
      C(MPM, OptLevel, ThinOrFullLTOPhase::None);
#else
      C(MPM, OptLevel);
#endif
  }

  if (ExtraPassesLen) {
    if (auto Err =
            PB.parsePassPipeline(MPM, StringRef(ExtraPasses, ExtraPassesLen))) {
      std::string ErrMsg = toString(std::move(Err));
      LLVMRustSetLastError(ErrMsg.c_str());
      return LLVMRustResult::Failure;
    }
  }

  if (NeedThinLTOBufferPasses) {
    MPM.addPass(CanonicalizeAliasesPass());
    MPM.addPass(NameAnonGlobalPass());
  }
  // For `-Copt-level=0`, ThinLTO, or LTO.
  if (ThinLTOBufferRef && *ThinLTOBufferRef == nullptr) {
    if (EmitThinLTO) {
      MPM.addPass(ThinLTOBitcodeWriterPass(
          ThinLTODataOS, EmitThinLTOSummary ? &ThinLinkDataOS : nullptr));
    } else {
      MPM.addPass(BitcodeWriterPass(ThinLTODataOS));
    }
    *ThinLTOBufferRef = ThinLTOBuffer.release();
  }

  // now load "-enzyme" pass:
#ifdef ENZYME
  if (RunEnzyme) {

    if (PrintBeforeEnzyme) {
      // Handle the Rust flag `-Zautodiff=PrintModBefore`.
      std::string Banner = "Module before EnzymeNewPM";
      MPM.addPass(PrintModulePass(outs(), Banner, true, false));
    }

    registerEnzymeAndPassPipeline(PB, false);
    if (auto Err = PB.parsePassPipeline(MPM, "enzyme")) {
      std::string ErrMsg = toString(std::move(Err));
      LLVMRustSetLastError(ErrMsg.c_str());
      return LLVMRustResult::Failure;
    }

    if (PrintAfterEnzyme) {
      // Handle the Rust flag `-Zautodiff=PrintModAfter`.
      std::string Banner = "Module after EnzymeNewPM";
      MPM.addPass(PrintModulePass(outs(), Banner, true, false));
    }
  }
#endif
  if (PrintPasses) {
    // Print all passes from the PM:
    std::string Pipeline;
    raw_string_ostream SOS(Pipeline);
    MPM.printPipeline(SOS, [&PIC](StringRef ClassName) {
      auto PassName = PIC.getPassNameForClassName(ClassName);
      return PassName.empty() ? ClassName : PassName;
    });
    outs() << Pipeline;
    outs() << "\n";
  }

  // Upgrade all calls to old intrinsics first.
  for (Module::iterator I = TheModule->begin(), E = TheModule->end(); I != E;)
    UpgradeCallsToIntrinsic(&*I++); // must be post-increment, as we remove

  MPM.run(*TheModule, MAM);
  return LLVMRustResult::Success;
}

// Callback to demangle function name
// Parameters:
// * name to be demangled
// * name len
// * output buffer
// * output buffer len
// Returns len of demangled string, or 0 if demangle failed.
typedef size_t (*DemangleFn)(const char *, size_t, char *, size_t);

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
      Value = CI->getCalledOperand();
    } else if (const InvokeInst *II = dyn_cast<InvokeInst>(I)) {
      Name = "invoke";
      Value = II->getCalledOperand();
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

} // namespace

extern "C" LLVMRustResult LLVMRustPrintModule(LLVMModuleRef M, const char *Path,
                                              DemangleFn Demangle) {
  std::string ErrorInfo;
  std::error_code EC;
  auto OS = raw_fd_ostream(Path, EC, sys::fs::OF_None);
  if (EC)
    ErrorInfo = EC.message();
  if (ErrorInfo != "") {
    LLVMRustSetLastError(ErrorInfo.c_str());
    return LLVMRustResult::Failure;
  }

  auto AAW = RustAssemblyAnnotationWriter(Demangle);
  auto FOS = formatted_raw_ostream(OS);
  unwrap(M)->print(FOS, &AAW);

  return LLVMRustResult::Success;
}

extern "C" void LLVMRustPrintPasses() {
  PassBuilder PB;
  PB.printPassNames(outs());
}

extern "C" void LLVMRustRunRestrictionPass(LLVMModuleRef M, char **Symbols,
                                           size_t Len) {
  auto PreserveFunctions = [=](const GlobalValue &GV) {
    // Preserve LLVM-injected, ASAN-related symbols.
    // See also https://github.com/rust-lang/rust/issues/113404.
    if (GV.getName() == "___asan_globals_registered") {
      return true;
    }

    // Preserve symbols exported from Rust modules.
    for (size_t I = 0; I < Len; I++) {
      if (GV.getName() == Symbols[I]) {
        return true;
      }
    }
    return false;
  };

  internalizeModule(*unwrap(M), PreserveFunctions);
}

extern "C" void
LLVMRustSetDataLayoutFromTargetMachine(LLVMModuleRef Module,
                                       LLVMTargetMachineRef TMR) {
  TargetMachine *Target = unwrap(TMR);
  unwrap(Module)->setDataLayout(Target->createDataLayout());
}

extern "C" void LLVMRustSetModulePICLevel(LLVMModuleRef M) {
  unwrap(M)->setPICLevel(PICLevel::Level::BigPIC);
}

extern "C" void LLVMRustSetModulePIELevel(LLVMModuleRef M) {
  unwrap(M)->setPIELevel(PIELevel::Level::Large);
}

extern "C" void LLVMRustSetModuleCodeModel(LLVMModuleRef M,
                                           LLVMRustCodeModel Model) {
  auto CM = fromRust(Model);
  if (!CM)
    return;
  unwrap(M)->setCodeModel(*CM);
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
#if LLVM_VERSION_GE(20, 0)
  FunctionImporter::ImportListsTy ImportLists;
#else
  DenseMap<StringRef, FunctionImporter::ImportMapTy> ImportLists;
#endif
  DenseMap<StringRef, FunctionImporter::ExportSetTy> ExportLists;
  DenseMap<StringRef, GVSummaryMapTy> ModuleToDefinedGVSummaries;
  StringMap<std::map<GlobalValue::GUID, GlobalValue::LinkageTypes>> ResolvedODR;

  LLVMRustThinLTOData() : Index(/* HaveGVs = */ false) {}
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
extern "C" LLVMRustThinLTOData *
LLVMRustCreateThinLTOData(LLVMRustThinLTOModule *modules, size_t num_modules,
                          const char **preserved_symbols, size_t num_symbols) {
  auto Ret = std::make_unique<LLVMRustThinLTOData>();

  // Load each module's summary and merge it into one combined index
  for (size_t i = 0; i < num_modules; i++) {
    auto module = &modules[i];
    auto buffer = StringRef(module->data, module->len);
    auto mem_buffer = MemoryBufferRef(buffer, module->identifier);

    Ret->ModuleMap[module->identifier] = mem_buffer;

    if (Error Err = readModuleSummaryIndex(mem_buffer, Ret->Index)) {
      LLVMRustSetLastError(toString(std::move(Err)).c_str());
      return nullptr;
    }
  }

  // Collect for each module the list of function it defines (GUID -> Summary)
  Ret->Index.collectDefinedGVSummariesPerModule(
      Ret->ModuleToDefinedGVSummaries);

  // Convert the preserved symbols set from string to GUID, this is then needed
  // for internalization.
  for (size_t i = 0; i < num_symbols; i++) {
    auto GUID = GlobalValue::getGUID(preserved_symbols[i]);
    Ret->GUIDPreservedSymbols.insert(GUID);
  }

  // Collect the import/export lists for all modules from the call-graph in the
  // combined index
  //
  // This is copied from `lib/LTO/ThinLTOCodeGenerator.cpp`
  auto deadIsPrevailing = [&](GlobalValue::GUID G) {
    return PrevailingType::Unknown;
  };
  // We don't have a complete picture in our use of ThinLTO, just our immediate
  // crate, so we need `ImportEnabled = false` to limit internalization.
  // Otherwise, we sometimes lose `static` values -- see #60184.
  computeDeadSymbolsWithConstProp(Ret->Index, Ret->GUIDPreservedSymbols,
                                  deadIsPrevailing,
                                  /* ImportEnabled = */ false);
  // Resolve LinkOnce/Weak symbols, this has to be computed early be cause it
  // impacts the caching.
  //
  // This is copied from `lib/LTO/ThinLTOCodeGenerator.cpp` with some of this
  // being lifted from `lib/LTO/LTO.cpp` as well
  DenseMap<GlobalValue::GUID, const GlobalValueSummary *> PrevailingCopy;
  for (auto &I : Ret->Index) {
    if (I.second.SummaryList.size() > 1)
      PrevailingCopy[I.first] =
          getFirstDefinitionForLinker(I.second.SummaryList);
  }
  auto isPrevailing = [&](GlobalValue::GUID GUID, const GlobalValueSummary *S) {
    const auto &Prevailing = PrevailingCopy.find(GUID);
    if (Prevailing == PrevailingCopy.end())
      return true;
    return Prevailing->second == S;
  };
  ComputeCrossModuleImport(Ret->Index, Ret->ModuleToDefinedGVSummaries,
                           isPrevailing, Ret->ImportLists, Ret->ExportLists);

  auto recordNewLinkage = [&](StringRef ModuleIdentifier,
                              GlobalValue::GUID GUID,
                              GlobalValue::LinkageTypes NewLinkage) {
    Ret->ResolvedODR[ModuleIdentifier][GUID] = NewLinkage;
  };

  // Uses FromPrevailing visibility scheme which works for many binary
  // formats. We probably could and should use ELF visibility scheme for many of
  // our targets, however.
  lto::Config conf;
  thinLTOResolvePrevailingInIndex(conf, Ret->Index, isPrevailing,
                                  recordNewLinkage, Ret->GUIDPreservedSymbols);

  // Here we calculate an `ExportedGUIDs` set for use in the `isExported`
  // callback below. This callback below will dictate the linkage for all
  // summaries in the index, and we basically just only want to ensure that dead
  // symbols are internalized. Otherwise everything that's already external
  // linkage will stay as external, and internal will stay as internal.
  std::set<GlobalValue::GUID> ExportedGUIDs;
  for (auto &List : Ret->Index) {
    for (auto &GVS : List.second.SummaryList) {
      if (GlobalValue::isLocalLinkage(GVS->linkage()))
        continue;
      auto GUID = GVS->getOriginalName();
      if (GVS->flags().Live)
        ExportedGUIDs.insert(GUID);
    }
  }
  auto isExported = [&](StringRef ModuleIdentifier, ValueInfo VI) {
    const auto &ExportList = Ret->ExportLists.find(ModuleIdentifier);
    return (ExportList != Ret->ExportLists.end() &&
            ExportList->second.count(VI)) ||
           ExportedGUIDs.count(VI.getGUID());
  };
  thinLTOInternalizeAndPromoteInIndex(Ret->Index, isExported, isPrevailing);

  return Ret.release();
}

extern "C" void LLVMRustFreeThinLTOData(LLVMRustThinLTOData *Data) {
  delete Data;
}

// Below are the various passes that happen *per module* when doing ThinLTO.
//
// In other words, these are the functions that are all run concurrently
// with one another, one per module. The passes here correspond to the analysis
// passes in `lib/LTO/ThinLTOCodeGenerator.cpp`, currently found in the
// `ProcessThinLTOModule` function. Here they're split up into separate steps
// so rustc can save off the intermediate bytecode between each step.

static bool clearDSOLocalOnDeclarations(Module &Mod, TargetMachine &TM) {
  // When linking an ELF shared object, dso_local should be dropped. We
  // conservatively do this for -fpic.
  bool ClearDSOLocalOnDeclarations = TM.getTargetTriple().isOSBinFormatELF() &&
                                     TM.getRelocationModel() != Reloc::Static &&
                                     Mod.getPIELevel() == PIELevel::Default;
  return ClearDSOLocalOnDeclarations;
}

extern "C" void LLVMRustPrepareThinLTORename(const LLVMRustThinLTOData *Data,
                                             LLVMModuleRef M,
                                             LLVMTargetMachineRef TM) {
  Module &Mod = *unwrap(M);
  TargetMachine &Target = *unwrap(TM);

  bool ClearDSOLocal = clearDSOLocalOnDeclarations(Mod, Target);
  renameModuleForThinLTO(Mod, Data->Index, ClearDSOLocal);
}

extern "C" bool
LLVMRustPrepareThinLTOResolveWeak(const LLVMRustThinLTOData *Data,
                                  LLVMModuleRef M) {
  Module &Mod = *unwrap(M);
  const auto &DefinedGlobals =
      Data->ModuleToDefinedGVSummaries.lookup(Mod.getModuleIdentifier());
  thinLTOFinalizeInModule(Mod, DefinedGlobals, /*PropagateAttrs=*/true);
  return true;
}

extern "C" bool
LLVMRustPrepareThinLTOInternalize(const LLVMRustThinLTOData *Data,
                                  LLVMModuleRef M) {
  Module &Mod = *unwrap(M);
  const auto &DefinedGlobals =
      Data->ModuleToDefinedGVSummaries.lookup(Mod.getModuleIdentifier());
  thinLTOInternalizeModule(Mod, DefinedGlobals);
  return true;
}

extern "C" bool LLVMRustPrepareThinLTOImport(const LLVMRustThinLTOData *Data,
                                             LLVMModuleRef M,
                                             LLVMTargetMachineRef TM) {
  Module &Mod = *unwrap(M);
  TargetMachine &Target = *unwrap(TM);

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

    auto *WasmCustomSections =
        (*MOrErr)->getNamedMetadata("wasm.custom_sections");
    if (WasmCustomSections)
      WasmCustomSections->eraseFromParent();

    // `llvm.ident` named metadata also gets duplicated.
    auto *llvmIdent = (*MOrErr)->getNamedMetadata("llvm.ident");
    if (llvmIdent)
      llvmIdent->eraseFromParent();

    return MOrErr;
  };
  bool ClearDSOLocal = clearDSOLocalOnDeclarations(Mod, Target);
  auto Importer = FunctionImporter(Data->Index, Loader, ClearDSOLocal);
  Expected<bool> Result = Importer.importFunctions(Mod, ImportList);
  if (!Result) {
    LLVMRustSetLastError(toString(Result.takeError()).c_str());
    return false;
  }
  return true;
}

extern "C" LLVMRustThinLTOBuffer *
LLVMRustThinLTOBufferCreate(LLVMModuleRef M, bool is_thin, bool emit_summary) {
  auto Ret = std::make_unique<LLVMRustThinLTOBuffer>();
  {
    auto OS = raw_string_ostream(Ret->data);
    auto ThinLinkOS = raw_string_ostream(Ret->thin_link_data);
    {
      if (is_thin) {
        PassBuilder PB;
        LoopAnalysisManager LAM;
        FunctionAnalysisManager FAM;
        CGSCCAnalysisManager CGAM;
        ModuleAnalysisManager MAM;
        PB.registerModuleAnalyses(MAM);
        PB.registerCGSCCAnalyses(CGAM);
        PB.registerFunctionAnalyses(FAM);
        PB.registerLoopAnalyses(LAM);
        PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);
        ModulePassManager MPM;
        // We only pass ThinLinkOS to be filled in if we want the summary,
        // because otherwise LLVM does extra work and may double-emit some
        // errors or warnings.
        MPM.addPass(
            ThinLTOBitcodeWriterPass(OS, emit_summary ? &ThinLinkOS : nullptr));
        MPM.run(*unwrap(M), MAM);
      } else {
        WriteBitcodeToFile(*unwrap(M), OS);
      }
    }
  }
  return Ret.release();
}

extern "C" void LLVMRustThinLTOBufferFree(LLVMRustThinLTOBuffer *Buffer) {
  delete Buffer;
}

extern "C" const void *
LLVMRustThinLTOBufferPtr(const LLVMRustThinLTOBuffer *Buffer) {
  return Buffer->data.data();
}

extern "C" size_t
LLVMRustThinLTOBufferLen(const LLVMRustThinLTOBuffer *Buffer) {
  return Buffer->data.length();
}

extern "C" const void *
LLVMRustThinLTOBufferThinLinkDataPtr(const LLVMRustThinLTOBuffer *Buffer) {
  return Buffer->thin_link_data.data();
}

extern "C" size_t
LLVMRustThinLTOBufferThinLinkDataLen(const LLVMRustThinLTOBuffer *Buffer) {
  return Buffer->thin_link_data.length();
}

// This is what we used to parse upstream bitcode for actual ThinLTO
// processing. We'll call this once per module optimized through ThinLTO, and
// it'll be called concurrently on many threads.
extern "C" LLVMModuleRef LLVMRustParseBitcodeForLTO(LLVMContextRef Context,
                                                    const char *data,
                                                    size_t len,
                                                    const char *identifier) {
  auto Data = StringRef(data, len);
  auto Buffer = MemoryBufferRef(Data, identifier);
  unwrap(Context)->enableDebugTypeODRUniquing();
  Expected<std::unique_ptr<Module>> SrcOrError =
      parseBitcodeFile(Buffer, *unwrap(Context));
  if (!SrcOrError) {
    LLVMRustSetLastError(toString(SrcOrError.takeError()).c_str());
    return nullptr;
  }
  return wrap(std::move(*SrcOrError).release());
}

// Find a section of an object file by name. Fail if the section is missing or
// empty.
extern "C" const char *LLVMRustGetSliceFromObjectDataByName(const char *data,
                                                            size_t len,
                                                            const char *name,
                                                            size_t name_len,
                                                            size_t *out_len) {
  *out_len = 0;
  auto Name = StringRef(name, name_len);
  auto Data = StringRef(data, len);
  auto Buffer = MemoryBufferRef(Data, ""); // The id is unused.
  file_magic Type = identify_magic(Buffer.getBuffer());
  Expected<std::unique_ptr<object::ObjectFile>> ObjFileOrError =
      object::ObjectFile::createObjectFile(Buffer, Type);
  if (!ObjFileOrError) {
    LLVMRustSetLastError(toString(ObjFileOrError.takeError()).c_str());
    return nullptr;
  }
  for (const object::SectionRef &Sec : (*ObjFileOrError)->sections()) {
    Expected<StringRef> SecName = Sec.getName();
    if (SecName && *SecName == Name) {
      Expected<StringRef> SectionOrError = Sec.getContents();
      if (!SectionOrError) {
        LLVMRustSetLastError(toString(SectionOrError.takeError()).c_str());
        return nullptr;
      }
      *out_len = SectionOrError->size();
      return SectionOrError->data();
    }
  }
  LLVMRustSetLastError("could not find requested section");
  return nullptr;
}

// Computes the LTO cache key for the provided 'ModId' in the given 'Data',
// storing the result in 'KeyOut'.
// Currently, this cache key is a SHA-1 hash of anything that could affect
// the result of optimizing this module (e.g. module imports, exports, liveness
// of access globals, etc).
// The precise details are determined by LLVM in `computeLTOCacheKey`, which is
// used during the normal linker-plugin incremental thin-LTO process.
extern "C" void LLVMRustComputeLTOCacheKey(RustStringRef KeyOut,
                                           const char *ModId,
                                           LLVMRustThinLTOData *Data) {
  SmallString<40> Key;
  llvm::lto::Config conf;
  const auto &ImportList = Data->ImportLists.lookup(ModId);
  const auto &ExportList = Data->ExportLists.lookup(ModId);
  const auto &ResolvedODR = Data->ResolvedODR.lookup(ModId);
  const auto &DefinedGlobals = Data->ModuleToDefinedGVSummaries.lookup(ModId);
#if LLVM_VERSION_GE(20, 0)
  DenseSet<GlobalValue::GUID> CfiFunctionDefs;
  DenseSet<GlobalValue::GUID> CfiFunctionDecls;
#else
  std::set<GlobalValue::GUID> CfiFunctionDefs;
  std::set<GlobalValue::GUID> CfiFunctionDecls;
#endif

  // Based on the 'InProcessThinBackend' constructor in LLVM
#if LLVM_VERSION_GE(21, 0)
  for (auto &Name : Data->Index.cfiFunctionDefs().symbols())
    CfiFunctionDefs.insert(
        GlobalValue::getGUID(GlobalValue::dropLLVMManglingEscape(Name)));
  for (auto &Name : Data->Index.cfiFunctionDecls().symbols())
    CfiFunctionDecls.insert(
        GlobalValue::getGUID(GlobalValue::dropLLVMManglingEscape(Name)));
#else
  for (auto &Name : Data->Index.cfiFunctionDefs())
    CfiFunctionDefs.insert(
        GlobalValue::getGUID(GlobalValue::dropLLVMManglingEscape(Name)));
  for (auto &Name : Data->Index.cfiFunctionDecls())
    CfiFunctionDecls.insert(
        GlobalValue::getGUID(GlobalValue::dropLLVMManglingEscape(Name)));
#endif

#if LLVM_VERSION_GE(20, 0)
  Key = llvm::computeLTOCacheKey(conf, Data->Index, ModId, ImportList,
                                 ExportList, ResolvedODR, DefinedGlobals,
                                 CfiFunctionDefs, CfiFunctionDecls);
#else
  llvm::computeLTOCacheKey(Key, conf, Data->Index, ModId, ImportList,
                           ExportList, ResolvedODR, DefinedGlobals,
                           CfiFunctionDefs, CfiFunctionDecls);
#endif

  auto OS = RawRustStringOstream(KeyOut);
  OS << Key.str();
}
