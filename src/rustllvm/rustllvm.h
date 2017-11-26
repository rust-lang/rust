// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#include "llvm-c/BitReader.h"
#include "llvm-c/Core.h"
#include "llvm-c/ExecutionEngine.h"
#include "llvm-c/Object.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Analysis/Lint.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/Interpreter.h"
#include "llvm/ExecutionEngine/MCJIT.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IRPrintingPasses.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/ArchiveWriter.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/Memory.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/Instrumentation.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Vectorize.h"

#define LLVM_VERSION_GE(major, minor)                                          \
  (LLVM_VERSION_MAJOR > (major) ||                                             \
   LLVM_VERSION_MAJOR == (major) && LLVM_VERSION_MINOR >= (minor))

#define LLVM_VERSION_EQ(major, minor)                                          \
  (LLVM_VERSION_MAJOR == (major) && LLVM_VERSION_MINOR == (minor))

#define LLVM_VERSION_LE(major, minor)                                          \
  (LLVM_VERSION_MAJOR < (major) ||                                             \
   LLVM_VERSION_MAJOR == (major) && LLVM_VERSION_MINOR <= (minor))

#define LLVM_VERSION_LT(major, minor) (!LLVM_VERSION_GE((major), (minor)))

#if LLVM_VERSION_GE(3, 7)
#include "llvm/IR/LegacyPassManager.h"
#else
#include "llvm/PassManager.h"
#endif

#if LLVM_VERSION_GE(4, 0)
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#else
#include "llvm/Bitcode/ReaderWriter.h"
#endif

enum class LLVMRustResult { Success, Failure };

typedef llvm::object::Archive::Child *LLVMRustArchiveChildRef;
typedef llvm::object::Archive::Child const *LLVMRustArchiveChildConstRef;
typedef llvm::object::OwningBinary<llvm::object::Archive> *LLVMRustArchiveRef;
typedef struct RustArchiveIterator *LLVMRustArchiveIteratorRef;
typedef struct RustArchiveMember *LLVMRustArchiveMemberRef;

enum class LLVMRustArchiveKind {
  Other,
  GNU,
  MIPS64,
  BSD,
  COFF,
};

extern "C" LLVMRustArchiveRef LLVMRustOpenArchive(const char *Path);
extern "C" void LLVMRustDestroyArchive(LLVMRustArchiveRef RustArchive);
extern "C" LLVMRustArchiveIteratorRef
LLVMRustArchiveIteratorNew(LLVMRustArchiveRef RustArchive);
extern "C" LLVMRustArchiveChildConstRef
LLVMRustArchiveIteratorNext(LLVMRustArchiveIteratorRef RAI);
extern "C" void LLVMRustArchiveChildFree(LLVMRustArchiveChildRef Child);
extern "C" void LLVMRustArchiveIteratorFree(LLVMRustArchiveIteratorRef RAI);
extern "C" const char *
LLVMRustArchiveChildName(LLVMRustArchiveChildConstRef Child, size_t *Size);
extern "C" const char *LLVMRustArchiveChildData(LLVMRustArchiveChildRef Child,
                                                size_t *Size);
extern "C" LLVMRustArchiveMemberRef
LLVMRustArchiveMemberNew(char *Filename, char *Name,
                         LLVMRustArchiveChildRef Child);
extern "C" void LLVMRustArchiveMemberFree(LLVMRustArchiveMemberRef Member);
extern "C" LLVMRustResult
LLVMRustWriteArchive(char *Dst, size_t NumMembers,
                     const LLVMRustArchiveMemberRef *NewMembers,
                     bool WriteSymbtab, LLVMRustArchiveKind RustKind);

typedef struct LLVMOpaqueDebugLoc *LLVMDebugLocRef;
typedef struct LLVMOpaqueRustJITMemoryManager *LLVMRustJITMemoryManagerRef;
typedef struct LLVMOpaqueSMDiagnostic *LLVMSMDiagnosticRef;
typedef struct LLVMOpaqueTwine *LLVMTwineRef;
typedef struct OpaqueRustString *RustStringRef;

void LLVMRustSetLastError(const char *);

/// Semantically a subset of the C++ enum llvm::Attribute::AttrKind,
/// though it is not ABI compatible (since it's a C++ enum)
enum LLVMRustAttribute {
  AlwaysInline = 0,
  ByVal = 1,
  Cold = 2,
  InlineHint = 3,
  MinSize = 4,
  Naked = 5,
  NoAlias = 6,
  NoCapture = 7,
  NoInline = 8,
  NonNull = 9,
  NoRedZone = 10,
  NoReturn = 11,
  NoUnwind = 12,
  OptimizeForSize = 13,
  ReadOnly = 14,
  SExt = 15,
  StructRet = 16,
  UWTable = 17,
  ZExt = 18,
  InReg = 19,
  SanitizeThread = 20,
  SanitizeAddress = 21,
  SanitizeMemory = 22,
};

enum class LLVMRustDiagnosticKind {
  Other,
  InlineAsm,
  StackSize,
  DebugMetadataVersion,
  SampleProfile,
  OptimizationRemark,
  OptimizationRemarkMissed,
  OptimizationRemarkAnalysis,
  OptimizationRemarkAnalysisFPCommute,
  OptimizationRemarkAnalysisAliasing,
  OptimizationRemarkOther,
  OptimizationFailure,
};

extern "C" char *LLVMRustGetLastError(void);

extern "C" void LLVMRustAddCallSiteAttribute(LLVMValueRef Instr, unsigned Index,
                                             LLVMRustAttribute RustAttr);
extern "C" void LLVMRustAddFunctionAttribute(LLVMValueRef Fn, unsigned Index,
                                             LLVMRustAttribute RustAttr);
extern "C" void LLVMRustAddFunctionAttrStringValue(LLVMValueRef Fn,
                                                   unsigned Index,
                                                   const char *Name,
                                                   const char *Value);
extern "C" void LLVMRustRemoveFunctionAttributes(LLVMValueRef Fn,
                                                 unsigned Index,
                                                 LLVMRustAttribute RustAttr);

enum class LLVMRustAsmDialect {
  Other,
  Att,
  Intel,
};

typedef llvm::DIBuilder *LLVMRustDIBuilderRef;

// These values **must** match debuginfo::DIFlags! They also *happen*
// to match LLVM, but that isn't required as we do giant sets of
// matching below. The value shouldn't be directly passed to LLVM.
enum class LLVMRustDIFlags : uint32_t {
  FlagZero = 0,
  FlagPrivate = 1,
  FlagProtected = 2,
  FlagPublic = 3,
  FlagFwdDecl = (1 << 2),
  FlagAppleBlock = (1 << 3),
  FlagBlockByrefStruct = (1 << 4),
  FlagVirtual = (1 << 5),
  FlagArtificial = (1 << 6),
  FlagExplicit = (1 << 7),
  FlagPrototyped = (1 << 8),
  FlagObjcClassComplete = (1 << 9),
  FlagObjectPointer = (1 << 10),
  FlagVector = (1 << 11),
  FlagStaticMember = (1 << 12),
  FlagLValueReference = (1 << 13),
  FlagRValueReference = (1 << 14),
  FlagMainSubprogram      = (1 << 21),
  // Do not add values that are not supported by the minimum LLVM
  // version we support!
};


#if LLVM_VERSION_LT(5, 0)
typedef struct LLVMOpaqueMetadata *LLVMMetadataRef;
#endif

extern "C" void LLVMRustWriteTwineToString(LLVMTwineRef T, RustStringRef Str);

extern "C" void LLVMRustUnpackOptimizationDiagnostic(
    LLVMDiagnosticInfoRef DI, RustStringRef PassNameOut,
    LLVMValueRef *FunctionOut, unsigned* Line, unsigned* Column,
    RustStringRef FilenameOut, RustStringRef MessageOut);
extern "C" void
LLVMRustUnpackInlineAsmDiagnostic(LLVMDiagnosticInfoRef DI, unsigned *CookieOut,
                                  LLVMTwineRef *MessageOut,
                                  LLVMValueRef *InstructionOut);
extern "C" LLVMRustDiagnosticKind
LLVMRustGetDiagInfoKind(LLVMDiagnosticInfoRef DI);

extern "C" void LLVMRustWriteDebugLocToString(LLVMContextRef C,
                                              LLVMDebugLocRef DL,
                                              RustStringRef Str);

enum class LLVMRustFileType {
  Other,
  AssemblyFile,
  ObjectFile,
};

enum class LLVMRustSynchronizationScope {
  Other,
  SingleThread,
  CrossThread,
};

extern "C" void LLVMRustSetInlineAsmDiagnosticHandler(
    LLVMContextRef C, llvm::LLVMContext::InlineAsmDiagHandlerTy H, void *CX);

typedef llvm::OperandBundleDef *LLVMRustOperandBundleDefRef;

extern "C" LLVMRustOperandBundleDefRef LLVMRustBuildOperandBundleDef(const char *Name,
                                                           LLVMValueRef *Inputs,
                                                           unsigned NumInputs);
extern "C" void LLVMRustFreeOperandBundleDef(LLVMRustOperandBundleDefRef Bundle);

extern "C" void LLVMRustSetComdat(LLVMModuleRef M, LLVMValueRef V,
                                  const char *Name);
extern "C" void LLVMRustUnsetComdat(LLVMValueRef V);

enum class LLVMRustLinkage {
  ExternalLinkage = 0,
  AvailableExternallyLinkage = 1,
  LinkOnceAnyLinkage = 2,
  LinkOnceODRLinkage = 3,
  WeakAnyLinkage = 4,
  WeakODRLinkage = 5,
  AppendingLinkage = 6,
  InternalLinkage = 7,
  PrivateLinkage = 8,
  ExternalWeakLinkage = 9,
  CommonLinkage = 10,
};

enum class LLVMRustVisibility {
  Default = 0,
  Hidden = 1,
  Protected = 2,
};

enum class LLVMRustPassKind {
  Other,
  Function,
  Module,
};

enum class LLVMRustCodeModel {
  Other,
  Default,
  JITDefault,
  Small,
  Kernel,
  Medium,
  Large,
};

enum class LLVMRustCodeGenOptLevel {
  Other,
  None,
  Less,
  Default,
  Aggressive,
};

extern "C" void LLVMRustStringWriteImpl(RustStringRef Str, const char *Ptr,
                                        size_t Size);

class RawRustStringOstream : public llvm::raw_ostream {
  RustStringRef Str;
  uint64_t Pos;

  void write_impl(const char *Ptr, size_t Size) override {
    LLVMRustStringWriteImpl(Str, Ptr, Size);
    Pos += Size;
  }

  uint64_t current_pos() const override { return Pos; }

public:
  explicit RawRustStringOstream(RustStringRef Str) : Str(Str), Pos(0) {}

  ~RawRustStringOstream() {
    // LLVM requires this.
    flush();
  }
};
