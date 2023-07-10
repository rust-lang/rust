#include "LLVMWrapper.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DiagnosticHandler.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsARM.h"
#include "llvm/IR/LLVMRemarkStreamer.h"
#include "llvm/IR/Mangler.h"
#include "llvm/Remarks/RemarkStreamer.h"
#include "llvm/Remarks/RemarkSerializer.h"
#include "llvm/Remarks/RemarkFormat.h"
#include "llvm/Support/ToolOutputFile.h"
#if LLVM_VERSION_GE(16, 0)
#include "llvm/Support/ModRef.h"
#endif
#include "llvm/Object/Archive.h"
#include "llvm/Object/COFFImportFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Pass.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/Support/Signals.h"
#if LLVM_VERSION_LT(16, 0)
#include "llvm/ADT/Optional.h"
#endif

#include <iostream>

//===----------------------------------------------------------------------===
//
// This file defines alternate interfaces to core functions that are more
// readily callable by Rust's FFI.
//
//===----------------------------------------------------------------------===

using namespace llvm;
using namespace llvm::sys;
using namespace llvm::object;

// LLVMAtomicOrdering is already an enum - don't create another
// one.
static AtomicOrdering fromRust(LLVMAtomicOrdering Ordering) {
  switch (Ordering) {
  case LLVMAtomicOrderingNotAtomic:
    return AtomicOrdering::NotAtomic;
  case LLVMAtomicOrderingUnordered:
    return AtomicOrdering::Unordered;
  case LLVMAtomicOrderingMonotonic:
    return AtomicOrdering::Monotonic;
  case LLVMAtomicOrderingAcquire:
    return AtomicOrdering::Acquire;
  case LLVMAtomicOrderingRelease:
    return AtomicOrdering::Release;
  case LLVMAtomicOrderingAcquireRelease:
    return AtomicOrdering::AcquireRelease;
  case LLVMAtomicOrderingSequentiallyConsistent:
    return AtomicOrdering::SequentiallyConsistent;
  }

  report_fatal_error("Invalid LLVMAtomicOrdering value!");
}

static LLVM_THREAD_LOCAL char *LastError;

// Custom error handler for fatal LLVM errors.
//
// Notably it exits the process with code 101, unlike LLVM's default of 1.
static void FatalErrorHandler(void *UserData,
                              const char* Reason,
                              bool GenCrashDiag) {
  // Do the same thing that the default error handler does.
  std::cerr << "LLVM ERROR: " << Reason << std::endl;

  // Since this error handler exits the process, we have to run any cleanup that
  // LLVM would run after handling the error. This might change with an LLVM
  // upgrade.
  sys::RunInterruptHandlers();

  exit(101);
}

extern "C" void LLVMRustInstallFatalErrorHandler() {
  install_fatal_error_handler(FatalErrorHandler);
}

extern "C" void LLVMRustDisableSystemDialogsOnCrash() {
  sys::DisableSystemDialogsOnCrash();
}

extern "C" char *LLVMRustGetLastError(void) {
  char *Ret = LastError;
  LastError = nullptr;
  return Ret;
}

extern "C" void LLVMRustSetLastError(const char *Err) {
  free((void *)LastError);
  LastError = strdup(Err);
}

extern "C" LLVMContextRef LLVMRustContextCreate(bool shouldDiscardNames) {
  auto ctx = new LLVMContext();
  ctx->setDiscardValueNames(shouldDiscardNames);
  return wrap(ctx);
}

extern "C" void LLVMRustSetNormalizedTarget(LLVMModuleRef M,
                                            const char *Triple) {
  unwrap(M)->setTargetTriple(Triple::normalize(Triple));
}

extern "C" void LLVMRustPrintPassTimings() {
  raw_fd_ostream OS(2, false); // stderr.
  TimerGroup::printAll(OS);
}

extern "C" LLVMValueRef LLVMRustGetNamedValue(LLVMModuleRef M, const char *Name,
                                              size_t NameLen) {
  return wrap(unwrap(M)->getNamedValue(StringRef(Name, NameLen)));
}

enum class LLVMRustTailCallKind {
  None,
  Tail,
  MustTail,
  NoTail,
};

static CallInst::TailCallKind fromRust(LLVMRustTailCallKind Kind) {
  switch (Kind) {
  case LLVMRustTailCallKind::None:
    return CallInst::TailCallKind::TCK_None;
  case LLVMRustTailCallKind::Tail:
    return CallInst::TailCallKind::TCK_Tail;
  case LLVMRustTailCallKind::MustTail:
    return CallInst::TailCallKind::TCK_MustTail;
  case LLVMRustTailCallKind::NoTail:
    return CallInst::TailCallKind::TCK_NoTail;
  default:
    report_fatal_error("bad CallInst::TailCallKind.");
  }
}

extern "C" void LLVMRustSetTailCallKind(LLVMValueRef Call, LLVMRustTailCallKind TCK) {
  unwrap<CallInst>(Call)->setTailCallKind(fromRust(TCK));
}

extern "C" LLVMValueRef LLVMRustGetOrInsertFunction(LLVMModuleRef M,
                                                    const char *Name,
                                                    size_t NameLen,
                                                    LLVMTypeRef FunctionTy) {
  return wrap(unwrap(M)
                  ->getOrInsertFunction(StringRef(Name, NameLen),
                                        unwrap<FunctionType>(FunctionTy))
                  .getCallee()
  );
}

extern "C" LLVMValueRef
LLVMRustGetOrInsertGlobal(LLVMModuleRef M, const char *Name, size_t NameLen, LLVMTypeRef Ty) {
  Module *Mod = unwrap(M);
  StringRef NameRef(Name, NameLen);

  // We don't use Module::getOrInsertGlobal because that returns a Constant*,
  // which may either be the real GlobalVariable*, or a constant bitcast of it
  // if our type doesn't match the original declaration. We always want the
  // GlobalVariable* so we can access linkage, visibility, etc.
  GlobalVariable *GV = Mod->getGlobalVariable(NameRef, true);
  if (!GV)
    GV = new GlobalVariable(*Mod, unwrap(Ty), false,
                            GlobalValue::ExternalLinkage, nullptr, NameRef);
  return wrap(GV);
}

extern "C" LLVMValueRef
LLVMRustInsertPrivateGlobal(LLVMModuleRef M, LLVMTypeRef Ty) {
  return wrap(new GlobalVariable(*unwrap(M),
                                 unwrap(Ty),
                                 false,
                                 GlobalValue::PrivateLinkage,
                                 nullptr));
}

static Attribute::AttrKind fromRust(LLVMRustAttribute Kind) {
  switch (Kind) {
  case AlwaysInline:
    return Attribute::AlwaysInline;
  case ByVal:
    return Attribute::ByVal;
  case Cold:
    return Attribute::Cold;
  case InlineHint:
    return Attribute::InlineHint;
  case MinSize:
    return Attribute::MinSize;
  case Naked:
    return Attribute::Naked;
  case NoAlias:
    return Attribute::NoAlias;
  case NoCapture:
    return Attribute::NoCapture;
  case NoCfCheck:
    return Attribute::NoCfCheck;
  case NoInline:
    return Attribute::NoInline;
  case NonNull:
    return Attribute::NonNull;
  case NoRedZone:
    return Attribute::NoRedZone;
  case NoReturn:
    return Attribute::NoReturn;
  case NoUnwind:
    return Attribute::NoUnwind;
  case OptimizeForSize:
    return Attribute::OptimizeForSize;
  case ReadOnly:
    return Attribute::ReadOnly;
  case SExt:
    return Attribute::SExt;
  case StructRet:
    return Attribute::StructRet;
  case UWTable:
    return Attribute::UWTable;
  case ZExt:
    return Attribute::ZExt;
  case InReg:
    return Attribute::InReg;
  case SanitizeThread:
    return Attribute::SanitizeThread;
  case SanitizeAddress:
    return Attribute::SanitizeAddress;
  case SanitizeMemory:
    return Attribute::SanitizeMemory;
  case NonLazyBind:
    return Attribute::NonLazyBind;
  case OptimizeNone:
    return Attribute::OptimizeNone;
  case ReturnsTwice:
    return Attribute::ReturnsTwice;
  case ReadNone:
    return Attribute::ReadNone;
  case SanitizeHWAddress:
    return Attribute::SanitizeHWAddress;
  case WillReturn:
    return Attribute::WillReturn;
  case StackProtectReq:
    return Attribute::StackProtectReq;
  case StackProtectStrong:
    return Attribute::StackProtectStrong;
  case StackProtect:
    return Attribute::StackProtect;
  case NoUndef:
    return Attribute::NoUndef;
  case SanitizeMemTag:
    return Attribute::SanitizeMemTag;
  case ShadowCallStack:
    return Attribute::ShadowCallStack;
  case AllocSize:
    return Attribute::AllocSize;
#if LLVM_VERSION_GE(15, 0)
  case AllocatedPointer:
    return Attribute::AllocatedPointer;
  case AllocAlign:
    return Attribute::AllocAlign;
#endif
  case SanitizeSafeStack:
    return Attribute::SafeStack;
  }
  report_fatal_error("bad AttributeKind");
}

template<typename T> static inline void AddAttributes(T *t, unsigned Index,
                                                      LLVMAttributeRef *Attrs, size_t AttrsLen) {
  AttributeList PAL = t->getAttributes();
  AttrBuilder B(t->getContext());
  for (LLVMAttributeRef Attr : ArrayRef<LLVMAttributeRef>(Attrs, AttrsLen))
    B.addAttribute(unwrap(Attr));
  AttributeList PALNew = PAL.addAttributesAtIndex(t->getContext(), Index, B);
  t->setAttributes(PALNew);
}

extern "C" void LLVMRustAddFunctionAttributes(LLVMValueRef Fn, unsigned Index,
                                              LLVMAttributeRef *Attrs, size_t AttrsLen) {
  Function *F = unwrap<Function>(Fn);
  AddAttributes(F, Index, Attrs, AttrsLen);
}

extern "C" void LLVMRustAddCallSiteAttributes(LLVMValueRef Instr, unsigned Index,
                                              LLVMAttributeRef *Attrs, size_t AttrsLen) {
  CallBase *Call = unwrap<CallBase>(Instr);
  AddAttributes(Call, Index, Attrs, AttrsLen);
}

extern "C" LLVMAttributeRef LLVMRustCreateAttrNoValue(LLVMContextRef C,
                                                      LLVMRustAttribute RustAttr) {
  return wrap(Attribute::get(*unwrap(C), fromRust(RustAttr)));
}

extern "C" LLVMAttributeRef LLVMRustCreateAlignmentAttr(LLVMContextRef C,
                                                        uint64_t Bytes) {
  return wrap(Attribute::getWithAlignment(*unwrap(C), llvm::Align(Bytes)));
}

extern "C" LLVMAttributeRef LLVMRustCreateDereferenceableAttr(LLVMContextRef C,
                                                              uint64_t Bytes) {
  return wrap(Attribute::getWithDereferenceableBytes(*unwrap(C), Bytes));
}

extern "C" LLVMAttributeRef LLVMRustCreateDereferenceableOrNullAttr(LLVMContextRef C,
                                                                    uint64_t Bytes) {
  return wrap(Attribute::getWithDereferenceableOrNullBytes(*unwrap(C), Bytes));
}

extern "C" LLVMAttributeRef LLVMRustCreateByValAttr(LLVMContextRef C, LLVMTypeRef Ty) {
  return wrap(Attribute::getWithByValType(*unwrap(C), unwrap(Ty)));
}

extern "C" LLVMAttributeRef LLVMRustCreateStructRetAttr(LLVMContextRef C, LLVMTypeRef Ty) {
  return wrap(Attribute::getWithStructRetType(*unwrap(C), unwrap(Ty)));
}

extern "C" LLVMAttributeRef LLVMRustCreateElementTypeAttr(LLVMContextRef C, LLVMTypeRef Ty) {
#if LLVM_VERSION_GE(15, 0)
  return wrap(Attribute::get(*unwrap(C), Attribute::ElementType, unwrap(Ty)));
#else
  report_fatal_error("Should not be needed on LLVM < 15");
#endif
}

extern "C" LLVMAttributeRef LLVMRustCreateUWTableAttr(LLVMContextRef C, bool Async) {
#if LLVM_VERSION_LT(15, 0)
  return wrap(Attribute::get(*unwrap(C), Attribute::UWTable));
#else
  return wrap(Attribute::getWithUWTableKind(
      *unwrap(C), Async ? UWTableKind::Async : UWTableKind::Sync));
#endif
}

extern "C" LLVMAttributeRef LLVMRustCreateAllocSizeAttr(LLVMContextRef C, uint32_t ElementSizeArg) {
  return wrap(Attribute::getWithAllocSizeArgs(*unwrap(C), ElementSizeArg,
#if LLVM_VERSION_LT(16, 0)
                                              None
#else
                                              std::nullopt
#endif
                                              ));
}

#if LLVM_VERSION_GE(15, 0)

// These values **must** match ffi::AllocKindFlags.
// It _happens_ to match the LLVM values of llvm::AllocFnKind,
// but that's happenstance and we do explicit conversions before
// passing them to LLVM.
enum class LLVMRustAllocKindFlags : uint64_t {
  Unknown = 0,
  Alloc = 1,
  Realloc = 1 << 1,
  Free = 1 << 2,
  Uninitialized = 1 << 3,
  Zeroed = 1 << 4,
  Aligned = 1 << 5,
};

static LLVMRustAllocKindFlags operator&(LLVMRustAllocKindFlags A, LLVMRustAllocKindFlags B) {
  return static_cast<LLVMRustAllocKindFlags>(static_cast<uint64_t>(A) &
                                      static_cast<uint64_t>(B));
}

static bool isSet(LLVMRustAllocKindFlags F) { return F != LLVMRustAllocKindFlags::Unknown; }

static llvm::AllocFnKind allocKindFromRust(LLVMRustAllocKindFlags F) {
  llvm::AllocFnKind AFK = llvm::AllocFnKind::Unknown;
  if (isSet(F & LLVMRustAllocKindFlags::Alloc)) {
    AFK |= llvm::AllocFnKind::Alloc;
  }
  if (isSet(F & LLVMRustAllocKindFlags::Realloc)) {
    AFK |= llvm::AllocFnKind::Realloc;
  }
  if (isSet(F & LLVMRustAllocKindFlags::Free)) {
    AFK |= llvm::AllocFnKind::Free;
  }
  if (isSet(F & LLVMRustAllocKindFlags::Uninitialized)) {
    AFK |= llvm::AllocFnKind::Uninitialized;
  }
  if (isSet(F & LLVMRustAllocKindFlags::Zeroed)) {
    AFK |= llvm::AllocFnKind::Zeroed;
  }
  if (isSet(F & LLVMRustAllocKindFlags::Aligned)) {
    AFK |= llvm::AllocFnKind::Aligned;
  }
  return AFK;
}
#endif

extern "C" LLVMAttributeRef LLVMRustCreateAllocKindAttr(LLVMContextRef C, uint64_t AllocKindArg) {
#if LLVM_VERSION_GE(15, 0)
  return wrap(Attribute::get(*unwrap(C), Attribute::AllocKind,
      static_cast<uint64_t>(allocKindFromRust(static_cast<LLVMRustAllocKindFlags>(AllocKindArg)))));
#else
  report_fatal_error(
      "allockind attributes are new in LLVM 15 and should not be used on older LLVMs");
#endif
}

// Simplified representation of `MemoryEffects` across the FFI boundary.
//
// Each variant corresponds to one of the static factory methods on `MemoryEffects`.
enum class LLVMRustMemoryEffects {
  None,
  ReadOnly,
  InaccessibleMemOnly,
};

extern "C" LLVMAttributeRef LLVMRustCreateMemoryEffectsAttr(LLVMContextRef C,
                                                            LLVMRustMemoryEffects Effects) {
#if LLVM_VERSION_GE(16, 0)
  switch (Effects) {
    case LLVMRustMemoryEffects::None:
      return wrap(Attribute::getWithMemoryEffects(*unwrap(C), MemoryEffects::none()));
    case LLVMRustMemoryEffects::ReadOnly:
      return wrap(Attribute::getWithMemoryEffects(*unwrap(C), MemoryEffects::readOnly()));
    case LLVMRustMemoryEffects::InaccessibleMemOnly:
      return wrap(Attribute::getWithMemoryEffects(*unwrap(C),
                                                  MemoryEffects::inaccessibleMemOnly()));
    default:
      report_fatal_error("bad MemoryEffects.");
  }
#else
  switch (Effects) {
    case LLVMRustMemoryEffects::None:
      return wrap(Attribute::get(*unwrap(C), Attribute::ReadNone));
    case LLVMRustMemoryEffects::ReadOnly:
      return wrap(Attribute::get(*unwrap(C), Attribute::ReadOnly));
    case LLVMRustMemoryEffects::InaccessibleMemOnly:
      return wrap(Attribute::get(*unwrap(C), Attribute::InaccessibleMemOnly));
    default:
      report_fatal_error("bad MemoryEffects.");
  }
#endif
}

// Enable a fast-math flag
//
// https://llvm.org/docs/LangRef.html#fast-math-flags
extern "C" void LLVMRustSetFastMath(LLVMValueRef V) {
  if (auto I = dyn_cast<Instruction>(unwrap<Value>(V))) {
    I->setFast(true);
  }
}

extern "C" LLVMValueRef
LLVMRustBuildAtomicLoad(LLVMBuilderRef B, LLVMTypeRef Ty, LLVMValueRef Source,
                        const char *Name, LLVMAtomicOrdering Order) {
  Value *Ptr = unwrap(Source);
  LoadInst *LI = unwrap(B)->CreateLoad(unwrap(Ty), Ptr, Name);
  LI->setAtomic(fromRust(Order));
  return wrap(LI);
}

extern "C" LLVMValueRef LLVMRustBuildAtomicStore(LLVMBuilderRef B,
                                                 LLVMValueRef V,
                                                 LLVMValueRef Target,
                                                 LLVMAtomicOrdering Order) {
  StoreInst *SI = unwrap(B)->CreateStore(unwrap(V), unwrap(Target));
  SI->setAtomic(fromRust(Order));
  return wrap(SI);
}

enum class LLVMRustAsmDialect {
  Att,
  Intel,
};

static InlineAsm::AsmDialect fromRust(LLVMRustAsmDialect Dialect) {
  switch (Dialect) {
  case LLVMRustAsmDialect::Att:
    return InlineAsm::AD_ATT;
  case LLVMRustAsmDialect::Intel:
    return InlineAsm::AD_Intel;
  default:
    report_fatal_error("bad AsmDialect.");
  }
}

extern "C" LLVMValueRef
LLVMRustInlineAsm(LLVMTypeRef Ty, char *AsmString, size_t AsmStringLen,
                  char *Constraints, size_t ConstraintsLen,
                  LLVMBool HasSideEffects, LLVMBool IsAlignStack,
                  LLVMRustAsmDialect Dialect, LLVMBool CanThrow) {
  return wrap(InlineAsm::get(unwrap<FunctionType>(Ty),
                             StringRef(AsmString, AsmStringLen),
                             StringRef(Constraints, ConstraintsLen),
                             HasSideEffects, IsAlignStack,
                             fromRust(Dialect), CanThrow));
}

extern "C" bool LLVMRustInlineAsmVerify(LLVMTypeRef Ty, char *Constraints,
                                        size_t ConstraintsLen) {
#if LLVM_VERSION_LT(15, 0)
  return InlineAsm::Verify(unwrap<FunctionType>(Ty),
                           StringRef(Constraints, ConstraintsLen));
#else
  // llvm::Error converts to true if it is an error.
  return !llvm::errorToBool(InlineAsm::verify(
      unwrap<FunctionType>(Ty), StringRef(Constraints, ConstraintsLen)));
#endif
}

typedef DIBuilder *LLVMRustDIBuilderRef;

template <typename DIT> DIT *unwrapDIPtr(LLVMMetadataRef Ref) {
  return (DIT *)(Ref ? unwrap<MDNode>(Ref) : nullptr);
}

#define DIDescriptor DIScope
#define DIArray DINodeArray
#define unwrapDI unwrapDIPtr

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
  FlagExternalTypeRef = (1 << 15),
  FlagIntroducedVirtual = (1 << 18),
  FlagBitField = (1 << 19),
  FlagNoReturn = (1 << 20),
  // Do not add values that are not supported by the minimum LLVM
  // version we support! see llvm/include/llvm/IR/DebugInfoFlags.def
};

inline LLVMRustDIFlags operator&(LLVMRustDIFlags A, LLVMRustDIFlags B) {
  return static_cast<LLVMRustDIFlags>(static_cast<uint32_t>(A) &
                                      static_cast<uint32_t>(B));
}

inline LLVMRustDIFlags operator|(LLVMRustDIFlags A, LLVMRustDIFlags B) {
  return static_cast<LLVMRustDIFlags>(static_cast<uint32_t>(A) |
                                      static_cast<uint32_t>(B));
}

inline LLVMRustDIFlags &operator|=(LLVMRustDIFlags &A, LLVMRustDIFlags B) {
  return A = A | B;
}

inline bool isSet(LLVMRustDIFlags F) { return F != LLVMRustDIFlags::FlagZero; }

inline LLVMRustDIFlags visibility(LLVMRustDIFlags F) {
  return static_cast<LLVMRustDIFlags>(static_cast<uint32_t>(F) & 0x3);
}

static DINode::DIFlags fromRust(LLVMRustDIFlags Flags) {
  DINode::DIFlags Result = DINode::DIFlags::FlagZero;

  switch (visibility(Flags)) {
  case LLVMRustDIFlags::FlagPrivate:
    Result |= DINode::DIFlags::FlagPrivate;
    break;
  case LLVMRustDIFlags::FlagProtected:
    Result |= DINode::DIFlags::FlagProtected;
    break;
  case LLVMRustDIFlags::FlagPublic:
    Result |= DINode::DIFlags::FlagPublic;
    break;
  default:
    // The rest are handled below
    break;
  }

  if (isSet(Flags & LLVMRustDIFlags::FlagFwdDecl)) {
    Result |= DINode::DIFlags::FlagFwdDecl;
  }
  if (isSet(Flags & LLVMRustDIFlags::FlagAppleBlock)) {
    Result |= DINode::DIFlags::FlagAppleBlock;
  }
  if (isSet(Flags & LLVMRustDIFlags::FlagVirtual)) {
    Result |= DINode::DIFlags::FlagVirtual;
  }
  if (isSet(Flags & LLVMRustDIFlags::FlagArtificial)) {
    Result |= DINode::DIFlags::FlagArtificial;
  }
  if (isSet(Flags & LLVMRustDIFlags::FlagExplicit)) {
    Result |= DINode::DIFlags::FlagExplicit;
  }
  if (isSet(Flags & LLVMRustDIFlags::FlagPrototyped)) {
    Result |= DINode::DIFlags::FlagPrototyped;
  }
  if (isSet(Flags & LLVMRustDIFlags::FlagObjcClassComplete)) {
    Result |= DINode::DIFlags::FlagObjcClassComplete;
  }
  if (isSet(Flags & LLVMRustDIFlags::FlagObjectPointer)) {
    Result |= DINode::DIFlags::FlagObjectPointer;
  }
  if (isSet(Flags & LLVMRustDIFlags::FlagVector)) {
    Result |= DINode::DIFlags::FlagVector;
  }
  if (isSet(Flags & LLVMRustDIFlags::FlagStaticMember)) {
    Result |= DINode::DIFlags::FlagStaticMember;
  }
  if (isSet(Flags & LLVMRustDIFlags::FlagLValueReference)) {
    Result |= DINode::DIFlags::FlagLValueReference;
  }
  if (isSet(Flags & LLVMRustDIFlags::FlagRValueReference)) {
    Result |= DINode::DIFlags::FlagRValueReference;
  }
  if (isSet(Flags & LLVMRustDIFlags::FlagIntroducedVirtual)) {
    Result |= DINode::DIFlags::FlagIntroducedVirtual;
  }
  if (isSet(Flags & LLVMRustDIFlags::FlagBitField)) {
    Result |= DINode::DIFlags::FlagBitField;
  }
  if (isSet(Flags & LLVMRustDIFlags::FlagNoReturn)) {
    Result |= DINode::DIFlags::FlagNoReturn;
  }

  return Result;
}

// These values **must** match debuginfo::DISPFlags! They also *happen*
// to match LLVM, but that isn't required as we do giant sets of
// matching below. The value shouldn't be directly passed to LLVM.
enum class LLVMRustDISPFlags : uint32_t {
  SPFlagZero = 0,
  SPFlagVirtual = 1,
  SPFlagPureVirtual = 2,
  SPFlagLocalToUnit = (1 << 2),
  SPFlagDefinition = (1 << 3),
  SPFlagOptimized = (1 << 4),
  SPFlagMainSubprogram = (1 << 5),
  // Do not add values that are not supported by the minimum LLVM
  // version we support! see llvm/include/llvm/IR/DebugInfoFlags.def
  // (In LLVM < 8, createFunction supported these as separate bool arguments.)
};

inline LLVMRustDISPFlags operator&(LLVMRustDISPFlags A, LLVMRustDISPFlags B) {
  return static_cast<LLVMRustDISPFlags>(static_cast<uint32_t>(A) &
                                      static_cast<uint32_t>(B));
}

inline LLVMRustDISPFlags operator|(LLVMRustDISPFlags A, LLVMRustDISPFlags B) {
  return static_cast<LLVMRustDISPFlags>(static_cast<uint32_t>(A) |
                                      static_cast<uint32_t>(B));
}

inline LLVMRustDISPFlags &operator|=(LLVMRustDISPFlags &A, LLVMRustDISPFlags B) {
  return A = A | B;
}

inline bool isSet(LLVMRustDISPFlags F) { return F != LLVMRustDISPFlags::SPFlagZero; }

inline LLVMRustDISPFlags virtuality(LLVMRustDISPFlags F) {
  return static_cast<LLVMRustDISPFlags>(static_cast<uint32_t>(F) & 0x3);
}

static DISubprogram::DISPFlags fromRust(LLVMRustDISPFlags SPFlags) {
  DISubprogram::DISPFlags Result = DISubprogram::DISPFlags::SPFlagZero;

  switch (virtuality(SPFlags)) {
  case LLVMRustDISPFlags::SPFlagVirtual:
    Result |= DISubprogram::DISPFlags::SPFlagVirtual;
    break;
  case LLVMRustDISPFlags::SPFlagPureVirtual:
    Result |= DISubprogram::DISPFlags::SPFlagPureVirtual;
    break;
  default:
    // The rest are handled below
    break;
  }

  if (isSet(SPFlags & LLVMRustDISPFlags::SPFlagLocalToUnit)) {
    Result |= DISubprogram::DISPFlags::SPFlagLocalToUnit;
  }
  if (isSet(SPFlags & LLVMRustDISPFlags::SPFlagDefinition)) {
    Result |= DISubprogram::DISPFlags::SPFlagDefinition;
  }
  if (isSet(SPFlags & LLVMRustDISPFlags::SPFlagOptimized)) {
    Result |= DISubprogram::DISPFlags::SPFlagOptimized;
  }
  if (isSet(SPFlags & LLVMRustDISPFlags::SPFlagMainSubprogram)) {
    Result |= DISubprogram::DISPFlags::SPFlagMainSubprogram;
  }

  return Result;
}

enum class LLVMRustDebugEmissionKind {
  NoDebug,
  FullDebug,
  LineTablesOnly,
  DebugDirectivesOnly,
};

static DICompileUnit::DebugEmissionKind fromRust(LLVMRustDebugEmissionKind Kind) {
  switch (Kind) {
  case LLVMRustDebugEmissionKind::NoDebug:
    return DICompileUnit::DebugEmissionKind::NoDebug;
  case LLVMRustDebugEmissionKind::FullDebug:
    return DICompileUnit::DebugEmissionKind::FullDebug;
  case LLVMRustDebugEmissionKind::LineTablesOnly:
    return DICompileUnit::DebugEmissionKind::LineTablesOnly;
  case LLVMRustDebugEmissionKind::DebugDirectivesOnly:
    return DICompileUnit::DebugEmissionKind::DebugDirectivesOnly;
  default:
    report_fatal_error("bad DebugEmissionKind.");
  }
}

enum class LLVMRustChecksumKind {
  None,
  MD5,
  SHA1,
  SHA256,
};

#if LLVM_VERSION_LT(16, 0)
static Optional<DIFile::ChecksumKind> fromRust(LLVMRustChecksumKind Kind) {
#else
static std::optional<DIFile::ChecksumKind> fromRust(LLVMRustChecksumKind Kind) {
#endif
  switch (Kind) {
  case LLVMRustChecksumKind::None:
#if LLVM_VERSION_LT(16, 0)
    return None;
#else
    return std::nullopt;
#endif
  case LLVMRustChecksumKind::MD5:
    return DIFile::ChecksumKind::CSK_MD5;
  case LLVMRustChecksumKind::SHA1:
    return DIFile::ChecksumKind::CSK_SHA1;
  case LLVMRustChecksumKind::SHA256:
    return DIFile::ChecksumKind::CSK_SHA256;
  default:
    report_fatal_error("bad ChecksumKind.");
  }
}

extern "C" uint32_t LLVMRustDebugMetadataVersion() {
  return DEBUG_METADATA_VERSION;
}

extern "C" uint32_t LLVMRustVersionPatch() { return LLVM_VERSION_PATCH; }

extern "C" uint32_t LLVMRustVersionMinor() { return LLVM_VERSION_MINOR; }

extern "C" uint32_t LLVMRustVersionMajor() { return LLVM_VERSION_MAJOR; }

extern "C" void LLVMRustAddModuleFlag(
    LLVMModuleRef M,
    Module::ModFlagBehavior MergeBehavior,
    const char *Name,
    uint32_t Value) {
  unwrap(M)->addModuleFlag(MergeBehavior, Name, Value);
}

extern "C" bool LLVMRustHasModuleFlag(LLVMModuleRef M, const char *Name,
                                      size_t Len) {
  return unwrap(M)->getModuleFlag(StringRef(Name, Len)) != nullptr;
}

extern "C" void LLVMRustGlobalAddMetadata(
    LLVMValueRef Global, unsigned Kind, LLVMMetadataRef MD) {
  unwrap<GlobalObject>(Global)->addMetadata(Kind, *unwrap<MDNode>(MD));
}

extern "C" LLVMRustDIBuilderRef LLVMRustDIBuilderCreate(LLVMModuleRef M) {
  return new DIBuilder(*unwrap(M));
}

extern "C" void LLVMRustDIBuilderDispose(LLVMRustDIBuilderRef Builder) {
  delete Builder;
}

extern "C" void LLVMRustDIBuilderFinalize(LLVMRustDIBuilderRef Builder) {
  Builder->finalize();
}

extern "C" LLVMMetadataRef LLVMRustDIBuilderCreateCompileUnit(
    LLVMRustDIBuilderRef Builder, unsigned Lang, LLVMMetadataRef FileRef,
    const char *Producer, size_t ProducerLen, bool isOptimized,
    const char *Flags, unsigned RuntimeVer,
    const char *SplitName, size_t SplitNameLen,
    LLVMRustDebugEmissionKind Kind,
    uint64_t DWOId, bool SplitDebugInlining) {
  auto *File = unwrapDI<DIFile>(FileRef);

  return wrap(Builder->createCompileUnit(Lang, File, StringRef(Producer, ProducerLen),
                                         isOptimized, Flags, RuntimeVer,
                                         StringRef(SplitName, SplitNameLen),
                                         fromRust(Kind), DWOId, SplitDebugInlining));
}

extern "C" LLVMMetadataRef LLVMRustDIBuilderCreateFile(
    LLVMRustDIBuilderRef Builder,
    const char *Filename, size_t FilenameLen,
    const char *Directory, size_t DirectoryLen, LLVMRustChecksumKind CSKind,
    const char *Checksum, size_t ChecksumLen) {

#if LLVM_VERSION_LT(16, 0)
  Optional<DIFile::ChecksumKind> llvmCSKind = fromRust(CSKind);
#else
  std::optional<DIFile::ChecksumKind> llvmCSKind = fromRust(CSKind);
#endif

#if LLVM_VERSION_LT(16, 0)
  Optional<DIFile::ChecksumInfo<StringRef>> CSInfo{};
#else
  std::optional<DIFile::ChecksumInfo<StringRef>> CSInfo{};
#endif
  if (llvmCSKind)
    CSInfo.emplace(*llvmCSKind, StringRef{Checksum, ChecksumLen});
  return wrap(Builder->createFile(StringRef(Filename, FilenameLen),
                                  StringRef(Directory, DirectoryLen),
                                  CSInfo));
}

extern "C" LLVMMetadataRef
LLVMRustDIBuilderCreateSubroutineType(LLVMRustDIBuilderRef Builder,
                                      LLVMMetadataRef ParameterTypes) {
  return wrap(Builder->createSubroutineType(
      DITypeRefArray(unwrap<MDTuple>(ParameterTypes))));
}

extern "C" LLVMMetadataRef LLVMRustDIBuilderCreateFunction(
    LLVMRustDIBuilderRef Builder, LLVMMetadataRef Scope,
    const char *Name, size_t NameLen,
    const char *LinkageName, size_t LinkageNameLen,
    LLVMMetadataRef File, unsigned LineNo,
    LLVMMetadataRef Ty, unsigned ScopeLine, LLVMRustDIFlags Flags,
    LLVMRustDISPFlags SPFlags, LLVMValueRef MaybeFn, LLVMMetadataRef TParam,
    LLVMMetadataRef Decl) {
  DITemplateParameterArray TParams =
      DITemplateParameterArray(unwrap<MDTuple>(TParam));
  DISubprogram::DISPFlags llvmSPFlags = fromRust(SPFlags);
  DINode::DIFlags llvmFlags = fromRust(Flags);
  DISubprogram *Sub = Builder->createFunction(
      unwrapDI<DIScope>(Scope),
      StringRef(Name, NameLen),
      StringRef(LinkageName, LinkageNameLen),
      unwrapDI<DIFile>(File), LineNo,
      unwrapDI<DISubroutineType>(Ty), ScopeLine, llvmFlags,
      llvmSPFlags, TParams, unwrapDIPtr<DISubprogram>(Decl));
  if (MaybeFn)
    unwrap<Function>(MaybeFn)->setSubprogram(Sub);
  return wrap(Sub);
}

extern "C" LLVMMetadataRef LLVMRustDIBuilderCreateMethod(
    LLVMRustDIBuilderRef Builder, LLVMMetadataRef Scope,
    const char *Name, size_t NameLen,
    const char *LinkageName, size_t LinkageNameLen,
    LLVMMetadataRef File, unsigned LineNo,
    LLVMMetadataRef Ty, LLVMRustDIFlags Flags,
    LLVMRustDISPFlags SPFlags, LLVMMetadataRef TParam) {
  DITemplateParameterArray TParams =
      DITemplateParameterArray(unwrap<MDTuple>(TParam));
  DISubprogram::DISPFlags llvmSPFlags = fromRust(SPFlags);
  DINode::DIFlags llvmFlags = fromRust(Flags);
  DISubprogram *Sub = Builder->createMethod(
      unwrapDI<DIScope>(Scope),
      StringRef(Name, NameLen),
      StringRef(LinkageName, LinkageNameLen),
      unwrapDI<DIFile>(File), LineNo,
      unwrapDI<DISubroutineType>(Ty),
      0, 0, nullptr, // VTable params aren't used
      llvmFlags, llvmSPFlags, TParams);
  return wrap(Sub);
}

extern "C" LLVMMetadataRef LLVMRustDIBuilderCreateBasicType(
    LLVMRustDIBuilderRef Builder, const char *Name, size_t NameLen,
    uint64_t SizeInBits, unsigned Encoding) {
  return wrap(Builder->createBasicType(StringRef(Name, NameLen), SizeInBits, Encoding));
}

extern "C" LLVMMetadataRef LLVMRustDIBuilderCreateTypedef(
    LLVMRustDIBuilderRef Builder, LLVMMetadataRef Type, const char *Name, size_t NameLen,
    LLVMMetadataRef File, unsigned LineNo, LLVMMetadataRef Scope) {
  return wrap(Builder->createTypedef(
    unwrap<DIType>(Type), StringRef(Name, NameLen), unwrap<DIFile>(File),
    LineNo, unwrapDIPtr<DIScope>(Scope)));
}

extern "C" LLVMMetadataRef LLVMRustDIBuilderCreatePointerType(
    LLVMRustDIBuilderRef Builder, LLVMMetadataRef PointeeTy,
    uint64_t SizeInBits, uint32_t AlignInBits, unsigned AddressSpace,
    const char *Name, size_t NameLen) {
  return wrap(Builder->createPointerType(unwrapDI<DIType>(PointeeTy),
                                         SizeInBits, AlignInBits,
                                         AddressSpace,
                                         StringRef(Name, NameLen)));
}

extern "C" LLVMMetadataRef LLVMRustDIBuilderCreateStructType(
    LLVMRustDIBuilderRef Builder, LLVMMetadataRef Scope,
    const char *Name, size_t NameLen,
    LLVMMetadataRef File, unsigned LineNumber, uint64_t SizeInBits,
    uint32_t AlignInBits, LLVMRustDIFlags Flags,
    LLVMMetadataRef DerivedFrom, LLVMMetadataRef Elements,
    unsigned RunTimeLang, LLVMMetadataRef VTableHolder,
    const char *UniqueId, size_t UniqueIdLen) {
  return wrap(Builder->createStructType(
      unwrapDI<DIDescriptor>(Scope), StringRef(Name, NameLen),
      unwrapDI<DIFile>(File), LineNumber,
      SizeInBits, AlignInBits, fromRust(Flags), unwrapDI<DIType>(DerivedFrom),
      DINodeArray(unwrapDI<MDTuple>(Elements)), RunTimeLang,
      unwrapDI<DIType>(VTableHolder), StringRef(UniqueId, UniqueIdLen)));
}

extern "C" LLVMMetadataRef LLVMRustDIBuilderCreateVariantPart(
    LLVMRustDIBuilderRef Builder, LLVMMetadataRef Scope,
    const char *Name, size_t NameLen,
    LLVMMetadataRef File, unsigned LineNumber, uint64_t SizeInBits,
    uint32_t AlignInBits, LLVMRustDIFlags Flags, LLVMMetadataRef Discriminator,
    LLVMMetadataRef Elements, const char *UniqueId, size_t UniqueIdLen) {
  return wrap(Builder->createVariantPart(
      unwrapDI<DIDescriptor>(Scope), StringRef(Name, NameLen),
      unwrapDI<DIFile>(File), LineNumber,
      SizeInBits, AlignInBits, fromRust(Flags), unwrapDI<DIDerivedType>(Discriminator),
      DINodeArray(unwrapDI<MDTuple>(Elements)), StringRef(UniqueId, UniqueIdLen)));
}

extern "C" LLVMMetadataRef LLVMRustDIBuilderCreateMemberType(
    LLVMRustDIBuilderRef Builder, LLVMMetadataRef Scope,
    const char *Name, size_t NameLen,
    LLVMMetadataRef File, unsigned LineNo, uint64_t SizeInBits,
    uint32_t AlignInBits, uint64_t OffsetInBits, LLVMRustDIFlags Flags,
    LLVMMetadataRef Ty) {
  return wrap(Builder->createMemberType(unwrapDI<DIDescriptor>(Scope),
                                        StringRef(Name, NameLen),
                                        unwrapDI<DIFile>(File), LineNo,
                                        SizeInBits, AlignInBits, OffsetInBits,
                                        fromRust(Flags), unwrapDI<DIType>(Ty)));
}

extern "C" LLVMMetadataRef LLVMRustDIBuilderCreateVariantMemberType(
    LLVMRustDIBuilderRef Builder, LLVMMetadataRef Scope,
    const char *Name, size_t NameLen, LLVMMetadataRef File, unsigned LineNo,
    uint64_t SizeInBits, uint32_t AlignInBits, uint64_t OffsetInBits, LLVMValueRef Discriminant,
    LLVMRustDIFlags Flags, LLVMMetadataRef Ty) {
  llvm::ConstantInt* D = nullptr;
  if (Discriminant) {
    D = unwrap<llvm::ConstantInt>(Discriminant);
  }
  return wrap(Builder->createVariantMemberType(unwrapDI<DIDescriptor>(Scope),
                                               StringRef(Name, NameLen),
                                               unwrapDI<DIFile>(File), LineNo,
                                               SizeInBits, AlignInBits, OffsetInBits, D,
                                               fromRust(Flags), unwrapDI<DIType>(Ty)));
}

extern "C" LLVMMetadataRef LLVMRustDIBuilderCreateStaticMemberType(
    LLVMRustDIBuilderRef Builder,
    LLVMMetadataRef Scope,
    const char *Name,
    size_t NameLen,
    LLVMMetadataRef File,
    unsigned LineNo,
    LLVMMetadataRef Ty,
    LLVMRustDIFlags Flags,
    LLVMValueRef val,
    uint32_t AlignInBits
) {
  return wrap(Builder->createStaticMemberType(
    unwrapDI<DIDescriptor>(Scope),
    StringRef(Name, NameLen),
    unwrapDI<DIFile>(File),
    LineNo,
    unwrapDI<DIType>(Ty),
    fromRust(Flags),
    unwrap<llvm::ConstantInt>(val),
    AlignInBits
  ));
}

extern "C" LLVMMetadataRef LLVMRustDIBuilderCreateLexicalBlock(
    LLVMRustDIBuilderRef Builder, LLVMMetadataRef Scope,
    LLVMMetadataRef File, unsigned Line, unsigned Col) {
  return wrap(Builder->createLexicalBlock(unwrapDI<DIDescriptor>(Scope),
                                          unwrapDI<DIFile>(File), Line, Col));
}

extern "C" LLVMMetadataRef
LLVMRustDIBuilderCreateLexicalBlockFile(LLVMRustDIBuilderRef Builder,
                                        LLVMMetadataRef Scope,
                                        LLVMMetadataRef File) {
  return wrap(Builder->createLexicalBlockFile(unwrapDI<DIDescriptor>(Scope),
                                              unwrapDI<DIFile>(File)));
}

extern "C" LLVMMetadataRef LLVMRustDIBuilderCreateStaticVariable(
    LLVMRustDIBuilderRef Builder, LLVMMetadataRef Context,
    const char *Name, size_t NameLen,
    const char *LinkageName, size_t LinkageNameLen,
    LLVMMetadataRef File, unsigned LineNo,
    LLVMMetadataRef Ty, bool IsLocalToUnit, LLVMValueRef V,
    LLVMMetadataRef Decl = nullptr, uint32_t AlignInBits = 0) {
  llvm::GlobalVariable *InitVal = cast<llvm::GlobalVariable>(unwrap(V));

  llvm::DIExpression *InitExpr = nullptr;
  if (llvm::ConstantInt *IntVal = llvm::dyn_cast<llvm::ConstantInt>(InitVal)) {
    InitExpr = Builder->createConstantValueExpression(
        IntVal->getValue().getSExtValue());
  } else if (llvm::ConstantFP *FPVal =
                 llvm::dyn_cast<llvm::ConstantFP>(InitVal)) {
    InitExpr = Builder->createConstantValueExpression(
        FPVal->getValueAPF().bitcastToAPInt().getZExtValue());
  }

  llvm::DIGlobalVariableExpression *VarExpr = Builder->createGlobalVariableExpression(
      unwrapDI<DIDescriptor>(Context), StringRef(Name, NameLen),
      StringRef(LinkageName, LinkageNameLen),
      unwrapDI<DIFile>(File), LineNo, unwrapDI<DIType>(Ty), IsLocalToUnit,
      /* isDefined */ true,
      InitExpr, unwrapDIPtr<MDNode>(Decl),
      /* templateParams */ nullptr,
      AlignInBits);

  InitVal->setMetadata("dbg", VarExpr);

  return wrap(VarExpr);
}

extern "C" LLVMMetadataRef LLVMRustDIBuilderCreateVariable(
    LLVMRustDIBuilderRef Builder, unsigned Tag, LLVMMetadataRef Scope,
    const char *Name, size_t NameLen,
    LLVMMetadataRef File, unsigned LineNo,
    LLVMMetadataRef Ty, bool AlwaysPreserve, LLVMRustDIFlags Flags,
    unsigned ArgNo, uint32_t AlignInBits) {
  if (Tag == 0x100) { // DW_TAG_auto_variable
    return wrap(Builder->createAutoVariable(
        unwrapDI<DIDescriptor>(Scope), StringRef(Name, NameLen),
        unwrapDI<DIFile>(File), LineNo,
        unwrapDI<DIType>(Ty), AlwaysPreserve, fromRust(Flags), AlignInBits));
  } else {
    return wrap(Builder->createParameterVariable(
        unwrapDI<DIDescriptor>(Scope), StringRef(Name, NameLen), ArgNo,
        unwrapDI<DIFile>(File), LineNo,
        unwrapDI<DIType>(Ty), AlwaysPreserve, fromRust(Flags)));
  }
}

extern "C" LLVMMetadataRef
LLVMRustDIBuilderCreateArrayType(LLVMRustDIBuilderRef Builder, uint64_t Size,
                                 uint32_t AlignInBits, LLVMMetadataRef Ty,
                                 LLVMMetadataRef Subscripts) {
  return wrap(
      Builder->createArrayType(Size, AlignInBits, unwrapDI<DIType>(Ty),
                               DINodeArray(unwrapDI<MDTuple>(Subscripts))));
}

extern "C" LLVMMetadataRef
LLVMRustDIBuilderGetOrCreateSubrange(LLVMRustDIBuilderRef Builder, int64_t Lo,
                                     int64_t Count) {
  return wrap(Builder->getOrCreateSubrange(Lo, Count));
}

extern "C" LLVMMetadataRef
LLVMRustDIBuilderGetOrCreateArray(LLVMRustDIBuilderRef Builder,
                                  LLVMMetadataRef *Ptr, unsigned Count) {
  Metadata **DataValue = unwrap(Ptr);
  return wrap(
      Builder->getOrCreateArray(ArrayRef<Metadata *>(DataValue, Count)).get());
}

extern "C" LLVMValueRef LLVMRustDIBuilderInsertDeclareAtEnd(
    LLVMRustDIBuilderRef Builder, LLVMValueRef V, LLVMMetadataRef VarInfo,
    uint64_t *AddrOps, unsigned AddrOpsCount, LLVMMetadataRef DL,
    LLVMBasicBlockRef InsertAtEnd) {
  return wrap(Builder->insertDeclare(
      unwrap(V), unwrap<DILocalVariable>(VarInfo),
      Builder->createExpression(llvm::ArrayRef<uint64_t>(AddrOps, AddrOpsCount)),
      DebugLoc(cast<MDNode>(unwrap(DL))),
      unwrap(InsertAtEnd)));
}

extern "C" LLVMMetadataRef LLVMRustDIBuilderCreateEnumerator(
    LLVMRustDIBuilderRef Builder, const char *Name, size_t NameLen,
    const uint64_t Value[2], unsigned SizeInBits, bool IsUnsigned) {
  return wrap(Builder->createEnumerator(StringRef(Name, NameLen),
      APSInt(APInt(SizeInBits, ArrayRef<uint64_t>(Value, 2)), IsUnsigned)));
}

extern "C" LLVMMetadataRef LLVMRustDIBuilderCreateEnumerationType(
    LLVMRustDIBuilderRef Builder, LLVMMetadataRef Scope,
    const char *Name, size_t NameLen,
    LLVMMetadataRef File, unsigned LineNumber, uint64_t SizeInBits,
    uint32_t AlignInBits, LLVMMetadataRef Elements,
    LLVMMetadataRef ClassTy, bool IsScoped) {
  return wrap(Builder->createEnumerationType(
      unwrapDI<DIDescriptor>(Scope), StringRef(Name, NameLen),
      unwrapDI<DIFile>(File), LineNumber,
      SizeInBits, AlignInBits, DINodeArray(unwrapDI<MDTuple>(Elements)),
      unwrapDI<DIType>(ClassTy), "", IsScoped));
}

extern "C" LLVMMetadataRef LLVMRustDIBuilderCreateUnionType(
    LLVMRustDIBuilderRef Builder, LLVMMetadataRef Scope,
    const char *Name, size_t NameLen,
    LLVMMetadataRef File, unsigned LineNumber, uint64_t SizeInBits,
    uint32_t AlignInBits, LLVMRustDIFlags Flags, LLVMMetadataRef Elements,
    unsigned RunTimeLang, const char *UniqueId, size_t UniqueIdLen) {
  return wrap(Builder->createUnionType(
      unwrapDI<DIDescriptor>(Scope), StringRef(Name, NameLen), unwrapDI<DIFile>(File),
      LineNumber, SizeInBits, AlignInBits, fromRust(Flags),
      DINodeArray(unwrapDI<MDTuple>(Elements)), RunTimeLang,
      StringRef(UniqueId, UniqueIdLen)));
}

extern "C" LLVMMetadataRef LLVMRustDIBuilderCreateTemplateTypeParameter(
    LLVMRustDIBuilderRef Builder, LLVMMetadataRef Scope,
    const char *Name, size_t NameLen, LLVMMetadataRef Ty) {
  bool IsDefault = false; // FIXME: should we ever set this true?
  return wrap(Builder->createTemplateTypeParameter(
      unwrapDI<DIDescriptor>(Scope), StringRef(Name, NameLen), unwrapDI<DIType>(Ty), IsDefault));
}

extern "C" LLVMMetadataRef LLVMRustDIBuilderCreateNameSpace(
    LLVMRustDIBuilderRef Builder, LLVMMetadataRef Scope,
    const char *Name, size_t NameLen, bool ExportSymbols) {
  return wrap(Builder->createNameSpace(
      unwrapDI<DIDescriptor>(Scope), StringRef(Name, NameLen), ExportSymbols
  ));
}

extern "C" void
LLVMRustDICompositeTypeReplaceArrays(LLVMRustDIBuilderRef Builder,
                                     LLVMMetadataRef CompositeTy,
                                     LLVMMetadataRef Elements,
                                     LLVMMetadataRef Params) {
  DICompositeType *Tmp = unwrapDI<DICompositeType>(CompositeTy);
  Builder->replaceArrays(Tmp, DINodeArray(unwrap<MDTuple>(Elements)),
                         DINodeArray(unwrap<MDTuple>(Params)));
}

extern "C" LLVMMetadataRef
LLVMRustDIBuilderCreateDebugLocation(unsigned Line, unsigned Column,
                                     LLVMMetadataRef ScopeRef,
                                     LLVMMetadataRef InlinedAt) {
  MDNode *Scope = unwrapDIPtr<MDNode>(ScopeRef);
  DILocation *Loc = DILocation::get(
      Scope->getContext(), Line, Column, Scope,
      unwrapDIPtr<MDNode>(InlinedAt));
  return wrap(Loc);
}

extern "C" uint64_t LLVMRustDIBuilderCreateOpDeref() {
  return dwarf::DW_OP_deref;
}

extern "C" uint64_t LLVMRustDIBuilderCreateOpPlusUconst() {
  return dwarf::DW_OP_plus_uconst;
}

extern "C" int64_t LLVMRustDIBuilderCreateOpLLVMFragment() {
  return dwarf::DW_OP_LLVM_fragment;
}

extern "C" void LLVMRustWriteTypeToString(LLVMTypeRef Ty, RustStringRef Str) {
  RawRustStringOstream OS(Str);
  unwrap<llvm::Type>(Ty)->print(OS);
}

extern "C" void LLVMRustWriteValueToString(LLVMValueRef V,
                                           RustStringRef Str) {
  RawRustStringOstream OS(Str);
  if (!V) {
    OS << "(null)";
  } else {
    OS << "(";
    unwrap<llvm::Value>(V)->getType()->print(OS);
    OS << ":";
    unwrap<llvm::Value>(V)->print(OS);
    OS << ")";
  }
}

// LLVMArrayType function does not support 64-bit ElementCount
// FIXME: replace with LLVMArrayType2 when bumped minimal version to llvm-17
// https://github.com/llvm/llvm-project/commit/35276f16e5a2cae0dfb49c0fbf874d4d2f177acc
extern "C" LLVMTypeRef LLVMRustArrayType(LLVMTypeRef ElementTy,
                                         uint64_t ElementCount) {
  return wrap(ArrayType::get(unwrap(ElementTy), ElementCount));
}

DEFINE_SIMPLE_CONVERSION_FUNCTIONS(Twine, LLVMTwineRef)

extern "C" void LLVMRustWriteTwineToString(LLVMTwineRef T, RustStringRef Str) {
  RawRustStringOstream OS(Str);
  unwrap(T)->print(OS);
}

extern "C" void LLVMRustUnpackOptimizationDiagnostic(
    LLVMDiagnosticInfoRef DI, RustStringRef PassNameOut,
    LLVMValueRef *FunctionOut, unsigned* Line, unsigned* Column,
    RustStringRef FilenameOut, RustStringRef MessageOut) {
  // Undefined to call this not on an optimization diagnostic!
  llvm::DiagnosticInfoOptimizationBase *Opt =
      static_cast<llvm::DiagnosticInfoOptimizationBase *>(unwrap(DI));

  RawRustStringOstream PassNameOS(PassNameOut);
  PassNameOS << Opt->getPassName();
  *FunctionOut = wrap(&Opt->getFunction());

  RawRustStringOstream FilenameOS(FilenameOut);
  DiagnosticLocation loc = Opt->getLocation();
  if (loc.isValid()) {
    *Line = loc.getLine();
    *Column = loc.getColumn();
    FilenameOS << loc.getAbsolutePath();
  }

  RawRustStringOstream MessageOS(MessageOut);
  MessageOS << Opt->getMsg();
}

enum class LLVMRustDiagnosticLevel {
    Error,
    Warning,
    Note,
    Remark,
};

extern "C" void
LLVMRustUnpackInlineAsmDiagnostic(LLVMDiagnosticInfoRef DI,
                                  LLVMRustDiagnosticLevel *LevelOut,
                                  unsigned *CookieOut,
                                  LLVMTwineRef *MessageOut) {
  // Undefined to call this not on an inline assembly diagnostic!
  llvm::DiagnosticInfoInlineAsm *IA =
      static_cast<llvm::DiagnosticInfoInlineAsm *>(unwrap(DI));

  *CookieOut = IA->getLocCookie();
  *MessageOut = wrap(&IA->getMsgStr());

  switch (IA->getSeverity()) {
    case DS_Error:
      *LevelOut = LLVMRustDiagnosticLevel::Error;
      break;
    case DS_Warning:
      *LevelOut = LLVMRustDiagnosticLevel::Warning;
      break;
    case DS_Note:
      *LevelOut = LLVMRustDiagnosticLevel::Note;
      break;
    case DS_Remark:
      *LevelOut = LLVMRustDiagnosticLevel::Remark;
      break;
    default:
      report_fatal_error("Invalid LLVMRustDiagnosticLevel value!");
  }
}

extern "C" void LLVMRustWriteDiagnosticInfoToString(LLVMDiagnosticInfoRef DI,
                                                    RustStringRef Str) {
  RawRustStringOstream OS(Str);
  DiagnosticPrinterRawOStream DP(OS);
  unwrap(DI)->print(DP);
}

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
  PGOProfile,
  Linker,
  Unsupported,
  SrcMgr,
};

static LLVMRustDiagnosticKind toRust(DiagnosticKind Kind) {
  switch (Kind) {
  case DK_InlineAsm:
    return LLVMRustDiagnosticKind::InlineAsm;
  case DK_StackSize:
    return LLVMRustDiagnosticKind::StackSize;
  case DK_DebugMetadataVersion:
    return LLVMRustDiagnosticKind::DebugMetadataVersion;
  case DK_SampleProfile:
    return LLVMRustDiagnosticKind::SampleProfile;
  case DK_OptimizationRemark:
  case DK_MachineOptimizationRemark:
    return LLVMRustDiagnosticKind::OptimizationRemark;
  case DK_OptimizationRemarkMissed:
  case DK_MachineOptimizationRemarkMissed:
    return LLVMRustDiagnosticKind::OptimizationRemarkMissed;
  case DK_OptimizationRemarkAnalysis:
  case DK_MachineOptimizationRemarkAnalysis:
    return LLVMRustDiagnosticKind::OptimizationRemarkAnalysis;
  case DK_OptimizationRemarkAnalysisFPCommute:
    return LLVMRustDiagnosticKind::OptimizationRemarkAnalysisFPCommute;
  case DK_OptimizationRemarkAnalysisAliasing:
    return LLVMRustDiagnosticKind::OptimizationRemarkAnalysisAliasing;
  case DK_PGOProfile:
    return LLVMRustDiagnosticKind::PGOProfile;
  case DK_Linker:
    return LLVMRustDiagnosticKind::Linker;
  case DK_Unsupported:
    return LLVMRustDiagnosticKind::Unsupported;
  case DK_SrcMgr:
    return LLVMRustDiagnosticKind::SrcMgr;
  default:
    return (Kind >= DK_FirstRemark && Kind <= DK_LastRemark)
               ? LLVMRustDiagnosticKind::OptimizationRemarkOther
               : LLVMRustDiagnosticKind::Other;
  }
}

extern "C" LLVMRustDiagnosticKind
LLVMRustGetDiagInfoKind(LLVMDiagnosticInfoRef DI) {
  return toRust((DiagnosticKind)unwrap(DI)->getKind());
}

// This is kept distinct from LLVMGetTypeKind, because when
// a new type kind is added, the Rust-side enum must be
// updated or UB will result.
extern "C" LLVMTypeKind LLVMRustGetTypeKind(LLVMTypeRef Ty) {
  switch (unwrap(Ty)->getTypeID()) {
  case Type::VoidTyID:
    return LLVMVoidTypeKind;
  case Type::HalfTyID:
    return LLVMHalfTypeKind;
  case Type::FloatTyID:
    return LLVMFloatTypeKind;
  case Type::DoubleTyID:
    return LLVMDoubleTypeKind;
  case Type::X86_FP80TyID:
    return LLVMX86_FP80TypeKind;
  case Type::FP128TyID:
    return LLVMFP128TypeKind;
  case Type::PPC_FP128TyID:
    return LLVMPPC_FP128TypeKind;
  case Type::LabelTyID:
    return LLVMLabelTypeKind;
  case Type::MetadataTyID:
    return LLVMMetadataTypeKind;
  case Type::IntegerTyID:
    return LLVMIntegerTypeKind;
  case Type::FunctionTyID:
    return LLVMFunctionTypeKind;
  case Type::StructTyID:
    return LLVMStructTypeKind;
  case Type::ArrayTyID:
    return LLVMArrayTypeKind;
  case Type::PointerTyID:
    return LLVMPointerTypeKind;
  case Type::FixedVectorTyID:
    return LLVMVectorTypeKind;
  case Type::X86_MMXTyID:
    return LLVMX86_MMXTypeKind;
  case Type::TokenTyID:
    return LLVMTokenTypeKind;
  case Type::ScalableVectorTyID:
    return LLVMScalableVectorTypeKind;
  case Type::BFloatTyID:
    return LLVMBFloatTypeKind;
  case Type::X86_AMXTyID:
    return LLVMX86_AMXTypeKind;
  default:
    {
      std::string error;
      llvm::raw_string_ostream stream(error);
      stream << "Rust does not support the TypeID: " << unwrap(Ty)->getTypeID()
             << " for the type: " << *unwrap(Ty);
      stream.flush();
      report_fatal_error(error.c_str());
    }
  }
}

DEFINE_SIMPLE_CONVERSION_FUNCTIONS(SMDiagnostic, LLVMSMDiagnosticRef)

extern "C" LLVMSMDiagnosticRef LLVMRustGetSMDiagnostic(
    LLVMDiagnosticInfoRef DI, unsigned *Cookie) {
  llvm::DiagnosticInfoSrcMgr *SM = static_cast<llvm::DiagnosticInfoSrcMgr *>(unwrap(DI));
  *Cookie = SM->getLocCookie();
  return wrap(&SM->getSMDiag());
}

extern "C" bool LLVMRustUnpackSMDiagnostic(LLVMSMDiagnosticRef DRef,
                                           RustStringRef MessageOut,
                                           RustStringRef BufferOut,
                                           LLVMRustDiagnosticLevel* LevelOut,
                                           unsigned* LocOut,
                                           unsigned* RangesOut,
                                           size_t* NumRanges) {
  SMDiagnostic& D = *unwrap(DRef);
  RawRustStringOstream MessageOS(MessageOut);
  MessageOS << D.getMessage();

  switch (D.getKind()) {
    case SourceMgr::DK_Error:
      *LevelOut = LLVMRustDiagnosticLevel::Error;
      break;
    case SourceMgr::DK_Warning:
      *LevelOut = LLVMRustDiagnosticLevel::Warning;
      break;
    case SourceMgr::DK_Note:
      *LevelOut = LLVMRustDiagnosticLevel::Note;
      break;
    case SourceMgr::DK_Remark:
      *LevelOut = LLVMRustDiagnosticLevel::Remark;
      break;
    default:
      report_fatal_error("Invalid LLVMRustDiagnosticLevel value!");
  }

  if (D.getLoc() == SMLoc())
    return false;

  const SourceMgr &LSM = *D.getSourceMgr();
  const MemoryBuffer *LBuf = LSM.getMemoryBuffer(LSM.FindBufferContainingLoc(D.getLoc()));
  LLVMRustStringWriteImpl(BufferOut, LBuf->getBufferStart(), LBuf->getBufferSize());

  *LocOut = D.getLoc().getPointer() - LBuf->getBufferStart();

  *NumRanges = std::min(*NumRanges, D.getRanges().size());
  size_t LineStart = *LocOut - (size_t)D.getColumnNo();
  for (size_t i = 0; i < *NumRanges; i++) {
    RangesOut[i * 2] = LineStart + D.getRanges()[i].first;
    RangesOut[i * 2 + 1] = LineStart + D.getRanges()[i].second;
  }

  return true;
}

extern "C" OperandBundleDef *LLVMRustBuildOperandBundleDef(const char *Name,
                                                           LLVMValueRef *Inputs,
                                                           unsigned NumInputs) {
  return new OperandBundleDef(Name, ArrayRef<Value*>(unwrap(Inputs), NumInputs));
}

extern "C" void LLVMRustFreeOperandBundleDef(OperandBundleDef *Bundle) {
  delete Bundle;
}

extern "C" LLVMValueRef LLVMRustBuildCall(LLVMBuilderRef B, LLVMTypeRef Ty, LLVMValueRef Fn,
                                          LLVMValueRef *Args, unsigned NumArgs,
                                          OperandBundleDef **OpBundles,
                                          unsigned NumOpBundles) {
  Value *Callee = unwrap(Fn);
  FunctionType *FTy = unwrap<FunctionType>(Ty);
  return wrap(unwrap(B)->CreateCall(
      FTy, Callee, ArrayRef<Value*>(unwrap(Args), NumArgs),
      ArrayRef<OperandBundleDef>(*OpBundles, NumOpBundles)));
}

extern "C" LLVMValueRef LLVMRustGetInstrProfIncrementIntrinsic(LLVMModuleRef M) {
  return wrap(llvm::Intrinsic::getDeclaration(unwrap(M),
              (llvm::Intrinsic::ID)llvm::Intrinsic::instrprof_increment));
}

extern "C" LLVMValueRef LLVMRustBuildMemCpy(LLVMBuilderRef B,
                                            LLVMValueRef Dst, unsigned DstAlign,
                                            LLVMValueRef Src, unsigned SrcAlign,
                                            LLVMValueRef Size, bool IsVolatile) {
  return wrap(unwrap(B)->CreateMemCpy(
      unwrap(Dst), MaybeAlign(DstAlign),
      unwrap(Src), MaybeAlign(SrcAlign),
      unwrap(Size), IsVolatile));
}

extern "C" LLVMValueRef LLVMRustBuildMemMove(LLVMBuilderRef B,
                                             LLVMValueRef Dst, unsigned DstAlign,
                                             LLVMValueRef Src, unsigned SrcAlign,
                                             LLVMValueRef Size, bool IsVolatile) {
  return wrap(unwrap(B)->CreateMemMove(
      unwrap(Dst), MaybeAlign(DstAlign),
      unwrap(Src), MaybeAlign(SrcAlign),
      unwrap(Size), IsVolatile));
}

extern "C" LLVMValueRef LLVMRustBuildMemSet(LLVMBuilderRef B,
                                            LLVMValueRef Dst, unsigned DstAlign,
                                            LLVMValueRef Val,
                                            LLVMValueRef Size, bool IsVolatile) {
  return wrap(unwrap(B)->CreateMemSet(
      unwrap(Dst), unwrap(Val), unwrap(Size), MaybeAlign(DstAlign), IsVolatile));
}

extern "C" LLVMValueRef
LLVMRustBuildInvoke(LLVMBuilderRef B, LLVMTypeRef Ty, LLVMValueRef Fn,
                    LLVMValueRef *Args, unsigned NumArgs,
                    LLVMBasicBlockRef Then, LLVMBasicBlockRef Catch,
                    OperandBundleDef **OpBundles, unsigned NumOpBundles,
                    const char *Name) {
  Value *Callee = unwrap(Fn);
  FunctionType *FTy = unwrap<FunctionType>(Ty);
  return wrap(unwrap(B)->CreateInvoke(FTy, Callee, unwrap(Then), unwrap(Catch),
                                      ArrayRef<Value*>(unwrap(Args), NumArgs),
                                      ArrayRef<OperandBundleDef>(*OpBundles, NumOpBundles),
                                      Name));
}

extern "C" void LLVMRustPositionBuilderAtStart(LLVMBuilderRef B,
                                               LLVMBasicBlockRef BB) {
  auto Point = unwrap(BB)->getFirstInsertionPt();
  unwrap(B)->SetInsertPoint(unwrap(BB), Point);
}

extern "C" void LLVMRustSetComdat(LLVMModuleRef M, LLVMValueRef V,
                                  const char *Name, size_t NameLen) {
  Triple TargetTriple(unwrap(M)->getTargetTriple());
  GlobalObject *GV = unwrap<GlobalObject>(V);
  if (TargetTriple.supportsCOMDAT()) {
    StringRef NameRef(Name, NameLen);
    GV->setComdat(unwrap(M)->getOrInsertComdat(NameRef));
  }
}

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

static LLVMRustLinkage toRust(LLVMLinkage Linkage) {
  switch (Linkage) {
  case LLVMExternalLinkage:
    return LLVMRustLinkage::ExternalLinkage;
  case LLVMAvailableExternallyLinkage:
    return LLVMRustLinkage::AvailableExternallyLinkage;
  case LLVMLinkOnceAnyLinkage:
    return LLVMRustLinkage::LinkOnceAnyLinkage;
  case LLVMLinkOnceODRLinkage:
    return LLVMRustLinkage::LinkOnceODRLinkage;
  case LLVMWeakAnyLinkage:
    return LLVMRustLinkage::WeakAnyLinkage;
  case LLVMWeakODRLinkage:
    return LLVMRustLinkage::WeakODRLinkage;
  case LLVMAppendingLinkage:
    return LLVMRustLinkage::AppendingLinkage;
  case LLVMInternalLinkage:
    return LLVMRustLinkage::InternalLinkage;
  case LLVMPrivateLinkage:
    return LLVMRustLinkage::PrivateLinkage;
  case LLVMExternalWeakLinkage:
    return LLVMRustLinkage::ExternalWeakLinkage;
  case LLVMCommonLinkage:
    return LLVMRustLinkage::CommonLinkage;
  default:
    report_fatal_error("Invalid LLVMRustLinkage value!");
  }
}

static LLVMLinkage fromRust(LLVMRustLinkage Linkage) {
  switch (Linkage) {
  case LLVMRustLinkage::ExternalLinkage:
    return LLVMExternalLinkage;
  case LLVMRustLinkage::AvailableExternallyLinkage:
    return LLVMAvailableExternallyLinkage;
  case LLVMRustLinkage::LinkOnceAnyLinkage:
    return LLVMLinkOnceAnyLinkage;
  case LLVMRustLinkage::LinkOnceODRLinkage:
    return LLVMLinkOnceODRLinkage;
  case LLVMRustLinkage::WeakAnyLinkage:
    return LLVMWeakAnyLinkage;
  case LLVMRustLinkage::WeakODRLinkage:
    return LLVMWeakODRLinkage;
  case LLVMRustLinkage::AppendingLinkage:
    return LLVMAppendingLinkage;
  case LLVMRustLinkage::InternalLinkage:
    return LLVMInternalLinkage;
  case LLVMRustLinkage::PrivateLinkage:
    return LLVMPrivateLinkage;
  case LLVMRustLinkage::ExternalWeakLinkage:
    return LLVMExternalWeakLinkage;
  case LLVMRustLinkage::CommonLinkage:
    return LLVMCommonLinkage;
  }
  report_fatal_error("Invalid LLVMRustLinkage value!");
}

extern "C" LLVMRustLinkage LLVMRustGetLinkage(LLVMValueRef V) {
  return toRust(LLVMGetLinkage(V));
}

extern "C" void LLVMRustSetLinkage(LLVMValueRef V,
                                   LLVMRustLinkage RustLinkage) {
  LLVMSetLinkage(V, fromRust(RustLinkage));
}

extern "C" bool LLVMRustConstIntGetZExtValue(LLVMValueRef CV, uint64_t *value) {
    auto C = unwrap<llvm::ConstantInt>(CV);
    if (C->getBitWidth() > 64)
      return false;
    *value = C->getZExtValue();
    return true;
}

// Returns true if both high and low were successfully set. Fails in case constant wasn’t any of
// the common sizes (1, 8, 16, 32, 64, 128 bits)
extern "C" bool LLVMRustConstInt128Get(LLVMValueRef CV, bool sext, uint64_t *high, uint64_t *low)
{
    auto C = unwrap<llvm::ConstantInt>(CV);
    if (C->getBitWidth() > 128) { return false; }
    APInt AP;
#if LLVM_VERSION_GE(15, 0)
    if (sext) {
        AP = C->getValue().sext(128);
    } else {
        AP = C->getValue().zext(128);
    }
#else
    if (sext) {
        AP = C->getValue().sextOrSelf(128);
    } else {
        AP = C->getValue().zextOrSelf(128);
    }
#endif
    *low = AP.getLoBits(64).getZExtValue();
    *high = AP.getHiBits(64).getZExtValue();
    return true;
}

enum class LLVMRustVisibility {
  Default = 0,
  Hidden = 1,
  Protected = 2,
};

static LLVMRustVisibility toRust(LLVMVisibility Vis) {
  switch (Vis) {
  case LLVMDefaultVisibility:
    return LLVMRustVisibility::Default;
  case LLVMHiddenVisibility:
    return LLVMRustVisibility::Hidden;
  case LLVMProtectedVisibility:
    return LLVMRustVisibility::Protected;
  }
  report_fatal_error("Invalid LLVMRustVisibility value!");
}

static LLVMVisibility fromRust(LLVMRustVisibility Vis) {
  switch (Vis) {
  case LLVMRustVisibility::Default:
    return LLVMDefaultVisibility;
  case LLVMRustVisibility::Hidden:
    return LLVMHiddenVisibility;
  case LLVMRustVisibility::Protected:
    return LLVMProtectedVisibility;
  }
  report_fatal_error("Invalid LLVMRustVisibility value!");
}

extern "C" LLVMRustVisibility LLVMRustGetVisibility(LLVMValueRef V) {
  return toRust(LLVMGetVisibility(V));
}

extern "C" void LLVMRustSetVisibility(LLVMValueRef V,
                                      LLVMRustVisibility RustVisibility) {
  LLVMSetVisibility(V, fromRust(RustVisibility));
}

extern "C" void LLVMRustSetDSOLocal(LLVMValueRef Global, bool is_dso_local) {
  unwrap<GlobalValue>(Global)->setDSOLocal(is_dso_local);
}

struct LLVMRustModuleBuffer {
  std::string data;
};

extern "C" LLVMRustModuleBuffer*
LLVMRustModuleBufferCreate(LLVMModuleRef M) {
  auto Ret = std::make_unique<LLVMRustModuleBuffer>();
  {
    raw_string_ostream OS(Ret->data);
    WriteBitcodeToFile(*unwrap(M), OS);
  }
  return Ret.release();
}

extern "C" void
LLVMRustModuleBufferFree(LLVMRustModuleBuffer *Buffer) {
  delete Buffer;
}

extern "C" const void*
LLVMRustModuleBufferPtr(const LLVMRustModuleBuffer *Buffer) {
  return Buffer->data.data();
}

extern "C" size_t
LLVMRustModuleBufferLen(const LLVMRustModuleBuffer *Buffer) {
  return Buffer->data.length();
}

extern "C" uint64_t
LLVMRustModuleCost(LLVMModuleRef M) {
  auto f = unwrap(M)->functions();
  return std::distance(std::begin(f), std::end(f));
}

extern "C" void
LLVMRustModuleInstructionStats(LLVMModuleRef M, RustStringRef Str)
{
  RawRustStringOstream OS(Str);
  llvm::json::OStream JOS(OS);
  auto Module = unwrap(M);

  JOS.object([&] {
    JOS.attribute("module", Module->getName());
    JOS.attribute("total", Module->getInstructionCount());
  });
}

// Vector reductions:
extern "C" LLVMValueRef
LLVMRustBuildVectorReduceFAdd(LLVMBuilderRef B, LLVMValueRef Acc, LLVMValueRef Src) {
    return wrap(unwrap(B)->CreateFAddReduce(unwrap(Acc),unwrap(Src)));
}
extern "C" LLVMValueRef
LLVMRustBuildVectorReduceFMul(LLVMBuilderRef B, LLVMValueRef Acc, LLVMValueRef Src) {
    return wrap(unwrap(B)->CreateFMulReduce(unwrap(Acc),unwrap(Src)));
}
extern "C" LLVMValueRef
LLVMRustBuildVectorReduceAdd(LLVMBuilderRef B, LLVMValueRef Src) {
    return wrap(unwrap(B)->CreateAddReduce(unwrap(Src)));
}
extern "C" LLVMValueRef
LLVMRustBuildVectorReduceMul(LLVMBuilderRef B, LLVMValueRef Src) {
    return wrap(unwrap(B)->CreateMulReduce(unwrap(Src)));
}
extern "C" LLVMValueRef
LLVMRustBuildVectorReduceAnd(LLVMBuilderRef B, LLVMValueRef Src) {
    return wrap(unwrap(B)->CreateAndReduce(unwrap(Src)));
}
extern "C" LLVMValueRef
LLVMRustBuildVectorReduceOr(LLVMBuilderRef B, LLVMValueRef Src) {
    return wrap(unwrap(B)->CreateOrReduce(unwrap(Src)));
}
extern "C" LLVMValueRef
LLVMRustBuildVectorReduceXor(LLVMBuilderRef B, LLVMValueRef Src) {
    return wrap(unwrap(B)->CreateXorReduce(unwrap(Src)));
}
extern "C" LLVMValueRef
LLVMRustBuildVectorReduceMin(LLVMBuilderRef B, LLVMValueRef Src, bool IsSigned) {
    return wrap(unwrap(B)->CreateIntMinReduce(unwrap(Src), IsSigned));
}
extern "C" LLVMValueRef
LLVMRustBuildVectorReduceMax(LLVMBuilderRef B, LLVMValueRef Src, bool IsSigned) {
    return wrap(unwrap(B)->CreateIntMaxReduce(unwrap(Src), IsSigned));
}
extern "C" LLVMValueRef
LLVMRustBuildVectorReduceFMin(LLVMBuilderRef B, LLVMValueRef Src, bool NoNaN) {
  Instruction *I = unwrap(B)->CreateFPMinReduce(unwrap(Src));
  I->setHasNoNaNs(NoNaN);
  return wrap(I);
}
extern "C" LLVMValueRef
LLVMRustBuildVectorReduceFMax(LLVMBuilderRef B, LLVMValueRef Src, bool NoNaN) {
  Instruction *I = unwrap(B)->CreateFPMaxReduce(unwrap(Src));
  I->setHasNoNaNs(NoNaN);
  return wrap(I);
}

extern "C" LLVMValueRef
LLVMRustBuildMinNum(LLVMBuilderRef B, LLVMValueRef LHS, LLVMValueRef RHS) {
    return wrap(unwrap(B)->CreateMinNum(unwrap(LHS),unwrap(RHS)));
}
extern "C" LLVMValueRef
LLVMRustBuildMaxNum(LLVMBuilderRef B, LLVMValueRef LHS, LLVMValueRef RHS) {
    return wrap(unwrap(B)->CreateMaxNum(unwrap(LHS),unwrap(RHS)));
}

// This struct contains all necessary info about a symbol exported from a DLL.
struct LLVMRustCOFFShortExport {
  const char* name;
  bool ordinal_present;
  // The value of `ordinal` is only meaningful if `ordinal_present` is true.
  uint16_t ordinal;
};

// Machine must be a COFF machine type, as defined in PE specs.
extern "C" LLVMRustResult LLVMRustWriteImportLibrary(
  const char* ImportName,
  const char* Path,
  const LLVMRustCOFFShortExport* Exports,
  size_t NumExports,
  uint16_t Machine,
  bool MinGW)
{
  std::vector<llvm::object::COFFShortExport> ConvertedExports;
  ConvertedExports.reserve(NumExports);

  for (size_t i = 0; i < NumExports; ++i) {
    bool ordinal_present = Exports[i].ordinal_present;
    uint16_t ordinal = ordinal_present ? Exports[i].ordinal : 0;
    ConvertedExports.push_back(llvm::object::COFFShortExport{
      Exports[i].name,  // Name
      std::string{},    // ExtName
      std::string{},    // SymbolName
      std::string{},    // AliasTarget
      ordinal,          // Ordinal
      ordinal_present,  // Noname
      false,            // Data
      false,            // Private
      false             // Constant
    });
  }

  auto Error = llvm::object::writeImportLibrary(
    ImportName,
    Path,
    ConvertedExports,
    static_cast<llvm::COFF::MachineTypes>(Machine),
    MinGW);
  if (Error) {
    std::string errorString;
    llvm::raw_string_ostream stream(errorString);
    stream << Error;
    stream.flush();
    LLVMRustSetLastError(errorString.c_str());
    return LLVMRustResult::Failure;
  } else {
    return LLVMRustResult::Success;
  }
}

// Transfers ownership of DiagnosticHandler unique_ptr to the caller.
extern "C" DiagnosticHandler *
LLVMRustContextGetDiagnosticHandler(LLVMContextRef C) {
  std::unique_ptr<DiagnosticHandler> DH = unwrap(C)->getDiagnosticHandler();
  return DH.release();
}

// Sets unique_ptr to object of DiagnosticHandler to provide custom diagnostic
// handling. Ownership of the handler is moved to the LLVMContext.
extern "C" void LLVMRustContextSetDiagnosticHandler(LLVMContextRef C,
                                                    DiagnosticHandler *DH) {
  unwrap(C)->setDiagnosticHandler(std::unique_ptr<DiagnosticHandler>(DH));
}

using LLVMDiagnosticHandlerTy = DiagnosticHandler::DiagnosticHandlerTy;

// Configures a diagnostic handler that invokes provided callback when a
// backend needs to emit a diagnostic.
//
// When RemarkAllPasses is true, remarks are enabled for all passes. Otherwise
// the RemarkPasses array specifies individual passes for which remarks will be
// enabled.
//
// If RemarkFilePath is not NULL, optimization remarks will be streamed directly into this file,
// bypassing the diagnostics handler.
extern "C" void LLVMRustContextConfigureDiagnosticHandler(
    LLVMContextRef C, LLVMDiagnosticHandlerTy DiagnosticHandlerCallback,
    void *DiagnosticHandlerContext, bool RemarkAllPasses,
    const char * const * RemarkPasses, size_t RemarkPassesLen,
    const char * RemarkFilePath
) {

  class RustDiagnosticHandler final : public DiagnosticHandler {
  public:
    RustDiagnosticHandler(
      LLVMDiagnosticHandlerTy DiagnosticHandlerCallback,
      void *DiagnosticHandlerContext,
      bool RemarkAllPasses,
      std::vector<std::string> RemarkPasses,
      std::unique_ptr<ToolOutputFile> RemarksFile,
      std::unique_ptr<llvm::remarks::RemarkStreamer> RemarkStreamer,
      std::unique_ptr<LLVMRemarkStreamer> LlvmRemarkStreamer
    )
        : DiagnosticHandlerCallback(DiagnosticHandlerCallback),
          DiagnosticHandlerContext(DiagnosticHandlerContext),
          RemarkAllPasses(RemarkAllPasses),
          RemarkPasses(std::move(RemarkPasses)),
          RemarksFile(std::move(RemarksFile)),
          RemarkStreamer(std::move(RemarkStreamer)),
          LlvmRemarkStreamer(std::move(LlvmRemarkStreamer)) {}

    virtual bool handleDiagnostics(const DiagnosticInfo &DI) override {
      if (this->LlvmRemarkStreamer) {
        if (auto *OptDiagBase = dyn_cast<DiagnosticInfoOptimizationBase>(&DI)) {
          if (OptDiagBase->isEnabled()) {
            this->LlvmRemarkStreamer->emit(*OptDiagBase);
            return true;
          }
        }
      }
      if (DiagnosticHandlerCallback) {
        DiagnosticHandlerCallback(DI, DiagnosticHandlerContext);
        return true;
      }
      return false;
    }

    bool isAnalysisRemarkEnabled(StringRef PassName) const override {
      return isRemarkEnabled(PassName);
    }

    bool isMissedOptRemarkEnabled(StringRef PassName) const override {
      return isRemarkEnabled(PassName);
    }

    bool isPassedOptRemarkEnabled(StringRef PassName) const override {
      return isRemarkEnabled(PassName);
    }

    bool isAnyRemarkEnabled() const override {
      return RemarkAllPasses || !RemarkPasses.empty();
    }

  private:
    bool isRemarkEnabled(StringRef PassName) const {
      if (RemarkAllPasses)
        return true;

      for (auto &Pass : RemarkPasses)
        if (Pass == PassName)
          return true;

      return false;
    }

    LLVMDiagnosticHandlerTy DiagnosticHandlerCallback = nullptr;
    void *DiagnosticHandlerContext = nullptr;

    bool RemarkAllPasses = false;
    std::vector<std::string> RemarkPasses;

    // Since LlvmRemarkStreamer contains a pointer to RemarkStreamer, the ordering of the three
    // members below is important.
    std::unique_ptr<ToolOutputFile> RemarksFile;
    std::unique_ptr<llvm::remarks::RemarkStreamer> RemarkStreamer;
    std::unique_ptr<LLVMRemarkStreamer> LlvmRemarkStreamer;
  };

  std::vector<std::string> Passes;
  for (size_t I = 0; I != RemarkPassesLen; ++I)
  {
    Passes.push_back(RemarkPasses[I]);
  }

  // We need to hold onto both the streamers and the opened file
  std::unique_ptr<ToolOutputFile> RemarkFile;
  std::unique_ptr<llvm::remarks::RemarkStreamer> RemarkStreamer;
  std::unique_ptr<LLVMRemarkStreamer> LlvmRemarkStreamer;

  if (RemarkFilePath != nullptr) {
    std::error_code EC;
    RemarkFile = std::make_unique<ToolOutputFile>(
      RemarkFilePath,
      EC,
      llvm::sys::fs::OF_TextWithCRLF
    );
    if (EC) {
      std::string Error = std::string("Cannot create remark file: ") +
              toString(errorCodeToError(EC));
      report_fatal_error(Twine(Error));
    }

    // Do not delete the file after we gather remarks
    RemarkFile->keep();

    auto RemarkSerializer = remarks::createRemarkSerializer(
      llvm::remarks::Format::YAML,
      remarks::SerializerMode::Separate,
      RemarkFile->os()
    );
    if (Error E = RemarkSerializer.takeError())
    {
      std::string Error = std::string("Cannot create remark serializer: ") + toString(std::move(E));
      report_fatal_error(Twine(Error));
    }
    RemarkStreamer = std::make_unique<llvm::remarks::RemarkStreamer>(std::move(*RemarkSerializer));
    LlvmRemarkStreamer = std::make_unique<LLVMRemarkStreamer>(*RemarkStreamer);
  }

  unwrap(C)->setDiagnosticHandler(std::make_unique<RustDiagnosticHandler>(
    DiagnosticHandlerCallback,
    DiagnosticHandlerContext,
    RemarkAllPasses,
    Passes,
    std::move(RemarkFile),
    std::move(RemarkStreamer),
    std::move(LlvmRemarkStreamer)
  ));
}

extern "C" void LLVMRustGetMangledName(LLVMValueRef V, RustStringRef Str) {
  RawRustStringOstream OS(Str);
  GlobalValue *GV = unwrap<GlobalValue>(V);
  Mangler().getNameWithPrefix(OS, GV, true);
}

// LLVMGetAggregateElement was added in LLVM 15. For earlier LLVM versions just
// use its implementation.
#if LLVM_VERSION_LT(15, 0)
extern "C" LLVMValueRef LLVMGetAggregateElement(LLVMValueRef C, unsigned Idx) {
    return wrap(unwrap<Constant>(C)->getAggregateElement(Idx));
}
#endif

extern "C" int32_t LLVMRustGetElementTypeArgIndex(LLVMValueRef CallSite) {
#if LLVM_VERSION_GE(15, 0)
    auto *CB = unwrap<CallBase>(CallSite);
    switch (CB->getIntrinsicID()) {
        case Intrinsic::arm_ldrex:
            return 0;
        case Intrinsic::arm_strex:
            return 1;
    }
#endif
    return -1;
}

extern "C" bool LLVMRustIsBitcode(char *ptr, size_t len) {
  return identify_magic(StringRef(ptr, len)) == file_magic::bitcode;
}
