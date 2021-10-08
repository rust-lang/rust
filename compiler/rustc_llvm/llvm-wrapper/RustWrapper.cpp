#include "LLVMWrapper.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/COFFImportFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Bitcode/BitcodeWriterPass.h"
#include "llvm/Support/Signals.h"
#include "llvm/ADT/Optional.h"

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
#if LLVM_VERSION_LT(14, 0)
                              const std::string& Reason,
#else
                              const char* Reason,
#endif
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

extern "C" char *LLVMRustGetLastError(void) {
  char *Ret = LastError;
  LastError = nullptr;
  return Ret;
}

extern "C" unsigned int LLVMRustGetInstructionCount(LLVMModuleRef M) {
  return unwrap(M)->getInstructionCount();
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
  StringRef NameRef(Name, NameLen);
  return wrap(unwrap(M)->getOrInsertGlobal(NameRef, unwrap(Ty)));
}

extern "C" LLVMValueRef
LLVMRustInsertPrivateGlobal(LLVMModuleRef M, LLVMTypeRef Ty) {
  return wrap(new GlobalVariable(*unwrap(M),
                                 unwrap(Ty),
                                 false,
                                 GlobalValue::PrivateLinkage,
                                 nullptr));
}

extern "C" LLVMTypeRef LLVMRustMetadataTypeInContext(LLVMContextRef C) {
  return wrap(Type::getMetadataTy(*unwrap(C)));
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
  case InaccessibleMemOnly:
    return Attribute::InaccessibleMemOnly;
  case SanitizeHWAddress:
    return Attribute::SanitizeHWAddress;
  case WillReturn:
    return Attribute::WillReturn;
  }
  report_fatal_error("bad AttributeKind");
}

template<typename T> static inline void AddAttribute(T *t, unsigned Index, Attribute Attr) {
#if LLVM_VERSION_LT(14, 0)
  t->addAttribute(Index, Attr);
#else
  t->addAttributeAtIndex(Index, Attr);
#endif
}

extern "C" void LLVMRustAddCallSiteAttribute(LLVMValueRef Instr, unsigned Index,
                                             LLVMRustAttribute RustAttr) {
  CallBase *Call = unwrap<CallBase>(Instr);
  Attribute Attr = Attribute::get(Call->getContext(), fromRust(RustAttr));
  AddAttribute(Call, Index, Attr);
}

extern "C" void LLVMRustAddCallSiteAttrString(LLVMValueRef Instr, unsigned Index,
                                              const char *Name) {
  CallBase *Call = unwrap<CallBase>(Instr);
  Attribute Attr = Attribute::get(Call->getContext(), Name);
  AddAttribute(Call, Index, Attr);
}

extern "C" void LLVMRustAddAlignmentCallSiteAttr(LLVMValueRef Instr,
                                                 unsigned Index,
                                                 uint32_t Bytes) {
  CallBase *Call = unwrap<CallBase>(Instr);
  Attribute Attr = Attribute::getWithAlignment(Call->getContext(), Align(Bytes));
  AddAttribute(Call, Index, Attr);
}

extern "C" void LLVMRustAddDereferenceableCallSiteAttr(LLVMValueRef Instr,
                                                       unsigned Index,
                                                       uint64_t Bytes) {
  CallBase *Call = unwrap<CallBase>(Instr);
  Attribute Attr = Attribute::getWithDereferenceableBytes(Call->getContext(), Bytes);
  AddAttribute(Call, Index, Attr);
}

extern "C" void LLVMRustAddDereferenceableOrNullCallSiteAttr(LLVMValueRef Instr,
                                                             unsigned Index,
                                                             uint64_t Bytes) {
  CallBase *Call = unwrap<CallBase>(Instr);
  Attribute Attr = Attribute::getWithDereferenceableOrNullBytes(Call->getContext(), Bytes);
  AddAttribute(Call, Index, Attr);
}

extern "C" void LLVMRustAddByValCallSiteAttr(LLVMValueRef Instr, unsigned Index,
                                             LLVMTypeRef Ty) {
  CallBase *Call = unwrap<CallBase>(Instr);
  Attribute Attr = Attribute::getWithByValType(Call->getContext(), unwrap(Ty));
  AddAttribute(Call, Index, Attr);
}

extern "C" void LLVMRustAddStructRetCallSiteAttr(LLVMValueRef Instr, unsigned Index,
                                                 LLVMTypeRef Ty) {
  CallBase *Call = unwrap<CallBase>(Instr);
#if LLVM_VERSION_GE(12, 0)
  Attribute Attr = Attribute::getWithStructRetType(Call->getContext(), unwrap(Ty));
#else
  Attribute Attr = Attribute::get(Call->getContext(), Attribute::StructRet);
#endif
  AddAttribute(Call, Index, Attr);
}

extern "C" void LLVMRustAddFunctionAttribute(LLVMValueRef Fn, unsigned Index,
                                             LLVMRustAttribute RustAttr) {
  Function *A = unwrap<Function>(Fn);
  Attribute Attr = Attribute::get(A->getContext(), fromRust(RustAttr));
  AddAttribute(A, Index, Attr);
}

extern "C" void LLVMRustAddAlignmentAttr(LLVMValueRef Fn,
                                         unsigned Index,
                                         uint32_t Bytes) {
  Function *A = unwrap<Function>(Fn);
  AddAttribute(A, Index, Attribute::getWithAlignment(
      A->getContext(), llvm::Align(Bytes)));
}

extern "C" void LLVMRustAddDereferenceableAttr(LLVMValueRef Fn, unsigned Index,
                                               uint64_t Bytes) {
  Function *A = unwrap<Function>(Fn);
  AddAttribute(A, Index, Attribute::getWithDereferenceableBytes(A->getContext(),
                                                                Bytes));
}

extern "C" void LLVMRustAddDereferenceableOrNullAttr(LLVMValueRef Fn,
                                                     unsigned Index,
                                                     uint64_t Bytes) {
  Function *A = unwrap<Function>(Fn);
  AddAttribute(A, Index, Attribute::getWithDereferenceableOrNullBytes(
      A->getContext(), Bytes));
}

extern "C" void LLVMRustAddByValAttr(LLVMValueRef Fn, unsigned Index,
                                     LLVMTypeRef Ty) {
  Function *F = unwrap<Function>(Fn);
  Attribute Attr = Attribute::getWithByValType(F->getContext(), unwrap(Ty));
  AddAttribute(F, Index, Attr);
}

extern "C" void LLVMRustAddStructRetAttr(LLVMValueRef Fn, unsigned Index,
                                         LLVMTypeRef Ty) {
  Function *F = unwrap<Function>(Fn);
#if LLVM_VERSION_GE(12, 0)
  Attribute Attr = Attribute::getWithStructRetType(F->getContext(), unwrap(Ty));
#else
  Attribute Attr = Attribute::get(F->getContext(), Attribute::StructRet);
#endif
  AddAttribute(F, Index, Attr);
}

extern "C" void LLVMRustAddFunctionAttrStringValue(LLVMValueRef Fn,
                                                   unsigned Index,
                                                   const char *Name,
                                                   const char *Value) {
  Function *F = unwrap<Function>(Fn);
  AddAttribute(F, Index, Attribute::get(
      F->getContext(), StringRef(Name), StringRef(Value)));
}

extern "C" void LLVMRustRemoveFunctionAttributes(LLVMValueRef Fn,
                                                 unsigned Index,
                                                 LLVMRustAttribute RustAttr) {
  Function *F = unwrap<Function>(Fn);
  Attribute Attr = Attribute::get(F->getContext(), fromRust(RustAttr));
  AttrBuilder B(Attr);
  auto PAL = F->getAttributes();
  AttributeList PALNew;
#if LLVM_VERSION_LT(14, 0)
  PALNew = PAL.removeAttributes(F->getContext(), Index, B);
#else
  PALNew = PAL.removeAttributesAtIndex(F->getContext(), Index, B);
#endif
  F->setAttributes(PALNew);
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

// FIXME: Use the C-API LLVMBuildAtomicCmpXchg and LLVMSetWeak
// once we raise our minimum support to LLVM 10.
extern "C" LLVMValueRef
LLVMRustBuildAtomicCmpXchg(LLVMBuilderRef B, LLVMValueRef Target,
                           LLVMValueRef Old, LLVMValueRef Source,
                           LLVMAtomicOrdering Order,
                           LLVMAtomicOrdering FailureOrder, LLVMBool Weak) {
#if LLVM_VERSION_GE(13,0)
  // Rust probably knows the alignment of the target value and should be able to
  // specify something more precise than MaybeAlign here. See also
  // https://reviews.llvm.org/D97224 which may be a useful reference.
  AtomicCmpXchgInst *ACXI = unwrap(B)->CreateAtomicCmpXchg(
      unwrap(Target), unwrap(Old), unwrap(Source), llvm::MaybeAlign(), fromRust(Order),
      fromRust(FailureOrder));
#else
  AtomicCmpXchgInst *ACXI = unwrap(B)->CreateAtomicCmpXchg(
      unwrap(Target), unwrap(Old), unwrap(Source), fromRust(Order),
      fromRust(FailureOrder));
#endif
  ACXI->setWeak(Weak);
  return wrap(ACXI);
}

enum class LLVMRustSynchronizationScope {
  SingleThread,
  CrossThread,
};

static SyncScope::ID fromRust(LLVMRustSynchronizationScope Scope) {
  switch (Scope) {
  case LLVMRustSynchronizationScope::SingleThread:
    return SyncScope::SingleThread;
  case LLVMRustSynchronizationScope::CrossThread:
    return SyncScope::System;
  default:
    report_fatal_error("bad SynchronizationScope.");
  }
}

extern "C" LLVMValueRef
LLVMRustBuildAtomicFence(LLVMBuilderRef B, LLVMAtomicOrdering Order,
                         LLVMRustSynchronizationScope Scope) {
  return wrap(unwrap(B)->CreateFence(fromRust(Order), fromRust(Scope)));
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
                  LLVMRustAsmDialect Dialect) {
  return wrap(InlineAsm::get(unwrap<FunctionType>(Ty),
                             StringRef(AsmString, AsmStringLen),
                             StringRef(Constraints, ConstraintsLen),
                             HasSideEffects, IsAlignStack, fromRust(Dialect)));
}

extern "C" bool LLVMRustInlineAsmVerify(LLVMTypeRef Ty, char *Constraints,
                                        size_t ConstraintsLen) {
  return InlineAsm::Verify(unwrap<FunctionType>(Ty),
                           StringRef(Constraints, ConstraintsLen));
}

extern "C" void LLVMRustAppendModuleInlineAsm(LLVMModuleRef M, const char *Asm,
                                              size_t AsmLen) {
  unwrap(M)->appendModuleInlineAsm(StringRef(Asm, AsmLen));
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
};

static DICompileUnit::DebugEmissionKind fromRust(LLVMRustDebugEmissionKind Kind) {
  switch (Kind) {
  case LLVMRustDebugEmissionKind::NoDebug:
    return DICompileUnit::DebugEmissionKind::NoDebug;
  case LLVMRustDebugEmissionKind::FullDebug:
    return DICompileUnit::DebugEmissionKind::FullDebug;
  case LLVMRustDebugEmissionKind::LineTablesOnly:
    return DICompileUnit::DebugEmissionKind::LineTablesOnly;
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

static Optional<DIFile::ChecksumKind> fromRust(LLVMRustChecksumKind Kind) {
  switch (Kind) {
  case LLVMRustChecksumKind::None:
    return None;
  case LLVMRustChecksumKind::MD5:
    return DIFile::ChecksumKind::CSK_MD5;
  case LLVMRustChecksumKind::SHA1:
    return DIFile::ChecksumKind::CSK_SHA1;
#if (LLVM_VERSION_MAJOR >= 11)
  case LLVMRustChecksumKind::SHA256:
    return DIFile::ChecksumKind::CSK_SHA256;
#endif
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

extern "C" void LLVMRustAddModuleFlag(LLVMModuleRef M, const char *Name,
                                      uint32_t Value) {
  unwrap(M)->addModuleFlag(Module::Warning, Name, Value);
}

extern "C" LLVMValueRef LLVMRustMetadataAsValue(LLVMContextRef C, LLVMMetadataRef MD) {
  return wrap(MetadataAsValue::get(*unwrap(C), unwrap(MD)));
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
  Optional<DIFile::ChecksumKind> llvmCSKind = fromRust(CSKind);
  Optional<DIFile::ChecksumInfo<StringRef>> CSInfo{};
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
    int64_t *AddrOps, unsigned AddrOpsCount, LLVMMetadataRef DL,
    LLVMBasicBlockRef InsertAtEnd) {
  return wrap(Builder->insertDeclare(
      unwrap(V), unwrap<DILocalVariable>(VarInfo),
      Builder->createExpression(llvm::ArrayRef<int64_t>(AddrOps, AddrOpsCount)),
      DebugLoc(cast<MDNode>(unwrap(DL))),
      unwrap(InsertAtEnd)));
}

extern "C" LLVMMetadataRef LLVMRustDIBuilderCreateEnumerator(
    LLVMRustDIBuilderRef Builder, const char *Name, size_t NameLen,
    int64_t Value, bool IsUnsigned) {
  return wrap(Builder->createEnumerator(StringRef(Name, NameLen), Value, IsUnsigned));
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
#if LLVM_VERSION_GE(11, 0)
  bool IsDefault = false; // FIXME: should we ever set this true?
  return wrap(Builder->createTemplateTypeParameter(
      unwrapDI<DIDescriptor>(Scope), StringRef(Name, NameLen), unwrapDI<DIType>(Ty), IsDefault));
#else
  return wrap(Builder->createTemplateTypeParameter(
      unwrapDI<DIDescriptor>(Scope), StringRef(Name, NameLen), unwrapDI<DIType>(Ty)));
#endif
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
#if LLVM_VERSION_GE(12, 0)
  MDNode *Scope = unwrapDIPtr<MDNode>(ScopeRef);
  DILocation *Loc = DILocation::get(
      Scope->getContext(), Line, Column, Scope,
      unwrapDIPtr<MDNode>(InlinedAt));
  return wrap(Loc);
#else
  DebugLoc debug_loc = DebugLoc::get(Line, Column, unwrapDIPtr<MDNode>(ScopeRef),
                                     unwrapDIPtr<MDNode>(InlinedAt));
  return wrap(debug_loc.getAsMDNode());
#endif
}

extern "C" int64_t LLVMRustDIBuilderCreateOpDeref() {
  return dwarf::DW_OP_deref;
}

extern "C" int64_t LLVMRustDIBuilderCreateOpPlusUconst() {
  return dwarf::DW_OP_plus_uconst;
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
    return LLVMRustDiagnosticKind::OptimizationRemark;
  case DK_OptimizationRemarkMissed:
    return LLVMRustDiagnosticKind::OptimizationRemarkMissed;
  case DK_OptimizationRemarkAnalysis:
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
#if LLVM_VERSION_GE(13, 0)
  case DK_SrcMgr:
    return LLVMRustDiagnosticKind::SrcMgr;
#endif
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
#if LLVM_VERSION_GE(11, 0)
  case Type::FixedVectorTyID:
    return LLVMVectorTypeKind;
#else
  case Type::VectorTyID:
    return LLVMVectorTypeKind;
#endif
  case Type::X86_MMXTyID:
    return LLVMX86_MMXTypeKind;
  case Type::TokenTyID:
    return LLVMTokenTypeKind;
#if LLVM_VERSION_GE(11, 0)
  case Type::ScalableVectorTyID:
    return LLVMScalableVectorTypeKind;
  case Type::BFloatTyID:
    return LLVMBFloatTypeKind;
#endif
#if LLVM_VERSION_GE(12, 0)
  case Type::X86_AMXTyID:
    return LLVMX86_AMXTypeKind;
#endif
  }
  report_fatal_error("Unhandled TypeID.");
}

DEFINE_SIMPLE_CONVERSION_FUNCTIONS(SMDiagnostic, LLVMSMDiagnosticRef)

#if LLVM_VERSION_LT(13, 0)
using LLVMInlineAsmDiagHandlerTy = LLVMContext::InlineAsmDiagHandlerTy;
#else
using LLVMInlineAsmDiagHandlerTy = void*;
#endif

extern "C" void LLVMRustSetInlineAsmDiagnosticHandler(
    LLVMContextRef C, LLVMInlineAsmDiagHandlerTy H, void *CX) {
  // Diagnostic handlers were unified in LLVM change 5de2d189e6ad, so starting
  // with LLVM 13 this function is gone.
#if LLVM_VERSION_LT(13, 0)
  unwrap(C)->setInlineAsmDiagnosticHandler(H, CX);
#endif
}

extern "C" LLVMSMDiagnosticRef LLVMRustGetSMDiagnostic(
    LLVMDiagnosticInfoRef DI, unsigned *Cookie) {
#if LLVM_VERSION_GE(13, 0)
  llvm::DiagnosticInfoSrcMgr *SM = static_cast<llvm::DiagnosticInfoSrcMgr *>(unwrap(DI));
  *Cookie = SM->getLocCookie();
  return wrap(&SM->getSMDiag());
#else
  report_fatal_error("Shouldn't get called on older versions");
#endif
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

extern "C" LLVMValueRef LLVMRustBuildCleanupPad(LLVMBuilderRef B,
                                                LLVMValueRef ParentPad,
                                                unsigned ArgCount,
                                                LLVMValueRef *LLArgs,
                                                const char *Name) {
  Value **Args = unwrap(LLArgs);
  if (ParentPad == nullptr) {
    Type *Ty = Type::getTokenTy(unwrap(B)->getContext());
    ParentPad = wrap(Constant::getNullValue(Ty));
  }
  return wrap(unwrap(B)->CreateCleanupPad(
      unwrap(ParentPad), ArrayRef<Value *>(Args, ArgCount), Name));
}

extern "C" LLVMValueRef LLVMRustBuildCleanupRet(LLVMBuilderRef B,
                                                LLVMValueRef CleanupPad,
                                                LLVMBasicBlockRef UnwindBB) {
  CleanupPadInst *Inst = cast<CleanupPadInst>(unwrap(CleanupPad));
  return wrap(unwrap(B)->CreateCleanupRet(Inst, unwrap(UnwindBB)));
}

extern "C" LLVMValueRef
LLVMRustBuildCatchPad(LLVMBuilderRef B, LLVMValueRef ParentPad,
                      unsigned ArgCount, LLVMValueRef *LLArgs, const char *Name) {
  Value **Args = unwrap(LLArgs);
  return wrap(unwrap(B)->CreateCatchPad(
      unwrap(ParentPad), ArrayRef<Value *>(Args, ArgCount), Name));
}

extern "C" LLVMValueRef LLVMRustBuildCatchRet(LLVMBuilderRef B,
                                              LLVMValueRef Pad,
                                              LLVMBasicBlockRef BB) {
  return wrap(unwrap(B)->CreateCatchRet(cast<CatchPadInst>(unwrap(Pad)),
                                              unwrap(BB)));
}

extern "C" LLVMValueRef LLVMRustBuildCatchSwitch(LLVMBuilderRef B,
                                                 LLVMValueRef ParentPad,
                                                 LLVMBasicBlockRef BB,
                                                 unsigned NumHandlers,
                                                 const char *Name) {
  if (ParentPad == nullptr) {
    Type *Ty = Type::getTokenTy(unwrap(B)->getContext());
    ParentPad = wrap(Constant::getNullValue(Ty));
  }
  return wrap(unwrap(B)->CreateCatchSwitch(unwrap(ParentPad), unwrap(BB),
                                                 NumHandlers, Name));
}

extern "C" void LLVMRustAddHandler(LLVMValueRef CatchSwitchRef,
                                   LLVMBasicBlockRef Handler) {
  Value *CatchSwitch = unwrap(CatchSwitchRef);
  cast<CatchSwitchInst>(CatchSwitch)->addHandler(unwrap(Handler));
}

extern "C" OperandBundleDef *LLVMRustBuildOperandBundleDef(const char *Name,
                                                           LLVMValueRef *Inputs,
                                                           unsigned NumInputs) {
  return new OperandBundleDef(Name, makeArrayRef(unwrap(Inputs), NumInputs));
}

extern "C" void LLVMRustFreeOperandBundleDef(OperandBundleDef *Bundle) {
  delete Bundle;
}

extern "C" LLVMValueRef LLVMRustBuildCall(LLVMBuilderRef B, LLVMTypeRef Ty, LLVMValueRef Fn,
                                          LLVMValueRef *Args, unsigned NumArgs,
                                          OperandBundleDef *Bundle) {
  Value *Callee = unwrap(Fn);
  FunctionType *FTy = unwrap<FunctionType>(Ty);
  unsigned Len = Bundle ? 1 : 0;
  ArrayRef<OperandBundleDef> Bundles = makeArrayRef(Bundle, Len);
  return wrap(unwrap(B)->CreateCall(
      FTy, Callee, makeArrayRef(unwrap(Args), NumArgs), Bundles));
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
                    OperandBundleDef *Bundle, const char *Name) {
  Value *Callee = unwrap(Fn);
  FunctionType *FTy = unwrap<FunctionType>(Ty);
  unsigned Len = Bundle ? 1 : 0;
  ArrayRef<OperandBundleDef> Bundles = makeArrayRef(Bundle, Len);
  return wrap(unwrap(B)->CreateInvoke(FTy, Callee, unwrap(Then), unwrap(Catch),
                                      makeArrayRef(unwrap(Args), NumArgs),
                                      Bundles, Name));
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

extern "C" void LLVMRustUnsetComdat(LLVMValueRef V) {
  GlobalObject *GV = unwrap<GlobalObject>(V);
  GV->setComdat(nullptr);
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

extern "C" LLVMValueRef LLVMRustConstInBoundsGEP2(LLVMTypeRef Ty,
                                                  LLVMValueRef ConstantVal,
                                                  LLVMValueRef *ConstantIndices,
                                                  unsigned NumIndices) {
  ArrayRef<Constant *> IdxList(unwrap<Constant>(ConstantIndices, NumIndices),
                               NumIndices);
  Constant *Val = unwrap<Constant>(ConstantVal);
  return wrap(ConstantExpr::getInBoundsGetElementPtr(unwrap(Ty), Val, IdxList));
}

// Returns true if both high and low were successfully set. Fails in case constant wasn’t any of
// the common sizes (1, 8, 16, 32, 64, 128 bits)
extern "C" bool LLVMRustConstInt128Get(LLVMValueRef CV, bool sext, uint64_t *high, uint64_t *low)
{
    auto C = unwrap<llvm::ConstantInt>(CV);
    if (C->getBitWidth() > 128) { return false; }
    APInt AP;
    if (sext) {
        AP = C->getValue().sextOrSelf(128);
    } else {
        AP = C->getValue().zextOrSelf(128);
    }
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

// Oh hey, a binding that makes sense for once? (because LLVM’s own do not)
extern "C" LLVMValueRef LLVMRustBuildIntCast(LLVMBuilderRef B, LLVMValueRef Val,
                                             LLVMTypeRef DestTy, bool isSigned) {
  return wrap(unwrap(B)->CreateIntCast(unwrap(Val), unwrap(DestTy), isSigned, ""));
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
    {
      legacy::PassManager PM;
      PM.add(createBitcodeWriterPass(OS));
      PM.run(*unwrap(M));
    }
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
#if LLVM_VERSION_GE(12, 0)
  Instruction *I = unwrap(B)->CreateFPMinReduce(unwrap(Src));
  I->setHasNoNaNs(NoNaN);
  return wrap(I);
#else
  return wrap(unwrap(B)->CreateFPMinReduce(unwrap(Src), NoNaN));
#endif
}
extern "C" LLVMValueRef
LLVMRustBuildVectorReduceFMax(LLVMBuilderRef B, LLVMValueRef Src, bool NoNaN) {
#if LLVM_VERSION_GE(12, 0)
  Instruction *I = unwrap(B)->CreateFPMaxReduce(unwrap(Src));
  I->setHasNoNaNs(NoNaN);
  return wrap(I);
#else
  return wrap(unwrap(B)->CreateFPMaxReduce(unwrap(Src), NoNaN));
#endif
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
