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
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Bitcode/BitcodeWriterPass.h"

#include "llvm/IR/CallSite.h"

#if LLVM_VERSION_GE(5, 0)
#include "llvm/ADT/Optional.h"
#endif

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

extern "C" LLVMMemoryBufferRef
LLVMRustCreateMemoryBufferWithContentsOfFile(const char *Path) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> BufOr =
      MemoryBuffer::getFile(Path, -1, false);
  if (!BufOr) {
    LLVMRustSetLastError(BufOr.getError().message().c_str());
    return nullptr;
  }
  return wrap(BufOr.get().release());
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

extern "C" LLVMValueRef LLVMRustGetNamedValue(LLVMModuleRef M,
                                              const char *Name) {
  return wrap(unwrap(M)->getNamedValue(Name));
}

extern "C" LLVMValueRef LLVMRustGetOrInsertFunction(LLVMModuleRef M,
                                                    const char *Name,
                                                    LLVMTypeRef FunctionTy) {
  return wrap(
      unwrap(M)->getOrInsertFunction(Name, unwrap<FunctionType>(FunctionTy)));
}

extern "C" LLVMValueRef
LLVMRustGetOrInsertGlobal(LLVMModuleRef M, const char *Name, LLVMTypeRef Ty) {
  return wrap(unwrap(M)->getOrInsertGlobal(Name, unwrap(Ty)));
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
  }
  report_fatal_error("bad AttributeKind");
}

extern "C" void LLVMRustAddCallSiteAttribute(LLVMValueRef Instr, unsigned Index,
                                             LLVMRustAttribute RustAttr) {
  CallSite Call = CallSite(unwrap<Instruction>(Instr));
  Attribute Attr = Attribute::get(Call->getContext(), fromRust(RustAttr));
#if LLVM_VERSION_GE(5, 0)
  Call.addAttribute(Index, Attr);
#else
  AttrBuilder B(Attr);
  Call.setAttributes(Call.getAttributes().addAttributes(
      Call->getContext(), Index,
      AttributeSet::get(Call->getContext(), Index, B)));
#endif
}

extern "C" void LLVMRustAddAlignmentCallSiteAttr(LLVMValueRef Instr,
                                                 unsigned Index,
                                                 uint32_t Bytes) {
  CallSite Call = CallSite(unwrap<Instruction>(Instr));
  AttrBuilder B;
  B.addAlignmentAttr(Bytes);
#if LLVM_VERSION_GE(5, 0)
  Call.setAttributes(Call.getAttributes().addAttributes(
      Call->getContext(), Index, B));
#else
  Call.setAttributes(Call.getAttributes().addAttributes(
      Call->getContext(), Index,
      AttributeSet::get(Call->getContext(), Index, B)));
#endif
}

extern "C" void LLVMRustAddDereferenceableCallSiteAttr(LLVMValueRef Instr,
                                                       unsigned Index,
                                                       uint64_t Bytes) {
  CallSite Call = CallSite(unwrap<Instruction>(Instr));
  AttrBuilder B;
  B.addDereferenceableAttr(Bytes);
#if LLVM_VERSION_GE(5, 0)
  Call.setAttributes(Call.getAttributes().addAttributes(
      Call->getContext(), Index, B));
#else
  Call.setAttributes(Call.getAttributes().addAttributes(
      Call->getContext(), Index,
      AttributeSet::get(Call->getContext(), Index, B)));
#endif
}

extern "C" void LLVMRustAddDereferenceableOrNullCallSiteAttr(LLVMValueRef Instr,
                                                             unsigned Index,
                                                             uint64_t Bytes) {
  CallSite Call = CallSite(unwrap<Instruction>(Instr));
  AttrBuilder B;
  B.addDereferenceableOrNullAttr(Bytes);
#if LLVM_VERSION_GE(5, 0)
  Call.setAttributes(Call.getAttributes().addAttributes(
      Call->getContext(), Index, B));
#else
  Call.setAttributes(Call.getAttributes().addAttributes(
      Call->getContext(), Index,
      AttributeSet::get(Call->getContext(), Index, B)));
#endif
}

extern "C" void LLVMRustAddFunctionAttribute(LLVMValueRef Fn, unsigned Index,
                                             LLVMRustAttribute RustAttr) {
  Function *A = unwrap<Function>(Fn);
  Attribute Attr = Attribute::get(A->getContext(), fromRust(RustAttr));
  AttrBuilder B(Attr);
#if LLVM_VERSION_GE(5, 0)
  A->addAttributes(Index, B);
#else
  A->addAttributes(Index, AttributeSet::get(A->getContext(), Index, B));
#endif
}

extern "C" void LLVMRustAddAlignmentAttr(LLVMValueRef Fn,
                                         unsigned Index,
                                         uint32_t Bytes) {
  Function *A = unwrap<Function>(Fn);
  AttrBuilder B;
  B.addAlignmentAttr(Bytes);
#if LLVM_VERSION_GE(5, 0)
  A->addAttributes(Index, B);
#else
  A->addAttributes(Index, AttributeSet::get(A->getContext(), Index, B));
#endif
}

extern "C" void LLVMRustAddDereferenceableAttr(LLVMValueRef Fn, unsigned Index,
                                               uint64_t Bytes) {
  Function *A = unwrap<Function>(Fn);
  AttrBuilder B;
  B.addDereferenceableAttr(Bytes);
#if LLVM_VERSION_GE(5, 0)
  A->addAttributes(Index, B);
#else
  A->addAttributes(Index, AttributeSet::get(A->getContext(), Index, B));
#endif
}

extern "C" void LLVMRustAddDereferenceableOrNullAttr(LLVMValueRef Fn,
                                                     unsigned Index,
                                                     uint64_t Bytes) {
  Function *A = unwrap<Function>(Fn);
  AttrBuilder B;
  B.addDereferenceableOrNullAttr(Bytes);
#if LLVM_VERSION_GE(5, 0)
  A->addAttributes(Index, B);
#else
  A->addAttributes(Index, AttributeSet::get(A->getContext(), Index, B));
#endif
}

extern "C" void LLVMRustAddFunctionAttrStringValue(LLVMValueRef Fn,
                                                   unsigned Index,
                                                   const char *Name,
                                                   const char *Value) {
  Function *F = unwrap<Function>(Fn);
  AttrBuilder B;
  B.addAttribute(Name, Value);
#if LLVM_VERSION_GE(5, 0)
  F->addAttributes(Index, B);
#else
  F->addAttributes(Index, AttributeSet::get(F->getContext(), Index, B));
#endif
}

extern "C" void LLVMRustRemoveFunctionAttributes(LLVMValueRef Fn,
                                                 unsigned Index,
                                                 LLVMRustAttribute RustAttr) {
  Function *F = unwrap<Function>(Fn);
  Attribute Attr = Attribute::get(F->getContext(), fromRust(RustAttr));
  AttrBuilder B(Attr);
  auto PAL = F->getAttributes();
#if LLVM_VERSION_GE(5, 0)
  auto PALNew = PAL.removeAttributes(F->getContext(), Index, B);
#else
  const AttributeSet PALNew = PAL.removeAttributes(
      F->getContext(), Index, AttributeSet::get(F->getContext(), Index, B));
#endif
  F->setAttributes(PALNew);
}

// enable fpmath flag UnsafeAlgebra
extern "C" void LLVMRustSetHasUnsafeAlgebra(LLVMValueRef V) {
  if (auto I = dyn_cast<Instruction>(unwrap<Value>(V))) {
#if LLVM_VERSION_GE(6, 0)
    I->setFast(true);
#else
    I->setHasUnsafeAlgebra(true);
#endif
  }
}

extern "C" LLVMValueRef
LLVMRustBuildAtomicLoad(LLVMBuilderRef B, LLVMValueRef Source, const char *Name,
                        LLVMAtomicOrdering Order) {
  LoadInst *LI = new LoadInst(unwrap(Source), 0);
  LI->setAtomic(fromRust(Order));
  return wrap(unwrap(B)->Insert(LI, Name));
}

extern "C" LLVMValueRef LLVMRustBuildAtomicStore(LLVMBuilderRef B,
                                                 LLVMValueRef V,
                                                 LLVMValueRef Target,
                                                 LLVMAtomicOrdering Order) {
  StoreInst *SI = new StoreInst(unwrap(V), unwrap(Target));
  SI->setAtomic(fromRust(Order));
  return wrap(unwrap(B)->Insert(SI));
}

extern "C" LLVMValueRef
LLVMRustBuildAtomicCmpXchg(LLVMBuilderRef B, LLVMValueRef Target,
                           LLVMValueRef Old, LLVMValueRef Source,
                           LLVMAtomicOrdering Order,
                           LLVMAtomicOrdering FailureOrder, LLVMBool Weak) {
  AtomicCmpXchgInst *ACXI = unwrap(B)->CreateAtomicCmpXchg(
      unwrap(Target), unwrap(Old), unwrap(Source), fromRust(Order),
      fromRust(FailureOrder));
  ACXI->setWeak(Weak);
  return wrap(ACXI);
}

enum class LLVMRustSynchronizationScope {
  Other,
  SingleThread,
  CrossThread,
};

#if LLVM_VERSION_GE(5, 0)
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
#else
static SynchronizationScope fromRust(LLVMRustSynchronizationScope Scope) {
  switch (Scope) {
  case LLVMRustSynchronizationScope::SingleThread:
    return SingleThread;
  case LLVMRustSynchronizationScope::CrossThread:
    return CrossThread;
  default:
    report_fatal_error("bad SynchronizationScope.");
  }
}
#endif

extern "C" LLVMValueRef
LLVMRustBuildAtomicFence(LLVMBuilderRef B, LLVMAtomicOrdering Order,
                         LLVMRustSynchronizationScope Scope) {
  return wrap(unwrap(B)->CreateFence(fromRust(Order), fromRust(Scope)));
}

enum class LLVMRustAsmDialect {
  Other,
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

extern "C" LLVMValueRef LLVMRustInlineAsm(LLVMTypeRef Ty, char *AsmString,
                                          char *Constraints,
                                          LLVMBool HasSideEffects,
                                          LLVMBool IsAlignStack,
                                          LLVMRustAsmDialect Dialect) {
  return wrap(InlineAsm::get(unwrap<FunctionType>(Ty), AsmString, Constraints,
                             HasSideEffects, IsAlignStack, fromRust(Dialect)));
}

extern "C" void LLVMRustAppendModuleInlineAsm(LLVMModuleRef M, const char *Asm) {
  unwrap(M)->appendModuleInlineAsm(StringRef(Asm));
}

typedef DIBuilder *LLVMRustDIBuilderRef;

#if LLVM_VERSION_LT(5, 0)
typedef struct LLVMOpaqueMetadata *LLVMMetadataRef;

namespace llvm {
DEFINE_ISA_CONVERSION_FUNCTIONS(Metadata, LLVMMetadataRef)

inline Metadata **unwrap(LLVMMetadataRef *Vals) {
  return reinterpret_cast<Metadata **>(Vals);
}
}
#endif

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
  FlagMainSubprogram = (1 << 21),
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

#if LLVM_VERSION_GE(4, 0)
static DINode::DIFlags fromRust(LLVMRustDIFlags Flags) {
  DINode::DIFlags Result = DINode::DIFlags::FlagZero;
#else
static unsigned fromRust(LLVMRustDIFlags Flags) {
  unsigned Result = 0;
#endif

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
  if (isSet(Flags & LLVMRustDIFlags::FlagBlockByrefStruct)) {
    Result |= DINode::DIFlags::FlagBlockByrefStruct;
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
#if LLVM_VERSION_LE(4, 0)
  if (isSet(Flags & LLVMRustDIFlags::FlagExternalTypeRef)) {
    Result |= DINode::DIFlags::FlagExternalTypeRef;
  }
#endif
  if (isSet(Flags & LLVMRustDIFlags::FlagIntroducedVirtual)) {
    Result |= DINode::DIFlags::FlagIntroducedVirtual;
  }
  if (isSet(Flags & LLVMRustDIFlags::FlagBitField)) {
    Result |= DINode::DIFlags::FlagBitField;
  }
#if LLVM_RUSTLLVM || LLVM_VERSION_GE(4, 0)
  if (isSet(Flags & LLVMRustDIFlags::FlagNoReturn)) {
    Result |= DINode::DIFlags::FlagNoReturn;
  }
  if (isSet(Flags & LLVMRustDIFlags::FlagMainSubprogram)) {
    Result |= DINode::DIFlags::FlagMainSubprogram;
  }
#endif

  return Result;
}

extern "C" uint32_t LLVMRustDebugMetadataVersion() {
  return DEBUG_METADATA_VERSION;
}

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
    const char *Producer, bool isOptimized, const char *Flags,
    unsigned RuntimeVer, const char *SplitName) {
  auto *File = unwrapDI<DIFile>(FileRef);

#if LLVM_VERSION_GE(4, 0)
  return wrap(Builder->createCompileUnit(Lang, File, Producer, isOptimized,
                                         Flags, RuntimeVer, SplitName));
#else
  return wrap(Builder->createCompileUnit(Lang, File->getFilename(),
      File->getDirectory(), Producer, isOptimized,
      Flags, RuntimeVer, SplitName));
#endif
}

extern "C" LLVMMetadataRef
LLVMRustDIBuilderCreateFile(LLVMRustDIBuilderRef Builder, const char *Filename,
                            const char *Directory) {
  return wrap(Builder->createFile(Filename, Directory));
}

extern "C" LLVMMetadataRef
LLVMRustDIBuilderCreateSubroutineType(LLVMRustDIBuilderRef Builder,
                                      LLVMMetadataRef File,
                                      LLVMMetadataRef ParameterTypes) {
  return wrap(Builder->createSubroutineType(
      DITypeRefArray(unwrap<MDTuple>(ParameterTypes))));
}

extern "C" LLVMMetadataRef LLVMRustDIBuilderCreateFunction(
    LLVMRustDIBuilderRef Builder, LLVMMetadataRef Scope, const char *Name,
    const char *LinkageName, LLVMMetadataRef File, unsigned LineNo,
    LLVMMetadataRef Ty, bool IsLocalToUnit, bool IsDefinition,
    unsigned ScopeLine, LLVMRustDIFlags Flags, bool IsOptimized,
    LLVMValueRef Fn, LLVMMetadataRef TParam, LLVMMetadataRef Decl) {
  DITemplateParameterArray TParams =
      DITemplateParameterArray(unwrap<MDTuple>(TParam));
  DISubprogram *Sub = Builder->createFunction(
      unwrapDI<DIScope>(Scope), Name, LinkageName, unwrapDI<DIFile>(File),
      LineNo, unwrapDI<DISubroutineType>(Ty), IsLocalToUnit, IsDefinition,
      ScopeLine, fromRust(Flags), IsOptimized, TParams,
      unwrapDIPtr<DISubprogram>(Decl));
  unwrap<Function>(Fn)->setSubprogram(Sub);
  return wrap(Sub);
}

extern "C" LLVMMetadataRef
LLVMRustDIBuilderCreateBasicType(LLVMRustDIBuilderRef Builder, const char *Name,
                                 uint64_t SizeInBits, uint32_t AlignInBits,
                                 unsigned Encoding) {
  return wrap(Builder->createBasicType(Name, SizeInBits,
#if LLVM_VERSION_LE(3, 9)
                                       AlignInBits,
#endif
                                       Encoding));
}

extern "C" LLVMMetadataRef LLVMRustDIBuilderCreatePointerType(
    LLVMRustDIBuilderRef Builder, LLVMMetadataRef PointeeTy,
    uint64_t SizeInBits, uint32_t AlignInBits, const char *Name) {
  return wrap(Builder->createPointerType(unwrapDI<DIType>(PointeeTy),
                                         SizeInBits, AlignInBits,
#if LLVM_VERSION_GE(5, 0)
                                         /* DWARFAddressSpace */ None,
#endif
                                         Name));
}

extern "C" LLVMMetadataRef LLVMRustDIBuilderCreateStructType(
    LLVMRustDIBuilderRef Builder, LLVMMetadataRef Scope, const char *Name,
    LLVMMetadataRef File, unsigned LineNumber, uint64_t SizeInBits,
    uint32_t AlignInBits, LLVMRustDIFlags Flags,
    LLVMMetadataRef DerivedFrom, LLVMMetadataRef Elements,
    unsigned RunTimeLang, LLVMMetadataRef VTableHolder,
    const char *UniqueId) {
  return wrap(Builder->createStructType(
      unwrapDI<DIDescriptor>(Scope), Name, unwrapDI<DIFile>(File), LineNumber,
      SizeInBits, AlignInBits, fromRust(Flags), unwrapDI<DIType>(DerivedFrom),
      DINodeArray(unwrapDI<MDTuple>(Elements)), RunTimeLang,
      unwrapDI<DIType>(VTableHolder), UniqueId));
}

extern "C" LLVMMetadataRef LLVMRustDIBuilderCreateMemberType(
    LLVMRustDIBuilderRef Builder, LLVMMetadataRef Scope, const char *Name,
    LLVMMetadataRef File, unsigned LineNo, uint64_t SizeInBits,
    uint32_t AlignInBits, uint64_t OffsetInBits, LLVMRustDIFlags Flags,
    LLVMMetadataRef Ty) {
  return wrap(Builder->createMemberType(unwrapDI<DIDescriptor>(Scope), Name,
                                        unwrapDI<DIFile>(File), LineNo,
                                        SizeInBits, AlignInBits, OffsetInBits,
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
    LLVMRustDIBuilderRef Builder, LLVMMetadataRef Context, const char *Name,
    const char *LinkageName, LLVMMetadataRef File, unsigned LineNo,
    LLVMMetadataRef Ty, bool IsLocalToUnit, LLVMValueRef V,
    LLVMMetadataRef Decl = nullptr, uint32_t AlignInBits = 0) {
  llvm::GlobalVariable *InitVal = cast<llvm::GlobalVariable>(unwrap(V));

#if LLVM_VERSION_GE(4, 0)
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
      unwrapDI<DIDescriptor>(Context), Name, LinkageName,
      unwrapDI<DIFile>(File), LineNo, unwrapDI<DIType>(Ty), IsLocalToUnit,
      InitExpr, unwrapDIPtr<MDNode>(Decl), AlignInBits);

  InitVal->setMetadata("dbg", VarExpr);

  return wrap(VarExpr);
#else
  return wrap(Builder->createGlobalVariable(
      unwrapDI<DIDescriptor>(Context), Name, LinkageName,
      unwrapDI<DIFile>(File), LineNo, unwrapDI<DIType>(Ty), IsLocalToUnit,
      InitVal, unwrapDIPtr<MDNode>(Decl)));
#endif
}

extern "C" LLVMMetadataRef LLVMRustDIBuilderCreateVariable(
    LLVMRustDIBuilderRef Builder, unsigned Tag, LLVMMetadataRef Scope,
    const char *Name, LLVMMetadataRef File, unsigned LineNo,
    LLVMMetadataRef Ty, bool AlwaysPreserve, LLVMRustDIFlags Flags,
    unsigned ArgNo, uint32_t AlignInBits) {
  if (Tag == 0x100) { // DW_TAG_auto_variable
    return wrap(Builder->createAutoVariable(
        unwrapDI<DIDescriptor>(Scope), Name, unwrapDI<DIFile>(File), LineNo,
        unwrapDI<DIType>(Ty), AlwaysPreserve, fromRust(Flags)
#if LLVM_VERSION_GE(4, 0)
        ,
  AlignInBits
#endif
        ));
  } else {
    return wrap(Builder->createParameterVariable(
        unwrapDI<DIDescriptor>(Scope), Name, ArgNo, unwrapDI<DIFile>(File),
        LineNo, unwrapDI<DIType>(Ty), AlwaysPreserve, fromRust(Flags)));
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
LLVMRustDIBuilderCreateVectorType(LLVMRustDIBuilderRef Builder, uint64_t Size,
                                  uint32_t AlignInBits, LLVMMetadataRef Ty,
                                  LLVMMetadataRef Subscripts) {
  return wrap(
      Builder->createVectorType(Size, AlignInBits, unwrapDI<DIType>(Ty),
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
    int64_t *AddrOps, unsigned AddrOpsCount, LLVMValueRef DL,
    LLVMBasicBlockRef InsertAtEnd) {
  return wrap(Builder->insertDeclare(
      unwrap(V), unwrap<DILocalVariable>(VarInfo),
      Builder->createExpression(llvm::ArrayRef<int64_t>(AddrOps, AddrOpsCount)),
      DebugLoc(cast<MDNode>(unwrap<MetadataAsValue>(DL)->getMetadata())),
      unwrap(InsertAtEnd)));
}

extern "C" LLVMMetadataRef
LLVMRustDIBuilderCreateEnumerator(LLVMRustDIBuilderRef Builder,
                                  const char *Name, uint64_t Val) {
  return wrap(Builder->createEnumerator(Name, Val));
}

extern "C" LLVMMetadataRef LLVMRustDIBuilderCreateEnumerationType(
    LLVMRustDIBuilderRef Builder, LLVMMetadataRef Scope, const char *Name,
    LLVMMetadataRef File, unsigned LineNumber, uint64_t SizeInBits,
    uint32_t AlignInBits, LLVMMetadataRef Elements,
    LLVMMetadataRef ClassTy) {
  return wrap(Builder->createEnumerationType(
      unwrapDI<DIDescriptor>(Scope), Name, unwrapDI<DIFile>(File), LineNumber,
      SizeInBits, AlignInBits, DINodeArray(unwrapDI<MDTuple>(Elements)),
      unwrapDI<DIType>(ClassTy)));
}

extern "C" LLVMMetadataRef LLVMRustDIBuilderCreateUnionType(
    LLVMRustDIBuilderRef Builder, LLVMMetadataRef Scope, const char *Name,
    LLVMMetadataRef File, unsigned LineNumber, uint64_t SizeInBits,
    uint32_t AlignInBits, LLVMRustDIFlags Flags, LLVMMetadataRef Elements,
    unsigned RunTimeLang, const char *UniqueId) {
  return wrap(Builder->createUnionType(
      unwrapDI<DIDescriptor>(Scope), Name, unwrapDI<DIFile>(File), LineNumber,
      SizeInBits, AlignInBits, fromRust(Flags),
      DINodeArray(unwrapDI<MDTuple>(Elements)), RunTimeLang, UniqueId));
}

extern "C" LLVMMetadataRef LLVMRustDIBuilderCreateTemplateTypeParameter(
    LLVMRustDIBuilderRef Builder, LLVMMetadataRef Scope, const char *Name,
    LLVMMetadataRef Ty, LLVMMetadataRef File, unsigned LineNo,
    unsigned ColumnNo) {
  return wrap(Builder->createTemplateTypeParameter(
      unwrapDI<DIDescriptor>(Scope), Name, unwrapDI<DIType>(Ty)));
}

extern "C" LLVMMetadataRef
LLVMRustDIBuilderCreateNameSpace(LLVMRustDIBuilderRef Builder,
                                 LLVMMetadataRef Scope, const char *Name,
                                 LLVMMetadataRef File, unsigned LineNo) {
  return wrap(Builder->createNameSpace(
      unwrapDI<DIDescriptor>(Scope), Name
#if LLVM_VERSION_LT(5, 0)
      ,
      unwrapDI<DIFile>(File), LineNo
#endif
#if LLVM_VERSION_GE(4, 0)
      ,
      false // ExportSymbols (only relevant for C++ anonymous namespaces)
#endif
      ));
}

extern "C" void
LLVMRustDICompositeTypeSetTypeArray(LLVMRustDIBuilderRef Builder,
                                    LLVMMetadataRef CompositeTy,
                                    LLVMMetadataRef TyArray) {
  DICompositeType *Tmp = unwrapDI<DICompositeType>(CompositeTy);
  Builder->replaceArrays(Tmp, DINodeArray(unwrap<MDTuple>(TyArray)));
}

extern "C" LLVMValueRef
LLVMRustDIBuilderCreateDebugLocation(LLVMContextRef ContextRef, unsigned Line,
                                     unsigned Column, LLVMMetadataRef Scope,
                                     LLVMMetadataRef InlinedAt) {
  LLVMContext &Context = *unwrap(ContextRef);

  DebugLoc debug_loc = DebugLoc::get(Line, Column, unwrapDIPtr<MDNode>(Scope),
                                     unwrapDIPtr<MDNode>(InlinedAt));

  return wrap(MetadataAsValue::get(Context, debug_loc.getAsMDNode()));
}

extern "C" int64_t LLVMRustDIBuilderCreateOpDeref() {
  return dwarf::DW_OP_deref;
}

extern "C" int64_t LLVMRustDIBuilderCreateOpPlusUconst() {
#if LLVM_VERSION_GE(5, 0)
  return dwarf::DW_OP_plus_uconst;
#else
  // older LLVM used `plus` to behave like `plus_uconst`.
  return dwarf::DW_OP_plus;
#endif
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

extern "C" bool LLVMRustLinkInExternalBitcode(LLVMModuleRef DstRef, char *BC,
                                              size_t Len) {
  Module *Dst = unwrap(DstRef);

  std::unique_ptr<MemoryBuffer> Buf =
      MemoryBuffer::getMemBufferCopy(StringRef(BC, Len));

#if LLVM_VERSION_GE(4, 0)
  Expected<std::unique_ptr<Module>> SrcOrError =
      llvm::getLazyBitcodeModule(Buf->getMemBufferRef(), Dst->getContext());
  if (!SrcOrError) {
    LLVMRustSetLastError(toString(SrcOrError.takeError()).c_str());
    return false;
  }

  auto Src = std::move(*SrcOrError);
#else
  ErrorOr<std::unique_ptr<Module>> Src =
      llvm::getLazyBitcodeModule(std::move(Buf), Dst->getContext());
  if (!Src) {
    LLVMRustSetLastError(Src.getError().message().c_str());
    return false;
  }
#endif

  std::string Err;

  raw_string_ostream Stream(Err);
  DiagnosticPrinterRawOStream DP(Stream);
#if LLVM_VERSION_GE(4, 0)
  if (Linker::linkModules(*Dst, std::move(Src))) {
#else
  if (Linker::linkModules(*Dst, std::move(Src.get()))) {
#endif
    LLVMRustSetLastError(Err.c_str());
    return false;
  }
  return true;
}

// Note that the two following functions look quite similar to the
// LLVMGetSectionName function. Sadly, it appears that this function only
// returns a char* pointer, which isn't guaranteed to be null-terminated. The
// function provided by LLVM doesn't return the length, so we've created our own
// function which returns the length as well as the data pointer.
//
// For an example of this not returning a null terminated string, see
// lib/Object/COFFObjectFile.cpp in the getSectionName function. One of the
// branches explicitly creates a StringRef without a null terminator, and then
// that's returned.

inline section_iterator *unwrap(LLVMSectionIteratorRef SI) {
  return reinterpret_cast<section_iterator *>(SI);
}

extern "C" size_t LLVMRustGetSectionName(LLVMSectionIteratorRef SI,
                                         const char **Ptr) {
  StringRef Ret;
  if (std::error_code EC = (*unwrap(SI))->getName(Ret))
    report_fatal_error(EC.message());
  *Ptr = Ret.data();
  return Ret.size();
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
#if LLVM_VERSION_GE(5,0)
  DiagnosticLocation loc = Opt->getLocation();
  if (loc.isValid()) {
    *Line = loc.getLine();
    *Column = loc.getColumn();
    FilenameOS << loc.getFilename();
  }
#else
  const DebugLoc &loc = Opt->getDebugLoc();
  if (loc) {
    *Line = loc.getLine();
    *Column = loc.getCol();
    FilenameOS << cast<DIScope>(loc.getScope())->getFilename();
  }
#endif

  RawRustStringOstream MessageOS(MessageOut);
  MessageOS << Opt->getMsg();
}

extern "C" void
LLVMRustUnpackInlineAsmDiagnostic(LLVMDiagnosticInfoRef DI, unsigned *CookieOut,
                                  LLVMTwineRef *MessageOut,
                                  LLVMValueRef *InstructionOut) {
  // Undefined to call this not on an inline assembly diagnostic!
  llvm::DiagnosticInfoInlineAsm *IA =
      static_cast<llvm::DiagnosticInfoInlineAsm *>(unwrap(DI));

  *CookieOut = IA->getLocCookie();
  *MessageOut = wrap(&IA->getMsgStr());
  *InstructionOut = wrap(IA->getInstruction());
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
  case Type::VectorTyID:
    return LLVMVectorTypeKind;
  case Type::X86_MMXTyID:
    return LLVMX86_MMXTypeKind;
  case Type::TokenTyID:
    return LLVMTokenTypeKind;
  }
  report_fatal_error("Unhandled TypeID.");
}

DEFINE_SIMPLE_CONVERSION_FUNCTIONS(SMDiagnostic, LLVMSMDiagnosticRef)

extern "C" void LLVMRustSetInlineAsmDiagnosticHandler(
    LLVMContextRef C, LLVMContext::InlineAsmDiagHandlerTy H, void *CX) {
  unwrap(C)->setInlineAsmDiagnosticHandler(H, CX);
}

extern "C" void LLVMRustWriteSMDiagnosticToString(LLVMSMDiagnosticRef D,
                                                  RustStringRef Str) {
  RawRustStringOstream OS(Str);
  unwrap(D)->print("", OS);
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

extern "C" LLVMValueRef LLVMRustBuildCall(LLVMBuilderRef B, LLVMValueRef Fn,
                                          LLVMValueRef *Args, unsigned NumArgs,
                                          OperandBundleDef *Bundle,
                                          const char *Name) {
  unsigned Len = Bundle ? 1 : 0;
  ArrayRef<OperandBundleDef> Bundles = makeArrayRef(Bundle, Len);
  return wrap(unwrap(B)->CreateCall(
      unwrap(Fn), makeArrayRef(unwrap(Args), NumArgs), Bundles, Name));
}

extern "C" LLVMValueRef
LLVMRustBuildInvoke(LLVMBuilderRef B, LLVMValueRef Fn, LLVMValueRef *Args,
                    unsigned NumArgs, LLVMBasicBlockRef Then,
                    LLVMBasicBlockRef Catch, OperandBundleDef *Bundle,
                    const char *Name) {
  unsigned Len = Bundle ? 1 : 0;
  ArrayRef<OperandBundleDef> Bundles = makeArrayRef(Bundle, Len);
  return wrap(unwrap(B)->CreateInvoke(unwrap(Fn), unwrap(Then), unwrap(Catch),
                                      makeArrayRef(unwrap(Args), NumArgs),
                                      Bundles, Name));
}

extern "C" void LLVMRustPositionBuilderAtStart(LLVMBuilderRef B,
                                               LLVMBasicBlockRef BB) {
  auto Point = unwrap(BB)->getFirstInsertionPt();
  unwrap(B)->SetInsertPoint(unwrap(BB), Point);
}

extern "C" void LLVMRustSetComdat(LLVMModuleRef M, LLVMValueRef V,
                                  const char *Name) {
  Triple TargetTriple(unwrap(M)->getTargetTriple());
  GlobalObject *GV = unwrap<GlobalObject>(V);
  if (!TargetTriple.isOSBinFormatMachO()) {
    GV->setComdat(unwrap(M)->getOrInsertComdat(Name));
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

struct LLVMRustModuleBuffer {
  std::string data;
};

extern "C" LLVMRustModuleBuffer*
LLVMRustModuleBufferCreate(LLVMModuleRef M) {
  auto Ret = llvm::make_unique<LLVMRustModuleBuffer>();
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
