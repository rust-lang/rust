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
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/ObjectFile.h"

#include "llvm/IR/CallSite.h"

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

  llvm_unreachable("Invalid LLVMAtomicOrdering value!");
}

static char *LastError;

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

void LLVMRustSetLastError(const char *Err) {
  free((void *)LastError);
  LastError = strdup(Err);
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
  }
  llvm_unreachable("bad AttributeKind");
}

extern "C" void LLVMRustAddCallSiteAttribute(LLVMValueRef Instr, unsigned Index,
                                             LLVMRustAttribute RustAttr) {
  CallSite Call = CallSite(unwrap<Instruction>(Instr));
  Attribute Attr = Attribute::get(Call->getContext(), fromRust(RustAttr));
  AttrBuilder B(Attr);
  Call.setAttributes(Call.getAttributes().addAttributes(
      Call->getContext(), Index,
      AttributeSet::get(Call->getContext(), Index, B)));
}

extern "C" void LLVMRustAddDereferenceableCallSiteAttr(LLVMValueRef Instr,
                                                       unsigned Index,
                                                       uint64_t Bytes) {
  CallSite Call = CallSite(unwrap<Instruction>(Instr));
  AttrBuilder B;
  B.addDereferenceableAttr(Bytes);
  Call.setAttributes(Call.getAttributes().addAttributes(
      Call->getContext(), Index,
      AttributeSet::get(Call->getContext(), Index, B)));
}

extern "C" void LLVMRustAddFunctionAttribute(LLVMValueRef Fn, unsigned Index,
                                             LLVMRustAttribute RustAttr) {
  Function *A = unwrap<Function>(Fn);
  Attribute Attr = Attribute::get(A->getContext(), fromRust(RustAttr));
  AttrBuilder B(Attr);
  A->addAttributes(Index, AttributeSet::get(A->getContext(), Index, B));
}

extern "C" void LLVMRustAddDereferenceableAttr(LLVMValueRef Fn, unsigned Index,
                                               uint64_t Bytes) {
  Function *A = unwrap<Function>(Fn);
  AttrBuilder B;
  B.addDereferenceableAttr(Bytes);
  A->addAttributes(Index, AttributeSet::get(A->getContext(), Index, B));
}

extern "C" void LLVMRustAddFunctionAttrStringValue(LLVMValueRef Fn,
                                                   unsigned Index,
                                                   const char *Name,
                                                   const char *Value) {
  Function *F = unwrap<Function>(Fn);
  AttrBuilder B;
  B.addAttribute(Name, Value);
  F->addAttributes(Index, AttributeSet::get(F->getContext(), Index, B));
}

extern "C" void LLVMRustRemoveFunctionAttributes(LLVMValueRef Fn,
                                                 unsigned Index,
                                                 LLVMRustAttribute RustAttr) {
  Function *F = unwrap<Function>(Fn);
  const AttributeSet PAL = F->getAttributes();
  Attribute Attr = Attribute::get(F->getContext(), fromRust(RustAttr));
  AttrBuilder B(Attr);
  const AttributeSet PALNew = PAL.removeAttributes(
      F->getContext(), Index, AttributeSet::get(F->getContext(), Index, B));
  F->setAttributes(PALNew);
}

// enable fpmath flag UnsafeAlgebra
extern "C" void LLVMRustSetHasUnsafeAlgebra(LLVMValueRef V) {
  if (auto I = dyn_cast<Instruction>(unwrap<Value>(V))) {
    I->setHasUnsafeAlgebra(true);
  }
}

extern "C" LLVMValueRef
LLVMRustBuildAtomicLoad(LLVMBuilderRef B, LLVMValueRef Source, const char *Name,
                        LLVMAtomicOrdering Order, unsigned Alignment) {
  LoadInst *LI = new LoadInst(unwrap(Source), 0);
  LI->setAtomic(fromRust(Order));
  LI->setAlignment(Alignment);
  return wrap(unwrap(B)->Insert(LI, Name));
}

extern "C" LLVMValueRef LLVMRustBuildAtomicStore(LLVMBuilderRef B,
                                                 LLVMValueRef V,
                                                 LLVMValueRef Target,
                                                 LLVMAtomicOrdering Order,
                                                 unsigned Alignment) {
  StoreInst *SI = new StoreInst(unwrap(V), unwrap(Target));
  SI->setAtomic(fromRust(Order));
  SI->setAlignment(Alignment);
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

static SynchronizationScope fromRust(LLVMRustSynchronizationScope Scope) {
  switch (Scope) {
  case LLVMRustSynchronizationScope::SingleThread:
    return SingleThread;
  case LLVMRustSynchronizationScope::CrossThread:
    return CrossThread;
  default:
    llvm_unreachable("bad SynchronizationScope.");
  }
}

extern "C" LLVMValueRef
LLVMRustBuildAtomicFence(LLVMBuilderRef B, LLVMAtomicOrdering Order,
                         LLVMRustSynchronizationScope Scope) {
  return wrap(unwrap(B)->CreateFence(fromRust(Order), fromRust(Scope)));
}

extern "C" void LLVMRustSetDebug(int Enabled) {
#ifndef NDEBUG
  DebugFlag = Enabled;
#endif
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
    llvm_unreachable("bad AsmDialect.");
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

typedef DIBuilder *LLVMRustDIBuilderRef;

typedef struct LLVMOpaqueMetadata *LLVMRustMetadataRef;

namespace llvm {
DEFINE_ISA_CONVERSION_FUNCTIONS(Metadata, LLVMRustMetadataRef)

inline Metadata **unwrap(LLVMRustMetadataRef *Vals) {
  return reinterpret_cast<Metadata **>(Vals);
}
}

template <typename DIT> DIT *unwrapDIPtr(LLVMRustMetadataRef Ref) {
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
  // Do not add values that are not supported by the minimum LLVM
  // version we support!
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

extern "C" LLVMRustDIBuilderRef LLVMRustDIBuilderCreate(LLVMModuleRef M) {
  return new DIBuilder(*unwrap(M));
}

extern "C" void LLVMRustDIBuilderDispose(LLVMRustDIBuilderRef Builder) {
  delete Builder;
}

extern "C" void LLVMRustDIBuilderFinalize(LLVMRustDIBuilderRef Builder) {
  Builder->finalize();
}

extern "C" LLVMRustMetadataRef LLVMRustDIBuilderCreateCompileUnit(
    LLVMRustDIBuilderRef Builder, unsigned Lang, const char *File,
    const char *Dir, const char *Producer, bool isOptimized, const char *Flags,
    unsigned RuntimeVer, const char *SplitName) {
  return wrap(Builder->createCompileUnit(Lang, File, Dir, Producer, isOptimized,
                                         Flags, RuntimeVer, SplitName));
}

extern "C" LLVMRustMetadataRef
LLVMRustDIBuilderCreateFile(LLVMRustDIBuilderRef Builder, const char *Filename,
                            const char *Directory) {
  return wrap(Builder->createFile(Filename, Directory));
}

extern "C" LLVMRustMetadataRef
LLVMRustDIBuilderCreateSubroutineType(LLVMRustDIBuilderRef Builder,
                                      LLVMRustMetadataRef File,
                                      LLVMRustMetadataRef ParameterTypes) {
  return wrap(Builder->createSubroutineType(
#if LLVM_VERSION_EQ(3, 7)
      unwrapDI<DIFile>(File),
#endif
      DITypeRefArray(unwrap<MDTuple>(ParameterTypes))));
}

extern "C" LLVMRustMetadataRef LLVMRustDIBuilderCreateFunction(
    LLVMRustDIBuilderRef Builder, LLVMRustMetadataRef Scope, const char *Name,
    const char *LinkageName, LLVMRustMetadataRef File, unsigned LineNo,
    LLVMRustMetadataRef Ty, bool IsLocalToUnit, bool IsDefinition,
    unsigned ScopeLine, LLVMRustDIFlags Flags, bool IsOptimized,
    LLVMValueRef Fn, LLVMRustMetadataRef TParam, LLVMRustMetadataRef Decl) {
#if LLVM_VERSION_GE(3, 8)
  DITemplateParameterArray TParams =
      DITemplateParameterArray(unwrap<MDTuple>(TParam));
  DISubprogram *Sub = Builder->createFunction(
      unwrapDI<DIScope>(Scope), Name, LinkageName, unwrapDI<DIFile>(File),
      LineNo, unwrapDI<DISubroutineType>(Ty), IsLocalToUnit, IsDefinition,
      ScopeLine, fromRust(Flags), IsOptimized, TParams,
      unwrapDIPtr<DISubprogram>(Decl));
  unwrap<Function>(Fn)->setSubprogram(Sub);
  return wrap(Sub);
#else
  return wrap(Builder->createFunction(
      unwrapDI<DIScope>(Scope), Name, LinkageName, unwrapDI<DIFile>(File),
      LineNo, unwrapDI<DISubroutineType>(Ty), IsLocalToUnit, IsDefinition,
      ScopeLine, fromRust(Flags), IsOptimized, unwrap<Function>(Fn),
      unwrapDIPtr<MDNode>(TParam), unwrapDIPtr<MDNode>(Decl)));
#endif
}

extern "C" LLVMRustMetadataRef
LLVMRustDIBuilderCreateBasicType(LLVMRustDIBuilderRef Builder, const char *Name,
                                 uint64_t SizeInBits, uint64_t AlignInBits,
                                 unsigned Encoding) {
  return wrap(Builder->createBasicType(Name, SizeInBits,
#if LLVM_VERSION_LE(3, 9)
                                       AlignInBits,
#endif
                                       Encoding));
}

extern "C" LLVMRustMetadataRef LLVMRustDIBuilderCreatePointerType(
    LLVMRustDIBuilderRef Builder, LLVMRustMetadataRef PointeeTy,
    uint64_t SizeInBits, uint64_t AlignInBits, const char *Name) {
  return wrap(Builder->createPointerType(unwrapDI<DIType>(PointeeTy),
                                         SizeInBits, AlignInBits, Name));
}

extern "C" LLVMRustMetadataRef LLVMRustDIBuilderCreateStructType(
    LLVMRustDIBuilderRef Builder, LLVMRustMetadataRef Scope, const char *Name,
    LLVMRustMetadataRef File, unsigned LineNumber, uint64_t SizeInBits,
    uint64_t AlignInBits, LLVMRustDIFlags Flags,
    LLVMRustMetadataRef DerivedFrom, LLVMRustMetadataRef Elements,
    unsigned RunTimeLang, LLVMRustMetadataRef VTableHolder,
    const char *UniqueId) {
  return wrap(Builder->createStructType(
      unwrapDI<DIDescriptor>(Scope), Name, unwrapDI<DIFile>(File), LineNumber,
      SizeInBits, AlignInBits, fromRust(Flags), unwrapDI<DIType>(DerivedFrom),
      DINodeArray(unwrapDI<MDTuple>(Elements)), RunTimeLang,
      unwrapDI<DIType>(VTableHolder), UniqueId));
}

extern "C" LLVMRustMetadataRef LLVMRustDIBuilderCreateMemberType(
    LLVMRustDIBuilderRef Builder, LLVMRustMetadataRef Scope, const char *Name,
    LLVMRustMetadataRef File, unsigned LineNo, uint64_t SizeInBits,
    uint64_t AlignInBits, uint64_t OffsetInBits, LLVMRustDIFlags Flags,
    LLVMRustMetadataRef Ty) {
  return wrap(Builder->createMemberType(unwrapDI<DIDescriptor>(Scope), Name,
                                        unwrapDI<DIFile>(File), LineNo,
                                        SizeInBits, AlignInBits, OffsetInBits,
                                        fromRust(Flags), unwrapDI<DIType>(Ty)));
}

extern "C" LLVMRustMetadataRef LLVMRustDIBuilderCreateLexicalBlock(
    LLVMRustDIBuilderRef Builder, LLVMRustMetadataRef Scope,
    LLVMRustMetadataRef File, unsigned Line, unsigned Col) {
  return wrap(Builder->createLexicalBlock(unwrapDI<DIDescriptor>(Scope),
                                          unwrapDI<DIFile>(File), Line, Col));
}

extern "C" LLVMRustMetadataRef
LLVMRustDIBuilderCreateLexicalBlockFile(LLVMRustDIBuilderRef Builder,
                                        LLVMRustMetadataRef Scope,
                                        LLVMRustMetadataRef File) {
  return wrap(Builder->createLexicalBlockFile(unwrapDI<DIDescriptor>(Scope),
                                              unwrapDI<DIFile>(File)));
}

extern "C" LLVMRustMetadataRef LLVMRustDIBuilderCreateStaticVariable(
    LLVMRustDIBuilderRef Builder, LLVMRustMetadataRef Context, const char *Name,
    const char *LinkageName, LLVMRustMetadataRef File, unsigned LineNo,
    LLVMRustMetadataRef Ty, bool IsLocalToUnit, LLVMValueRef V,
    LLVMRustMetadataRef Decl = nullptr, uint64_t AlignInBits = 0) {
  Constant *InitVal = cast<Constant>(unwrap(V));

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
#endif

  return wrap(Builder->createGlobalVariable(
      unwrapDI<DIDescriptor>(Context), Name, LinkageName,
      unwrapDI<DIFile>(File), LineNo, unwrapDI<DIType>(Ty), IsLocalToUnit,
#if LLVM_VERSION_GE(4, 0)
      InitExpr,
#else
      InitVal,
#endif
      unwrapDIPtr<MDNode>(Decl)
#if LLVM_VERSION_GE(4, 0)
      ,
      AlignInBits
#endif
      ));
}

extern "C" LLVMRustMetadataRef LLVMRustDIBuilderCreateVariable(
    LLVMRustDIBuilderRef Builder, unsigned Tag, LLVMRustMetadataRef Scope,
    const char *Name, LLVMRustMetadataRef File, unsigned LineNo,
    LLVMRustMetadataRef Ty, bool AlwaysPreserve, LLVMRustDIFlags Flags,
    unsigned ArgNo, uint64_t AlignInBits) {
#if LLVM_VERSION_GE(3, 8)
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
#else
  return wrap(Builder->createLocalVariable(
      Tag, unwrapDI<DIDescriptor>(Scope), Name, unwrapDI<DIFile>(File), LineNo,
      unwrapDI<DIType>(Ty), AlwaysPreserve, fromRust(Flags), ArgNo));
#endif
}

extern "C" LLVMRustMetadataRef
LLVMRustDIBuilderCreateArrayType(LLVMRustDIBuilderRef Builder, uint64_t Size,
                                 uint64_t AlignInBits, LLVMRustMetadataRef Ty,
                                 LLVMRustMetadataRef Subscripts) {
  return wrap(
      Builder->createArrayType(Size, AlignInBits, unwrapDI<DIType>(Ty),
                               DINodeArray(unwrapDI<MDTuple>(Subscripts))));
}

extern "C" LLVMRustMetadataRef
LLVMRustDIBuilderCreateVectorType(LLVMRustDIBuilderRef Builder, uint64_t Size,
                                  uint64_t AlignInBits, LLVMRustMetadataRef Ty,
                                  LLVMRustMetadataRef Subscripts) {
  return wrap(
      Builder->createVectorType(Size, AlignInBits, unwrapDI<DIType>(Ty),
                                DINodeArray(unwrapDI<MDTuple>(Subscripts))));
}

extern "C" LLVMRustMetadataRef
LLVMRustDIBuilderGetOrCreateSubrange(LLVMRustDIBuilderRef Builder, int64_t Lo,
                                     int64_t Count) {
  return wrap(Builder->getOrCreateSubrange(Lo, Count));
}

extern "C" LLVMRustMetadataRef
LLVMRustDIBuilderGetOrCreateArray(LLVMRustDIBuilderRef Builder,
                                  LLVMRustMetadataRef *Ptr, unsigned Count) {
  Metadata **DataValue = unwrap(Ptr);
  return wrap(
      Builder->getOrCreateArray(ArrayRef<Metadata *>(DataValue, Count)).get());
}

extern "C" LLVMValueRef LLVMRustDIBuilderInsertDeclareAtEnd(
    LLVMRustDIBuilderRef Builder, LLVMValueRef V, LLVMRustMetadataRef VarInfo,
    int64_t *AddrOps, unsigned AddrOpsCount, LLVMValueRef DL,
    LLVMBasicBlockRef InsertAtEnd) {
  return wrap(Builder->insertDeclare(
      unwrap(V), unwrap<DILocalVariable>(VarInfo),
      Builder->createExpression(llvm::ArrayRef<int64_t>(AddrOps, AddrOpsCount)),
      DebugLoc(cast<MDNode>(unwrap<MetadataAsValue>(DL)->getMetadata())),
      unwrap(InsertAtEnd)));
}

extern "C" LLVMRustMetadataRef
LLVMRustDIBuilderCreateEnumerator(LLVMRustDIBuilderRef Builder,
                                  const char *Name, uint64_t Val) {
  return wrap(Builder->createEnumerator(Name, Val));
}

extern "C" LLVMRustMetadataRef LLVMRustDIBuilderCreateEnumerationType(
    LLVMRustDIBuilderRef Builder, LLVMRustMetadataRef Scope, const char *Name,
    LLVMRustMetadataRef File, unsigned LineNumber, uint64_t SizeInBits,
    uint64_t AlignInBits, LLVMRustMetadataRef Elements,
    LLVMRustMetadataRef ClassTy) {
  return wrap(Builder->createEnumerationType(
      unwrapDI<DIDescriptor>(Scope), Name, unwrapDI<DIFile>(File), LineNumber,
      SizeInBits, AlignInBits, DINodeArray(unwrapDI<MDTuple>(Elements)),
      unwrapDI<DIType>(ClassTy)));
}

extern "C" LLVMRustMetadataRef LLVMRustDIBuilderCreateUnionType(
    LLVMRustDIBuilderRef Builder, LLVMRustMetadataRef Scope, const char *Name,
    LLVMRustMetadataRef File, unsigned LineNumber, uint64_t SizeInBits,
    uint64_t AlignInBits, LLVMRustDIFlags Flags, LLVMRustMetadataRef Elements,
    unsigned RunTimeLang, const char *UniqueId) {
  return wrap(Builder->createUnionType(
      unwrapDI<DIDescriptor>(Scope), Name, unwrapDI<DIFile>(File), LineNumber,
      SizeInBits, AlignInBits, fromRust(Flags),
      DINodeArray(unwrapDI<MDTuple>(Elements)), RunTimeLang, UniqueId));
}

extern "C" LLVMRustMetadataRef LLVMRustDIBuilderCreateTemplateTypeParameter(
    LLVMRustDIBuilderRef Builder, LLVMRustMetadataRef Scope, const char *Name,
    LLVMRustMetadataRef Ty, LLVMRustMetadataRef File, unsigned LineNo,
    unsigned ColumnNo) {
  return wrap(Builder->createTemplateTypeParameter(
      unwrapDI<DIDescriptor>(Scope), Name, unwrapDI<DIType>(Ty)));
}

extern "C" LLVMRustMetadataRef
LLVMRustDIBuilderCreateNameSpace(LLVMRustDIBuilderRef Builder,
                                 LLVMRustMetadataRef Scope, const char *Name,
                                 LLVMRustMetadataRef File, unsigned LineNo) {
  return wrap(Builder->createNameSpace(
      unwrapDI<DIDescriptor>(Scope), Name, unwrapDI<DIFile>(File), LineNo
#if LLVM_VERSION_GE(4, 0)
      ,
      false // ExportSymbols (only relevant for C++ anonymous namespaces)
#endif
      ));
}

extern "C" void
LLVMRustDICompositeTypeSetTypeArray(LLVMRustDIBuilderRef Builder,
                                    LLVMRustMetadataRef CompositeTy,
                                    LLVMRustMetadataRef TyArray) {
  DICompositeType *Tmp = unwrapDI<DICompositeType>(CompositeTy);
  Builder->replaceArrays(Tmp, DINodeArray(unwrap<MDTuple>(TyArray)));
}

extern "C" LLVMValueRef
LLVMRustDIBuilderCreateDebugLocation(LLVMContextRef ContextRef, unsigned Line,
                                     unsigned Column, LLVMRustMetadataRef Scope,
                                     LLVMRustMetadataRef InlinedAt) {
  LLVMContext &Context = *unwrap(ContextRef);

  DebugLoc debug_loc = DebugLoc::get(Line, Column, unwrapDIPtr<MDNode>(Scope),
                                     unwrapDIPtr<MDNode>(InlinedAt));

  return wrap(MetadataAsValue::get(Context, debug_loc.getAsMDNode()));
}

extern "C" int64_t LLVMRustDIBuilderCreateOpDeref() {
  return dwarf::DW_OP_deref;
}

extern "C" int64_t LLVMRustDIBuilderCreateOpPlus() { return dwarf::DW_OP_plus; }

extern "C" void LLVMRustWriteTypeToString(LLVMTypeRef Ty, RustStringRef Str) {
  RawRustStringOstream OS(Str);
  unwrap<llvm::Type>(Ty)->print(OS);
}

extern "C" void LLVMRustWriteValueToString(LLVMValueRef V,
                                           RustStringRef Str) {
  RawRustStringOstream OS(Str);
  OS << "(";
  unwrap<llvm::Value>(V)->getType()->print(OS);
  OS << ":";
  unwrap<llvm::Value>(V)->print(OS);
  OS << ")";
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
#elif LLVM_VERSION_GE(3, 8)
  if (Linker::linkModules(*Dst, std::move(Src.get()))) {
#else
  if (Linker::LinkModules(Dst, Src->get(),
                          [&](const DiagnosticInfo &DI) { DI.print(DP); })) {
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
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(DebugLoc, LLVMDebugLocRef)

extern "C" void LLVMRustWriteTwineToString(LLVMTwineRef T, RustStringRef Str) {
  RawRustStringOstream OS(Str);
  unwrap(T)->print(OS);
}

extern "C" void LLVMRustUnpackOptimizationDiagnostic(
    LLVMDiagnosticInfoRef DI, RustStringRef PassNameOut,
    LLVMValueRef *FunctionOut, LLVMDebugLocRef *DebugLocOut,
    RustStringRef MessageOut) {
  // Undefined to call this not on an optimization diagnostic!
  llvm::DiagnosticInfoOptimizationBase *Opt =
      static_cast<llvm::DiagnosticInfoOptimizationBase *>(unwrap(DI));

  RawRustStringOstream PassNameOS(PassNameOut);
  PassNameOS << Opt->getPassName();
  *FunctionOut = wrap(&Opt->getFunction());
  *DebugLocOut = wrap(&Opt->getDebugLoc());
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
#if LLVM_VERSION_GE(3, 8)
  case DK_OptimizationRemarkAnalysisFPCommute:
    return LLVMRustDiagnosticKind::OptimizationRemarkAnalysisFPCommute;
  case DK_OptimizationRemarkAnalysisAliasing:
    return LLVMRustDiagnosticKind::OptimizationRemarkAnalysisAliasing;
#endif
  default:
#if LLVM_VERSION_GE(3, 9)
    return (Kind >= DK_FirstRemark && Kind <= DK_LastRemark)
               ? LLVMRustDiagnosticKind::OptimizationRemarkOther
               : LLVMRustDiagnosticKind::Other;
#else
    return LLVMRustDiagnosticKind::Other;
#endif
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
#if LLVM_VERSION_GE(3, 8)
  case Type::TokenTyID:
    return LLVMTokenTypeKind;
#endif
  }
  llvm_unreachable("Unhandled TypeID.");
}

extern "C" void LLVMRustWriteDebugLocToString(LLVMContextRef C,
                                              LLVMDebugLocRef DL,
                                              RustStringRef Str) {
  RawRustStringOstream OS(Str);
  unwrap(DL)->print(OS);
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

extern "C" LLVMValueRef
LLVMRustBuildLandingPad(LLVMBuilderRef B, LLVMTypeRef Ty,
                        LLVMValueRef PersFn, unsigned NumClauses,
                        const char *Name, LLVMValueRef F) {
  return LLVMBuildLandingPad(B, Ty, PersFn, NumClauses, Name);
}

extern "C" LLVMValueRef LLVMRustBuildCleanupPad(LLVMBuilderRef B,
                                                LLVMValueRef ParentPad,
                                                unsigned ArgCount,
                                                LLVMValueRef *LLArgs,
                                                const char *Name) {
#if LLVM_VERSION_GE(3, 8)
  Value **Args = unwrap(LLArgs);
  if (ParentPad == nullptr) {
    Type *Ty = Type::getTokenTy(unwrap(B)->getContext());
    ParentPad = wrap(Constant::getNullValue(Ty));
  }
  return wrap(unwrap(B)->CreateCleanupPad(
      unwrap(ParentPad), ArrayRef<Value *>(Args, ArgCount), Name));
#else
  return nullptr;
#endif
}

extern "C" LLVMValueRef LLVMRustBuildCleanupRet(LLVMBuilderRef B,
                                                LLVMValueRef CleanupPad,
                                                LLVMBasicBlockRef UnwindBB) {
#if LLVM_VERSION_GE(3, 8)
  CleanupPadInst *Inst = cast<CleanupPadInst>(unwrap(CleanupPad));
  return wrap(unwrap(B)->CreateCleanupRet(Inst, unwrap(UnwindBB)));
#else
  return nullptr;
#endif
}

extern "C" LLVMValueRef
LLVMRustBuildCatchPad(LLVMBuilderRef B, LLVMValueRef ParentPad,
                      unsigned ArgCount, LLVMValueRef *LLArgs, const char *Name) {
#if LLVM_VERSION_GE(3, 8)
  Value **Args = unwrap(LLArgs);
  return wrap(unwrap(B)->CreateCatchPad(
      unwrap(ParentPad), ArrayRef<Value *>(Args, ArgCount), Name));
#else
  return nullptr;
#endif
}

extern "C" LLVMValueRef LLVMRustBuildCatchRet(LLVMBuilderRef B,
                                              LLVMValueRef Pad,
                                              LLVMBasicBlockRef BB) {
#if LLVM_VERSION_GE(3, 8)
  return wrap(unwrap(B)->CreateCatchRet(cast<CatchPadInst>(unwrap(Pad)),
                                              unwrap(BB)));
#else
  return nullptr;
#endif
}

extern "C" LLVMValueRef LLVMRustBuildCatchSwitch(LLVMBuilderRef B,
                                                 LLVMValueRef ParentPad,
                                                 LLVMBasicBlockRef BB,
                                                 unsigned NumHandlers,
                                                 const char *Name) {
#if LLVM_VERSION_GE(3, 8)
  if (ParentPad == nullptr) {
    Type *Ty = Type::getTokenTy(unwrap(B)->getContext());
    ParentPad = wrap(Constant::getNullValue(Ty));
  }
  return wrap(unwrap(B)->CreateCatchSwitch(unwrap(ParentPad), unwrap(BB),
                                                 NumHandlers, Name));
#else
  return nullptr;
#endif
}

extern "C" void LLVMRustAddHandler(LLVMValueRef CatchSwitchRef,
                                   LLVMBasicBlockRef Handler) {
#if LLVM_VERSION_GE(3, 8)
  Value *CatchSwitch = unwrap(CatchSwitchRef);
  cast<CatchSwitchInst>(CatchSwitch)->addHandler(unwrap(Handler));
#endif
}

extern "C" void LLVMRustSetPersonalityFn(LLVMBuilderRef B,
                                         LLVMValueRef Personality) {
#if LLVM_VERSION_GE(3, 8)
  unwrap(B)->GetInsertBlock()->getParent()->setPersonalityFn(
      cast<Function>(unwrap(Personality)));
#endif
}

#if LLVM_VERSION_GE(3, 8)
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
#else
extern "C" void *LLVMRustBuildOperandBundleDef(const char *Name,
                                               LLVMValueRef *Inputs,
                                               unsigned NumInputs) {
  return nullptr;
}

extern "C" void LLVMRustFreeOperandBundleDef(void *Bundle) {}

extern "C" LLVMValueRef LLVMRustBuildCall(LLVMBuilderRef B, LLVMValueRef Fn,
                                          LLVMValueRef *Args, unsigned NumArgs,
                                          void *Bundle, const char *Name) {
  return LLVMBuildCall(B, Fn, Args, NumArgs, Name);
}

extern "C" LLVMValueRef
LLVMRustBuildInvoke(LLVMBuilderRef B, LLVMValueRef Fn, LLVMValueRef *Args,
                    unsigned NumArgs, LLVMBasicBlockRef Then,
                    LLVMBasicBlockRef Catch, void *Bundle, const char *Name) {
  return LLVMBuildInvoke(B, Fn, Args, NumArgs, Then, Catch, Name);
}
#endif

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
    llvm_unreachable("Invalid LLVMRustLinkage value!");
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
  llvm_unreachable("Invalid LLVMRustLinkage value!");
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

extern "C" LLVMContextRef LLVMRustGetValueContext(LLVMValueRef V) {
  return wrap(&unwrap(V)->getContext());
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
  llvm_unreachable("Invalid LLVMRustVisibility value!");
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
  llvm_unreachable("Invalid LLVMRustVisibility value!");
}

extern "C" LLVMRustVisibility LLVMRustGetVisibility(LLVMValueRef V) {
  return toRust(LLVMGetVisibility(V));
}

extern "C" void LLVMRustSetVisibility(LLVMValueRef V,
                                      LLVMRustVisibility RustVisibility) {
  LLVMSetVisibility(V, fromRust(RustVisibility));
}
