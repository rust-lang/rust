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
#include "llvm/Object/Archive.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/Instructions.h"

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
static AtomicOrdering from_rust(LLVMAtomicOrdering Ordering) {
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
  ErrorOr<std::unique_ptr<MemoryBuffer>> buf_or = MemoryBuffer::getFile(Path,
                                                                        -1,
                                                                        false);
  if (!buf_or) {
      LLVMRustSetLastError(buf_or.getError().message().c_str());
      return nullptr;
  }
  return wrap(buf_or.get().release());
}

extern "C" char *LLVMRustGetLastError(void) {
  char *ret = LastError;
  LastError = NULL;
  return ret;
}

void LLVMRustSetLastError(const char *err) {
  free((void*) LastError);
  LastError = strdup(err);
}

extern "C" void
LLVMRustSetNormalizedTarget(LLVMModuleRef M, const char *triple) {
    unwrap(M)->setTargetTriple(Triple::normalize(triple));
}

extern "C" void LLVMRustPrintPassTimings() {
  raw_fd_ostream OS (2, false); // stderr.
  TimerGroup::printAll(OS);
}

extern "C" LLVMValueRef LLVMRustGetNamedValue(LLVMModuleRef M,
					      const char* Name) {
    return wrap(unwrap(M)->getNamedValue(Name));
}

extern "C" LLVMValueRef LLVMRustGetOrInsertFunction(LLVMModuleRef M,
						    const char* Name,
						    LLVMTypeRef FunctionTy) {
  return wrap(unwrap(M)->getOrInsertFunction(Name,
                                             unwrap<FunctionType>(FunctionTy)));
}

extern "C" LLVMValueRef LLVMRustGetOrInsertGlobal(LLVMModuleRef M,
						  const char* Name,
						  LLVMTypeRef Ty) {
  return wrap(unwrap(M)->getOrInsertGlobal(Name, unwrap(Ty)));
}

extern "C" LLVMTypeRef LLVMRustMetadataTypeInContext(LLVMContextRef C) {
  return wrap(Type::getMetadataTy(*unwrap(C)));
}

static Attribute::AttrKind
from_rust(LLVMRustAttribute kind) {
  switch (kind) {
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
    default:
      llvm_unreachable("bad AttributeKind");
  }
}

extern "C" void LLVMRustAddCallSiteAttribute(LLVMValueRef Instr, unsigned index, LLVMRustAttribute attr) {
  CallSite Call = CallSite(unwrap<Instruction>(Instr));
  Attribute Attr = Attribute::get(Call->getContext(), from_rust(attr));
  AttrBuilder B(Attr);
  Call.setAttributes(
    Call.getAttributes().addAttributes(Call->getContext(), index,
                                       AttributeSet::get(Call->getContext(),
                                                         index, B)));
}

extern "C" void LLVMRustAddDereferenceableCallSiteAttr(LLVMValueRef Instr,
                                                      unsigned index,
                                                      uint64_t bytes)
{
  CallSite Call = CallSite(unwrap<Instruction>(Instr));
  AttrBuilder B;
  B.addDereferenceableAttr(bytes);
  Call.setAttributes(
    Call.getAttributes().addAttributes(Call->getContext(), index,
                                       AttributeSet::get(Call->getContext(),
                                                         index, B)));
}

extern "C" void LLVMRustAddFunctionAttribute(LLVMValueRef Fn,
					     unsigned index,
					     LLVMRustAttribute attr)
{
  Function *A = unwrap<Function>(Fn);
  Attribute Attr = Attribute::get(A->getContext(), from_rust(attr));
  AttrBuilder B(Attr);
  A->addAttributes(index, AttributeSet::get(A->getContext(), index, B));
}

extern "C" void LLVMRustAddDereferenceableAttr(LLVMValueRef Fn,
					       unsigned index,
					       uint64_t bytes)
{
  Function *A = unwrap<Function>(Fn);
  AttrBuilder B;
  B.addDereferenceableAttr(bytes);
  A->addAttributes(index, AttributeSet::get(A->getContext(), index, B));
}

extern "C" void LLVMRustAddFunctionAttrStringValue(LLVMValueRef Fn,
						   unsigned index,
						   const char *Name,
						   const char *Value) {
  Function *F = unwrap<Function>(Fn);
  AttrBuilder B;
  B.addAttribute(Name, Value);
  F->addAttributes(index, AttributeSet::get(F->getContext(), index, B));
}

extern "C" void LLVMRustRemoveFunctionAttributes(LLVMValueRef Fn,
						 unsigned index,
						 LLVMRustAttribute attr)
{
  Function *F = unwrap<Function>(Fn);
  const AttributeSet PAL = F->getAttributes();
  Attribute Attr = Attribute::get(F->getContext(), from_rust(attr));
  AttrBuilder B(Attr);
  const AttributeSet PALnew =
    PAL.removeAttributes(F->getContext(), index,
                         AttributeSet::get(F->getContext(), index, B));
  F->setAttributes(PALnew);
}

// enable fpmath flag UnsafeAlgebra
extern "C" void LLVMRustSetHasUnsafeAlgebra(LLVMValueRef V) {
    if (auto I = dyn_cast<Instruction>(unwrap<Value>(V))) {
        I->setHasUnsafeAlgebra(true);
    }
}

extern "C" LLVMValueRef LLVMRustBuildAtomicLoad(LLVMBuilderRef B,
						LLVMValueRef source,
						const char* Name,
						LLVMAtomicOrdering order,
						unsigned alignment) {
    LoadInst* li = new LoadInst(unwrap(source),0);
    li->setAtomic(from_rust(order));
    li->setAlignment(alignment);
    return wrap(unwrap(B)->Insert(li, Name));
}

extern "C" LLVMValueRef LLVMRustBuildAtomicStore(LLVMBuilderRef B,
						 LLVMValueRef val,
						 LLVMValueRef target,
						 LLVMAtomicOrdering order,
						 unsigned alignment) {
    StoreInst* si = new StoreInst(unwrap(val),unwrap(target));
    si->setAtomic(from_rust(order));
    si->setAlignment(alignment);
    return wrap(unwrap(B)->Insert(si));
}

extern "C" LLVMValueRef LLVMRustBuildAtomicCmpXchg(LLVMBuilderRef B,
                                               LLVMValueRef target,
                                               LLVMValueRef old,
                                               LLVMValueRef source,
                                               LLVMAtomicOrdering order,
                                               LLVMAtomicOrdering failure_order,
                                               LLVMBool weak) {
    AtomicCmpXchgInst* acxi = unwrap(B)->CreateAtomicCmpXchg(
        unwrap(target),
        unwrap(old),
        unwrap(source),
        from_rust(order),
	from_rust(failure_order));
    acxi->setWeak(weak);
    return wrap(acxi);
}

enum class LLVMRustSynchronizationScope {
    Other,
    SingleThread,
    CrossThread,
};

static SynchronizationScope
from_rust(LLVMRustSynchronizationScope scope)
{
    switch (scope) {
    case LLVMRustSynchronizationScope::SingleThread:
        return SingleThread;
    case LLVMRustSynchronizationScope::CrossThread:
        return CrossThread;
    default:
        llvm_unreachable("bad SynchronizationScope.");
    }
}

extern "C" LLVMValueRef LLVMRustBuildAtomicFence(
    LLVMBuilderRef B,
    LLVMAtomicOrdering order,
    LLVMRustSynchronizationScope scope)
{
    return wrap(unwrap(B)->CreateFence(from_rust(order), from_rust(scope)));
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

static InlineAsm::AsmDialect
from_rust(LLVMRustAsmDialect dialect)
{
    switch (dialect) {
    case LLVMRustAsmDialect::Att:
        return InlineAsm::AD_ATT;
    case LLVMRustAsmDialect::Intel:
        return InlineAsm::AD_Intel;
    default:
        llvm_unreachable("bad AsmDialect.");
    }
}

extern "C" LLVMValueRef LLVMRustInlineAsm(LLVMTypeRef Ty,
					  char *AsmString,
					  char *Constraints,
					  LLVMBool HasSideEffects,
					  LLVMBool IsAlignStack,
					  LLVMRustAsmDialect Dialect) {
    return wrap(InlineAsm::get(unwrap<FunctionType>(Ty), AsmString,
                               Constraints, HasSideEffects,
                               IsAlignStack, from_rust(Dialect)));
}

typedef DIBuilder* LLVMRustDIBuilderRef;

typedef struct LLVMOpaqueMetadata *LLVMRustMetadataRef;

namespace llvm {
DEFINE_ISA_CONVERSION_FUNCTIONS(Metadata, LLVMRustMetadataRef)

inline Metadata **unwrap(LLVMRustMetadataRef *Vals) {
  return reinterpret_cast<Metadata**>(Vals);
}
}

template<typename DIT>
DIT* unwrapDIptr(LLVMRustMetadataRef ref) {
    return (DIT*) (ref ? unwrap<MDNode>(ref) : NULL);
}

#define DIDescriptor DIScope
#define DIArray DINodeArray
#define unwrapDI unwrapDIptr

// These values **must** match debuginfo::DIFlags! They also *happen*
// to match LLVM, but that isn't required as we do giant sets of
// matching below. The value shouldn't be directly passed to LLVM.
enum class LLVMRustDIFlags : uint32_t {
    FlagZero                = 0,
    FlagPrivate             = 1,
    FlagProtected           = 2,
    FlagPublic              = 3,
    FlagFwdDecl             = (1 << 2),
    FlagAppleBlock          = (1 << 3),
    FlagBlockByrefStruct    = (1 << 4),
    FlagVirtual             = (1 << 5),
    FlagArtificial          = (1 << 6),
    FlagExplicit            = (1 << 7),
    FlagPrototyped          = (1 << 8),
    FlagObjcClassComplete   = (1 << 9),
    FlagObjectPointer       = (1 << 10),
    FlagVector              = (1 << 11),
    FlagStaticMember        = (1 << 12),
    FlagLValueReference     = (1 << 13),
    FlagRValueReference     = (1 << 14),
    // Do not add values that are not supported by the minimum LLVM
    // version we support!
};

inline LLVMRustDIFlags operator& (LLVMRustDIFlags a, LLVMRustDIFlags b) {
    return static_cast<LLVMRustDIFlags>(static_cast<uint32_t>(a) & static_cast<uint32_t>(b));
}

inline LLVMRustDIFlags operator| (LLVMRustDIFlags a, LLVMRustDIFlags b) {
    return static_cast<LLVMRustDIFlags>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}

inline LLVMRustDIFlags& operator|= (LLVMRustDIFlags& a, LLVMRustDIFlags b) {
    return a = a | b;
}

inline bool is_set(LLVMRustDIFlags f) {
    return f != LLVMRustDIFlags::FlagZero;
}

inline LLVMRustDIFlags visibility(LLVMRustDIFlags f) {
    return static_cast<LLVMRustDIFlags>(static_cast<uint32_t>(f) & 0x3);
}

#if LLVM_VERSION_GE(4, 0)
static DINode::DIFlags from_rust(LLVMRustDIFlags flags) {
    DINode::DIFlags result = DINode::DIFlags::FlagZero;
#else
static unsigned from_rust(LLVMRustDIFlags flags) {
    unsigned result = 0;
#endif

    switch (visibility(flags)) {
    case LLVMRustDIFlags::FlagPrivate:
        result |= DINode::DIFlags::FlagPrivate;
        break;
    case LLVMRustDIFlags::FlagProtected:
        result |= DINode::DIFlags::FlagProtected;
        break;
    case LLVMRustDIFlags::FlagPublic:
        result |= DINode::DIFlags::FlagPublic;
        break;
    default:
        // The rest are handled below
        break;
    }

    if (is_set(flags & LLVMRustDIFlags::FlagFwdDecl))             { result |= DINode::DIFlags::FlagFwdDecl; }
    if (is_set(flags & LLVMRustDIFlags::FlagAppleBlock))          { result |= DINode::DIFlags::FlagAppleBlock; }
    if (is_set(flags & LLVMRustDIFlags::FlagBlockByrefStruct))    { result |= DINode::DIFlags::FlagBlockByrefStruct; }
    if (is_set(flags & LLVMRustDIFlags::FlagVirtual))             { result |= DINode::DIFlags::FlagVirtual; }
    if (is_set(flags & LLVMRustDIFlags::FlagArtificial))          { result |= DINode::DIFlags::FlagArtificial; }
    if (is_set(flags & LLVMRustDIFlags::FlagExplicit))            { result |= DINode::DIFlags::FlagExplicit; }
    if (is_set(flags & LLVMRustDIFlags::FlagPrototyped))          { result |= DINode::DIFlags::FlagPrototyped; }
    if (is_set(flags & LLVMRustDIFlags::FlagObjcClassComplete))   { result |= DINode::DIFlags::FlagObjcClassComplete; }
    if (is_set(flags & LLVMRustDIFlags::FlagObjectPointer))       { result |= DINode::DIFlags::FlagObjectPointer; }
    if (is_set(flags & LLVMRustDIFlags::FlagVector))              { result |= DINode::DIFlags::FlagVector; }
    if (is_set(flags & LLVMRustDIFlags::FlagStaticMember))        { result |= DINode::DIFlags::FlagStaticMember; }
    if (is_set(flags & LLVMRustDIFlags::FlagLValueReference))     { result |= DINode::DIFlags::FlagLValueReference; }
    if (is_set(flags & LLVMRustDIFlags::FlagRValueReference))     { result |= DINode::DIFlags::FlagRValueReference; }

    return result;
}

extern "C" uint32_t LLVMRustDebugMetadataVersion() {
    return DEBUG_METADATA_VERSION;
}

extern "C" uint32_t LLVMRustVersionMinor() {
  return LLVM_VERSION_MINOR;
}

extern "C" uint32_t LLVMRustVersionMajor() {
  return LLVM_VERSION_MAJOR;
}

extern "C" void LLVMRustAddModuleFlag(LLVMModuleRef M,
                                      const char *name,
                                      uint32_t value) {
    unwrap(M)->addModuleFlag(Module::Warning, name, value);
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
    LLVMRustDIBuilderRef Builder,
    unsigned Lang,
    const char* File,
    const char* Dir,
    const char* Producer,
    bool isOptimized,
    const char* Flags,
    unsigned RuntimeVer,
    const char* SplitName) {
    return wrap(Builder->createCompileUnit(Lang,
                                           File,
                                           Dir,
                                           Producer,
                                           isOptimized,
                                           Flags,
                                           RuntimeVer,
                                           SplitName));
}

extern "C" LLVMRustMetadataRef LLVMRustDIBuilderCreateFile(
    LLVMRustDIBuilderRef Builder,
    const char* Filename,
    const char* Directory) {
    return wrap(Builder->createFile(Filename, Directory));
}

extern "C" LLVMRustMetadataRef LLVMRustDIBuilderCreateSubroutineType(
    LLVMRustDIBuilderRef Builder,
    LLVMRustMetadataRef File,
    LLVMRustMetadataRef ParameterTypes) {
    return wrap(Builder->createSubroutineType(
#if LLVM_VERSION_EQ(3, 7)
        unwrapDI<DIFile>(File),
#endif
        DITypeRefArray(unwrap<MDTuple>(ParameterTypes))));
}

extern "C" LLVMRustMetadataRef LLVMRustDIBuilderCreateFunction(
    LLVMRustDIBuilderRef Builder,
    LLVMRustMetadataRef Scope,
    const char* Name,
    const char* LinkageName,
    LLVMRustMetadataRef File,
    unsigned LineNo,
    LLVMRustMetadataRef Ty,
    bool isLocalToUnit,
    bool isDefinition,
    unsigned ScopeLine,
    LLVMRustDIFlags Flags,
    bool isOptimized,
    LLVMValueRef Fn,
    LLVMRustMetadataRef TParam,
    LLVMRustMetadataRef Decl) {
#if LLVM_VERSION_GE(3, 8)
    DITemplateParameterArray TParams =
        DITemplateParameterArray(unwrap<MDTuple>(TParam));
    DISubprogram *Sub = Builder->createFunction(
        unwrapDI<DIScope>(Scope), Name, LinkageName,
        unwrapDI<DIFile>(File), LineNo,
        unwrapDI<DISubroutineType>(Ty), isLocalToUnit, isDefinition, ScopeLine,
        from_rust(Flags), isOptimized,
        TParams,
        unwrapDIptr<DISubprogram>(Decl));
    unwrap<Function>(Fn)->setSubprogram(Sub);
    return wrap(Sub);
#else
    return wrap(Builder->createFunction(
        unwrapDI<DIScope>(Scope), Name, LinkageName,
        unwrapDI<DIFile>(File), LineNo,
        unwrapDI<DISubroutineType>(Ty), isLocalToUnit, isDefinition, ScopeLine,
        from_rust(Flags), isOptimized,
        unwrap<Function>(Fn),
        unwrapDIptr<MDNode>(TParam),
        unwrapDIptr<MDNode>(Decl)));
#endif
}

extern "C" LLVMRustMetadataRef LLVMRustDIBuilderCreateBasicType(
    LLVMRustDIBuilderRef Builder,
    const char* Name,
    uint64_t SizeInBits,
    uint64_t AlignInBits,
    unsigned Encoding) {
    return wrap(Builder->createBasicType(
        Name, SizeInBits,
        AlignInBits, Encoding));
}

extern "C" LLVMRustMetadataRef LLVMRustDIBuilderCreatePointerType(
    LLVMRustDIBuilderRef Builder,
    LLVMRustMetadataRef PointeeTy,
    uint64_t SizeInBits,
    uint64_t AlignInBits,
    const char* Name) {
    return wrap(Builder->createPointerType(
        unwrapDI<DIType>(PointeeTy), SizeInBits, AlignInBits, Name));
}

extern "C" LLVMRustMetadataRef LLVMRustDIBuilderCreateStructType(
    LLVMRustDIBuilderRef Builder,
    LLVMRustMetadataRef Scope,
    const char* Name,
    LLVMRustMetadataRef File,
    unsigned LineNumber,
    uint64_t SizeInBits,
    uint64_t AlignInBits,
    LLVMRustDIFlags Flags,
    LLVMRustMetadataRef DerivedFrom,
    LLVMRustMetadataRef Elements,
    unsigned RunTimeLang,
    LLVMRustMetadataRef VTableHolder,
    const char *UniqueId) {
    return wrap(Builder->createStructType(
        unwrapDI<DIDescriptor>(Scope),
        Name,
        unwrapDI<DIFile>(File),
        LineNumber,
        SizeInBits,
        AlignInBits,
        from_rust(Flags),
        unwrapDI<DIType>(DerivedFrom),
        DINodeArray(unwrapDI<MDTuple>(Elements)),
        RunTimeLang,
        unwrapDI<DIType>(VTableHolder),
        UniqueId
        ));
}

extern "C" LLVMRustMetadataRef LLVMRustDIBuilderCreateMemberType(
    LLVMRustDIBuilderRef Builder,
    LLVMRustMetadataRef Scope,
    const char* Name,
    LLVMRustMetadataRef File,
    unsigned LineNo,
    uint64_t SizeInBits,
    uint64_t AlignInBits,
    uint64_t OffsetInBits,
    LLVMRustDIFlags Flags,
    LLVMRustMetadataRef Ty) {
    return wrap(Builder->createMemberType(
        unwrapDI<DIDescriptor>(Scope), Name,
        unwrapDI<DIFile>(File), LineNo,
        SizeInBits, AlignInBits, OffsetInBits, from_rust(Flags),
        unwrapDI<DIType>(Ty)));
}

extern "C" LLVMRustMetadataRef LLVMRustDIBuilderCreateLexicalBlock(
    LLVMRustDIBuilderRef Builder,
    LLVMRustMetadataRef Scope,
    LLVMRustMetadataRef File,
    unsigned Line,
    unsigned Col) {
    return wrap(Builder->createLexicalBlock(
        unwrapDI<DIDescriptor>(Scope),
        unwrapDI<DIFile>(File), Line, Col
        ));
}

extern "C" LLVMRustMetadataRef LLVMRustDIBuilderCreateLexicalBlockFile(
    LLVMRustDIBuilderRef Builder,
    LLVMRustMetadataRef Scope,
    LLVMRustMetadataRef File) {
    return wrap(Builder->createLexicalBlockFile(
        unwrapDI<DIDescriptor>(Scope),
        unwrapDI<DIFile>(File)));
}

extern "C" LLVMRustMetadataRef LLVMRustDIBuilderCreateStaticVariable(
    LLVMRustDIBuilderRef Builder,
    LLVMRustMetadataRef Context,
    const char* Name,
    const char* LinkageName,
    LLVMRustMetadataRef File,
    unsigned LineNo,
    LLVMRustMetadataRef Ty,
    bool isLocalToUnit,
    LLVMValueRef Val,
    LLVMRustMetadataRef Decl = NULL) {
    return wrap(Builder->createGlobalVariable(unwrapDI<DIDescriptor>(Context),
        Name,
        LinkageName,
        unwrapDI<DIFile>(File),
        LineNo,
        unwrapDI<DIType>(Ty),
        isLocalToUnit,
        cast<Constant>(unwrap(Val)),
        unwrapDIptr<MDNode>(Decl)));
}

extern "C" LLVMRustMetadataRef LLVMRustDIBuilderCreateVariable(
    LLVMRustDIBuilderRef Builder,
    unsigned Tag,
    LLVMRustMetadataRef Scope,
    const char* Name,
    LLVMRustMetadataRef File,
    unsigned LineNo,
    LLVMRustMetadataRef Ty,
    bool AlwaysPreserve,
    LLVMRustDIFlags Flags,
    unsigned ArgNo) {
#if LLVM_VERSION_GE(3, 8)
    if (Tag == 0x100) { // DW_TAG_auto_variable
        return wrap(Builder->createAutoVariable(
            unwrapDI<DIDescriptor>(Scope), Name,
            unwrapDI<DIFile>(File),
            LineNo,
            unwrapDI<DIType>(Ty), AlwaysPreserve, from_rust(Flags)));
    } else {
        return wrap(Builder->createParameterVariable(
            unwrapDI<DIDescriptor>(Scope), Name, ArgNo,
            unwrapDI<DIFile>(File),
            LineNo,
            unwrapDI<DIType>(Ty), AlwaysPreserve, from_rust(Flags)));
    }
#else
    return wrap(Builder->createLocalVariable(Tag,
        unwrapDI<DIDescriptor>(Scope), Name,
        unwrapDI<DIFile>(File),
        LineNo,
        unwrapDI<DIType>(Ty), AlwaysPreserve, from_rust(Flags), ArgNo));
#endif
}

extern "C" LLVMRustMetadataRef LLVMRustDIBuilderCreateArrayType(
    LLVMRustDIBuilderRef Builder,
    uint64_t Size,
    uint64_t AlignInBits,
    LLVMRustMetadataRef Ty,
    LLVMRustMetadataRef Subscripts) {
    return wrap(Builder->createArrayType(Size, AlignInBits,
        unwrapDI<DIType>(Ty),
        DINodeArray(unwrapDI<MDTuple>(Subscripts))
    ));
}

extern "C" LLVMRustMetadataRef LLVMRustDIBuilderCreateVectorType(
    LLVMRustDIBuilderRef Builder,
    uint64_t Size,
    uint64_t AlignInBits,
    LLVMRustMetadataRef Ty,
    LLVMRustMetadataRef Subscripts) {
    return wrap(Builder->createVectorType(Size, AlignInBits,
        unwrapDI<DIType>(Ty),
        DINodeArray(unwrapDI<MDTuple>(Subscripts))
    ));
}

extern "C" LLVMRustMetadataRef LLVMRustDIBuilderGetOrCreateSubrange(
    LLVMRustDIBuilderRef Builder,
    int64_t Lo,
    int64_t Count) {
    return wrap(Builder->getOrCreateSubrange(Lo, Count));
}

extern "C" LLVMRustMetadataRef LLVMRustDIBuilderGetOrCreateArray(
    LLVMRustDIBuilderRef Builder,
    LLVMRustMetadataRef* Ptr,
    unsigned Count) {
    Metadata **DataValue = unwrap(Ptr);
    return wrap(Builder->getOrCreateArray(
        ArrayRef<Metadata*>(DataValue, Count)).get());
}

extern "C" LLVMValueRef LLVMRustDIBuilderInsertDeclareAtEnd(
    LLVMRustDIBuilderRef Builder,
    LLVMValueRef Val,
    LLVMRustMetadataRef VarInfo,
    int64_t* AddrOps,
    unsigned AddrOpsCount,
    LLVMValueRef DL,
    LLVMBasicBlockRef InsertAtEnd) {
    return wrap(Builder->insertDeclare(
        unwrap(Val),
        unwrap<DILocalVariable>(VarInfo),
        Builder->createExpression(
          llvm::ArrayRef<int64_t>(AddrOps, AddrOpsCount)),
        DebugLoc(cast<MDNode>(unwrap<MetadataAsValue>(DL)->getMetadata())),
        unwrap(InsertAtEnd)));
}

extern "C" LLVMRustMetadataRef LLVMRustDIBuilderCreateEnumerator(
    LLVMRustDIBuilderRef Builder,
    const char* Name,
    uint64_t Val)
{
    return wrap(Builder->createEnumerator(Name, Val));
}

extern "C" LLVMRustMetadataRef LLVMRustDIBuilderCreateEnumerationType(
    LLVMRustDIBuilderRef Builder,
    LLVMRustMetadataRef Scope,
    const char* Name,
    LLVMRustMetadataRef File,
    unsigned LineNumber,
    uint64_t SizeInBits,
    uint64_t AlignInBits,
    LLVMRustMetadataRef Elements,
    LLVMRustMetadataRef ClassType)
{
    return wrap(Builder->createEnumerationType(
        unwrapDI<DIDescriptor>(Scope),
        Name,
        unwrapDI<DIFile>(File),
        LineNumber,
        SizeInBits,
        AlignInBits,
        DINodeArray(unwrapDI<MDTuple>(Elements)),
        unwrapDI<DIType>(ClassType)));
}

extern "C" LLVMRustMetadataRef LLVMRustDIBuilderCreateUnionType(
    LLVMRustDIBuilderRef Builder,
    LLVMRustMetadataRef Scope,
    const char* Name,
    LLVMRustMetadataRef File,
    unsigned LineNumber,
    uint64_t SizeInBits,
    uint64_t AlignInBits,
    LLVMRustDIFlags Flags,
    LLVMRustMetadataRef Elements,
    unsigned RunTimeLang,
    const char* UniqueId)
{
    return wrap(Builder->createUnionType(
        unwrapDI<DIDescriptor>(Scope),
        Name,
        unwrapDI<DIFile>(File),
        LineNumber,
        SizeInBits,
        AlignInBits,
        from_rust(Flags),
        DINodeArray(unwrapDI<MDTuple>(Elements)),
        RunTimeLang,
        UniqueId
        ));
}

extern "C" LLVMRustMetadataRef LLVMRustDIBuilderCreateTemplateTypeParameter(
    LLVMRustDIBuilderRef Builder,
    LLVMRustMetadataRef Scope,
    const char* Name,
    LLVMRustMetadataRef Ty,
    LLVMRustMetadataRef File,
    unsigned LineNo,
    unsigned ColumnNo)
{
    return wrap(Builder->createTemplateTypeParameter(
      unwrapDI<DIDescriptor>(Scope),
      Name,
      unwrapDI<DIType>(Ty)
      ));
}

extern "C" LLVMRustMetadataRef LLVMRustDIBuilderCreateNameSpace(
    LLVMRustDIBuilderRef Builder,
    LLVMRustMetadataRef Scope,
    const char* Name,
    LLVMRustMetadataRef File,
    unsigned LineNo)
{
    return wrap(Builder->createNameSpace(
        unwrapDI<DIDescriptor>(Scope),
        Name,
        unwrapDI<DIFile>(File),
        LineNo
#if LLVM_VERSION_GE(4, 0)
        , false // ExportSymbols (only relevant for C++ anonymous namespaces)
#endif
    ));
}

extern "C" void LLVMRustDICompositeTypeSetTypeArray(
    LLVMRustDIBuilderRef Builder,
    LLVMRustMetadataRef CompositeType,
    LLVMRustMetadataRef TypeArray)
{
    DICompositeType *tmp = unwrapDI<DICompositeType>(CompositeType);
    Builder->replaceArrays(tmp, DINodeArray(unwrap<MDTuple>(TypeArray)));
}

extern "C" LLVMValueRef LLVMRustDIBuilderCreateDebugLocation(
  LLVMContextRef Context,
  unsigned Line,
  unsigned Column,
  LLVMRustMetadataRef Scope,
  LLVMRustMetadataRef InlinedAt)
{
    LLVMContext& context = *unwrap(Context);

    DebugLoc debug_loc = DebugLoc::get(Line,
                                       Column,
                                       unwrapDIptr<MDNode>(Scope),
                                       unwrapDIptr<MDNode>(InlinedAt));

    return wrap(MetadataAsValue::get(context, debug_loc.getAsMDNode()));
}

extern "C" int64_t LLVMRustDIBuilderCreateOpDeref()
{
    return dwarf::DW_OP_deref;
}

extern "C" int64_t LLVMRustDIBuilderCreateOpPlus()
{
    return dwarf::DW_OP_plus;
}

extern "C" void LLVMRustWriteTypeToString(LLVMTypeRef Type, RustStringRef str) {
    raw_rust_string_ostream os(str);
    unwrap<llvm::Type>(Type)->print(os);
}

extern "C" void LLVMRustWriteValueToString(LLVMValueRef Value, RustStringRef str) {
    raw_rust_string_ostream os(str);
    os << "(";
    unwrap<llvm::Value>(Value)->getType()->print(os);
    os << ":";
    unwrap<llvm::Value>(Value)->print(os);
    os << ")";
}

extern "C" bool
LLVMRustLinkInExternalBitcode(LLVMModuleRef dst, char *bc, size_t len) {
    Module *Dst = unwrap(dst);

    std::unique_ptr<MemoryBuffer> buf = MemoryBuffer::getMemBufferCopy(StringRef(bc, len));

#if LLVM_VERSION_GE(4, 0)
    Expected<std::unique_ptr<Module>> SrcOrError =
        llvm::getLazyBitcodeModule(buf->getMemBufferRef(), Dst->getContext());
    if (!SrcOrError) {
        LLVMRustSetLastError(toString(SrcOrError.takeError()).c_str());
        return false;
    }

    auto Src = std::move(*SrcOrError);
#else
    ErrorOr<std::unique_ptr<Module>> Src =
        llvm::getLazyBitcodeModule(std::move(buf), Dst->getContext());
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
    if (Linker::LinkModules(Dst, Src->get(), [&](const DiagnosticInfo &DI) { DI.print(DP); })) {
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
    return reinterpret_cast<section_iterator*>(SI);
}

extern "C" size_t
LLVMRustGetSectionName(LLVMSectionIteratorRef SI, const char **ptr) {
    StringRef ret;
    if (std::error_code ec = (*unwrap(SI))->getName(ret))
      report_fatal_error(ec.message());
    *ptr = ret.data();
    return ret.size();
}

// LLVMArrayType function does not support 64-bit ElementCount
extern "C" LLVMTypeRef
LLVMRustArrayType(LLVMTypeRef ElementType, uint64_t ElementCount) {
    return wrap(ArrayType::get(unwrap(ElementType), ElementCount));
}

DEFINE_SIMPLE_CONVERSION_FUNCTIONS(Twine, LLVMTwineRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(DebugLoc, LLVMDebugLocRef)

extern "C" void
LLVMRustWriteTwineToString(LLVMTwineRef T, RustStringRef str) {
    raw_rust_string_ostream os(str);
    unwrap(T)->print(os);
}

extern "C" void
LLVMRustUnpackOptimizationDiagnostic(
    LLVMDiagnosticInfoRef di,
    RustStringRef pass_name_out,
    LLVMValueRef *function_out,
    LLVMDebugLocRef *debugloc_out,
    RustStringRef message_out)
{
    // Undefined to call this not on an optimization diagnostic!
    llvm::DiagnosticInfoOptimizationBase *opt
        = static_cast<llvm::DiagnosticInfoOptimizationBase*>(unwrap(di));

    raw_rust_string_ostream pass_name_os(pass_name_out);
    pass_name_os << opt->getPassName();
    *function_out = wrap(&opt->getFunction());
    *debugloc_out = wrap(&opt->getDebugLoc());
    raw_rust_string_ostream message_os(message_out);
    message_os << opt->getMsg();
}

extern "C" void
LLVMRustUnpackInlineAsmDiagnostic(
    LLVMDiagnosticInfoRef di,
    unsigned *cookie_out,
    LLVMTwineRef *message_out,
    LLVMValueRef *instruction_out)
{
    // Undefined to call this not on an inline assembly diagnostic!
    llvm::DiagnosticInfoInlineAsm *ia
        = static_cast<llvm::DiagnosticInfoInlineAsm*>(unwrap(di));

    *cookie_out = ia->getLocCookie();
    *message_out = wrap(&ia->getMsgStr());
    *instruction_out = wrap(ia->getInstruction());
}

extern "C" void LLVMRustWriteDiagnosticInfoToString(LLVMDiagnosticInfoRef di, RustStringRef str) {
    raw_rust_string_ostream os(str);
    DiagnosticPrinterRawOStream dp(os);
    unwrap(di)->print(dp);
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

static LLVMRustDiagnosticKind
to_rust(DiagnosticKind kind)
{
    switch (kind) {
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
        return (kind >= DK_FirstRemark && kind <= DK_LastRemark) ?
            LLVMRustDiagnosticKind::OptimizationRemarkOther :
            LLVMRustDiagnosticKind::Other;
#else
        return LLVMRustDiagnosticKind::Other;
#endif
  }
}

extern "C" LLVMRustDiagnosticKind LLVMRustGetDiagInfoKind(LLVMDiagnosticInfoRef di) {
    return to_rust((DiagnosticKind) unwrap(di)->getKind());
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

extern "C" void LLVMRustWriteDebugLocToString(
    LLVMContextRef C,
    LLVMDebugLocRef dl,
    RustStringRef str)
{
    raw_rust_string_ostream os(str);
    unwrap(dl)->print(os);
}

DEFINE_SIMPLE_CONVERSION_FUNCTIONS(SMDiagnostic, LLVMSMDiagnosticRef)

extern "C" void LLVMRustSetInlineAsmDiagnosticHandler(
    LLVMContextRef C,
    LLVMContext::InlineAsmDiagHandlerTy H,
    void *CX)
{
    unwrap(C)->setInlineAsmDiagnosticHandler(H, CX);
}

extern "C" void LLVMRustWriteSMDiagnosticToString(LLVMSMDiagnosticRef d,
						  RustStringRef str) {
    raw_rust_string_ostream os(str);
    unwrap(d)->print("", os);
}

extern "C" LLVMValueRef
LLVMRustBuildLandingPad(LLVMBuilderRef Builder,
                        LLVMTypeRef Ty,
                        LLVMValueRef PersFn,
                        unsigned NumClauses,
                        const char* Name,
                        LLVMValueRef F) {
    return LLVMBuildLandingPad(Builder, Ty, PersFn, NumClauses, Name);
}

extern "C" LLVMValueRef
LLVMRustBuildCleanupPad(LLVMBuilderRef Builder,
                        LLVMValueRef ParentPad,
                        unsigned ArgCnt,
                        LLVMValueRef *LLArgs,
                        const char *Name) {
#if LLVM_VERSION_GE(3, 8)
    Value **Args = unwrap(LLArgs);
    if (ParentPad == NULL) {
        Type *Ty = Type::getTokenTy(unwrap(Builder)->getContext());
        ParentPad = wrap(Constant::getNullValue(Ty));
    }
    return wrap(unwrap(Builder)->CreateCleanupPad(unwrap(ParentPad),
                                                  ArrayRef<Value*>(Args, ArgCnt),
                                                  Name));
#else
    return NULL;
#endif
}

extern "C" LLVMValueRef
LLVMRustBuildCleanupRet(LLVMBuilderRef Builder,
                        LLVMValueRef CleanupPad,
                        LLVMBasicBlockRef UnwindBB) {
#if LLVM_VERSION_GE(3, 8)
    CleanupPadInst *Inst = cast<CleanupPadInst>(unwrap(CleanupPad));
    return wrap(unwrap(Builder)->CreateCleanupRet(Inst, unwrap(UnwindBB)));
#else
    return NULL;
#endif
}

extern "C" LLVMValueRef
LLVMRustBuildCatchPad(LLVMBuilderRef Builder,
                      LLVMValueRef ParentPad,
                      unsigned ArgCnt,
                      LLVMValueRef *LLArgs,
                      const char *Name) {
#if LLVM_VERSION_GE(3, 8)
    Value **Args = unwrap(LLArgs);
    return wrap(unwrap(Builder)->CreateCatchPad(unwrap(ParentPad),
                                                ArrayRef<Value*>(Args, ArgCnt),
                                                Name));
#else
    return NULL;
#endif
}

extern "C" LLVMValueRef
LLVMRustBuildCatchRet(LLVMBuilderRef Builder,
                      LLVMValueRef Pad,
                      LLVMBasicBlockRef BB) {
#if LLVM_VERSION_GE(3, 8)
    return wrap(unwrap(Builder)->CreateCatchRet(cast<CatchPadInst>(unwrap(Pad)),
                                                unwrap(BB)));
#else
    return NULL;
#endif
}

extern "C" LLVMValueRef
LLVMRustBuildCatchSwitch(LLVMBuilderRef Builder,
                         LLVMValueRef ParentPad,
                         LLVMBasicBlockRef BB,
                         unsigned NumHandlers,
                         const char *Name) {
#if LLVM_VERSION_GE(3, 8)
    if (ParentPad == NULL) {
        Type *Ty = Type::getTokenTy(unwrap(Builder)->getContext());
        ParentPad = wrap(Constant::getNullValue(Ty));
    }
    return wrap(unwrap(Builder)->CreateCatchSwitch(unwrap(ParentPad),
                                                   unwrap(BB),
                                                   NumHandlers,
                                                   Name));
#else
    return NULL;
#endif
}

extern "C" void
LLVMRustAddHandler(LLVMValueRef CatchSwitchRef,
                   LLVMBasicBlockRef Handler) {
#if LLVM_VERSION_GE(3, 8)
    Value *CatchSwitch = unwrap(CatchSwitchRef);
    cast<CatchSwitchInst>(CatchSwitch)->addHandler(unwrap(Handler));
#endif
}

extern "C" void
LLVMRustSetPersonalityFn(LLVMBuilderRef B,
                         LLVMValueRef Personality) {
#if LLVM_VERSION_GE(3, 8)
    unwrap(B)->GetInsertBlock()
             ->getParent()
             ->setPersonalityFn(cast<Function>(unwrap(Personality)));
#endif
}

#if LLVM_VERSION_GE(3, 8)
extern "C" OperandBundleDef*
LLVMRustBuildOperandBundleDef(const char *Name,
                              LLVMValueRef *Inputs,
                              unsigned NumInputs) {
  return new OperandBundleDef(Name, makeArrayRef(unwrap(Inputs), NumInputs));
}

extern "C" void
LLVMRustFreeOperandBundleDef(OperandBundleDef* Bundle) {
  delete Bundle;
}

extern "C" LLVMValueRef
LLVMRustBuildCall(LLVMBuilderRef B,
                    LLVMValueRef Fn,
                    LLVMValueRef *Args,
                    unsigned NumArgs,
                    OperandBundleDef *Bundle,
                    const char *Name) {
    unsigned len = Bundle ? 1 : 0;
    ArrayRef<OperandBundleDef> Bundles = makeArrayRef(Bundle, len);
    return wrap(unwrap(B)->CreateCall(unwrap(Fn),
                                      makeArrayRef(unwrap(Args), NumArgs),
                                      Bundles,
                                      Name));
}

extern "C" LLVMValueRef
LLVMRustBuildInvoke(LLVMBuilderRef B,
                    LLVMValueRef Fn,
                    LLVMValueRef *Args,
                    unsigned NumArgs,
                    LLVMBasicBlockRef Then,
                    LLVMBasicBlockRef Catch,
                    OperandBundleDef *Bundle,
                    const char *Name) {
    unsigned len = Bundle ? 1 : 0;
    ArrayRef<OperandBundleDef> Bundles = makeArrayRef(Bundle, len);
    return wrap(unwrap(B)->CreateInvoke(unwrap(Fn), unwrap(Then), unwrap(Catch),
                                        makeArrayRef(unwrap(Args), NumArgs),
                                        Bundles,
                                        Name));
}
#else
extern "C" void*
LLVMRustBuildOperandBundleDef(const char *Name,
                              LLVMValueRef *Inputs,
                              unsigned NumInputs) {
  return NULL;
}

extern "C" void
LLVMRustFreeOperandBundleDef(void* Bundle) {
}

extern "C" LLVMValueRef
LLVMRustBuildCall(LLVMBuilderRef B,
                    LLVMValueRef Fn,
                    LLVMValueRef *Args,
                    unsigned NumArgs,
                    void *Bundle,
                    const char *Name) {
    return LLVMBuildCall(B, Fn, Args, NumArgs, Name);
}

extern "C" LLVMValueRef
LLVMRustBuildInvoke(LLVMBuilderRef B,
                    LLVMValueRef Fn,
                    LLVMValueRef *Args,
                    unsigned NumArgs,
                    LLVMBasicBlockRef Then,
                    LLVMBasicBlockRef Catch,
                    void *Bundle,
                    const char *Name) {
    return LLVMBuildInvoke(B, Fn, Args, NumArgs, Then, Catch, Name);
}
#endif

extern "C" void LLVMRustPositionBuilderAtStart(LLVMBuilderRef B, LLVMBasicBlockRef BB) {
    auto point = unwrap(BB)->getFirstInsertionPt();
    unwrap(B)->SetInsertPoint(unwrap(BB), point);
}

extern "C" void LLVMRustSetComdat(LLVMModuleRef M, LLVMValueRef V, const char *Name) {
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

static LLVMRustLinkage to_rust(LLVMLinkage linkage) {
    switch (linkage) {
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

static LLVMLinkage from_rust(LLVMRustLinkage linkage) {
    switch (linkage) {
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
        default:
            llvm_unreachable("Invalid LLVMRustLinkage value!");
    }
}

extern "C" LLVMRustLinkage LLVMRustGetLinkage(LLVMValueRef V) {
    return to_rust(LLVMGetLinkage(V));
}

extern "C" void LLVMRustSetLinkage(LLVMValueRef V, LLVMRustLinkage RustLinkage) {
    LLVMSetLinkage(V, from_rust(RustLinkage));
}

extern "C" LLVMContextRef LLVMRustGetValueContext(LLVMValueRef V) {
    return wrap(&unwrap(V)->getContext());
}

enum class LLVMRustVisibility {
    Default = 0,
    Hidden = 1,
    Protected = 2,
};

static LLVMRustVisibility to_rust(LLVMVisibility vis) {
    switch (vis) {
        case LLVMDefaultVisibility:
            return LLVMRustVisibility::Default;
        case LLVMHiddenVisibility:
            return LLVMRustVisibility::Hidden;
        case LLVMProtectedVisibility:
            return LLVMRustVisibility::Protected;

        default:
            llvm_unreachable("Invalid LLVMRustVisibility value!");
    }
}

static LLVMVisibility from_rust(LLVMRustVisibility vis) {
    switch (vis) {
        case LLVMRustVisibility::Default:
            return LLVMDefaultVisibility;
        case LLVMRustVisibility::Hidden:
            return LLVMHiddenVisibility;
        case LLVMRustVisibility::Protected:
            return LLVMProtectedVisibility;

        default:
            llvm_unreachable("Invalid LLVMRustVisibility value!");
    }
}

extern "C" LLVMRustVisibility LLVMRustGetVisibility(LLVMValueRef V) {
    return to_rust(LLVMGetVisibility(V));
}

extern "C" void LLVMRustSetVisibility(LLVMValueRef V, LLVMRustVisibility RustVisibility) {
    LLVMSetVisibility(V, from_rust(RustVisibility));
}
