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

extern "C" void LLVMRustAddCallSiteAttribute(LLVMValueRef Instr, unsigned index, uint64_t Val) {
  CallSite Call = CallSite(unwrap<Instruction>(Instr));
  AttrBuilder B;
  B.addRawValue(Val);
  Call.setAttributes(
    Call.getAttributes().addAttributes(Call->getContext(), index,
                                       AttributeSet::get(Call->getContext(),
                                                         index, B)));
}


extern "C" void LLVMRustAddDereferenceableCallSiteAttr(LLVMValueRef Instr,
						       unsigned idx,
						       uint64_t b)
{
  CallSite Call = CallSite(unwrap<Instruction>(Instr));
  AttrBuilder B;
  B.addDereferenceableAttr(b);
  Call.setAttributes(
    Call.getAttributes().addAttributes(Call->getContext(), idx,
                                       AttributeSet::get(Call->getContext(),
                                                         idx, B)));
}

extern "C" void LLVMRustAddFunctionAttribute(LLVMValueRef Fn,
					     unsigned index,
					     uint64_t Val)
{
  Function *A = unwrap<Function>(Fn);
  AttrBuilder B;
  B.addRawValue(Val);
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

extern "C" void LLVMRustAddFunctionAttrString(LLVMValueRef Fn,
					      unsigned index,
					      const char *Name)
{
  Function *F = unwrap<Function>(Fn);
  AttrBuilder B;
  B.addAttribute(Name);
  F->addAttributes(index, AttributeSet::get(F->getContext(), index, B));
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
						 uint64_t Val)
{
  Function *A = unwrap<Function>(Fn);
  const AttributeSet PAL = A->getAttributes();
  AttrBuilder B(Val);
  const AttributeSet PALnew =
    PAL.removeAttributes(A->getContext(), index,
                         AttributeSet::get(A->getContext(), index, B));
  A->setAttributes(PALnew);
}

extern "C" void LLVMRustRemoveFunctionAttrString(LLVMValueRef fn,
						 unsigned index,
						 const char *Name)
{
  Function *f = unwrap<Function>(fn);
  LLVMContext &C = f->getContext();
  AttrBuilder B;
  B.addAttribute(Name);
  AttributeSet to_remove = AttributeSet::get(C, index, B);

  AttributeSet attrs = f->getAttributes();
  f->setAttributes(attrs.removeAttributes(f->getContext(),
                                          index,
                                          to_remove));
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
#if LLVM_VERSION_MINOR == 7
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
    unsigned Flags,
    bool isOptimized,
    LLVMValueRef Fn,
    LLVMRustMetadataRef TParam,
    LLVMRustMetadataRef Decl) {
#if LLVM_VERSION_MINOR >= 8
    DITemplateParameterArray TParams =
        DITemplateParameterArray(unwrap<MDTuple>(TParam));
    DISubprogram *Sub = Builder->createFunction(
        unwrapDI<DIScope>(Scope), Name, LinkageName,
        unwrapDI<DIFile>(File), LineNo,
        unwrapDI<DISubroutineType>(Ty), isLocalToUnit, isDefinition, ScopeLine,
        Flags, isOptimized,
        TParams,
        unwrapDIptr<DISubprogram>(Decl));
    unwrap<Function>(Fn)->setSubprogram(Sub);
    return wrap(Sub);
#else
    return wrap(Builder->createFunction(
        unwrapDI<DIScope>(Scope), Name, LinkageName,
        unwrapDI<DIFile>(File), LineNo,
        unwrapDI<DISubroutineType>(Ty), isLocalToUnit, isDefinition, ScopeLine,
        Flags, isOptimized,
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
    unsigned Flags,
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
        Flags,
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
    unsigned Flags,
    LLVMRustMetadataRef Ty) {
    return wrap(Builder->createMemberType(
        unwrapDI<DIDescriptor>(Scope), Name,
        unwrapDI<DIFile>(File), LineNo,
        SizeInBits, AlignInBits, OffsetInBits, Flags,
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
    unsigned Flags,
    int64_t* AddrOps,
    unsigned AddrOpsCount,
    unsigned ArgNo) {
#if LLVM_VERSION_MINOR >= 8
    if (Tag == 0x100) { // DW_TAG_auto_variable
        return wrap(Builder->createAutoVariable(
            unwrapDI<DIDescriptor>(Scope), Name,
            unwrapDI<DIFile>(File),
            LineNo,
            unwrapDI<DIType>(Ty), AlwaysPreserve, Flags));
    } else {
        return wrap(Builder->createParameterVariable(
            unwrapDI<DIDescriptor>(Scope), Name, ArgNo,
            unwrapDI<DIFile>(File),
            LineNo,
            unwrapDI<DIType>(Ty), AlwaysPreserve, Flags));
    }
#else
    return wrap(Builder->createLocalVariable(Tag,
        unwrapDI<DIDescriptor>(Scope), Name,
        unwrapDI<DIFile>(File),
        LineNo,
        unwrapDI<DIType>(Ty), AlwaysPreserve, Flags, ArgNo));
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

extern "C" LLVMValueRef LLVMRustDIBuilderInsertDeclareBefore(
    LLVMRustDIBuilderRef Builder,
    LLVMValueRef Val,
    LLVMRustMetadataRef VarInfo,
    int64_t* AddrOps,
    unsigned AddrOpsCount,
    LLVMValueRef DL,
    LLVMValueRef InsertBefore) {
    return wrap(Builder->insertDeclare(
        unwrap(Val),
        unwrap<DILocalVariable>(VarInfo),
        Builder->createExpression(
          llvm::ArrayRef<int64_t>(AddrOps, AddrOpsCount)),
        DebugLoc(cast<MDNode>(unwrap<MetadataAsValue>(DL)->getMetadata())),
        unwrap<Instruction>(InsertBefore)));
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
    unsigned Flags,
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
        Flags,
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
        LineNo));
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
    ErrorOr<std::unique_ptr<Module>> Src =
        llvm::getLazyBitcodeModule(std::move(buf), Dst->getContext());
    if (!Src) {
        LLVMRustSetLastError(Src.getError().message().c_str());
        return false;
    }

    std::string Err;

    raw_string_ostream Stream(Err);
    DiagnosticPrinterRawOStream DP(Stream);
#if LLVM_VERSION_MINOR >= 8
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
    const char **pass_name_out,
    LLVMValueRef *function_out,
    LLVMDebugLocRef *debugloc_out,
    LLVMTwineRef *message_out)
{
    // Undefined to call this not on an optimization diagnostic!
    llvm::DiagnosticInfoOptimizationBase *opt
        = static_cast<llvm::DiagnosticInfoOptimizationBase*>(unwrap(di));

    *pass_name_out = opt->getPassName();
    *function_out = wrap(&opt->getFunction());
    *debugloc_out = wrap(&opt->getDebugLoc());
    *message_out = wrap(&opt->getMsg());
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
#if LLVM_VERSION_MINOR >= 8
    case DK_OptimizationRemarkAnalysisFPCommute:
        return LLVMRustDiagnosticKind::OptimizationRemarkAnalysisFPCommute;
    case DK_OptimizationRemarkAnalysisAliasing:
        return LLVMRustDiagnosticKind::OptimizationRemarkAnalysisAliasing;
#endif
    default:
#if LLVM_VERSION_MINOR >= 9
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
#if LLVM_VERSION_MINOR >= 8
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
#if LLVM_VERSION_MINOR >= 8
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
#if LLVM_VERSION_MINOR >= 8
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
#if LLVM_VERSION_MINOR >= 8
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
#if LLVM_VERSION_MINOR >= 8
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
#if LLVM_VERSION_MINOR >= 8
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
#if LLVM_VERSION_MINOR >= 8
    Value *CatchSwitch = unwrap(CatchSwitchRef);
    cast<CatchSwitchInst>(CatchSwitch)->addHandler(unwrap(Handler));
#endif
}

extern "C" void
LLVMRustSetPersonalityFn(LLVMBuilderRef B,
                         LLVMValueRef Personality) {
#if LLVM_VERSION_MINOR >= 8
    unwrap(B)->GetInsertBlock()
             ->getParent()
             ->setPersonalityFn(cast<Function>(unwrap(Personality)));
#endif
}

#if LLVM_VERSION_MINOR >= 8
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
