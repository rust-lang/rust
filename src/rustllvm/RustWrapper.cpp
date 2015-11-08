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

extern "C" LLVMValueRef LLVMRustConstSmallInt(LLVMTypeRef IntTy, unsigned N,
                                              LLVMBool SignExtend) {
  return LLVMConstInt(IntTy, (unsigned long long)N, SignExtend);
}

extern "C" LLVMValueRef LLVMRustConstInt(LLVMTypeRef IntTy,
           unsigned N_hi,
           unsigned N_lo,
           LLVMBool SignExtend) {
  unsigned long long N = N_hi;
  N <<= 32;
  N |= N_lo;
  return LLVMConstInt(IntTy, N, SignExtend);
}

extern "C" void LLVMRustPrintPassTimings() {
  raw_fd_ostream OS (2, false); // stderr.
  TimerGroup::printAll(OS);
}

extern "C" LLVMValueRef LLVMGetNamedValue(LLVMModuleRef M,
                                          const char* Name) {
    return wrap(unwrap(M)->getNamedValue(Name));
}

extern "C" LLVMValueRef LLVMGetOrInsertFunction(LLVMModuleRef M,
                                                const char* Name,
                                                LLVMTypeRef FunctionTy) {
  return wrap(unwrap(M)->getOrInsertFunction(Name,
                                             unwrap<FunctionType>(FunctionTy)));
}

extern "C" LLVMValueRef LLVMGetOrInsertGlobal(LLVMModuleRef M,
                                              const char* Name,
                                              LLVMTypeRef Ty) {
  return wrap(unwrap(M)->getOrInsertGlobal(Name, unwrap(Ty)));
}

extern "C" LLVMTypeRef LLVMMetadataTypeInContext(LLVMContextRef C) {
  return wrap(Type::getMetadataTy(*unwrap(C)));
}

extern "C" void LLVMAddCallSiteAttribute(LLVMValueRef Instr, unsigned index, uint64_t Val) {
  CallSite Call = CallSite(unwrap<Instruction>(Instr));
  AttrBuilder B;
  B.addRawValue(Val);
  Call.setAttributes(
    Call.getAttributes().addAttributes(Call->getContext(), index,
                                       AttributeSet::get(Call->getContext(),
                                                         index, B)));
}


extern "C" void LLVMAddDereferenceableCallSiteAttr(LLVMValueRef Instr, unsigned idx, uint64_t b) {
  CallSite Call = CallSite(unwrap<Instruction>(Instr));
  AttrBuilder B;
  B.addDereferenceableAttr(b);
  Call.setAttributes(
    Call.getAttributes().addAttributes(Call->getContext(), idx,
                                       AttributeSet::get(Call->getContext(),
                                                         idx, B)));
}

extern "C" void LLVMAddFunctionAttribute(LLVMValueRef Fn, unsigned index,
                                         uint64_t Val) {
  Function *A = unwrap<Function>(Fn);
  AttrBuilder B;
  B.addRawValue(Val);
  A->addAttributes(index, AttributeSet::get(A->getContext(), index, B));
}

extern "C" void LLVMAddDereferenceableAttr(LLVMValueRef Fn, unsigned index, uint64_t bytes) {
  Function *A = unwrap<Function>(Fn);
  AttrBuilder B;
  B.addDereferenceableAttr(bytes);
  A->addAttributes(index, AttributeSet::get(A->getContext(), index, B));
}

extern "C" void LLVMAddFunctionAttrString(LLVMValueRef Fn, unsigned index, const char *Name) {
  Function *F = unwrap<Function>(Fn);
  AttrBuilder B;
  B.addAttribute(Name);
  F->addAttributes(index, AttributeSet::get(F->getContext(), index, B));
}

extern "C" void LLVMAddFunctionAttrStringValue(LLVMValueRef Fn, unsigned index,
                                               const char *Name,
                                               const char *Value) {
  Function *F = unwrap<Function>(Fn);
  AttrBuilder B;
  B.addAttribute(Name, Value);
  F->addAttributes(index, AttributeSet::get(F->getContext(), index, B));
}

extern "C" void LLVMRemoveFunctionAttrString(LLVMValueRef fn, unsigned index, const char *Name) {
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

extern "C" LLVMValueRef LLVMBuildAtomicLoad(LLVMBuilderRef B,
                                            LLVMValueRef source,
                                            const char* Name,
                                            AtomicOrdering order,
                                            unsigned alignment) {
    LoadInst* li = new LoadInst(unwrap(source),0);
    li->setVolatile(true);
    li->setAtomic(order);
    li->setAlignment(alignment);
    return wrap(unwrap(B)->Insert(li, Name));
}

extern "C" LLVMValueRef LLVMBuildAtomicStore(LLVMBuilderRef B,
                                             LLVMValueRef val,
                                             LLVMValueRef target,
                                             AtomicOrdering order,
                                             unsigned alignment) {
    StoreInst* si = new StoreInst(unwrap(val),unwrap(target));
    si->setVolatile(true);
    si->setAtomic(order);
    si->setAlignment(alignment);
    return wrap(unwrap(B)->Insert(si));
}

extern "C" LLVMValueRef LLVMBuildAtomicCmpXchg(LLVMBuilderRef B,
                                               LLVMValueRef target,
                                               LLVMValueRef old,
                                               LLVMValueRef source,
                                               AtomicOrdering order,
                                               AtomicOrdering failure_order) {
    return wrap(unwrap(B)->CreateAtomicCmpXchg(unwrap(target), unwrap(old),
                                               unwrap(source), order,
                                               failure_order
                                               ));
}
extern "C" LLVMValueRef LLVMBuildAtomicFence(LLVMBuilderRef B,
                                             AtomicOrdering order,
                                             SynchronizationScope scope) {
    return wrap(unwrap(B)->CreateFence(order, scope));
}

extern "C" void LLVMSetDebug(int Enabled) {
#ifndef NDEBUG
  DebugFlag = Enabled;
#endif
}

extern "C" LLVMValueRef LLVMInlineAsm(LLVMTypeRef Ty,
                                      char *AsmString,
                                      char *Constraints,
                                      LLVMBool HasSideEffects,
                                      LLVMBool IsAlignStack,
                                      unsigned Dialect) {
    return wrap(InlineAsm::get(unwrap<FunctionType>(Ty), AsmString,
                               Constraints, HasSideEffects,
                               IsAlignStack, (InlineAsm::AsmDialect) Dialect));
}

typedef DIBuilder* DIBuilderRef;

#if LLVM_VERSION_MINOR >= 6
typedef struct LLVMOpaqueMetadata *LLVMMetadataRef;

namespace llvm {
DEFINE_ISA_CONVERSION_FUNCTIONS(Metadata, LLVMMetadataRef)

inline Metadata **unwrap(LLVMMetadataRef *Vals) {
  return reinterpret_cast<Metadata**>(Vals);
}
}
#else
typedef LLVMValueRef LLVMMetadataRef;
#endif

template<typename DIT>
DIT* unwrapDIptr(LLVMMetadataRef ref) {
    return (DIT*) (ref ? unwrap<MDNode>(ref) : NULL);
}

#if LLVM_VERSION_MINOR <= 6
template<typename DIT>
DIT unwrapDI(LLVMMetadataRef ref) {
    return DIT(ref ? unwrap<MDNode>(ref) : NULL);
}
#else
#define DIDescriptor DIScope
#define DIArray DINodeArray
#define unwrapDI unwrapDIptr
#endif

#if LLVM_VERSION_MINOR <= 5
#define DISubroutineType DICompositeType
#endif

extern "C" uint32_t LLVMRustDebugMetadataVersion() {
    return DEBUG_METADATA_VERSION;
}

extern "C" uint32_t LLVMVersionMinor() {
  return LLVM_VERSION_MINOR;
}

extern "C" uint32_t LLVMVersionMajor() {
  return LLVM_VERSION_MAJOR;
}

extern "C" void LLVMRustAddModuleFlag(LLVMModuleRef M,
                                      const char *name,
                                      uint32_t value) {
    unwrap(M)->addModuleFlag(Module::Warning, name, value);
}

extern "C" DIBuilderRef LLVMDIBuilderCreate(LLVMModuleRef M) {
    return new DIBuilder(*unwrap(M));
}

extern "C" void LLVMDIBuilderDispose(DIBuilderRef Builder) {
    delete Builder;
}

extern "C" void LLVMDIBuilderFinalize(DIBuilderRef Builder) {
    Builder->finalize();
}

extern "C" LLVMMetadataRef LLVMDIBuilderCreateCompileUnit(
    DIBuilderRef Builder,
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

extern "C" LLVMMetadataRef LLVMDIBuilderCreateFile(
    DIBuilderRef Builder,
    const char* Filename,
    const char* Directory) {
    return wrap(Builder->createFile(Filename, Directory));
}

extern "C" LLVMMetadataRef LLVMDIBuilderCreateSubroutineType(
    DIBuilderRef Builder,
    LLVMMetadataRef File,
    LLVMMetadataRef ParameterTypes) {
    return wrap(Builder->createSubroutineType(
        unwrapDI<DIFile>(File),
#if LLVM_VERSION_MINOR >= 7
        DITypeRefArray(unwrap<MDTuple>(ParameterTypes))));
#elif LLVM_VERSION_MINOR >= 6
        unwrapDI<DITypeArray>(ParameterTypes)));
#else
        unwrapDI<DIArray>(ParameterTypes)));
#endif
}

extern "C" LLVMMetadataRef LLVMDIBuilderCreateFunction(
    DIBuilderRef Builder,
    LLVMMetadataRef Scope,
    const char* Name,
    const char* LinkageName,
    LLVMMetadataRef File,
    unsigned LineNo,
    LLVMMetadataRef Ty,
    bool isLocalToUnit,
    bool isDefinition,
    unsigned ScopeLine,
    unsigned Flags,
    bool isOptimized,
    LLVMValueRef Fn,
    LLVMMetadataRef TParam,
    LLVMMetadataRef Decl) {
    return wrap(Builder->createFunction(
        unwrapDI<DIScope>(Scope), Name, LinkageName,
        unwrapDI<DIFile>(File), LineNo,
        unwrapDI<DISubroutineType>(Ty), isLocalToUnit, isDefinition, ScopeLine,
        Flags, isOptimized,
        unwrap<Function>(Fn),
        unwrapDIptr<MDNode>(TParam),
        unwrapDIptr<MDNode>(Decl)));
}

extern "C" LLVMMetadataRef LLVMDIBuilderCreateBasicType(
    DIBuilderRef Builder,
    const char* Name,
    uint64_t SizeInBits,
    uint64_t AlignInBits,
    unsigned Encoding) {
    return wrap(Builder->createBasicType(
        Name, SizeInBits,
        AlignInBits, Encoding));
}

extern "C" LLVMMetadataRef LLVMDIBuilderCreatePointerType(
    DIBuilderRef Builder,
    LLVMMetadataRef PointeeTy,
    uint64_t SizeInBits,
    uint64_t AlignInBits,
    const char* Name) {
    return wrap(Builder->createPointerType(
        unwrapDI<DIType>(PointeeTy), SizeInBits, AlignInBits, Name));
}

extern "C" LLVMMetadataRef LLVMDIBuilderCreateStructType(
    DIBuilderRef Builder,
    LLVMMetadataRef Scope,
    const char* Name,
    LLVMMetadataRef File,
    unsigned LineNumber,
    uint64_t SizeInBits,
    uint64_t AlignInBits,
    unsigned Flags,
    LLVMMetadataRef DerivedFrom,
    LLVMMetadataRef Elements,
    unsigned RunTimeLang,
    LLVMMetadataRef VTableHolder,
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
#if LLVM_VERSION_MINOR >= 7
        DINodeArray(unwrapDI<MDTuple>(Elements)),
#else
        unwrapDI<DIArray>(Elements),
#endif
        RunTimeLang,
        unwrapDI<DIType>(VTableHolder),
        UniqueId
        ));
}

extern "C" LLVMMetadataRef LLVMDIBuilderCreateMemberType(
    DIBuilderRef Builder,
    LLVMMetadataRef Scope,
    const char* Name,
    LLVMMetadataRef File,
    unsigned LineNo,
    uint64_t SizeInBits,
    uint64_t AlignInBits,
    uint64_t OffsetInBits,
    unsigned Flags,
    LLVMMetadataRef Ty) {
    return wrap(Builder->createMemberType(
        unwrapDI<DIDescriptor>(Scope), Name,
        unwrapDI<DIFile>(File), LineNo,
        SizeInBits, AlignInBits, OffsetInBits, Flags,
        unwrapDI<DIType>(Ty)));
}

extern "C" LLVMMetadataRef LLVMDIBuilderCreateLexicalBlock(
    DIBuilderRef Builder,
    LLVMMetadataRef Scope,
    LLVMMetadataRef File,
    unsigned Line,
    unsigned Col) {
    return wrap(Builder->createLexicalBlock(
        unwrapDI<DIDescriptor>(Scope),
        unwrapDI<DIFile>(File), Line, Col
#if LLVM_VERSION_MINOR == 5
        , 0
#endif
        ));
}

extern "C" LLVMMetadataRef LLVMDIBuilderCreateStaticVariable(
    DIBuilderRef Builder,
    LLVMMetadataRef Context,
    const char* Name,
    const char* LinkageName,
    LLVMMetadataRef File,
    unsigned LineNo,
    LLVMMetadataRef Ty,
    bool isLocalToUnit,
    LLVMValueRef Val,
    LLVMMetadataRef Decl = NULL) {
#if LLVM_VERSION_MINOR >= 6
    return wrap(Builder->createGlobalVariable(unwrapDI<DIDescriptor>(Context),
#else
    return wrap(Builder->createStaticVariable(unwrapDI<DIDescriptor>(Context),
#endif
        Name,
        LinkageName,
        unwrapDI<DIFile>(File),
        LineNo,
        unwrapDI<DIType>(Ty),
        isLocalToUnit,
        cast<Constant>(unwrap(Val)),
        unwrapDIptr<MDNode>(Decl)));
}

extern "C" LLVMMetadataRef LLVMDIBuilderCreateVariable(
    DIBuilderRef Builder,
    unsigned Tag,
    LLVMMetadataRef Scope,
    const char* Name,
    LLVMMetadataRef File,
    unsigned LineNo,
    LLVMMetadataRef Ty,
    bool AlwaysPreserve,
    unsigned Flags,
    int64_t* AddrOps,
    unsigned AddrOpsCount,
    unsigned ArgNo) {
#if LLVM_VERSION_MINOR == 5
    if (AddrOpsCount > 0) {
        SmallVector<llvm::Value *, 16> addr_ops;
        llvm::Type *Int64Ty = Type::getInt64Ty(unwrap<MDNode>(Scope)->getContext());
        for (unsigned i = 0; i < AddrOpsCount; ++i)
            addr_ops.push_back(ConstantInt::get(Int64Ty, AddrOps[i]));

        return wrap(Builder->createComplexVariable(
            Tag,
            unwrapDI<DIDescriptor>(Scope),
            Name,
            unwrapDI<DIFile>(File),
            LineNo,
            unwrapDI<DIType>(Ty),
            addr_ops,
            ArgNo
        ));
    }
#endif
    return wrap(Builder->createLocalVariable(Tag,
        unwrapDI<DIDescriptor>(Scope), Name,
        unwrapDI<DIFile>(File),
        LineNo,
        unwrapDI<DIType>(Ty), AlwaysPreserve, Flags, ArgNo));
}

extern "C" LLVMMetadataRef LLVMDIBuilderCreateArrayType(
    DIBuilderRef Builder,
    uint64_t Size,
    uint64_t AlignInBits,
    LLVMMetadataRef Ty,
    LLVMMetadataRef Subscripts) {
    return wrap(Builder->createArrayType(Size, AlignInBits,
        unwrapDI<DIType>(Ty),
#if LLVM_VERSION_MINOR >= 7
        DINodeArray(unwrapDI<MDTuple>(Subscripts))
#else
        unwrapDI<DIArray>(Subscripts)
#endif
    ));
}

extern "C" LLVMMetadataRef LLVMDIBuilderCreateVectorType(
    DIBuilderRef Builder,
    uint64_t Size,
    uint64_t AlignInBits,
    LLVMMetadataRef Ty,
    LLVMMetadataRef Subscripts) {
    return wrap(Builder->createVectorType(Size, AlignInBits,
        unwrapDI<DIType>(Ty),
#if LLVM_VERSION_MINOR >= 7
        DINodeArray(unwrapDI<MDTuple>(Subscripts))
#else
        unwrapDI<DIArray>(Subscripts)
#endif
    ));
}

extern "C" LLVMMetadataRef LLVMDIBuilderGetOrCreateSubrange(
    DIBuilderRef Builder,
    int64_t Lo,
    int64_t Count) {
    return wrap(Builder->getOrCreateSubrange(Lo, Count));
}

extern "C" LLVMMetadataRef LLVMDIBuilderGetOrCreateArray(
    DIBuilderRef Builder,
    LLVMMetadataRef* Ptr,
    unsigned Count) {
#if LLVM_VERSION_MINOR >= 7
    Metadata **DataValue = unwrap(Ptr);
    return wrap(Builder->getOrCreateArray(
        ArrayRef<Metadata*>(DataValue, Count)).get());
#else
    return wrap(Builder->getOrCreateArray(
#if LLVM_VERSION_MINOR >= 6
        ArrayRef<Metadata*>(unwrap(Ptr), Count)));
#else
        ArrayRef<Value*>(reinterpret_cast<Value**>(Ptr), Count)));
#endif
#endif
}

extern "C" LLVMValueRef LLVMDIBuilderInsertDeclareAtEnd(
    DIBuilderRef Builder,
    LLVMValueRef Val,
    LLVMMetadataRef VarInfo,
    int64_t* AddrOps,
    unsigned AddrOpsCount,
    LLVMValueRef DL,
    LLVMBasicBlockRef InsertAtEnd) {
    return wrap(Builder->insertDeclare(
        unwrap(Val),
#if LLVM_VERSION_MINOR >= 7
        unwrap<DILocalVariable>(VarInfo),
#else
        unwrapDI<DIVariable>(VarInfo),
#endif
#if LLVM_VERSION_MINOR >= 6
        Builder->createExpression(
          llvm::ArrayRef<int64_t>(AddrOps, AddrOpsCount)),
#endif
#if LLVM_VERSION_MINOR >= 7
        DebugLoc(cast<MDNode>(unwrap<MetadataAsValue>(DL)->getMetadata())),
#endif
        unwrap(InsertAtEnd)));
}

extern "C" LLVMValueRef LLVMDIBuilderInsertDeclareBefore(
    DIBuilderRef Builder,
    LLVMValueRef Val,
    LLVMMetadataRef VarInfo,
    int64_t* AddrOps,
    unsigned AddrOpsCount,
    LLVMValueRef DL,
    LLVMValueRef InsertBefore) {
#if LLVM_VERSION_MINOR >= 6
#endif
    return wrap(Builder->insertDeclare(
        unwrap(Val),
#if LLVM_VERSION_MINOR >= 7
        unwrap<DILocalVariable>(VarInfo),
#else
        unwrapDI<DIVariable>(VarInfo),
#endif
#if LLVM_VERSION_MINOR >= 6
        Builder->createExpression(
          llvm::ArrayRef<int64_t>(AddrOps, AddrOpsCount)),
#endif
#if LLVM_VERSION_MINOR >= 7
        DebugLoc(cast<MDNode>(unwrap<MetadataAsValue>(DL)->getMetadata())),
#endif
        unwrap<Instruction>(InsertBefore)));
}

extern "C" LLVMMetadataRef LLVMDIBuilderCreateEnumerator(
    DIBuilderRef Builder,
    const char* Name,
    uint64_t Val)
{
    return wrap(Builder->createEnumerator(Name, Val));
}

extern "C" LLVMMetadataRef LLVMDIBuilderCreateEnumerationType(
    DIBuilderRef Builder,
    LLVMMetadataRef Scope,
    const char* Name,
    LLVMMetadataRef File,
    unsigned LineNumber,
    uint64_t SizeInBits,
    uint64_t AlignInBits,
    LLVMMetadataRef Elements,
    LLVMMetadataRef ClassType)
{
    return wrap(Builder->createEnumerationType(
        unwrapDI<DIDescriptor>(Scope),
        Name,
        unwrapDI<DIFile>(File),
        LineNumber,
        SizeInBits,
        AlignInBits,
#if LLVM_VERSION_MINOR >= 7
        DINodeArray(unwrapDI<MDTuple>(Elements)),
#else
        unwrapDI<DIArray>(Elements),
#endif
        unwrapDI<DIType>(ClassType)));
}

extern "C" LLVMMetadataRef LLVMDIBuilderCreateUnionType(
    DIBuilderRef Builder,
    LLVMMetadataRef Scope,
    const char* Name,
    LLVMMetadataRef File,
    unsigned LineNumber,
    uint64_t SizeInBits,
    uint64_t AlignInBits,
    unsigned Flags,
    LLVMMetadataRef Elements,
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
#if LLVM_VERSION_MINOR >= 7
        DINodeArray(unwrapDI<MDTuple>(Elements)),
#else
        unwrapDI<DIArray>(Elements),
#endif
        RunTimeLang,
        UniqueId
        ));
}

extern "C" LLVMMetadataRef LLVMDIBuilderCreateTemplateTypeParameter(
    DIBuilderRef Builder,
    LLVMMetadataRef Scope,
    const char* Name,
    LLVMMetadataRef Ty,
    LLVMMetadataRef File,
    unsigned LineNo,
    unsigned ColumnNo)
{
    return wrap(Builder->createTemplateTypeParameter(
      unwrapDI<DIDescriptor>(Scope),
      Name,
      unwrapDI<DIType>(Ty)
#if LLVM_VERSION_MINOR <= 6
      ,
      unwrapDI<MDNode*>(File),
      LineNo,
      ColumnNo
#endif
      ));
}

extern "C" int64_t LLVMDIBuilderCreateOpDeref()
{
    return dwarf::DW_OP_deref;
}

extern "C" int64_t LLVMDIBuilderCreateOpPlus()
{
    return dwarf::DW_OP_plus;
}

extern "C" LLVMMetadataRef LLVMDIBuilderCreateNameSpace(
    DIBuilderRef Builder,
    LLVMMetadataRef Scope,
    const char* Name,
    LLVMMetadataRef File,
    unsigned LineNo)
{
    return wrap(Builder->createNameSpace(
        unwrapDI<DIDescriptor>(Scope),
        Name,
        unwrapDI<DIFile>(File),
        LineNo));
}

extern "C" void LLVMDICompositeTypeSetTypeArray(
    DIBuilderRef Builder,
    LLVMMetadataRef CompositeType,
    LLVMMetadataRef TypeArray)
{
#if LLVM_VERSION_MINOR >= 7
    DICompositeType *tmp = unwrapDI<DICompositeType>(CompositeType);
    Builder->replaceArrays(tmp, DINodeArray(unwrap<MDTuple>(TypeArray)));
#elif LLVM_VERSION_MINOR >= 6
    DICompositeType tmp = unwrapDI<DICompositeType>(CompositeType);
    Builder->replaceArrays(tmp, unwrapDI<DIArray>(TypeArray));
#else
    unwrapDI<DICompositeType>(CompositeType).setTypeArray(unwrapDI<DIArray>(TypeArray));
#endif
}

extern "C" LLVMValueRef LLVMDIBuilderCreateDebugLocation(
  LLVMContextRef Context,
  unsigned Line,
  unsigned Column,
  LLVMMetadataRef Scope,
  LLVMMetadataRef InlinedAt) {

    LLVMContext& context = *unwrap(Context);

    DebugLoc debug_loc = DebugLoc::get(Line,
                                       Column,
                                       unwrapDIptr<MDNode>(Scope),
                                       unwrapDIptr<MDNode>(InlinedAt));

#if LLVM_VERSION_MINOR >= 6
    return wrap(MetadataAsValue::get(context, debug_loc.getAsMDNode(
#if LLVM_VERSION_MINOR <= 6
            context
#endif
        )));
#else
    return wrap(debug_loc.getAsMDNode(context));
#endif
}

extern "C" void LLVMWriteTypeToString(LLVMTypeRef Type, RustStringRef str) {
    raw_rust_string_ostream os(str);
    unwrap<llvm::Type>(Type)->print(os);
}

extern "C" void LLVMWriteValueToString(LLVMValueRef Value, RustStringRef str) {
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
#if LLVM_VERSION_MINOR >= 6
    std::unique_ptr<MemoryBuffer> buf = MemoryBuffer::getMemBufferCopy(StringRef(bc, len));
#if LLVM_VERSION_MINOR >= 7
    ErrorOr<std::unique_ptr<Module>> Src =
        llvm::getLazyBitcodeModule(std::move(buf), Dst->getContext());
#else
    ErrorOr<Module *> Src = llvm::getLazyBitcodeModule(std::move(buf), Dst->getContext());
#endif
#else
    MemoryBuffer* buf = MemoryBuffer::getMemBufferCopy(StringRef(bc, len));
    ErrorOr<Module *> Src = llvm::getLazyBitcodeModule(buf, Dst->getContext());
#endif
    if (!Src) {
        LLVMRustSetLastError(Src.getError().message().c_str());
#if LLVM_VERSION_MINOR == 5
        delete buf;
#endif
        return false;
    }

    std::string Err;

#if LLVM_VERSION_MINOR >= 6
    raw_string_ostream Stream(Err);
    DiagnosticPrinterRawOStream DP(Stream);
#if LLVM_VERSION_MINOR >= 7
    if (Linker::LinkModules(Dst, Src->get(), [&](const DiagnosticInfo &DI) { DI.print(DP); })) {
#else
    if (Linker::LinkModules(Dst, *Src, [&](const DiagnosticInfo &DI) { DI.print(DP); })) {
#endif
#else
    if (Linker::LinkModules(Dst, *Src, Linker::DestroySource, &Err)) {
#endif
        LLVMRustSetLastError(Err.c_str());
        return false;
    }
    return true;
}

extern "C" void
LLVMRustSetDLLStorageClass(LLVMValueRef Value,
                           GlobalValue::DLLStorageClassTypes Class) {
    GlobalValue *V = unwrap<GlobalValue>(Value);
    V->setDLLStorageClass(Class);
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

extern "C" int
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
LLVMWriteTwineToString(LLVMTwineRef T, RustStringRef str) {
    raw_rust_string_ostream os(str);
    unwrap(T)->print(os);
}

extern "C" void
LLVMUnpackOptimizationDiagnostic(
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
LLVMUnpackInlineAsmDiagnostic(
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

extern "C" void LLVMWriteDiagnosticInfoToString(LLVMDiagnosticInfoRef di, RustStringRef str) {
    raw_rust_string_ostream os(str);
    DiagnosticPrinterRawOStream dp(os);
    unwrap(di)->print(dp);
}

extern "C" int LLVMGetDiagInfoKind(LLVMDiagnosticInfoRef di) {
    return unwrap(di)->getKind();
}

extern "C" void LLVMWriteDebugLocToString(
    LLVMContextRef C,
    LLVMDebugLocRef dl,
    RustStringRef str)
{
    raw_rust_string_ostream os(str);
#if LLVM_VERSION_MINOR >= 7
    unwrap(dl)->print(os);
#else
    unwrap(dl)->print(*unwrap(C), os);
#endif
}

DEFINE_SIMPLE_CONVERSION_FUNCTIONS(SMDiagnostic, LLVMSMDiagnosticRef)

extern "C" void LLVMSetInlineAsmDiagnosticHandler(
    LLVMContextRef C,
    LLVMContext::InlineAsmDiagHandlerTy H,
    void *CX)
{
    unwrap(C)->setInlineAsmDiagnosticHandler(H, CX);
}

extern "C" void LLVMWriteSMDiagnosticToString(LLVMSMDiagnosticRef d, RustStringRef str) {
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
