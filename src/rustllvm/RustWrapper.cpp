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

//===----------------------------------------------------------------------===
//
// This file defines alternate interfaces to core functions that are more
// readily callable by Rust's FFI.
//
//===----------------------------------------------------------------------===

using namespace llvm;
using namespace llvm::sys;

const char *LLVMRustError;

extern "C" LLVMMemoryBufferRef
LLVMRustCreateMemoryBufferWithContentsOfFile(const char *Path) {
  LLVMMemoryBufferRef MemBuf = NULL;
  LLVMCreateMemoryBufferWithContentsOfFile(Path, &MemBuf,
    const_cast<char **>(&LLVMRustError));
  return MemBuf;
}

extern "C" const char *LLVMRustGetLastError(void) {
  return LLVMRustError;
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

extern "C" LLVMValueRef LLVMGetOrInsertFunction(LLVMModuleRef M,
                                                const char* Name,
                                                LLVMTypeRef FunctionTy) {
  return wrap(unwrap(M)->getOrInsertFunction(Name,
                                             unwrap<FunctionType>(FunctionTy)));
}

extern "C" LLVMTypeRef LLVMMetadataTypeInContext(LLVMContextRef C) {
  return wrap(Type::getMetadataTy(*unwrap(C)));
}

extern "C" void LLVMAddFunctionAttrString(LLVMValueRef fn, const char *Name) {
  unwrap<Function>(fn)->addFnAttr(Name);
}


extern "C" void LLVMAddReturnAttribute(LLVMValueRef Fn, LLVMAttribute PA) {
  Function *A = unwrap<Function>(Fn);
  AttrBuilder B(PA);
  A->addAttributes(AttributeSet::ReturnIndex,
                   AttributeSet::get(A->getContext(), AttributeSet::ReturnIndex,  B));
}

extern "C" void LLVMRemoveReturnAttribute(LLVMValueRef Fn, LLVMAttribute PA) {
  Function *A = unwrap<Function>(Fn);
  AttrBuilder B(PA);
  A->removeAttributes(AttributeSet::ReturnIndex,
                      AttributeSet::get(A->getContext(), AttributeSet::ReturnIndex,  B));
}

extern "C" void LLVMAddColdAttribute(LLVMValueRef Fn) {
  Function *A = unwrap<Function>(Fn);
  A->addAttribute(AttributeSet::FunctionIndex, Attribute::Cold);
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
                                               AtomicOrdering order) {
    return wrap(unwrap(B)->CreateAtomicCmpXchg(unwrap(target), unwrap(old),
                                               unwrap(source), order));
}
extern "C" LLVMValueRef LLVMBuildAtomicFence(LLVMBuilderRef B, AtomicOrdering order) {
    return wrap(unwrap(B)->CreateFence(order));
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

/**
 * This function is intended to be a threadsafe interface into enabling a
 * multithreaded LLVM. This is invoked at the start of the translation phase of
 * compilation to ensure that LLVM is ready.
 *
 * All of trans properly isolates LLVM with the use of a different
 * LLVMContextRef per task, thus allowing parallel compilation of different
 * crates in the same process. At the time of this writing, the use case for
 * this is unit tests for rusti, but there are possible other applications.
 */
extern "C" bool LLVMRustStartMultithreading() {
    static Mutex lock;
    bool ret = true;
    assert(lock.acquire());
    if (!LLVMIsMultithreaded()) {
        ret = LLVMStartMultithreaded();
    }
    assert(lock.release());
    return ret;
}


typedef DIBuilder* DIBuilderRef;

template<typename DIT>
DIT unwrapDI(LLVMValueRef ref) {
    return DIT(ref ? unwrap<MDNode>(ref) : NULL);
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

extern "C" void LLVMDIBuilderCreateCompileUnit(
    DIBuilderRef Builder,
    unsigned Lang,
    const char* File,
    const char* Dir,
    const char* Producer,
    bool isOptimized,
    const char* Flags,
    unsigned RuntimeVer,
    const char* SplitName) {
    Builder->createCompileUnit(Lang, File, Dir, Producer, isOptimized,
        Flags, RuntimeVer, SplitName);
}

extern "C" LLVMValueRef LLVMDIBuilderCreateFile(
    DIBuilderRef Builder,
    const char* Filename,
    const char* Directory) {
    return wrap(Builder->createFile(Filename, Directory));
}

extern "C" LLVMValueRef LLVMDIBuilderCreateSubroutineType(
    DIBuilderRef Builder,
    LLVMValueRef File,
    LLVMValueRef ParameterTypes) {
    return wrap(Builder->createSubroutineType(
        unwrapDI<DIFile>(File),
        unwrapDI<DIArray>(ParameterTypes)));
}

extern "C" LLVMValueRef LLVMDIBuilderCreateFunction(
    DIBuilderRef Builder,
    LLVMValueRef Scope,
    const char* Name,
    const char* LinkageName,
    LLVMValueRef File,
    unsigned LineNo,
    LLVMValueRef Ty,
    bool isLocalToUnit,
    bool isDefinition,
    unsigned ScopeLine,
    unsigned Flags,
    bool isOptimized,
    LLVMValueRef Fn,
    LLVMValueRef TParam,
    LLVMValueRef Decl) {
    return wrap(Builder->createFunction(
        unwrapDI<DIScope>(Scope), Name, LinkageName,
        unwrapDI<DIFile>(File), LineNo,
        unwrapDI<DICompositeType>(Ty), isLocalToUnit, isDefinition, ScopeLine,
        Flags, isOptimized,
        unwrap<Function>(Fn),
        unwrapDI<MDNode*>(TParam),
        unwrapDI<MDNode*>(Decl)));
}

extern "C" LLVMValueRef LLVMDIBuilderCreateBasicType(
    DIBuilderRef Builder,
    const char* Name,
    uint64_t SizeInBits,
    uint64_t AlignInBits,
    unsigned Encoding) {
    return wrap(Builder->createBasicType(
        Name, SizeInBits,
        AlignInBits, Encoding));
}

extern "C" LLVMValueRef LLVMDIBuilderCreatePointerType(
    DIBuilderRef Builder,
    LLVMValueRef PointeeTy,
    uint64_t SizeInBits,
    uint64_t AlignInBits,
    const char* Name) {
    return wrap(Builder->createPointerType(
        unwrapDI<DIType>(PointeeTy), SizeInBits, AlignInBits, Name));
}

extern "C" LLVMValueRef LLVMDIBuilderCreateStructType(
    DIBuilderRef Builder,
    LLVMValueRef Scope,
    const char* Name,
    LLVMValueRef File,
    unsigned LineNumber,
    uint64_t SizeInBits,
    uint64_t AlignInBits,
    unsigned Flags,
    LLVMValueRef DerivedFrom,
    LLVMValueRef Elements,
    unsigned RunTimeLang,
    LLVMValueRef VTableHolder,
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
        unwrapDI<DIArray>(Elements),
        RunTimeLang,
        unwrapDI<DIType>(VTableHolder),
        UniqueId));
}

extern "C" LLVMValueRef LLVMDIBuilderCreateMemberType(
    DIBuilderRef Builder,
    LLVMValueRef Scope,
    const char* Name,
    LLVMValueRef File,
    unsigned LineNo,
    uint64_t SizeInBits,
    uint64_t AlignInBits,
    uint64_t OffsetInBits,
    unsigned Flags,
    LLVMValueRef Ty) {
    return wrap(Builder->createMemberType(
        unwrapDI<DIDescriptor>(Scope), Name,
        unwrapDI<DIFile>(File), LineNo,
        SizeInBits, AlignInBits, OffsetInBits, Flags,
        unwrapDI<DIType>(Ty)));
}

extern "C" LLVMValueRef LLVMDIBuilderCreateLexicalBlock(
    DIBuilderRef Builder,
    LLVMValueRef Scope,
    LLVMValueRef File,
    unsigned Line,
    unsigned Col) {
    return wrap(Builder->createLexicalBlock(
        unwrapDI<DIDescriptor>(Scope),
        unwrapDI<DIFile>(File), Line, Col));
}

extern "C" LLVMValueRef LLVMDIBuilderCreateLocalVariable(
    DIBuilderRef Builder,
    unsigned Tag,
    LLVMValueRef Scope,
    const char* Name,
    LLVMValueRef File,
    unsigned LineNo,
    LLVMValueRef Ty,
    bool AlwaysPreserve,
    unsigned Flags,
    unsigned ArgNo) {
    return wrap(Builder->createLocalVariable(Tag,
        unwrapDI<DIDescriptor>(Scope), Name,
        unwrapDI<DIFile>(File),
        LineNo,
        unwrapDI<DIType>(Ty), AlwaysPreserve, Flags, ArgNo));
}

extern "C" LLVMValueRef LLVMDIBuilderCreateArrayType(
    DIBuilderRef Builder,
    uint64_t Size,
    uint64_t AlignInBits,
    LLVMValueRef Ty,
    LLVMValueRef Subscripts) {
    return wrap(Builder->createArrayType(Size, AlignInBits,
        unwrapDI<DIType>(Ty),
        unwrapDI<DIArray>(Subscripts)));
}

extern "C" LLVMValueRef LLVMDIBuilderCreateVectorType(
    DIBuilderRef Builder,
    uint64_t Size,
    uint64_t AlignInBits,
    LLVMValueRef Ty,
    LLVMValueRef Subscripts) {
    return wrap(Builder->createVectorType(Size, AlignInBits,
        unwrapDI<DIType>(Ty),
        unwrapDI<DIArray>(Subscripts)));
}

extern "C" LLVMValueRef LLVMDIBuilderGetOrCreateSubrange(
    DIBuilderRef Builder,
    int64_t Lo,
    int64_t Count) {
    return wrap(Builder->getOrCreateSubrange(Lo, Count));
}

extern "C" LLVMValueRef LLVMDIBuilderGetOrCreateArray(
    DIBuilderRef Builder,
    LLVMValueRef* Ptr,
    unsigned Count) {
    return wrap(Builder->getOrCreateArray(
        ArrayRef<Value*>(reinterpret_cast<Value**>(Ptr), Count)));
}

extern "C" LLVMValueRef LLVMDIBuilderInsertDeclareAtEnd(
    DIBuilderRef Builder,
    LLVMValueRef Val,
    LLVMValueRef VarInfo,
    LLVMBasicBlockRef InsertAtEnd) {
    return wrap(Builder->insertDeclare(
        unwrap(Val),
        unwrapDI<DIVariable>(VarInfo),
        unwrap(InsertAtEnd)));
}

extern "C" LLVMValueRef LLVMDIBuilderInsertDeclareBefore(
    DIBuilderRef Builder,
    LLVMValueRef Val,
    LLVMValueRef VarInfo,
    LLVMValueRef InsertBefore) {
    return wrap(Builder->insertDeclare(
        unwrap(Val),
        unwrapDI<DIVariable>(VarInfo),
        unwrap<Instruction>(InsertBefore)));
}

extern "C" LLVMValueRef LLVMDIBuilderCreateEnumerator(
    DIBuilderRef Builder,
    const char* Name,
    uint64_t Val)
{
    return wrap(Builder->createEnumerator(Name, Val));
}

extern "C" LLVMValueRef LLVMDIBuilderCreateEnumerationType(
    DIBuilderRef Builder,
    LLVMValueRef Scope,
    const char* Name,
    LLVMValueRef File,
    unsigned LineNumber,
    uint64_t SizeInBits,
    uint64_t AlignInBits,
    LLVMValueRef Elements,
    LLVMValueRef ClassType)
{
    return wrap(Builder->createEnumerationType(
        unwrapDI<DIDescriptor>(Scope),
        Name,
        unwrapDI<DIFile>(File),
        LineNumber,
        SizeInBits,
        AlignInBits,
        unwrapDI<DIArray>(Elements),
        unwrapDI<DIType>(ClassType)));
}

extern "C" LLVMValueRef LLVMDIBuilderCreateUnionType(
    DIBuilderRef Builder,
    LLVMValueRef Scope,
    const char* Name,
    LLVMValueRef File,
    unsigned LineNumber,
    uint64_t SizeInBits,
    uint64_t AlignInBits,
    unsigned Flags,
    LLVMValueRef Elements,
    unsigned RunTimeLang)
{
    return wrap(Builder->createUnionType(
        unwrapDI<DIDescriptor>(Scope),
        Name,
        unwrapDI<DIFile>(File),
        LineNumber,
        SizeInBits,
        AlignInBits,
        Flags,
        unwrapDI<DIArray>(Elements),
        RunTimeLang));
}

extern "C" void LLVMSetUnnamedAddr(LLVMValueRef Value, LLVMBool Unnamed) {
    unwrap<GlobalValue>(Value)->setUnnamedAddr(Unnamed);
}

extern "C" LLVMValueRef LLVMDIBuilderCreateTemplateTypeParameter(
    DIBuilderRef Builder,
    LLVMValueRef Scope,
    const char* Name,
    LLVMValueRef Ty,
    LLVMValueRef File,
    unsigned LineNo,
    unsigned ColumnNo)
{
    return wrap(Builder->createTemplateTypeParameter(
      unwrapDI<DIDescriptor>(Scope),
      Name,
      unwrapDI<DIType>(Ty),
      unwrapDI<MDNode*>(File),
      LineNo,
      ColumnNo));
}

extern "C" LLVMValueRef LLVMDIBuilderCreateOpDeref(LLVMTypeRef IntTy)
{
    return LLVMConstInt(IntTy, DIBuilder::OpDeref, true);
}

extern "C" LLVMValueRef LLVMDIBuilderCreateOpPlus(LLVMTypeRef IntTy)
{
    return LLVMConstInt(IntTy, DIBuilder::OpPlus, true);
}

extern "C" LLVMValueRef LLVMDIBuilderCreateComplexVariable(
    DIBuilderRef Builder,
    unsigned Tag,
    LLVMValueRef Scope,
    const char *Name,
    LLVMValueRef File,
    unsigned LineNo,
    LLVMValueRef Ty,
    LLVMValueRef* AddrOps,
    unsigned AddrOpsCount,
    unsigned ArgNo)
{
    llvm::ArrayRef<llvm::Value*> addr_ops((llvm::Value**)AddrOps, AddrOpsCount);

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

extern "C" LLVMValueRef LLVMDIBuilderCreateNameSpace(
    DIBuilderRef Builder,
    LLVMValueRef Scope,
    const char* Name,
    LLVMValueRef File,
    unsigned LineNo)
{
    return wrap(Builder->createNameSpace(
        unwrapDI<DIDescriptor>(Scope),
        Name,
        unwrapDI<DIFile>(File),
        LineNo));
}

extern "C" void LLVMDICompositeTypeSetTypeArray(
    LLVMValueRef CompositeType,
    LLVMValueRef TypeArray)
{
    unwrapDI<DICompositeType>(CompositeType).setTypeArray(unwrapDI<DIArray>(TypeArray));
}

extern "C" char *LLVMTypeToString(LLVMTypeRef Type) {
    std::string s;
    llvm::raw_string_ostream os(s);
    unwrap<llvm::Type>(Type)->print(os);
    return strdup(os.str().data());
}
