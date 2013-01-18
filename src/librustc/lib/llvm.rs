// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::prelude::*;

use core::cast;
use core::cmp;
use core::int;
use core::io;
use core::libc::{c_char, c_int, c_uint, c_longlong, c_ulonglong};
use core::option;
use core::ptr;
use core::str;
use core::uint;
use core::vec;
use std::map::HashMap;

type Opcode = u32;
type Bool = c_uint;

const True: Bool = 1 as Bool;
const False: Bool = 0 as Bool;

// Consts for the LLVM CallConv type, pre-cast to uint.

enum CallConv {
    CCallConv = 0,
    FastCallConv = 8,
    ColdCallConv = 9,
    X86StdcallCallConv = 64,
    X86FastcallCallConv = 65,
}

enum Visibility {
    LLVMDefaultVisibility = 0,
    HiddenVisibility = 1,
    ProtectedVisibility = 2,
}

enum Linkage {
    ExternalLinkage = 0,
    AvailableExternallyLinkage = 1,
    LinkOnceAnyLinkage = 2,
    LinkOnceODRLinkage = 3,
    LinkOnceODRAutoHideLinkage = 4,
    WeakAnyLinkage = 5,
    WeakODRLinkage = 6,
    AppendingLinkage = 7,
    InternalLinkage = 8,
    PrivateLinkage = 9,
    DLLImportLinkage = 10,
    DLLExportLinkage = 11,
    ExternalWeakLinkage = 12,
    GhostLinkage = 13,
    CommonLinkage = 14,
    LinkerPrivateLinkage = 15,
    LinkerPrivateWeakLinkage = 16,
}

enum Attribute {
    ZExtAttribute = 1,
    SExtAttribute = 2,
    NoReturnAttribute = 4,
    InRegAttribute = 8,
    StructRetAttribute = 16,
    NoUnwindAttribute = 32,
    NoAliasAttribute = 64,
    ByValAttribute = 128,
    NestAttribute = 256,
    ReadNoneAttribute = 512,
    ReadOnlyAttribute = 1024,
    NoInlineAttribute = 2048,
    AlwaysInlineAttribute = 4096,
    OptimizeForSizeAttribute = 8192,
    StackProtectAttribute = 16384,
    StackProtectReqAttribute = 32768,
    // 31 << 16
    AlignmentAttribute = 2031616,
    NoCaptureAttribute = 2097152,
    NoRedZoneAttribute = 4194304,
    NoImplicitFloatAttribute = 8388608,
    NakedAttribute = 16777216,
    InlineHintAttribute = 33554432,
    // 7 << 26
    StackAttribute = 469762048,
    ReturnsTwiceAttribute = 536870912,
    // 1 << 30
    UWTableAttribute = 1073741824,
    NonLazyBindAttribute = 2147483648,
}

// enum for the LLVM IntPredicate type
enum IntPredicate {
    IntEQ = 32,
    IntNE = 33,
    IntUGT = 34,
    IntUGE = 35,
    IntULT = 36,
    IntULE = 37,
    IntSGT = 38,
    IntSGE = 39,
    IntSLT = 40,
    IntSLE = 41,
}

// enum for the LLVM RealPredicate type
enum RealPredicate {
    RealPredicateFalse = 0,
    RealOEQ = 1,
    RealOGT = 2,
    RealOGE = 3,
    RealOLT = 4,
    RealOLE = 5,
    RealONE = 6,
    RealORD = 7,
    RealUNO = 8,
    RealUEQ = 9,
    RealUGT = 10,
    RealUGE = 11,
    RealULT = 12,
    RealULE = 13,
    RealUNE = 14,
    RealPredicateTrue = 15,
}

// enum for the LLVM TypeKind type - must stay in sync with the def of
// LLVMTypeKind in llvm/include/llvm-c/Core.h
enum TypeKind {
    Void      = 0,
    Half      = 1,
    Float     = 2,
    Double    = 3,
    X86_FP80  = 4,
    FP128     = 5,
    PPC_FP128 = 6,
    Label     = 7,
    Integer   = 8,
    Function  = 9,
    Struct    = 10,
    Array     = 11,
    Pointer   = 12,
    Vector    = 13,
    Metadata  = 14,
    X86_MMX   = 15
}

impl TypeKind : cmp::Eq {
    pure fn eq(&self, other: &TypeKind) -> bool {
        match ((*self), (*other)) {
            (Void, Void) => true,
            (Half, Half) => true,
            (Float, Float) => true,
            (Double, Double) => true,
            (X86_FP80, X86_FP80) => true,
            (FP128, FP128) => true,
            (PPC_FP128, PPC_FP128) => true,
            (Label, Label) => true,
            (Integer, Integer) => true,
            (Function, Function) => true,
            (Struct, Struct) => true,
            (Array, Array) => true,
            (Pointer, Pointer) => true,
            (Vector, Vector) => true,
            (Metadata, Metadata) => true,
            (X86_MMX, X86_MMX) => true,
            (Void, _) => false,
            (Half, _) => false,
            (Float, _) => false,
            (Double, _) => false,
            (X86_FP80, _) => false,
            (FP128, _) => false,
            (PPC_FP128, _) => false,
            (Label, _) => false,
            (Integer, _) => false,
            (Function, _) => false,
            (Struct, _) => false,
            (Array, _) => false,
            (Pointer, _) => false,
            (Vector, _) => false,
            (Metadata, _) => false,
            (X86_MMX, _) => false,
        }
    }
    pure fn ne(&self, other: &TypeKind) -> bool { !(*self).eq(other) }
}

enum AtomicBinOp {
    Xchg = 0,
    Add  = 1,
    Sub  = 2,
    And  = 3,
    Nand = 4,
    Or   = 5,
    Xor  = 6,
    Max  = 7,
    Min  = 8,
    UMax = 9,
    UMin = 10,
}

enum AtomicOrdering {
    NotAtomic = 0,
    Unordered = 1,
    Monotonic = 2,
    // Consume = 3,  // Not specified yet.
    Acquire = 4,
    Release = 5,
    AcquireRelease = 6,
    SequentiallyConsistent = 7
}

// FIXME: Not used right now, but will be once #2334 is fixed
// Consts for the LLVMCodeGenFileType type (in include/llvm/c/TargetMachine.h)
enum FileType {
    AssemblyFile = 0,
    ObjectFile = 1
}

// Opaque pointer types
enum Module_opaque {}
type ModuleRef = *Module_opaque;
enum Context_opaque {}
type ContextRef = *Context_opaque;
enum Type_opaque {}
type TypeRef = *Type_opaque;
enum Value_opaque {}
type ValueRef = *Value_opaque;
enum BasicBlock_opaque {}
type BasicBlockRef = *BasicBlock_opaque;
enum Builder_opaque {}
type BuilderRef = *Builder_opaque;
enum MemoryBuffer_opaque {}
type MemoryBufferRef = *MemoryBuffer_opaque;
enum PassManager_opaque {}
type PassManagerRef = *PassManager_opaque;
enum PassManagerBuilder_opaque {}
type PassManagerBuilderRef = *PassManagerBuilder_opaque;
enum Use_opaque {}
type UseRef = *Use_opaque;
enum TargetData_opaque {}
type TargetDataRef = *TargetData_opaque;
enum ObjectFile_opaque {}
type ObjectFileRef = *ObjectFile_opaque;
enum SectionIterator_opaque {}
type SectionIteratorRef = *SectionIterator_opaque;

#[link_args = "-Lrustllvm"]
#[link_name = "rustllvm"]
#[abi = "cdecl"]
extern mod llvm {
    #[legacy_exports];
    /* Create and destroy contexts. */
    unsafe fn LLVMContextCreate() -> ContextRef;
    unsafe fn LLVMGetGlobalContext() -> ContextRef;
    unsafe fn LLVMContextDispose(C: ContextRef);
    unsafe fn LLVMGetMDKindIDInContext(C: ContextRef,
                                       Name: *c_char,
                                       SLen: c_uint)
                                    -> c_uint;
    unsafe fn LLVMGetMDKindID(Name: *c_char, SLen: c_uint) -> c_uint;

    /* Create and destroy modules. */
    unsafe fn LLVMModuleCreateWithNameInContext(ModuleID: *c_char,
                                                C: ContextRef)
                                             -> ModuleRef;
    unsafe fn LLVMDisposeModule(M: ModuleRef);

    /** Data layout. See Module::getDataLayout. */
    unsafe fn LLVMGetDataLayout(M: ModuleRef) -> *c_char;
    unsafe fn LLVMSetDataLayout(M: ModuleRef, Triple: *c_char);

    /** Target triple. See Module::getTargetTriple. */
    unsafe fn LLVMGetTarget(M: ModuleRef) -> *c_char;
    unsafe fn LLVMSetTarget(M: ModuleRef, Triple: *c_char);

    /** See Module::dump. */
    unsafe fn LLVMDumpModule(M: ModuleRef);

    /** See Module::setModuleInlineAsm. */
    unsafe fn LLVMSetModuleInlineAsm(M: ModuleRef, Asm: *c_char);

    /** See llvm::LLVMTypeKind::getTypeID. */
    unsafe fn LLVMGetTypeKind(Ty: TypeRef) -> TypeKind;

    /** See llvm::LLVMType::getContext. */
    unsafe fn LLVMGetTypeContext(Ty: TypeRef) -> ContextRef;

    /* Operations on integer types */
    unsafe fn LLVMInt1TypeInContext(C: ContextRef) -> TypeRef;
    unsafe fn LLVMInt8TypeInContext(C: ContextRef) -> TypeRef;
    unsafe fn LLVMInt16TypeInContext(C: ContextRef) -> TypeRef;
    unsafe fn LLVMInt32TypeInContext(C: ContextRef) -> TypeRef;
    unsafe fn LLVMInt64TypeInContext(C: ContextRef) -> TypeRef;
    unsafe fn LLVMIntTypeInContext(C: ContextRef, NumBits: c_uint) -> TypeRef;

    unsafe fn LLVMInt1Type() -> TypeRef;
    unsafe fn LLVMInt8Type() -> TypeRef;
    unsafe fn LLVMInt16Type() -> TypeRef;
    unsafe fn LLVMInt32Type() -> TypeRef;
    unsafe fn LLVMInt64Type() -> TypeRef;
    unsafe fn LLVMIntType(NumBits: c_uint) -> TypeRef;
    unsafe fn LLVMGetIntTypeWidth(IntegerTy: TypeRef) -> c_uint;

    /* Operations on real types */
    unsafe fn LLVMFloatTypeInContext(C: ContextRef) -> TypeRef;
    unsafe fn LLVMDoubleTypeInContext(C: ContextRef) -> TypeRef;
    unsafe fn LLVMX86FP80TypeInContext(C: ContextRef) -> TypeRef;
    unsafe fn LLVMFP128TypeInContext(C: ContextRef) -> TypeRef;
    unsafe fn LLVMPPCFP128TypeInContext(C: ContextRef) -> TypeRef;

    unsafe fn LLVMFloatType() -> TypeRef;
    unsafe fn LLVMDoubleType() -> TypeRef;
    unsafe fn LLVMX86FP80Type() -> TypeRef;
    unsafe fn LLVMFP128Type() -> TypeRef;
    unsafe fn LLVMPPCFP128Type() -> TypeRef;

    /* Operations on function types */
    unsafe fn LLVMFunctionType(ReturnType: TypeRef, ParamTypes: *TypeRef,
                        ParamCount: c_uint, IsVarArg: Bool) -> TypeRef;
    unsafe fn LLVMIsFunctionVarArg(FunctionTy: TypeRef) -> Bool;
    unsafe fn LLVMGetReturnType(FunctionTy: TypeRef) -> TypeRef;
    unsafe fn LLVMCountParamTypes(FunctionTy: TypeRef) -> c_uint;
    unsafe fn LLVMGetParamTypes(FunctionTy: TypeRef, Dest: *TypeRef);

    /* Operations on struct types */
    unsafe fn LLVMStructTypeInContext(C: ContextRef, ElementTypes: *TypeRef,
                               ElementCount: c_uint,
                               Packed: Bool) -> TypeRef;
    unsafe fn LLVMStructType(ElementTypes: *TypeRef, ElementCount: c_uint,
                      Packed: Bool) -> TypeRef;
    unsafe fn LLVMCountStructElementTypes(StructTy: TypeRef) -> c_uint;
    unsafe fn LLVMGetStructElementTypes(StructTy: TypeRef,
                                        Dest: *mut TypeRef);
    unsafe fn LLVMIsPackedStruct(StructTy: TypeRef) -> Bool;

    /* Operations on array, pointer, and vector types (sequence types) */
    unsafe fn LLVMArrayType(ElementType: TypeRef,
                     ElementCount: c_uint) -> TypeRef;
    unsafe fn LLVMPointerType(ElementType: TypeRef,
                       AddressSpace: c_uint) -> TypeRef;
    unsafe fn LLVMVectorType(ElementType: TypeRef,
                      ElementCount: c_uint) -> TypeRef;

    unsafe fn LLVMGetElementType(Ty: TypeRef) -> TypeRef;
    unsafe fn LLVMGetArrayLength(ArrayTy: TypeRef) -> c_uint;
    unsafe fn LLVMGetPointerAddressSpace(PointerTy: TypeRef) -> c_uint;
    unsafe fn LLVMGetVectorSize(VectorTy: TypeRef) -> c_uint;

    /* Operations on other types */
    unsafe fn LLVMVoidTypeInContext(C: ContextRef) -> TypeRef;
    unsafe fn LLVMLabelTypeInContext(C: ContextRef) -> TypeRef;
    unsafe fn LLVMMetadataTypeInContext(C: ContextRef) -> TypeRef;

    unsafe fn LLVMVoidType() -> TypeRef;
    unsafe fn LLVMLabelType() -> TypeRef;
    unsafe fn LLVMMetadataType() -> TypeRef;

    /* Operations on all values */
    unsafe fn LLVMTypeOf(Val: ValueRef) -> TypeRef;
    unsafe fn LLVMGetValueName(Val: ValueRef) -> *c_char;
    unsafe fn LLVMSetValueName(Val: ValueRef, Name: *c_char);
    unsafe fn LLVMDumpValue(Val: ValueRef);
    unsafe fn LLVMReplaceAllUsesWith(OldVal: ValueRef, NewVal: ValueRef);
    unsafe fn LLVMHasMetadata(Val: ValueRef) -> c_int;
    unsafe fn LLVMGetMetadata(Val: ValueRef, KindID: c_uint) -> ValueRef;
    unsafe fn LLVMSetMetadata(Val: ValueRef, KindID: c_uint, Node: ValueRef);

    /* Operations on Uses */
    unsafe fn LLVMGetFirstUse(Val: ValueRef) -> UseRef;
    unsafe fn LLVMGetNextUse(U: UseRef) -> UseRef;
    unsafe fn LLVMGetUser(U: UseRef) -> ValueRef;
    unsafe fn LLVMGetUsedValue(U: UseRef) -> ValueRef;

    /* Operations on Users */
    unsafe fn LLVMGetOperand(Val: ValueRef, Index: c_uint) -> ValueRef;
    unsafe fn LLVMSetOperand(Val: ValueRef, Index: c_uint, Op: ValueRef);

    /* Operations on constants of any type */
    unsafe fn LLVMConstNull(Ty: TypeRef) -> ValueRef;
    /* all zeroes */
    unsafe fn LLVMConstAllOnes(Ty: TypeRef) -> ValueRef;
    /* only for int/vector */
    unsafe fn LLVMGetUndef(Ty: TypeRef) -> ValueRef;
    unsafe fn LLVMIsConstant(Val: ValueRef) -> Bool;
    unsafe fn LLVMIsNull(Val: ValueRef) -> Bool;
    unsafe fn LLVMIsUndef(Val: ValueRef) -> Bool;
    unsafe fn LLVMConstPointerNull(Ty: TypeRef) -> ValueRef;

    /* Operations on metadata */
    unsafe fn LLVMMDStringInContext(C: ContextRef,
                                    Str: *c_char,
                                    SLen: c_uint)
                                 -> ValueRef;
    unsafe fn LLVMMDString(Str: *c_char, SLen: c_uint) -> ValueRef;
    unsafe fn LLVMMDNodeInContext(C: ContextRef,
                                  Vals: *ValueRef,
                                  Count: c_uint)
                               -> ValueRef;
    unsafe fn LLVMMDNode(Vals: *ValueRef, Count: c_uint) -> ValueRef;
    unsafe fn LLVMAddNamedMetadataOperand(M: ModuleRef, Str: *c_char,
                                   Val: ValueRef);

    /* Operations on scalar constants */
    unsafe fn LLVMConstInt(IntTy: TypeRef,
                           N: c_ulonglong,
                           SignExtend: Bool)
                        -> ValueRef;
    unsafe fn LLVMConstIntOfString(IntTy: TypeRef,
                                   Text: *c_char,
                                   Radix: u8)
                                -> ValueRef;
    unsafe fn LLVMConstIntOfStringAndSize(IntTy: TypeRef, Text: *c_char,
                                   SLen: c_uint,
                                   Radix: u8) -> ValueRef;
    unsafe fn LLVMConstReal(RealTy: TypeRef, N: f64) -> ValueRef;
    unsafe fn LLVMConstRealOfString(RealTy: TypeRef,
                                    Text: *c_char)
                                 -> ValueRef;
    unsafe fn LLVMConstRealOfStringAndSize(RealTy: TypeRef, Text: *c_char,
                                    SLen: c_uint) -> ValueRef;
    unsafe fn LLVMConstIntGetZExtValue(ConstantVal: ValueRef) -> c_ulonglong;
    unsafe fn LLVMConstIntGetSExtValue(ConstantVal: ValueRef) -> c_longlong;


    /* Operations on composite constants */
    unsafe fn LLVMConstStringInContext(C: ContextRef,
                                       Str: *c_char,
                                       Length: c_uint,
                                       DontNullTerminate: Bool)
                                    -> ValueRef;
    unsafe fn LLVMConstStructInContext(C: ContextRef, ConstantVals: *ValueRef,
                                Count: c_uint, Packed: Bool) -> ValueRef;

    unsafe fn LLVMConstString(Str: *c_char, Length: c_uint,
                       DontNullTerminate: Bool) -> ValueRef;
    unsafe fn LLVMConstArray(ElementTy: TypeRef, ConstantVals: *ValueRef,
                      Length: c_uint) -> ValueRef;
    unsafe fn LLVMConstStruct(ConstantVals: *ValueRef,
                       Count: c_uint, Packed: Bool) -> ValueRef;
    unsafe fn LLVMConstVector(ScalarConstantVals: *ValueRef,
                       Size: c_uint) -> ValueRef;

    /* Constant expressions */
    unsafe fn LLVMAlignOf(Ty: TypeRef) -> ValueRef;
    unsafe fn LLVMSizeOf(Ty: TypeRef) -> ValueRef;
    unsafe fn LLVMConstNeg(ConstantVal: ValueRef) -> ValueRef;
    unsafe fn LLVMConstNSWNeg(ConstantVal: ValueRef) -> ValueRef;
    unsafe fn LLVMConstNUWNeg(ConstantVal: ValueRef) -> ValueRef;
    unsafe fn LLVMConstFNeg(ConstantVal: ValueRef) -> ValueRef;
    unsafe fn LLVMConstNot(ConstantVal: ValueRef) -> ValueRef;
    unsafe fn LLVMConstAdd(LHSConstant: ValueRef,
                           RHSConstant: ValueRef)
                        -> ValueRef;
    unsafe fn LLVMConstNSWAdd(LHSConstant: ValueRef,
                              RHSConstant: ValueRef)
                           -> ValueRef;
    unsafe fn LLVMConstNUWAdd(LHSConstant: ValueRef,
                              RHSConstant: ValueRef)
                           -> ValueRef;
    unsafe fn LLVMConstFAdd(LHSConstant: ValueRef,
                            RHSConstant: ValueRef)
                         -> ValueRef;
    unsafe fn LLVMConstSub(LHSConstant: ValueRef,
                           RHSConstant: ValueRef)
                        -> ValueRef;
    unsafe fn LLVMConstNSWSub(LHSConstant: ValueRef,
                              RHSConstant: ValueRef)
                           -> ValueRef;
    unsafe fn LLVMConstNUWSub(LHSConstant: ValueRef,
                              RHSConstant: ValueRef)
                           -> ValueRef;
    unsafe fn LLVMConstFSub(LHSConstant: ValueRef, RHSConstant: ValueRef) ->
       ValueRef;
    unsafe fn LLVMConstMul(LHSConstant: ValueRef,
                           RHSConstant: ValueRef)
                        -> ValueRef;
    unsafe fn LLVMConstNSWMul(LHSConstant: ValueRef,
                              RHSConstant: ValueRef)
                           -> ValueRef;
    unsafe fn LLVMConstNUWMul(LHSConstant: ValueRef,
                              RHSConstant: ValueRef)
                           -> ValueRef;
    unsafe fn LLVMConstFMul(LHSConstant: ValueRef,
                            RHSConstant: ValueRef)
                         -> ValueRef;
    unsafe fn LLVMConstUDiv(LHSConstant: ValueRef,
                            RHSConstant: ValueRef)
                         -> ValueRef;
    unsafe fn LLVMConstSDiv(LHSConstant: ValueRef,
                            RHSConstant: ValueRef)
                         -> ValueRef;
    unsafe fn LLVMConstExactSDiv(LHSConstant: ValueRef,
                                 RHSConstant: ValueRef)
                              -> ValueRef;
    unsafe fn LLVMConstFDiv(LHSConstant: ValueRef,
                            RHSConstant: ValueRef)
                         -> ValueRef;
    unsafe fn LLVMConstURem(LHSConstant: ValueRef,
                            RHSConstant: ValueRef)
                         -> ValueRef;
    unsafe fn LLVMConstSRem(LHSConstant: ValueRef,
                            RHSConstant: ValueRef)
                         -> ValueRef;
    unsafe fn LLVMConstFRem(LHSConstant: ValueRef,
                            RHSConstant: ValueRef)
                         -> ValueRef;
    unsafe fn LLVMConstAnd(LHSConstant: ValueRef,
                           RHSConstant: ValueRef)
                        -> ValueRef;
    unsafe fn LLVMConstOr(LHSConstant: ValueRef,
                          RHSConstant: ValueRef)
                       -> ValueRef;
    unsafe fn LLVMConstXor(LHSConstant: ValueRef,
                           RHSConstant: ValueRef)
                        -> ValueRef;
    unsafe fn LLVMConstShl(LHSConstant: ValueRef,
                           RHSConstant: ValueRef)
                        -> ValueRef;
    unsafe fn LLVMConstLShr(LHSConstant: ValueRef, RHSConstant: ValueRef) ->
       ValueRef;
    unsafe fn LLVMConstAShr(LHSConstant: ValueRef, RHSConstant: ValueRef) ->
       ValueRef;
    unsafe fn LLVMConstGEP(ConstantVal: ValueRef,
                    ConstantIndices: *ValueRef,
                    NumIndices: c_uint) -> ValueRef;
    unsafe fn LLVMConstInBoundsGEP(ConstantVal: ValueRef,
                                   ConstantIndices: *ValueRef,
                                   NumIndices: c_uint)
                                -> ValueRef;
    unsafe fn LLVMConstTrunc(ConstantVal: ValueRef,
                             ToType: TypeRef)
                          -> ValueRef;
    unsafe fn LLVMConstSExt(ConstantVal: ValueRef,
                            ToType: TypeRef)
                         -> ValueRef;
    unsafe fn LLVMConstZExt(ConstantVal: ValueRef,
                            ToType: TypeRef)
                         -> ValueRef;
    unsafe fn LLVMConstFPTrunc(ConstantVal: ValueRef,
                               ToType: TypeRef)
                            -> ValueRef;
    unsafe fn LLVMConstFPExt(ConstantVal: ValueRef,
                             ToType: TypeRef)
                          -> ValueRef;
    unsafe fn LLVMConstUIToFP(ConstantVal: ValueRef,
                              ToType: TypeRef)
                           -> ValueRef;
    unsafe fn LLVMConstSIToFP(ConstantVal: ValueRef,
                              ToType: TypeRef)
                           -> ValueRef;
    unsafe fn LLVMConstFPToUI(ConstantVal: ValueRef,
                              ToType: TypeRef)
                           -> ValueRef;
    unsafe fn LLVMConstFPToSI(ConstantVal: ValueRef,
                              ToType: TypeRef)
                           -> ValueRef;
    unsafe fn LLVMConstPtrToInt(ConstantVal: ValueRef,
                                ToType: TypeRef)
                             -> ValueRef;
    unsafe fn LLVMConstIntToPtr(ConstantVal: ValueRef,
                                ToType: TypeRef)
                             -> ValueRef;
    unsafe fn LLVMConstBitCast(ConstantVal: ValueRef,
                               ToType: TypeRef)
                            -> ValueRef;
    unsafe fn LLVMConstZExtOrBitCast(ConstantVal: ValueRef,
                                     ToType: TypeRef)
                                  -> ValueRef;
    unsafe fn LLVMConstSExtOrBitCast(ConstantVal: ValueRef,
                                     ToType: TypeRef)
                                  -> ValueRef;
    unsafe fn LLVMConstTruncOrBitCast(ConstantVal: ValueRef,
                                      ToType: TypeRef)
                                   -> ValueRef;
    unsafe fn LLVMConstPointerCast(ConstantVal: ValueRef,
                                   ToType: TypeRef)
                                -> ValueRef;
    unsafe fn LLVMConstIntCast(ConstantVal: ValueRef, ToType: TypeRef,
                        isSigned: Bool) -> ValueRef;
    unsafe fn LLVMConstFPCast(ConstantVal: ValueRef,
                              ToType: TypeRef)
                           -> ValueRef;
    unsafe fn LLVMConstSelect(ConstantCondition: ValueRef,
                              ConstantIfTrue: ValueRef,
                              ConstantIfFalse: ValueRef)
                           -> ValueRef;
    unsafe fn LLVMConstExtractElement(VectorConstant: ValueRef,
                               IndexConstant: ValueRef) -> ValueRef;
    unsafe fn LLVMConstInsertElement(VectorConstant: ValueRef,
                              ElementValueConstant: ValueRef,
                              IndexConstant: ValueRef) -> ValueRef;
    unsafe fn LLVMConstShuffleVector(VectorAConstant: ValueRef,
                              VectorBConstant: ValueRef,
                              MaskConstant: ValueRef) -> ValueRef;
    unsafe fn LLVMConstExtractValue(AggConstant: ValueRef, IdxList: *c_uint,
                             NumIdx: c_uint) -> ValueRef;
    unsafe fn LLVMConstInsertValue(AggConstant: ValueRef,
                            ElementValueConstant: ValueRef, IdxList: *c_uint,
                            NumIdx: c_uint) -> ValueRef;
    unsafe fn LLVMConstInlineAsm(Ty: TypeRef, AsmString: *c_char,
                          Constraints: *c_char, HasSideEffects: Bool,
                          IsAlignStack: Bool) -> ValueRef;
    unsafe fn LLVMBlockAddress(F: ValueRef, BB: BasicBlockRef) -> ValueRef;



    /* Operations on global variables, functions, and aliases (globals) */
    unsafe fn LLVMGetGlobalParent(Global: ValueRef) -> ModuleRef;
    unsafe fn LLVMIsDeclaration(Global: ValueRef) -> Bool;
    unsafe fn LLVMGetLinkage(Global: ValueRef) -> c_uint;
    unsafe fn LLVMSetLinkage(Global: ValueRef, Link: c_uint);
    unsafe fn LLVMGetSection(Global: ValueRef) -> *c_char;
    unsafe fn LLVMSetSection(Global: ValueRef, Section: *c_char);
    unsafe fn LLVMGetVisibility(Global: ValueRef) -> c_uint;
    unsafe fn LLVMSetVisibility(Global: ValueRef, Viz: c_uint);
    unsafe fn LLVMGetAlignment(Global: ValueRef) -> c_uint;
    unsafe fn LLVMSetAlignment(Global: ValueRef, Bytes: c_uint);


    /* Operations on global variables */
    unsafe fn LLVMAddGlobal(M: ModuleRef,
                            Ty: TypeRef,
                            Name: *c_char)
                         -> ValueRef;
    unsafe fn LLVMAddGlobalInAddressSpace(M: ModuleRef,
                                          Ty: TypeRef,
                                          Name: *c_char,
                                          AddressSpace: c_uint)
                                       -> ValueRef;
    unsafe fn LLVMGetNamedGlobal(M: ModuleRef, Name: *c_char) -> ValueRef;
    unsafe fn LLVMGetFirstGlobal(M: ModuleRef) -> ValueRef;
    unsafe fn LLVMGetLastGlobal(M: ModuleRef) -> ValueRef;
    unsafe fn LLVMGetNextGlobal(GlobalVar: ValueRef) -> ValueRef;
    unsafe fn LLVMGetPreviousGlobal(GlobalVar: ValueRef) -> ValueRef;
    unsafe fn LLVMDeleteGlobal(GlobalVar: ValueRef);
    unsafe fn LLVMGetInitializer(GlobalVar: ValueRef) -> ValueRef;
    unsafe fn LLVMSetInitializer(GlobalVar: ValueRef, ConstantVal: ValueRef);
    unsafe fn LLVMIsThreadLocal(GlobalVar: ValueRef) -> Bool;
    unsafe fn LLVMSetThreadLocal(GlobalVar: ValueRef, IsThreadLocal: Bool);
    unsafe fn LLVMIsGlobalConstant(GlobalVar: ValueRef) -> Bool;
    unsafe fn LLVMSetGlobalConstant(GlobalVar: ValueRef, IsConstant: Bool);

    /* Operations on aliases */
    unsafe fn LLVMAddAlias(M: ModuleRef, Ty: TypeRef, Aliasee: ValueRef,
                    Name: *c_char) -> ValueRef;

    /* Operations on functions */
    unsafe fn LLVMAddFunction(M: ModuleRef,
                              Name: *c_char,
                              FunctionTy: TypeRef)
                           -> ValueRef;
    unsafe fn LLVMGetNamedFunction(M: ModuleRef, Name: *c_char) -> ValueRef;
    unsafe fn LLVMGetFirstFunction(M: ModuleRef) -> ValueRef;
    unsafe fn LLVMGetLastFunction(M: ModuleRef) -> ValueRef;
    unsafe fn LLVMGetNextFunction(Fn: ValueRef) -> ValueRef;
    unsafe fn LLVMGetPreviousFunction(Fn: ValueRef) -> ValueRef;
    unsafe fn LLVMDeleteFunction(Fn: ValueRef);
    unsafe fn LLVMGetOrInsertFunction(M: ModuleRef, Name: *c_char,
                               FunctionTy: TypeRef) -> ValueRef;
    unsafe fn LLVMGetIntrinsicID(Fn: ValueRef) -> c_uint;
    unsafe fn LLVMGetFunctionCallConv(Fn: ValueRef) -> c_uint;
    unsafe fn LLVMSetFunctionCallConv(Fn: ValueRef, CC: c_uint);
    unsafe fn LLVMGetGC(Fn: ValueRef) -> *c_char;
    unsafe fn LLVMSetGC(Fn: ValueRef, Name: *c_char);
    unsafe fn LLVMAddFunctionAttr(Fn: ValueRef, PA: c_ulonglong, HighPA:
                           c_ulonglong);
    unsafe fn LLVMGetFunctionAttr(Fn: ValueRef) -> c_ulonglong;
    unsafe fn LLVMRemoveFunctionAttr(Fn: ValueRef, PA: c_ulonglong, HighPA:
                              c_ulonglong);

    /* Operations on parameters */
    unsafe fn LLVMCountParams(Fn: ValueRef) -> c_uint;
    unsafe fn LLVMGetParams(Fn: ValueRef, Params: *ValueRef);
    unsafe fn LLVMGetParam(Fn: ValueRef, Index: c_uint) -> ValueRef;
    unsafe fn LLVMGetParamParent(Inst: ValueRef) -> ValueRef;
    unsafe fn LLVMGetFirstParam(Fn: ValueRef) -> ValueRef;
    unsafe fn LLVMGetLastParam(Fn: ValueRef) -> ValueRef;
    unsafe fn LLVMGetNextParam(Arg: ValueRef) -> ValueRef;
    unsafe fn LLVMGetPreviousParam(Arg: ValueRef) -> ValueRef;
    unsafe fn LLVMAddAttribute(Arg: ValueRef, PA: c_uint);
    unsafe fn LLVMRemoveAttribute(Arg: ValueRef, PA: c_uint);
    unsafe fn LLVMGetAttribute(Arg: ValueRef) -> c_uint;
    unsafe fn LLVMSetParamAlignment(Arg: ValueRef, align: c_uint);

    /* Operations on basic blocks */
    unsafe fn LLVMBasicBlockAsValue(BB: BasicBlockRef) -> ValueRef;
    unsafe fn LLVMValueIsBasicBlock(Val: ValueRef) -> Bool;
    unsafe fn LLVMValueAsBasicBlock(Val: ValueRef) -> BasicBlockRef;
    unsafe fn LLVMGetBasicBlockParent(BB: BasicBlockRef) -> ValueRef;
    unsafe fn LLVMCountBasicBlocks(Fn: ValueRef) -> c_uint;
    unsafe fn LLVMGetBasicBlocks(Fn: ValueRef, BasicBlocks: *ValueRef);
    unsafe fn LLVMGetFirstBasicBlock(Fn: ValueRef) -> BasicBlockRef;
    unsafe fn LLVMGetLastBasicBlock(Fn: ValueRef) -> BasicBlockRef;
    unsafe fn LLVMGetNextBasicBlock(BB: BasicBlockRef) -> BasicBlockRef;
    unsafe fn LLVMGetPreviousBasicBlock(BB: BasicBlockRef) -> BasicBlockRef;
    unsafe fn LLVMGetEntryBasicBlock(Fn: ValueRef) -> BasicBlockRef;

    unsafe fn LLVMAppendBasicBlockInContext(C: ContextRef, Fn: ValueRef,
                                     Name: *c_char) -> BasicBlockRef;
    unsafe fn LLVMInsertBasicBlockInContext(C: ContextRef, BB: BasicBlockRef,
                                     Name: *c_char) -> BasicBlockRef;

    unsafe fn LLVMAppendBasicBlock(Fn: ValueRef,
                                   Name: *c_char)
                                -> BasicBlockRef;
    unsafe fn LLVMInsertBasicBlock(InsertBeforeBB: BasicBlockRef,
                                   Name: *c_char)
                                -> BasicBlockRef;
    unsafe fn LLVMDeleteBasicBlock(BB: BasicBlockRef);

    /* Operations on instructions */
    unsafe fn LLVMGetInstructionParent(Inst: ValueRef) -> BasicBlockRef;
    unsafe fn LLVMGetFirstInstruction(BB: BasicBlockRef) -> ValueRef;
    unsafe fn LLVMGetLastInstruction(BB: BasicBlockRef) -> ValueRef;
    unsafe fn LLVMGetNextInstruction(Inst: ValueRef) -> ValueRef;
    unsafe fn LLVMGetPreviousInstruction(Inst: ValueRef) -> ValueRef;

    /* Operations on call sites */
    unsafe fn LLVMSetInstructionCallConv(Instr: ValueRef, CC: c_uint);
    unsafe fn LLVMGetInstructionCallConv(Instr: ValueRef) -> c_uint;
    unsafe fn LLVMAddInstrAttribute(Instr: ValueRef,
                                    index: c_uint,
                                    IA: c_uint);
    unsafe fn LLVMRemoveInstrAttribute(Instr: ValueRef, index: c_uint,
                                IA: c_uint);
    unsafe fn LLVMSetInstrParamAlignment(Instr: ValueRef, index: c_uint,
                                  align: c_uint);

    /* Operations on call instructions (only) */
    unsafe fn LLVMIsTailCall(CallInst: ValueRef) -> Bool;
    unsafe fn LLVMSetTailCall(CallInst: ValueRef, IsTailCall: Bool);

    /* Operations on phi nodes */
    unsafe fn LLVMAddIncoming(PhiNode: ValueRef, IncomingValues: *ValueRef,
                       IncomingBlocks: *BasicBlockRef, Count: c_uint);
    unsafe fn LLVMCountIncoming(PhiNode: ValueRef) -> c_uint;
    unsafe fn LLVMGetIncomingValue(PhiNode: ValueRef,
                                   Index: c_uint)
                                -> ValueRef;
    unsafe fn LLVMGetIncomingBlock(PhiNode: ValueRef,
                            Index: c_uint) -> BasicBlockRef;

    /* Instruction builders */
    unsafe fn LLVMCreateBuilderInContext(C: ContextRef) -> BuilderRef;
    unsafe fn LLVMCreateBuilder() -> BuilderRef;
    unsafe fn LLVMPositionBuilder(Builder: BuilderRef, Block: BasicBlockRef,
                           Instr: ValueRef);
    unsafe fn LLVMPositionBuilderBefore(Builder: BuilderRef, Instr: ValueRef);
    unsafe fn LLVMPositionBuilderAtEnd(Builder: BuilderRef,
                                       Block: BasicBlockRef);
    unsafe fn LLVMGetInsertBlock(Builder: BuilderRef) -> BasicBlockRef;
    unsafe fn LLVMClearInsertionPosition(Builder: BuilderRef);
    unsafe fn LLVMInsertIntoBuilder(Builder: BuilderRef, Instr: ValueRef);
    unsafe fn LLVMInsertIntoBuilderWithName(Builder: BuilderRef,
                                            Instr: ValueRef,
                                            Name: *c_char);
    unsafe fn LLVMDisposeBuilder(Builder: BuilderRef);

    /* Metadata */
    unsafe fn LLVMSetCurrentDebugLocation(Builder: BuilderRef, L: ValueRef);
    unsafe fn LLVMGetCurrentDebugLocation(Builder: BuilderRef) -> ValueRef;
    unsafe fn LLVMSetInstDebugLocation(Builder: BuilderRef, Inst: ValueRef);

    /* Terminators */
    unsafe fn LLVMBuildRetVoid(B: BuilderRef) -> ValueRef;
    unsafe fn LLVMBuildRet(B: BuilderRef, V: ValueRef) -> ValueRef;
    unsafe fn LLVMBuildAggregateRet(B: BuilderRef, RetVals: *ValueRef,
                             N: c_uint) -> ValueRef;
    unsafe fn LLVMBuildBr(B: BuilderRef, Dest: BasicBlockRef) -> ValueRef;
    unsafe fn LLVMBuildCondBr(B: BuilderRef,
                              If: ValueRef,
                              Then: BasicBlockRef,
                              Else: BasicBlockRef)
                           -> ValueRef;
    unsafe fn LLVMBuildSwitch(B: BuilderRef, V: ValueRef, Else: BasicBlockRef,
                       NumCases: c_uint) -> ValueRef;
    unsafe fn LLVMBuildIndirectBr(B: BuilderRef, Addr: ValueRef,
                           NumDests: c_uint) -> ValueRef;
    unsafe fn LLVMBuildInvoke(B: BuilderRef, Fn: ValueRef, Args: *ValueRef,
                       NumArgs: c_uint, Then: BasicBlockRef,
                       Catch: BasicBlockRef, Name: *c_char) -> ValueRef;
    unsafe fn LLVMBuildLandingPad(B: BuilderRef,
                                  Ty: TypeRef,
                                  PersFn: ValueRef,
                                  NumClauses: c_uint,
                                  Name: *c_char)
                               -> ValueRef;
    unsafe fn LLVMBuildResume(B: BuilderRef, Exn: ValueRef) -> ValueRef;
    unsafe fn LLVMBuildUnreachable(B: BuilderRef) -> ValueRef;

    /* Add a case to the switch instruction */
    unsafe fn LLVMAddCase(Switch: ValueRef,
                          OnVal: ValueRef,
                          Dest: BasicBlockRef);

    /* Add a destination to the indirectbr instruction */
    unsafe fn LLVMAddDestination(IndirectBr: ValueRef, Dest: BasicBlockRef);

    /* Add a clause to the landing pad instruction */
    unsafe fn LLVMAddClause(LandingPad: ValueRef, ClauseVal: ValueRef);

    /* Set the cleanup on a landing pad instruction */
    unsafe fn LLVMSetCleanup(LandingPad: ValueRef, Val: Bool);

    /* Arithmetic */
    unsafe fn LLVMBuildAdd(B: BuilderRef, LHS: ValueRef, RHS: ValueRef,
                    Name: *c_char) -> ValueRef;
    unsafe fn LLVMBuildNSWAdd(B: BuilderRef, LHS: ValueRef, RHS: ValueRef,
                       Name: *c_char) -> ValueRef;
    unsafe fn LLVMBuildNUWAdd(B: BuilderRef, LHS: ValueRef, RHS: ValueRef,
                       Name: *c_char) -> ValueRef;
    unsafe fn LLVMBuildFAdd(B: BuilderRef, LHS: ValueRef, RHS: ValueRef,
                     Name: *c_char) -> ValueRef;
    unsafe fn LLVMBuildSub(B: BuilderRef, LHS: ValueRef, RHS: ValueRef,
                    Name: *c_char) -> ValueRef;
    unsafe fn LLVMBuildNSWSub(B: BuilderRef, LHS: ValueRef, RHS: ValueRef,
                       Name: *c_char) -> ValueRef;
    unsafe fn LLVMBuildNUWSub(B: BuilderRef, LHS: ValueRef, RHS: ValueRef,
                       Name: *c_char) -> ValueRef;
    unsafe fn LLVMBuildFSub(B: BuilderRef, LHS: ValueRef, RHS: ValueRef,
                     Name: *c_char) -> ValueRef;
    unsafe fn LLVMBuildMul(B: BuilderRef, LHS: ValueRef, RHS: ValueRef,
                    Name: *c_char) -> ValueRef;
    unsafe fn LLVMBuildNSWMul(B: BuilderRef, LHS: ValueRef, RHS: ValueRef,
                       Name: *c_char) -> ValueRef;
    unsafe fn LLVMBuildNUWMul(B: BuilderRef, LHS: ValueRef, RHS: ValueRef,
                       Name: *c_char) -> ValueRef;
    unsafe fn LLVMBuildFMul(B: BuilderRef, LHS: ValueRef, RHS: ValueRef,
                     Name: *c_char) -> ValueRef;
    unsafe fn LLVMBuildUDiv(B: BuilderRef, LHS: ValueRef, RHS: ValueRef,
                     Name: *c_char) -> ValueRef;
    unsafe fn LLVMBuildSDiv(B: BuilderRef, LHS: ValueRef, RHS: ValueRef,
                     Name: *c_char) -> ValueRef;
    unsafe fn LLVMBuildExactSDiv(B: BuilderRef, LHS: ValueRef, RHS: ValueRef,
                          Name: *c_char) -> ValueRef;
    unsafe fn LLVMBuildFDiv(B: BuilderRef, LHS: ValueRef, RHS: ValueRef,
                     Name: *c_char) -> ValueRef;
    unsafe fn LLVMBuildURem(B: BuilderRef, LHS: ValueRef, RHS: ValueRef,
                     Name: *c_char) -> ValueRef;
    unsafe fn LLVMBuildSRem(B: BuilderRef, LHS: ValueRef, RHS: ValueRef,
                     Name: *c_char) -> ValueRef;
    unsafe fn LLVMBuildFRem(B: BuilderRef, LHS: ValueRef, RHS: ValueRef,
                     Name: *c_char) -> ValueRef;
    unsafe fn LLVMBuildShl(B: BuilderRef, LHS: ValueRef, RHS: ValueRef,
                    Name: *c_char) -> ValueRef;
    unsafe fn LLVMBuildLShr(B: BuilderRef, LHS: ValueRef, RHS: ValueRef,
                     Name: *c_char) -> ValueRef;
    unsafe fn LLVMBuildAShr(B: BuilderRef, LHS: ValueRef, RHS: ValueRef,
                     Name: *c_char) -> ValueRef;
    unsafe fn LLVMBuildAnd(B: BuilderRef, LHS: ValueRef, RHS: ValueRef,
                    Name: *c_char) -> ValueRef;
    unsafe fn LLVMBuildOr(B: BuilderRef, LHS: ValueRef, RHS: ValueRef,
                   Name: *c_char) -> ValueRef;
    unsafe fn LLVMBuildXor(B: BuilderRef, LHS: ValueRef, RHS: ValueRef,
                    Name: *c_char) -> ValueRef;
    unsafe fn LLVMBuildBinOp(B: BuilderRef,
                             Op: Opcode,
                             LHS: ValueRef,
                             RHS: ValueRef,
                             Name: *c_char)
                          -> ValueRef;
    unsafe fn LLVMBuildNeg(B: BuilderRef,
                           V: ValueRef,
                           Name: *c_char)
                        -> ValueRef;
    unsafe fn LLVMBuildNSWNeg(B: BuilderRef,
                              V: ValueRef,
                              Name: *c_char)
                           -> ValueRef;
    unsafe fn LLVMBuildNUWNeg(B: BuilderRef,
                              V: ValueRef,
                              Name: *c_char)
                           -> ValueRef;
    unsafe fn LLVMBuildFNeg(B: BuilderRef,
                            V: ValueRef,
                            Name: *c_char)
                         -> ValueRef;
    unsafe fn LLVMBuildNot(B: BuilderRef,
                           V: ValueRef,
                           Name: *c_char)
                        -> ValueRef;

    /* Memory */
    unsafe fn LLVMBuildMalloc(B: BuilderRef,
                              Ty: TypeRef,
                              Name: *c_char)
                           -> ValueRef;
    unsafe fn LLVMBuildArrayMalloc(B: BuilderRef, Ty: TypeRef, Val: ValueRef,
                            Name: *c_char) -> ValueRef;
    unsafe fn LLVMBuildAlloca(B: BuilderRef,
                              Ty: TypeRef,
                              Name: *c_char)
                           -> ValueRef;
    unsafe fn LLVMBuildArrayAlloca(B: BuilderRef, Ty: TypeRef, Val: ValueRef,
                            Name: *c_char) -> ValueRef;
    unsafe fn LLVMBuildFree(B: BuilderRef, PointerVal: ValueRef) -> ValueRef;
    unsafe fn LLVMBuildLoad(B: BuilderRef,
                            PointerVal: ValueRef,
                            Name: *c_char)
                         -> ValueRef;
    unsafe fn LLVMBuildStore(B: BuilderRef, Val: ValueRef, Ptr: ValueRef) ->
       ValueRef;
    unsafe fn LLVMBuildGEP(B: BuilderRef,
                           Pointer: ValueRef,
                           Indices: *ValueRef,
                           NumIndices: c_uint,
                           Name: *c_char)
                        -> ValueRef;
    unsafe fn LLVMBuildInBoundsGEP(B: BuilderRef, Pointer: ValueRef,
                            Indices: *ValueRef, NumIndices: c_uint,
                            Name: *c_char)
       -> ValueRef;
    unsafe fn LLVMBuildStructGEP(B: BuilderRef,
                                 Pointer: ValueRef,
                                 Idx: c_uint,
                                 Name: *c_char)
                              -> ValueRef;
    unsafe fn LLVMBuildGlobalString(B: BuilderRef,
                                    Str: *c_char,
                                    Name: *c_char)
                                 -> ValueRef;
    unsafe fn LLVMBuildGlobalStringPtr(B: BuilderRef,
                                       Str: *c_char,
                                       Name: *c_char)
                                    -> ValueRef;

    /* Casts */
    unsafe fn LLVMBuildTrunc(B: BuilderRef, Val: ValueRef, DestTy: TypeRef,
                      Name: *c_char) -> ValueRef;
    unsafe fn LLVMBuildZExt(B: BuilderRef, Val: ValueRef, DestTy: TypeRef,
                     Name: *c_char) -> ValueRef;
    unsafe fn LLVMBuildSExt(B: BuilderRef, Val: ValueRef, DestTy: TypeRef,
                     Name: *c_char) -> ValueRef;
    unsafe fn LLVMBuildFPToUI(B: BuilderRef, Val: ValueRef, DestTy: TypeRef,
                       Name: *c_char) -> ValueRef;
    unsafe fn LLVMBuildFPToSI(B: BuilderRef, Val: ValueRef, DestTy: TypeRef,
                       Name: *c_char) -> ValueRef;
    unsafe fn LLVMBuildUIToFP(B: BuilderRef, Val: ValueRef, DestTy: TypeRef,
                       Name: *c_char) -> ValueRef;
    unsafe fn LLVMBuildSIToFP(B: BuilderRef, Val: ValueRef, DestTy: TypeRef,
                       Name: *c_char) -> ValueRef;
    unsafe fn LLVMBuildFPTrunc(B: BuilderRef, Val: ValueRef, DestTy: TypeRef,
                        Name: *c_char) -> ValueRef;
    unsafe fn LLVMBuildFPExt(B: BuilderRef, Val: ValueRef, DestTy: TypeRef,
                      Name: *c_char) -> ValueRef;
    unsafe fn LLVMBuildPtrToInt(B: BuilderRef, Val: ValueRef, DestTy: TypeRef,
                         Name: *c_char) -> ValueRef;
    unsafe fn LLVMBuildIntToPtr(B: BuilderRef, Val: ValueRef, DestTy: TypeRef,
                         Name: *c_char) -> ValueRef;
    unsafe fn LLVMBuildBitCast(B: BuilderRef, Val: ValueRef, DestTy: TypeRef,
                        Name: *c_char) -> ValueRef;
    unsafe fn LLVMBuildZExtOrBitCast(B: BuilderRef,
                                     Val: ValueRef,
                                     DestTy: TypeRef,
                                     Name: *c_char)
                                  -> ValueRef;
    unsafe fn LLVMBuildSExtOrBitCast(B: BuilderRef,
                                     Val: ValueRef,
                                     DestTy: TypeRef,
                                     Name: *c_char)
                                  -> ValueRef;
    unsafe fn LLVMBuildTruncOrBitCast(B: BuilderRef,
                                      Val: ValueRef,
                                      DestTy: TypeRef,
                                      Name: *c_char)
                                   -> ValueRef;
    unsafe fn LLVMBuildCast(B: BuilderRef, Op: Opcode, Val: ValueRef,
                     DestTy: TypeRef, Name: *c_char) -> ValueRef;
    unsafe fn LLVMBuildPointerCast(B: BuilderRef,
                                   Val: ValueRef,
                                   DestTy: TypeRef,
                                   Name: *c_char)
                                -> ValueRef;
    unsafe fn LLVMBuildIntCast(B: BuilderRef, Val: ValueRef, DestTy: TypeRef,
                        Name: *c_char) -> ValueRef;
    unsafe fn LLVMBuildFPCast(B: BuilderRef, Val: ValueRef, DestTy: TypeRef,
                       Name: *c_char) -> ValueRef;

    /* Comparisons */
    unsafe fn LLVMBuildICmp(B: BuilderRef, Op: c_uint, LHS: ValueRef,
                     RHS: ValueRef, Name: *c_char) -> ValueRef;
    unsafe fn LLVMBuildFCmp(B: BuilderRef, Op: c_uint, LHS: ValueRef,
                     RHS: ValueRef, Name: *c_char) -> ValueRef;

    /* Miscellaneous instructions */
    unsafe fn LLVMBuildPhi(B: BuilderRef,
                           Ty: TypeRef,
                           Name: *c_char)
                        -> ValueRef;
    unsafe fn LLVMBuildCall(B: BuilderRef, Fn: ValueRef, Args: *ValueRef,
                     NumArgs: c_uint, Name: *c_char) -> ValueRef;
    unsafe fn LLVMBuildSelect(B: BuilderRef, If: ValueRef, Then: ValueRef,
                       Else: ValueRef, Name: *c_char) -> ValueRef;
    unsafe fn LLVMBuildVAArg(B: BuilderRef, list: ValueRef, Ty: TypeRef,
                      Name: *c_char)
       -> ValueRef;
    unsafe fn LLVMBuildExtractElement(B: BuilderRef,
                                      VecVal: ValueRef,
                                      Index: ValueRef,
                                      Name: *c_char)
                                   -> ValueRef;
    unsafe fn LLVMBuildInsertElement(B: BuilderRef,
                                     VecVal: ValueRef,
                                     EltVal: ValueRef,
                                     Index: ValueRef,
                                     Name: *c_char)
                                  -> ValueRef;
    unsafe fn LLVMBuildShuffleVector(B: BuilderRef,
                                     V1: ValueRef,
                                     V2: ValueRef,
                                     Mask: ValueRef,
                                     Name: *c_char)
                                  -> ValueRef;
    unsafe fn LLVMBuildExtractValue(B: BuilderRef,
                                    AggVal: ValueRef,
                                    Index: c_uint,
                                    Name: *c_char)
                                 -> ValueRef;
    unsafe fn LLVMBuildInsertValue(B: BuilderRef,
                                   AggVal: ValueRef,
                                   EltVal: ValueRef,
                                   Index: c_uint,
                                   Name: *c_char)
                                -> ValueRef;

    unsafe fn LLVMBuildIsNull(B: BuilderRef, Val: ValueRef, Name: *c_char)
                           -> ValueRef;
    unsafe fn LLVMBuildIsNotNull(B: BuilderRef, Val: ValueRef, Name: *c_char)
                              -> ValueRef;
    unsafe fn LLVMBuildPtrDiff(B: BuilderRef, LHS: ValueRef, RHS: ValueRef,
                        Name: *c_char) -> ValueRef;

    /* Atomic Operations */
    unsafe fn LLVMBuildAtomicCmpXchg(B: BuilderRef, LHS: ValueRef,
                              CMP: ValueRef, RHS: ValueRef,
                              ++Order: AtomicOrdering) -> ValueRef;
    unsafe fn LLVMBuildAtomicRMW(B: BuilderRef, ++Op: AtomicBinOp,
                          LHS: ValueRef, RHS: ValueRef,
                          ++Order: AtomicOrdering) -> ValueRef;

    /* Selected entries from the downcasts. */
    unsafe fn LLVMIsATerminatorInst(Inst: ValueRef) -> ValueRef;

    /** Writes a module to the specified path. Returns 0 on success. */
    unsafe fn LLVMWriteBitcodeToFile(M: ModuleRef, Path: *c_char) -> c_int;

    /** Creates target data from a target layout string. */
    unsafe fn LLVMCreateTargetData(StringRep: *c_char) -> TargetDataRef;
    /** Adds the target data to the given pass manager. The pass manager
        references the target data only weakly. */
    unsafe fn LLVMAddTargetData(TD: TargetDataRef, PM: PassManagerRef);
    /** Number of bytes clobbered when doing a Store to *T. */
    unsafe fn LLVMStoreSizeOfType(TD: TargetDataRef, Ty: TypeRef)
        -> c_ulonglong;

    /** Number of bytes clobbered when doing a Store to *T. */
    unsafe fn LLVMSizeOfTypeInBits(TD: TargetDataRef, Ty: TypeRef)
        -> c_ulonglong;

    /** Distance between successive elements in an array of T.
    Includes ABI padding. */
    unsafe fn LLVMABISizeOfType(TD: TargetDataRef, Ty: TypeRef) -> c_uint;

    /** Returns the preferred alignment of a type. */
    unsafe fn LLVMPreferredAlignmentOfType(TD: TargetDataRef,
                                    Ty: TypeRef) -> c_uint;
    /** Returns the minimum alignment of a type. */
    unsafe fn LLVMABIAlignmentOfType(TD: TargetDataRef,
                              Ty: TypeRef) -> c_uint;
    /** Returns the minimum alignment of a type when part of a call frame. */
    unsafe fn LLVMCallFrameAlignmentOfType(TD: TargetDataRef,
                                    Ty: TypeRef) -> c_uint;

    /** Disposes target data. */
    unsafe fn LLVMDisposeTargetData(TD: TargetDataRef);

    /** Creates a pass manager. */
    unsafe fn LLVMCreatePassManager() -> PassManagerRef;
    /** Disposes a pass manager. */
    unsafe fn LLVMDisposePassManager(PM: PassManagerRef);
    /** Runs a pass manager on a module. */
    unsafe fn LLVMRunPassManager(PM: PassManagerRef, M: ModuleRef) -> Bool;

    /** Adds a verification pass. */
    unsafe fn LLVMAddVerifierPass(PM: PassManagerRef);

    unsafe fn LLVMAddGlobalOptimizerPass(PM: PassManagerRef);
    unsafe fn LLVMAddIPSCCPPass(PM: PassManagerRef);
    unsafe fn LLVMAddDeadArgEliminationPass(PM: PassManagerRef);
    unsafe fn LLVMAddInstructionCombiningPass(PM: PassManagerRef);
    unsafe fn LLVMAddCFGSimplificationPass(PM: PassManagerRef);
    unsafe fn LLVMAddFunctionInliningPass(PM: PassManagerRef);
    unsafe fn LLVMAddFunctionAttrsPass(PM: PassManagerRef);
    unsafe fn LLVMAddScalarReplAggregatesPass(PM: PassManagerRef);
    unsafe fn LLVMAddScalarReplAggregatesPassSSA(PM: PassManagerRef);
    unsafe fn LLVMAddJumpThreadingPass(PM: PassManagerRef);
    unsafe fn LLVMAddConstantPropagationPass(PM: PassManagerRef);
    unsafe fn LLVMAddReassociatePass(PM: PassManagerRef);
    unsafe fn LLVMAddLoopRotatePass(PM: PassManagerRef);
    unsafe fn LLVMAddLICMPass(PM: PassManagerRef);
    unsafe fn LLVMAddLoopUnswitchPass(PM: PassManagerRef);
    unsafe fn LLVMAddLoopDeletionPass(PM: PassManagerRef);
    unsafe fn LLVMAddLoopUnrollPass(PM: PassManagerRef);
    unsafe fn LLVMAddGVNPass(PM: PassManagerRef);
    unsafe fn LLVMAddMemCpyOptPass(PM: PassManagerRef);
    unsafe fn LLVMAddSCCPPass(PM: PassManagerRef);
    unsafe fn LLVMAddDeadStoreEliminationPass(PM: PassManagerRef);
    unsafe fn LLVMAddStripDeadPrototypesPass(PM: PassManagerRef);
    unsafe fn LLVMAddConstantMergePass(PM: PassManagerRef);
    unsafe fn LLVMAddArgumentPromotionPass(PM: PassManagerRef);
    unsafe fn LLVMAddTailCallEliminationPass(PM: PassManagerRef);
    unsafe fn LLVMAddIndVarSimplifyPass(PM: PassManagerRef);
    unsafe fn LLVMAddAggressiveDCEPass(PM: PassManagerRef);
    unsafe fn LLVMAddGlobalDCEPass(PM: PassManagerRef);
    unsafe fn LLVMAddCorrelatedValuePropagationPass(PM: PassManagerRef);
    unsafe fn LLVMAddPruneEHPass(PM: PassManagerRef);
    unsafe fn LLVMAddSimplifyLibCallsPass(PM: PassManagerRef);
    unsafe fn LLVMAddLoopIdiomPass(PM: PassManagerRef);
    unsafe fn LLVMAddEarlyCSEPass(PM: PassManagerRef);
    unsafe fn LLVMAddTypeBasedAliasAnalysisPass(PM: PassManagerRef);
    unsafe fn LLVMAddBasicAliasAnalysisPass(PM: PassManagerRef);

    unsafe fn LLVMPassManagerBuilderCreate() -> PassManagerBuilderRef;
    unsafe fn LLVMPassManagerBuilderDispose(PMB: PassManagerBuilderRef);
    unsafe fn LLVMPassManagerBuilderSetOptLevel(PMB: PassManagerBuilderRef,
                                         OptimizationLevel: c_uint);
    unsafe fn LLVMPassManagerBuilderSetSizeLevel(PMB: PassManagerBuilderRef,
                                          Value: Bool);
    unsafe fn LLVMPassManagerBuilderSetDisableUnitAtATime(
        PMB: PassManagerBuilderRef, Value: Bool);
    unsafe fn LLVMPassManagerBuilderSetDisableUnrollLoops(
        PMB: PassManagerBuilderRef, Value: Bool);
    unsafe fn LLVMPassManagerBuilderSetDisableSimplifyLibCalls
        (PMB: PassManagerBuilderRef, Value: Bool);
    unsafe fn LLVMPassManagerBuilderUseInlinerWithThreshold
        (PMB: PassManagerBuilderRef, threshold: c_uint);
    unsafe fn LLVMPassManagerBuilderPopulateModulePassManager
        (PMB: PassManagerBuilderRef, PM: PassManagerRef);

    unsafe fn LLVMPassManagerBuilderPopulateFunctionPassManager
        (PMB: PassManagerBuilderRef, PM: PassManagerRef);

    /** Destroys a memory buffer. */
    unsafe fn LLVMDisposeMemoryBuffer(MemBuf: MemoryBufferRef);


    /* Stuff that's in rustllvm/ because it's not upstream yet. */

    /** Opens an object file. */
    unsafe fn LLVMCreateObjectFile(MemBuf: MemoryBufferRef) -> ObjectFileRef;
    /** Closes an object file. */
    unsafe fn LLVMDisposeObjectFile(ObjFile: ObjectFileRef);

    /** Enumerates the sections in an object file. */
    unsafe fn LLVMGetSections(ObjFile: ObjectFileRef) -> SectionIteratorRef;
    /** Destroys a section iterator. */
    unsafe fn LLVMDisposeSectionIterator(SI: SectionIteratorRef);
    /** Returns true if the section iterator is at the end of the section
        list: */
    unsafe fn LLVMIsSectionIteratorAtEnd(ObjFile: ObjectFileRef,
                                  SI: SectionIteratorRef) -> Bool;
    /** Moves the section iterator to point to the next section. */
    unsafe fn LLVMMoveToNextSection(SI: SectionIteratorRef);
    /** Returns the current section name. */
    unsafe fn LLVMGetSectionName(SI: SectionIteratorRef) -> *c_char;
    /** Returns the current section size. */
    unsafe fn LLVMGetSectionSize(SI: SectionIteratorRef) -> c_ulonglong;
    /** Returns the current section contents as a string buffer. */
    unsafe fn LLVMGetSectionContents(SI: SectionIteratorRef) -> *c_char;

    /** Reads the given file and returns it as a memory buffer. Use
        LLVMDisposeMemoryBuffer() to get rid of it. */
    unsafe fn LLVMRustCreateMemoryBufferWithContentsOfFile(Path: *c_char) ->
       MemoryBufferRef;

    unsafe fn LLVMRustWriteOutputFile(PM: PassManagerRef, M: ModuleRef,
                               Triple: *c_char,
                               // FIXME: When #2334 is fixed, change
                               // c_uint to FileType
                               Output: *c_char, FileType: c_uint,
                               OptLevel: c_int,
                               EnableSegmentedStacks: bool) -> bool;

    /** Returns a string describing the last error caused by an LLVMRust*
        call. */
    unsafe fn LLVMRustGetLastError() -> *c_char;

    /** Prepare the JIT. Returns a memory manager that can load crates. */
    unsafe fn LLVMRustPrepareJIT(__morestack: *()) -> *();

    /** Load a crate into the memory manager. */
    unsafe fn LLVMRustLoadCrate(MM: *(),
                         Filename: *c_char) -> bool;

    /** Execute the JIT engine. */
    unsafe fn LLVMRustExecuteJIT(MM: *(),
                          PM: PassManagerRef,
                          M: ModuleRef,
                          OptLevel: c_int,
                          EnableSegmentedStacks: bool) -> *();

    /** Parses the bitcode in the given memory buffer. */
    unsafe fn LLVMRustParseBitcode(MemBuf: MemoryBufferRef) -> ModuleRef;

    /** Parses LLVM asm in the given file */
    unsafe fn LLVMRustParseAssemblyFile(Filename: *c_char) -> ModuleRef;

    unsafe fn LLVMRustAddPrintModulePass(PM: PassManagerRef, M: ModuleRef,
                                  Output: *c_char);

    /** Turn on LLVM pass-timing. */
    unsafe fn LLVMRustEnableTimePasses();

    /** Print the pass timings since static dtors aren't picking them up. */
    unsafe fn LLVMRustPrintPassTimings();

    unsafe fn LLVMStructCreateNamed(C: ContextRef, Name: *c_char) -> TypeRef;

    unsafe fn LLVMStructSetBody(StructTy: TypeRef, ElementTypes: *TypeRef,
                         ElementCount: c_uint, Packed: Bool);

    unsafe fn LLVMConstNamedStruct(S: TypeRef, ConstantVals: *ValueRef,
                            Count: c_uint) -> ValueRef;

    /** Enables LLVM debug output. */
    unsafe fn LLVMSetDebug(Enabled: c_int);
}

fn SetInstructionCallConv(Instr: ValueRef, CC: CallConv) {
    unsafe {
        llvm::LLVMSetInstructionCallConv(Instr, CC as c_uint);
    }
}
fn SetFunctionCallConv(Fn: ValueRef, CC: CallConv) {
    unsafe {
        llvm::LLVMSetFunctionCallConv(Fn, CC as c_uint);
    }
}
fn SetLinkage(Global: ValueRef, Link: Linkage) {
    unsafe {
        llvm::LLVMSetLinkage(Global, Link as c_uint);
    }
}

/* Memory-managed object interface to type handles. */

type type_names = @{type_names: HashMap<TypeRef, @str>,
                    named_types: HashMap<@str, TypeRef>};

fn associate_type(tn: type_names, s: @str, t: TypeRef) {
    assert tn.type_names.insert(t, s);
    assert tn.named_types.insert(s, t);
}

fn type_has_name(tn: type_names, t: TypeRef) -> Option<@str> {
    return tn.type_names.find(t);
}

fn name_has_type(tn: type_names, s: @str) -> Option<TypeRef> {
    return tn.named_types.find(s);
}

fn mk_type_names() -> type_names {
    @{type_names: HashMap(),
      named_types: HashMap()}
}

fn type_to_str(names: type_names, ty: TypeRef) -> @str {
    return type_to_str_inner(names, ~[], ty);
}

fn type_to_str_inner(names: type_names, +outer0: ~[TypeRef], ty: TypeRef) ->
   @str {
    unsafe {
        match type_has_name(names, ty) {
          option::Some(n) => return n,
          _ => {}
        }

        // FIXME #2543: Bad copy.
        let outer = vec::append_one(copy outer0, ty);

        let kind = llvm::LLVMGetTypeKind(ty);

        fn tys_str(names: type_names, outer: ~[TypeRef],
                   tys: ~[TypeRef]) -> @str {
            let mut s = ~"";
            let mut first: bool = true;
            for tys.each |t| {
                if first { first = false; } else { s += ~", "; }
                s += type_to_str_inner(names, outer, *t).to_owned();
            }
            // [Note at-str] FIXME #2543: Could rewrite this without the copy,
            // but need better @str support.
            return s.to_managed();
        }

        match kind {
          Void => return @"Void",
          Half => return @"Half",
          Float => return @"Float",
          Double => return @"Double",
          X86_FP80 => return @"X86_FP80",
          FP128 => return @"FP128",
          PPC_FP128 => return @"PPC_FP128",
          Label => return @"Label",
          Integer => {
            // See [Note at-str]
            return fmt!("i%d", llvm::LLVMGetIntTypeWidth(ty)
                        as int).to_managed();
          }
          Function => {
            let out_ty: TypeRef = llvm::LLVMGetReturnType(ty);
            let n_args = llvm::LLVMCountParamTypes(ty) as uint;
            let args = vec::from_elem(n_args, 0 as TypeRef);
            unsafe {
                llvm::LLVMGetParamTypes(ty, vec::raw::to_ptr(args));
            }
            // See [Note at-str]
            return fmt!("fn(%s) -> %s",
                        tys_str(names, outer, args),
                        type_to_str_inner(names, outer, out_ty)).to_managed();
          }
          Struct => {
            let n_elts = llvm::LLVMCountStructElementTypes(ty) as uint;
            let mut elts = vec::from_elem(n_elts, 0 as TypeRef);
            if elts.is_not_empty() {
                llvm::LLVMGetStructElementTypes(
                    ty, ptr::to_mut_unsafe_ptr(&mut elts[0]));
            }
            // See [Note at-str]
            return fmt!("{%s}", tys_str(names, outer, elts)).to_managed();
          }
          Array => {
            let el_ty = llvm::LLVMGetElementType(ty);
            // See [Note at-str]
            return fmt!("[%s@ x %u", type_to_str_inner(names, outer, el_ty),
                llvm::LLVMGetArrayLength(ty) as uint).to_managed();
          }
          Pointer => {
            let mut i = 0;
            for outer0.each |tout| {
                i += 1;
                if *tout as int == ty as int {
                    let n = outer0.len() - i;
                    // See [Note at-str]
                    return fmt!("*\\%d", n as int).to_managed();
                }
            }
            let addrstr = {
                let addrspace = llvm::LLVMGetPointerAddressSpace(ty) as uint;
                if addrspace == 0 {
                    ~""
                } else {
                    fmt!("addrspace(%u)", addrspace)
                }
            };
            // See [Note at-str]
            return fmt!("%s*%s", addrstr, type_to_str_inner(names,
                        outer,
                        llvm::LLVMGetElementType(ty))).to_managed();
          }
          Vector => return @"Vector",
          Metadata => return @"Metadata",
          X86_MMX => return @"X86_MMAX"
        }
    }
}

fn float_width(llt: TypeRef) -> uint {
    unsafe {
        return match llvm::LLVMGetTypeKind(llt) as int {
              1 => 32u,
              2 => 64u,
              3 => 80u,
              4 | 5 => 128u,
              _ => fail ~"llvm_float_width called on a non-float type"
            };
    }
}

fn fn_ty_param_tys(fn_ty: TypeRef) -> ~[TypeRef] unsafe {
    unsafe {
        let args = vec::from_elem(llvm::LLVMCountParamTypes(fn_ty) as uint,
                                 0 as TypeRef);
        llvm::LLVMGetParamTypes(fn_ty, vec::raw::to_ptr(args));
        return args;
    }
}

fn struct_element_types(struct_ty: TypeRef) -> ~[TypeRef] {
    unsafe {
        let count = llvm::LLVMCountStructElementTypes(struct_ty);
        let mut buf: ~[TypeRef] =
            vec::from_elem(count as uint,
                           cast::transmute::<uint,TypeRef>(0));
        if buf.len() > 0 {
            llvm::LLVMGetStructElementTypes(
                struct_ty, ptr::to_mut_unsafe_ptr(&mut buf[0]));
        }
        return move buf;
    }
}


/* Memory-managed interface to target data. */

struct target_data_res {
    TD: TargetDataRef,
    drop {
        unsafe {
            llvm::LLVMDisposeTargetData(self.TD);
        }
    }
}

fn target_data_res(TD: TargetDataRef) -> target_data_res {
    target_data_res {
        TD: TD
    }
}

type target_data = {lltd: TargetDataRef, dtor: @target_data_res};

fn mk_target_data(string_rep: ~str) -> target_data {
    let lltd =
        str::as_c_str(string_rep, |buf| unsafe {
            llvm::LLVMCreateTargetData(buf)
        });
    return {lltd: lltd, dtor: @target_data_res(lltd)};
}

/* Memory-managed interface to pass managers. */

struct pass_manager_res {
    PM: PassManagerRef,
    drop {
        unsafe {
            llvm::LLVMDisposePassManager(self.PM);
        }
    }
}

fn pass_manager_res(PM: PassManagerRef) -> pass_manager_res {
    pass_manager_res {
        PM: PM
    }
}

type pass_manager = {llpm: PassManagerRef, dtor: @pass_manager_res};

fn mk_pass_manager() -> pass_manager {
    unsafe {
        let llpm = llvm::LLVMCreatePassManager();
        return {llpm: llpm, dtor: @pass_manager_res(llpm)};
    }
}

/* Memory-managed interface to object files. */

struct object_file_res {
    ObjectFile: ObjectFileRef,
    drop {
        unsafe {
            llvm::LLVMDisposeObjectFile(self.ObjectFile);
        }
    }
}

fn object_file_res(ObjFile: ObjectFileRef) -> object_file_res {
    object_file_res {
        ObjectFile: ObjFile
    }
}

type object_file = {llof: ObjectFileRef, dtor: @object_file_res};

fn mk_object_file(llmb: MemoryBufferRef) -> Option<object_file> {
    unsafe {
        let llof = llvm::LLVMCreateObjectFile(llmb);
        if llof as int == 0 { return option::None::<object_file>; }
        return option::Some({llof: llof, dtor: @object_file_res(llof)});
    }
}

/* Memory-managed interface to section iterators. */

struct section_iter_res {
    SI: SectionIteratorRef,
    drop {
        unsafe {
            llvm::LLVMDisposeSectionIterator(self.SI);
        }
    }
}

fn section_iter_res(SI: SectionIteratorRef) -> section_iter_res {
    section_iter_res {
        SI: SI
    }
}

type section_iter = {llsi: SectionIteratorRef, dtor: @section_iter_res};

fn mk_section_iter(llof: ObjectFileRef) -> section_iter {
    unsafe {
        let llsi = llvm::LLVMGetSections(llof);
        return {llsi: llsi, dtor: @section_iter_res(llsi)};
    }
}

//
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
//
