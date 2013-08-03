// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use std::hashmap::HashMap;
use std::libc::{c_uint, c_ushort};
use std::option;

use middle::trans::type_::Type;

pub type Opcode = u32;
pub type Bool = c_uint;

pub static True: Bool = 1 as Bool;
pub static False: Bool = 0 as Bool;

// Consts for the LLVM CallConv type, pre-cast to uint.

pub enum CallConv {
    CCallConv = 0,
    FastCallConv = 8,
    ColdCallConv = 9,
    X86StdcallCallConv = 64,
    X86FastcallCallConv = 65,
}

pub enum Visibility {
    LLVMDefaultVisibility = 0,
    HiddenVisibility = 1,
    ProtectedVisibility = 2,
}

pub enum Linkage {
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

#[deriving(Clone)]
pub enum Attribute {
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
pub enum IntPredicate {
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
pub enum RealPredicate {
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

// The LLVM TypeKind type - must stay in sync with the def of
// LLVMTypeKind in llvm/include/llvm-c/Core.h
pub type TypeKind = u32;
pub static Void: TypeKind      = 0;
pub static Half: TypeKind      = 1;
pub static Float: TypeKind     = 2;
pub static Double: TypeKind    = 3;
pub static X86_FP80: TypeKind  = 4;
pub static FP128: TypeKind     = 5;
pub static PPC_FP128: TypeKind = 6;
pub static Label: TypeKind     = 7;
pub static Integer: TypeKind   = 8;
pub static Function: TypeKind  = 9;
pub static Struct: TypeKind    = 10;
pub static Array: TypeKind     = 11;
pub static Pointer: TypeKind   = 12;
pub static Vector: TypeKind    = 13;
pub static Metadata: TypeKind  = 14;
pub static X86_MMX: TypeKind   = 15;

pub enum AtomicBinOp {
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

pub enum AtomicOrdering {
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
pub enum FileType {
    AssemblyFile = 0,
    ObjectFile = 1
}

pub enum Metadata {
    MD_dbg = 0,
    MD_tbaa = 1,
    MD_prof = 2,
    MD_fpmath = 3,
    MD_range = 4,
    MD_tbaa_struct = 5
}

// Inline Asm Dialect
pub enum AsmDialect {
    AD_ATT   = 0,
    AD_Intel = 1
}

// Opaque pointer types
pub enum Module_opaque {}
pub type ModuleRef = *Module_opaque;
pub enum Context_opaque {}
pub type ContextRef = *Context_opaque;
pub enum Type_opaque {}
pub type TypeRef = *Type_opaque;
pub enum Value_opaque {}
pub type ValueRef = *Value_opaque;
pub enum BasicBlock_opaque {}
pub type BasicBlockRef = *BasicBlock_opaque;
pub enum Builder_opaque {}
pub type BuilderRef = *Builder_opaque;
pub enum ExecutionEngine_opaque {}
pub type ExecutionEngineRef = *ExecutionEngine_opaque;
pub enum MemoryBuffer_opaque {}
pub type MemoryBufferRef = *MemoryBuffer_opaque;
pub enum PassManager_opaque {}
pub type PassManagerRef = *PassManager_opaque;
pub enum PassManagerBuilder_opaque {}
pub type PassManagerBuilderRef = *PassManagerBuilder_opaque;
pub enum Use_opaque {}
pub type UseRef = *Use_opaque;
pub enum TargetData_opaque {}
pub type TargetDataRef = *TargetData_opaque;
pub enum ObjectFile_opaque {}
pub type ObjectFileRef = *ObjectFile_opaque;
pub enum SectionIterator_opaque {}
pub type SectionIteratorRef = *SectionIterator_opaque;
pub enum Pass_opaque {}
pub type PassRef = *Pass_opaque;

pub mod debuginfo {
    use super::{ValueRef};

    pub enum DIBuilder_opaque {}
    pub type DIBuilderRef = *DIBuilder_opaque;

    pub type DIDescriptor = ValueRef;
    pub type DIScope = DIDescriptor;
    pub type DILocation = DIDescriptor;
    pub type DIFile = DIScope;
    pub type DILexicalBlock = DIScope;
    pub type DISubprogram = DIScope;
    pub type DIType = DIDescriptor;
    pub type DIBasicType = DIType;
    pub type DIDerivedType = DIType;
    pub type DICompositeType = DIDerivedType;
    pub type DIVariable = DIDescriptor;
    pub type DIArray = DIDescriptor;
    pub type DISubrange = DIDescriptor;

    pub enum DIDescriptorFlags {
      FlagPrivate            = 1 << 0,
      FlagProtected          = 1 << 1,
      FlagFwdDecl            = 1 << 2,
      FlagAppleBlock         = 1 << 3,
      FlagBlockByrefStruct   = 1 << 4,
      FlagVirtual            = 1 << 5,
      FlagArtificial         = 1 << 6,
      FlagExplicit           = 1 << 7,
      FlagPrototyped         = 1 << 8,
      FlagObjcClassComplete  = 1 << 9,
      FlagObjectPointer      = 1 << 10,
      FlagVector             = 1 << 11,
      FlagStaticMember       = 1 << 12
    }
}

pub mod llvm {
    use super::{AtomicBinOp, AtomicOrdering, BasicBlockRef, ExecutionEngineRef};
    use super::{Bool, BuilderRef, ContextRef, MemoryBufferRef, ModuleRef};
    use super::{ObjectFileRef, Opcode, PassManagerRef, PassManagerBuilderRef};
    use super::{SectionIteratorRef, TargetDataRef, TypeKind, TypeRef, UseRef};
    use super::{ValueRef, PassRef};
    use super::debuginfo::*;
    use std::libc::{c_char, c_int, c_longlong, c_ushort, c_uint, c_ulonglong};

    #[link_args = "-Lrustllvm -lrustllvm"]
    #[link_name = "rustllvm"]
    #[abi = "cdecl"]
    extern {
        /* Create and destroy contexts. */
        #[fast_ffi]
        pub fn LLVMContextCreate() -> ContextRef;
        #[fast_ffi]
        pub fn LLVMContextDispose(C: ContextRef);
        #[fast_ffi]
        pub fn LLVMGetMDKindIDInContext(C: ContextRef,
                                        Name: *c_char,
                                        SLen: c_uint)
                                        -> c_uint;

        /* Create and destroy modules. */
        #[fast_ffi]
        pub fn LLVMModuleCreateWithNameInContext(ModuleID: *c_char,
                                                 C: ContextRef)
                                                 -> ModuleRef;
        #[fast_ffi]
        pub fn LLVMGetModuleContext(M: ModuleRef) -> ContextRef;
        #[fast_ffi]
        pub fn LLVMDisposeModule(M: ModuleRef);

        /** Data layout. See Module::getDataLayout. */
        #[fast_ffi]
        pub fn LLVMGetDataLayout(M: ModuleRef) -> *c_char;
        #[fast_ffi]
        pub fn LLVMSetDataLayout(M: ModuleRef, Triple: *c_char);

        /** Target triple. See Module::getTargetTriple. */
        #[fast_ffi]
        pub fn LLVMGetTarget(M: ModuleRef) -> *c_char;
        #[fast_ffi]
        pub fn LLVMSetTarget(M: ModuleRef, Triple: *c_char);

        /** See Module::dump. */
        #[fast_ffi]
        pub fn LLVMDumpModule(M: ModuleRef);

        /** See Module::setModuleInlineAsm. */
        #[fast_ffi]
        pub fn LLVMSetModuleInlineAsm(M: ModuleRef, Asm: *c_char);

        /** See llvm::LLVMTypeKind::getTypeID. */
        pub fn LLVMGetTypeKind(Ty: TypeRef) -> TypeKind;

        /** See llvm::LLVMType::getContext. */
        #[fast_ffi]
        pub fn LLVMGetTypeContext(Ty: TypeRef) -> ContextRef;

        /* Operations on integer types */
        #[fast_ffi]
        pub fn LLVMInt1TypeInContext(C: ContextRef) -> TypeRef;
        #[fast_ffi]
        pub fn LLVMInt8TypeInContext(C: ContextRef) -> TypeRef;
        #[fast_ffi]
        pub fn LLVMInt16TypeInContext(C: ContextRef) -> TypeRef;
        #[fast_ffi]
        pub fn LLVMInt32TypeInContext(C: ContextRef) -> TypeRef;
        #[fast_ffi]
        pub fn LLVMInt64TypeInContext(C: ContextRef) -> TypeRef;
        #[fast_ffi]
        pub fn LLVMIntTypeInContext(C: ContextRef, NumBits: c_uint)
                                    -> TypeRef;

        #[fast_ffi]
        pub fn LLVMGetIntTypeWidth(IntegerTy: TypeRef) -> c_uint;

        /* Operations on real types */
        #[fast_ffi]
        pub fn LLVMFloatTypeInContext(C: ContextRef) -> TypeRef;
        #[fast_ffi]
        pub fn LLVMDoubleTypeInContext(C: ContextRef) -> TypeRef;
        #[fast_ffi]
        pub fn LLVMX86FP80TypeInContext(C: ContextRef) -> TypeRef;
        #[fast_ffi]
        pub fn LLVMFP128TypeInContext(C: ContextRef) -> TypeRef;
        #[fast_ffi]
        pub fn LLVMPPCFP128TypeInContext(C: ContextRef) -> TypeRef;

        /* Operations on function types */
        #[fast_ffi]
        pub fn LLVMFunctionType(ReturnType: TypeRef,
                                ParamTypes: *TypeRef,
                                ParamCount: c_uint,
                                IsVarArg: Bool)
                                -> TypeRef;
        #[fast_ffi]
        pub fn LLVMIsFunctionVarArg(FunctionTy: TypeRef) -> Bool;
        #[fast_ffi]
        pub fn LLVMGetReturnType(FunctionTy: TypeRef) -> TypeRef;
        #[fast_ffi]
        pub fn LLVMCountParamTypes(FunctionTy: TypeRef) -> c_uint;
        #[fast_ffi]
        pub fn LLVMGetParamTypes(FunctionTy: TypeRef, Dest: *TypeRef);

        /* Operations on struct types */
        #[fast_ffi]
        pub fn LLVMStructTypeInContext(C: ContextRef,
                                       ElementTypes: *TypeRef,
                                       ElementCount: c_uint,
                                       Packed: Bool)
                                       -> TypeRef;
        #[fast_ffi]
        pub fn LLVMCountStructElementTypes(StructTy: TypeRef) -> c_uint;
        #[fast_ffi]
        pub fn LLVMGetStructElementTypes(StructTy: TypeRef,
                                         Dest: *mut TypeRef);
        #[fast_ffi]
        pub fn LLVMIsPackedStruct(StructTy: TypeRef) -> Bool;

        /* Operations on array, pointer, and vector types (sequence types) */
        #[fast_ffi]
        pub fn LLVMArrayType(ElementType: TypeRef, ElementCount: c_uint)
                             -> TypeRef;
        #[fast_ffi]
        pub fn LLVMPointerType(ElementType: TypeRef, AddressSpace: c_uint)
                               -> TypeRef;
        #[fast_ffi]
        pub fn LLVMVectorType(ElementType: TypeRef, ElementCount: c_uint)
                              -> TypeRef;

        #[fast_ffi]
        pub fn LLVMGetElementType(Ty: TypeRef) -> TypeRef;
        #[fast_ffi]
        pub fn LLVMGetArrayLength(ArrayTy: TypeRef) -> c_uint;
        #[fast_ffi]
        pub fn LLVMGetPointerAddressSpace(PointerTy: TypeRef) -> c_uint;
        #[fast_ffi]
        pub fn LLVMGetPointerToGlobal(EE: ExecutionEngineRef, V: ValueRef)
                                      -> *();
        #[fast_ffi]
        pub fn LLVMGetVectorSize(VectorTy: TypeRef) -> c_uint;

        /* Operations on other types */
        #[fast_ffi]
        pub fn LLVMVoidTypeInContext(C: ContextRef) -> TypeRef;
        #[fast_ffi]
        pub fn LLVMLabelTypeInContext(C: ContextRef) -> TypeRef;
        #[fast_ffi]
        pub fn LLVMMetadataTypeInContext(C: ContextRef) -> TypeRef;

        /* Operations on all values */
        #[fast_ffi]
        pub fn LLVMTypeOf(Val: ValueRef) -> TypeRef;
        #[fast_ffi]
        pub fn LLVMGetValueName(Val: ValueRef) -> *c_char;
        #[fast_ffi]
        pub fn LLVMSetValueName(Val: ValueRef, Name: *c_char);
        #[fast_ffi]
        pub fn LLVMDumpValue(Val: ValueRef);
        #[fast_ffi]
        pub fn LLVMReplaceAllUsesWith(OldVal: ValueRef, NewVal: ValueRef);
        #[fast_ffi]
        pub fn LLVMHasMetadata(Val: ValueRef) -> c_int;
        #[fast_ffi]
        pub fn LLVMGetMetadata(Val: ValueRef, KindID: c_uint) -> ValueRef;
        #[fast_ffi]
        pub fn LLVMSetMetadata(Val: ValueRef, KindID: c_uint, Node: ValueRef);

        /* Operations on Uses */
        #[fast_ffi]
        pub fn LLVMGetFirstUse(Val: ValueRef) -> UseRef;
        #[fast_ffi]
        pub fn LLVMGetNextUse(U: UseRef) -> UseRef;
        #[fast_ffi]
        pub fn LLVMGetUser(U: UseRef) -> ValueRef;
        #[fast_ffi]
        pub fn LLVMGetUsedValue(U: UseRef) -> ValueRef;

        /* Operations on Users */
        #[fast_ffi]
        pub fn LLVMGetNumOperands(Val: ValueRef) -> c_int;
        #[fast_ffi]
        pub fn LLVMGetOperand(Val: ValueRef, Index: c_uint) -> ValueRef;
        #[fast_ffi]
        pub fn LLVMSetOperand(Val: ValueRef, Index: c_uint, Op: ValueRef);

        /* Operations on constants of any type */
        #[fast_ffi]
        pub fn LLVMConstNull(Ty: TypeRef) -> ValueRef;
        /* all zeroes */
        #[fast_ffi]
        pub fn LLVMConstAllOnes(Ty: TypeRef) -> ValueRef;
        #[fast_ffi]
        pub fn LLVMConstICmp(Pred: c_ushort, V1: ValueRef, V2: ValueRef)
                             -> ValueRef;
        #[fast_ffi]
        pub fn LLVMConstFCmp(Pred: c_ushort, V1: ValueRef, V2: ValueRef)
                             -> ValueRef;
        /* only for int/vector */
        #[fast_ffi]
        pub fn LLVMGetUndef(Ty: TypeRef) -> ValueRef;
        #[fast_ffi]
        pub fn LLVMIsConstant(Val: ValueRef) -> Bool;
        #[fast_ffi]
        pub fn LLVMIsNull(Val: ValueRef) -> Bool;
        #[fast_ffi]
        pub fn LLVMIsUndef(Val: ValueRef) -> Bool;
        #[fast_ffi]
        pub fn LLVMConstPointerNull(Ty: TypeRef) -> ValueRef;

        /* Operations on metadata */
        #[fast_ffi]
        pub fn LLVMMDStringInContext(C: ContextRef,
                                     Str: *c_char,
                                     SLen: c_uint)
                                     -> ValueRef;
        #[fast_ffi]
        pub fn LLVMMDNodeInContext(C: ContextRef,
                                   Vals: *ValueRef,
                                   Count: c_uint)
                                   -> ValueRef;
        #[fast_ffi]
        pub fn LLVMAddNamedMetadataOperand(M: ModuleRef,
                                           Str: *c_char,
                                           Val: ValueRef);

        /* Operations on scalar constants */
        #[fast_ffi]
        pub fn LLVMConstInt(IntTy: TypeRef, N: c_ulonglong, SignExtend: Bool)
                            -> ValueRef;
        #[fast_ffi]
        pub fn LLVMConstIntOfString(IntTy: TypeRef, Text: *c_char, Radix: u8)
                                    -> ValueRef;
        #[fast_ffi]
        pub fn LLVMConstIntOfStringAndSize(IntTy: TypeRef,
                                           Text: *c_char,
                                           SLen: c_uint,
                                           Radix: u8)
                                           -> ValueRef;
        #[fast_ffi]
        pub fn LLVMConstReal(RealTy: TypeRef, N: f64) -> ValueRef;
        #[fast_ffi]
        pub fn LLVMConstRealOfString(RealTy: TypeRef, Text: *c_char)
                                     -> ValueRef;
        #[fast_ffi]
        pub fn LLVMConstRealOfStringAndSize(RealTy: TypeRef,
                                            Text: *c_char,
                                            SLen: c_uint)
                                            -> ValueRef;
        #[fast_ffi]
        pub fn LLVMConstIntGetZExtValue(ConstantVal: ValueRef) -> c_ulonglong;
        #[fast_ffi]
        pub fn LLVMConstIntGetSExtValue(ConstantVal: ValueRef) -> c_longlong;


        /* Operations on composite constants */
        #[fast_ffi]
        pub fn LLVMConstStringInContext(C: ContextRef,
                                        Str: *c_char,
                                        Length: c_uint,
                                        DontNullTerminate: Bool)
                                        -> ValueRef;
        #[fast_ffi]
        pub fn LLVMConstStructInContext(C: ContextRef,
                                        ConstantVals: *ValueRef,
                                        Count: c_uint,
                                        Packed: Bool)
                                        -> ValueRef;

        #[fast_ffi]
        pub fn LLVMConstArray(ElementTy: TypeRef,
                              ConstantVals: *ValueRef,
                              Length: c_uint)
                              -> ValueRef;
        #[fast_ffi]
        pub fn LLVMConstVector(ScalarConstantVals: *ValueRef, Size: c_uint)
                               -> ValueRef;

        /* Constant expressions */
        #[fast_ffi]
        pub fn LLVMAlignOf(Ty: TypeRef) -> ValueRef;
        #[fast_ffi]
        pub fn LLVMSizeOf(Ty: TypeRef) -> ValueRef;
        #[fast_ffi]
        pub fn LLVMConstNeg(ConstantVal: ValueRef) -> ValueRef;
        #[fast_ffi]
        pub fn LLVMConstNSWNeg(ConstantVal: ValueRef) -> ValueRef;
        #[fast_ffi]
        pub fn LLVMConstNUWNeg(ConstantVal: ValueRef) -> ValueRef;
        #[fast_ffi]
        pub fn LLVMConstFNeg(ConstantVal: ValueRef) -> ValueRef;
        #[fast_ffi]
        pub fn LLVMConstNot(ConstantVal: ValueRef) -> ValueRef;
        #[fast_ffi]
        pub fn LLVMConstAdd(LHSConstant: ValueRef, RHSConstant: ValueRef)
                            -> ValueRef;
        #[fast_ffi]
        pub fn LLVMConstNSWAdd(LHSConstant: ValueRef, RHSConstant: ValueRef)
                               -> ValueRef;
        #[fast_ffi]
        pub fn LLVMConstNUWAdd(LHSConstant: ValueRef, RHSConstant: ValueRef)
                               -> ValueRef;
        #[fast_ffi]
        pub fn LLVMConstFAdd(LHSConstant: ValueRef, RHSConstant: ValueRef)
                             -> ValueRef;
        #[fast_ffi]
        pub fn LLVMConstSub(LHSConstant: ValueRef, RHSConstant: ValueRef)
                            -> ValueRef;
        #[fast_ffi]
        pub fn LLVMConstNSWSub(LHSConstant: ValueRef, RHSConstant: ValueRef)
                               -> ValueRef;
        #[fast_ffi]
        pub fn LLVMConstNUWSub(LHSConstant: ValueRef, RHSConstant: ValueRef)
                               -> ValueRef;
        #[fast_ffi]
        pub fn LLVMConstFSub(LHSConstant: ValueRef, RHSConstant: ValueRef)
                             -> ValueRef;
        #[fast_ffi]
        pub fn LLVMConstMul(LHSConstant: ValueRef, RHSConstant: ValueRef)
                            -> ValueRef;
        #[fast_ffi]
        pub fn LLVMConstNSWMul(LHSConstant: ValueRef, RHSConstant: ValueRef)
                               -> ValueRef;
        #[fast_ffi]
        pub fn LLVMConstNUWMul(LHSConstant: ValueRef, RHSConstant: ValueRef)
                               -> ValueRef;
        #[fast_ffi]
        pub fn LLVMConstFMul(LHSConstant: ValueRef, RHSConstant: ValueRef)
                             -> ValueRef;
        #[fast_ffi]
        pub fn LLVMConstUDiv(LHSConstant: ValueRef, RHSConstant: ValueRef)
                             -> ValueRef;
        #[fast_ffi]
        pub fn LLVMConstSDiv(LHSConstant: ValueRef, RHSConstant: ValueRef)
                             -> ValueRef;
        #[fast_ffi]
        pub fn LLVMConstExactSDiv(LHSConstant: ValueRef,
                                  RHSConstant: ValueRef)
                                  -> ValueRef;
        #[fast_ffi]
        pub fn LLVMConstFDiv(LHSConstant: ValueRef, RHSConstant: ValueRef)
                             -> ValueRef;
        #[fast_ffi]
        pub fn LLVMConstURem(LHSConstant: ValueRef, RHSConstant: ValueRef)
                             -> ValueRef;
        #[fast_ffi]
        pub fn LLVMConstSRem(LHSConstant: ValueRef, RHSConstant: ValueRef)
                             -> ValueRef;
        #[fast_ffi]
        pub fn LLVMConstFRem(LHSConstant: ValueRef, RHSConstant: ValueRef)
                             -> ValueRef;
        #[fast_ffi]
        pub fn LLVMConstAnd(LHSConstant: ValueRef, RHSConstant: ValueRef)
                            -> ValueRef;
        #[fast_ffi]
        pub fn LLVMConstOr(LHSConstant: ValueRef, RHSConstant: ValueRef)
                           -> ValueRef;
        #[fast_ffi]
        pub fn LLVMConstXor(LHSConstant: ValueRef, RHSConstant: ValueRef)
                            -> ValueRef;
        #[fast_ffi]
        pub fn LLVMConstShl(LHSConstant: ValueRef, RHSConstant: ValueRef)
                            -> ValueRef;
        #[fast_ffi]
        pub fn LLVMConstLShr(LHSConstant: ValueRef, RHSConstant: ValueRef)
                             -> ValueRef;
        #[fast_ffi]
        pub fn LLVMConstAShr(LHSConstant: ValueRef, RHSConstant: ValueRef)
                             -> ValueRef;
        #[fast_ffi]
        pub fn LLVMConstGEP(ConstantVal: ValueRef,
                            ConstantIndices: *ValueRef,
                            NumIndices: c_uint)
                            -> ValueRef;
        #[fast_ffi]
        pub fn LLVMConstInBoundsGEP(ConstantVal: ValueRef,
                                    ConstantIndices: *ValueRef,
                                    NumIndices: c_uint)
                                    -> ValueRef;
        #[fast_ffi]
        pub fn LLVMConstTrunc(ConstantVal: ValueRef, ToType: TypeRef)
                              -> ValueRef;
        #[fast_ffi]
        pub fn LLVMConstSExt(ConstantVal: ValueRef, ToType: TypeRef)
                             -> ValueRef;
        #[fast_ffi]
        pub fn LLVMConstZExt(ConstantVal: ValueRef, ToType: TypeRef)
                             -> ValueRef;
        #[fast_ffi]
        pub fn LLVMConstFPTrunc(ConstantVal: ValueRef, ToType: TypeRef)
                                -> ValueRef;
        #[fast_ffi]
        pub fn LLVMConstFPExt(ConstantVal: ValueRef, ToType: TypeRef)
                              -> ValueRef;
        #[fast_ffi]
        pub fn LLVMConstUIToFP(ConstantVal: ValueRef, ToType: TypeRef)
                               -> ValueRef;
        #[fast_ffi]
        pub fn LLVMConstSIToFP(ConstantVal: ValueRef, ToType: TypeRef)
                               -> ValueRef;
        #[fast_ffi]
        pub fn LLVMConstFPToUI(ConstantVal: ValueRef, ToType: TypeRef)
                               -> ValueRef;
        #[fast_ffi]
        pub fn LLVMConstFPToSI(ConstantVal: ValueRef, ToType: TypeRef)
                               -> ValueRef;
        #[fast_ffi]
        pub fn LLVMConstPtrToInt(ConstantVal: ValueRef, ToType: TypeRef)
                                 -> ValueRef;
        #[fast_ffi]
        pub fn LLVMConstIntToPtr(ConstantVal: ValueRef, ToType: TypeRef)
                                 -> ValueRef;
        #[fast_ffi]
        pub fn LLVMConstBitCast(ConstantVal: ValueRef, ToType: TypeRef)
                                -> ValueRef;
        #[fast_ffi]
        pub fn LLVMConstZExtOrBitCast(ConstantVal: ValueRef, ToType: TypeRef)
                                      -> ValueRef;
        #[fast_ffi]
        pub fn LLVMConstSExtOrBitCast(ConstantVal: ValueRef, ToType: TypeRef)
                                      -> ValueRef;
        #[fast_ffi]
        pub fn LLVMConstTruncOrBitCast(ConstantVal: ValueRef, ToType: TypeRef)
                                       -> ValueRef;
        #[fast_ffi]
        pub fn LLVMConstPointerCast(ConstantVal: ValueRef, ToType: TypeRef)
                                    -> ValueRef;
        #[fast_ffi]
        pub fn LLVMConstIntCast(ConstantVal: ValueRef,
                                ToType: TypeRef,
                                isSigned: Bool)
                                -> ValueRef;
        #[fast_ffi]
        pub fn LLVMConstFPCast(ConstantVal: ValueRef, ToType: TypeRef)
                               -> ValueRef;
        #[fast_ffi]
        pub fn LLVMConstSelect(ConstantCondition: ValueRef,
                               ConstantIfTrue: ValueRef,
                               ConstantIfFalse: ValueRef)
                               -> ValueRef;
        #[fast_ffi]
        pub fn LLVMConstExtractElement(VectorConstant: ValueRef,
                                       IndexConstant: ValueRef)
                                       -> ValueRef;
        #[fast_ffi]
        pub fn LLVMConstInsertElement(VectorConstant: ValueRef,
                                      ElementValueConstant: ValueRef,
                                      IndexConstant: ValueRef)
                                      -> ValueRef;
        #[fast_ffi]
        pub fn LLVMConstShuffleVector(VectorAConstant: ValueRef,
                                      VectorBConstant: ValueRef,
                                      MaskConstant: ValueRef)
                                      -> ValueRef;
        #[fast_ffi]
        pub fn LLVMConstExtractValue(AggConstant: ValueRef,
                                     IdxList: *c_uint,
                                     NumIdx: c_uint)
                                     -> ValueRef;
        #[fast_ffi]
        pub fn LLVMConstInsertValue(AggConstant: ValueRef,
                                    ElementValueConstant: ValueRef,
                                    IdxList: *c_uint,
                                    NumIdx: c_uint)
                                    -> ValueRef;
        #[fast_ffi]
        pub fn LLVMConstInlineAsm(Ty: TypeRef,
                                  AsmString: *c_char,
                                  Constraints: *c_char,
                                  HasSideEffects: Bool,
                                  IsAlignStack: Bool)
                                  -> ValueRef;
        #[fast_ffi]
        pub fn LLVMBlockAddress(F: ValueRef, BB: BasicBlockRef) -> ValueRef;



        /* Operations on global variables, functions, and aliases (globals) */
        #[fast_ffi]
        pub fn LLVMGetGlobalParent(Global: ValueRef) -> ModuleRef;
        #[fast_ffi]
        pub fn LLVMIsDeclaration(Global: ValueRef) -> Bool;
        #[fast_ffi]
        pub fn LLVMGetLinkage(Global: ValueRef) -> c_uint;
        #[fast_ffi]
        pub fn LLVMSetLinkage(Global: ValueRef, Link: c_uint);
        #[fast_ffi]
        pub fn LLVMGetSection(Global: ValueRef) -> *c_char;
        #[fast_ffi]
        pub fn LLVMSetSection(Global: ValueRef, Section: *c_char);
        #[fast_ffi]
        pub fn LLVMGetVisibility(Global: ValueRef) -> c_uint;
        #[fast_ffi]
        pub fn LLVMSetVisibility(Global: ValueRef, Viz: c_uint);
        #[fast_ffi]
        pub fn LLVMGetAlignment(Global: ValueRef) -> c_uint;
        #[fast_ffi]
        pub fn LLVMSetAlignment(Global: ValueRef, Bytes: c_uint);


        /* Operations on global variables */
        #[fast_ffi]
        pub fn LLVMAddGlobal(M: ModuleRef, Ty: TypeRef, Name: *c_char)
                             -> ValueRef;
        #[fast_ffi]
        pub fn LLVMAddGlobalInAddressSpace(M: ModuleRef,
                                           Ty: TypeRef,
                                           Name: *c_char,
                                           AddressSpace: c_uint)
                                           -> ValueRef;
        #[fast_ffi]
        pub fn LLVMGetNamedGlobal(M: ModuleRef, Name: *c_char) -> ValueRef;
        #[fast_ffi]
        pub fn LLVMGetFirstGlobal(M: ModuleRef) -> ValueRef;
        #[fast_ffi]
        pub fn LLVMGetLastGlobal(M: ModuleRef) -> ValueRef;
        #[fast_ffi]
        pub fn LLVMGetNextGlobal(GlobalVar: ValueRef) -> ValueRef;
        #[fast_ffi]
        pub fn LLVMGetPreviousGlobal(GlobalVar: ValueRef) -> ValueRef;
        #[fast_ffi]
        pub fn LLVMDeleteGlobal(GlobalVar: ValueRef);
        #[fast_ffi]
        pub fn LLVMGetInitializer(GlobalVar: ValueRef) -> ValueRef;
        #[fast_ffi]
        pub fn LLVMSetInitializer(GlobalVar: ValueRef,
                                         ConstantVal: ValueRef);
        #[fast_ffi]
        pub fn LLVMIsThreadLocal(GlobalVar: ValueRef) -> Bool;
        #[fast_ffi]
        pub fn LLVMSetThreadLocal(GlobalVar: ValueRef, IsThreadLocal: Bool);
        #[fast_ffi]
        pub fn LLVMIsGlobalConstant(GlobalVar: ValueRef) -> Bool;
        #[fast_ffi]
        pub fn LLVMSetGlobalConstant(GlobalVar: ValueRef, IsConstant: Bool);

        /* Operations on aliases */
        #[fast_ffi]
        pub fn LLVMAddAlias(M: ModuleRef,
                            Ty: TypeRef,
                            Aliasee: ValueRef,
                            Name: *c_char)
                            -> ValueRef;

        /* Operations on functions */
        #[fast_ffi]
        pub fn LLVMAddFunction(M: ModuleRef,
                               Name: *c_char,
                               FunctionTy: TypeRef)
                               -> ValueRef;
        #[fast_ffi]
        pub fn LLVMGetNamedFunction(M: ModuleRef, Name: *c_char) -> ValueRef;
        #[fast_ffi]
        pub fn LLVMGetFirstFunction(M: ModuleRef) -> ValueRef;
        #[fast_ffi]
        pub fn LLVMGetLastFunction(M: ModuleRef) -> ValueRef;
        #[fast_ffi]
        pub fn LLVMGetNextFunction(Fn: ValueRef) -> ValueRef;
        #[fast_ffi]
        pub fn LLVMGetPreviousFunction(Fn: ValueRef) -> ValueRef;
        #[fast_ffi]
        pub fn LLVMDeleteFunction(Fn: ValueRef);
        #[fast_ffi]
        pub fn LLVMGetOrInsertFunction(M: ModuleRef,
                                       Name: *c_char,
                                       FunctionTy: TypeRef)
                                       -> ValueRef;
        #[fast_ffi]
        pub fn LLVMGetIntrinsicID(Fn: ValueRef) -> c_uint;
        #[fast_ffi]
        pub fn LLVMGetFunctionCallConv(Fn: ValueRef) -> c_uint;
        #[fast_ffi]
        pub fn LLVMSetFunctionCallConv(Fn: ValueRef, CC: c_uint);
        #[fast_ffi]
        pub fn LLVMGetGC(Fn: ValueRef) -> *c_char;
        #[fast_ffi]
        pub fn LLVMSetGC(Fn: ValueRef, Name: *c_char);
        #[fast_ffi]
        pub fn LLVMAddFunctionAttr(Fn: ValueRef, PA: c_uint, HighPA: c_uint);
        #[fast_ffi]
        pub fn LLVMGetFunctionAttr(Fn: ValueRef) -> c_ulonglong;
        #[fast_ffi]
        pub fn LLVMRemoveFunctionAttr(Fn: ValueRef,
                                      PA: c_ulonglong,
                                      HighPA: c_ulonglong);

        /* Operations on parameters */
        #[fast_ffi]
        pub fn LLVMCountParams(Fn: ValueRef) -> c_uint;
        #[fast_ffi]
        pub fn LLVMGetParams(Fn: ValueRef, Params: *ValueRef);
        #[fast_ffi]
        pub fn LLVMGetParam(Fn: ValueRef, Index: c_uint) -> ValueRef;
        #[fast_ffi]
        pub fn LLVMGetParamParent(Inst: ValueRef) -> ValueRef;
        #[fast_ffi]
        pub fn LLVMGetFirstParam(Fn: ValueRef) -> ValueRef;
        #[fast_ffi]
        pub fn LLVMGetLastParam(Fn: ValueRef) -> ValueRef;
        #[fast_ffi]
        pub fn LLVMGetNextParam(Arg: ValueRef) -> ValueRef;
        #[fast_ffi]
        pub fn LLVMGetPreviousParam(Arg: ValueRef) -> ValueRef;
        #[fast_ffi]
        pub fn LLVMAddAttribute(Arg: ValueRef, PA: c_uint);
        #[fast_ffi]
        pub fn LLVMRemoveAttribute(Arg: ValueRef, PA: c_uint);
        #[fast_ffi]
        pub fn LLVMGetAttribute(Arg: ValueRef) -> c_uint;
        #[fast_ffi]
        pub fn LLVMSetParamAlignment(Arg: ValueRef, align: c_uint);

        /* Operations on basic blocks */
        #[fast_ffi]
        pub fn LLVMBasicBlockAsValue(BB: BasicBlockRef) -> ValueRef;
        #[fast_ffi]
        pub fn LLVMValueIsBasicBlock(Val: ValueRef) -> Bool;
        #[fast_ffi]
        pub fn LLVMValueAsBasicBlock(Val: ValueRef) -> BasicBlockRef;
        #[fast_ffi]
        pub fn LLVMGetBasicBlockParent(BB: BasicBlockRef) -> ValueRef;
        #[fast_ffi]
        pub fn LLVMCountBasicBlocks(Fn: ValueRef) -> c_uint;
        #[fast_ffi]
        pub fn LLVMGetBasicBlocks(Fn: ValueRef, BasicBlocks: *ValueRef);
        #[fast_ffi]
        pub fn LLVMGetFirstBasicBlock(Fn: ValueRef) -> BasicBlockRef;
        #[fast_ffi]
        pub fn LLVMGetLastBasicBlock(Fn: ValueRef) -> BasicBlockRef;
        #[fast_ffi]
        pub fn LLVMGetNextBasicBlock(BB: BasicBlockRef) -> BasicBlockRef;
        #[fast_ffi]
        pub fn LLVMGetPreviousBasicBlock(BB: BasicBlockRef) -> BasicBlockRef;
        #[fast_ffi]
        pub fn LLVMGetEntryBasicBlock(Fn: ValueRef) -> BasicBlockRef;

        #[fast_ffi]
        pub fn LLVMAppendBasicBlockInContext(C: ContextRef,
                                             Fn: ValueRef,
                                             Name: *c_char)
                                             -> BasicBlockRef;
        #[fast_ffi]
        pub fn LLVMInsertBasicBlockInContext(C: ContextRef,
                                             BB: BasicBlockRef,
                                             Name: *c_char)
                                             -> BasicBlockRef;
        #[fast_ffi]
        pub fn LLVMDeleteBasicBlock(BB: BasicBlockRef);

        #[fast_ffi]
        pub fn LLVMMoveBasicBlockAfter(BB: BasicBlockRef,
                                       MoveAfter: BasicBlockRef);

        #[fast_ffi]
        pub fn LLVMMoveBasicBlockBefore(BB: BasicBlockRef,
                                        MoveBefore: BasicBlockRef);

        /* Operations on instructions */
        #[fast_ffi]
        pub fn LLVMGetInstructionParent(Inst: ValueRef) -> BasicBlockRef;
        #[fast_ffi]
        pub fn LLVMGetFirstInstruction(BB: BasicBlockRef) -> ValueRef;
        #[fast_ffi]
        pub fn LLVMGetLastInstruction(BB: BasicBlockRef) -> ValueRef;
        #[fast_ffi]
        pub fn LLVMGetNextInstruction(Inst: ValueRef) -> ValueRef;
        #[fast_ffi]
        pub fn LLVMGetPreviousInstruction(Inst: ValueRef) -> ValueRef;
        #[fast_ffi]
        pub fn LLVMInstructionEraseFromParent(Inst: ValueRef);

        /* Operations on call sites */
        #[fast_ffi]
        pub fn LLVMSetInstructionCallConv(Instr: ValueRef, CC: c_uint);
        #[fast_ffi]
        pub fn LLVMGetInstructionCallConv(Instr: ValueRef) -> c_uint;
        #[fast_ffi]
        pub fn LLVMAddInstrAttribute(Instr: ValueRef,
                                     index: c_uint,
                                     IA: c_uint);
        #[fast_ffi]
        pub fn LLVMRemoveInstrAttribute(Instr: ValueRef,
                                        index: c_uint,
                                        IA: c_uint);
        #[fast_ffi]
        pub fn LLVMSetInstrParamAlignment(Instr: ValueRef,
                                          index: c_uint,
                                          align: c_uint);

        /* Operations on call instructions (only) */
        #[fast_ffi]
        pub fn LLVMIsTailCall(CallInst: ValueRef) -> Bool;
        #[fast_ffi]
        pub fn LLVMSetTailCall(CallInst: ValueRef, IsTailCall: Bool);

        /* Operations on phi nodes */
        #[fast_ffi]
        pub fn LLVMAddIncoming(PhiNode: ValueRef,
                               IncomingValues: *ValueRef,
                               IncomingBlocks: *BasicBlockRef,
                               Count: c_uint);
        #[fast_ffi]
        pub fn LLVMCountIncoming(PhiNode: ValueRef) -> c_uint;
        #[fast_ffi]
        pub fn LLVMGetIncomingValue(PhiNode: ValueRef, Index: c_uint)
                                    -> ValueRef;
        #[fast_ffi]
        pub fn LLVMGetIncomingBlock(PhiNode: ValueRef, Index: c_uint)
                                    -> BasicBlockRef;

        /* Instruction builders */
        #[fast_ffi]
        pub fn LLVMCreateBuilderInContext(C: ContextRef) -> BuilderRef;
        #[fast_ffi]
        pub fn LLVMPositionBuilder(Builder: BuilderRef,
                                   Block: BasicBlockRef,
                                   Instr: ValueRef);
        #[fast_ffi]
        pub fn LLVMPositionBuilderBefore(Builder: BuilderRef,
                                         Instr: ValueRef);
        #[fast_ffi]
        pub fn LLVMPositionBuilderAtEnd(Builder: BuilderRef,
                                        Block: BasicBlockRef);
        #[fast_ffi]
        pub fn LLVMGetInsertBlock(Builder: BuilderRef) -> BasicBlockRef;
        #[fast_ffi]
        pub fn LLVMClearInsertionPosition(Builder: BuilderRef);
        #[fast_ffi]
        pub fn LLVMInsertIntoBuilder(Builder: BuilderRef, Instr: ValueRef);
        #[fast_ffi]
        pub fn LLVMInsertIntoBuilderWithName(Builder: BuilderRef,
                                             Instr: ValueRef,
                                             Name: *c_char);
        #[fast_ffi]
        pub fn LLVMDisposeBuilder(Builder: BuilderRef);
        #[fast_ffi]
        pub fn LLVMDisposeExecutionEngine(EE: ExecutionEngineRef);

        /* Metadata */
        #[fast_ffi]
        pub fn LLVMSetCurrentDebugLocation(Builder: BuilderRef, L: ValueRef);
        #[fast_ffi]
        pub fn LLVMGetCurrentDebugLocation(Builder: BuilderRef) -> ValueRef;
        #[fast_ffi]
        pub fn LLVMSetInstDebugLocation(Builder: BuilderRef, Inst: ValueRef);

        /* Terminators */
        #[fast_ffi]
        pub fn LLVMBuildRetVoid(B: BuilderRef) -> ValueRef;
        #[fast_ffi]
        pub fn LLVMBuildRet(B: BuilderRef, V: ValueRef) -> ValueRef;
        #[fast_ffi]
        pub fn LLVMBuildAggregateRet(B: BuilderRef,
                                     RetVals: *ValueRef,
                                     N: c_uint)
                                     -> ValueRef;
        #[fast_ffi]
        pub fn LLVMBuildBr(B: BuilderRef, Dest: BasicBlockRef) -> ValueRef;
        #[fast_ffi]
        pub fn LLVMBuildCondBr(B: BuilderRef,
                               If: ValueRef,
                               Then: BasicBlockRef,
                               Else: BasicBlockRef)
                               -> ValueRef;
        #[fast_ffi]
        pub fn LLVMBuildSwitch(B: BuilderRef,
                               V: ValueRef,
                               Else: BasicBlockRef,
                               NumCases: c_uint)
                               -> ValueRef;
        #[fast_ffi]
        pub fn LLVMBuildIndirectBr(B: BuilderRef,
                                   Addr: ValueRef,
                                   NumDests: c_uint)
                                   -> ValueRef;
        #[fast_ffi]
        pub fn LLVMBuildInvoke(B: BuilderRef,
                               Fn: ValueRef,
                               Args: *ValueRef,
                               NumArgs: c_uint,
                               Then: BasicBlockRef,
                               Catch: BasicBlockRef,
                               Name: *c_char)
                               -> ValueRef;
        #[fast_ffi]
        pub fn LLVMBuildLandingPad(B: BuilderRef,
                                   Ty: TypeRef,
                                   PersFn: ValueRef,
                                   NumClauses: c_uint,
                                   Name: *c_char)
                                   -> ValueRef;
        #[fast_ffi]
        pub fn LLVMBuildResume(B: BuilderRef, Exn: ValueRef) -> ValueRef;
        #[fast_ffi]
        pub fn LLVMBuildUnreachable(B: BuilderRef) -> ValueRef;

        /* Add a case to the switch instruction */
        #[fast_ffi]
        pub fn LLVMAddCase(Switch: ValueRef,
                           OnVal: ValueRef,
                           Dest: BasicBlockRef);

        /* Add a destination to the indirectbr instruction */
        #[fast_ffi]
        pub fn LLVMAddDestination(IndirectBr: ValueRef, Dest: BasicBlockRef);

        /* Add a clause to the landing pad instruction */
        #[fast_ffi]
        pub fn LLVMAddClause(LandingPad: ValueRef, ClauseVal: ValueRef);

        /* Set the cleanup on a landing pad instruction */
        #[fast_ffi]
        pub fn LLVMSetCleanup(LandingPad: ValueRef, Val: Bool);

        /* Arithmetic */
        #[fast_ffi]
        pub fn LLVMBuildAdd(B: BuilderRef,
                            LHS: ValueRef,
                            RHS: ValueRef,
                            Name: *c_char)
                            -> ValueRef;
        #[fast_ffi]
        pub fn LLVMBuildNSWAdd(B: BuilderRef,
                               LHS: ValueRef,
                               RHS: ValueRef,
                               Name: *c_char)
                               -> ValueRef;
        #[fast_ffi]
        pub fn LLVMBuildNUWAdd(B: BuilderRef,
                               LHS: ValueRef,
                               RHS: ValueRef,
                               Name: *c_char)
                               -> ValueRef;
        #[fast_ffi]
        pub fn LLVMBuildFAdd(B: BuilderRef,
                             LHS: ValueRef,
                             RHS: ValueRef,
                             Name: *c_char)
                             -> ValueRef;
        #[fast_ffi]
        pub fn LLVMBuildSub(B: BuilderRef,
                            LHS: ValueRef,
                            RHS: ValueRef,
                            Name: *c_char)
                            -> ValueRef;
        #[fast_ffi]
        pub fn LLVMBuildNSWSub(B: BuilderRef,
                               LHS: ValueRef,
                               RHS: ValueRef,
                               Name: *c_char)
                               -> ValueRef;
        #[fast_ffi]
        pub fn LLVMBuildNUWSub(B: BuilderRef,
                               LHS: ValueRef,
                               RHS: ValueRef,
                               Name: *c_char)
                               -> ValueRef;
        #[fast_ffi]
        pub fn LLVMBuildFSub(B: BuilderRef,
                             LHS: ValueRef,
                             RHS: ValueRef,
                             Name: *c_char)
                             -> ValueRef;
        #[fast_ffi]
        pub fn LLVMBuildMul(B: BuilderRef,
                            LHS: ValueRef,
                            RHS: ValueRef,
                            Name: *c_char)
                            -> ValueRef;
        #[fast_ffi]
        pub fn LLVMBuildNSWMul(B: BuilderRef,
                               LHS: ValueRef,
                               RHS: ValueRef,
                               Name: *c_char)
                               -> ValueRef;
        #[fast_ffi]
        pub fn LLVMBuildNUWMul(B: BuilderRef,
                               LHS: ValueRef,
                               RHS: ValueRef,
                               Name: *c_char)
                               -> ValueRef;
        #[fast_ffi]
        pub fn LLVMBuildFMul(B: BuilderRef,
                             LHS: ValueRef,
                             RHS: ValueRef,
                             Name: *c_char)
                             -> ValueRef;
        #[fast_ffi]
        pub fn LLVMBuildUDiv(B: BuilderRef,
                             LHS: ValueRef,
                             RHS: ValueRef,
                             Name: *c_char)
                             -> ValueRef;
        #[fast_ffi]
        pub fn LLVMBuildSDiv(B: BuilderRef,
                             LHS: ValueRef,
                             RHS: ValueRef,
                             Name: *c_char)
                             -> ValueRef;
        #[fast_ffi]
        pub fn LLVMBuildExactSDiv(B: BuilderRef,
                                  LHS: ValueRef,
                                  RHS: ValueRef,
                                  Name: *c_char)
                                  -> ValueRef;
        #[fast_ffi]
        pub fn LLVMBuildFDiv(B: BuilderRef,
                             LHS: ValueRef,
                             RHS: ValueRef,
                             Name: *c_char)
                             -> ValueRef;
        #[fast_ffi]
        pub fn LLVMBuildURem(B: BuilderRef,
                             LHS: ValueRef,
                             RHS: ValueRef,
                             Name: *c_char)
                             -> ValueRef;
        #[fast_ffi]
        pub fn LLVMBuildSRem(B: BuilderRef,
                             LHS: ValueRef,
                             RHS: ValueRef,
                             Name: *c_char)
                             -> ValueRef;
        #[fast_ffi]
        pub fn LLVMBuildFRem(B: BuilderRef,
                             LHS: ValueRef,
                             RHS: ValueRef,
                             Name: *c_char)
                             -> ValueRef;
        #[fast_ffi]
        pub fn LLVMBuildShl(B: BuilderRef,
                            LHS: ValueRef,
                            RHS: ValueRef,
                            Name: *c_char)
                            -> ValueRef;
        #[fast_ffi]
        pub fn LLVMBuildLShr(B: BuilderRef,
                             LHS: ValueRef,
                             RHS: ValueRef,
                             Name: *c_char)
                             -> ValueRef;
        #[fast_ffi]
        pub fn LLVMBuildAShr(B: BuilderRef,
                             LHS: ValueRef,
                             RHS: ValueRef,
                             Name: *c_char)
                             -> ValueRef;
        #[fast_ffi]
        pub fn LLVMBuildAnd(B: BuilderRef,
                            LHS: ValueRef,
                            RHS: ValueRef,
                            Name: *c_char)
                            -> ValueRef;
        #[fast_ffi]
        pub fn LLVMBuildOr(B: BuilderRef,
                           LHS: ValueRef,
                           RHS: ValueRef,
                           Name: *c_char)
                           -> ValueRef;
        #[fast_ffi]
        pub fn LLVMBuildXor(B: BuilderRef,
                            LHS: ValueRef,
                            RHS: ValueRef,
                            Name: *c_char)
                            -> ValueRef;
        #[fast_ffi]
        pub fn LLVMBuildBinOp(B: BuilderRef,
                              Op: Opcode,
                              LHS: ValueRef,
                              RHS: ValueRef,
                              Name: *c_char)
                              -> ValueRef;
        #[fast_ffi]
        pub fn LLVMBuildNeg(B: BuilderRef, V: ValueRef, Name: *c_char)
                            -> ValueRef;
        #[fast_ffi]
        pub fn LLVMBuildNSWNeg(B: BuilderRef, V: ValueRef, Name: *c_char)
                               -> ValueRef;
        #[fast_ffi]
        pub fn LLVMBuildNUWNeg(B: BuilderRef, V: ValueRef, Name: *c_char)
                               -> ValueRef;
        #[fast_ffi]
        pub fn LLVMBuildFNeg(B: BuilderRef, V: ValueRef, Name: *c_char)
                             -> ValueRef;
        #[fast_ffi]
        pub fn LLVMBuildNot(B: BuilderRef, V: ValueRef, Name: *c_char)
                            -> ValueRef;

        /* Memory */
        #[fast_ffi]
        pub fn LLVMBuildMalloc(B: BuilderRef, Ty: TypeRef, Name: *c_char)
                               -> ValueRef;
        #[fast_ffi]
        pub fn LLVMBuildArrayMalloc(B: BuilderRef,
                                    Ty: TypeRef,
                                    Val: ValueRef,
                                    Name: *c_char)
                                    -> ValueRef;
        #[fast_ffi]
        pub fn LLVMBuildAlloca(B: BuilderRef, Ty: TypeRef, Name: *c_char)
                               -> ValueRef;
        #[fast_ffi]
        pub fn LLVMBuildArrayAlloca(B: BuilderRef,
                                    Ty: TypeRef,
                                    Val: ValueRef,
                                    Name: *c_char)
                                    -> ValueRef;
        #[fast_ffi]
        pub fn LLVMBuildFree(B: BuilderRef, PointerVal: ValueRef) -> ValueRef;
        #[fast_ffi]
        pub fn LLVMBuildLoad(B: BuilderRef,
                             PointerVal: ValueRef,
                             Name: *c_char)
                             -> ValueRef;

        #[fast_ffi]
        pub fn LLVMBuildStore(B: BuilderRef, Val: ValueRef, Ptr: ValueRef)
                              -> ValueRef;

        #[fast_ffi]
        pub fn LLVMBuildGEP(B: BuilderRef,
                            Pointer: ValueRef,
                            Indices: *ValueRef,
                            NumIndices: c_uint,
                            Name: *c_char)
                            -> ValueRef;
        #[fast_ffi]
        pub fn LLVMBuildInBoundsGEP(B: BuilderRef,
                                    Pointer: ValueRef,
                                    Indices: *ValueRef,
                                    NumIndices: c_uint,
                                    Name: *c_char)
                                    -> ValueRef;
        #[fast_ffi]
        pub fn LLVMBuildStructGEP(B: BuilderRef,
                                  Pointer: ValueRef,
                                  Idx: c_uint,
                                  Name: *c_char)
                                  -> ValueRef;
        #[fast_ffi]
        pub fn LLVMBuildGlobalString(B: BuilderRef,
                                     Str: *c_char,
                                     Name: *c_char)
                                     -> ValueRef;
        #[fast_ffi]
        pub fn LLVMBuildGlobalStringPtr(B: BuilderRef,
                                        Str: *c_char,
                                        Name: *c_char)
                                        -> ValueRef;

        /* Casts */
        #[fast_ffi]
        pub fn LLVMBuildTrunc(B: BuilderRef,
                              Val: ValueRef,
                              DestTy: TypeRef,
                              Name: *c_char)
                              -> ValueRef;
        #[fast_ffi]
        pub fn LLVMBuildZExt(B: BuilderRef,
                             Val: ValueRef,
                             DestTy: TypeRef,
                             Name: *c_char)
                             -> ValueRef;
        #[fast_ffi]
        pub fn LLVMBuildSExt(B: BuilderRef,
                             Val: ValueRef,
                             DestTy: TypeRef,
                             Name: *c_char)
                             -> ValueRef;
        #[fast_ffi]
        pub fn LLVMBuildFPToUI(B: BuilderRef,
                               Val: ValueRef,
                               DestTy: TypeRef,
                               Name: *c_char)
                               -> ValueRef;
        #[fast_ffi]
        pub fn LLVMBuildFPToSI(B: BuilderRef,
                               Val: ValueRef,
                               DestTy: TypeRef,
                               Name: *c_char)
                               -> ValueRef;
        #[fast_ffi]
        pub fn LLVMBuildUIToFP(B: BuilderRef,
                               Val: ValueRef,
                               DestTy: TypeRef,
                               Name: *c_char)
                               -> ValueRef;
        #[fast_ffi]
        pub fn LLVMBuildSIToFP(B: BuilderRef,
                               Val: ValueRef,
                               DestTy: TypeRef,
                               Name: *c_char)
                               -> ValueRef;
        #[fast_ffi]
        pub fn LLVMBuildFPTrunc(B: BuilderRef,
                                Val: ValueRef,
                                DestTy: TypeRef,
                                Name: *c_char)
                                -> ValueRef;
        #[fast_ffi]
        pub fn LLVMBuildFPExt(B: BuilderRef,
                              Val: ValueRef,
                              DestTy: TypeRef,
                              Name: *c_char)
                              -> ValueRef;
        #[fast_ffi]
        pub fn LLVMBuildPtrToInt(B: BuilderRef,
                                 Val: ValueRef,
                                 DestTy: TypeRef,
                                 Name: *c_char)
                                 -> ValueRef;
        #[fast_ffi]
        pub fn LLVMBuildIntToPtr(B: BuilderRef,
                                 Val: ValueRef,
                                 DestTy: TypeRef,
                                 Name: *c_char)
                                 -> ValueRef;
        #[fast_ffi]
        pub fn LLVMBuildBitCast(B: BuilderRef,
                                Val: ValueRef,
                                DestTy: TypeRef,
                                Name: *c_char)
                                -> ValueRef;
        #[fast_ffi]
        pub fn LLVMBuildZExtOrBitCast(B: BuilderRef,
                                      Val: ValueRef,
                                      DestTy: TypeRef,
                                      Name: *c_char)
                                      -> ValueRef;
        #[fast_ffi]
        pub fn LLVMBuildSExtOrBitCast(B: BuilderRef,
                                      Val: ValueRef,
                                      DestTy: TypeRef,
                                      Name: *c_char)
                                      -> ValueRef;
        #[fast_ffi]
        pub fn LLVMBuildTruncOrBitCast(B: BuilderRef,
                                       Val: ValueRef,
                                       DestTy: TypeRef,
                                       Name: *c_char)
                                       -> ValueRef;
        #[fast_ffi]
        pub fn LLVMBuildCast(B: BuilderRef,
                             Op: Opcode,
                             Val: ValueRef,
                             DestTy: TypeRef,
                             Name: *c_char) -> ValueRef;
        #[fast_ffi]
        pub fn LLVMBuildPointerCast(B: BuilderRef,
                                    Val: ValueRef,
                                    DestTy: TypeRef,
                                    Name: *c_char)
                                    -> ValueRef;
        #[fast_ffi]
        pub fn LLVMBuildIntCast(B: BuilderRef,
                                Val: ValueRef,
                                DestTy: TypeRef,
                                Name: *c_char)
                                -> ValueRef;
        #[fast_ffi]
        pub fn LLVMBuildFPCast(B: BuilderRef,
                               Val: ValueRef,
                               DestTy: TypeRef,
                               Name: *c_char)
                               -> ValueRef;

        /* Comparisons */
        #[fast_ffi]
        pub fn LLVMBuildICmp(B: BuilderRef,
                             Op: c_uint,
                             LHS: ValueRef,
                             RHS: ValueRef,
                             Name: *c_char)
                             -> ValueRef;
        #[fast_ffi]
        pub fn LLVMBuildFCmp(B: BuilderRef,
                             Op: c_uint,
                             LHS: ValueRef,
                             RHS: ValueRef,
                             Name: *c_char)
                             -> ValueRef;

        /* Miscellaneous instructions */
        #[fast_ffi]
        pub fn LLVMBuildPhi(B: BuilderRef, Ty: TypeRef, Name: *c_char)
                            -> ValueRef;
        #[fast_ffi]
        pub fn LLVMBuildCall(B: BuilderRef,
                             Fn: ValueRef,
                             Args: *ValueRef,
                             NumArgs: c_uint,
                             Name: *c_char)
                             -> ValueRef;
        #[fast_ffi]
        pub fn LLVMBuildSelect(B: BuilderRef,
                               If: ValueRef,
                               Then: ValueRef,
                               Else: ValueRef,
                               Name: *c_char)
                               -> ValueRef;
        #[fast_ffi]
        pub fn LLVMBuildVAArg(B: BuilderRef,
                              list: ValueRef,
                              Ty: TypeRef,
                              Name: *c_char)
                              -> ValueRef;
        #[fast_ffi]
        pub fn LLVMBuildExtractElement(B: BuilderRef,
                                       VecVal: ValueRef,
                                       Index: ValueRef,
                                       Name: *c_char)
                                       -> ValueRef;
        #[fast_ffi]
        pub fn LLVMBuildInsertElement(B: BuilderRef,
                                      VecVal: ValueRef,
                                      EltVal: ValueRef,
                                      Index: ValueRef,
                                      Name: *c_char)
                                      -> ValueRef;
        #[fast_ffi]
        pub fn LLVMBuildShuffleVector(B: BuilderRef,
                                      V1: ValueRef,
                                      V2: ValueRef,
                                      Mask: ValueRef,
                                      Name: *c_char)
                                      -> ValueRef;
        #[fast_ffi]
        pub fn LLVMBuildExtractValue(B: BuilderRef,
                                     AggVal: ValueRef,
                                     Index: c_uint,
                                     Name: *c_char)
                                     -> ValueRef;
        #[fast_ffi]
        pub fn LLVMBuildInsertValue(B: BuilderRef,
                                    AggVal: ValueRef,
                                    EltVal: ValueRef,
                                    Index: c_uint,
                                    Name: *c_char)
                                    -> ValueRef;

        #[fast_ffi]
        pub fn LLVMBuildIsNull(B: BuilderRef, Val: ValueRef, Name: *c_char)
                               -> ValueRef;
        #[fast_ffi]
        pub fn LLVMBuildIsNotNull(B: BuilderRef, Val: ValueRef, Name: *c_char)
                                  -> ValueRef;
        #[fast_ffi]
        pub fn LLVMBuildPtrDiff(B: BuilderRef,
                                LHS: ValueRef,
                                RHS: ValueRef,
                                Name: *c_char)
                                -> ValueRef;

        /* Atomic Operations */
        pub fn LLVMBuildAtomicLoad(B: BuilderRef,
                                   PointerVal: ValueRef,
                                   Name: *c_char,
                                   Order: AtomicOrdering,
                                   Alignment: c_uint)
                                   -> ValueRef;

        pub fn LLVMBuildAtomicStore(B: BuilderRef,
                                    Val: ValueRef,
                                    Ptr: ValueRef,
                                    Order: AtomicOrdering,
                                    Alignment: c_uint)
                                    -> ValueRef;

        pub fn LLVMBuildAtomicCmpXchg(B: BuilderRef,
                                      LHS: ValueRef,
                                      CMP: ValueRef,
                                      RHS: ValueRef,
                                      Order: AtomicOrdering)
                                      -> ValueRef;
        pub fn LLVMBuildAtomicRMW(B: BuilderRef,
                                  Op: AtomicBinOp,
                                  LHS: ValueRef,
                                  RHS: ValueRef,
                                  Order: AtomicOrdering)
                                  -> ValueRef;

        pub fn LLVMBuildAtomicFence(B: BuilderRef, Order: AtomicOrdering);


        /* Selected entries from the downcasts. */
        #[fast_ffi]
        pub fn LLVMIsATerminatorInst(Inst: ValueRef) -> ValueRef;

        /** Writes a module to the specified path. Returns 0 on success. */
        #[fast_ffi]
        pub fn LLVMWriteBitcodeToFile(M: ModuleRef, Path: *c_char) -> c_int;

        /** Creates target data from a target layout string. */
        #[fast_ffi]
        pub fn LLVMCreateTargetData(StringRep: *c_char) -> TargetDataRef;
        /// Adds the target data to the given pass manager. The pass manager
        /// references the target data only weakly.
        #[fast_ffi]
        pub fn LLVMAddTargetData(TD: TargetDataRef, PM: PassManagerRef);
        /** Number of bytes clobbered when doing a Store to *T. */
        #[fast_ffi]
        pub fn LLVMStoreSizeOfType(TD: TargetDataRef, Ty: TypeRef)
                                   -> c_ulonglong;

        /** Number of bytes clobbered when doing a Store to *T. */
        #[fast_ffi]
        pub fn LLVMSizeOfTypeInBits(TD: TargetDataRef, Ty: TypeRef)
                                    -> c_ulonglong;

        /** Distance between successive elements in an array of T.
        Includes ABI padding. */
        #[fast_ffi]
        pub fn LLVMABISizeOfType(TD: TargetDataRef, Ty: TypeRef) -> c_uint;

        /** Returns the preferred alignment of a type. */
        #[fast_ffi]
        pub fn LLVMPreferredAlignmentOfType(TD: TargetDataRef, Ty: TypeRef)
                                            -> c_uint;
        /** Returns the minimum alignment of a type. */
        #[fast_ffi]
        pub fn LLVMABIAlignmentOfType(TD: TargetDataRef, Ty: TypeRef)
                                      -> c_uint;

        /// Computes the byte offset of the indexed struct element for a
        /// target.
        #[fast_ffi]
        pub fn LLVMOffsetOfElement(TD: TargetDataRef,
                                   StructTy: TypeRef,
                                   Element: c_uint)
                                   -> c_ulonglong;

        /**
         * Returns the minimum alignment of a type when part of a call frame.
         */
        #[fast_ffi]
        pub fn LLVMCallFrameAlignmentOfType(TD: TargetDataRef, Ty: TypeRef)
                                            -> c_uint;

        /** Disposes target data. */
        #[fast_ffi]
        pub fn LLVMDisposeTargetData(TD: TargetDataRef);

        /** Creates a pass manager. */
        #[fast_ffi]
        pub fn LLVMCreatePassManager() -> PassManagerRef;
        /** Creates a function-by-function pass manager */
        #[fast_ffi]
        pub fn LLVMCreateFunctionPassManagerForModule(M: ModuleRef)
                                                      -> PassManagerRef;

        /** Disposes a pass manager. */
        #[fast_ffi]
        pub fn LLVMDisposePassManager(PM: PassManagerRef);

        /** Runs a pass manager on a module. */
        #[fast_ffi]
        pub fn LLVMRunPassManager(PM: PassManagerRef, M: ModuleRef) -> Bool;

        /** Runs the function passes on the provided function. */
        #[fast_ffi]
        pub fn LLVMRunFunctionPassManager(FPM: PassManagerRef, F: ValueRef)
                                          -> Bool;

        /** Initializes all the function passes scheduled in the manager */
        #[fast_ffi]
        pub fn LLVMInitializeFunctionPassManager(FPM: PassManagerRef) -> Bool;

        /** Finalizes all the function passes scheduled in the manager */
        #[fast_ffi]
        pub fn LLVMFinalizeFunctionPassManager(FPM: PassManagerRef) -> Bool;

        #[fast_ffi]
        pub fn LLVMInitializePasses();

        #[fast_ffi]
        pub fn LLVMAddPass(PM: PassManagerRef, P: PassRef);

        #[fast_ffi]
        pub fn LLVMCreatePass(PassName: *c_char) -> PassRef;

        #[fast_ffi]
        pub fn LLVMDestroyPass(P: PassRef);

        /** Adds a verification pass. */
        #[fast_ffi]
        pub fn LLVMAddVerifierPass(PM: PassManagerRef);

        #[fast_ffi]
        pub fn LLVMAddGlobalOptimizerPass(PM: PassManagerRef);
        #[fast_ffi]
        pub fn LLVMAddIPSCCPPass(PM: PassManagerRef);
        #[fast_ffi]
        pub fn LLVMAddDeadArgEliminationPass(PM: PassManagerRef);
        #[fast_ffi]
        pub fn LLVMAddInstructionCombiningPass(PM: PassManagerRef);
        #[fast_ffi]
        pub fn LLVMAddCFGSimplificationPass(PM: PassManagerRef);
        #[fast_ffi]
        pub fn LLVMAddFunctionInliningPass(PM: PassManagerRef);
        #[fast_ffi]
        pub fn LLVMAddFunctionAttrsPass(PM: PassManagerRef);
        #[fast_ffi]
        pub fn LLVMAddScalarReplAggregatesPass(PM: PassManagerRef);
        #[fast_ffi]
        pub fn LLVMAddScalarReplAggregatesPassSSA(PM: PassManagerRef);
        #[fast_ffi]
        pub fn LLVMAddJumpThreadingPass(PM: PassManagerRef);
        #[fast_ffi]
        pub fn LLVMAddConstantPropagationPass(PM: PassManagerRef);
        #[fast_ffi]
        pub fn LLVMAddReassociatePass(PM: PassManagerRef);
        #[fast_ffi]
        pub fn LLVMAddLoopRotatePass(PM: PassManagerRef);
        #[fast_ffi]
        pub fn LLVMAddLICMPass(PM: PassManagerRef);
        #[fast_ffi]
        pub fn LLVMAddLoopUnswitchPass(PM: PassManagerRef);
        #[fast_ffi]
        pub fn LLVMAddLoopDeletionPass(PM: PassManagerRef);
        #[fast_ffi]
        pub fn LLVMAddLoopUnrollPass(PM: PassManagerRef);
        #[fast_ffi]
        pub fn LLVMAddGVNPass(PM: PassManagerRef);
        #[fast_ffi]
        pub fn LLVMAddMemCpyOptPass(PM: PassManagerRef);
        #[fast_ffi]
        pub fn LLVMAddSCCPPass(PM: PassManagerRef);
        #[fast_ffi]
        pub fn LLVMAddDeadStoreEliminationPass(PM: PassManagerRef);
        #[fast_ffi]
        pub fn LLVMAddStripDeadPrototypesPass(PM: PassManagerRef);
        #[fast_ffi]
        pub fn LLVMAddConstantMergePass(PM: PassManagerRef);
        #[fast_ffi]
        pub fn LLVMAddArgumentPromotionPass(PM: PassManagerRef);
        #[fast_ffi]
        pub fn LLVMAddTailCallEliminationPass(PM: PassManagerRef);
        #[fast_ffi]
        pub fn LLVMAddIndVarSimplifyPass(PM: PassManagerRef);
        #[fast_ffi]
        pub fn LLVMAddAggressiveDCEPass(PM: PassManagerRef);
        #[fast_ffi]
        pub fn LLVMAddGlobalDCEPass(PM: PassManagerRef);
        #[fast_ffi]
        pub fn LLVMAddCorrelatedValuePropagationPass(PM: PassManagerRef);
        #[fast_ffi]
        pub fn LLVMAddPruneEHPass(PM: PassManagerRef);
        #[fast_ffi]
        pub fn LLVMAddSimplifyLibCallsPass(PM: PassManagerRef);
        #[fast_ffi]
        pub fn LLVMAddLoopIdiomPass(PM: PassManagerRef);
        #[fast_ffi]
        pub fn LLVMAddEarlyCSEPass(PM: PassManagerRef);
        #[fast_ffi]
        pub fn LLVMAddTypeBasedAliasAnalysisPass(PM: PassManagerRef);
        #[fast_ffi]
        pub fn LLVMAddBasicAliasAnalysisPass(PM: PassManagerRef);

        #[fast_ffi]
        pub fn LLVMPassManagerBuilderCreate() -> PassManagerBuilderRef;
        #[fast_ffi]
        pub fn LLVMPassManagerBuilderDispose(PMB: PassManagerBuilderRef);
        #[fast_ffi]
        pub fn LLVMPassManagerBuilderSetOptLevel(PMB: PassManagerBuilderRef,
                                                 OptimizationLevel: c_uint);
        #[fast_ffi]
        pub fn LLVMPassManagerBuilderSetSizeLevel(PMB: PassManagerBuilderRef,
                                                  Value: Bool);
        #[fast_ffi]
        pub fn LLVMPassManagerBuilderSetDisableUnitAtATime(
            PMB: PassManagerBuilderRef,
            Value: Bool);
        #[fast_ffi]
        pub fn LLVMPassManagerBuilderSetDisableUnrollLoops(
            PMB: PassManagerBuilderRef,
            Value: Bool);
        #[fast_ffi]
        pub fn LLVMPassManagerBuilderSetDisableSimplifyLibCalls(
            PMB: PassManagerBuilderRef,
            Value: Bool);
        #[fast_ffi]
        pub fn LLVMPassManagerBuilderUseInlinerWithThreshold(
            PMB: PassManagerBuilderRef,
            threshold: c_uint);
        #[fast_ffi]
        pub fn LLVMPassManagerBuilderPopulateModulePassManager(
            PMB: PassManagerBuilderRef,
            PM: PassManagerRef);

        #[fast_ffi]
        pub fn LLVMPassManagerBuilderPopulateFunctionPassManager(
            PMB: PassManagerBuilderRef,
            PM: PassManagerRef);

        /** Destroys a memory buffer. */
        #[fast_ffi]
        pub fn LLVMDisposeMemoryBuffer(MemBuf: MemoryBufferRef);


        /* Stuff that's in rustllvm/ because it's not upstream yet. */

        /** Opens an object file. */
        #[fast_ffi]
        pub fn LLVMCreateObjectFile(MemBuf: MemoryBufferRef) -> ObjectFileRef;
        /** Closes an object file. */
        #[fast_ffi]
        pub fn LLVMDisposeObjectFile(ObjFile: ObjectFileRef);

        /** Enumerates the sections in an object file. */
        #[fast_ffi]
        pub fn LLVMGetSections(ObjFile: ObjectFileRef) -> SectionIteratorRef;
        /** Destroys a section iterator. */
        #[fast_ffi]
        pub fn LLVMDisposeSectionIterator(SI: SectionIteratorRef);
        /** Returns true if the section iterator is at the end of the section
            list: */
        #[fast_ffi]
        pub fn LLVMIsSectionIteratorAtEnd(ObjFile: ObjectFileRef,
                                          SI: SectionIteratorRef)
                                          -> Bool;
        /** Moves the section iterator to point to the next section. */
        #[fast_ffi]
        pub fn LLVMMoveToNextSection(SI: SectionIteratorRef);
        /** Returns the current section name. */
        #[fast_ffi]
        pub fn LLVMGetSectionName(SI: SectionIteratorRef) -> *c_char;
        /** Returns the current section size. */
        #[fast_ffi]
        pub fn LLVMGetSectionSize(SI: SectionIteratorRef) -> c_ulonglong;
        /** Returns the current section contents as a string buffer. */
        #[fast_ffi]
        pub fn LLVMGetSectionContents(SI: SectionIteratorRef) -> *c_char;

        /** Reads the given file and returns it as a memory buffer. Use
            LLVMDisposeMemoryBuffer() to get rid of it. */
        #[fast_ffi]
        pub fn LLVMRustCreateMemoryBufferWithContentsOfFile(Path: *c_char)
            -> MemoryBufferRef;

        #[fast_ffi]
        pub fn LLVMRustWriteOutputFile(PM: PassManagerRef,
                                       M: ModuleRef,
                                       Triple: *c_char,
                                       Feature: *c_char,
                                       Output: *c_char,
                                       // FIXME: When #2334 is fixed,
                                       // change c_uint to FileType
                                       FileType: c_uint,
                                       OptLevel: c_int,
                                       EnableSegmentedStacks: bool)
                                       -> bool;

        /** Returns a string describing the last error caused by an LLVMRust*
            call. */
        #[fast_ffi]
        pub fn LLVMRustGetLastError() -> *c_char;

        /** Prepare the JIT. Returns a memory manager that can load crates. */
        #[fast_ffi]
        pub fn LLVMRustPrepareJIT(__morestack: *()) -> *();

        /** Load a crate into the memory manager. */
        #[fast_ffi]
        pub fn LLVMRustLoadCrate(MM: *(), Filename: *c_char) -> bool;

        /** Execute the JIT engine. */
        #[fast_ffi]
        pub fn LLVMRustBuildJIT(MM: *(),
                                M: ModuleRef,
                                EnableSegmentedStacks: bool)
                                -> ExecutionEngineRef;

        /** Parses the bitcode in the given memory buffer. */
        #[fast_ffi]
        pub fn LLVMRustParseBitcode(MemBuf: MemoryBufferRef) -> ModuleRef;

        /** Parses LLVM asm in the given file */
        #[fast_ffi]
        pub fn LLVMRustParseAssemblyFile(Filename: *c_char, C: ContextRef)
                                         -> ModuleRef;

        #[fast_ffi]
        pub fn LLVMRustAddPrintModulePass(PM: PassManagerRef,
                                          M: ModuleRef,
                                          Output: *c_char);

        /** Turn on LLVM pass-timing. */
        #[fast_ffi]
        pub fn LLVMRustEnableTimePasses();

        /// Print the pass timings since static dtors aren't picking them up.
        #[fast_ffi]
        pub fn LLVMRustPrintPassTimings();

        #[fast_ffi]
        pub fn LLVMRustStartMultithreading() -> bool;

        #[fast_ffi]
        pub fn LLVMStructCreateNamed(C: ContextRef, Name: *c_char) -> TypeRef;

        #[fast_ffi]
        pub fn LLVMStructSetBody(StructTy: TypeRef,
                                 ElementTypes: *TypeRef,
                                 ElementCount: c_uint,
                                 Packed: Bool);

        #[fast_ffi]
        pub fn LLVMConstNamedStruct(S: TypeRef,
                                    ConstantVals: *ValueRef,
                                    Count: c_uint)
                                    -> ValueRef;

        /** Enables LLVM debug output. */
        #[fast_ffi]
        pub fn LLVMSetDebug(Enabled: c_int);

        /** Prepares inline assembly. */
        #[fast_ffi]
        pub fn LLVMInlineAsm(Ty: TypeRef,
                             AsmString: *c_char,
                             Constraints: *c_char,
                             SideEffects: Bool,
                             AlignStack: Bool,
                             Dialect: c_uint)
                             -> ValueRef;


        #[fast_ffi]
        pub fn LLVMDIBuilderCreate(M: ModuleRef) -> DIBuilderRef;

        #[fast_ffi]
        pub fn LLVMDIBuilderDispose(Builder: DIBuilderRef);

        #[fast_ffi]
        pub fn LLVMDIBuilderFinalize(Builder: DIBuilderRef);

        #[fast_ffi]
        pub fn LLVMDIBuilderCreateCompileUnit(Builder: DIBuilderRef,
                                              Lang: c_uint,
                                              File: *c_char,
                                              Dir: *c_char,
                                              Producer: *c_char,
                                              isOptimized: bool,
                                              Flags: *c_char,
                                              RuntimeVer: c_uint,
                                              SplitName: *c_char);

        #[fast_ffi]
        pub fn LLVMDIBuilderCreateFile(Builder: DIBuilderRef,
                                       Filename: *c_char,
                                       Directory: *c_char)
                                       -> DIFile;

        #[fast_ffi]
        pub fn LLVMDIBuilderCreateSubroutineType(Builder: DIBuilderRef,
                                                 File: DIFile,
                                                 ParameterTypes: DIArray)
                                                 -> DICompositeType;

        #[fast_ffi]
        pub fn LLVMDIBuilderCreateFunction(Builder: DIBuilderRef,
                                           Scope: DIDescriptor,
                                           Name: *c_char,
                                           LinkageName: *c_char,
                                           File: DIFile,
                                           LineNo: c_uint,
                                           Ty: DIType,
                                           isLocalToUnit: bool,
                                           isDefinition: bool,
                                           ScopeLine: c_uint,
                                           Flags: c_uint,
                                           isOptimized: bool,
                                           Fn: ValueRef,
                                           TParam: ValueRef,
                                           Decl: ValueRef)
                                           -> DISubprogram;

        #[fast_ffi]
        pub fn LLVMDIBuilderCreateBasicType(Builder: DIBuilderRef,
                                            Name: *c_char,
                                            SizeInBits: c_ulonglong,
                                            AlignInBits: c_ulonglong,
                                            Encoding: c_uint)
                                            -> DIBasicType;

        #[fast_ffi]
        pub fn LLVMDIBuilderCreatePointerType(Builder: DIBuilderRef,
                                              PointeeTy: DIType,
                                              SizeInBits: c_ulonglong,
                                              AlignInBits: c_ulonglong,
                                              Name: *c_char)
                                              -> DIDerivedType;

        #[fast_ffi]
        pub fn LLVMDIBuilderCreateStructType(Builder: DIBuilderRef,
                                             Scope: DIDescriptor,
                                             Name: *c_char,
                                             File: DIFile,
                                             LineNumber: c_uint,
                                             SizeInBits: c_ulonglong,
                                             AlignInBits: c_ulonglong,
                                             Flags: c_uint,
                                             DerivedFrom: DIType,
                                             Elements: DIArray,
                                             RunTimeLang: c_uint,
                                             VTableHolder: ValueRef)
                                             -> DICompositeType;

        #[fast_ffi]
        pub fn LLVMDIBuilderCreateMemberType(Builder: DIBuilderRef,
                                             Scope: DIDescriptor,
                                             Name: *c_char,
                                             File: DIFile,
                                             LineNo: c_uint,
                                             SizeInBits: c_ulonglong,
                                             AlignInBits: c_ulonglong,
                                             OffsetInBits: c_ulonglong,
                                             Flags: c_uint,
                                             Ty: DIType)
                                             -> DIDerivedType;

        #[fast_ffi]
        pub fn LLVMDIBuilderCreateLexicalBlock(Builder: DIBuilderRef,
                                               Scope: DIDescriptor,
                                               File: DIFile,
                                               Line: c_uint,
                                               Col: c_uint)
                                               -> DILexicalBlock;

        #[fast_ffi]
        pub fn LLVMDIBuilderCreateLocalVariable(Builder: DIBuilderRef,
                                                Tag: c_uint,
                                                Scope: DIDescriptor,
                                                Name: *c_char,
                                                File: DIFile,
                                                LineNo: c_uint,
                                                Ty: DIType,
                                                AlwaysPreserve: bool,
                                                Flags: c_uint,
                                                ArgNo: c_uint)
                                                -> DIVariable;

        #[fast_ffi]
        pub fn LLVMDIBuilderCreateArrayType(Builder: DIBuilderRef,
                                            Size: c_ulonglong,
                                            AlignInBits: c_ulonglong,
                                            Ty: DIType,
                                            Subscripts: DIArray)
                                            -> DIType;

        #[fast_ffi]
        pub fn LLVMDIBuilderCreateVectorType(Builder: DIBuilderRef,
                                             Size: c_ulonglong,
                                             AlignInBits: c_ulonglong,
                                             Ty: DIType,
                                             Subscripts: DIArray)
                                             -> DIType;

        #[fast_ffi]
        pub fn LLVMDIBuilderGetOrCreateSubrange(Builder: DIBuilderRef,
                                                Lo: c_longlong,
                                                Count: c_longlong)
                                                -> DISubrange;

        #[fast_ffi]
        pub fn LLVMDIBuilderGetOrCreateArray(Builder: DIBuilderRef,
                                             Ptr: *DIDescriptor,
                                             Count: c_uint)
                                             -> DIArray;

        #[fast_ffi]
        pub fn LLVMDIBuilderInsertDeclareAtEnd(Builder: DIBuilderRef,
                                               Val: ValueRef,
                                               VarInfo: DIVariable,
                                               InsertAtEnd: BasicBlockRef)
                                               -> ValueRef;

        #[fast_ffi]
        pub fn LLVMDIBuilderInsertDeclareBefore(Builder: DIBuilderRef,
                                                Val: ValueRef,
                                                VarInfo: DIVariable,
                                                InsertBefore: ValueRef)
                                                -> ValueRef;

        #[fast_ffi]
        pub fn LLVMDIBuilderCreateEnumerator(Builder: DIBuilderRef,
                                             Name: *c_char,
                                             Val: c_ulonglong)
                                             -> ValueRef;

        #[fast_ffi]
        pub fn LLVMDIBuilderCreateEnumerationType(Builder: DIBuilderRef,
                                                  Scope: ValueRef,
                                                  Name: *c_char,
                                                  File: ValueRef,
                                                  LineNumber: c_uint,
                                                  SizeInBits: c_ulonglong,
                                                  AlignInBits: c_ulonglong,
                                                  Elements: ValueRef,
                                                  ClassType: ValueRef)
                                                  -> ValueRef;

        #[fast_ffi]
        pub fn LLVMDIBuilderCreateUnionType(Builder: DIBuilderRef,
                                            Scope: ValueRef,
                                            Name: *c_char,
                                            File: ValueRef,
                                            LineNumber: c_uint,
                                            SizeInBits: c_ulonglong,
                                            AlignInBits: c_ulonglong,
                                            Flags: c_uint,
                                            Elements: ValueRef,
                                            RunTimeLang: c_uint)
                                            -> ValueRef;
    }
}

pub fn SetInstructionCallConv(Instr: ValueRef, CC: CallConv) {
    unsafe {
        llvm::LLVMSetInstructionCallConv(Instr, CC as c_uint);
    }
}
pub fn SetFunctionCallConv(Fn: ValueRef, CC: CallConv) {
    unsafe {
        llvm::LLVMSetFunctionCallConv(Fn, CC as c_uint);
    }
}
pub fn SetLinkage(Global: ValueRef, Link: Linkage) {
    unsafe {
        llvm::LLVMSetLinkage(Global, Link as c_uint);
    }
}

pub fn ConstICmp(Pred: IntPredicate, V1: ValueRef, V2: ValueRef) -> ValueRef {
    unsafe {
        llvm::LLVMConstICmp(Pred as c_ushort, V1, V2)
    }
}
pub fn ConstFCmp(Pred: RealPredicate, V1: ValueRef, V2: ValueRef) -> ValueRef {
    unsafe {
        llvm::LLVMConstFCmp(Pred as c_ushort, V1, V2)
    }
}
/* Memory-managed object interface to type handles. */

pub struct TypeNames {
    type_names: HashMap<TypeRef, ~str>,
    named_types: HashMap<~str, TypeRef>
}

impl TypeNames {
    pub fn new() -> TypeNames {
        TypeNames {
            type_names: HashMap::new(),
            named_types: HashMap::new()
        }
    }

    pub fn associate_type(&mut self, s: &str, t: &Type) {
        assert!(self.type_names.insert(t.to_ref(), s.to_owned()));
        assert!(self.named_types.insert(s.to_owned(), t.to_ref()));
    }

    pub fn find_name<'r>(&'r self, ty: &Type) -> Option<&'r str> {
        match self.type_names.find(&ty.to_ref()) {
            Some(a) => Some(a.slice(0, a.len())),
            None => None
        }
    }

    pub fn find_type(&self, s: &str) -> Option<Type> {
        self.named_types.find_equiv(&s).map_consume(|x| Type::from_ref(*x))
    }

    // We have a depth count, because we seem to make infinite types.
    pub fn type_to_str_depth(&self, ty: Type, depth: int) -> ~str {
        match self.find_name(&ty) {
            option::Some(name) => return name.to_owned(),
            None => ()
        }

        if depth == 0 {
            return ~"###";
        }

        unsafe {
            let kind = ty.kind();

            match kind {
                Void => ~"Void",
                Half => ~"Half",
                Float => ~"Float",
                Double => ~"Double",
                X86_FP80 => ~"X86_FP80",
                FP128 => ~"FP128",
                PPC_FP128 => ~"PPC_FP128",
                Label => ~"Label",
                Vector => ~"Vector",
                Metadata => ~"Metadata",
                X86_MMX => ~"X86_MMAX",
                Integer => {
                    fmt!("i%d", llvm::LLVMGetIntTypeWidth(ty.to_ref()) as int)
                }
                Function => {
                    let out_ty = ty.return_type();
                    let args = ty.func_params();
                    let args =
                        args.map(|&ty| self.type_to_str_depth(ty, depth-1)).connect(", ");
                    let out_ty = self.type_to_str_depth(out_ty, depth-1);
                    fmt!("fn(%s) -> %s", args, out_ty)
                }
                Struct => {
                    let tys = ty.field_types();
                    let tys = tys.map(|&ty| self.type_to_str_depth(ty, depth-1)).connect(", ");
                    fmt!("{%s}", tys)
                }
                Array => {
                    let el_ty = ty.element_type();
                    let el_ty = self.type_to_str_depth(el_ty, depth-1);
                    let len = ty.array_length();
                    fmt!("[%s x %u]", el_ty, len)
                }
                Pointer => {
                    let el_ty = ty.element_type();
                    let el_ty = self.type_to_str_depth(el_ty, depth-1);
                    fmt!("*%s", el_ty)
                }
                _ => fail!("Unknown Type Kind (%u)", kind as uint)
            }
        }
    }

    pub fn type_to_str(&self, ty: Type) -> ~str {
        self.type_to_str_depth(ty, 30)
    }

    pub fn val_to_str(&self, val: ValueRef) -> ~str {
        unsafe {
            let ty = Type::from_ref(llvm::LLVMTypeOf(val));
            self.type_to_str(ty)
        }
    }
}


/* Memory-managed interface to target data. */

pub struct target_data_res {
    TD: TargetDataRef,
}

impl Drop for target_data_res {
    fn drop(&self) {
        unsafe {
            llvm::LLVMDisposeTargetData(self.TD);
        }
    }
}

pub fn target_data_res(TD: TargetDataRef) -> target_data_res {
    target_data_res {
        TD: TD
    }
}

pub struct TargetData {
    lltd: TargetDataRef,
    dtor: @target_data_res
}

pub fn mk_target_data(string_rep: &str) -> TargetData {
    let lltd = do string_rep.as_c_str |buf| {
        unsafe { llvm::LLVMCreateTargetData(buf) }
    };

    TargetData {
        lltd: lltd,
        dtor: @target_data_res(lltd)
    }
}

/* Memory-managed interface to pass managers. */

pub struct pass_manager_res {
    PM: PassManagerRef,
}

impl Drop for pass_manager_res {
    fn drop(&self) {
        unsafe {
            llvm::LLVMDisposePassManager(self.PM);
        }
    }
}

pub fn pass_manager_res(PM: PassManagerRef) -> pass_manager_res {
    pass_manager_res {
        PM: PM
    }
}

pub struct PassManager {
    llpm: PassManagerRef,
    dtor: @pass_manager_res
}

pub fn mk_pass_manager() -> PassManager {
    unsafe {
        let llpm = llvm::LLVMCreatePassManager();

        PassManager {
            llpm: llpm,
            dtor: @pass_manager_res(llpm)
        }
    }
}

/* Memory-managed interface to object files. */

pub struct object_file_res {
    ObjectFile: ObjectFileRef,
}

impl Drop for object_file_res {
    fn drop(&self) {
        unsafe {
            llvm::LLVMDisposeObjectFile(self.ObjectFile);
        }
    }
}

pub fn object_file_res(ObjFile: ObjectFileRef) -> object_file_res {
    object_file_res {
        ObjectFile: ObjFile
    }
}

pub struct ObjectFile {
    llof: ObjectFileRef,
    dtor: @object_file_res
}

pub fn mk_object_file(llmb: MemoryBufferRef) -> Option<ObjectFile> {
    unsafe {
        let llof = llvm::LLVMCreateObjectFile(llmb);
        if llof as int == 0 { return option::None::<ObjectFile>; }

        option::Some(ObjectFile {
            llof: llof,
            dtor: @object_file_res(llof)
        })
    }
}

/* Memory-managed interface to section iterators. */

pub struct section_iter_res {
    SI: SectionIteratorRef,
}

impl Drop for section_iter_res {
    fn drop(&self) {
        unsafe {
            llvm::LLVMDisposeSectionIterator(self.SI);
        }
    }
}

pub fn section_iter_res(SI: SectionIteratorRef) -> section_iter_res {
    section_iter_res {
        SI: SI
    }
}

pub struct SectionIter {
    llsi: SectionIteratorRef,
    dtor: @section_iter_res
}

pub fn mk_section_iter(llof: ObjectFileRef) -> SectionIter {
    unsafe {
        let llsi = llvm::LLVMGetSections(llof);
        SectionIter {
            llsi: llsi,
            dtor: @section_iter_res(llsi)
        }
    }
}
