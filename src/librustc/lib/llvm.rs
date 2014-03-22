// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[allow(non_uppercase_pattern_statics)];
#[allow(non_camel_case_types)];

use std::c_str::ToCStr;
use std::cell::RefCell;
use collections::HashMap;
use std::libc::{c_uint, c_ushort, c_void, free};
use std::str::raw::from_c_str;

use middle::trans::type_::Type;

pub type Opcode = u32;
pub type Bool = c_uint;

pub static True: Bool = 1 as Bool;
pub static False: Bool = 0 as Bool;

// Consts for the LLVM CallConv type, pre-cast to uint.

#[deriving(Eq)]
pub enum CallConv {
    CCallConv = 0,
    FastCallConv = 8,
    ColdCallConv = 9,
    X86StdcallCallConv = 64,
    X86FastcallCallConv = 65,
    X86_64_Win64 = 79,
}

pub enum Visibility {
    LLVMDefaultVisibility = 0,
    HiddenVisibility = 1,
    ProtectedVisibility = 2,
}

// This enum omits the obsolete (and no-op) linkage types DLLImportLinkage,
// DLLExportLinkage, GhostLinkage and LinkOnceODRAutoHideLinkage.
// LinkerPrivateLinkage and LinkerPrivateWeakLinkage are not included either;
// they've been removed in upstream LLVM commit r203866.
pub enum Linkage {
    ExternalLinkage = 0,
    AvailableExternallyLinkage = 1,
    LinkOnceAnyLinkage = 2,
    LinkOnceODRLinkage = 3,
    WeakAnyLinkage = 5,
    WeakODRLinkage = 6,
    AppendingLinkage = 7,
    InternalLinkage = 8,
    PrivateLinkage = 9,
    ExternalWeakLinkage = 12,
    CommonLinkage = 14,
}

#[deriving(Clone)]
pub enum Attribute {
    ZExtAttribute = 1 << 0,
    SExtAttribute = 1 << 1,
    NoReturnAttribute = 1 << 2,
    InRegAttribute = 1 << 3,
    StructRetAttribute = 1 << 4,
    NoUnwindAttribute = 1 << 5,
    NoAliasAttribute = 1 << 6,
    ByValAttribute = 1 << 7,
    NestAttribute = 1 << 8,
    ReadNoneAttribute = 1 << 9,
    ReadOnlyAttribute = 1 << 10,
    NoInlineAttribute = 1 << 11,
    AlwaysInlineAttribute = 1 << 12,
    OptimizeForSizeAttribute = 1 << 13,
    StackProtectAttribute = 1 << 14,
    StackProtectReqAttribute = 1 << 15,
    AlignmentAttribute = 31 << 16,
    NoCaptureAttribute = 1 << 21,
    NoRedZoneAttribute = 1 << 22,
    NoImplicitFloatAttribute = 1 << 23,
    NakedAttribute = 1 << 24,
    InlineHintAttribute = 1 << 25,
    StackAttribute = 7 << 26,
    ReturnsTwiceAttribute = 1 << 29,
    UWTableAttribute = 1 << 30,
    NonLazyBindAttribute = 1 << 31,
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
#[deriving(Eq)]
#[repr(C)]
pub enum TypeKind {
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
    X86_MMX   = 15,
}

#[repr(C)]
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

#[repr(C)]
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

// Consts for the LLVMCodeGenFileType type (in include/llvm/c/TargetMachine.h)
#[repr(C)]
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

#[deriving(Eq)]
#[repr(C)]
pub enum CodeGenOptLevel {
    CodeGenLevelNone = 0,
    CodeGenLevelLess = 1,
    CodeGenLevelDefault = 2,
    CodeGenLevelAggressive = 3,
}

#[repr(C)]
pub enum RelocMode {
    RelocDefault = 0,
    RelocStatic = 1,
    RelocPIC = 2,
    RelocDynamicNoPic = 3,
}

#[repr(C)]
pub enum CodeGenModel {
    CodeModelDefault = 0,
    CodeModelJITDefault = 1,
    CodeModelSmall = 2,
    CodeModelKernel = 3,
    CodeModelMedium = 4,
    CodeModelLarge = 5,
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
pub enum TargetMachine_opaque {}
pub type TargetMachineRef = *TargetMachine_opaque;
pub enum Archive_opaque {}
pub type ArchiveRef = *Archive_opaque;

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
    use super::{ValueRef, TargetMachineRef, FileType, ArchiveRef};
    use super::{CodeGenModel, RelocMode, CodeGenOptLevel};
    use super::debuginfo::*;
    use std::libc::{c_char, c_int, c_longlong, c_ushort, c_uint, c_ulonglong,
                    size_t};

    // Link to our native llvm bindings (things that we need to use the C++ api
    // for) and because llvm is written in C++ we need to link against libstdc++
    //
    // You'll probably notice that there is an omission of all LLVM libraries
    // from this location. This is because the set of LLVM libraries that we
    // link to is mostly defined by LLVM, and the `llvm-config` tool is used to
    // figure out the exact set of libraries. To do this, the build system
    // generates an llvmdeps.rs file next to this one which will be
    // automatically updated whenever LLVM is updated to include an up-to-date
    // set of the libraries we need to link to LLVM for.
    #[link(name = "rustllvm", kind = "static")]
    extern {
        /* Create and destroy contexts. */
        pub fn LLVMContextCreate() -> ContextRef;
        pub fn LLVMContextDispose(C: ContextRef);
        pub fn LLVMGetMDKindIDInContext(C: ContextRef,
                                        Name: *c_char,
                                        SLen: c_uint)
                                        -> c_uint;

        /* Create and destroy modules. */
        pub fn LLVMModuleCreateWithNameInContext(ModuleID: *c_char,
                                                 C: ContextRef)
                                                 -> ModuleRef;
        pub fn LLVMGetModuleContext(M: ModuleRef) -> ContextRef;
        pub fn LLVMDisposeModule(M: ModuleRef);

        /** Data layout. See Module::getDataLayout. */
        pub fn LLVMGetDataLayout(M: ModuleRef) -> *c_char;
        pub fn LLVMSetDataLayout(M: ModuleRef, Triple: *c_char);

        /** Target triple. See Module::getTargetTriple. */
        pub fn LLVMGetTarget(M: ModuleRef) -> *c_char;
        pub fn LLVMSetTarget(M: ModuleRef, Triple: *c_char);

        /** See Module::dump. */
        pub fn LLVMDumpModule(M: ModuleRef);

        /** See Module::setModuleInlineAsm. */
        pub fn LLVMSetModuleInlineAsm(M: ModuleRef, Asm: *c_char);

        /** See llvm::LLVMTypeKind::getTypeID. */
        pub fn LLVMGetTypeKind(Ty: TypeRef) -> TypeKind;

        /** See llvm::LLVMType::getContext. */
        pub fn LLVMGetTypeContext(Ty: TypeRef) -> ContextRef;

        /* Operations on integer types */
        pub fn LLVMInt1TypeInContext(C: ContextRef) -> TypeRef;
        pub fn LLVMInt8TypeInContext(C: ContextRef) -> TypeRef;
        pub fn LLVMInt16TypeInContext(C: ContextRef) -> TypeRef;
        pub fn LLVMInt32TypeInContext(C: ContextRef) -> TypeRef;
        pub fn LLVMInt64TypeInContext(C: ContextRef) -> TypeRef;
        pub fn LLVMIntTypeInContext(C: ContextRef, NumBits: c_uint)
                                    -> TypeRef;

        pub fn LLVMGetIntTypeWidth(IntegerTy: TypeRef) -> c_uint;

        /* Operations on real types */
        pub fn LLVMFloatTypeInContext(C: ContextRef) -> TypeRef;
        pub fn LLVMDoubleTypeInContext(C: ContextRef) -> TypeRef;
        pub fn LLVMX86FP80TypeInContext(C: ContextRef) -> TypeRef;
        pub fn LLVMFP128TypeInContext(C: ContextRef) -> TypeRef;
        pub fn LLVMPPCFP128TypeInContext(C: ContextRef) -> TypeRef;

        /* Operations on function types */
        pub fn LLVMFunctionType(ReturnType: TypeRef,
                                ParamTypes: *TypeRef,
                                ParamCount: c_uint,
                                IsVarArg: Bool)
                                -> TypeRef;
        pub fn LLVMIsFunctionVarArg(FunctionTy: TypeRef) -> Bool;
        pub fn LLVMGetReturnType(FunctionTy: TypeRef) -> TypeRef;
        pub fn LLVMCountParamTypes(FunctionTy: TypeRef) -> c_uint;
        pub fn LLVMGetParamTypes(FunctionTy: TypeRef, Dest: *TypeRef);

        /* Operations on struct types */
        pub fn LLVMStructTypeInContext(C: ContextRef,
                                       ElementTypes: *TypeRef,
                                       ElementCount: c_uint,
                                       Packed: Bool)
                                       -> TypeRef;
        pub fn LLVMCountStructElementTypes(StructTy: TypeRef) -> c_uint;
        pub fn LLVMGetStructElementTypes(StructTy: TypeRef,
                                         Dest: *mut TypeRef);
        pub fn LLVMIsPackedStruct(StructTy: TypeRef) -> Bool;

        /* Operations on array, pointer, and vector types (sequence types) */
        pub fn LLVMArrayType(ElementType: TypeRef, ElementCount: c_uint)
                             -> TypeRef;
        pub fn LLVMPointerType(ElementType: TypeRef, AddressSpace: c_uint)
                               -> TypeRef;
        pub fn LLVMVectorType(ElementType: TypeRef, ElementCount: c_uint)
                              -> TypeRef;

        pub fn LLVMGetElementType(Ty: TypeRef) -> TypeRef;
        pub fn LLVMGetArrayLength(ArrayTy: TypeRef) -> c_uint;
        pub fn LLVMGetPointerAddressSpace(PointerTy: TypeRef) -> c_uint;
        pub fn LLVMGetPointerToGlobal(EE: ExecutionEngineRef, V: ValueRef)
                                      -> *();
        pub fn LLVMGetVectorSize(VectorTy: TypeRef) -> c_uint;

        /* Operations on other types */
        pub fn LLVMVoidTypeInContext(C: ContextRef) -> TypeRef;
        pub fn LLVMLabelTypeInContext(C: ContextRef) -> TypeRef;
        pub fn LLVMMetadataTypeInContext(C: ContextRef) -> TypeRef;

        /* Operations on all values */
        pub fn LLVMTypeOf(Val: ValueRef) -> TypeRef;
        pub fn LLVMGetValueName(Val: ValueRef) -> *c_char;
        pub fn LLVMSetValueName(Val: ValueRef, Name: *c_char);
        pub fn LLVMDumpValue(Val: ValueRef);
        pub fn LLVMReplaceAllUsesWith(OldVal: ValueRef, NewVal: ValueRef);
        pub fn LLVMHasMetadata(Val: ValueRef) -> c_int;
        pub fn LLVMGetMetadata(Val: ValueRef, KindID: c_uint) -> ValueRef;
        pub fn LLVMSetMetadata(Val: ValueRef, KindID: c_uint, Node: ValueRef);

        /* Operations on Uses */
        pub fn LLVMGetFirstUse(Val: ValueRef) -> UseRef;
        pub fn LLVMGetNextUse(U: UseRef) -> UseRef;
        pub fn LLVMGetUser(U: UseRef) -> ValueRef;
        pub fn LLVMGetUsedValue(U: UseRef) -> ValueRef;

        /* Operations on Users */
        pub fn LLVMGetNumOperands(Val: ValueRef) -> c_int;
        pub fn LLVMGetOperand(Val: ValueRef, Index: c_uint) -> ValueRef;
        pub fn LLVMSetOperand(Val: ValueRef, Index: c_uint, Op: ValueRef);

        /* Operations on constants of any type */
        pub fn LLVMConstNull(Ty: TypeRef) -> ValueRef;
        /* all zeroes */
        pub fn LLVMConstAllOnes(Ty: TypeRef) -> ValueRef;
        pub fn LLVMConstICmp(Pred: c_ushort, V1: ValueRef, V2: ValueRef)
                             -> ValueRef;
        pub fn LLVMConstFCmp(Pred: c_ushort, V1: ValueRef, V2: ValueRef)
                             -> ValueRef;
        /* only for int/vector */
        pub fn LLVMGetUndef(Ty: TypeRef) -> ValueRef;
        pub fn LLVMIsConstant(Val: ValueRef) -> Bool;
        pub fn LLVMIsNull(Val: ValueRef) -> Bool;
        pub fn LLVMIsUndef(Val: ValueRef) -> Bool;
        pub fn LLVMConstPointerNull(Ty: TypeRef) -> ValueRef;

        /* Operations on metadata */
        pub fn LLVMMDStringInContext(C: ContextRef,
                                     Str: *c_char,
                                     SLen: c_uint)
                                     -> ValueRef;
        pub fn LLVMMDNodeInContext(C: ContextRef,
                                   Vals: *ValueRef,
                                   Count: c_uint)
                                   -> ValueRef;
        pub fn LLVMAddNamedMetadataOperand(M: ModuleRef,
                                           Str: *c_char,
                                           Val: ValueRef);

        /* Operations on scalar constants */
        pub fn LLVMConstInt(IntTy: TypeRef, N: c_ulonglong, SignExtend: Bool)
                            -> ValueRef;
        pub fn LLVMConstIntOfString(IntTy: TypeRef, Text: *c_char, Radix: u8)
                                    -> ValueRef;
        pub fn LLVMConstIntOfStringAndSize(IntTy: TypeRef,
                                           Text: *c_char,
                                           SLen: c_uint,
                                           Radix: u8)
                                           -> ValueRef;
        pub fn LLVMConstReal(RealTy: TypeRef, N: f64) -> ValueRef;
        pub fn LLVMConstRealOfString(RealTy: TypeRef, Text: *c_char)
                                     -> ValueRef;
        pub fn LLVMConstRealOfStringAndSize(RealTy: TypeRef,
                                            Text: *c_char,
                                            SLen: c_uint)
                                            -> ValueRef;
        pub fn LLVMConstIntGetZExtValue(ConstantVal: ValueRef) -> c_ulonglong;
        pub fn LLVMConstIntGetSExtValue(ConstantVal: ValueRef) -> c_longlong;


        /* Operations on composite constants */
        pub fn LLVMConstStringInContext(C: ContextRef,
                                        Str: *c_char,
                                        Length: c_uint,
                                        DontNullTerminate: Bool)
                                        -> ValueRef;
        pub fn LLVMConstStructInContext(C: ContextRef,
                                        ConstantVals: *ValueRef,
                                        Count: c_uint,
                                        Packed: Bool)
                                        -> ValueRef;

        pub fn LLVMConstArray(ElementTy: TypeRef,
                              ConstantVals: *ValueRef,
                              Length: c_uint)
                              -> ValueRef;
        pub fn LLVMConstVector(ScalarConstantVals: *ValueRef, Size: c_uint)
                               -> ValueRef;

        /* Constant expressions */
        pub fn LLVMAlignOf(Ty: TypeRef) -> ValueRef;
        pub fn LLVMSizeOf(Ty: TypeRef) -> ValueRef;
        pub fn LLVMConstNeg(ConstantVal: ValueRef) -> ValueRef;
        pub fn LLVMConstNSWNeg(ConstantVal: ValueRef) -> ValueRef;
        pub fn LLVMConstNUWNeg(ConstantVal: ValueRef) -> ValueRef;
        pub fn LLVMConstFNeg(ConstantVal: ValueRef) -> ValueRef;
        pub fn LLVMConstNot(ConstantVal: ValueRef) -> ValueRef;
        pub fn LLVMConstAdd(LHSConstant: ValueRef, RHSConstant: ValueRef)
                            -> ValueRef;
        pub fn LLVMConstNSWAdd(LHSConstant: ValueRef, RHSConstant: ValueRef)
                               -> ValueRef;
        pub fn LLVMConstNUWAdd(LHSConstant: ValueRef, RHSConstant: ValueRef)
                               -> ValueRef;
        pub fn LLVMConstFAdd(LHSConstant: ValueRef, RHSConstant: ValueRef)
                             -> ValueRef;
        pub fn LLVMConstSub(LHSConstant: ValueRef, RHSConstant: ValueRef)
                            -> ValueRef;
        pub fn LLVMConstNSWSub(LHSConstant: ValueRef, RHSConstant: ValueRef)
                               -> ValueRef;
        pub fn LLVMConstNUWSub(LHSConstant: ValueRef, RHSConstant: ValueRef)
                               -> ValueRef;
        pub fn LLVMConstFSub(LHSConstant: ValueRef, RHSConstant: ValueRef)
                             -> ValueRef;
        pub fn LLVMConstMul(LHSConstant: ValueRef, RHSConstant: ValueRef)
                            -> ValueRef;
        pub fn LLVMConstNSWMul(LHSConstant: ValueRef, RHSConstant: ValueRef)
                               -> ValueRef;
        pub fn LLVMConstNUWMul(LHSConstant: ValueRef, RHSConstant: ValueRef)
                               -> ValueRef;
        pub fn LLVMConstFMul(LHSConstant: ValueRef, RHSConstant: ValueRef)
                             -> ValueRef;
        pub fn LLVMConstUDiv(LHSConstant: ValueRef, RHSConstant: ValueRef)
                             -> ValueRef;
        pub fn LLVMConstSDiv(LHSConstant: ValueRef, RHSConstant: ValueRef)
                             -> ValueRef;
        pub fn LLVMConstExactSDiv(LHSConstant: ValueRef,
                                  RHSConstant: ValueRef)
                                  -> ValueRef;
        pub fn LLVMConstFDiv(LHSConstant: ValueRef, RHSConstant: ValueRef)
                             -> ValueRef;
        pub fn LLVMConstURem(LHSConstant: ValueRef, RHSConstant: ValueRef)
                             -> ValueRef;
        pub fn LLVMConstSRem(LHSConstant: ValueRef, RHSConstant: ValueRef)
                             -> ValueRef;
        pub fn LLVMConstFRem(LHSConstant: ValueRef, RHSConstant: ValueRef)
                             -> ValueRef;
        pub fn LLVMConstAnd(LHSConstant: ValueRef, RHSConstant: ValueRef)
                            -> ValueRef;
        pub fn LLVMConstOr(LHSConstant: ValueRef, RHSConstant: ValueRef)
                           -> ValueRef;
        pub fn LLVMConstXor(LHSConstant: ValueRef, RHSConstant: ValueRef)
                            -> ValueRef;
        pub fn LLVMConstShl(LHSConstant: ValueRef, RHSConstant: ValueRef)
                            -> ValueRef;
        pub fn LLVMConstLShr(LHSConstant: ValueRef, RHSConstant: ValueRef)
                             -> ValueRef;
        pub fn LLVMConstAShr(LHSConstant: ValueRef, RHSConstant: ValueRef)
                             -> ValueRef;
        pub fn LLVMConstGEP(ConstantVal: ValueRef,
                            ConstantIndices: *ValueRef,
                            NumIndices: c_uint)
                            -> ValueRef;
        pub fn LLVMConstInBoundsGEP(ConstantVal: ValueRef,
                                    ConstantIndices: *ValueRef,
                                    NumIndices: c_uint)
                                    -> ValueRef;
        pub fn LLVMConstTrunc(ConstantVal: ValueRef, ToType: TypeRef)
                              -> ValueRef;
        pub fn LLVMConstSExt(ConstantVal: ValueRef, ToType: TypeRef)
                             -> ValueRef;
        pub fn LLVMConstZExt(ConstantVal: ValueRef, ToType: TypeRef)
                             -> ValueRef;
        pub fn LLVMConstFPTrunc(ConstantVal: ValueRef, ToType: TypeRef)
                                -> ValueRef;
        pub fn LLVMConstFPExt(ConstantVal: ValueRef, ToType: TypeRef)
                              -> ValueRef;
        pub fn LLVMConstUIToFP(ConstantVal: ValueRef, ToType: TypeRef)
                               -> ValueRef;
        pub fn LLVMConstSIToFP(ConstantVal: ValueRef, ToType: TypeRef)
                               -> ValueRef;
        pub fn LLVMConstFPToUI(ConstantVal: ValueRef, ToType: TypeRef)
                               -> ValueRef;
        pub fn LLVMConstFPToSI(ConstantVal: ValueRef, ToType: TypeRef)
                               -> ValueRef;
        pub fn LLVMConstPtrToInt(ConstantVal: ValueRef, ToType: TypeRef)
                                 -> ValueRef;
        pub fn LLVMConstIntToPtr(ConstantVal: ValueRef, ToType: TypeRef)
                                 -> ValueRef;
        pub fn LLVMConstBitCast(ConstantVal: ValueRef, ToType: TypeRef)
                                -> ValueRef;
        pub fn LLVMConstZExtOrBitCast(ConstantVal: ValueRef, ToType: TypeRef)
                                      -> ValueRef;
        pub fn LLVMConstSExtOrBitCast(ConstantVal: ValueRef, ToType: TypeRef)
                                      -> ValueRef;
        pub fn LLVMConstTruncOrBitCast(ConstantVal: ValueRef, ToType: TypeRef)
                                       -> ValueRef;
        pub fn LLVMConstPointerCast(ConstantVal: ValueRef, ToType: TypeRef)
                                    -> ValueRef;
        pub fn LLVMConstIntCast(ConstantVal: ValueRef,
                                ToType: TypeRef,
                                isSigned: Bool)
                                -> ValueRef;
        pub fn LLVMConstFPCast(ConstantVal: ValueRef, ToType: TypeRef)
                               -> ValueRef;
        pub fn LLVMConstSelect(ConstantCondition: ValueRef,
                               ConstantIfTrue: ValueRef,
                               ConstantIfFalse: ValueRef)
                               -> ValueRef;
        pub fn LLVMConstExtractElement(VectorConstant: ValueRef,
                                       IndexConstant: ValueRef)
                                       -> ValueRef;
        pub fn LLVMConstInsertElement(VectorConstant: ValueRef,
                                      ElementValueConstant: ValueRef,
                                      IndexConstant: ValueRef)
                                      -> ValueRef;
        pub fn LLVMConstShuffleVector(VectorAConstant: ValueRef,
                                      VectorBConstant: ValueRef,
                                      MaskConstant: ValueRef)
                                      -> ValueRef;
        pub fn LLVMConstExtractValue(AggConstant: ValueRef,
                                     IdxList: *c_uint,
                                     NumIdx: c_uint)
                                     -> ValueRef;
        pub fn LLVMConstInsertValue(AggConstant: ValueRef,
                                    ElementValueConstant: ValueRef,
                                    IdxList: *c_uint,
                                    NumIdx: c_uint)
                                    -> ValueRef;
        pub fn LLVMConstInlineAsm(Ty: TypeRef,
                                  AsmString: *c_char,
                                  Constraints: *c_char,
                                  HasSideEffects: Bool,
                                  IsAlignStack: Bool)
                                  -> ValueRef;
        pub fn LLVMBlockAddress(F: ValueRef, BB: BasicBlockRef) -> ValueRef;



        /* Operations on global variables, functions, and aliases (globals) */
        pub fn LLVMGetGlobalParent(Global: ValueRef) -> ModuleRef;
        pub fn LLVMIsDeclaration(Global: ValueRef) -> Bool;
        pub fn LLVMGetLinkage(Global: ValueRef) -> c_uint;
        pub fn LLVMSetLinkage(Global: ValueRef, Link: c_uint);
        pub fn LLVMGetSection(Global: ValueRef) -> *c_char;
        pub fn LLVMSetSection(Global: ValueRef, Section: *c_char);
        pub fn LLVMGetVisibility(Global: ValueRef) -> c_uint;
        pub fn LLVMSetVisibility(Global: ValueRef, Viz: c_uint);
        pub fn LLVMGetAlignment(Global: ValueRef) -> c_uint;
        pub fn LLVMSetAlignment(Global: ValueRef, Bytes: c_uint);


        /* Operations on global variables */
        pub fn LLVMAddGlobal(M: ModuleRef, Ty: TypeRef, Name: *c_char)
                             -> ValueRef;
        pub fn LLVMAddGlobalInAddressSpace(M: ModuleRef,
                                           Ty: TypeRef,
                                           Name: *c_char,
                                           AddressSpace: c_uint)
                                           -> ValueRef;
        pub fn LLVMGetNamedGlobal(M: ModuleRef, Name: *c_char) -> ValueRef;
        pub fn LLVMGetFirstGlobal(M: ModuleRef) -> ValueRef;
        pub fn LLVMGetLastGlobal(M: ModuleRef) -> ValueRef;
        pub fn LLVMGetNextGlobal(GlobalVar: ValueRef) -> ValueRef;
        pub fn LLVMGetPreviousGlobal(GlobalVar: ValueRef) -> ValueRef;
        pub fn LLVMDeleteGlobal(GlobalVar: ValueRef);
        pub fn LLVMGetInitializer(GlobalVar: ValueRef) -> ValueRef;
        pub fn LLVMSetInitializer(GlobalVar: ValueRef,
                                         ConstantVal: ValueRef);
        pub fn LLVMIsThreadLocal(GlobalVar: ValueRef) -> Bool;
        pub fn LLVMSetThreadLocal(GlobalVar: ValueRef, IsThreadLocal: Bool);
        pub fn LLVMIsGlobalConstant(GlobalVar: ValueRef) -> Bool;
        pub fn LLVMSetGlobalConstant(GlobalVar: ValueRef, IsConstant: Bool);

        /* Operations on aliases */
        pub fn LLVMAddAlias(M: ModuleRef,
                            Ty: TypeRef,
                            Aliasee: ValueRef,
                            Name: *c_char)
                            -> ValueRef;

        /* Operations on functions */
        pub fn LLVMAddFunction(M: ModuleRef,
                               Name: *c_char,
                               FunctionTy: TypeRef)
                               -> ValueRef;
        pub fn LLVMGetNamedFunction(M: ModuleRef, Name: *c_char) -> ValueRef;
        pub fn LLVMGetFirstFunction(M: ModuleRef) -> ValueRef;
        pub fn LLVMGetLastFunction(M: ModuleRef) -> ValueRef;
        pub fn LLVMGetNextFunction(Fn: ValueRef) -> ValueRef;
        pub fn LLVMGetPreviousFunction(Fn: ValueRef) -> ValueRef;
        pub fn LLVMDeleteFunction(Fn: ValueRef);
        pub fn LLVMGetOrInsertFunction(M: ModuleRef,
                                       Name: *c_char,
                                       FunctionTy: TypeRef)
                                       -> ValueRef;
        pub fn LLVMGetIntrinsicID(Fn: ValueRef) -> c_uint;
        pub fn LLVMGetFunctionCallConv(Fn: ValueRef) -> c_uint;
        pub fn LLVMSetFunctionCallConv(Fn: ValueRef, CC: c_uint);
        pub fn LLVMGetGC(Fn: ValueRef) -> *c_char;
        pub fn LLVMSetGC(Fn: ValueRef, Name: *c_char);
        pub fn LLVMAddFunctionAttr(Fn: ValueRef, PA: c_uint);
        pub fn LLVMAddFunctionAttrString(Fn: ValueRef, Name: *c_char);
        pub fn LLVMGetFunctionAttr(Fn: ValueRef) -> c_ulonglong;

        pub fn LLVMAddReturnAttribute(Fn: ValueRef, PA: c_uint);
        pub fn LLVMRemoveReturnAttribute(Fn: ValueRef, PA: c_uint);

        pub fn LLVMAddColdAttribute(Fn: ValueRef);

        pub fn LLVMRemoveFunctionAttr(Fn: ValueRef,
                                      PA: c_ulonglong,
                                      HighPA: c_ulonglong);

        /* Operations on parameters */
        pub fn LLVMCountParams(Fn: ValueRef) -> c_uint;
        pub fn LLVMGetParams(Fn: ValueRef, Params: *ValueRef);
        pub fn LLVMGetParam(Fn: ValueRef, Index: c_uint) -> ValueRef;
        pub fn LLVMGetParamParent(Inst: ValueRef) -> ValueRef;
        pub fn LLVMGetFirstParam(Fn: ValueRef) -> ValueRef;
        pub fn LLVMGetLastParam(Fn: ValueRef) -> ValueRef;
        pub fn LLVMGetNextParam(Arg: ValueRef) -> ValueRef;
        pub fn LLVMGetPreviousParam(Arg: ValueRef) -> ValueRef;
        pub fn LLVMAddAttribute(Arg: ValueRef, PA: c_uint);
        pub fn LLVMRemoveAttribute(Arg: ValueRef, PA: c_uint);
        pub fn LLVMGetAttribute(Arg: ValueRef) -> c_uint;
        pub fn LLVMSetParamAlignment(Arg: ValueRef, align: c_uint);

        /* Operations on basic blocks */
        pub fn LLVMBasicBlockAsValue(BB: BasicBlockRef) -> ValueRef;
        pub fn LLVMValueIsBasicBlock(Val: ValueRef) -> Bool;
        pub fn LLVMValueAsBasicBlock(Val: ValueRef) -> BasicBlockRef;
        pub fn LLVMGetBasicBlockParent(BB: BasicBlockRef) -> ValueRef;
        pub fn LLVMCountBasicBlocks(Fn: ValueRef) -> c_uint;
        pub fn LLVMGetBasicBlocks(Fn: ValueRef, BasicBlocks: *ValueRef);
        pub fn LLVMGetFirstBasicBlock(Fn: ValueRef) -> BasicBlockRef;
        pub fn LLVMGetLastBasicBlock(Fn: ValueRef) -> BasicBlockRef;
        pub fn LLVMGetNextBasicBlock(BB: BasicBlockRef) -> BasicBlockRef;
        pub fn LLVMGetPreviousBasicBlock(BB: BasicBlockRef) -> BasicBlockRef;
        pub fn LLVMGetEntryBasicBlock(Fn: ValueRef) -> BasicBlockRef;

        pub fn LLVMAppendBasicBlockInContext(C: ContextRef,
                                             Fn: ValueRef,
                                             Name: *c_char)
                                             -> BasicBlockRef;
        pub fn LLVMInsertBasicBlockInContext(C: ContextRef,
                                             BB: BasicBlockRef,
                                             Name: *c_char)
                                             -> BasicBlockRef;
        pub fn LLVMDeleteBasicBlock(BB: BasicBlockRef);

        pub fn LLVMMoveBasicBlockAfter(BB: BasicBlockRef,
                                       MoveAfter: BasicBlockRef);

        pub fn LLVMMoveBasicBlockBefore(BB: BasicBlockRef,
                                        MoveBefore: BasicBlockRef);

        /* Operations on instructions */
        pub fn LLVMGetInstructionParent(Inst: ValueRef) -> BasicBlockRef;
        pub fn LLVMGetFirstInstruction(BB: BasicBlockRef) -> ValueRef;
        pub fn LLVMGetLastInstruction(BB: BasicBlockRef) -> ValueRef;
        pub fn LLVMGetNextInstruction(Inst: ValueRef) -> ValueRef;
        pub fn LLVMGetPreviousInstruction(Inst: ValueRef) -> ValueRef;
        pub fn LLVMInstructionEraseFromParent(Inst: ValueRef);

        /* Operations on call sites */
        pub fn LLVMSetInstructionCallConv(Instr: ValueRef, CC: c_uint);
        pub fn LLVMGetInstructionCallConv(Instr: ValueRef) -> c_uint;
        pub fn LLVMAddInstrAttribute(Instr: ValueRef,
                                     index: c_uint,
                                     IA: c_uint);
        pub fn LLVMRemoveInstrAttribute(Instr: ValueRef,
                                        index: c_uint,
                                        IA: c_uint);
        pub fn LLVMSetInstrParamAlignment(Instr: ValueRef,
                                          index: c_uint,
                                          align: c_uint);

        /* Operations on call instructions (only) */
        pub fn LLVMIsTailCall(CallInst: ValueRef) -> Bool;
        pub fn LLVMSetTailCall(CallInst: ValueRef, IsTailCall: Bool);

        /* Operations on load/store instructions (only) */
        pub fn LLVMGetVolatile(MemoryAccessInst: ValueRef) -> Bool;
        pub fn LLVMSetVolatile(MemoryAccessInst: ValueRef, volatile: Bool);

        /* Operations on phi nodes */
        pub fn LLVMAddIncoming(PhiNode: ValueRef,
                               IncomingValues: *ValueRef,
                               IncomingBlocks: *BasicBlockRef,
                               Count: c_uint);
        pub fn LLVMCountIncoming(PhiNode: ValueRef) -> c_uint;
        pub fn LLVMGetIncomingValue(PhiNode: ValueRef, Index: c_uint)
                                    -> ValueRef;
        pub fn LLVMGetIncomingBlock(PhiNode: ValueRef, Index: c_uint)
                                    -> BasicBlockRef;

        /* Instruction builders */
        pub fn LLVMCreateBuilderInContext(C: ContextRef) -> BuilderRef;
        pub fn LLVMPositionBuilder(Builder: BuilderRef,
                                   Block: BasicBlockRef,
                                   Instr: ValueRef);
        pub fn LLVMPositionBuilderBefore(Builder: BuilderRef,
                                         Instr: ValueRef);
        pub fn LLVMPositionBuilderAtEnd(Builder: BuilderRef,
                                        Block: BasicBlockRef);
        pub fn LLVMGetInsertBlock(Builder: BuilderRef) -> BasicBlockRef;
        pub fn LLVMClearInsertionPosition(Builder: BuilderRef);
        pub fn LLVMInsertIntoBuilder(Builder: BuilderRef, Instr: ValueRef);
        pub fn LLVMInsertIntoBuilderWithName(Builder: BuilderRef,
                                             Instr: ValueRef,
                                             Name: *c_char);
        pub fn LLVMDisposeBuilder(Builder: BuilderRef);
        pub fn LLVMDisposeExecutionEngine(EE: ExecutionEngineRef);

        /* Metadata */
        pub fn LLVMSetCurrentDebugLocation(Builder: BuilderRef, L: ValueRef);
        pub fn LLVMGetCurrentDebugLocation(Builder: BuilderRef) -> ValueRef;
        pub fn LLVMSetInstDebugLocation(Builder: BuilderRef, Inst: ValueRef);

        /* Terminators */
        pub fn LLVMBuildRetVoid(B: BuilderRef) -> ValueRef;
        pub fn LLVMBuildRet(B: BuilderRef, V: ValueRef) -> ValueRef;
        pub fn LLVMBuildAggregateRet(B: BuilderRef,
                                     RetVals: *ValueRef,
                                     N: c_uint)
                                     -> ValueRef;
        pub fn LLVMBuildBr(B: BuilderRef, Dest: BasicBlockRef) -> ValueRef;
        pub fn LLVMBuildCondBr(B: BuilderRef,
                               If: ValueRef,
                               Then: BasicBlockRef,
                               Else: BasicBlockRef)
                               -> ValueRef;
        pub fn LLVMBuildSwitch(B: BuilderRef,
                               V: ValueRef,
                               Else: BasicBlockRef,
                               NumCases: c_uint)
                               -> ValueRef;
        pub fn LLVMBuildIndirectBr(B: BuilderRef,
                                   Addr: ValueRef,
                                   NumDests: c_uint)
                                   -> ValueRef;
        pub fn LLVMBuildInvoke(B: BuilderRef,
                               Fn: ValueRef,
                               Args: *ValueRef,
                               NumArgs: c_uint,
                               Then: BasicBlockRef,
                               Catch: BasicBlockRef,
                               Name: *c_char)
                               -> ValueRef;
        pub fn LLVMBuildLandingPad(B: BuilderRef,
                                   Ty: TypeRef,
                                   PersFn: ValueRef,
                                   NumClauses: c_uint,
                                   Name: *c_char)
                                   -> ValueRef;
        pub fn LLVMBuildResume(B: BuilderRef, Exn: ValueRef) -> ValueRef;
        pub fn LLVMBuildUnreachable(B: BuilderRef) -> ValueRef;

        /* Add a case to the switch instruction */
        pub fn LLVMAddCase(Switch: ValueRef,
                           OnVal: ValueRef,
                           Dest: BasicBlockRef);

        /* Add a destination to the indirectbr instruction */
        pub fn LLVMAddDestination(IndirectBr: ValueRef, Dest: BasicBlockRef);

        /* Add a clause to the landing pad instruction */
        pub fn LLVMAddClause(LandingPad: ValueRef, ClauseVal: ValueRef);

        /* Set the cleanup on a landing pad instruction */
        pub fn LLVMSetCleanup(LandingPad: ValueRef, Val: Bool);

        /* Arithmetic */
        pub fn LLVMBuildAdd(B: BuilderRef,
                            LHS: ValueRef,
                            RHS: ValueRef,
                            Name: *c_char)
                            -> ValueRef;
        pub fn LLVMBuildNSWAdd(B: BuilderRef,
                               LHS: ValueRef,
                               RHS: ValueRef,
                               Name: *c_char)
                               -> ValueRef;
        pub fn LLVMBuildNUWAdd(B: BuilderRef,
                               LHS: ValueRef,
                               RHS: ValueRef,
                               Name: *c_char)
                               -> ValueRef;
        pub fn LLVMBuildFAdd(B: BuilderRef,
                             LHS: ValueRef,
                             RHS: ValueRef,
                             Name: *c_char)
                             -> ValueRef;
        pub fn LLVMBuildSub(B: BuilderRef,
                            LHS: ValueRef,
                            RHS: ValueRef,
                            Name: *c_char)
                            -> ValueRef;
        pub fn LLVMBuildNSWSub(B: BuilderRef,
                               LHS: ValueRef,
                               RHS: ValueRef,
                               Name: *c_char)
                               -> ValueRef;
        pub fn LLVMBuildNUWSub(B: BuilderRef,
                               LHS: ValueRef,
                               RHS: ValueRef,
                               Name: *c_char)
                               -> ValueRef;
        pub fn LLVMBuildFSub(B: BuilderRef,
                             LHS: ValueRef,
                             RHS: ValueRef,
                             Name: *c_char)
                             -> ValueRef;
        pub fn LLVMBuildMul(B: BuilderRef,
                            LHS: ValueRef,
                            RHS: ValueRef,
                            Name: *c_char)
                            -> ValueRef;
        pub fn LLVMBuildNSWMul(B: BuilderRef,
                               LHS: ValueRef,
                               RHS: ValueRef,
                               Name: *c_char)
                               -> ValueRef;
        pub fn LLVMBuildNUWMul(B: BuilderRef,
                               LHS: ValueRef,
                               RHS: ValueRef,
                               Name: *c_char)
                               -> ValueRef;
        pub fn LLVMBuildFMul(B: BuilderRef,
                             LHS: ValueRef,
                             RHS: ValueRef,
                             Name: *c_char)
                             -> ValueRef;
        pub fn LLVMBuildUDiv(B: BuilderRef,
                             LHS: ValueRef,
                             RHS: ValueRef,
                             Name: *c_char)
                             -> ValueRef;
        pub fn LLVMBuildSDiv(B: BuilderRef,
                             LHS: ValueRef,
                             RHS: ValueRef,
                             Name: *c_char)
                             -> ValueRef;
        pub fn LLVMBuildExactSDiv(B: BuilderRef,
                                  LHS: ValueRef,
                                  RHS: ValueRef,
                                  Name: *c_char)
                                  -> ValueRef;
        pub fn LLVMBuildFDiv(B: BuilderRef,
                             LHS: ValueRef,
                             RHS: ValueRef,
                             Name: *c_char)
                             -> ValueRef;
        pub fn LLVMBuildURem(B: BuilderRef,
                             LHS: ValueRef,
                             RHS: ValueRef,
                             Name: *c_char)
                             -> ValueRef;
        pub fn LLVMBuildSRem(B: BuilderRef,
                             LHS: ValueRef,
                             RHS: ValueRef,
                             Name: *c_char)
                             -> ValueRef;
        pub fn LLVMBuildFRem(B: BuilderRef,
                             LHS: ValueRef,
                             RHS: ValueRef,
                             Name: *c_char)
                             -> ValueRef;
        pub fn LLVMBuildShl(B: BuilderRef,
                            LHS: ValueRef,
                            RHS: ValueRef,
                            Name: *c_char)
                            -> ValueRef;
        pub fn LLVMBuildLShr(B: BuilderRef,
                             LHS: ValueRef,
                             RHS: ValueRef,
                             Name: *c_char)
                             -> ValueRef;
        pub fn LLVMBuildAShr(B: BuilderRef,
                             LHS: ValueRef,
                             RHS: ValueRef,
                             Name: *c_char)
                             -> ValueRef;
        pub fn LLVMBuildAnd(B: BuilderRef,
                            LHS: ValueRef,
                            RHS: ValueRef,
                            Name: *c_char)
                            -> ValueRef;
        pub fn LLVMBuildOr(B: BuilderRef,
                           LHS: ValueRef,
                           RHS: ValueRef,
                           Name: *c_char)
                           -> ValueRef;
        pub fn LLVMBuildXor(B: BuilderRef,
                            LHS: ValueRef,
                            RHS: ValueRef,
                            Name: *c_char)
                            -> ValueRef;
        pub fn LLVMBuildBinOp(B: BuilderRef,
                              Op: Opcode,
                              LHS: ValueRef,
                              RHS: ValueRef,
                              Name: *c_char)
                              -> ValueRef;
        pub fn LLVMBuildNeg(B: BuilderRef, V: ValueRef, Name: *c_char)
                            -> ValueRef;
        pub fn LLVMBuildNSWNeg(B: BuilderRef, V: ValueRef, Name: *c_char)
                               -> ValueRef;
        pub fn LLVMBuildNUWNeg(B: BuilderRef, V: ValueRef, Name: *c_char)
                               -> ValueRef;
        pub fn LLVMBuildFNeg(B: BuilderRef, V: ValueRef, Name: *c_char)
                             -> ValueRef;
        pub fn LLVMBuildNot(B: BuilderRef, V: ValueRef, Name: *c_char)
                            -> ValueRef;

        /* Memory */
        pub fn LLVMBuildMalloc(B: BuilderRef, Ty: TypeRef, Name: *c_char)
                               -> ValueRef;
        pub fn LLVMBuildArrayMalloc(B: BuilderRef,
                                    Ty: TypeRef,
                                    Val: ValueRef,
                                    Name: *c_char)
                                    -> ValueRef;
        pub fn LLVMBuildAlloca(B: BuilderRef, Ty: TypeRef, Name: *c_char)
                               -> ValueRef;
        pub fn LLVMBuildArrayAlloca(B: BuilderRef,
                                    Ty: TypeRef,
                                    Val: ValueRef,
                                    Name: *c_char)
                                    -> ValueRef;
        pub fn LLVMBuildFree(B: BuilderRef, PointerVal: ValueRef) -> ValueRef;
        pub fn LLVMBuildLoad(B: BuilderRef,
                             PointerVal: ValueRef,
                             Name: *c_char)
                             -> ValueRef;

        pub fn LLVMBuildStore(B: BuilderRef, Val: ValueRef, Ptr: ValueRef)
                              -> ValueRef;

        pub fn LLVMBuildGEP(B: BuilderRef,
                            Pointer: ValueRef,
                            Indices: *ValueRef,
                            NumIndices: c_uint,
                            Name: *c_char)
                            -> ValueRef;
        pub fn LLVMBuildInBoundsGEP(B: BuilderRef,
                                    Pointer: ValueRef,
                                    Indices: *ValueRef,
                                    NumIndices: c_uint,
                                    Name: *c_char)
                                    -> ValueRef;
        pub fn LLVMBuildStructGEP(B: BuilderRef,
                                  Pointer: ValueRef,
                                  Idx: c_uint,
                                  Name: *c_char)
                                  -> ValueRef;
        pub fn LLVMBuildGlobalString(B: BuilderRef,
                                     Str: *c_char,
                                     Name: *c_char)
                                     -> ValueRef;
        pub fn LLVMBuildGlobalStringPtr(B: BuilderRef,
                                        Str: *c_char,
                                        Name: *c_char)
                                        -> ValueRef;

        /* Casts */
        pub fn LLVMBuildTrunc(B: BuilderRef,
                              Val: ValueRef,
                              DestTy: TypeRef,
                              Name: *c_char)
                              -> ValueRef;
        pub fn LLVMBuildZExt(B: BuilderRef,
                             Val: ValueRef,
                             DestTy: TypeRef,
                             Name: *c_char)
                             -> ValueRef;
        pub fn LLVMBuildSExt(B: BuilderRef,
                             Val: ValueRef,
                             DestTy: TypeRef,
                             Name: *c_char)
                             -> ValueRef;
        pub fn LLVMBuildFPToUI(B: BuilderRef,
                               Val: ValueRef,
                               DestTy: TypeRef,
                               Name: *c_char)
                               -> ValueRef;
        pub fn LLVMBuildFPToSI(B: BuilderRef,
                               Val: ValueRef,
                               DestTy: TypeRef,
                               Name: *c_char)
                               -> ValueRef;
        pub fn LLVMBuildUIToFP(B: BuilderRef,
                               Val: ValueRef,
                               DestTy: TypeRef,
                               Name: *c_char)
                               -> ValueRef;
        pub fn LLVMBuildSIToFP(B: BuilderRef,
                               Val: ValueRef,
                               DestTy: TypeRef,
                               Name: *c_char)
                               -> ValueRef;
        pub fn LLVMBuildFPTrunc(B: BuilderRef,
                                Val: ValueRef,
                                DestTy: TypeRef,
                                Name: *c_char)
                                -> ValueRef;
        pub fn LLVMBuildFPExt(B: BuilderRef,
                              Val: ValueRef,
                              DestTy: TypeRef,
                              Name: *c_char)
                              -> ValueRef;
        pub fn LLVMBuildPtrToInt(B: BuilderRef,
                                 Val: ValueRef,
                                 DestTy: TypeRef,
                                 Name: *c_char)
                                 -> ValueRef;
        pub fn LLVMBuildIntToPtr(B: BuilderRef,
                                 Val: ValueRef,
                                 DestTy: TypeRef,
                                 Name: *c_char)
                                 -> ValueRef;
        pub fn LLVMBuildBitCast(B: BuilderRef,
                                Val: ValueRef,
                                DestTy: TypeRef,
                                Name: *c_char)
                                -> ValueRef;
        pub fn LLVMBuildZExtOrBitCast(B: BuilderRef,
                                      Val: ValueRef,
                                      DestTy: TypeRef,
                                      Name: *c_char)
                                      -> ValueRef;
        pub fn LLVMBuildSExtOrBitCast(B: BuilderRef,
                                      Val: ValueRef,
                                      DestTy: TypeRef,
                                      Name: *c_char)
                                      -> ValueRef;
        pub fn LLVMBuildTruncOrBitCast(B: BuilderRef,
                                       Val: ValueRef,
                                       DestTy: TypeRef,
                                       Name: *c_char)
                                       -> ValueRef;
        pub fn LLVMBuildCast(B: BuilderRef,
                             Op: Opcode,
                             Val: ValueRef,
                             DestTy: TypeRef,
                             Name: *c_char) -> ValueRef;
        pub fn LLVMBuildPointerCast(B: BuilderRef,
                                    Val: ValueRef,
                                    DestTy: TypeRef,
                                    Name: *c_char)
                                    -> ValueRef;
        pub fn LLVMBuildIntCast(B: BuilderRef,
                                Val: ValueRef,
                                DestTy: TypeRef,
                                Name: *c_char)
                                -> ValueRef;
        pub fn LLVMBuildFPCast(B: BuilderRef,
                               Val: ValueRef,
                               DestTy: TypeRef,
                               Name: *c_char)
                               -> ValueRef;

        /* Comparisons */
        pub fn LLVMBuildICmp(B: BuilderRef,
                             Op: c_uint,
                             LHS: ValueRef,
                             RHS: ValueRef,
                             Name: *c_char)
                             -> ValueRef;
        pub fn LLVMBuildFCmp(B: BuilderRef,
                             Op: c_uint,
                             LHS: ValueRef,
                             RHS: ValueRef,
                             Name: *c_char)
                             -> ValueRef;

        /* Miscellaneous instructions */
        pub fn LLVMBuildPhi(B: BuilderRef, Ty: TypeRef, Name: *c_char)
                            -> ValueRef;
        pub fn LLVMBuildCall(B: BuilderRef,
                             Fn: ValueRef,
                             Args: *ValueRef,
                             NumArgs: c_uint,
                             Name: *c_char)
                             -> ValueRef;
        pub fn LLVMBuildSelect(B: BuilderRef,
                               If: ValueRef,
                               Then: ValueRef,
                               Else: ValueRef,
                               Name: *c_char)
                               -> ValueRef;
        pub fn LLVMBuildVAArg(B: BuilderRef,
                              list: ValueRef,
                              Ty: TypeRef,
                              Name: *c_char)
                              -> ValueRef;
        pub fn LLVMBuildExtractElement(B: BuilderRef,
                                       VecVal: ValueRef,
                                       Index: ValueRef,
                                       Name: *c_char)
                                       -> ValueRef;
        pub fn LLVMBuildInsertElement(B: BuilderRef,
                                      VecVal: ValueRef,
                                      EltVal: ValueRef,
                                      Index: ValueRef,
                                      Name: *c_char)
                                      -> ValueRef;
        pub fn LLVMBuildShuffleVector(B: BuilderRef,
                                      V1: ValueRef,
                                      V2: ValueRef,
                                      Mask: ValueRef,
                                      Name: *c_char)
                                      -> ValueRef;
        pub fn LLVMBuildExtractValue(B: BuilderRef,
                                     AggVal: ValueRef,
                                     Index: c_uint,
                                     Name: *c_char)
                                     -> ValueRef;
        pub fn LLVMBuildInsertValue(B: BuilderRef,
                                    AggVal: ValueRef,
                                    EltVal: ValueRef,
                                    Index: c_uint,
                                    Name: *c_char)
                                    -> ValueRef;

        pub fn LLVMBuildIsNull(B: BuilderRef, Val: ValueRef, Name: *c_char)
                               -> ValueRef;
        pub fn LLVMBuildIsNotNull(B: BuilderRef, Val: ValueRef, Name: *c_char)
                                  -> ValueRef;
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
                                  Order: AtomicOrdering,
                                  SingleThreaded: Bool)
                                  -> ValueRef;

        pub fn LLVMBuildAtomicFence(B: BuilderRef, Order: AtomicOrdering);


        /* Selected entries from the downcasts. */
        pub fn LLVMIsATerminatorInst(Inst: ValueRef) -> ValueRef;
        pub fn LLVMIsAStoreInst(Inst: ValueRef) -> ValueRef;

        /** Writes a module to the specified path. Returns 0 on success. */
        pub fn LLVMWriteBitcodeToFile(M: ModuleRef, Path: *c_char) -> c_int;

        /** Creates target data from a target layout string. */
        pub fn LLVMCreateTargetData(StringRep: *c_char) -> TargetDataRef;
        /// Adds the target data to the given pass manager. The pass manager
        /// references the target data only weakly.
        pub fn LLVMAddTargetData(TD: TargetDataRef, PM: PassManagerRef);
        /** Number of bytes clobbered when doing a Store to *T. */
        pub fn LLVMStoreSizeOfType(TD: TargetDataRef, Ty: TypeRef)
                                   -> c_ulonglong;

        /** Number of bytes clobbered when doing a Store to *T. */
        pub fn LLVMSizeOfTypeInBits(TD: TargetDataRef, Ty: TypeRef)
                                    -> c_ulonglong;

        /** Distance between successive elements in an array of T.
        Includes ABI padding. */
        pub fn LLVMABISizeOfType(TD: TargetDataRef, Ty: TypeRef) -> c_uint;

        /** Returns the preferred alignment of a type. */
        pub fn LLVMPreferredAlignmentOfType(TD: TargetDataRef, Ty: TypeRef)
                                            -> c_uint;
        /** Returns the minimum alignment of a type. */
        pub fn LLVMABIAlignmentOfType(TD: TargetDataRef, Ty: TypeRef)
                                      -> c_uint;

        /// Computes the byte offset of the indexed struct element for a
        /// target.
        pub fn LLVMOffsetOfElement(TD: TargetDataRef,
                                   StructTy: TypeRef,
                                   Element: c_uint)
                                   -> c_ulonglong;

        /**
         * Returns the minimum alignment of a type when part of a call frame.
         */
        pub fn LLVMCallFrameAlignmentOfType(TD: TargetDataRef, Ty: TypeRef)
                                            -> c_uint;

        /** Disposes target data. */
        pub fn LLVMDisposeTargetData(TD: TargetDataRef);

        /** Creates a pass manager. */
        pub fn LLVMCreatePassManager() -> PassManagerRef;

        /** Creates a function-by-function pass manager */
        pub fn LLVMCreateFunctionPassManagerForModule(M: ModuleRef)
                                                      -> PassManagerRef;

        /** Disposes a pass manager. */
        pub fn LLVMDisposePassManager(PM: PassManagerRef);

        /** Runs a pass manager on a module. */
        pub fn LLVMRunPassManager(PM: PassManagerRef, M: ModuleRef) -> Bool;

        /** Runs the function passes on the provided function. */
        pub fn LLVMRunFunctionPassManager(FPM: PassManagerRef, F: ValueRef)
                                          -> Bool;

        /** Initializes all the function passes scheduled in the manager */
        pub fn LLVMInitializeFunctionPassManager(FPM: PassManagerRef) -> Bool;

        /** Finalizes all the function passes scheduled in the manager */
        pub fn LLVMFinalizeFunctionPassManager(FPM: PassManagerRef) -> Bool;

        pub fn LLVMInitializePasses();

        /** Adds a verification pass. */
        pub fn LLVMAddVerifierPass(PM: PassManagerRef);

        pub fn LLVMAddGlobalOptimizerPass(PM: PassManagerRef);
        pub fn LLVMAddIPSCCPPass(PM: PassManagerRef);
        pub fn LLVMAddDeadArgEliminationPass(PM: PassManagerRef);
        pub fn LLVMAddInstructionCombiningPass(PM: PassManagerRef);
        pub fn LLVMAddCFGSimplificationPass(PM: PassManagerRef);
        pub fn LLVMAddFunctionInliningPass(PM: PassManagerRef);
        pub fn LLVMAddFunctionAttrsPass(PM: PassManagerRef);
        pub fn LLVMAddScalarReplAggregatesPass(PM: PassManagerRef);
        pub fn LLVMAddScalarReplAggregatesPassSSA(PM: PassManagerRef);
        pub fn LLVMAddJumpThreadingPass(PM: PassManagerRef);
        pub fn LLVMAddConstantPropagationPass(PM: PassManagerRef);
        pub fn LLVMAddReassociatePass(PM: PassManagerRef);
        pub fn LLVMAddLoopRotatePass(PM: PassManagerRef);
        pub fn LLVMAddLICMPass(PM: PassManagerRef);
        pub fn LLVMAddLoopUnswitchPass(PM: PassManagerRef);
        pub fn LLVMAddLoopDeletionPass(PM: PassManagerRef);
        pub fn LLVMAddLoopUnrollPass(PM: PassManagerRef);
        pub fn LLVMAddGVNPass(PM: PassManagerRef);
        pub fn LLVMAddMemCpyOptPass(PM: PassManagerRef);
        pub fn LLVMAddSCCPPass(PM: PassManagerRef);
        pub fn LLVMAddDeadStoreEliminationPass(PM: PassManagerRef);
        pub fn LLVMAddStripDeadPrototypesPass(PM: PassManagerRef);
        pub fn LLVMAddConstantMergePass(PM: PassManagerRef);
        pub fn LLVMAddArgumentPromotionPass(PM: PassManagerRef);
        pub fn LLVMAddTailCallEliminationPass(PM: PassManagerRef);
        pub fn LLVMAddIndVarSimplifyPass(PM: PassManagerRef);
        pub fn LLVMAddAggressiveDCEPass(PM: PassManagerRef);
        pub fn LLVMAddGlobalDCEPass(PM: PassManagerRef);
        pub fn LLVMAddCorrelatedValuePropagationPass(PM: PassManagerRef);
        pub fn LLVMAddPruneEHPass(PM: PassManagerRef);
        pub fn LLVMAddSimplifyLibCallsPass(PM: PassManagerRef);
        pub fn LLVMAddLoopIdiomPass(PM: PassManagerRef);
        pub fn LLVMAddEarlyCSEPass(PM: PassManagerRef);
        pub fn LLVMAddTypeBasedAliasAnalysisPass(PM: PassManagerRef);
        pub fn LLVMAddBasicAliasAnalysisPass(PM: PassManagerRef);

        pub fn LLVMPassManagerBuilderCreate() -> PassManagerBuilderRef;
        pub fn LLVMPassManagerBuilderDispose(PMB: PassManagerBuilderRef);
        pub fn LLVMPassManagerBuilderSetOptLevel(PMB: PassManagerBuilderRef,
                                                 OptimizationLevel: c_uint);
        pub fn LLVMPassManagerBuilderSetSizeLevel(PMB: PassManagerBuilderRef,
                                                  Value: Bool);
        pub fn LLVMPassManagerBuilderSetDisableUnitAtATime(
            PMB: PassManagerBuilderRef,
            Value: Bool);
        pub fn LLVMPassManagerBuilderSetDisableUnrollLoops(
            PMB: PassManagerBuilderRef,
            Value: Bool);
        pub fn LLVMPassManagerBuilderSetDisableSimplifyLibCalls(
            PMB: PassManagerBuilderRef,
            Value: Bool);
        pub fn LLVMPassManagerBuilderUseInlinerWithThreshold(
            PMB: PassManagerBuilderRef,
            threshold: c_uint);
        pub fn LLVMPassManagerBuilderPopulateModulePassManager(
            PMB: PassManagerBuilderRef,
            PM: PassManagerRef);

        pub fn LLVMPassManagerBuilderPopulateFunctionPassManager(
            PMB: PassManagerBuilderRef,
            PM: PassManagerRef);
        pub fn LLVMPassManagerBuilderPopulateLTOPassManager(
            PMB: PassManagerBuilderRef,
            PM: PassManagerRef,
            Internalize: Bool,
            RunInliner: Bool);

        /** Destroys a memory buffer. */
        pub fn LLVMDisposeMemoryBuffer(MemBuf: MemoryBufferRef);


        /* Stuff that's in rustllvm/ because it's not upstream yet. */

        /** Opens an object file. */
        pub fn LLVMCreateObjectFile(MemBuf: MemoryBufferRef) -> ObjectFileRef;
        /** Closes an object file. */
        pub fn LLVMDisposeObjectFile(ObjFile: ObjectFileRef);

        /** Enumerates the sections in an object file. */
        pub fn LLVMGetSections(ObjFile: ObjectFileRef) -> SectionIteratorRef;
        /** Destroys a section iterator. */
        pub fn LLVMDisposeSectionIterator(SI: SectionIteratorRef);
        /** Returns true if the section iterator is at the end of the section
            list: */
        pub fn LLVMIsSectionIteratorAtEnd(ObjFile: ObjectFileRef,
                                          SI: SectionIteratorRef)
                                          -> Bool;
        /** Moves the section iterator to point to the next section. */
        pub fn LLVMMoveToNextSection(SI: SectionIteratorRef);
        /** Returns the current section name. */
        pub fn LLVMGetSectionName(SI: SectionIteratorRef) -> *c_char;
        /** Returns the current section size. */
        pub fn LLVMGetSectionSize(SI: SectionIteratorRef) -> c_ulonglong;
        /** Returns the current section contents as a string buffer. */
        pub fn LLVMGetSectionContents(SI: SectionIteratorRef) -> *c_char;

        /** Reads the given file and returns it as a memory buffer. Use
            LLVMDisposeMemoryBuffer() to get rid of it. */
        pub fn LLVMRustCreateMemoryBufferWithContentsOfFile(Path: *c_char)
            -> MemoryBufferRef;
        /** Borrows the contents of the memory buffer (doesn't copy it) */
        pub fn LLVMCreateMemoryBufferWithMemoryRange(InputData: *c_char,
                                                     InputDataLength: size_t,
                                                     BufferName: *c_char,
                                                     RequiresNull: Bool)
            -> MemoryBufferRef;
        pub fn LLVMCreateMemoryBufferWithMemoryRangeCopy(InputData: *c_char,
                                                         InputDataLength: size_t,
                                                         BufferName: *c_char)
            -> MemoryBufferRef;

        pub fn LLVMIsMultithreaded() -> Bool;
        pub fn LLVMStartMultithreaded() -> Bool;

        /** Returns a string describing the last error caused by an LLVMRust*
            call. */
        pub fn LLVMRustGetLastError() -> *c_char;

        /// Print the pass timings since static dtors aren't picking them up.
        pub fn LLVMRustPrintPassTimings();

        pub fn LLVMStructCreateNamed(C: ContextRef, Name: *c_char) -> TypeRef;

        pub fn LLVMStructSetBody(StructTy: TypeRef,
                                 ElementTypes: *TypeRef,
                                 ElementCount: c_uint,
                                 Packed: Bool);

        pub fn LLVMConstNamedStruct(S: TypeRef,
                                    ConstantVals: *ValueRef,
                                    Count: c_uint)
                                    -> ValueRef;

        /** Enables LLVM debug output. */
        pub fn LLVMSetDebug(Enabled: c_int);

        /** Prepares inline assembly. */
        pub fn LLVMInlineAsm(Ty: TypeRef,
                             AsmString: *c_char,
                             Constraints: *c_char,
                             SideEffects: Bool,
                             AlignStack: Bool,
                             Dialect: c_uint)
                             -> ValueRef;

        pub static LLVMRustDebugMetadataVersion: u32;

        pub fn LLVMRustAddModuleFlag(M: ModuleRef,
                                     name: *c_char,
                                     value: u32);

        pub fn LLVMDIBuilderCreate(M: ModuleRef) -> DIBuilderRef;

        pub fn LLVMDIBuilderDispose(Builder: DIBuilderRef);

        pub fn LLVMDIBuilderFinalize(Builder: DIBuilderRef);

        pub fn LLVMDIBuilderCreateCompileUnit(Builder: DIBuilderRef,
                                              Lang: c_uint,
                                              File: *c_char,
                                              Dir: *c_char,
                                              Producer: *c_char,
                                              isOptimized: bool,
                                              Flags: *c_char,
                                              RuntimeVer: c_uint,
                                              SplitName: *c_char);

        pub fn LLVMDIBuilderCreateFile(Builder: DIBuilderRef,
                                       Filename: *c_char,
                                       Directory: *c_char)
                                       -> DIFile;

        pub fn LLVMDIBuilderCreateSubroutineType(Builder: DIBuilderRef,
                                                 File: DIFile,
                                                 ParameterTypes: DIArray)
                                                 -> DICompositeType;

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

        pub fn LLVMDIBuilderCreateBasicType(Builder: DIBuilderRef,
                                            Name: *c_char,
                                            SizeInBits: c_ulonglong,
                                            AlignInBits: c_ulonglong,
                                            Encoding: c_uint)
                                            -> DIBasicType;

        pub fn LLVMDIBuilderCreatePointerType(Builder: DIBuilderRef,
                                              PointeeTy: DIType,
                                              SizeInBits: c_ulonglong,
                                              AlignInBits: c_ulonglong,
                                              Name: *c_char)
                                              -> DIDerivedType;

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
                                             VTableHolder: ValueRef,
                                             UniqueId: *c_char)
                                             -> DICompositeType;

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

        pub fn LLVMDIBuilderCreateLexicalBlock(Builder: DIBuilderRef,
                                               Scope: DIDescriptor,
                                               File: DIFile,
                                               Line: c_uint,
                                               Col: c_uint)
                                               -> DILexicalBlock;

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

        pub fn LLVMDIBuilderCreateArrayType(Builder: DIBuilderRef,
                                            Size: c_ulonglong,
                                            AlignInBits: c_ulonglong,
                                            Ty: DIType,
                                            Subscripts: DIArray)
                                            -> DIType;

        pub fn LLVMDIBuilderCreateVectorType(Builder: DIBuilderRef,
                                             Size: c_ulonglong,
                                             AlignInBits: c_ulonglong,
                                             Ty: DIType,
                                             Subscripts: DIArray)
                                             -> DIType;

        pub fn LLVMDIBuilderGetOrCreateSubrange(Builder: DIBuilderRef,
                                                Lo: c_longlong,
                                                Count: c_longlong)
                                                -> DISubrange;

        pub fn LLVMDIBuilderGetOrCreateArray(Builder: DIBuilderRef,
                                             Ptr: *DIDescriptor,
                                             Count: c_uint)
                                             -> DIArray;

        pub fn LLVMDIBuilderInsertDeclareAtEnd(Builder: DIBuilderRef,
                                               Val: ValueRef,
                                               VarInfo: DIVariable,
                                               InsertAtEnd: BasicBlockRef)
                                               -> ValueRef;

        pub fn LLVMDIBuilderInsertDeclareBefore(Builder: DIBuilderRef,
                                                Val: ValueRef,
                                                VarInfo: DIVariable,
                                                InsertBefore: ValueRef)
                                                -> ValueRef;

        pub fn LLVMDIBuilderCreateEnumerator(Builder: DIBuilderRef,
                                             Name: *c_char,
                                             Val: c_ulonglong)
                                             -> ValueRef;

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

        pub fn LLVMDIBuilderCreateUnionType(Builder: DIBuilderRef,
                                            Scope: ValueRef,
                                            Name: *c_char,
                                            File: ValueRef,
                                            LineNumber: c_uint,
                                            SizeInBits: c_ulonglong,
                                            AlignInBits: c_ulonglong,
                                            Flags: c_uint,
                                            Elements: ValueRef,
                                            RunTimeLang: c_uint,
                                            UniqueId: *c_char)
                                            -> ValueRef;

        pub fn LLVMSetUnnamedAddr(GlobalVar: ValueRef, UnnamedAddr: Bool);

        pub fn LLVMDIBuilderCreateTemplateTypeParameter(Builder: DIBuilderRef,
                                                        Scope: ValueRef,
                                                        Name: *c_char,
                                                        Ty: ValueRef,
                                                        File: ValueRef,
                                                        LineNo: c_uint,
                                                        ColumnNo: c_uint)
                                                        -> ValueRef;

        pub fn LLVMDIBuilderCreateOpDeref(IntType: TypeRef) -> ValueRef;

        pub fn LLVMDIBuilderCreateOpPlus(IntType: TypeRef) -> ValueRef;

        pub fn LLVMDIBuilderCreateComplexVariable(Builder: DIBuilderRef,
            Tag: c_uint,
            Scope: ValueRef,
            Name: *c_char,
            File: ValueRef,
            LineNo: c_uint,
            Ty: ValueRef,
            AddrOps: *ValueRef,
            AddrOpsCount: c_uint,
            ArgNo: c_uint)
            -> ValueRef;

        pub fn LLVMDIBuilderCreateNameSpace(Builder: DIBuilderRef,
                                            Scope: ValueRef,
                                            Name: *c_char,
                                            File: ValueRef,
                                            LineNo: c_uint)
                                            -> ValueRef;

        pub fn LLVMDICompositeTypeSetTypeArray(CompositeType: ValueRef, TypeArray: ValueRef);
        pub fn LLVMTypeToString(Type: TypeRef) -> *c_char;
        pub fn LLVMValueToString(value_ref: ValueRef) -> *c_char;

        pub fn LLVMIsAArgument(value_ref: ValueRef) -> ValueRef;

        pub fn LLVMIsAAllocaInst(value_ref: ValueRef) -> ValueRef;

        pub fn LLVMInitializeX86TargetInfo();
        pub fn LLVMInitializeX86Target();
        pub fn LLVMInitializeX86TargetMC();
        pub fn LLVMInitializeX86AsmPrinter();
        pub fn LLVMInitializeX86AsmParser();
        pub fn LLVMInitializeARMTargetInfo();
        pub fn LLVMInitializeARMTarget();
        pub fn LLVMInitializeARMTargetMC();
        pub fn LLVMInitializeARMAsmPrinter();
        pub fn LLVMInitializeARMAsmParser();
        pub fn LLVMInitializeMipsTargetInfo();
        pub fn LLVMInitializeMipsTarget();
        pub fn LLVMInitializeMipsTargetMC();
        pub fn LLVMInitializeMipsAsmPrinter();
        pub fn LLVMInitializeMipsAsmParser();

        pub fn LLVMRustAddPass(PM: PassManagerRef, Pass: *c_char) -> bool;
        pub fn LLVMRustCreateTargetMachine(Triple: *c_char,
                                           CPU: *c_char,
                                           Features: *c_char,
                                           Model: CodeGenModel,
                                           Reloc: RelocMode,
                                           Level: CodeGenOptLevel,
                                           EnableSegstk: bool,
                                           UseSoftFP: bool,
                                           NoFramePointerElim: bool) -> TargetMachineRef;
        pub fn LLVMRustDisposeTargetMachine(T: TargetMachineRef);
        pub fn LLVMRustAddAnalysisPasses(T: TargetMachineRef,
                                         PM: PassManagerRef,
                                         M: ModuleRef);
        pub fn LLVMRustAddBuilderLibraryInfo(PMB: PassManagerBuilderRef,
                                             M: ModuleRef);
        pub fn LLVMRustAddLibraryInfo(PM: PassManagerRef, M: ModuleRef);
        pub fn LLVMRustRunFunctionPassManager(PM: PassManagerRef, M: ModuleRef);
        pub fn LLVMRustWriteOutputFile(T: TargetMachineRef,
                                       PM: PassManagerRef,
                                       M: ModuleRef,
                                       Output: *c_char,
                                       FileType: FileType) -> bool;
        pub fn LLVMRustPrintModule(PM: PassManagerRef,
                                   M: ModuleRef,
                                   Output: *c_char);
        pub fn LLVMRustSetLLVMOptions(Argc: c_int, Argv: **c_char);
        pub fn LLVMRustPrintPasses();
        pub fn LLVMRustSetNormalizedTarget(M: ModuleRef, triple: *c_char);
        pub fn LLVMRustAddAlwaysInlinePass(P: PassManagerBuilderRef,
                                           AddLifetimes: bool);
        pub fn LLVMRustLinkInExternalBitcode(M: ModuleRef,
                                             bc: *c_char,
                                             len: size_t) -> bool;
        pub fn LLVMRustRunRestrictionPass(M: ModuleRef,
                                          syms: **c_char,
                                          len: size_t);
        pub fn LLVMRustMarkAllFunctionsNounwind(M: ModuleRef);

        pub fn LLVMRustOpenArchive(path: *c_char) -> ArchiveRef;
        pub fn LLVMRustArchiveReadSection(AR: ArchiveRef, name: *c_char,
                                          out_len: *mut size_t) -> *c_char;
        pub fn LLVMRustDestroyArchive(AR: ArchiveRef);

        pub fn LLVMRustSetDLLExportStorageClass(V: ValueRef);
        pub fn LLVMVersionMinor() -> c_int;
    }
}

pub fn SetInstructionCallConv(instr: ValueRef, cc: CallConv) {
    unsafe {
        llvm::LLVMSetInstructionCallConv(instr, cc as c_uint);
    }
}
pub fn SetFunctionCallConv(fn_: ValueRef, cc: CallConv) {
    unsafe {
        llvm::LLVMSetFunctionCallConv(fn_, cc as c_uint);
    }
}
pub fn SetLinkage(global: ValueRef, link: Linkage) {
    unsafe {
        llvm::LLVMSetLinkage(global, link as c_uint);
    }
}

pub fn SetUnnamedAddr(global: ValueRef, unnamed: bool) {
    unsafe {
        llvm::LLVMSetUnnamedAddr(global, unnamed as Bool);
    }
}

pub fn set_thread_local(global: ValueRef, is_thread_local: bool) {
    unsafe {
        llvm::LLVMSetThreadLocal(global, is_thread_local as Bool);
    }
}

pub fn ConstICmp(pred: IntPredicate, v1: ValueRef, v2: ValueRef) -> ValueRef {
    unsafe {
        llvm::LLVMConstICmp(pred as c_ushort, v1, v2)
    }
}
pub fn ConstFCmp(pred: RealPredicate, v1: ValueRef, v2: ValueRef) -> ValueRef {
    unsafe {
        llvm::LLVMConstFCmp(pred as c_ushort, v1, v2)
    }
}

pub fn SetFunctionAttribute(fn_: ValueRef, attr: Attribute) {
    unsafe {
        llvm::LLVMAddFunctionAttr(fn_, attr as c_uint)
    }
}
/* Memory-managed object interface to type handles. */

pub struct TypeNames {
    named_types: RefCell<HashMap<~str, TypeRef>>,
}

impl TypeNames {
    pub fn new() -> TypeNames {
        TypeNames {
            named_types: RefCell::new(HashMap::new())
        }
    }

    pub fn associate_type(&self, s: &str, t: &Type) {
        assert!(self.named_types.borrow_mut().insert(s.to_owned(), t.to_ref()));
    }

    pub fn find_type(&self, s: &str) -> Option<Type> {
        self.named_types.borrow().find_equiv(&s).map(|x| Type::from_ref(*x))
    }

    pub fn type_to_str(&self, ty: Type) -> ~str {
        unsafe {
            let s = llvm::LLVMTypeToString(ty.to_ref());
            let ret = from_c_str(s);
            free(s as *mut c_void);
            ret
        }
    }

    pub fn types_to_str(&self, tys: &[Type]) -> ~str {
        let strs = tys.map(|t| self.type_to_str(*t));
        format!("[{}]", strs.connect(","))
    }

    pub fn val_to_str(&self, val: ValueRef) -> ~str {
        unsafe {
            let s = llvm::LLVMValueToString(val);
            let ret = from_c_str(s);
            free(s as *mut c_void);
            ret
        }
    }
}

/* Memory-managed interface to target data. */

pub struct target_data_res {
    td: TargetDataRef,
}

impl Drop for target_data_res {
    fn drop(&mut self) {
        unsafe {
            llvm::LLVMDisposeTargetData(self.td);
        }
    }
}

pub fn target_data_res(td: TargetDataRef) -> target_data_res {
    target_data_res {
        td: td
    }
}

pub struct TargetData {
    lltd: TargetDataRef,
    dtor: @target_data_res
}

pub fn mk_target_data(string_rep: &str) -> TargetData {
    let lltd = string_rep.with_c_str(|buf| {
        unsafe { llvm::LLVMCreateTargetData(buf) }
    });

    TargetData {
        lltd: lltd,
        dtor: @target_data_res(lltd)
    }
}

/* Memory-managed interface to pass managers. */

pub struct pass_manager_res {
    pm: PassManagerRef,
}

impl Drop for pass_manager_res {
    fn drop(&mut self) {
        unsafe {
            llvm::LLVMDisposePassManager(self.pm);
        }
    }
}

pub fn pass_manager_res(pm: PassManagerRef) -> pass_manager_res {
    pass_manager_res {
        pm: pm
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

pub struct ObjectFile {
    llof: ObjectFileRef,
}

impl ObjectFile {
    // This will take ownership of llmb
    pub fn new(llmb: MemoryBufferRef) -> Option<ObjectFile> {
        unsafe {
            let llof = llvm::LLVMCreateObjectFile(llmb);
            if llof as int == 0 {
                // LLVMCreateObjectFile took ownership of llmb
                return None
            }

            Some(ObjectFile {
                llof: llof,
            })
        }
    }
}

impl Drop for ObjectFile {
    fn drop(&mut self) {
        unsafe {
            llvm::LLVMDisposeObjectFile(self.llof);
        }
    }
}

/* Memory-managed interface to section iterators. */

pub struct section_iter_res {
    si: SectionIteratorRef,
}

impl Drop for section_iter_res {
    fn drop(&mut self) {
        unsafe {
            llvm::LLVMDisposeSectionIterator(self.si);
        }
    }
}

pub fn section_iter_res(si: SectionIteratorRef) -> section_iter_res {
    section_iter_res {
        si: si
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
