// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// FIXME: Rename 'DIGlobalVariable' to 'DIGlobalVariableExpression'
// once support for LLVM 3.9 is dropped.
//
// This method was changed in this LLVM patch:
// https://reviews.llvm.org/D26769

use super::debuginfo::{
    DIBuilder, DIDescriptor_opaque, DIDescriptor, DIFile, DILexicalBlock, DISubprogram, DIType_opaque,
    DIType, DIBasicType, DIDerivedType, DICompositeType, DIScope_opaque, DIScope, DIVariable,
    DIGlobalVariable, DIArray_opaque, DIArray, DISubrange, DITemplateTypeParameter, DIEnumerator,
    DINameSpace, DIFlags,
};

use libc::{c_uint, c_int, size_t, c_char};
use libc::{c_longlong, c_ulonglong, c_void};

use std::ptr::NonNull;

use super::RustStringRef;

pub type Opcode = u32;
pub type Bool = c_uint;

pub const True: Bool = 1 as Bool;
pub const False: Bool = 0 as Bool;

#[derive(Copy, Clone, PartialEq)]
#[repr(C)]
pub enum LLVMRustResult {
    Success,
    Failure,
}
// Consts for the LLVM CallConv type, pre-cast to usize.

/// LLVM CallingConv::ID. Should we wrap this?
#[derive(Copy, Clone, PartialEq, Debug)]
#[repr(C)]
pub enum CallConv {
    CCallConv = 0,
    FastCallConv = 8,
    ColdCallConv = 9,
    X86StdcallCallConv = 64,
    X86FastcallCallConv = 65,
    ArmAapcsCallConv = 67,
    Msp430Intr = 69,
    X86_ThisCall = 70,
    PtxKernel = 71,
    X86_64_SysV = 78,
    X86_64_Win64 = 79,
    X86_VectorCall = 80,
    X86_Intr = 83,
    AmdGpuKernel = 91,
}

/// LLVMRustLinkage
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
#[repr(C)]
pub enum Linkage {
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
}

// LLVMRustVisibility
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
#[repr(C)]
pub enum Visibility {
    Default = 0,
    Hidden = 1,
    Protected = 2,
}

/// LLVMDiagnosticSeverity
#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub enum DiagnosticSeverity {
    Error = 0,
    Warning = 1,
    Remark = 2,
    Note = 3,
}

/// LLVMDLLStorageClass
#[derive(Copy, Clone)]
#[repr(C)]
pub enum DLLStorageClass {
    Default = 0,
    DllImport = 1, // Function to be imported from DLL.
    DllExport = 2, // Function to be accessible from DLL.
}

/// Matches LLVMRustAttribute in rustllvm.h
/// Semantically a subset of the C++ enum llvm::Attribute::AttrKind,
/// though it is not ABI compatible (since it's a C++ enum)
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub enum Attribute {
    AlwaysInline    = 0,
    ByVal           = 1,
    Cold            = 2,
    InlineHint      = 3,
    MinSize         = 4,
    Naked           = 5,
    NoAlias         = 6,
    NoCapture       = 7,
    NoInline        = 8,
    NonNull         = 9,
    NoRedZone       = 10,
    NoReturn        = 11,
    NoUnwind        = 12,
    OptimizeForSize = 13,
    ReadOnly        = 14,
    SExt            = 15,
    StructRet       = 16,
    UWTable         = 17,
    ZExt            = 18,
    InReg           = 19,
    SanitizeThread  = 20,
    SanitizeAddress = 21,
    SanitizeMemory  = 22,
}

/// LLVMIntPredicate
#[derive(Copy, Clone)]
#[repr(C)]
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

/// LLVMRealPredicate
#[derive(Copy, Clone)]
#[repr(C)]
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

/// LLVMTypeKind
#[derive(Copy, Clone, PartialEq, Debug)]
#[repr(C)]
pub enum TypeKind {
    Void = 0,
    Half = 1,
    Float = 2,
    Double = 3,
    X86_FP80 = 4,
    FP128 = 5,
    PPC_FP128 = 6,
    Label = 7,
    Integer = 8,
    Function = 9,
    Struct = 10,
    Array = 11,
    Pointer = 12,
    Vector = 13,
    Metadata = 14,
    X86_MMX = 15,
    Token = 16,
}

/// LLVMAtomicRmwBinOp
#[derive(Copy, Clone)]
#[repr(C)]
pub enum AtomicRmwBinOp {
    AtomicXchg = 0,
    AtomicAdd = 1,
    AtomicSub = 2,
    AtomicAnd = 3,
    AtomicNand = 4,
    AtomicOr = 5,
    AtomicXor = 6,
    AtomicMax = 7,
    AtomicMin = 8,
    AtomicUMax = 9,
    AtomicUMin = 10,
}

/// LLVMAtomicOrdering
#[derive(Copy, Clone)]
#[repr(C)]
pub enum AtomicOrdering {
    NotAtomic = 0,
    Unordered = 1,
    Monotonic = 2,
    // Consume = 3,  // Not specified yet.
    Acquire = 4,
    Release = 5,
    AcquireRelease = 6,
    SequentiallyConsistent = 7,
}

/// LLVMRustSynchronizationScope
#[derive(Copy, Clone)]
#[repr(C)]
pub enum SynchronizationScope {
    Other,
    SingleThread,
    CrossThread,
}

/// LLVMRustFileType
#[derive(Copy, Clone)]
#[repr(C)]
pub enum FileType {
    Other,
    AssemblyFile,
    ObjectFile,
}

/// LLVMMetadataType
#[derive(Copy, Clone)]
#[repr(C)]
pub enum MetadataType {
    MD_dbg = 0,
    MD_tbaa = 1,
    MD_prof = 2,
    MD_fpmath = 3,
    MD_range = 4,
    MD_tbaa_struct = 5,
    MD_invariant_load = 6,
    MD_alias_scope = 7,
    MD_noalias = 8,
    MD_nontemporal = 9,
    MD_mem_parallel_loop_access = 10,
    MD_nonnull = 11,
}

/// LLVMRustAsmDialect
#[derive(Copy, Clone)]
#[repr(C)]
pub enum AsmDialect {
    Other,
    Att,
    Intel,
}

/// LLVMRustCodeGenOptLevel
#[derive(Copy, Clone, PartialEq)]
#[repr(C)]
pub enum CodeGenOptLevel {
    Other,
    None,
    Less,
    Default,
    Aggressive,
}

/// LLVMRelocMode
#[derive(Copy, Clone, PartialEq)]
#[repr(C)]
pub enum RelocMode {
    Default,
    Static,
    PIC,
    DynamicNoPic,
    ROPI,
    RWPI,
    ROPI_RWPI,
}

/// LLVMRustCodeModel
#[derive(Copy, Clone)]
#[repr(C)]
pub enum CodeModel {
    Other,
    Small,
    Kernel,
    Medium,
    Large,
    None,
}

/// LLVMRustDiagnosticKind
#[derive(Copy, Clone)]
#[repr(C)]
pub enum DiagnosticKind {
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
}

/// LLVMRustArchiveKind
#[derive(Copy, Clone)]
#[repr(C)]
pub enum ArchiveKind {
    Other,
    K_GNU,
    K_BSD,
    K_COFF,
}

/// LLVMRustPassKind
#[derive(Copy, Clone, PartialEq, Debug)]
#[repr(C)]
pub enum PassKind {
    Other,
    Function,
    Module,
}

/// LLVMRustThinLTOData
extern { pub type ThinLTOData; }

/// LLVMRustThinLTOBuffer
extern { pub type ThinLTOBuffer; }

/// LLVMRustThinLTOModule
#[repr(C)]
pub struct ThinLTOModule {
    pub identifier: *const c_char,
    pub data: *const u8,
    pub len: usize,
}

/// LLVMThreadLocalMode
#[derive(Copy, Clone)]
#[repr(C)]
pub enum ThreadLocalMode {
  NotThreadLocal,
  GeneralDynamic,
  LocalDynamic,
  InitialExec,
  LocalExec
}

// Opaque pointer types
extern { pub type Module; }
extern { pub type Context; }
extern { pub type Type; }
extern { pub type Value_opaque; }
pub type ValueRef = *mut Value_opaque;
extern { pub type Metadata_opaque; }
pub type MetadataRef = *mut Metadata_opaque;
extern { pub type BasicBlock_opaque; }
pub type BasicBlockRef = *mut BasicBlock_opaque;
extern { pub type Builder; }
extern { pub type ExecutionEngine_opaque; }
pub type ExecutionEngineRef = *mut ExecutionEngine_opaque;
extern { pub type MemoryBuffer_opaque; }
pub type MemoryBufferRef = *mut MemoryBuffer_opaque;
extern { pub type PassManager_opaque; }
pub type PassManagerRef = *mut PassManager_opaque;
extern { pub type PassManagerBuilder_opaque; }
pub type PassManagerBuilderRef = *mut PassManagerBuilder_opaque;
extern { pub type Use_opaque; }
pub type UseRef = *mut Use_opaque;
extern { pub type TargetData_opaque; }
pub type TargetDataRef = *mut TargetData_opaque;
extern { pub type ObjectFile_opaque; }
pub type ObjectFileRef = *mut ObjectFile_opaque;
extern { pub type SectionIterator_opaque; }
pub type SectionIteratorRef = *mut SectionIterator_opaque;
extern { pub type Pass_opaque; }
pub type PassRef = *mut Pass_opaque;
extern { pub type TargetMachine; }
pub type TargetMachineRef = *const TargetMachine;
extern { pub type Archive_opaque; }
pub type ArchiveRef = *mut Archive_opaque;
extern { pub type ArchiveIterator_opaque; }
pub type ArchiveIteratorRef = *mut ArchiveIterator_opaque;
extern { pub type ArchiveChild_opaque; }
pub type ArchiveChildRef = *mut ArchiveChild_opaque;
extern { pub type Twine_opaque; }
pub type TwineRef = *mut Twine_opaque;
extern { pub type DiagnosticInfo_opaque; }
pub type DiagnosticInfoRef = *mut DiagnosticInfo_opaque;
extern { pub type DebugLoc_opaque; }
pub type DebugLocRef = *mut DebugLoc_opaque;
extern { pub type SMDiagnostic_opaque; }
pub type SMDiagnosticRef = *mut SMDiagnostic_opaque;
extern { pub type RustArchiveMember_opaque; }
pub type RustArchiveMemberRef = *mut RustArchiveMember_opaque;
extern { pub type OperandBundleDef_opaque; }
pub type OperandBundleDefRef = *mut OperandBundleDef_opaque;
extern { pub type Linker_opaque; }
pub type LinkerRef = *mut Linker_opaque;

pub type DiagnosticHandler = unsafe extern "C" fn(DiagnosticInfoRef, *mut c_void);
pub type InlineAsmDiagHandler = unsafe extern "C" fn(SMDiagnosticRef, *const c_void, c_uint);


pub mod debuginfo {
    use super::Metadata_opaque;

    extern { pub type DIBuilder; }

    pub type DIDescriptor_opaque = Metadata_opaque;
    pub type DIDescriptor = *mut DIDescriptor_opaque;
    pub type DIScope_opaque = DIDescriptor_opaque;
    pub type DIScope = *mut DIScope_opaque;
    pub type DILocation = DIDescriptor;
    pub type DIFile = DIScope;
    pub type DILexicalBlock = DIScope;
    pub type DISubprogram = DIScope;
    pub type DINameSpace = DIScope;
    pub type DIType_opaque = DIDescriptor_opaque;
    pub type DIType = *mut DIType_opaque;
    pub type DIBasicType = DIType;
    pub type DIDerivedType = DIType;
    pub type DICompositeType = DIDerivedType;
    pub type DIVariable = DIDescriptor;
    pub type DIGlobalVariable = DIDescriptor;
    pub type DIArray_opaque = DIDescriptor_opaque;
    pub type DIArray = *mut DIArray_opaque;
    pub type DISubrange = DIDescriptor;
    pub type DIEnumerator = DIDescriptor;
    pub type DITemplateTypeParameter = DIDescriptor;

    // These values **must** match with LLVMRustDIFlags!!
    bitflags! {
        #[repr(C)]
        #[derive(Default)]
        pub struct DIFlags: ::libc::uint32_t {
            const FlagZero                = 0;
            const FlagPrivate             = 1;
            const FlagProtected           = 2;
            const FlagPublic              = 3;
            const FlagFwdDecl             = (1 << 2);
            const FlagAppleBlock          = (1 << 3);
            const FlagBlockByrefStruct    = (1 << 4);
            const FlagVirtual             = (1 << 5);
            const FlagArtificial          = (1 << 6);
            const FlagExplicit            = (1 << 7);
            const FlagPrototyped          = (1 << 8);
            const FlagObjcClassComplete   = (1 << 9);
            const FlagObjectPointer       = (1 << 10);
            const FlagVector              = (1 << 11);
            const FlagStaticMember        = (1 << 12);
            const FlagLValueReference     = (1 << 13);
            const FlagRValueReference     = (1 << 14);
            const FlagExternalTypeRef     = (1 << 15);
            const FlagIntroducedVirtual   = (1 << 18);
            const FlagBitField            = (1 << 19);
            const FlagNoReturn            = (1 << 20);
            const FlagMainSubprogram      = (1 << 21);
        }
    }
}

extern { pub type ModuleBuffer; }

#[allow(improper_ctypes)] // TODO remove this (use for NonNull)
extern "C" {
    // Create and destroy contexts.
    pub fn LLVMRustContextCreate(shouldDiscardNames: bool) -> &'static mut Context;
    pub fn LLVMContextDispose(C: &'static mut Context);
    pub fn LLVMGetMDKindIDInContext(C: &Context, Name: *const c_char, SLen: c_uint) -> c_uint;

    // Create modules.
    pub fn LLVMModuleCreateWithNameInContext(ModuleID: *const c_char, C: &Context) -> &Module;
    pub fn LLVMGetModuleContext(M: &Module) -> &Context;
    pub fn LLVMCloneModule(M: &Module) -> &Module;

    /// Data layout. See Module::getDataLayout.
    pub fn LLVMGetDataLayout(M: &Module) -> *const c_char;
    pub fn LLVMSetDataLayout(M: &Module, Triple: *const c_char);

    /// See Module::dump.
    pub fn LLVMDumpModule(M: &Module);

    /// See Module::setModuleInlineAsm.
    pub fn LLVMSetModuleInlineAsm(M: &Module, Asm: *const c_char);
    pub fn LLVMRustAppendModuleInlineAsm(M: &Module, Asm: *const c_char);

    /// See llvm::LLVMTypeKind::getTypeID.
    pub fn LLVMRustGetTypeKind(Ty: &Type) -> TypeKind;

    // Operations on integer types
    pub fn LLVMInt1TypeInContext(C: &Context) -> &Type;
    pub fn LLVMInt8TypeInContext(C: &Context) -> &Type;
    pub fn LLVMInt16TypeInContext(C: &Context) -> &Type;
    pub fn LLVMInt32TypeInContext(C: &Context) -> &Type;
    pub fn LLVMInt64TypeInContext(C: &Context) -> &Type;
    pub fn LLVMIntTypeInContext(C: &Context, NumBits: c_uint) -> &Type;

    pub fn LLVMGetIntTypeWidth(IntegerTy: &Type) -> c_uint;

    // Operations on real types
    pub fn LLVMFloatTypeInContext(C: &Context) -> &Type;
    pub fn LLVMDoubleTypeInContext(C: &Context) -> &Type;

    // Operations on function types
    pub fn LLVMFunctionType(ReturnType: &'a Type,
                            ParamTypes: *const &'a Type,
                            ParamCount: c_uint,
                            IsVarArg: Bool)
                            -> &'a Type;
    pub fn LLVMGetReturnType(FunctionTy: &Type) -> &Type;
    pub fn LLVMCountParamTypes(FunctionTy: &Type) -> c_uint;
    pub fn LLVMGetParamTypes(FunctionTy: &'a Type, Dest: *mut &'a Type);

    // Operations on struct types
    pub fn LLVMStructTypeInContext(C: &'a Context,
                                   ElementTypes: *const &'a Type,
                                   ElementCount: c_uint,
                                   Packed: Bool)
                                   -> &'a Type;
    pub fn LLVMIsPackedStruct(StructTy: &Type) -> Bool;

    // Operations on array, pointer, and vector types (sequence types)
    pub fn LLVMRustArrayType(ElementType: &Type, ElementCount: u64) -> &Type;
    pub fn LLVMPointerType(ElementType: &Type, AddressSpace: c_uint) -> &Type;
    pub fn LLVMVectorType(ElementType: &Type, ElementCount: c_uint) -> &Type;

    pub fn LLVMGetElementType(Ty: &Type) -> &Type;
    pub fn LLVMGetVectorSize(VectorTy: &Type) -> c_uint;

    // Operations on other types
    pub fn LLVMVoidTypeInContext(C: &Context) -> &Type;
    pub fn LLVMX86MMXTypeInContext(C: &Context) -> &Type;
    pub fn LLVMRustMetadataTypeInContext(C: &Context) -> &Type;

    // Operations on all values
    pub fn LLVMTypeOf(Val: &Value_opaque) -> &Type;
    pub fn LLVMGetValueName(Val: ValueRef) -> *const c_char;
    pub fn LLVMSetValueName(Val: ValueRef, Name: *const c_char);
    pub fn LLVMReplaceAllUsesWith(OldVal: ValueRef, NewVal: ValueRef);
    pub fn LLVMSetMetadata(Val: ValueRef, KindID: c_uint, Node: ValueRef);

    // Operations on Uses
    pub fn LLVMGetFirstUse(Val: ValueRef) -> UseRef;
    pub fn LLVMGetNextUse(U: UseRef) -> UseRef;
    pub fn LLVMGetUser(U: UseRef) -> ValueRef;

    // Operations on Users
    pub fn LLVMGetOperand(Val: ValueRef, Index: c_uint) -> ValueRef;

    // Operations on constants of any type
    pub fn LLVMConstNull(Ty: &Type) -> ValueRef;
    pub fn LLVMConstICmp(Pred: IntPredicate, V1: ValueRef, V2: ValueRef) -> ValueRef;
    pub fn LLVMConstFCmp(Pred: RealPredicate, V1: ValueRef, V2: ValueRef) -> ValueRef;
    pub fn LLVMGetUndef(Ty: &Type) -> ValueRef;

    // Operations on metadata
    pub fn LLVMMDStringInContext(C: &Context, Str: *const c_char, SLen: c_uint) -> ValueRef;
    pub fn LLVMMDNodeInContext(C: &Context, Vals: *const ValueRef, Count: c_uint) -> ValueRef;
    pub fn LLVMAddNamedMetadataOperand(M: &Module, Name: *const c_char, Val: ValueRef);

    // Operations on scalar constants
    pub fn LLVMConstInt(IntTy: &Type, N: c_ulonglong, SignExtend: Bool) -> ValueRef;
    pub fn LLVMConstIntOfArbitraryPrecision(IntTy: &Type, Wn: c_uint, Ws: *const u64) -> ValueRef;
    pub fn LLVMConstIntGetZExtValue(ConstantVal: ValueRef) -> c_ulonglong;
    pub fn LLVMConstIntGetSExtValue(ConstantVal: ValueRef) -> c_longlong;
    pub fn LLVMRustConstInt128Get(ConstantVal: ValueRef, SExt: bool,
                                  high: *mut u64, low: *mut u64) -> bool;
    pub fn LLVMConstRealGetDouble (ConstantVal: ValueRef, losesInfo: *mut Bool) -> f64;


    // Operations on composite constants
    pub fn LLVMConstStringInContext(C: &Context,
                                    Str: *const c_char,
                                    Length: c_uint,
                                    DontNullTerminate: Bool)
                                    -> ValueRef;
    pub fn LLVMConstStructInContext(C: &Context,
                                    ConstantVals: *const ValueRef,
                                    Count: c_uint,
                                    Packed: Bool)
                                    -> ValueRef;

    pub fn LLVMConstArray(ElementTy: &Type,
                          ConstantVals: *const ValueRef,
                          Length: c_uint)
                          -> ValueRef;
    pub fn LLVMConstVector(ScalarConstantVals: *const ValueRef, Size: c_uint) -> ValueRef;

    // Constant expressions
    pub fn LLVMSizeOf(Ty: &Type) -> ValueRef;
    pub fn LLVMConstNeg(ConstantVal: ValueRef) -> ValueRef;
    pub fn LLVMConstFNeg(ConstantVal: ValueRef) -> ValueRef;
    pub fn LLVMConstNot(ConstantVal: ValueRef) -> ValueRef;
    pub fn LLVMConstAdd(LHSConstant: ValueRef, RHSConstant: ValueRef) -> ValueRef;
    pub fn LLVMConstFAdd(LHSConstant: ValueRef, RHSConstant: ValueRef) -> ValueRef;
    pub fn LLVMConstSub(LHSConstant: ValueRef, RHSConstant: ValueRef) -> ValueRef;
    pub fn LLVMConstFSub(LHSConstant: ValueRef, RHSConstant: ValueRef) -> ValueRef;
    pub fn LLVMConstMul(LHSConstant: ValueRef, RHSConstant: ValueRef) -> ValueRef;
    pub fn LLVMConstFMul(LHSConstant: ValueRef, RHSConstant: ValueRef) -> ValueRef;
    pub fn LLVMConstUDiv(LHSConstant: ValueRef, RHSConstant: ValueRef) -> ValueRef;
    pub fn LLVMConstSDiv(LHSConstant: ValueRef, RHSConstant: ValueRef) -> ValueRef;
    pub fn LLVMConstFDiv(LHSConstant: ValueRef, RHSConstant: ValueRef) -> ValueRef;
    pub fn LLVMConstURem(LHSConstant: ValueRef, RHSConstant: ValueRef) -> ValueRef;
    pub fn LLVMConstSRem(LHSConstant: ValueRef, RHSConstant: ValueRef) -> ValueRef;
    pub fn LLVMConstFRem(LHSConstant: ValueRef, RHSConstant: ValueRef) -> ValueRef;
    pub fn LLVMConstAnd(LHSConstant: ValueRef, RHSConstant: ValueRef) -> ValueRef;
    pub fn LLVMConstOr(LHSConstant: ValueRef, RHSConstant: ValueRef) -> ValueRef;
    pub fn LLVMConstXor(LHSConstant: ValueRef, RHSConstant: ValueRef) -> ValueRef;
    pub fn LLVMConstShl(LHSConstant: ValueRef, RHSConstant: ValueRef) -> ValueRef;
    pub fn LLVMConstLShr(LHSConstant: ValueRef, RHSConstant: ValueRef) -> ValueRef;
    pub fn LLVMConstAShr(LHSConstant: ValueRef, RHSConstant: ValueRef) -> ValueRef;
    pub fn LLVMConstGEP(
        ConstantVal: ValueRef,
        ConstantIndices: *const ValueRef,
        NumIndices: c_uint,
    ) -> ValueRef;
    pub fn LLVMConstInBoundsGEP(
        ConstantVal: ValueRef,
        ConstantIndices: *const ValueRef,
        NumIndices: c_uint,
    ) -> ValueRef;
    pub fn LLVMConstTrunc(ConstantVal: ValueRef, ToType: &Type) -> ValueRef;
    pub fn LLVMConstZExt(ConstantVal: ValueRef, ToType: &Type) -> ValueRef;
    pub fn LLVMConstUIToFP(ConstantVal: ValueRef, ToType: &Type) -> ValueRef;
    pub fn LLVMConstSIToFP(ConstantVal: ValueRef, ToType: &Type) -> ValueRef;
    pub fn LLVMConstFPToUI(ConstantVal: ValueRef, ToType: &Type) -> ValueRef;
    pub fn LLVMConstFPToSI(ConstantVal: ValueRef, ToType: &Type) -> ValueRef;
    pub fn LLVMConstPtrToInt(ConstantVal: ValueRef, ToType: &Type) -> ValueRef;
    pub fn LLVMConstIntToPtr(ConstantVal: ValueRef, ToType: &Type) -> ValueRef;
    pub fn LLVMConstBitCast(ConstantVal: ValueRef, ToType: &Type) -> ValueRef;
    pub fn LLVMConstPointerCast(ConstantVal: ValueRef, ToType: &Type) -> ValueRef;
    pub fn LLVMConstIntCast(ConstantVal: ValueRef, ToType: &Type, isSigned: Bool) -> ValueRef;
    pub fn LLVMConstFPCast(ConstantVal: ValueRef, ToType: &Type) -> ValueRef;
    pub fn LLVMConstExtractValue(AggConstant: ValueRef,
                                 IdxList: *const c_uint,
                                 NumIdx: c_uint)
                                 -> ValueRef;
    pub fn LLVMConstInlineAsm(Ty: &Type,
                              AsmString: *const c_char,
                              Constraints: *const c_char,
                              HasSideEffects: Bool,
                              IsAlignStack: Bool)
                              -> ValueRef;


    // Operations on global variables, functions, and aliases (globals)
    pub fn LLVMIsDeclaration(Global: ValueRef) -> Bool;
    pub fn LLVMRustGetLinkage(Global: ValueRef) -> Linkage;
    pub fn LLVMRustSetLinkage(Global: ValueRef, RustLinkage: Linkage);
    pub fn LLVMGetSection(Global: ValueRef) -> *const c_char;
    pub fn LLVMSetSection(Global: ValueRef, Section: *const c_char);
    pub fn LLVMRustGetVisibility(Global: ValueRef) -> Visibility;
    pub fn LLVMRustSetVisibility(Global: ValueRef, Viz: Visibility);
    pub fn LLVMGetAlignment(Global: ValueRef) -> c_uint;
    pub fn LLVMSetAlignment(Global: ValueRef, Bytes: c_uint);
    pub fn LLVMSetDLLStorageClass(V: ValueRef, C: DLLStorageClass);


    // Operations on global variables
    pub fn LLVMIsAGlobalVariable(GlobalVar: ValueRef) -> ValueRef;
    pub fn LLVMAddGlobal(M: &Module, Ty: &Type, Name: *const c_char) -> ValueRef;
    pub fn LLVMGetNamedGlobal(M: &Module, Name: *const c_char) -> ValueRef;
    pub fn LLVMRustGetOrInsertGlobal(M: &Module, Name: *const c_char, T: &Type) -> ValueRef;
    pub fn LLVMGetFirstGlobal(M: &Module) -> ValueRef;
    pub fn LLVMGetNextGlobal(GlobalVar: ValueRef) -> ValueRef;
    pub fn LLVMDeleteGlobal(GlobalVar: ValueRef);
    pub fn LLVMGetInitializer(GlobalVar: ValueRef) -> ValueRef;
    pub fn LLVMSetInitializer(GlobalVar: ValueRef, ConstantVal: ValueRef);
    pub fn LLVMSetThreadLocal(GlobalVar: ValueRef, IsThreadLocal: Bool);
    pub fn LLVMSetThreadLocalMode(GlobalVar: ValueRef, Mode: ThreadLocalMode);
    pub fn LLVMIsGlobalConstant(GlobalVar: ValueRef) -> Bool;
    pub fn LLVMSetGlobalConstant(GlobalVar: ValueRef, IsConstant: Bool);
    pub fn LLVMRustGetNamedValue(M: &Module, Name: *const c_char) -> ValueRef;
    pub fn LLVMSetTailCall(CallInst: ValueRef, IsTailCall: Bool);

    // Operations on functions
    pub fn LLVMAddFunction(M: &Module, Name: *const c_char, FunctionTy: &Type) -> ValueRef;
    pub fn LLVMGetNamedFunction(M: &Module, Name: *const c_char) -> ValueRef;
    pub fn LLVMGetFirstFunction(M: &Module) -> ValueRef;
    pub fn LLVMGetNextFunction(Fn: ValueRef) -> ValueRef;
    pub fn LLVMRustGetOrInsertFunction(M: &Module,
                                       Name: *const c_char,
                                       FunctionTy: &Type)
                                       -> ValueRef;
    pub fn LLVMSetFunctionCallConv(Fn: ValueRef, CC: c_uint);
    pub fn LLVMRustAddAlignmentAttr(Fn: ValueRef, index: c_uint, bytes: u32);
    pub fn LLVMRustAddDereferenceableAttr(Fn: ValueRef, index: c_uint, bytes: u64);
    pub fn LLVMRustAddDereferenceableOrNullAttr(Fn: ValueRef, index: c_uint, bytes: u64);
    pub fn LLVMRustAddFunctionAttribute(Fn: ValueRef, index: c_uint, attr: Attribute);
    pub fn LLVMRustAddFunctionAttrStringValue(Fn: ValueRef,
                                              index: c_uint,
                                              Name: *const c_char,
                                              Value: *const c_char);
    pub fn LLVMRustRemoveFunctionAttributes(Fn: ValueRef, index: c_uint, attr: Attribute);

    // Operations on parameters
    pub fn LLVMCountParams(Fn: ValueRef) -> c_uint;
    pub fn LLVMGetParam(Fn: ValueRef, Index: c_uint) -> ValueRef;

    // Operations on basic blocks
    pub fn LLVMBasicBlockAsValue(BB: BasicBlockRef) -> ValueRef;
    pub fn LLVMGetBasicBlockParent(BB: BasicBlockRef) -> ValueRef;
    pub fn LLVMAppendBasicBlockInContext(C: &Context,
                                         Fn: ValueRef,
                                         Name: *const c_char)
                                         -> BasicBlockRef;
    pub fn LLVMDeleteBasicBlock(BB: BasicBlockRef);

    // Operations on instructions
    pub fn LLVMGetInstructionParent(Inst: ValueRef) -> BasicBlockRef;
    pub fn LLVMGetFirstBasicBlock(Fn: ValueRef) -> BasicBlockRef;
    pub fn LLVMGetFirstInstruction(BB: BasicBlockRef) -> ValueRef;
    pub fn LLVMInstructionEraseFromParent(Inst: ValueRef);

    // Operations on call sites
    pub fn LLVMSetInstructionCallConv(Instr: ValueRef, CC: c_uint);
    pub fn LLVMRustAddCallSiteAttribute(Instr: ValueRef, index: c_uint, attr: Attribute);
    pub fn LLVMRustAddAlignmentCallSiteAttr(Instr: ValueRef, index: c_uint, bytes: u32);
    pub fn LLVMRustAddDereferenceableCallSiteAttr(Instr: ValueRef, index: c_uint, bytes: u64);
    pub fn LLVMRustAddDereferenceableOrNullCallSiteAttr(Instr: ValueRef,
                                                        index: c_uint,
                                                        bytes: u64);

    // Operations on load/store instructions (only)
    pub fn LLVMSetVolatile(MemoryAccessInst: ValueRef, volatile: Bool);

    // Operations on phi nodes
    pub fn LLVMAddIncoming(PhiNode: ValueRef,
                           IncomingValues: *const ValueRef,
                           IncomingBlocks: *const BasicBlockRef,
                           Count: c_uint);

    // Instruction builders
    pub fn LLVMCreateBuilderInContext(C: &Context) -> &Builder;
    pub fn LLVMPositionBuilder(Builder: &Builder, Block: BasicBlockRef, Instr: ValueRef);
    pub fn LLVMPositionBuilderBefore(Builder: &Builder, Instr: ValueRef);
    pub fn LLVMPositionBuilderAtEnd(Builder: &Builder, Block: BasicBlockRef);
    pub fn LLVMGetInsertBlock(Builder: &Builder) -> BasicBlockRef;
    pub fn LLVMDisposeBuilder(Builder: &Builder);

    // Metadata
    pub fn LLVMSetCurrentDebugLocation(Builder: &Builder, L: Option<NonNull<Value_opaque>>);
    pub fn LLVMGetCurrentDebugLocation(Builder: &Builder) -> ValueRef;
    pub fn LLVMSetInstDebugLocation(Builder: &Builder, Inst: ValueRef);

    // Terminators
    pub fn LLVMBuildRetVoid(B: &Builder) -> ValueRef;
    pub fn LLVMBuildRet(B: &Builder, V: ValueRef) -> ValueRef;
    pub fn LLVMBuildAggregateRet(B: &Builder, RetVals: *const ValueRef, N: c_uint) -> ValueRef;
    pub fn LLVMBuildBr(B: &Builder, Dest: BasicBlockRef) -> ValueRef;
    pub fn LLVMBuildCondBr(B: &Builder,
                           If: ValueRef,
                           Then: BasicBlockRef,
                           Else: BasicBlockRef)
                           -> ValueRef;
    pub fn LLVMBuildSwitch(B: &Builder,
                           V: ValueRef,
                           Else: BasicBlockRef,
                           NumCases: c_uint)
                           -> ValueRef;
    pub fn LLVMBuildIndirectBr(B: &Builder, Addr: ValueRef, NumDests: c_uint) -> ValueRef;
    pub fn LLVMRustBuildInvoke(B: &Builder,
                               Fn: ValueRef,
                               Args: *const ValueRef,
                               NumArgs: c_uint,
                               Then: BasicBlockRef,
                               Catch: BasicBlockRef,
                               Bundle: Option<NonNull<OperandBundleDef_opaque>>,
                               Name: *const c_char)
                               -> ValueRef;
    pub fn LLVMBuildLandingPad(B: &'a Builder,
                               Ty: &'a Type,
                               PersFn: ValueRef,
                               NumClauses: c_uint,
                               Name: *const c_char)
                               -> ValueRef;
    pub fn LLVMBuildResume(B: &Builder, Exn: ValueRef) -> ValueRef;
    pub fn LLVMBuildUnreachable(B: &Builder) -> ValueRef;

    pub fn LLVMRustBuildCleanupPad(B: &Builder,
                                   ParentPad: Option<NonNull<Value_opaque>>,
                                   ArgCnt: c_uint,
                                   Args: *const ValueRef,
                                   Name: *const c_char)
                                   -> ValueRef;
    pub fn LLVMRustBuildCleanupRet(B: &Builder,
                                   CleanupPad: ValueRef,
                                   UnwindBB: Option<NonNull<BasicBlock_opaque>>)
                                   -> ValueRef;
    pub fn LLVMRustBuildCatchPad(B: &Builder,
                                 ParentPad: ValueRef,
                                 ArgCnt: c_uint,
                                 Args: *const ValueRef,
                                 Name: *const c_char)
                                 -> ValueRef;
    pub fn LLVMRustBuildCatchRet(B: &Builder, Pad: ValueRef, BB: BasicBlockRef) -> ValueRef;
    pub fn LLVMRustBuildCatchSwitch(Builder: &Builder,
                                    ParentPad: Option<NonNull<Value_opaque>>,
                                    BB: Option<NonNull<BasicBlock_opaque>>,
                                    NumHandlers: c_uint,
                                    Name: *const c_char)
                                    -> ValueRef;
    pub fn LLVMRustAddHandler(CatchSwitch: ValueRef, Handler: BasicBlockRef);
    pub fn LLVMSetPersonalityFn(Func: ValueRef, Pers: ValueRef);

    // Add a case to the switch instruction
    pub fn LLVMAddCase(Switch: ValueRef, OnVal: ValueRef, Dest: BasicBlockRef);

    // Add a clause to the landing pad instruction
    pub fn LLVMAddClause(LandingPad: ValueRef, ClauseVal: ValueRef);

    // Set the cleanup on a landing pad instruction
    pub fn LLVMSetCleanup(LandingPad: ValueRef, Val: Bool);

    // Arithmetic
    pub fn LLVMBuildAdd(B: &Builder,
                        LHS: ValueRef,
                        RHS: ValueRef,
                        Name: *const c_char)
                        -> ValueRef;
    pub fn LLVMBuildNSWAdd(B: &Builder,
                           LHS: ValueRef,
                           RHS: ValueRef,
                           Name: *const c_char)
                           -> ValueRef;
    pub fn LLVMBuildNUWAdd(B: &Builder,
                           LHS: ValueRef,
                           RHS: ValueRef,
                           Name: *const c_char)
                           -> ValueRef;
    pub fn LLVMBuildFAdd(B: &Builder,
                         LHS: ValueRef,
                         RHS: ValueRef,
                         Name: *const c_char)
                         -> ValueRef;
    pub fn LLVMBuildSub(B: &Builder,
                        LHS: ValueRef,
                        RHS: ValueRef,
                        Name: *const c_char)
                        -> ValueRef;
    pub fn LLVMBuildNSWSub(B: &Builder,
                           LHS: ValueRef,
                           RHS: ValueRef,
                           Name: *const c_char)
                           -> ValueRef;
    pub fn LLVMBuildNUWSub(B: &Builder,
                           LHS: ValueRef,
                           RHS: ValueRef,
                           Name: *const c_char)
                           -> ValueRef;
    pub fn LLVMBuildFSub(B: &Builder,
                         LHS: ValueRef,
                         RHS: ValueRef,
                         Name: *const c_char)
                         -> ValueRef;
    pub fn LLVMBuildMul(B: &Builder,
                        LHS: ValueRef,
                        RHS: ValueRef,
                        Name: *const c_char)
                        -> ValueRef;
    pub fn LLVMBuildNSWMul(B: &Builder,
                           LHS: ValueRef,
                           RHS: ValueRef,
                           Name: *const c_char)
                           -> ValueRef;
    pub fn LLVMBuildNUWMul(B: &Builder,
                           LHS: ValueRef,
                           RHS: ValueRef,
                           Name: *const c_char)
                           -> ValueRef;
    pub fn LLVMBuildFMul(B: &Builder,
                         LHS: ValueRef,
                         RHS: ValueRef,
                         Name: *const c_char)
                         -> ValueRef;
    pub fn LLVMBuildUDiv(B: &Builder,
                         LHS: ValueRef,
                         RHS: ValueRef,
                         Name: *const c_char)
                         -> ValueRef;
    pub fn LLVMBuildExactUDiv(B: &Builder,
                              LHS: ValueRef,
                              RHS: ValueRef,
                              Name: *const c_char)
                              -> ValueRef;
    pub fn LLVMBuildSDiv(B: &Builder,
                         LHS: ValueRef,
                         RHS: ValueRef,
                         Name: *const c_char)
                         -> ValueRef;
    pub fn LLVMBuildExactSDiv(B: &Builder,
                              LHS: ValueRef,
                              RHS: ValueRef,
                              Name: *const c_char)
                              -> ValueRef;
    pub fn LLVMBuildFDiv(B: &Builder,
                         LHS: ValueRef,
                         RHS: ValueRef,
                         Name: *const c_char)
                         -> ValueRef;
    pub fn LLVMBuildURem(B: &Builder,
                         LHS: ValueRef,
                         RHS: ValueRef,
                         Name: *const c_char)
                         -> ValueRef;
    pub fn LLVMBuildSRem(B: &Builder,
                         LHS: ValueRef,
                         RHS: ValueRef,
                         Name: *const c_char)
                         -> ValueRef;
    pub fn LLVMBuildFRem(B: &Builder,
                         LHS: ValueRef,
                         RHS: ValueRef,
                         Name: *const c_char)
                         -> ValueRef;
    pub fn LLVMBuildShl(B: &Builder,
                        LHS: ValueRef,
                        RHS: ValueRef,
                        Name: *const c_char)
                        -> ValueRef;
    pub fn LLVMBuildLShr(B: &Builder,
                         LHS: ValueRef,
                         RHS: ValueRef,
                         Name: *const c_char)
                         -> ValueRef;
    pub fn LLVMBuildAShr(B: &Builder,
                         LHS: ValueRef,
                         RHS: ValueRef,
                         Name: *const c_char)
                         -> ValueRef;
    pub fn LLVMBuildAnd(B: &Builder,
                        LHS: ValueRef,
                        RHS: ValueRef,
                        Name: *const c_char)
                        -> ValueRef;
    pub fn LLVMBuildOr(B: &Builder,
                       LHS: ValueRef,
                       RHS: ValueRef,
                       Name: *const c_char)
                       -> ValueRef;
    pub fn LLVMBuildXor(B: &Builder,
                        LHS: ValueRef,
                        RHS: ValueRef,
                        Name: *const c_char)
                        -> ValueRef;
    pub fn LLVMBuildBinOp(B: &Builder,
                          Op: Opcode,
                          LHS: ValueRef,
                          RHS: ValueRef,
                          Name: *const c_char)
                          -> ValueRef;
    pub fn LLVMBuildNeg(B: &Builder, V: ValueRef, Name: *const c_char) -> ValueRef;
    pub fn LLVMBuildNSWNeg(B: &Builder, V: ValueRef, Name: *const c_char) -> ValueRef;
    pub fn LLVMBuildNUWNeg(B: &Builder, V: ValueRef, Name: *const c_char) -> ValueRef;
    pub fn LLVMBuildFNeg(B: &Builder, V: ValueRef, Name: *const c_char) -> ValueRef;
    pub fn LLVMBuildNot(B: &Builder, V: ValueRef, Name: *const c_char) -> ValueRef;
    pub fn LLVMRustSetHasUnsafeAlgebra(Instr: ValueRef);

    // Memory
    pub fn LLVMBuildAlloca(B: &Builder, Ty: &Type, Name: *const c_char) -> ValueRef;
    pub fn LLVMBuildFree(B: &Builder, PointerVal: ValueRef) -> ValueRef;
    pub fn LLVMBuildLoad(B: &Builder, PointerVal: ValueRef, Name: *const c_char) -> ValueRef;

    pub fn LLVMBuildStore(B: &Builder, Val: ValueRef, Ptr: ValueRef) -> ValueRef;

    pub fn LLVMBuildGEP(B: &Builder,
                        Pointer: ValueRef,
                        Indices: *const ValueRef,
                        NumIndices: c_uint,
                        Name: *const c_char)
                        -> ValueRef;
    pub fn LLVMBuildInBoundsGEP(B: &Builder,
                                Pointer: ValueRef,
                                Indices: *const ValueRef,
                                NumIndices: c_uint,
                                Name: *const c_char)
                                -> ValueRef;
    pub fn LLVMBuildStructGEP(B: &Builder,
                              Pointer: ValueRef,
                              Idx: c_uint,
                              Name: *const c_char)
                              -> ValueRef;
    pub fn LLVMBuildGlobalString(B: &Builder,
                                 Str: *const c_char,
                                 Name: *const c_char)
                                 -> ValueRef;
    pub fn LLVMBuildGlobalStringPtr(B: &Builder,
                                    Str: *const c_char,
                                    Name: *const c_char)
                                    -> ValueRef;

    // Casts
    pub fn LLVMBuildTrunc(B: &'a Builder,
                          Val: ValueRef,
                          DestTy: &'a Type,
                          Name: *const c_char)
                          -> ValueRef;
    pub fn LLVMBuildZExt(B: &'a Builder,
                         Val: ValueRef,
                         DestTy: &'a Type,
                         Name: *const c_char)
                         -> ValueRef;
    pub fn LLVMBuildSExt(B: &'a Builder,
                         Val: ValueRef,
                         DestTy: &'a Type,
                         Name: *const c_char)
                         -> ValueRef;
    pub fn LLVMBuildFPToUI(B: &'a Builder,
                           Val: ValueRef,
                           DestTy: &'a Type,
                           Name: *const c_char)
                           -> ValueRef;
    pub fn LLVMBuildFPToSI(B: &'a Builder,
                           Val: ValueRef,
                           DestTy: &'a Type,
                           Name: *const c_char)
                           -> ValueRef;
    pub fn LLVMBuildUIToFP(B: &'a Builder,
                           Val: ValueRef,
                           DestTy: &'a Type,
                           Name: *const c_char)
                           -> ValueRef;
    pub fn LLVMBuildSIToFP(B: &'a Builder,
                           Val: ValueRef,
                           DestTy: &'a Type,
                           Name: *const c_char)
                           -> ValueRef;
    pub fn LLVMBuildFPTrunc(B: &'a Builder,
                            Val: ValueRef,
                            DestTy: &'a Type,
                            Name: *const c_char)
                            -> ValueRef;
    pub fn LLVMBuildFPExt(B: &'a Builder,
                          Val: ValueRef,
                          DestTy: &'a Type,
                          Name: *const c_char)
                          -> ValueRef;
    pub fn LLVMBuildPtrToInt(B: &'a Builder,
                             Val: ValueRef,
                             DestTy: &'a Type,
                             Name: *const c_char)
                             -> ValueRef;
    pub fn LLVMBuildIntToPtr(B: &'a Builder,
                             Val: ValueRef,
                             DestTy: &'a Type,
                             Name: *const c_char)
                             -> ValueRef;
    pub fn LLVMBuildBitCast(B: &'a Builder,
                            Val: ValueRef,
                            DestTy: &'a Type,
                            Name: *const c_char)
                            -> ValueRef;
    pub fn LLVMBuildZExtOrBitCast(B: &'a Builder,
                                  Val: ValueRef,
                                  DestTy: &'a Type,
                                  Name: *const c_char)
                                  -> ValueRef;
    pub fn LLVMBuildSExtOrBitCast(B: &'a Builder,
                                  Val: ValueRef,
                                  DestTy: &'a Type,
                                  Name: *const c_char)
                                  -> ValueRef;
    pub fn LLVMBuildTruncOrBitCast(B: &'a Builder,
                                   Val: ValueRef,
                                   DestTy: &'a Type,
                                   Name: *const c_char)
                                   -> ValueRef;
    pub fn LLVMBuildCast(B: &'a Builder,
                         Op: Opcode,
                         Val: ValueRef,
                         DestTy: &'a Type,
                         Name: *const c_char)
                         -> ValueRef;
    pub fn LLVMBuildPointerCast(B: &'a Builder,
                                Val: ValueRef,
                                DestTy: &'a Type,
                                Name: *const c_char)
                                -> ValueRef;
    pub fn LLVMRustBuildIntCast(B: &'a Builder,
                                Val: ValueRef,
                                DestTy: &'a Type,
                                IsSized: bool)
                                -> ValueRef;
    pub fn LLVMBuildFPCast(B: &'a Builder,
                           Val: ValueRef,
                           DestTy: &'a Type,
                           Name: *const c_char)
                           -> ValueRef;

    // Comparisons
    pub fn LLVMBuildICmp(B: &Builder,
                         Op: c_uint,
                         LHS: ValueRef,
                         RHS: ValueRef,
                         Name: *const c_char)
                         -> ValueRef;
    pub fn LLVMBuildFCmp(B: &Builder,
                         Op: c_uint,
                         LHS: ValueRef,
                         RHS: ValueRef,
                         Name: *const c_char)
                         -> ValueRef;

    // Miscellaneous instructions
    pub fn LLVMBuildPhi(B: &Builder, Ty: &Type, Name: *const c_char) -> ValueRef;
    pub fn LLVMRustBuildCall(B: &Builder,
                             Fn: ValueRef,
                             Args: *const ValueRef,
                             NumArgs: c_uint,
                             Bundle: Option<NonNull<OperandBundleDef_opaque>>,
                             Name: *const c_char)
                             -> ValueRef;
    pub fn LLVMBuildSelect(B: &Builder,
                           If: ValueRef,
                           Then: ValueRef,
                           Else: ValueRef,
                           Name: *const c_char)
                           -> ValueRef;
    pub fn LLVMBuildVAArg(B: &'a Builder,
                          list: ValueRef,
                          Ty: &'a Type,
                          Name: *const c_char)
                          -> ValueRef;
    pub fn LLVMBuildExtractElement(B: &Builder,
                                   VecVal: ValueRef,
                                   Index: ValueRef,
                                   Name: *const c_char)
                                   -> ValueRef;
    pub fn LLVMBuildInsertElement(B: &Builder,
                                  VecVal: ValueRef,
                                  EltVal: ValueRef,
                                  Index: ValueRef,
                                  Name: *const c_char)
                                  -> ValueRef;
    pub fn LLVMBuildShuffleVector(B: &Builder,
                                  V1: ValueRef,
                                  V2: ValueRef,
                                  Mask: ValueRef,
                                  Name: *const c_char)
                                  -> ValueRef;
    pub fn LLVMBuildExtractValue(B: &Builder,
                                 AggVal: ValueRef,
                                 Index: c_uint,
                                 Name: *const c_char)
                                 -> ValueRef;
    pub fn LLVMBuildInsertValue(B: &Builder,
                                AggVal: ValueRef,
                                EltVal: ValueRef,
                                Index: c_uint,
                                Name: *const c_char)
                                -> ValueRef;

    pub fn LLVMRustBuildVectorReduceFAdd(B: &Builder,
                                         Acc: ValueRef,
                                         Src: ValueRef)
                                         -> ValueRef;
    pub fn LLVMRustBuildVectorReduceFMul(B: &Builder,
                                         Acc: ValueRef,
                                         Src: ValueRef)
                                         -> ValueRef;
    pub fn LLVMRustBuildVectorReduceAdd(B: &Builder,
                                        Src: ValueRef)
                                        -> ValueRef;
    pub fn LLVMRustBuildVectorReduceMul(B: &Builder,
                                        Src: ValueRef)
                                        -> ValueRef;
    pub fn LLVMRustBuildVectorReduceAnd(B: &Builder,
                                        Src: ValueRef)
                                        -> ValueRef;
    pub fn LLVMRustBuildVectorReduceOr(B: &Builder,
                                       Src: ValueRef)
                                       -> ValueRef;
    pub fn LLVMRustBuildVectorReduceXor(B: &Builder,
                                        Src: ValueRef)
                                        -> ValueRef;
    pub fn LLVMRustBuildVectorReduceMin(B: &Builder,
                                        Src: ValueRef,
                                        IsSigned: bool)
                                        -> ValueRef;
    pub fn LLVMRustBuildVectorReduceMax(B: &Builder,
                                        Src: ValueRef,
                                        IsSigned: bool)
                                        -> ValueRef;
    pub fn LLVMRustBuildVectorReduceFMin(B: &Builder,
                                         Src: ValueRef,
                                         IsNaN: bool)
                                         -> ValueRef;
    pub fn LLVMRustBuildVectorReduceFMax(B: &Builder,
                                         Src: ValueRef,
                                         IsNaN: bool)
                                         -> ValueRef;

    pub fn LLVMRustBuildMinNum(B: &Builder, LHS: ValueRef, LHS: ValueRef) -> ValueRef;
    pub fn LLVMRustBuildMaxNum(B: &Builder, LHS: ValueRef, LHS: ValueRef) -> ValueRef;

    pub fn LLVMBuildIsNull(B: &Builder, Val: ValueRef, Name: *const c_char) -> ValueRef;
    pub fn LLVMBuildIsNotNull(B: &Builder, Val: ValueRef, Name: *const c_char) -> ValueRef;
    pub fn LLVMBuildPtrDiff(B: &Builder,
                            LHS: ValueRef,
                            RHS: ValueRef,
                            Name: *const c_char)
                            -> ValueRef;

    // Atomic Operations
    pub fn LLVMRustBuildAtomicLoad(B: &Builder,
                                   PointerVal: ValueRef,
                                   Name: *const c_char,
                                   Order: AtomicOrdering)
                                   -> ValueRef;

    pub fn LLVMRustBuildAtomicStore(B: &Builder,
                                    Val: ValueRef,
                                    Ptr: ValueRef,
                                    Order: AtomicOrdering)
                                    -> ValueRef;

    pub fn LLVMRustBuildAtomicCmpXchg(B: &Builder,
                                      LHS: ValueRef,
                                      CMP: ValueRef,
                                      RHS: ValueRef,
                                      Order: AtomicOrdering,
                                      FailureOrder: AtomicOrdering,
                                      Weak: Bool)
                                      -> ValueRef;

    pub fn LLVMBuildAtomicRMW(B: &Builder,
                              Op: AtomicRmwBinOp,
                              LHS: ValueRef,
                              RHS: ValueRef,
                              Order: AtomicOrdering,
                              SingleThreaded: Bool)
                              -> ValueRef;

    pub fn LLVMRustBuildAtomicFence(B: &Builder,
                                    Order: AtomicOrdering,
                                    Scope: SynchronizationScope);


    // Selected entries from the downcasts.
    pub fn LLVMIsATerminatorInst(Inst: ValueRef) -> ValueRef;
    pub fn LLVMIsAStoreInst(Inst: ValueRef) -> ValueRef;

    /// Writes a module to the specified path. Returns 0 on success.
    pub fn LLVMWriteBitcodeToFile(M: &Module, Path: *const c_char) -> c_int;

    /// Creates target data from a target layout string.
    pub fn LLVMCreateTargetData(StringRep: *const c_char) -> TargetDataRef;

    /// Disposes target data.
    pub fn LLVMDisposeTargetData(TD: TargetDataRef);

    /// Creates a pass manager.
    pub fn LLVMCreatePassManager() -> PassManagerRef;

    /// Creates a function-by-function pass manager
    pub fn LLVMCreateFunctionPassManagerForModule(M: &Module) -> PassManagerRef;

    /// Disposes a pass manager.
    pub fn LLVMDisposePassManager(PM: PassManagerRef);

    /// Runs a pass manager on a module.
    pub fn LLVMRunPassManager(PM: PassManagerRef, M: &Module) -> Bool;

    pub fn LLVMInitializePasses();

    pub fn LLVMPassManagerBuilderCreate() -> PassManagerBuilderRef;
    pub fn LLVMPassManagerBuilderDispose(PMB: PassManagerBuilderRef);
    pub fn LLVMPassManagerBuilderSetSizeLevel(PMB: PassManagerBuilderRef, Value: Bool);
    pub fn LLVMPassManagerBuilderSetDisableUnrollLoops(PMB: PassManagerBuilderRef, Value: Bool);
    pub fn LLVMPassManagerBuilderUseInlinerWithThreshold(PMB: PassManagerBuilderRef,
                                                         threshold: c_uint);
    pub fn LLVMPassManagerBuilderPopulateModulePassManager(PMB: PassManagerBuilderRef,
                                                           PM: PassManagerRef);

    pub fn LLVMPassManagerBuilderPopulateFunctionPassManager(PMB: PassManagerBuilderRef,
                                                             PM: PassManagerRef);
    pub fn LLVMPassManagerBuilderPopulateLTOPassManager(PMB: PassManagerBuilderRef,
                                                        PM: PassManagerRef,
                                                        Internalize: Bool,
                                                        RunInliner: Bool);
    pub fn LLVMRustPassManagerBuilderPopulateThinLTOPassManager(
        PMB: PassManagerBuilderRef,
        PM: PassManagerRef) -> bool;

    // Stuff that's in rustllvm/ because it's not upstream yet.

    /// Opens an object file.
    pub fn LLVMCreateObjectFile(MemBuf: MemoryBufferRef) -> ObjectFileRef;
    /// Closes an object file.
    pub fn LLVMDisposeObjectFile(ObjFile: ObjectFileRef);

    /// Enumerates the sections in an object file.
    pub fn LLVMGetSections(ObjFile: ObjectFileRef) -> SectionIteratorRef;
    /// Destroys a section iterator.
    pub fn LLVMDisposeSectionIterator(SI: SectionIteratorRef);
    /// Returns true if the section iterator is at the end of the section
    /// list:
    pub fn LLVMIsSectionIteratorAtEnd(ObjFile: ObjectFileRef, SI: SectionIteratorRef) -> Bool;
    /// Moves the section iterator to point to the next section.
    pub fn LLVMMoveToNextSection(SI: SectionIteratorRef);
    /// Returns the current section size.
    pub fn LLVMGetSectionSize(SI: SectionIteratorRef) -> c_ulonglong;
    /// Returns the current section contents as a string buffer.
    pub fn LLVMGetSectionContents(SI: SectionIteratorRef) -> *const c_char;

    /// Reads the given file and returns it as a memory buffer. Use
    /// LLVMDisposeMemoryBuffer() to get rid of it.
    pub fn LLVMRustCreateMemoryBufferWithContentsOfFile(Path: *const c_char) -> MemoryBufferRef;

    pub fn LLVMStartMultithreaded() -> Bool;

    /// Returns a string describing the last error caused by an LLVMRust* call.
    pub fn LLVMRustGetLastError() -> *const c_char;

    /// Print the pass timings since static dtors aren't picking them up.
    pub fn LLVMRustPrintPassTimings();

    pub fn LLVMStructCreateNamed(C: &Context, Name: *const c_char) -> &Type;

    pub fn LLVMStructSetBody(StructTy: &'a Type,
                             ElementTypes: *const &'a Type,
                             ElementCount: c_uint,
                             Packed: Bool);

    /// Prepares inline assembly.
    pub fn LLVMRustInlineAsm(Ty: &Type,
                             AsmString: *const c_char,
                             Constraints: *const c_char,
                             SideEffects: Bool,
                             AlignStack: Bool,
                             Dialect: AsmDialect)
                             -> ValueRef;

    pub fn LLVMRustDebugMetadataVersion() -> u32;
    pub fn LLVMRustVersionMajor() -> u32;
    pub fn LLVMRustVersionMinor() -> u32;

    pub fn LLVMRustAddModuleFlag(M: &Module, name: *const c_char, value: u32);

    pub fn LLVMRustMetadataAsValue(C: &Context, MD: MetadataRef) -> ValueRef;

    pub fn LLVMRustDIBuilderCreate(M: &Module) -> &DIBuilder;

    pub fn LLVMRustDIBuilderDispose(Builder: &DIBuilder);

    pub fn LLVMRustDIBuilderFinalize(Builder: &DIBuilder);

    pub fn LLVMRustDIBuilderCreateCompileUnit(Builder: &DIBuilder,
                                              Lang: c_uint,
                                              File: DIFile,
                                              Producer: *const c_char,
                                              isOptimized: bool,
                                              Flags: *const c_char,
                                              RuntimeVer: c_uint,
                                              SplitName: *const c_char)
                                              -> DIDescriptor;

    pub fn LLVMRustDIBuilderCreateFile(Builder: &DIBuilder,
                                       Filename: *const c_char,
                                       Directory: *const c_char)
                                       -> DIFile;

    pub fn LLVMRustDIBuilderCreateSubroutineType(Builder: &DIBuilder,
                                                 File: DIFile,
                                                 ParameterTypes: DIArray)
                                                 -> DICompositeType;

    pub fn LLVMRustDIBuilderCreateFunction(Builder: &DIBuilder,
                                           Scope: DIDescriptor,
                                           Name: *const c_char,
                                           LinkageName: *const c_char,
                                           File: DIFile,
                                           LineNo: c_uint,
                                           Ty: DIType,
                                           isLocalToUnit: bool,
                                           isDefinition: bool,
                                           ScopeLine: c_uint,
                                           Flags: DIFlags,
                                           isOptimized: bool,
                                           Fn: ValueRef,
                                           TParam: DIArray,
                                           Decl: Option<NonNull<DIDescriptor_opaque>>)
                                           -> DISubprogram;

    pub fn LLVMRustDIBuilderCreateBasicType(Builder: &DIBuilder,
                                            Name: *const c_char,
                                            SizeInBits: u64,
                                            AlignInBits: u32,
                                            Encoding: c_uint)
                                            -> DIBasicType;

    pub fn LLVMRustDIBuilderCreatePointerType(Builder: &DIBuilder,
                                              PointeeTy: DIType,
                                              SizeInBits: u64,
                                              AlignInBits: u32,
                                              Name: *const c_char)
                                              -> DIDerivedType;

    pub fn LLVMRustDIBuilderCreateStructType(Builder: &DIBuilder,
                                             Scope: Option<NonNull<DIDescriptor_opaque>>,
                                             Name: *const c_char,
                                             File: DIFile,
                                             LineNumber: c_uint,
                                             SizeInBits: u64,
                                             AlignInBits: u32,
                                             Flags: DIFlags,
                                             DerivedFrom: Option<NonNull<DIType_opaque>>,
                                             Elements: DIArray,
                                             RunTimeLang: c_uint,
                                             VTableHolder: Option<NonNull<DIType_opaque>>,
                                             UniqueId: *const c_char)
                                             -> DICompositeType;

    pub fn LLVMRustDIBuilderCreateMemberType(Builder: &DIBuilder,
                                             Scope: DIDescriptor,
                                             Name: *const c_char,
                                             File: DIFile,
                                             LineNo: c_uint,
                                             SizeInBits: u64,
                                             AlignInBits: u32,
                                             OffsetInBits: u64,
                                             Flags: DIFlags,
                                             Ty: DIType)
                                             -> DIDerivedType;

    pub fn LLVMRustDIBuilderCreateLexicalBlock(Builder: &DIBuilder,
                                               Scope: DIScope,
                                               File: DIFile,
                                               Line: c_uint,
                                               Col: c_uint)
                                               -> DILexicalBlock;

    pub fn LLVMRustDIBuilderCreateLexicalBlockFile(Builder: &DIBuilder,
                                                   Scope: DIScope,
                                                   File: DIFile)
                                                   -> DILexicalBlock;

    pub fn LLVMRustDIBuilderCreateStaticVariable(Builder: &DIBuilder,
                                                 Context: Option<NonNull<DIScope_opaque>>,
                                                 Name: *const c_char,
                                                 LinkageName: *const c_char,
                                                 File: DIFile,
                                                 LineNo: c_uint,
                                                 Ty: DIType,
                                                 isLocalToUnit: bool,
                                                 Val: ValueRef,
                                                 Decl: Option<NonNull<DIDescriptor_opaque>>,
                                                 AlignInBits: u32)
                                                 -> DIGlobalVariable;

    pub fn LLVMRustDIBuilderCreateVariable(Builder: &DIBuilder,
                                           Tag: c_uint,
                                           Scope: DIDescriptor,
                                           Name: *const c_char,
                                           File: DIFile,
                                           LineNo: c_uint,
                                           Ty: DIType,
                                           AlwaysPreserve: bool,
                                           Flags: DIFlags,
                                           ArgNo: c_uint,
                                           AlignInBits: u32)
                                           -> DIVariable;

    pub fn LLVMRustDIBuilderCreateArrayType(Builder: &DIBuilder,
                                            Size: u64,
                                            AlignInBits: u32,
                                            Ty: DIType,
                                            Subscripts: DIArray)
                                            -> DIType;

    pub fn LLVMRustDIBuilderCreateVectorType(Builder: &DIBuilder,
                                             Size: u64,
                                             AlignInBits: u32,
                                             Ty: DIType,
                                             Subscripts: DIArray)
                                             -> DIType;

    pub fn LLVMRustDIBuilderGetOrCreateSubrange(Builder: &DIBuilder,
                                                Lo: i64,
                                                Count: i64)
                                                -> DISubrange;

    pub fn LLVMRustDIBuilderGetOrCreateArray(Builder: &DIBuilder,
                                             Ptr: *const Option<NonNull<DIDescriptor_opaque>>,
                                             Count: c_uint)
                                             -> DIArray;

    pub fn LLVMRustDIBuilderInsertDeclareAtEnd(Builder: &DIBuilder,
                                               Val: ValueRef,
                                               VarInfo: DIVariable,
                                               AddrOps: *const i64,
                                               AddrOpsCount: c_uint,
                                               DL: ValueRef,
                                               InsertAtEnd: BasicBlockRef)
                                               -> ValueRef;

    pub fn LLVMRustDIBuilderCreateEnumerator(Builder: &DIBuilder,
                                             Name: *const c_char,
                                             Val: u64)
                                             -> DIEnumerator;

    pub fn LLVMRustDIBuilderCreateEnumerationType(Builder: &DIBuilder,
                                                  Scope: DIScope,
                                                  Name: *const c_char,
                                                  File: DIFile,
                                                  LineNumber: c_uint,
                                                  SizeInBits: u64,
                                                  AlignInBits: u32,
                                                  Elements: DIArray,
                                                  ClassType: DIType)
                                                  -> DIType;

    pub fn LLVMRustDIBuilderCreateUnionType(Builder: &DIBuilder,
                                            Scope: DIScope,
                                            Name: *const c_char,
                                            File: DIFile,
                                            LineNumber: c_uint,
                                            SizeInBits: u64,
                                            AlignInBits: u32,
                                            Flags: DIFlags,
                                            Elements: Option<NonNull<DIArray_opaque>>,
                                            RunTimeLang: c_uint,
                                            UniqueId: *const c_char)
                                            -> DIType;

    pub fn LLVMSetUnnamedAddr(GlobalVar: ValueRef, UnnamedAddr: Bool);

    pub fn LLVMRustDIBuilderCreateTemplateTypeParameter(Builder: &DIBuilder,
                                                        Scope: Option<NonNull<DIScope_opaque>>,
                                                        Name: *const c_char,
                                                        Ty: DIType,
                                                        File: DIFile,
                                                        LineNo: c_uint,
                                                        ColumnNo: c_uint)
                                                        -> DITemplateTypeParameter;


    pub fn LLVMRustDIBuilderCreateNameSpace(Builder: &DIBuilder,
                                            Scope: Option<NonNull<DIScope_opaque>>,
                                            Name: *const c_char,
                                            File: DIFile,
                                            LineNo: c_uint)
                                            -> DINameSpace;
    pub fn LLVMRustDICompositeTypeSetTypeArray(Builder: &DIBuilder,
                                               CompositeType: DIType,
                                               TypeArray: DIArray);


    pub fn LLVMRustDIBuilderCreateDebugLocation(Context: &Context,
                                                Line: c_uint,
                                                Column: c_uint,
                                                Scope: DIScope,
                                                InlinedAt: Option<NonNull<Metadata_opaque>>)
                                                -> ValueRef;
    pub fn LLVMRustDIBuilderCreateOpDeref() -> i64;
    pub fn LLVMRustDIBuilderCreateOpPlusUconst() -> i64;

    pub fn LLVMRustWriteTypeToString(Type: &Type, s: RustStringRef);
    pub fn LLVMRustWriteValueToString(value_ref: ValueRef, s: RustStringRef);

    pub fn LLVMIsAConstantInt(value_ref: ValueRef) -> ValueRef;
    pub fn LLVMIsAConstantFP(value_ref: ValueRef) -> ValueRef;

    pub fn LLVMRustPassKind(Pass: PassRef) -> PassKind;
    pub fn LLVMRustFindAndCreatePass(Pass: *const c_char) -> PassRef;
    pub fn LLVMRustAddPass(PM: PassManagerRef, Pass: PassRef);

    pub fn LLVMRustHasFeature(T: TargetMachineRef, s: *const c_char) -> bool;

    pub fn LLVMRustPrintTargetCPUs(T: TargetMachineRef);
    pub fn LLVMRustPrintTargetFeatures(T: TargetMachineRef);

    pub fn LLVMRustCreateTargetMachine(Triple: *const c_char,
                                       CPU: *const c_char,
                                       Features: *const c_char,
                                       Model: CodeModel,
                                       Reloc: RelocMode,
                                       Level: CodeGenOptLevel,
                                       UseSoftFP: bool,
                                       PositionIndependentExecutable: bool,
                                       FunctionSections: bool,
                                       DataSections: bool,
                                       TrapUnreachable: bool,
                                       Singlethread: bool)
                                       -> Option<&'static mut TargetMachine>;
    pub fn LLVMRustDisposeTargetMachine(T: &'static mut TargetMachine);
    pub fn LLVMRustAddAnalysisPasses(T: TargetMachineRef, PM: PassManagerRef, M: &Module);
    pub fn LLVMRustAddBuilderLibraryInfo(PMB: PassManagerBuilderRef,
                                         M: &Module,
                                         DisableSimplifyLibCalls: bool);
    pub fn LLVMRustConfigurePassManagerBuilder(PMB: PassManagerBuilderRef,
                                               OptLevel: CodeGenOptLevel,
                                               MergeFunctions: bool,
                                               SLPVectorize: bool,
                                               LoopVectorize: bool,
                                               PrepareForThinLTO: bool,
                                               PGOGenPath: *const c_char,
                                               PGOUsePath: *const c_char);
    pub fn LLVMRustAddLibraryInfo(PM: PassManagerRef,
                                  M: &Module,
                                  DisableSimplifyLibCalls: bool);
    pub fn LLVMRustRunFunctionPassManager(PM: PassManagerRef, M: &Module);
    pub fn LLVMRustWriteOutputFile(T: TargetMachineRef,
                                   PM: PassManagerRef,
                                   M: &Module,
                                   Output: *const c_char,
                                   FileType: FileType)
                                   -> LLVMRustResult;
    pub fn LLVMRustPrintModule(PM: PassManagerRef,
                               M: &Module,
                               Output: *const c_char,
                               Demangle: extern fn(*const c_char,
                                                   size_t,
                                                   *mut c_char,
                                                   size_t) -> size_t);
    pub fn LLVMRustSetLLVMOptions(Argc: c_int, Argv: *const *const c_char);
    pub fn LLVMRustPrintPasses();
    pub fn LLVMRustSetNormalizedTarget(M: &Module, triple: *const c_char);
    pub fn LLVMRustAddAlwaysInlinePass(P: PassManagerBuilderRef, AddLifetimes: bool);
    pub fn LLVMRustRunRestrictionPass(M: &Module, syms: *const *const c_char, len: size_t);
    pub fn LLVMRustMarkAllFunctionsNounwind(M: &Module);

    pub fn LLVMRustOpenArchive(path: *const c_char) -> ArchiveRef;
    pub fn LLVMRustArchiveIteratorNew(AR: ArchiveRef) -> ArchiveIteratorRef;
    pub fn LLVMRustArchiveIteratorNext(AIR: ArchiveIteratorRef) -> ArchiveChildRef;
    pub fn LLVMRustArchiveChildName(ACR: ArchiveChildRef, size: *mut size_t) -> *const c_char;
    pub fn LLVMRustArchiveChildData(ACR: ArchiveChildRef, size: *mut size_t) -> *const c_char;
    pub fn LLVMRustArchiveChildFree(ACR: ArchiveChildRef);
    pub fn LLVMRustArchiveIteratorFree(AIR: ArchiveIteratorRef);
    pub fn LLVMRustDestroyArchive(AR: ArchiveRef);

    pub fn LLVMRustGetSectionName(SI: SectionIteratorRef, data: *mut *const c_char) -> size_t;

    pub fn LLVMRustWriteTwineToString(T: TwineRef, s: RustStringRef);

    pub fn LLVMContextSetDiagnosticHandler(C: &Context,
                                           Handler: DiagnosticHandler,
                                           DiagnosticContext: *mut c_void);

    pub fn LLVMRustUnpackOptimizationDiagnostic(DI: DiagnosticInfoRef,
                                                pass_name_out: RustStringRef,
                                                function_out: *mut ValueRef,
                                                loc_line_out: *mut c_uint,
                                                loc_column_out: *mut c_uint,
                                                loc_filename_out: RustStringRef,
                                                message_out: RustStringRef);
    pub fn LLVMRustUnpackInlineAsmDiagnostic(DI: DiagnosticInfoRef,
                                             cookie_out: *mut c_uint,
                                             message_out: *mut TwineRef,
                                             instruction_out: *mut ValueRef);

    pub fn LLVMRustWriteDiagnosticInfoToString(DI: DiagnosticInfoRef, s: RustStringRef);
    pub fn LLVMRustGetDiagInfoKind(DI: DiagnosticInfoRef) -> DiagnosticKind;

    pub fn LLVMRustSetInlineAsmDiagnosticHandler(C: &Context,
                                                 H: InlineAsmDiagHandler,
                                                 CX: *mut c_void);

    pub fn LLVMRustWriteSMDiagnosticToString(d: SMDiagnosticRef, s: RustStringRef);

    pub fn LLVMRustWriteArchive(Dst: *const c_char,
                                NumMembers: size_t,
                                Members: *const RustArchiveMemberRef,
                                WriteSymbtab: bool,
                                Kind: ArchiveKind)
                                -> LLVMRustResult;
    pub fn LLVMRustArchiveMemberNew(Filename: *const c_char,
                                    Name: *const c_char,
                                    Child: Option<NonNull<ArchiveChild_opaque>>)
                                    -> RustArchiveMemberRef;
    pub fn LLVMRustArchiveMemberFree(Member: RustArchiveMemberRef);

    pub fn LLVMRustSetDataLayoutFromTargetMachine(M: &Module, TM: TargetMachineRef);

    pub fn LLVMRustBuildOperandBundleDef(Name: *const c_char,
                                         Inputs: *const ValueRef,
                                         NumInputs: c_uint)
                                         -> OperandBundleDefRef;
    pub fn LLVMRustFreeOperandBundleDef(Bundle: OperandBundleDefRef);

    pub fn LLVMRustPositionBuilderAtStart(B: &Builder, BB: BasicBlockRef);

    pub fn LLVMRustSetComdat(M: &Module, V: ValueRef, Name: *const c_char);
    pub fn LLVMRustUnsetComdat(V: ValueRef);
    pub fn LLVMRustSetModulePIELevel(M: &Module);
    pub fn LLVMRustModuleBufferCreate(M: &Module) -> *mut ModuleBuffer;
    pub fn LLVMRustModuleBufferPtr(p: *const ModuleBuffer) -> *const u8;
    pub fn LLVMRustModuleBufferLen(p: *const ModuleBuffer) -> usize;
    pub fn LLVMRustModuleBufferFree(p: *mut ModuleBuffer);
    pub fn LLVMRustModuleCost(M: &Module) -> u64;

    pub fn LLVMRustThinLTOAvailable() -> bool;
    pub fn LLVMRustPGOAvailable() -> bool;
    pub fn LLVMRustWriteThinBitcodeToFile(PMR: PassManagerRef,
                                          M: &Module,
                                          BC: *const c_char) -> bool;
    pub fn LLVMRustThinLTOBufferCreate(M: &Module) -> *mut ThinLTOBuffer;
    pub fn LLVMRustThinLTOBufferFree(M: *mut ThinLTOBuffer);
    pub fn LLVMRustThinLTOBufferPtr(M: *const ThinLTOBuffer) -> *const c_char;
    pub fn LLVMRustThinLTOBufferLen(M: *const ThinLTOBuffer) -> size_t;
    pub fn LLVMRustCreateThinLTOData(
        Modules: *const ThinLTOModule,
        NumModules: c_uint,
        PreservedSymbols: *const *const c_char,
        PreservedSymbolsLen: c_uint,
    ) -> *mut ThinLTOData;
    pub fn LLVMRustPrepareThinLTORename(
        Data: *const ThinLTOData,
        Module: &Module,
    ) -> bool;
    pub fn LLVMRustPrepareThinLTOResolveWeak(
        Data: *const ThinLTOData,
        Module: &Module,
    ) -> bool;
    pub fn LLVMRustPrepareThinLTOInternalize(
        Data: *const ThinLTOData,
        Module: &Module,
    ) -> bool;
    pub fn LLVMRustPrepareThinLTOImport(
        Data: *const ThinLTOData,
        Module: &Module,
    ) -> bool;
    pub fn LLVMRustFreeThinLTOData(Data: *mut ThinLTOData);
    pub fn LLVMRustParseBitcodeForThinLTO(
        Context: &Context,
        Data: *const u8,
        len: usize,
        Identifier: *const c_char,
    ) -> Option<&Module>;
    pub fn LLVMGetModuleIdentifier(M: &Module, size: *mut usize) -> *const c_char;
    pub fn LLVMRustThinLTOGetDICompileUnit(M: &Module,
                                           CU1: *mut *mut c_void,
                                           CU2: *mut *mut c_void);
    pub fn LLVMRustThinLTOPatchDICompileUnit(M: &Module, CU: *mut c_void);

    pub fn LLVMRustLinkerNew(M: &Module) -> LinkerRef;
    pub fn LLVMRustLinkerAdd(linker: LinkerRef,
                             bytecode: *const c_char,
                             bytecode_len: usize) -> bool;
    pub fn LLVMRustLinkerFree(linker: LinkerRef);
}
