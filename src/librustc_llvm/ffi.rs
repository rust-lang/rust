// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use debuginfo::{DIBuilderRef, DIDescriptor, DIFile, DILexicalBlock, DISubprogram, DIType,
                DIBasicType, DIDerivedType, DICompositeType, DIScope, DIVariable,
                DIGlobalVariable, DIArray, DISubrange, DITemplateTypeParameter, DIEnumerator,
                DINameSpace, DIFlags};

use libc::{c_uint, c_int, size_t, c_char};
use libc::{c_longlong, c_ulonglong, c_void};

use RustStringRef;

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
    PtxKernel = 71,
    X86_64_SysV = 78,
    X86_64_Win64 = 79,
    X86_VectorCall = 80,
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
    Default = 0,
    Static = 1,
    PIC = 2,
    DynamicNoPic = 3,
}

/// LLVMRustCodeModel
#[derive(Copy, Clone)]
#[repr(C)]
pub enum CodeModel {
    Other,
    Default,
    JITDefault,
    Small,
    Kernel,
    Medium,
    Large,
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
}

/// LLVMRustArchiveKind
#[derive(Copy, Clone)]
#[repr(C)]
pub enum ArchiveKind {
    Other,
    K_GNU,
    K_MIPS64,
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

// Opaque pointer types
#[allow(missing_copy_implementations)]
pub enum Module_opaque {}
pub type ModuleRef = *mut Module_opaque;
#[allow(missing_copy_implementations)]
pub enum Context_opaque {}
pub type ContextRef = *mut Context_opaque;
#[allow(missing_copy_implementations)]
pub enum Type_opaque {}
pub type TypeRef = *mut Type_opaque;
#[allow(missing_copy_implementations)]
pub enum Value_opaque {}
pub type ValueRef = *mut Value_opaque;
#[allow(missing_copy_implementations)]
pub enum Metadata_opaque {}
pub type MetadataRef = *mut Metadata_opaque;
#[allow(missing_copy_implementations)]
pub enum BasicBlock_opaque {}
pub type BasicBlockRef = *mut BasicBlock_opaque;
#[allow(missing_copy_implementations)]
pub enum Builder_opaque {}
pub type BuilderRef = *mut Builder_opaque;
#[allow(missing_copy_implementations)]
pub enum ExecutionEngine_opaque {}
pub type ExecutionEngineRef = *mut ExecutionEngine_opaque;
#[allow(missing_copy_implementations)]
pub enum MemoryBuffer_opaque {}
pub type MemoryBufferRef = *mut MemoryBuffer_opaque;
#[allow(missing_copy_implementations)]
pub enum PassManager_opaque {}
pub type PassManagerRef = *mut PassManager_opaque;
#[allow(missing_copy_implementations)]
pub enum PassManagerBuilder_opaque {}
pub type PassManagerBuilderRef = *mut PassManagerBuilder_opaque;
#[allow(missing_copy_implementations)]
pub enum Use_opaque {}
pub type UseRef = *mut Use_opaque;
#[allow(missing_copy_implementations)]
pub enum TargetData_opaque {}
pub type TargetDataRef = *mut TargetData_opaque;
#[allow(missing_copy_implementations)]
pub enum ObjectFile_opaque {}
pub type ObjectFileRef = *mut ObjectFile_opaque;
#[allow(missing_copy_implementations)]
pub enum SectionIterator_opaque {}
pub type SectionIteratorRef = *mut SectionIterator_opaque;
#[allow(missing_copy_implementations)]
pub enum Pass_opaque {}
pub type PassRef = *mut Pass_opaque;
#[allow(missing_copy_implementations)]
pub enum TargetMachine_opaque {}
pub type TargetMachineRef = *mut TargetMachine_opaque;
pub enum Archive_opaque {}
pub type ArchiveRef = *mut Archive_opaque;
pub enum ArchiveIterator_opaque {}
pub type ArchiveIteratorRef = *mut ArchiveIterator_opaque;
pub enum ArchiveChild_opaque {}
pub type ArchiveChildRef = *mut ArchiveChild_opaque;
#[allow(missing_copy_implementations)]
pub enum Twine_opaque {}
pub type TwineRef = *mut Twine_opaque;
#[allow(missing_copy_implementations)]
pub enum DiagnosticInfo_opaque {}
pub type DiagnosticInfoRef = *mut DiagnosticInfo_opaque;
#[allow(missing_copy_implementations)]
pub enum DebugLoc_opaque {}
pub type DebugLocRef = *mut DebugLoc_opaque;
#[allow(missing_copy_implementations)]
pub enum SMDiagnostic_opaque {}
pub type SMDiagnosticRef = *mut SMDiagnostic_opaque;
#[allow(missing_copy_implementations)]
pub enum RustArchiveMember_opaque {}
pub type RustArchiveMemberRef = *mut RustArchiveMember_opaque;
#[allow(missing_copy_implementations)]
pub enum OperandBundleDef_opaque {}
pub type OperandBundleDefRef = *mut OperandBundleDef_opaque;

pub type DiagnosticHandler = unsafe extern "C" fn(DiagnosticInfoRef, *mut c_void);
pub type InlineAsmDiagHandler = unsafe extern "C" fn(SMDiagnosticRef, *const c_void, c_uint);


pub mod debuginfo {
    use super::MetadataRef;

    #[allow(missing_copy_implementations)]
    pub enum DIBuilder_opaque {}
    pub type DIBuilderRef = *mut DIBuilder_opaque;

    pub type DIDescriptor = MetadataRef;
    pub type DIScope = DIDescriptor;
    pub type DILocation = DIDescriptor;
    pub type DIFile = DIScope;
    pub type DILexicalBlock = DIScope;
    pub type DISubprogram = DIScope;
    pub type DINameSpace = DIScope;
    pub type DIType = DIDescriptor;
    pub type DIBasicType = DIType;
    pub type DIDerivedType = DIType;
    pub type DICompositeType = DIDerivedType;
    pub type DIVariable = DIDescriptor;
    pub type DIGlobalVariable = DIDescriptor;
    pub type DIArray = DIDescriptor;
    pub type DISubrange = DIDescriptor;
    pub type DIEnumerator = DIDescriptor;
    pub type DITemplateTypeParameter = DIDescriptor;

    // These values **must** match with LLVMRustDIFlags!!
    bitflags! {
        #[repr(C)]
        #[derive(Debug, Default)]
        flags DIFlags: ::libc::uint32_t {
            const FlagZero                = 0,
            const FlagPrivate             = 1,
            const FlagProtected           = 2,
            const FlagPublic              = 3,
            const FlagFwdDecl             = (1 << 2),
            const FlagAppleBlock          = (1 << 3),
            const FlagBlockByrefStruct    = (1 << 4),
            const FlagVirtual             = (1 << 5),
            const FlagArtificial          = (1 << 6),
            const FlagExplicit            = (1 << 7),
            const FlagPrototyped          = (1 << 8),
            const FlagObjcClassComplete   = (1 << 9),
            const FlagObjectPointer       = (1 << 10),
            const FlagVector              = (1 << 11),
            const FlagStaticMember        = (1 << 12),
            const FlagLValueReference     = (1 << 13),
            const FlagRValueReference     = (1 << 14),
        }
    }
}


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
#[link(name = "rustllvm", kind = "static")] // not quite true but good enough
extern "C" {
    // Create and destroy contexts.
    pub fn LLVMContextCreate() -> ContextRef;
    pub fn LLVMContextDispose(C: ContextRef);
    pub fn LLVMGetMDKindIDInContext(C: ContextRef, Name: *const c_char, SLen: c_uint) -> c_uint;

    // Create and destroy modules.
    pub fn LLVMModuleCreateWithNameInContext(ModuleID: *const c_char, C: ContextRef) -> ModuleRef;
    pub fn LLVMGetModuleContext(M: ModuleRef) -> ContextRef;
    pub fn LLVMCloneModule(M: ModuleRef) -> ModuleRef;
    pub fn LLVMDisposeModule(M: ModuleRef);

    /// Data layout. See Module::getDataLayout.
    pub fn LLVMGetDataLayout(M: ModuleRef) -> *const c_char;
    pub fn LLVMSetDataLayout(M: ModuleRef, Triple: *const c_char);

    /// See Module::dump.
    pub fn LLVMDumpModule(M: ModuleRef);

    /// See Module::setModuleInlineAsm.
    pub fn LLVMSetModuleInlineAsm(M: ModuleRef, Asm: *const c_char);

    /// See llvm::LLVMTypeKind::getTypeID.
    pub fn LLVMRustGetTypeKind(Ty: TypeRef) -> TypeKind;

    /// See llvm::Value::getContext
    pub fn LLVMRustGetValueContext(V: ValueRef) -> ContextRef;

    // Operations on integer types
    pub fn LLVMInt1TypeInContext(C: ContextRef) -> TypeRef;
    pub fn LLVMInt8TypeInContext(C: ContextRef) -> TypeRef;
    pub fn LLVMInt16TypeInContext(C: ContextRef) -> TypeRef;
    pub fn LLVMInt32TypeInContext(C: ContextRef) -> TypeRef;
    pub fn LLVMInt64TypeInContext(C: ContextRef) -> TypeRef;
    pub fn LLVMIntTypeInContext(C: ContextRef, NumBits: c_uint) -> TypeRef;

    pub fn LLVMGetIntTypeWidth(IntegerTy: TypeRef) -> c_uint;

    // Operations on real types
    pub fn LLVMFloatTypeInContext(C: ContextRef) -> TypeRef;
    pub fn LLVMDoubleTypeInContext(C: ContextRef) -> TypeRef;

    // Operations on function types
    pub fn LLVMFunctionType(ReturnType: TypeRef,
                            ParamTypes: *const TypeRef,
                            ParamCount: c_uint,
                            IsVarArg: Bool)
                            -> TypeRef;
    pub fn LLVMGetReturnType(FunctionTy: TypeRef) -> TypeRef;
    pub fn LLVMCountParamTypes(FunctionTy: TypeRef) -> c_uint;
    pub fn LLVMGetParamTypes(FunctionTy: TypeRef, Dest: *mut TypeRef);

    // Operations on struct types
    pub fn LLVMStructTypeInContext(C: ContextRef,
                                   ElementTypes: *const TypeRef,
                                   ElementCount: c_uint,
                                   Packed: Bool)
                                   -> TypeRef;
    pub fn LLVMCountStructElementTypes(StructTy: TypeRef) -> c_uint;
    pub fn LLVMGetStructElementTypes(StructTy: TypeRef, Dest: *mut TypeRef);
    pub fn LLVMIsPackedStruct(StructTy: TypeRef) -> Bool;

    // Operations on array, pointer, and vector types (sequence types)
    pub fn LLVMRustArrayType(ElementType: TypeRef, ElementCount: u64) -> TypeRef;
    pub fn LLVMPointerType(ElementType: TypeRef, AddressSpace: c_uint) -> TypeRef;
    pub fn LLVMVectorType(ElementType: TypeRef, ElementCount: c_uint) -> TypeRef;

    pub fn LLVMGetElementType(Ty: TypeRef) -> TypeRef;
    pub fn LLVMGetArrayLength(ArrayTy: TypeRef) -> c_uint;
    pub fn LLVMGetVectorSize(VectorTy: TypeRef) -> c_uint;

    // Operations on other types
    pub fn LLVMVoidTypeInContext(C: ContextRef) -> TypeRef;
    pub fn LLVMRustMetadataTypeInContext(C: ContextRef) -> TypeRef;

    // Operations on all values
    pub fn LLVMTypeOf(Val: ValueRef) -> TypeRef;
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
    pub fn LLVMConstNull(Ty: TypeRef) -> ValueRef;
    pub fn LLVMConstICmp(Pred: IntPredicate, V1: ValueRef, V2: ValueRef) -> ValueRef;
    pub fn LLVMConstFCmp(Pred: RealPredicate, V1: ValueRef, V2: ValueRef) -> ValueRef;
    // only for isize/vector
    pub fn LLVMGetUndef(Ty: TypeRef) -> ValueRef;
    pub fn LLVMIsNull(Val: ValueRef) -> Bool;
    pub fn LLVMIsUndef(Val: ValueRef) -> Bool;

    // Operations on metadata
    pub fn LLVMMDNodeInContext(C: ContextRef, Vals: *const ValueRef, Count: c_uint) -> ValueRef;

    // Operations on scalar constants
    pub fn LLVMConstInt(IntTy: TypeRef, N: c_ulonglong, SignExtend: Bool) -> ValueRef;
    pub fn LLVMConstIntOfArbitraryPrecision(IntTy: TypeRef, Wn: c_uint, Ws: *const u64) -> ValueRef;
    pub fn LLVMConstReal(RealTy: TypeRef, N: f64) -> ValueRef;
    pub fn LLVMConstIntGetZExtValue(ConstantVal: ValueRef) -> c_ulonglong;
    pub fn LLVMConstIntGetSExtValue(ConstantVal: ValueRef) -> c_longlong;
    pub fn LLVMRustConstInt128Get(ConstantVal: ValueRef, SExt: bool,
                                  high: *mut u64, low: *mut u64) -> bool;


    // Operations on composite constants
    pub fn LLVMConstStringInContext(C: ContextRef,
                                    Str: *const c_char,
                                    Length: c_uint,
                                    DontNullTerminate: Bool)
                                    -> ValueRef;
    pub fn LLVMConstStructInContext(C: ContextRef,
                                    ConstantVals: *const ValueRef,
                                    Count: c_uint,
                                    Packed: Bool)
                                    -> ValueRef;

    pub fn LLVMConstArray(ElementTy: TypeRef,
                          ConstantVals: *const ValueRef,
                          Length: c_uint)
                          -> ValueRef;
    pub fn LLVMConstVector(ScalarConstantVals: *const ValueRef, Size: c_uint) -> ValueRef;

    // Constant expressions
    pub fn LLVMSizeOf(Ty: TypeRef) -> ValueRef;
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
    pub fn LLVMConstTrunc(ConstantVal: ValueRef, ToType: TypeRef) -> ValueRef;
    pub fn LLVMConstZExt(ConstantVal: ValueRef, ToType: TypeRef) -> ValueRef;
    pub fn LLVMConstUIToFP(ConstantVal: ValueRef, ToType: TypeRef) -> ValueRef;
    pub fn LLVMConstSIToFP(ConstantVal: ValueRef, ToType: TypeRef) -> ValueRef;
    pub fn LLVMConstFPToUI(ConstantVal: ValueRef, ToType: TypeRef) -> ValueRef;
    pub fn LLVMConstFPToSI(ConstantVal: ValueRef, ToType: TypeRef) -> ValueRef;
    pub fn LLVMConstPtrToInt(ConstantVal: ValueRef, ToType: TypeRef) -> ValueRef;
    pub fn LLVMConstIntToPtr(ConstantVal: ValueRef, ToType: TypeRef) -> ValueRef;
    pub fn LLVMConstBitCast(ConstantVal: ValueRef, ToType: TypeRef) -> ValueRef;
    pub fn LLVMConstPointerCast(ConstantVal: ValueRef, ToType: TypeRef) -> ValueRef;
    pub fn LLVMConstIntCast(ConstantVal: ValueRef, ToType: TypeRef, isSigned: Bool) -> ValueRef;
    pub fn LLVMConstFPCast(ConstantVal: ValueRef, ToType: TypeRef) -> ValueRef;
    pub fn LLVMConstExtractValue(AggConstant: ValueRef,
                                 IdxList: *const c_uint,
                                 NumIdx: c_uint)
                                 -> ValueRef;
    pub fn LLVMConstInlineAsm(Ty: TypeRef,
                              AsmString: *const c_char,
                              Constraints: *const c_char,
                              HasSideEffects: Bool,
                              IsAlignStack: Bool)
                              -> ValueRef;


    // Operations on global variables, functions, and aliases (globals)
    pub fn LLVMGetGlobalParent(Global: ValueRef) -> ModuleRef;
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
    pub fn LLVMAddGlobal(M: ModuleRef, Ty: TypeRef, Name: *const c_char) -> ValueRef;
    pub fn LLVMGetNamedGlobal(M: ModuleRef, Name: *const c_char) -> ValueRef;
    pub fn LLVMRustGetOrInsertGlobal(M: ModuleRef, Name: *const c_char, T: TypeRef) -> ValueRef;
    pub fn LLVMGetFirstGlobal(M: ModuleRef) -> ValueRef;
    pub fn LLVMGetNextGlobal(GlobalVar: ValueRef) -> ValueRef;
    pub fn LLVMDeleteGlobal(GlobalVar: ValueRef);
    pub fn LLVMGetInitializer(GlobalVar: ValueRef) -> ValueRef;
    pub fn LLVMSetInitializer(GlobalVar: ValueRef, ConstantVal: ValueRef);
    pub fn LLVMSetThreadLocal(GlobalVar: ValueRef, IsThreadLocal: Bool);
    pub fn LLVMIsGlobalConstant(GlobalVar: ValueRef) -> Bool;
    pub fn LLVMSetGlobalConstant(GlobalVar: ValueRef, IsConstant: Bool);
    pub fn LLVMRustGetNamedValue(M: ModuleRef, Name: *const c_char) -> ValueRef;

    // Operations on functions
    pub fn LLVMAddFunction(M: ModuleRef, Name: *const c_char, FunctionTy: TypeRef) -> ValueRef;
    pub fn LLVMGetNamedFunction(M: ModuleRef, Name: *const c_char) -> ValueRef;
    pub fn LLVMGetFirstFunction(M: ModuleRef) -> ValueRef;
    pub fn LLVMGetNextFunction(Fn: ValueRef) -> ValueRef;
    pub fn LLVMRustGetOrInsertFunction(M: ModuleRef,
                                       Name: *const c_char,
                                       FunctionTy: TypeRef)
                                       -> ValueRef;
    pub fn LLVMSetFunctionCallConv(Fn: ValueRef, CC: c_uint);
    pub fn LLVMRustAddDereferenceableAttr(Fn: ValueRef, index: c_uint, bytes: u64);
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
    pub fn LLVMAppendBasicBlockInContext(C: ContextRef,
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
    pub fn LLVMRustAddDereferenceableCallSiteAttr(Instr: ValueRef, index: c_uint, bytes: u64);

    // Operations on load/store instructions (only)
    pub fn LLVMSetVolatile(MemoryAccessInst: ValueRef, volatile: Bool);

    // Operations on phi nodes
    pub fn LLVMAddIncoming(PhiNode: ValueRef,
                           IncomingValues: *const ValueRef,
                           IncomingBlocks: *const BasicBlockRef,
                           Count: c_uint);

    // Instruction builders
    pub fn LLVMCreateBuilderInContext(C: ContextRef) -> BuilderRef;
    pub fn LLVMPositionBuilder(Builder: BuilderRef, Block: BasicBlockRef, Instr: ValueRef);
    pub fn LLVMPositionBuilderBefore(Builder: BuilderRef, Instr: ValueRef);
    pub fn LLVMPositionBuilderAtEnd(Builder: BuilderRef, Block: BasicBlockRef);
    pub fn LLVMGetInsertBlock(Builder: BuilderRef) -> BasicBlockRef;
    pub fn LLVMDisposeBuilder(Builder: BuilderRef);

    // Metadata
    pub fn LLVMSetCurrentDebugLocation(Builder: BuilderRef, L: ValueRef);
    pub fn LLVMGetCurrentDebugLocation(Builder: BuilderRef) -> ValueRef;
    pub fn LLVMSetInstDebugLocation(Builder: BuilderRef, Inst: ValueRef);

    // Terminators
    pub fn LLVMBuildRetVoid(B: BuilderRef) -> ValueRef;
    pub fn LLVMBuildRet(B: BuilderRef, V: ValueRef) -> ValueRef;
    pub fn LLVMBuildAggregateRet(B: BuilderRef, RetVals: *const ValueRef, N: c_uint) -> ValueRef;
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
    pub fn LLVMBuildIndirectBr(B: BuilderRef, Addr: ValueRef, NumDests: c_uint) -> ValueRef;
    pub fn LLVMRustBuildInvoke(B: BuilderRef,
                               Fn: ValueRef,
                               Args: *const ValueRef,
                               NumArgs: c_uint,
                               Then: BasicBlockRef,
                               Catch: BasicBlockRef,
                               Bundle: OperandBundleDefRef,
                               Name: *const c_char)
                               -> ValueRef;
    pub fn LLVMRustBuildLandingPad(B: BuilderRef,
                                   Ty: TypeRef,
                                   PersFn: ValueRef,
                                   NumClauses: c_uint,
                                   Name: *const c_char,
                                   F: ValueRef)
                                   -> ValueRef;
    pub fn LLVMBuildResume(B: BuilderRef, Exn: ValueRef) -> ValueRef;
    pub fn LLVMBuildUnreachable(B: BuilderRef) -> ValueRef;

    pub fn LLVMRustBuildCleanupPad(B: BuilderRef,
                                   ParentPad: ValueRef,
                                   ArgCnt: c_uint,
                                   Args: *const ValueRef,
                                   Name: *const c_char)
                                   -> ValueRef;
    pub fn LLVMRustBuildCleanupRet(B: BuilderRef,
                                   CleanupPad: ValueRef,
                                   UnwindBB: BasicBlockRef)
                                   -> ValueRef;
    pub fn LLVMRustBuildCatchPad(B: BuilderRef,
                                 ParentPad: ValueRef,
                                 ArgCnt: c_uint,
                                 Args: *const ValueRef,
                                 Name: *const c_char)
                                 -> ValueRef;
    pub fn LLVMRustBuildCatchRet(B: BuilderRef, Pad: ValueRef, BB: BasicBlockRef) -> ValueRef;
    pub fn LLVMRustBuildCatchSwitch(Builder: BuilderRef,
                                    ParentPad: ValueRef,
                                    BB: BasicBlockRef,
                                    NumHandlers: c_uint,
                                    Name: *const c_char)
                                    -> ValueRef;
    pub fn LLVMRustAddHandler(CatchSwitch: ValueRef, Handler: BasicBlockRef);
    pub fn LLVMRustSetPersonalityFn(B: BuilderRef, Pers: ValueRef);

    // Add a case to the switch instruction
    pub fn LLVMAddCase(Switch: ValueRef, OnVal: ValueRef, Dest: BasicBlockRef);

    // Add a clause to the landing pad instruction
    pub fn LLVMAddClause(LandingPad: ValueRef, ClauseVal: ValueRef);

    // Set the cleanup on a landing pad instruction
    pub fn LLVMSetCleanup(LandingPad: ValueRef, Val: Bool);

    // Arithmetic
    pub fn LLVMBuildAdd(B: BuilderRef,
                        LHS: ValueRef,
                        RHS: ValueRef,
                        Name: *const c_char)
                        -> ValueRef;
    pub fn LLVMBuildNSWAdd(B: BuilderRef,
                           LHS: ValueRef,
                           RHS: ValueRef,
                           Name: *const c_char)
                           -> ValueRef;
    pub fn LLVMBuildNUWAdd(B: BuilderRef,
                           LHS: ValueRef,
                           RHS: ValueRef,
                           Name: *const c_char)
                           -> ValueRef;
    pub fn LLVMBuildFAdd(B: BuilderRef,
                         LHS: ValueRef,
                         RHS: ValueRef,
                         Name: *const c_char)
                         -> ValueRef;
    pub fn LLVMBuildSub(B: BuilderRef,
                        LHS: ValueRef,
                        RHS: ValueRef,
                        Name: *const c_char)
                        -> ValueRef;
    pub fn LLVMBuildNSWSub(B: BuilderRef,
                           LHS: ValueRef,
                           RHS: ValueRef,
                           Name: *const c_char)
                           -> ValueRef;
    pub fn LLVMBuildNUWSub(B: BuilderRef,
                           LHS: ValueRef,
                           RHS: ValueRef,
                           Name: *const c_char)
                           -> ValueRef;
    pub fn LLVMBuildFSub(B: BuilderRef,
                         LHS: ValueRef,
                         RHS: ValueRef,
                         Name: *const c_char)
                         -> ValueRef;
    pub fn LLVMBuildMul(B: BuilderRef,
                        LHS: ValueRef,
                        RHS: ValueRef,
                        Name: *const c_char)
                        -> ValueRef;
    pub fn LLVMBuildNSWMul(B: BuilderRef,
                           LHS: ValueRef,
                           RHS: ValueRef,
                           Name: *const c_char)
                           -> ValueRef;
    pub fn LLVMBuildNUWMul(B: BuilderRef,
                           LHS: ValueRef,
                           RHS: ValueRef,
                           Name: *const c_char)
                           -> ValueRef;
    pub fn LLVMBuildFMul(B: BuilderRef,
                         LHS: ValueRef,
                         RHS: ValueRef,
                         Name: *const c_char)
                         -> ValueRef;
    pub fn LLVMBuildUDiv(B: BuilderRef,
                         LHS: ValueRef,
                         RHS: ValueRef,
                         Name: *const c_char)
                         -> ValueRef;
    pub fn LLVMBuildSDiv(B: BuilderRef,
                         LHS: ValueRef,
                         RHS: ValueRef,
                         Name: *const c_char)
                         -> ValueRef;
    pub fn LLVMBuildExactSDiv(B: BuilderRef,
                              LHS: ValueRef,
                              RHS: ValueRef,
                              Name: *const c_char)
                              -> ValueRef;
    pub fn LLVMBuildFDiv(B: BuilderRef,
                         LHS: ValueRef,
                         RHS: ValueRef,
                         Name: *const c_char)
                         -> ValueRef;
    pub fn LLVMBuildURem(B: BuilderRef,
                         LHS: ValueRef,
                         RHS: ValueRef,
                         Name: *const c_char)
                         -> ValueRef;
    pub fn LLVMBuildSRem(B: BuilderRef,
                         LHS: ValueRef,
                         RHS: ValueRef,
                         Name: *const c_char)
                         -> ValueRef;
    pub fn LLVMBuildFRem(B: BuilderRef,
                         LHS: ValueRef,
                         RHS: ValueRef,
                         Name: *const c_char)
                         -> ValueRef;
    pub fn LLVMBuildShl(B: BuilderRef,
                        LHS: ValueRef,
                        RHS: ValueRef,
                        Name: *const c_char)
                        -> ValueRef;
    pub fn LLVMBuildLShr(B: BuilderRef,
                         LHS: ValueRef,
                         RHS: ValueRef,
                         Name: *const c_char)
                         -> ValueRef;
    pub fn LLVMBuildAShr(B: BuilderRef,
                         LHS: ValueRef,
                         RHS: ValueRef,
                         Name: *const c_char)
                         -> ValueRef;
    pub fn LLVMBuildAnd(B: BuilderRef,
                        LHS: ValueRef,
                        RHS: ValueRef,
                        Name: *const c_char)
                        -> ValueRef;
    pub fn LLVMBuildOr(B: BuilderRef,
                       LHS: ValueRef,
                       RHS: ValueRef,
                       Name: *const c_char)
                       -> ValueRef;
    pub fn LLVMBuildXor(B: BuilderRef,
                        LHS: ValueRef,
                        RHS: ValueRef,
                        Name: *const c_char)
                        -> ValueRef;
    pub fn LLVMBuildBinOp(B: BuilderRef,
                          Op: Opcode,
                          LHS: ValueRef,
                          RHS: ValueRef,
                          Name: *const c_char)
                          -> ValueRef;
    pub fn LLVMBuildNeg(B: BuilderRef, V: ValueRef, Name: *const c_char) -> ValueRef;
    pub fn LLVMBuildNSWNeg(B: BuilderRef, V: ValueRef, Name: *const c_char) -> ValueRef;
    pub fn LLVMBuildNUWNeg(B: BuilderRef, V: ValueRef, Name: *const c_char) -> ValueRef;
    pub fn LLVMBuildFNeg(B: BuilderRef, V: ValueRef, Name: *const c_char) -> ValueRef;
    pub fn LLVMBuildNot(B: BuilderRef, V: ValueRef, Name: *const c_char) -> ValueRef;
    pub fn LLVMRustSetHasUnsafeAlgebra(Instr: ValueRef);

    // Memory
    pub fn LLVMBuildAlloca(B: BuilderRef, Ty: TypeRef, Name: *const c_char) -> ValueRef;
    pub fn LLVMBuildFree(B: BuilderRef, PointerVal: ValueRef) -> ValueRef;
    pub fn LLVMBuildLoad(B: BuilderRef, PointerVal: ValueRef, Name: *const c_char) -> ValueRef;

    pub fn LLVMBuildStore(B: BuilderRef, Val: ValueRef, Ptr: ValueRef) -> ValueRef;

    pub fn LLVMBuildGEP(B: BuilderRef,
                        Pointer: ValueRef,
                        Indices: *const ValueRef,
                        NumIndices: c_uint,
                        Name: *const c_char)
                        -> ValueRef;
    pub fn LLVMBuildInBoundsGEP(B: BuilderRef,
                                Pointer: ValueRef,
                                Indices: *const ValueRef,
                                NumIndices: c_uint,
                                Name: *const c_char)
                                -> ValueRef;
    pub fn LLVMBuildStructGEP(B: BuilderRef,
                              Pointer: ValueRef,
                              Idx: c_uint,
                              Name: *const c_char)
                              -> ValueRef;
    pub fn LLVMBuildGlobalString(B: BuilderRef,
                                 Str: *const c_char,
                                 Name: *const c_char)
                                 -> ValueRef;
    pub fn LLVMBuildGlobalStringPtr(B: BuilderRef,
                                    Str: *const c_char,
                                    Name: *const c_char)
                                    -> ValueRef;

    // Casts
    pub fn LLVMBuildTrunc(B: BuilderRef,
                          Val: ValueRef,
                          DestTy: TypeRef,
                          Name: *const c_char)
                          -> ValueRef;
    pub fn LLVMBuildZExt(B: BuilderRef,
                         Val: ValueRef,
                         DestTy: TypeRef,
                         Name: *const c_char)
                         -> ValueRef;
    pub fn LLVMBuildSExt(B: BuilderRef,
                         Val: ValueRef,
                         DestTy: TypeRef,
                         Name: *const c_char)
                         -> ValueRef;
    pub fn LLVMBuildFPToUI(B: BuilderRef,
                           Val: ValueRef,
                           DestTy: TypeRef,
                           Name: *const c_char)
                           -> ValueRef;
    pub fn LLVMBuildFPToSI(B: BuilderRef,
                           Val: ValueRef,
                           DestTy: TypeRef,
                           Name: *const c_char)
                           -> ValueRef;
    pub fn LLVMBuildUIToFP(B: BuilderRef,
                           Val: ValueRef,
                           DestTy: TypeRef,
                           Name: *const c_char)
                           -> ValueRef;
    pub fn LLVMBuildSIToFP(B: BuilderRef,
                           Val: ValueRef,
                           DestTy: TypeRef,
                           Name: *const c_char)
                           -> ValueRef;
    pub fn LLVMBuildFPTrunc(B: BuilderRef,
                            Val: ValueRef,
                            DestTy: TypeRef,
                            Name: *const c_char)
                            -> ValueRef;
    pub fn LLVMBuildFPExt(B: BuilderRef,
                          Val: ValueRef,
                          DestTy: TypeRef,
                          Name: *const c_char)
                          -> ValueRef;
    pub fn LLVMBuildPtrToInt(B: BuilderRef,
                             Val: ValueRef,
                             DestTy: TypeRef,
                             Name: *const c_char)
                             -> ValueRef;
    pub fn LLVMBuildIntToPtr(B: BuilderRef,
                             Val: ValueRef,
                             DestTy: TypeRef,
                             Name: *const c_char)
                             -> ValueRef;
    pub fn LLVMBuildBitCast(B: BuilderRef,
                            Val: ValueRef,
                            DestTy: TypeRef,
                            Name: *const c_char)
                            -> ValueRef;
    pub fn LLVMBuildZExtOrBitCast(B: BuilderRef,
                                  Val: ValueRef,
                                  DestTy: TypeRef,
                                  Name: *const c_char)
                                  -> ValueRef;
    pub fn LLVMBuildSExtOrBitCast(B: BuilderRef,
                                  Val: ValueRef,
                                  DestTy: TypeRef,
                                  Name: *const c_char)
                                  -> ValueRef;
    pub fn LLVMBuildTruncOrBitCast(B: BuilderRef,
                                   Val: ValueRef,
                                   DestTy: TypeRef,
                                   Name: *const c_char)
                                   -> ValueRef;
    pub fn LLVMBuildCast(B: BuilderRef,
                         Op: Opcode,
                         Val: ValueRef,
                         DestTy: TypeRef,
                         Name: *const c_char)
                         -> ValueRef;
    pub fn LLVMBuildPointerCast(B: BuilderRef,
                                Val: ValueRef,
                                DestTy: TypeRef,
                                Name: *const c_char)
                                -> ValueRef;
    pub fn LLVMBuildIntCast(B: BuilderRef,
                            Val: ValueRef,
                            DestTy: TypeRef,
                            Name: *const c_char)
                            -> ValueRef;
    pub fn LLVMBuildFPCast(B: BuilderRef,
                           Val: ValueRef,
                           DestTy: TypeRef,
                           Name: *const c_char)
                           -> ValueRef;

    // Comparisons
    pub fn LLVMBuildICmp(B: BuilderRef,
                         Op: c_uint,
                         LHS: ValueRef,
                         RHS: ValueRef,
                         Name: *const c_char)
                         -> ValueRef;
    pub fn LLVMBuildFCmp(B: BuilderRef,
                         Op: c_uint,
                         LHS: ValueRef,
                         RHS: ValueRef,
                         Name: *const c_char)
                         -> ValueRef;

    // Miscellaneous instructions
    pub fn LLVMBuildPhi(B: BuilderRef, Ty: TypeRef, Name: *const c_char) -> ValueRef;
    pub fn LLVMRustBuildCall(B: BuilderRef,
                             Fn: ValueRef,
                             Args: *const ValueRef,
                             NumArgs: c_uint,
                             Bundle: OperandBundleDefRef,
                             Name: *const c_char)
                             -> ValueRef;
    pub fn LLVMBuildSelect(B: BuilderRef,
                           If: ValueRef,
                           Then: ValueRef,
                           Else: ValueRef,
                           Name: *const c_char)
                           -> ValueRef;
    pub fn LLVMBuildVAArg(B: BuilderRef,
                          list: ValueRef,
                          Ty: TypeRef,
                          Name: *const c_char)
                          -> ValueRef;
    pub fn LLVMBuildExtractElement(B: BuilderRef,
                                   VecVal: ValueRef,
                                   Index: ValueRef,
                                   Name: *const c_char)
                                   -> ValueRef;
    pub fn LLVMBuildInsertElement(B: BuilderRef,
                                  VecVal: ValueRef,
                                  EltVal: ValueRef,
                                  Index: ValueRef,
                                  Name: *const c_char)
                                  -> ValueRef;
    pub fn LLVMBuildShuffleVector(B: BuilderRef,
                                  V1: ValueRef,
                                  V2: ValueRef,
                                  Mask: ValueRef,
                                  Name: *const c_char)
                                  -> ValueRef;
    pub fn LLVMBuildExtractValue(B: BuilderRef,
                                 AggVal: ValueRef,
                                 Index: c_uint,
                                 Name: *const c_char)
                                 -> ValueRef;
    pub fn LLVMBuildInsertValue(B: BuilderRef,
                                AggVal: ValueRef,
                                EltVal: ValueRef,
                                Index: c_uint,
                                Name: *const c_char)
                                -> ValueRef;

    pub fn LLVMBuildIsNull(B: BuilderRef, Val: ValueRef, Name: *const c_char) -> ValueRef;
    pub fn LLVMBuildIsNotNull(B: BuilderRef, Val: ValueRef, Name: *const c_char) -> ValueRef;
    pub fn LLVMBuildPtrDiff(B: BuilderRef,
                            LHS: ValueRef,
                            RHS: ValueRef,
                            Name: *const c_char)
                            -> ValueRef;

    // Atomic Operations
    pub fn LLVMRustBuildAtomicLoad(B: BuilderRef,
                                   PointerVal: ValueRef,
                                   Name: *const c_char,
                                   Order: AtomicOrdering,
                                   Alignment: c_uint)
                                   -> ValueRef;

    pub fn LLVMRustBuildAtomicStore(B: BuilderRef,
                                    Val: ValueRef,
                                    Ptr: ValueRef,
                                    Order: AtomicOrdering,
                                    Alignment: c_uint)
                                    -> ValueRef;

    pub fn LLVMRustBuildAtomicCmpXchg(B: BuilderRef,
                                      LHS: ValueRef,
                                      CMP: ValueRef,
                                      RHS: ValueRef,
                                      Order: AtomicOrdering,
                                      FailureOrder: AtomicOrdering,
                                      Weak: Bool)
                                      -> ValueRef;

    pub fn LLVMBuildAtomicRMW(B: BuilderRef,
                              Op: AtomicRmwBinOp,
                              LHS: ValueRef,
                              RHS: ValueRef,
                              Order: AtomicOrdering,
                              SingleThreaded: Bool)
                              -> ValueRef;

    pub fn LLVMRustBuildAtomicFence(B: BuilderRef,
                                    Order: AtomicOrdering,
                                    Scope: SynchronizationScope);


    // Selected entries from the downcasts.
    pub fn LLVMIsATerminatorInst(Inst: ValueRef) -> ValueRef;
    pub fn LLVMIsAStoreInst(Inst: ValueRef) -> ValueRef;

    /// Writes a module to the specified path. Returns 0 on success.
    pub fn LLVMWriteBitcodeToFile(M: ModuleRef, Path: *const c_char) -> c_int;

    /// Creates target data from a target layout string.
    pub fn LLVMCreateTargetData(StringRep: *const c_char) -> TargetDataRef;
    /// Number of bytes clobbered when doing a Store to *T.
    pub fn LLVMSizeOfTypeInBits(TD: TargetDataRef, Ty: TypeRef) -> c_ulonglong;

    /// Distance between successive elements in an array of T. Includes ABI padding.
    pub fn LLVMABISizeOfType(TD: TargetDataRef, Ty: TypeRef) -> c_ulonglong;

    /// Returns the preferred alignment of a type.
    pub fn LLVMPreferredAlignmentOfType(TD: TargetDataRef, Ty: TypeRef) -> c_uint;
    /// Returns the minimum alignment of a type.
    pub fn LLVMABIAlignmentOfType(TD: TargetDataRef, Ty: TypeRef) -> c_uint;

    /// Computes the byte offset of the indexed struct element for a
    /// target.
    pub fn LLVMOffsetOfElement(TD: TargetDataRef,
                               StructTy: TypeRef,
                               Element: c_uint)
                               -> c_ulonglong;

    /// Disposes target data.
    pub fn LLVMDisposeTargetData(TD: TargetDataRef);

    /// Creates a pass manager.
    pub fn LLVMCreatePassManager() -> PassManagerRef;

    /// Creates a function-by-function pass manager
    pub fn LLVMCreateFunctionPassManagerForModule(M: ModuleRef) -> PassManagerRef;

    /// Disposes a pass manager.
    pub fn LLVMDisposePassManager(PM: PassManagerRef);

    /// Runs a pass manager on a module.
    pub fn LLVMRunPassManager(PM: PassManagerRef, M: ModuleRef) -> Bool;

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

    pub fn LLVMStructCreateNamed(C: ContextRef, Name: *const c_char) -> TypeRef;

    pub fn LLVMStructSetBody(StructTy: TypeRef,
                             ElementTypes: *const TypeRef,
                             ElementCount: c_uint,
                             Packed: Bool);

    pub fn LLVMConstNamedStruct(S: TypeRef,
                                ConstantVals: *const ValueRef,
                                Count: c_uint)
                                -> ValueRef;

    /// Enables LLVM debug output.
    pub fn LLVMRustSetDebug(Enabled: c_int);

    /// Prepares inline assembly.
    pub fn LLVMRustInlineAsm(Ty: TypeRef,
                             AsmString: *const c_char,
                             Constraints: *const c_char,
                             SideEffects: Bool,
                             AlignStack: Bool,
                             Dialect: AsmDialect)
                             -> ValueRef;

    pub fn LLVMRustDebugMetadataVersion() -> u32;
    pub fn LLVMRustVersionMajor() -> u32;
    pub fn LLVMRustVersionMinor() -> u32;

    pub fn LLVMRustAddModuleFlag(M: ModuleRef, name: *const c_char, value: u32);

    pub fn LLVMRustDIBuilderCreate(M: ModuleRef) -> DIBuilderRef;

    pub fn LLVMRustDIBuilderDispose(Builder: DIBuilderRef);

    pub fn LLVMRustDIBuilderFinalize(Builder: DIBuilderRef);

    pub fn LLVMRustDIBuilderCreateCompileUnit(Builder: DIBuilderRef,
                                              Lang: c_uint,
                                              File: *const c_char,
                                              Dir: *const c_char,
                                              Producer: *const c_char,
                                              isOptimized: bool,
                                              Flags: *const c_char,
                                              RuntimeVer: c_uint,
                                              SplitName: *const c_char)
                                              -> DIDescriptor;

    pub fn LLVMRustDIBuilderCreateFile(Builder: DIBuilderRef,
                                       Filename: *const c_char,
                                       Directory: *const c_char)
                                       -> DIFile;

    pub fn LLVMRustDIBuilderCreateSubroutineType(Builder: DIBuilderRef,
                                                 File: DIFile,
                                                 ParameterTypes: DIArray)
                                                 -> DICompositeType;

    pub fn LLVMRustDIBuilderCreateFunction(Builder: DIBuilderRef,
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
                                           Decl: DIDescriptor)
                                           -> DISubprogram;

    pub fn LLVMRustDIBuilderCreateBasicType(Builder: DIBuilderRef,
                                            Name: *const c_char,
                                            SizeInBits: u64,
                                            AlignInBits: u64,
                                            Encoding: c_uint)
                                            -> DIBasicType;

    pub fn LLVMRustDIBuilderCreatePointerType(Builder: DIBuilderRef,
                                              PointeeTy: DIType,
                                              SizeInBits: u64,
                                              AlignInBits: u64,
                                              Name: *const c_char)
                                              -> DIDerivedType;

    pub fn LLVMRustDIBuilderCreateStructType(Builder: DIBuilderRef,
                                             Scope: DIDescriptor,
                                             Name: *const c_char,
                                             File: DIFile,
                                             LineNumber: c_uint,
                                             SizeInBits: u64,
                                             AlignInBits: u64,
                                             Flags: DIFlags,
                                             DerivedFrom: DIType,
                                             Elements: DIArray,
                                             RunTimeLang: c_uint,
                                             VTableHolder: DIType,
                                             UniqueId: *const c_char)
                                             -> DICompositeType;

    pub fn LLVMRustDIBuilderCreateMemberType(Builder: DIBuilderRef,
                                             Scope: DIDescriptor,
                                             Name: *const c_char,
                                             File: DIFile,
                                             LineNo: c_uint,
                                             SizeInBits: u64,
                                             AlignInBits: u64,
                                             OffsetInBits: u64,
                                             Flags: DIFlags,
                                             Ty: DIType)
                                             -> DIDerivedType;

    pub fn LLVMRustDIBuilderCreateLexicalBlock(Builder: DIBuilderRef,
                                               Scope: DIScope,
                                               File: DIFile,
                                               Line: c_uint,
                                               Col: c_uint)
                                               -> DILexicalBlock;

    pub fn LLVMRustDIBuilderCreateLexicalBlockFile(Builder: DIBuilderRef,
                                                   Scope: DIScope,
                                                   File: DIFile)
                                                   -> DILexicalBlock;

    pub fn LLVMRustDIBuilderCreateStaticVariable(Builder: DIBuilderRef,
                                                 Context: DIScope,
                                                 Name: *const c_char,
                                                 LinkageName: *const c_char,
                                                 File: DIFile,
                                                 LineNo: c_uint,
                                                 Ty: DIType,
                                                 isLocalToUnit: bool,
                                                 Val: ValueRef,
                                                 Decl: DIDescriptor,
                                                 AlignInBits: u64)
                                                 -> DIGlobalVariable;

    pub fn LLVMRustDIBuilderCreateVariable(Builder: DIBuilderRef,
                                           Tag: c_uint,
                                           Scope: DIDescriptor,
                                           Name: *const c_char,
                                           File: DIFile,
                                           LineNo: c_uint,
                                           Ty: DIType,
                                           AlwaysPreserve: bool,
                                           Flags: DIFlags,
                                           ArgNo: c_uint,
                                           AlignInBits: u64)
                                           -> DIVariable;

    pub fn LLVMRustDIBuilderCreateArrayType(Builder: DIBuilderRef,
                                            Size: u64,
                                            AlignInBits: u64,
                                            Ty: DIType,
                                            Subscripts: DIArray)
                                            -> DIType;

    pub fn LLVMRustDIBuilderCreateVectorType(Builder: DIBuilderRef,
                                             Size: u64,
                                             AlignInBits: u64,
                                             Ty: DIType,
                                             Subscripts: DIArray)
                                             -> DIType;

    pub fn LLVMRustDIBuilderGetOrCreateSubrange(Builder: DIBuilderRef,
                                                Lo: i64,
                                                Count: i64)
                                                -> DISubrange;

    pub fn LLVMRustDIBuilderGetOrCreateArray(Builder: DIBuilderRef,
                                             Ptr: *const DIDescriptor,
                                             Count: c_uint)
                                             -> DIArray;

    pub fn LLVMRustDIBuilderInsertDeclareAtEnd(Builder: DIBuilderRef,
                                               Val: ValueRef,
                                               VarInfo: DIVariable,
                                               AddrOps: *const i64,
                                               AddrOpsCount: c_uint,
                                               DL: ValueRef,
                                               InsertAtEnd: BasicBlockRef)
                                               -> ValueRef;

    pub fn LLVMRustDIBuilderCreateEnumerator(Builder: DIBuilderRef,
                                             Name: *const c_char,
                                             Val: u64)
                                             -> DIEnumerator;

    pub fn LLVMRustDIBuilderCreateEnumerationType(Builder: DIBuilderRef,
                                                  Scope: DIScope,
                                                  Name: *const c_char,
                                                  File: DIFile,
                                                  LineNumber: c_uint,
                                                  SizeInBits: u64,
                                                  AlignInBits: u64,
                                                  Elements: DIArray,
                                                  ClassType: DIType)
                                                  -> DIType;

    pub fn LLVMRustDIBuilderCreateUnionType(Builder: DIBuilderRef,
                                            Scope: DIScope,
                                            Name: *const c_char,
                                            File: DIFile,
                                            LineNumber: c_uint,
                                            SizeInBits: u64,
                                            AlignInBits: u64,
                                            Flags: DIFlags,
                                            Elements: DIArray,
                                            RunTimeLang: c_uint,
                                            UniqueId: *const c_char)
                                            -> DIType;

    pub fn LLVMSetUnnamedAddr(GlobalVar: ValueRef, UnnamedAddr: Bool);

    pub fn LLVMRustDIBuilderCreateTemplateTypeParameter(Builder: DIBuilderRef,
                                                        Scope: DIScope,
                                                        Name: *const c_char,
                                                        Ty: DIType,
                                                        File: DIFile,
                                                        LineNo: c_uint,
                                                        ColumnNo: c_uint)
                                                        -> DITemplateTypeParameter;


    pub fn LLVMRustDIBuilderCreateNameSpace(Builder: DIBuilderRef,
                                            Scope: DIScope,
                                            Name: *const c_char,
                                            File: DIFile,
                                            LineNo: c_uint)
                                            -> DINameSpace;
    pub fn LLVMRustDICompositeTypeSetTypeArray(Builder: DIBuilderRef,
                                               CompositeType: DIType,
                                               TypeArray: DIArray);


    pub fn LLVMRustDIBuilderCreateDebugLocation(Context: ContextRef,
                                                Line: c_uint,
                                                Column: c_uint,
                                                Scope: DIScope,
                                                InlinedAt: MetadataRef)
                                                -> ValueRef;
    pub fn LLVMRustDIBuilderCreateOpDeref() -> i64;
    pub fn LLVMRustDIBuilderCreateOpPlus() -> i64;

    pub fn LLVMRustWriteTypeToString(Type: TypeRef, s: RustStringRef);
    pub fn LLVMRustWriteValueToString(value_ref: ValueRef, s: RustStringRef);

    pub fn LLVMIsAConstantInt(value_ref: ValueRef) -> ValueRef;

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
                                       DataSections: bool)
                                       -> TargetMachineRef;
    pub fn LLVMRustDisposeTargetMachine(T: TargetMachineRef);
    pub fn LLVMRustAddAnalysisPasses(T: TargetMachineRef, PM: PassManagerRef, M: ModuleRef);
    pub fn LLVMRustAddBuilderLibraryInfo(PMB: PassManagerBuilderRef,
                                         M: ModuleRef,
                                         DisableSimplifyLibCalls: bool);
    pub fn LLVMRustConfigurePassManagerBuilder(PMB: PassManagerBuilderRef,
                                               OptLevel: CodeGenOptLevel,
                                               MergeFunctions: bool,
                                               SLPVectorize: bool,
                                               LoopVectorize: bool);
    pub fn LLVMRustAddLibraryInfo(PM: PassManagerRef,
                                  M: ModuleRef,
                                  DisableSimplifyLibCalls: bool);
    pub fn LLVMRustRunFunctionPassManager(PM: PassManagerRef, M: ModuleRef);
    pub fn LLVMRustWriteOutputFile(T: TargetMachineRef,
                                   PM: PassManagerRef,
                                   M: ModuleRef,
                                   Output: *const c_char,
                                   FileType: FileType)
                                   -> LLVMRustResult;
    pub fn LLVMRustPrintModule(PM: PassManagerRef, M: ModuleRef, Output: *const c_char);
    pub fn LLVMRustSetLLVMOptions(Argc: c_int, Argv: *const *const c_char);
    pub fn LLVMRustPrintPasses();
    pub fn LLVMRustSetNormalizedTarget(M: ModuleRef, triple: *const c_char);
    pub fn LLVMRustAddAlwaysInlinePass(P: PassManagerBuilderRef, AddLifetimes: bool);
    pub fn LLVMRustLinkInExternalBitcode(M: ModuleRef, bc: *const c_char, len: size_t) -> bool;
    pub fn LLVMRustRunRestrictionPass(M: ModuleRef, syms: *const *const c_char, len: size_t);
    pub fn LLVMRustMarkAllFunctionsNounwind(M: ModuleRef);

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

    pub fn LLVMContextSetDiagnosticHandler(C: ContextRef,
                                           Handler: DiagnosticHandler,
                                           DiagnosticContext: *mut c_void);

    pub fn LLVMRustUnpackOptimizationDiagnostic(DI: DiagnosticInfoRef,
                                                pass_name_out: RustStringRef,
                                                function_out: *mut ValueRef,
                                                debugloc_out: *mut DebugLocRef,
                                                message_out: RustStringRef);
    pub fn LLVMRustUnpackInlineAsmDiagnostic(DI: DiagnosticInfoRef,
                                             cookie_out: *mut c_uint,
                                             message_out: *mut TwineRef,
                                             instruction_out: *mut ValueRef);

    pub fn LLVMRustWriteDiagnosticInfoToString(DI: DiagnosticInfoRef, s: RustStringRef);
    pub fn LLVMRustGetDiagInfoKind(DI: DiagnosticInfoRef) -> DiagnosticKind;

    pub fn LLVMRustWriteDebugLocToString(C: ContextRef, DL: DebugLocRef, s: RustStringRef);

    pub fn LLVMRustSetInlineAsmDiagnosticHandler(C: ContextRef,
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
                                    Child: ArchiveChildRef)
                                    -> RustArchiveMemberRef;
    pub fn LLVMRustArchiveMemberFree(Member: RustArchiveMemberRef);

    pub fn LLVMRustSetDataLayoutFromTargetMachine(M: ModuleRef, TM: TargetMachineRef);
    pub fn LLVMRustGetModuleDataLayout(M: ModuleRef) -> TargetDataRef;

    pub fn LLVMRustBuildOperandBundleDef(Name: *const c_char,
                                         Inputs: *const ValueRef,
                                         NumInputs: c_uint)
                                         -> OperandBundleDefRef;
    pub fn LLVMRustFreeOperandBundleDef(Bundle: OperandBundleDefRef);

    pub fn LLVMRustPositionBuilderAtStart(B: BuilderRef, BB: BasicBlockRef);

    pub fn LLVMRustSetComdat(M: ModuleRef, V: ValueRef, Name: *const c_char);
    pub fn LLVMRustUnsetComdat(V: ValueRef);
    pub fn LLVMRustSetModulePIELevel(M: ModuleRef);
}


// LLVM requires symbols from this library, but apparently they're not printed
// during llvm-config?
#[cfg(windows)]
#[link(name = "ole32")]
extern "C" {}
