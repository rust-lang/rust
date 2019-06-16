use super::debuginfo::{
    DIBuilder, DIDescriptor, DIFile, DILexicalBlock, DISubprogram, DIType,
    DIBasicType, DIDerivedType, DICompositeType, DIScope, DIVariable,
    DIGlobalVariableExpression, DIArray, DISubrange, DITemplateTypeParameter, DIEnumerator,
    DINameSpace, DIFlags, DISPFlags, DebugEmissionKind,
};

use libc::{c_uint, c_int, size_t, c_char};
use libc::{c_ulonglong, c_void};

use std::marker::PhantomData;

use super::RustString;

pub type Bool = c_uint;

pub const True: Bool = 1 as Bool;
pub const False: Bool = 0 as Bool;

#[derive(Copy, Clone, PartialEq)]
#[repr(C)]
#[allow(dead_code)] // Variants constructed by C++.
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

/// LLVMDLLStorageClass
#[derive(Copy, Clone)]
#[repr(C)]
pub enum DLLStorageClass {
    #[allow(dead_code)]
    Default = 0,
    DllImport = 1, // Function to be imported from DLL.
    #[allow(dead_code)]
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
    NonLazyBind     = 23,
    OptimizeNone    = 24,
    ReturnsTwice    = 25,
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

impl IntPredicate {
    pub fn from_generic(intpre: rustc_codegen_ssa::common::IntPredicate) -> Self {
        match intpre {
            rustc_codegen_ssa::common::IntPredicate::IntEQ => IntPredicate::IntEQ,
            rustc_codegen_ssa::common::IntPredicate::IntNE => IntPredicate::IntNE,
            rustc_codegen_ssa::common::IntPredicate::IntUGT => IntPredicate::IntUGT,
            rustc_codegen_ssa::common::IntPredicate::IntUGE => IntPredicate::IntUGE,
            rustc_codegen_ssa::common::IntPredicate::IntULT => IntPredicate::IntULT,
            rustc_codegen_ssa::common::IntPredicate::IntULE => IntPredicate::IntULE,
            rustc_codegen_ssa::common::IntPredicate::IntSGT => IntPredicate::IntSGT,
            rustc_codegen_ssa::common::IntPredicate::IntSGE => IntPredicate::IntSGE,
            rustc_codegen_ssa::common::IntPredicate::IntSLT => IntPredicate::IntSLT,
            rustc_codegen_ssa::common::IntPredicate::IntSLE => IntPredicate::IntSLE,
        }
    }
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

impl RealPredicate {
    pub fn from_generic(realpred: rustc_codegen_ssa::common::RealPredicate) -> Self {
        match realpred {
            rustc_codegen_ssa::common::RealPredicate::RealPredicateFalse =>
                RealPredicate::RealPredicateFalse,
            rustc_codegen_ssa::common::RealPredicate::RealOEQ => RealPredicate::RealOEQ,
            rustc_codegen_ssa::common::RealPredicate::RealOGT => RealPredicate::RealOGT,
            rustc_codegen_ssa::common::RealPredicate::RealOGE => RealPredicate::RealOGE,
            rustc_codegen_ssa::common::RealPredicate::RealOLT => RealPredicate::RealOLT,
            rustc_codegen_ssa::common::RealPredicate::RealOLE => RealPredicate::RealOLE,
            rustc_codegen_ssa::common::RealPredicate::RealONE => RealPredicate::RealONE,
            rustc_codegen_ssa::common::RealPredicate::RealORD => RealPredicate::RealORD,
            rustc_codegen_ssa::common::RealPredicate::RealUNO => RealPredicate::RealUNO,
            rustc_codegen_ssa::common::RealPredicate::RealUEQ => RealPredicate::RealUEQ,
            rustc_codegen_ssa::common::RealPredicate::RealUGT => RealPredicate::RealUGT,
            rustc_codegen_ssa::common::RealPredicate::RealUGE => RealPredicate::RealUGE,
            rustc_codegen_ssa::common::RealPredicate::RealULT => RealPredicate::RealULT,
            rustc_codegen_ssa::common::RealPredicate::RealULE => RealPredicate::RealULE,
            rustc_codegen_ssa::common::RealPredicate::RealUNE => RealPredicate::RealUNE,
            rustc_codegen_ssa::common::RealPredicate::RealPredicateTrue =>
                RealPredicate::RealPredicateTrue
        }
    }
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

impl TypeKind {
    pub fn to_generic(self) -> rustc_codegen_ssa::common::TypeKind {
        match self {
            TypeKind::Void => rustc_codegen_ssa::common::TypeKind::Void,
            TypeKind::Half => rustc_codegen_ssa::common::TypeKind::Half,
            TypeKind::Float => rustc_codegen_ssa::common::TypeKind::Float,
            TypeKind::Double => rustc_codegen_ssa::common::TypeKind::Double,
            TypeKind::X86_FP80 => rustc_codegen_ssa::common::TypeKind::X86_FP80,
            TypeKind::FP128 => rustc_codegen_ssa::common::TypeKind::FP128,
            TypeKind::PPC_FP128 => rustc_codegen_ssa::common::TypeKind::PPC_FP128,
            TypeKind::Label => rustc_codegen_ssa::common::TypeKind::Label,
            TypeKind::Integer => rustc_codegen_ssa::common::TypeKind::Integer,
            TypeKind::Function => rustc_codegen_ssa::common::TypeKind::Function,
            TypeKind::Struct => rustc_codegen_ssa::common::TypeKind::Struct,
            TypeKind::Array => rustc_codegen_ssa::common::TypeKind::Array,
            TypeKind::Pointer => rustc_codegen_ssa::common::TypeKind::Pointer,
            TypeKind::Vector => rustc_codegen_ssa::common::TypeKind::Vector,
            TypeKind::Metadata => rustc_codegen_ssa::common::TypeKind::Metadata,
            TypeKind::X86_MMX => rustc_codegen_ssa::common::TypeKind::X86_MMX,
            TypeKind::Token => rustc_codegen_ssa::common::TypeKind::Token,
        }
    }
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

impl AtomicRmwBinOp {
    pub fn from_generic(op: rustc_codegen_ssa::common::AtomicRmwBinOp) -> Self {
        match op {
            rustc_codegen_ssa::common::AtomicRmwBinOp::AtomicXchg => AtomicRmwBinOp::AtomicXchg,
            rustc_codegen_ssa::common::AtomicRmwBinOp::AtomicAdd => AtomicRmwBinOp::AtomicAdd,
            rustc_codegen_ssa::common::AtomicRmwBinOp::AtomicSub => AtomicRmwBinOp::AtomicSub,
            rustc_codegen_ssa::common::AtomicRmwBinOp::AtomicAnd => AtomicRmwBinOp::AtomicAnd,
            rustc_codegen_ssa::common::AtomicRmwBinOp::AtomicNand => AtomicRmwBinOp::AtomicNand,
            rustc_codegen_ssa::common::AtomicRmwBinOp::AtomicOr => AtomicRmwBinOp::AtomicOr,
            rustc_codegen_ssa::common::AtomicRmwBinOp::AtomicXor => AtomicRmwBinOp::AtomicXor,
            rustc_codegen_ssa::common::AtomicRmwBinOp::AtomicMax => AtomicRmwBinOp::AtomicMax,
            rustc_codegen_ssa::common::AtomicRmwBinOp::AtomicMin => AtomicRmwBinOp::AtomicMin,
            rustc_codegen_ssa::common::AtomicRmwBinOp::AtomicUMax => AtomicRmwBinOp::AtomicUMax,
            rustc_codegen_ssa::common::AtomicRmwBinOp::AtomicUMin => AtomicRmwBinOp::AtomicUMin
        }
    }
}

/// LLVMAtomicOrdering
#[derive(Copy, Clone)]
#[repr(C)]
pub enum AtomicOrdering {
    #[allow(dead_code)]
    NotAtomic = 0,
    Unordered = 1,
    Monotonic = 2,
    // Consume = 3,  // Not specified yet.
    Acquire = 4,
    Release = 5,
    AcquireRelease = 6,
    SequentiallyConsistent = 7,
}

impl AtomicOrdering {
    pub fn from_generic(ao: rustc_codegen_ssa::common::AtomicOrdering) -> Self {
        match ao {
            rustc_codegen_ssa::common::AtomicOrdering::NotAtomic => AtomicOrdering::NotAtomic,
            rustc_codegen_ssa::common::AtomicOrdering::Unordered => AtomicOrdering::Unordered,
            rustc_codegen_ssa::common::AtomicOrdering::Monotonic => AtomicOrdering::Monotonic,
            rustc_codegen_ssa::common::AtomicOrdering::Acquire => AtomicOrdering::Acquire,
            rustc_codegen_ssa::common::AtomicOrdering::Release => AtomicOrdering::Release,
            rustc_codegen_ssa::common::AtomicOrdering::AcquireRelease =>
                AtomicOrdering::AcquireRelease,
            rustc_codegen_ssa::common::AtomicOrdering::SequentiallyConsistent =>
                AtomicOrdering::SequentiallyConsistent
        }
    }
}


/// LLVMRustSynchronizationScope
#[derive(Copy, Clone)]
#[repr(C)]
pub enum SynchronizationScope {
    // FIXME: figure out if this variant is needed at all.
    #[allow(dead_code)]
    Other,
    SingleThread,
    CrossThread,
}

impl SynchronizationScope {
    pub fn from_generic(sc: rustc_codegen_ssa::common::SynchronizationScope) -> Self {
        match sc {
            rustc_codegen_ssa::common::SynchronizationScope::Other => SynchronizationScope::Other,
            rustc_codegen_ssa::common::SynchronizationScope::SingleThread =>
                SynchronizationScope::SingleThread,
            rustc_codegen_ssa::common::SynchronizationScope::CrossThread =>
                SynchronizationScope::CrossThread,
        }
    }
}

/// LLVMRustFileType
#[derive(Copy, Clone)]
#[repr(C)]
pub enum FileType {
    // FIXME: figure out if this variant is needed at all.
    #[allow(dead_code)]
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
    // FIXME: figure out if this variant is needed at all.
    #[allow(dead_code)]
    Other,
    Att,
    Intel,
}

impl AsmDialect {
    pub fn from_generic(asm: syntax::ast::AsmDialect) -> Self {
        match asm {
            syntax::ast::AsmDialect::Att => AsmDialect::Att,
            syntax::ast::AsmDialect::Intel => AsmDialect::Intel
        }
    }
}

/// LLVMRustCodeGenOptLevel
#[derive(Copy, Clone, PartialEq)]
#[repr(C)]
pub enum CodeGenOptLevel {
    // FIXME: figure out if this variant is needed at all.
    #[allow(dead_code)]
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
    // FIXME: figure out if this variant is needed at all.
    #[allow(dead_code)]
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
#[allow(dead_code)] // Variants constructed by C++.
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
    Linker,
}

/// LLVMRustArchiveKind
#[derive(Copy, Clone)]
#[repr(C)]
pub enum ArchiveKind {
    // FIXME: figure out if this variant is needed at all.
    #[allow(dead_code)]
    Other,
    K_GNU,
    K_BSD,
    K_COFF,
}

/// LLVMRustPassKind
#[derive(Copy, Clone, PartialEq, Debug)]
#[repr(C)]
#[allow(dead_code)] // Variants constructed by C++.
pub enum PassKind {
    Other,
    Function,
    Module,
}

/// LLVMRustThinLTOData
extern { pub type ThinLTOData; }

/// LLVMRustThinLTOBuffer
extern { pub type ThinLTOBuffer; }

// LLVMRustModuleNameCallback
pub type ThinLTOModuleNameCallback =
    unsafe extern "C" fn(*mut c_void, *const c_char, *const c_char);

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

extern { type Opaque; }
#[repr(C)]
struct InvariantOpaque<'a> {
    _marker: PhantomData<&'a mut &'a ()>,
    _opaque: Opaque,
}

// Opaque pointer types
extern { pub type Module; }
extern { pub type Context; }
extern { pub type Type; }
extern { pub type Value; }
extern { pub type Metadata; }
extern { pub type BasicBlock; }
#[repr(C)]
pub struct Builder<'a>(InvariantOpaque<'a>);
extern { pub type MemoryBuffer; }
#[repr(C)]
pub struct PassManager<'a>(InvariantOpaque<'a>);
extern { pub type PassManagerBuilder; }
extern { pub type ObjectFile; }
#[repr(C)]
pub struct SectionIterator<'a>(InvariantOpaque<'a>);
extern { pub type Pass; }
extern { pub type TargetMachine; }
extern { pub type Archive; }
#[repr(C)]
pub struct ArchiveIterator<'a>(InvariantOpaque<'a>);
#[repr(C)]
pub struct ArchiveChild<'a>(InvariantOpaque<'a>);
extern { pub type Twine; }
extern { pub type DiagnosticInfo; }
extern { pub type SMDiagnostic; }
#[repr(C)]
pub struct RustArchiveMember<'a>(InvariantOpaque<'a>);
#[repr(C)]
pub struct OperandBundleDef<'a>(InvariantOpaque<'a>);
#[repr(C)]
pub struct Linker<'a>(InvariantOpaque<'a>);

pub type DiagnosticHandler = unsafe extern "C" fn(&DiagnosticInfo, *mut c_void);
pub type InlineAsmDiagHandler = unsafe extern "C" fn(&SMDiagnostic, *const c_void, c_uint);


pub mod debuginfo {
    use super::{InvariantOpaque, Metadata};

    #[repr(C)]
    pub struct DIBuilder<'a>(InvariantOpaque<'a>);

    pub type DIDescriptor = Metadata;
    pub type DIScope = DIDescriptor;
    pub type DIFile = DIScope;
    pub type DILexicalBlock = DIScope;
    pub type DISubprogram = DIScope;
    pub type DINameSpace = DIScope;
    pub type DIType = DIDescriptor;
    pub type DIBasicType = DIType;
    pub type DIDerivedType = DIType;
    pub type DICompositeType = DIDerivedType;
    pub type DIVariable = DIDescriptor;
    pub type DIGlobalVariableExpression = DIDescriptor;
    pub type DIArray = DIDescriptor;
    pub type DISubrange = DIDescriptor;
    pub type DIEnumerator = DIDescriptor;
    pub type DITemplateTypeParameter = DIDescriptor;

    // These values **must** match with LLVMRustDIFlags!!
    bitflags! {
        #[repr(transparent)]
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
        }
    }

    // These values **must** match with LLVMRustDISPFlags!!
    bitflags! {
        #[repr(transparent)]
        #[derive(Default)]
        pub struct DISPFlags: ::libc::uint32_t {
            const SPFlagZero              = 0;
            const SPFlagVirtual           = 1;
            const SPFlagPureVirtual       = 2;
            const SPFlagLocalToUnit       = (1 << 2);
            const SPFlagDefinition        = (1 << 3);
            const SPFlagOptimized         = (1 << 4);
            const SPFlagMainSubprogram    = (1 << 5);
        }
    }

    /// LLVMRustDebugEmissionKind
    #[derive(Copy, Clone)]
    #[repr(C)]
    pub enum DebugEmissionKind {
        NoDebug,
        FullDebug,
        LineTablesOnly,
    }

    impl DebugEmissionKind {
        pub fn from_generic(kind: rustc::session::config::DebugInfo) -> Self {
            use rustc::session::config::DebugInfo;
            match kind {
                DebugInfo::None => DebugEmissionKind::NoDebug,
                DebugInfo::Limited => DebugEmissionKind::LineTablesOnly,
                DebugInfo::Full => DebugEmissionKind::FullDebug,
            }
        }
    }
}

extern { pub type ModuleBuffer; }

extern "C" {
    pub fn LLVMRustInstallFatalErrorHandler();

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
    pub fn LLVMCountParamTypes(FunctionTy: &Type) -> c_uint;
    pub fn LLVMGetParamTypes(FunctionTy: &'a Type, Dest: *mut &'a Type);

    // Operations on struct types
    pub fn LLVMStructTypeInContext(C: &'a Context,
                                   ElementTypes: *const &'a Type,
                                   ElementCount: c_uint,
                                   Packed: Bool)
                                   -> &'a Type;

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
    pub fn LLVMTypeOf(Val: &Value) -> &Type;
    pub fn LLVMGetValueName(Val: &Value) -> *const c_char;
    pub fn LLVMSetValueName(Val: &Value, Name: *const c_char);
    pub fn LLVMReplaceAllUsesWith(OldVal: &'a Value, NewVal: &'a Value);
    pub fn LLVMSetMetadata(Val: &'a Value, KindID: c_uint, Node: &'a Value);

    // Operations on constants of any type
    pub fn LLVMConstNull(Ty: &Type) -> &Value;
    pub fn LLVMGetUndef(Ty: &Type) -> &Value;

    // Operations on metadata
    pub fn LLVMMDStringInContext(C: &Context, Str: *const c_char, SLen: c_uint) -> &Value;
    pub fn LLVMMDNodeInContext(C: &'a Context, Vals: *const &'a Value, Count: c_uint) -> &'a Value;
    pub fn LLVMAddNamedMetadataOperand(M: &'a Module, Name: *const c_char, Val: &'a Value);

    // Operations on scalar constants
    pub fn LLVMConstInt(IntTy: &Type, N: c_ulonglong, SignExtend: Bool) -> &Value;
    pub fn LLVMConstIntOfArbitraryPrecision(IntTy: &Type, Wn: c_uint, Ws: *const u64) -> &Value;
    pub fn LLVMConstIntGetZExtValue(ConstantVal: &Value) -> c_ulonglong;
    pub fn LLVMRustConstInt128Get(ConstantVal: &Value, SExt: bool,
                                  high: &mut u64, low: &mut u64) -> bool;
    pub fn LLVMConstRealGetDouble (ConstantVal: &Value, losesInfo: &mut Bool) -> f64;


    // Operations on composite constants
    pub fn LLVMConstStringInContext(C: &Context,
                                    Str: *const c_char,
                                    Length: c_uint,
                                    DontNullTerminate: Bool)
                                    -> &Value;
    pub fn LLVMConstStructInContext(C: &'a Context,
                                    ConstantVals: *const &'a Value,
                                    Count: c_uint,
                                    Packed: Bool)
                                    -> &'a Value;

    pub fn LLVMConstArray(ElementTy: &'a Type,
                          ConstantVals: *const &'a Value,
                          Length: c_uint)
                          -> &'a Value;
    pub fn LLVMConstVector(ScalarConstantVals: *const &Value, Size: c_uint) -> &Value;

    // Constant expressions
    pub fn LLVMConstInBoundsGEP(
        ConstantVal: &'a Value,
        ConstantIndices: *const &'a Value,
        NumIndices: c_uint,
    ) -> &'a Value;
    pub fn LLVMConstZExt(ConstantVal: &'a Value, ToType: &'a Type) -> &'a Value;
    pub fn LLVMConstPtrToInt(ConstantVal: &'a Value, ToType: &'a Type) -> &'a Value;
    pub fn LLVMConstIntToPtr(ConstantVal: &'a Value, ToType: &'a Type) -> &'a Value;
    pub fn LLVMConstBitCast(ConstantVal: &'a Value, ToType: &'a Type) -> &'a Value;
    pub fn LLVMConstPointerCast(ConstantVal: &'a Value, ToType: &'a Type) -> &'a Value;
    pub fn LLVMConstExtractValue(AggConstant: &Value,
                                 IdxList: *const c_uint,
                                 NumIdx: c_uint)
                                 -> &Value;

    // Operations on global variables, functions, and aliases (globals)
    pub fn LLVMIsDeclaration(Global: &Value) -> Bool;
    pub fn LLVMRustGetLinkage(Global: &Value) -> Linkage;
    pub fn LLVMRustSetLinkage(Global: &Value, RustLinkage: Linkage);
    pub fn LLVMSetSection(Global: &Value, Section: *const c_char);
    pub fn LLVMRustGetVisibility(Global: &Value) -> Visibility;
    pub fn LLVMRustSetVisibility(Global: &Value, Viz: Visibility);
    pub fn LLVMGetAlignment(Global: &Value) -> c_uint;
    pub fn LLVMSetAlignment(Global: &Value, Bytes: c_uint);
    pub fn LLVMSetDLLStorageClass(V: &Value, C: DLLStorageClass);


    // Operations on global variables
    pub fn LLVMIsAGlobalVariable(GlobalVar: &Value) -> Option<&Value>;
    pub fn LLVMAddGlobal(M: &'a Module, Ty: &'a Type, Name: *const c_char) -> &'a Value;
    pub fn LLVMGetNamedGlobal(M: &Module, Name: *const c_char) -> Option<&Value>;
    pub fn LLVMRustGetOrInsertGlobal(M: &'a Module, Name: *const c_char, T: &'a Type) -> &'a Value;
    pub fn LLVMRustInsertPrivateGlobal(M: &'a Module, T: &'a Type) -> &'a Value;
    pub fn LLVMGetFirstGlobal(M: &Module) -> Option<&Value>;
    pub fn LLVMGetNextGlobal(GlobalVar: &Value) -> Option<&Value>;
    pub fn LLVMDeleteGlobal(GlobalVar: &Value);
    pub fn LLVMGetInitializer(GlobalVar: &Value) -> Option<&Value>;
    pub fn LLVMSetInitializer(GlobalVar: &'a Value, ConstantVal: &'a Value);
    pub fn LLVMSetThreadLocal(GlobalVar: &Value, IsThreadLocal: Bool);
    pub fn LLVMSetThreadLocalMode(GlobalVar: &Value, Mode: ThreadLocalMode);
    pub fn LLVMIsGlobalConstant(GlobalVar: &Value) -> Bool;
    pub fn LLVMSetGlobalConstant(GlobalVar: &Value, IsConstant: Bool);
    pub fn LLVMRustGetNamedValue(M: &Module, Name: *const c_char) -> Option<&Value>;
    pub fn LLVMSetTailCall(CallInst: &Value, IsTailCall: Bool);

    // Operations on functions
    pub fn LLVMRustGetOrInsertFunction(M: &'a Module,
                                       Name: *const c_char,
                                       FunctionTy: &'a Type)
                                       -> &'a Value;
    pub fn LLVMSetFunctionCallConv(Fn: &Value, CC: c_uint);
    pub fn LLVMRustAddAlignmentAttr(Fn: &Value, index: c_uint, bytes: u32);
    pub fn LLVMRustAddDereferenceableAttr(Fn: &Value, index: c_uint, bytes: u64);
    pub fn LLVMRustAddDereferenceableOrNullAttr(Fn: &Value, index: c_uint, bytes: u64);
    pub fn LLVMRustAddFunctionAttribute(Fn: &Value, index: c_uint, attr: Attribute);
    pub fn LLVMRustAddFunctionAttrStringValue(Fn: &Value,
                                              index: c_uint,
                                              Name: *const c_char,
                                              Value: *const c_char);
    pub fn LLVMRustRemoveFunctionAttributes(Fn: &Value, index: c_uint, attr: Attribute);

    // Operations on parameters
    pub fn LLVMCountParams(Fn: &Value) -> c_uint;
    pub fn LLVMGetParam(Fn: &Value, Index: c_uint) -> &Value;

    // Operations on basic blocks
    pub fn LLVMGetBasicBlockParent(BB: &BasicBlock) -> &Value;
    pub fn LLVMAppendBasicBlockInContext(C: &'a Context,
                                         Fn: &'a Value,
                                         Name: *const c_char)
                                         -> &'a BasicBlock;
    pub fn LLVMDeleteBasicBlock(BB: &BasicBlock);

    // Operations on instructions
    pub fn LLVMGetFirstBasicBlock(Fn: &Value) -> &BasicBlock;

    // Operations on call sites
    pub fn LLVMSetInstructionCallConv(Instr: &Value, CC: c_uint);
    pub fn LLVMRustAddCallSiteAttribute(Instr: &Value, index: c_uint, attr: Attribute);
    pub fn LLVMRustAddAlignmentCallSiteAttr(Instr: &Value, index: c_uint, bytes: u32);
    pub fn LLVMRustAddDereferenceableCallSiteAttr(Instr: &Value, index: c_uint, bytes: u64);
    pub fn LLVMRustAddDereferenceableOrNullCallSiteAttr(Instr: &Value,
                                                        index: c_uint,
                                                        bytes: u64);

    // Operations on load/store instructions (only)
    pub fn LLVMSetVolatile(MemoryAccessInst: &Value, volatile: Bool);

    // Operations on phi nodes
    pub fn LLVMAddIncoming(PhiNode: &'a Value,
                           IncomingValues: *const &'a Value,
                           IncomingBlocks: *const &'a BasicBlock,
                           Count: c_uint);

    // Instruction builders
    pub fn LLVMCreateBuilderInContext(C: &'a Context) -> &'a mut Builder<'a>;
    pub fn LLVMPositionBuilderAtEnd(Builder: &Builder<'a>, Block: &'a BasicBlock);
    pub fn LLVMGetInsertBlock(Builder: &Builder<'a>) -> &'a BasicBlock;
    pub fn LLVMDisposeBuilder(Builder: &'a mut Builder<'a>);

    // Metadata
    pub fn LLVMSetCurrentDebugLocation(Builder: &Builder<'a>, L: Option<&'a Value>);
    pub fn LLVMGetCurrentDebugLocation(Builder: &Builder<'a>) -> &'a Value;
    pub fn LLVMSetInstDebugLocation(Builder: &Builder<'a>, Inst: &'a Value);

    // Terminators
    pub fn LLVMBuildRetVoid(B: &Builder<'a>) -> &'a Value;
    pub fn LLVMBuildRet(B: &Builder<'a>, V: &'a Value) -> &'a Value;
    pub fn LLVMBuildBr(B: &Builder<'a>, Dest: &'a BasicBlock) -> &'a Value;
    pub fn LLVMBuildCondBr(B: &Builder<'a>,
                           If: &'a Value,
                           Then: &'a BasicBlock,
                           Else: &'a BasicBlock)
                           -> &'a Value;
    pub fn LLVMBuildSwitch(B: &Builder<'a>,
                           V: &'a Value,
                           Else: &'a BasicBlock,
                           NumCases: c_uint)
                           -> &'a Value;
    pub fn LLVMRustBuildInvoke(B: &Builder<'a>,
                               Fn: &'a Value,
                               Args: *const &'a Value,
                               NumArgs: c_uint,
                               Then: &'a BasicBlock,
                               Catch: &'a BasicBlock,
                               Bundle: Option<&OperandBundleDef<'a>>,
                               Name: *const c_char)
                               -> &'a Value;
    pub fn LLVMBuildLandingPad(B: &Builder<'a>,
                               Ty: &'a Type,
                               PersFn: &'a Value,
                               NumClauses: c_uint,
                               Name: *const c_char)
                               -> &'a Value;
    pub fn LLVMBuildResume(B: &Builder<'a>, Exn: &'a Value) -> &'a Value;
    pub fn LLVMBuildUnreachable(B: &Builder<'a>) -> &'a Value;

    pub fn LLVMRustBuildCleanupPad(B: &Builder<'a>,
                                   ParentPad: Option<&'a Value>,
                                   ArgCnt: c_uint,
                                   Args: *const &'a Value,
                                   Name: *const c_char)
                                   -> Option<&'a Value>;
    pub fn LLVMRustBuildCleanupRet(B: &Builder<'a>,
                                   CleanupPad: &'a Value,
                                   UnwindBB: Option<&'a BasicBlock>)
                                   -> Option<&'a Value>;
    pub fn LLVMRustBuildCatchPad(B: &Builder<'a>,
                                 ParentPad: &'a Value,
                                 ArgCnt: c_uint,
                                 Args: *const &'a Value,
                                 Name: *const c_char)
                                 -> Option<&'a Value>;
    pub fn LLVMRustBuildCatchRet(
        B: &Builder<'a>,
        Pad: &'a Value,
        BB: &'a BasicBlock,
    ) -> Option<&'a Value>;
    pub fn LLVMRustBuildCatchSwitch(Builder: &Builder<'a>,
                                    ParentPad: Option<&'a Value>,
                                    BB: Option<&'a BasicBlock>,
                                    NumHandlers: c_uint,
                                    Name: *const c_char)
                                    -> Option<&'a Value>;
    pub fn LLVMRustAddHandler(CatchSwitch: &'a Value, Handler: &'a BasicBlock);
    pub fn LLVMSetPersonalityFn(Func: &'a Value, Pers: &'a Value);

    // Add a case to the switch instruction
    pub fn LLVMAddCase(Switch: &'a Value, OnVal: &'a Value, Dest: &'a BasicBlock);

    // Add a clause to the landing pad instruction
    pub fn LLVMAddClause(LandingPad: &'a Value, ClauseVal: &'a Value);

    // Set the cleanup on a landing pad instruction
    pub fn LLVMSetCleanup(LandingPad: &Value, Val: Bool);

    // Arithmetic
    pub fn LLVMBuildAdd(B: &Builder<'a>,
                        LHS: &'a Value,
                        RHS: &'a Value,
                        Name: *const c_char)
                        -> &'a Value;
    pub fn LLVMBuildFAdd(B: &Builder<'a>,
                         LHS: &'a Value,
                         RHS: &'a Value,
                         Name: *const c_char)
                         -> &'a Value;
    pub fn LLVMBuildSub(B: &Builder<'a>,
                        LHS: &'a Value,
                        RHS: &'a Value,
                        Name: *const c_char)
                        -> &'a Value;
    pub fn LLVMBuildFSub(B: &Builder<'a>,
                         LHS: &'a Value,
                         RHS: &'a Value,
                         Name: *const c_char)
                         -> &'a Value;
    pub fn LLVMBuildMul(B: &Builder<'a>,
                        LHS: &'a Value,
                        RHS: &'a Value,
                        Name: *const c_char)
                        -> &'a Value;
    pub fn LLVMBuildFMul(B: &Builder<'a>,
                         LHS: &'a Value,
                         RHS: &'a Value,
                         Name: *const c_char)
                         -> &'a Value;
    pub fn LLVMBuildUDiv(B: &Builder<'a>,
                         LHS: &'a Value,
                         RHS: &'a Value,
                         Name: *const c_char)
                         -> &'a Value;
    pub fn LLVMBuildExactUDiv(B: &Builder<'a>,
                              LHS: &'a Value,
                              RHS: &'a Value,
                              Name: *const c_char)
                              -> &'a Value;
    pub fn LLVMBuildSDiv(B: &Builder<'a>,
                         LHS: &'a Value,
                         RHS: &'a Value,
                         Name: *const c_char)
                         -> &'a Value;
    pub fn LLVMBuildExactSDiv(B: &Builder<'a>,
                              LHS: &'a Value,
                              RHS: &'a Value,
                              Name: *const c_char)
                              -> &'a Value;
    pub fn LLVMBuildFDiv(B: &Builder<'a>,
                         LHS: &'a Value,
                         RHS: &'a Value,
                         Name: *const c_char)
                         -> &'a Value;
    pub fn LLVMBuildURem(B: &Builder<'a>,
                         LHS: &'a Value,
                         RHS: &'a Value,
                         Name: *const c_char)
                         -> &'a Value;
    pub fn LLVMBuildSRem(B: &Builder<'a>,
                         LHS: &'a Value,
                         RHS: &'a Value,
                         Name: *const c_char)
                         -> &'a Value;
    pub fn LLVMBuildFRem(B: &Builder<'a>,
                         LHS: &'a Value,
                         RHS: &'a Value,
                         Name: *const c_char)
                         -> &'a Value;
    pub fn LLVMBuildShl(B: &Builder<'a>,
                        LHS: &'a Value,
                        RHS: &'a Value,
                        Name: *const c_char)
                        -> &'a Value;
    pub fn LLVMBuildLShr(B: &Builder<'a>,
                         LHS: &'a Value,
                         RHS: &'a Value,
                         Name: *const c_char)
                         -> &'a Value;
    pub fn LLVMBuildAShr(B: &Builder<'a>,
                         LHS: &'a Value,
                         RHS: &'a Value,
                         Name: *const c_char)
                         -> &'a Value;
    pub fn LLVMBuildNSWAdd(B: &Builder<'a>,
                           LHS: &'a Value,
                           RHS: &'a Value,
                           Name: *const c_char)
                           -> &'a Value;
    pub fn LLVMBuildNUWAdd(B: &Builder<'a>,
                           LHS: &'a Value,
                           RHS: &'a Value,
                           Name: *const c_char)
                           -> &'a Value;
    pub fn LLVMBuildNSWSub(B: &Builder<'a>,
                           LHS: &'a Value,
                           RHS: &'a Value,
                           Name: *const c_char)
                           -> &'a Value;
    pub fn LLVMBuildNUWSub(B: &Builder<'a>,
                           LHS: &'a Value,
                           RHS: &'a Value,
                           Name: *const c_char)
                           -> &'a Value;
    pub fn LLVMBuildNSWMul(B: &Builder<'a>,
                           LHS: &'a Value,
                           RHS: &'a Value,
                           Name: *const c_char)
                           -> &'a Value;
    pub fn LLVMBuildNUWMul(B: &Builder<'a>,
                           LHS: &'a Value,
                           RHS: &'a Value,
                           Name: *const c_char)
                           -> &'a Value;
    pub fn LLVMBuildAnd(B: &Builder<'a>,
                        LHS: &'a Value,
                        RHS: &'a Value,
                        Name: *const c_char)
                        -> &'a Value;
    pub fn LLVMBuildOr(B: &Builder<'a>,
                       LHS: &'a Value,
                       RHS: &'a Value,
                       Name: *const c_char)
                       -> &'a Value;
    pub fn LLVMBuildXor(B: &Builder<'a>,
                        LHS: &'a Value,
                        RHS: &'a Value,
                        Name: *const c_char)
                        -> &'a Value;
    pub fn LLVMBuildNeg(B: &Builder<'a>, V: &'a Value, Name: *const c_char) -> &'a Value;
    pub fn LLVMBuildFNeg(B: &Builder<'a>, V: &'a Value, Name: *const c_char) -> &'a Value;
    pub fn LLVMBuildNot(B: &Builder<'a>, V: &'a Value, Name: *const c_char) -> &'a Value;
    pub fn LLVMRustSetHasUnsafeAlgebra(Instr: &Value);

    // Memory
    pub fn LLVMBuildAlloca(B: &Builder<'a>, Ty: &'a Type, Name: *const c_char) -> &'a Value;
    pub fn LLVMBuildArrayAlloca(B: &Builder<'a>,
                                Ty: &'a Type,
                                Val: &'a Value,
                                Name: *const c_char)
                                -> &'a Value;
    pub fn LLVMBuildLoad(B: &Builder<'a>, PointerVal: &'a Value, Name: *const c_char) -> &'a Value;

    pub fn LLVMBuildStore(B: &Builder<'a>, Val: &'a Value, Ptr: &'a Value) -> &'a Value;

    pub fn LLVMBuildGEP(B: &Builder<'a>,
                        Pointer: &'a Value,
                        Indices: *const &'a Value,
                        NumIndices: c_uint,
                        Name: *const c_char)
                        -> &'a Value;
    pub fn LLVMBuildInBoundsGEP(B: &Builder<'a>,
                                Pointer: &'a Value,
                                Indices: *const &'a Value,
                                NumIndices: c_uint,
                                Name: *const c_char)
                                -> &'a Value;
    pub fn LLVMBuildStructGEP(B: &Builder<'a>,
                              Pointer: &'a Value,
                              Idx: c_uint,
                              Name: *const c_char)
                              -> &'a Value;

    // Casts
    pub fn LLVMBuildTrunc(B: &Builder<'a>,
                          Val: &'a Value,
                          DestTy: &'a Type,
                          Name: *const c_char)
                          -> &'a Value;
    pub fn LLVMBuildZExt(B: &Builder<'a>,
                         Val: &'a Value,
                         DestTy: &'a Type,
                         Name: *const c_char)
                         -> &'a Value;
    pub fn LLVMBuildSExt(B: &Builder<'a>,
                         Val: &'a Value,
                         DestTy: &'a Type,
                         Name: *const c_char)
                         -> &'a Value;
    pub fn LLVMBuildFPToUI(B: &Builder<'a>,
                           Val: &'a Value,
                           DestTy: &'a Type,
                           Name: *const c_char)
                           -> &'a Value;
    pub fn LLVMBuildFPToSI(B: &Builder<'a>,
                           Val: &'a Value,
                           DestTy: &'a Type,
                           Name: *const c_char)
                           -> &'a Value;
    pub fn LLVMBuildUIToFP(B: &Builder<'a>,
                           Val: &'a Value,
                           DestTy: &'a Type,
                           Name: *const c_char)
                           -> &'a Value;
    pub fn LLVMBuildSIToFP(B: &Builder<'a>,
                           Val: &'a Value,
                           DestTy: &'a Type,
                           Name: *const c_char)
                           -> &'a Value;
    pub fn LLVMBuildFPTrunc(B: &Builder<'a>,
                            Val: &'a Value,
                            DestTy: &'a Type,
                            Name: *const c_char)
                            -> &'a Value;
    pub fn LLVMBuildFPExt(B: &Builder<'a>,
                          Val: &'a Value,
                          DestTy: &'a Type,
                          Name: *const c_char)
                          -> &'a Value;
    pub fn LLVMBuildPtrToInt(B: &Builder<'a>,
                             Val: &'a Value,
                             DestTy: &'a Type,
                             Name: *const c_char)
                             -> &'a Value;
    pub fn LLVMBuildIntToPtr(B: &Builder<'a>,
                             Val: &'a Value,
                             DestTy: &'a Type,
                             Name: *const c_char)
                             -> &'a Value;
    pub fn LLVMBuildBitCast(B: &Builder<'a>,
                            Val: &'a Value,
                            DestTy: &'a Type,
                            Name: *const c_char)
                            -> &'a Value;
    pub fn LLVMBuildPointerCast(B: &Builder<'a>,
                                Val: &'a Value,
                                DestTy: &'a Type,
                                Name: *const c_char)
                                -> &'a Value;
    pub fn LLVMRustBuildIntCast(B: &Builder<'a>,
                                Val: &'a Value,
                                DestTy: &'a Type,
                                IsSized: bool)
                                -> &'a Value;

    // Comparisons
    pub fn LLVMBuildICmp(B: &Builder<'a>,
                         Op: c_uint,
                         LHS: &'a Value,
                         RHS: &'a Value,
                         Name: *const c_char)
                         -> &'a Value;
    pub fn LLVMBuildFCmp(B: &Builder<'a>,
                         Op: c_uint,
                         LHS: &'a Value,
                         RHS: &'a Value,
                         Name: *const c_char)
                         -> &'a Value;

    // Miscellaneous instructions
    pub fn LLVMBuildPhi(B: &Builder<'a>, Ty: &'a Type, Name: *const c_char) -> &'a Value;
    pub fn LLVMRustBuildCall(B: &Builder<'a>,
                             Fn: &'a Value,
                             Args: *const &'a Value,
                             NumArgs: c_uint,
                             Bundle: Option<&OperandBundleDef<'a>>,
                             Name: *const c_char)
                             -> &'a Value;
    pub fn LLVMRustBuildMemCpy(B: &Builder<'a>,
                               Dst: &'a Value,
                               DstAlign: c_uint,
                               Src: &'a Value,
                               SrcAlign: c_uint,
                               Size: &'a Value,
                               IsVolatile: bool)
                               -> &'a Value;
    pub fn LLVMRustBuildMemMove(B: &Builder<'a>,
                                Dst: &'a Value,
                                DstAlign: c_uint,
                                Src: &'a Value,
                                SrcAlign: c_uint,
                                Size: &'a Value,
                                IsVolatile: bool)
                                -> &'a Value;
    pub fn LLVMBuildSelect(B: &Builder<'a>,
                           If: &'a Value,
                           Then: &'a Value,
                           Else: &'a Value,
                           Name: *const c_char)
                           -> &'a Value;
    pub fn LLVMBuildVAArg(B: &Builder<'a>,
                          list: &'a Value,
                          Ty: &'a Type,
                          Name: *const c_char)
                          -> &'a Value;
    pub fn LLVMBuildExtractElement(B: &Builder<'a>,
                                   VecVal: &'a Value,
                                   Index: &'a Value,
                                   Name: *const c_char)
                                   -> &'a Value;
    pub fn LLVMBuildInsertElement(B: &Builder<'a>,
                                  VecVal: &'a Value,
                                  EltVal: &'a Value,
                                  Index: &'a Value,
                                  Name: *const c_char)
                                  -> &'a Value;
    pub fn LLVMBuildShuffleVector(B: &Builder<'a>,
                                  V1: &'a Value,
                                  V2: &'a Value,
                                  Mask: &'a Value,
                                  Name: *const c_char)
                                  -> &'a Value;
    pub fn LLVMBuildExtractValue(B: &Builder<'a>,
                                 AggVal: &'a Value,
                                 Index: c_uint,
                                 Name: *const c_char)
                                 -> &'a Value;
    pub fn LLVMBuildInsertValue(B: &Builder<'a>,
                                AggVal: &'a Value,
                                EltVal: &'a Value,
                                Index: c_uint,
                                Name: *const c_char)
                                -> &'a Value;

    pub fn LLVMRustBuildVectorReduceFAdd(B: &Builder<'a>,
                                         Acc: &'a Value,
                                         Src: &'a Value)
                                         -> &'a Value;
    pub fn LLVMRustBuildVectorReduceFMul(B: &Builder<'a>,
                                         Acc: &'a Value,
                                         Src: &'a Value)
                                         -> &'a Value;
    pub fn LLVMRustBuildVectorReduceAdd(B: &Builder<'a>,
                                        Src: &'a Value)
                                        -> &'a Value;
    pub fn LLVMRustBuildVectorReduceMul(B: &Builder<'a>,
                                        Src: &'a Value)
                                        -> &'a Value;
    pub fn LLVMRustBuildVectorReduceAnd(B: &Builder<'a>,
                                        Src: &'a Value)
                                        -> &'a Value;
    pub fn LLVMRustBuildVectorReduceOr(B: &Builder<'a>,
                                       Src: &'a Value)
                                       -> &'a Value;
    pub fn LLVMRustBuildVectorReduceXor(B: &Builder<'a>,
                                        Src: &'a Value)
                                        -> &'a Value;
    pub fn LLVMRustBuildVectorReduceMin(B: &Builder<'a>,
                                        Src: &'a Value,
                                        IsSigned: bool)
                                        -> &'a Value;
    pub fn LLVMRustBuildVectorReduceMax(B: &Builder<'a>,
                                        Src: &'a Value,
                                        IsSigned: bool)
                                        -> &'a Value;
    pub fn LLVMRustBuildVectorReduceFMin(B: &Builder<'a>,
                                         Src: &'a Value,
                                         IsNaN: bool)
                                         -> &'a Value;
    pub fn LLVMRustBuildVectorReduceFMax(B: &Builder<'a>,
                                         Src: &'a Value,
                                         IsNaN: bool)
                                         -> &'a Value;

    pub fn LLVMRustBuildMinNum(
        B: &Builder<'a>,
        LHS: &'a Value,
        LHS: &'a Value,
    ) -> &'a Value;
    pub fn LLVMRustBuildMaxNum(
        B: &Builder<'a>,
        LHS: &'a Value,
        LHS: &'a Value,
    ) -> &'a Value;

    // Atomic Operations
    pub fn LLVMRustBuildAtomicLoad(B: &Builder<'a>,
                                   PointerVal: &'a Value,
                                   Name: *const c_char,
                                   Order: AtomicOrdering)
                                   -> &'a Value;

    pub fn LLVMRustBuildAtomicStore(B: &Builder<'a>,
                                    Val: &'a Value,
                                    Ptr: &'a Value,
                                    Order: AtomicOrdering)
                                    -> &'a Value;

    pub fn LLVMRustBuildAtomicCmpXchg(B: &Builder<'a>,
                                      LHS: &'a Value,
                                      CMP: &'a Value,
                                      RHS: &'a Value,
                                      Order: AtomicOrdering,
                                      FailureOrder: AtomicOrdering,
                                      Weak: Bool)
                                      -> &'a Value;

    pub fn LLVMBuildAtomicRMW(B: &Builder<'a>,
                              Op: AtomicRmwBinOp,
                              LHS: &'a Value,
                              RHS: &'a Value,
                              Order: AtomicOrdering,
                              SingleThreaded: Bool)
                              -> &'a Value;

    pub fn LLVMRustBuildAtomicFence(B: &Builder<'_>,
                                    Order: AtomicOrdering,
                                    Scope: SynchronizationScope);

    /// Writes a module to the specified path. Returns 0 on success.
    pub fn LLVMWriteBitcodeToFile(M: &Module, Path: *const c_char) -> c_int;

    /// Creates a pass manager.
    pub fn LLVMCreatePassManager() -> &'a mut PassManager<'a>;

    /// Creates a function-by-function pass manager
    pub fn LLVMCreateFunctionPassManagerForModule(M: &'a Module) -> &'a mut PassManager<'a>;

    /// Disposes a pass manager.
    pub fn LLVMDisposePassManager(PM: &'a mut PassManager<'a>);

    /// Runs a pass manager on a module.
    pub fn LLVMRunPassManager(PM: &PassManager<'a>, M: &'a Module) -> Bool;

    pub fn LLVMInitializePasses();

    pub fn LLVMPassManagerBuilderCreate() -> &'static mut PassManagerBuilder;
    pub fn LLVMPassManagerBuilderDispose(PMB: &'static mut PassManagerBuilder);
    pub fn LLVMPassManagerBuilderSetSizeLevel(PMB: &PassManagerBuilder, Value: Bool);
    pub fn LLVMPassManagerBuilderSetDisableUnrollLoops(PMB: &PassManagerBuilder, Value: Bool);
    pub fn LLVMPassManagerBuilderUseInlinerWithThreshold(PMB: &PassManagerBuilder,
                                                         threshold: c_uint);
    pub fn LLVMPassManagerBuilderPopulateModulePassManager(PMB: &PassManagerBuilder,
                                                           PM: &PassManager<'_>);

    pub fn LLVMPassManagerBuilderPopulateFunctionPassManager(PMB: &PassManagerBuilder,
                                                             PM: &PassManager<'_>);
    pub fn LLVMPassManagerBuilderPopulateLTOPassManager(PMB: &PassManagerBuilder,
                                                        PM: &PassManager<'_>,
                                                        Internalize: Bool,
                                                        RunInliner: Bool);
    pub fn LLVMRustPassManagerBuilderPopulateThinLTOPassManager(
        PMB: &PassManagerBuilder,
        PM: &PassManager<'_>);

    // Stuff that's in rustllvm/ because it's not upstream yet.

    /// Opens an object file.
    pub fn LLVMCreateObjectFile(
        MemBuf: &'static mut MemoryBuffer,
    ) -> Option<&'static mut ObjectFile>;
    /// Closes an object file.
    pub fn LLVMDisposeObjectFile(ObjFile: &'static mut ObjectFile);

    /// Enumerates the sections in an object file.
    pub fn LLVMGetSections(ObjFile: &'a ObjectFile) -> &'a mut SectionIterator<'a>;
    /// Destroys a section iterator.
    pub fn LLVMDisposeSectionIterator(SI: &'a mut SectionIterator<'a>);
    /// Returns `true` if the section iterator is at the end of the section
    /// list:
    pub fn LLVMIsSectionIteratorAtEnd(ObjFile: &'a ObjectFile, SI: &SectionIterator<'a>) -> Bool;
    /// Moves the section iterator to point to the next section.
    pub fn LLVMMoveToNextSection(SI: &SectionIterator<'_>);
    /// Returns the current section size.
    pub fn LLVMGetSectionSize(SI: &SectionIterator<'_>) -> c_ulonglong;
    /// Returns the current section contents as a string buffer.
    pub fn LLVMGetSectionContents(SI: &SectionIterator<'_>) -> *const c_char;

    /// Reads the given file and returns it as a memory buffer. Use
    /// LLVMDisposeMemoryBuffer() to get rid of it.
    pub fn LLVMRustCreateMemoryBufferWithContentsOfFile(
        Path: *const c_char,
    ) -> Option<&'static mut MemoryBuffer>;

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
                             -> &Value;
    pub fn LLVMRustInlineAsmVerify(Ty: &Type,
                                   Constraints: *const c_char)
                                   -> bool;

    pub fn LLVMRustDebugMetadataVersion() -> u32;
    pub fn LLVMRustVersionMajor() -> u32;
    pub fn LLVMRustVersionMinor() -> u32;

    pub fn LLVMRustAddModuleFlag(M: &Module, name: *const c_char, value: u32);

    pub fn LLVMRustMetadataAsValue(C: &'a Context, MD: &'a Metadata) -> &'a Value;

    pub fn LLVMRustDIBuilderCreate(M: &'a Module) -> &'a mut DIBuilder<'a>;

    pub fn LLVMRustDIBuilderDispose(Builder: &'a mut DIBuilder<'a>);

    pub fn LLVMRustDIBuilderFinalize(Builder: &DIBuilder<'_>);

    pub fn LLVMRustDIBuilderCreateCompileUnit(Builder: &DIBuilder<'a>,
                                              Lang: c_uint,
                                              File: &'a DIFile,
                                              Producer: *const c_char,
                                              isOptimized: bool,
                                              Flags: *const c_char,
                                              RuntimeVer: c_uint,
                                              SplitName: *const c_char,
                                              kind: DebugEmissionKind)
                                              -> &'a DIDescriptor;

    pub fn LLVMRustDIBuilderCreateFile(Builder: &DIBuilder<'a>,
                                       Filename: *const c_char,
                                       Directory: *const c_char)
                                       -> &'a DIFile;

    pub fn LLVMRustDIBuilderCreateSubroutineType(Builder: &DIBuilder<'a>,
                                                 File: &'a DIFile,
                                                 ParameterTypes: &'a DIArray)
                                                 -> &'a DICompositeType;

    pub fn LLVMRustDIBuilderCreateFunction(Builder: &DIBuilder<'a>,
                                           Scope: &'a DIDescriptor,
                                           Name: *const c_char,
                                           LinkageName: *const c_char,
                                           File: &'a DIFile,
                                           LineNo: c_uint,
                                           Ty: &'a DIType,
                                           ScopeLine: c_uint,
                                           Flags: DIFlags,
                                           SPFlags: DISPFlags,
                                           Fn: &'a Value,
                                           TParam: &'a DIArray,
                                           Decl: Option<&'a DIDescriptor>)
                                           -> &'a DISubprogram;

    pub fn LLVMRustDIBuilderCreateBasicType(Builder: &DIBuilder<'a>,
                                            Name: *const c_char,
                                            SizeInBits: u64,
                                            AlignInBits: u32,
                                            Encoding: c_uint)
                                            -> &'a DIBasicType;

    pub fn LLVMRustDIBuilderCreatePointerType(Builder: &DIBuilder<'a>,
                                              PointeeTy: &'a DIType,
                                              SizeInBits: u64,
                                              AlignInBits: u32,
                                              Name: *const c_char)
                                              -> &'a DIDerivedType;

    pub fn LLVMRustDIBuilderCreateStructType(Builder: &DIBuilder<'a>,
                                             Scope: Option<&'a DIDescriptor>,
                                             Name: *const c_char,
                                             File: &'a DIFile,
                                             LineNumber: c_uint,
                                             SizeInBits: u64,
                                             AlignInBits: u32,
                                             Flags: DIFlags,
                                             DerivedFrom: Option<&'a DIType>,
                                             Elements: &'a DIArray,
                                             RunTimeLang: c_uint,
                                             VTableHolder: Option<&'a DIType>,
                                             UniqueId: *const c_char)
                                             -> &'a DICompositeType;

    pub fn LLVMRustDIBuilderCreateMemberType(Builder: &DIBuilder<'a>,
                                             Scope: &'a DIDescriptor,
                                             Name: *const c_char,
                                             File: &'a DIFile,
                                             LineNo: c_uint,
                                             SizeInBits: u64,
                                             AlignInBits: u32,
                                             OffsetInBits: u64,
                                             Flags: DIFlags,
                                             Ty: &'a DIType)
                                             -> &'a DIDerivedType;

    pub fn LLVMRustDIBuilderCreateVariantMemberType(Builder: &DIBuilder<'a>,
                                                    Scope: &'a DIScope,
                                                    Name: *const c_char,
                                                    File: &'a DIFile,
                                                    LineNumber: c_uint,
                                                    SizeInBits: u64,
                                                    AlignInBits: u32,
                                                    OffsetInBits: u64,
                                                    Discriminant: Option<&'a Value>,
                                                    Flags: DIFlags,
                                                    Ty: &'a DIType)
                                                    -> &'a DIType;

    pub fn LLVMRustDIBuilderCreateLexicalBlock(Builder: &DIBuilder<'a>,
                                               Scope: &'a DIScope,
                                               File: &'a DIFile,
                                               Line: c_uint,
                                               Col: c_uint)
                                               -> &'a DILexicalBlock;

    pub fn LLVMRustDIBuilderCreateLexicalBlockFile(Builder: &DIBuilder<'a>,
                                                   Scope: &'a DIScope,
                                                   File: &'a DIFile)
                                                   -> &'a DILexicalBlock;

    pub fn LLVMRustDIBuilderCreateStaticVariable(Builder: &DIBuilder<'a>,
                                                 Context: Option<&'a DIScope>,
                                                 Name: *const c_char,
                                                 LinkageName: *const c_char,
                                                 File: &'a DIFile,
                                                 LineNo: c_uint,
                                                 Ty: &'a DIType,
                                                 isLocalToUnit: bool,
                                                 Val: &'a Value,
                                                 Decl: Option<&'a DIDescriptor>,
                                                 AlignInBits: u32)
                                                 -> &'a DIGlobalVariableExpression;

    pub fn LLVMRustDIBuilderCreateVariable(Builder: &DIBuilder<'a>,
                                           Tag: c_uint,
                                           Scope: &'a DIDescriptor,
                                           Name: *const c_char,
                                           File: &'a DIFile,
                                           LineNo: c_uint,
                                           Ty: &'a DIType,
                                           AlwaysPreserve: bool,
                                           Flags: DIFlags,
                                           ArgNo: c_uint,
                                           AlignInBits: u32)
                                           -> &'a DIVariable;

    pub fn LLVMRustDIBuilderCreateArrayType(Builder: &DIBuilder<'a>,
                                            Size: u64,
                                            AlignInBits: u32,
                                            Ty: &'a DIType,
                                            Subscripts: &'a DIArray)
                                            -> &'a DIType;

    pub fn LLVMRustDIBuilderGetOrCreateSubrange(Builder: &DIBuilder<'a>,
                                                Lo: i64,
                                                Count: i64)
                                                -> &'a DISubrange;

    pub fn LLVMRustDIBuilderGetOrCreateArray(Builder: &DIBuilder<'a>,
                                             Ptr: *const Option<&'a DIDescriptor>,
                                             Count: c_uint)
                                             -> &'a DIArray;

    pub fn LLVMRustDIBuilderInsertDeclareAtEnd(Builder: &DIBuilder<'a>,
                                               Val: &'a Value,
                                               VarInfo: &'a DIVariable,
                                               AddrOps: *const i64,
                                               AddrOpsCount: c_uint,
                                               DL: &'a Value,
                                               InsertAtEnd: &'a BasicBlock)
                                               -> &'a Value;

    pub fn LLVMRustDIBuilderCreateEnumerator(Builder: &DIBuilder<'a>,
                                             Name: *const c_char,
                                             Val: u64)
                                             -> &'a DIEnumerator;

    pub fn LLVMRustDIBuilderCreateEnumerationType(Builder: &DIBuilder<'a>,
                                                  Scope: &'a DIScope,
                                                  Name: *const c_char,
                                                  File: &'a DIFile,
                                                  LineNumber: c_uint,
                                                  SizeInBits: u64,
                                                  AlignInBits: u32,
                                                  Elements: &'a DIArray,
                                                  ClassType: &'a DIType,
                                                  IsScoped: bool)
                                                  -> &'a DIType;

    pub fn LLVMRustDIBuilderCreateUnionType(Builder: &DIBuilder<'a>,
                                            Scope: &'a DIScope,
                                            Name: *const c_char,
                                            File: &'a DIFile,
                                            LineNumber: c_uint,
                                            SizeInBits: u64,
                                            AlignInBits: u32,
                                            Flags: DIFlags,
                                            Elements: Option<&'a DIArray>,
                                            RunTimeLang: c_uint,
                                            UniqueId: *const c_char)
                                            -> &'a DIType;

    pub fn LLVMRustDIBuilderCreateVariantPart(Builder: &DIBuilder<'a>,
                                              Scope: &'a DIScope,
                                              Name: *const c_char,
                                              File: &'a DIFile,
                                              LineNo: c_uint,
                                              SizeInBits: u64,
                                              AlignInBits: u32,
                                              Flags: DIFlags,
                                              Discriminator: Option<&'a DIDerivedType>,
                                              Elements: &'a DIArray,
                                              UniqueId: *const c_char)
                                              -> &'a DIDerivedType;

    pub fn LLVMSetUnnamedAddr(GlobalVar: &Value, UnnamedAddr: Bool);

    pub fn LLVMRustDIBuilderCreateTemplateTypeParameter(Builder: &DIBuilder<'a>,
                                                        Scope: Option<&'a DIScope>,
                                                        Name: *const c_char,
                                                        Ty: &'a DIType,
                                                        File: &'a DIFile,
                                                        LineNo: c_uint,
                                                        ColumnNo: c_uint)
                                                        -> &'a DITemplateTypeParameter;


    pub fn LLVMRustDIBuilderCreateNameSpace(Builder: &DIBuilder<'a>,
                                            Scope: Option<&'a DIScope>,
                                            Name: *const c_char,
                                            File: &'a DIFile,
                                            LineNo: c_uint)
                                            -> &'a DINameSpace;

    pub fn LLVMRustDICompositeTypeReplaceArrays(Builder: &DIBuilder<'a>,
                                                CompositeType: &'a DIType,
                                                Elements: Option<&'a DIArray>,
                                                Params: Option<&'a DIArray>);


    pub fn LLVMRustDIBuilderCreateDebugLocation(Context: &'a Context,
                                                Line: c_uint,
                                                Column: c_uint,
                                                Scope: &'a DIScope,
                                                InlinedAt: Option<&'a Metadata>)
                                                -> &'a Value;
    pub fn LLVMRustDIBuilderCreateOpDeref() -> i64;
    pub fn LLVMRustDIBuilderCreateOpPlusUconst() -> i64;

    #[allow(improper_ctypes)]
    pub fn LLVMRustWriteTypeToString(Type: &Type, s: &RustString);
    #[allow(improper_ctypes)]
    pub fn LLVMRustWriteValueToString(value_ref: &Value, s: &RustString);

    pub fn LLVMIsAConstantInt(value_ref: &Value) -> Option<&Value>;
    pub fn LLVMIsAConstantFP(value_ref: &Value) -> Option<&Value>;

    pub fn LLVMRustPassKind(Pass: &Pass) -> PassKind;
    pub fn LLVMRustFindAndCreatePass(Pass: *const c_char) -> Option<&'static mut Pass>;
    pub fn LLVMRustAddPass(PM: &PassManager<'_>, Pass: &'static mut Pass);

    pub fn LLVMRustHasFeature(T: &TargetMachine, s: *const c_char) -> bool;

    pub fn LLVMRustPrintTargetCPUs(T: &TargetMachine);
    pub fn LLVMRustPrintTargetFeatures(T: &TargetMachine);

    pub fn LLVMRustGetHostCPUName(len: *mut usize) -> *const c_char;
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
                                       Singlethread: bool,
                                       AsmComments: bool,
                                       EmitStackSizeSection: bool)
                                       -> Option<&'static mut TargetMachine>;
    pub fn LLVMRustDisposeTargetMachine(T: &'static mut TargetMachine);
    pub fn LLVMRustAddAnalysisPasses(T: &'a TargetMachine, PM: &PassManager<'a>, M: &'a Module);
    pub fn LLVMRustAddBuilderLibraryInfo(PMB: &'a PassManagerBuilder,
                                         M: &'a Module,
                                         DisableSimplifyLibCalls: bool);
    pub fn LLVMRustConfigurePassManagerBuilder(PMB: &PassManagerBuilder,
                                               OptLevel: CodeGenOptLevel,
                                               MergeFunctions: bool,
                                               SLPVectorize: bool,
                                               LoopVectorize: bool,
                                               PrepareForThinLTO: bool,
                                               PGOGenPath: *const c_char,
                                               PGOUsePath: *const c_char);
    pub fn LLVMRustAddLibraryInfo(PM: &PassManager<'a>,
                                  M: &'a Module,
                                  DisableSimplifyLibCalls: bool);
    pub fn LLVMRustRunFunctionPassManager(PM: &PassManager<'a>, M: &'a Module);
    pub fn LLVMRustWriteOutputFile(T: &'a TargetMachine,
                                   PM: &PassManager<'a>,
                                   M: &'a Module,
                                   Output: *const c_char,
                                   FileType: FileType)
                                   -> LLVMRustResult;
    pub fn LLVMRustPrintModule(PM: &PassManager<'a>,
                               M: &'a Module,
                               Output: *const c_char,
                               Demangle: extern fn(*const c_char,
                                                   size_t,
                                                   *mut c_char,
                                                   size_t) -> size_t,
                               ) -> LLVMRustResult;
    pub fn LLVMRustSetLLVMOptions(Argc: c_int, Argv: *const *const c_char);
    pub fn LLVMRustPrintPasses();
    pub fn LLVMRustSetNormalizedTarget(M: &Module, triple: *const c_char);
    pub fn LLVMRustAddAlwaysInlinePass(P: &PassManagerBuilder, AddLifetimes: bool);
    pub fn LLVMRustRunRestrictionPass(M: &Module, syms: *const *const c_char, len: size_t);
    pub fn LLVMRustMarkAllFunctionsNounwind(M: &Module);

    pub fn LLVMRustOpenArchive(path: *const c_char) -> Option<&'static mut Archive>;
    pub fn LLVMRustArchiveIteratorNew(AR: &'a Archive) -> &'a mut ArchiveIterator<'a>;
    pub fn LLVMRustArchiveIteratorNext(
        AIR: &ArchiveIterator<'a>,
    ) -> Option<&'a mut ArchiveChild<'a>>;
    pub fn LLVMRustArchiveChildName(ACR: &ArchiveChild<'_>, size: &mut size_t) -> *const c_char;
    pub fn LLVMRustArchiveChildData(ACR: &ArchiveChild<'_>, size: &mut size_t) -> *const c_char;
    pub fn LLVMRustArchiveChildFree(ACR: &'a mut ArchiveChild<'a>);
    pub fn LLVMRustArchiveIteratorFree(AIR: &'a mut ArchiveIterator<'a>);
    pub fn LLVMRustDestroyArchive(AR: &'static mut Archive);

    pub fn LLVMRustGetSectionName(SI: &SectionIterator<'_>, data: &mut *const c_char) -> size_t;

    #[allow(improper_ctypes)]
    pub fn LLVMRustWriteTwineToString(T: &Twine, s: &RustString);

    pub fn LLVMContextSetDiagnosticHandler(C: &Context,
                                           Handler: DiagnosticHandler,
                                           DiagnosticContext: *mut c_void);

    #[allow(improper_ctypes)]
    pub fn LLVMRustUnpackOptimizationDiagnostic(DI: &'a DiagnosticInfo,
                                                pass_name_out: &RustString,
                                                function_out: &mut Option<&'a Value>,
                                                loc_line_out: &mut c_uint,
                                                loc_column_out: &mut c_uint,
                                                loc_filename_out: &RustString,
                                                message_out: &RustString);

    pub fn LLVMRustUnpackInlineAsmDiagnostic(DI: &'a DiagnosticInfo,
                                             cookie_out: &mut c_uint,
                                             message_out: &mut Option<&'a Twine>,
                                             instruction_out: &mut Option<&'a Value>);

    #[allow(improper_ctypes)]
    pub fn LLVMRustWriteDiagnosticInfoToString(DI: &DiagnosticInfo, s: &RustString);
    pub fn LLVMRustGetDiagInfoKind(DI: &DiagnosticInfo) -> DiagnosticKind;

    pub fn LLVMRustSetInlineAsmDiagnosticHandler(C: &Context,
                                                 H: InlineAsmDiagHandler,
                                                 CX: *mut c_void);

    #[allow(improper_ctypes)]
    pub fn LLVMRustWriteSMDiagnosticToString(d: &SMDiagnostic, s: &RustString);

    pub fn LLVMRustWriteArchive(Dst: *const c_char,
                                NumMembers: size_t,
                                Members: *const &RustArchiveMember<'_>,
                                WriteSymbtab: bool,
                                Kind: ArchiveKind)
                                -> LLVMRustResult;
    pub fn LLVMRustArchiveMemberNew(Filename: *const c_char,
                                    Name: *const c_char,
                                    Child: Option<&ArchiveChild<'a>>)
                                    -> &'a mut RustArchiveMember<'a>;
    pub fn LLVMRustArchiveMemberFree(Member: &'a mut RustArchiveMember<'a>);

    pub fn LLVMRustSetDataLayoutFromTargetMachine(M: &'a Module, TM: &'a TargetMachine);

    pub fn LLVMRustBuildOperandBundleDef(Name: *const c_char,
                                         Inputs: *const &'a Value,
                                         NumInputs: c_uint)
                                         -> &'a mut OperandBundleDef<'a>;
    pub fn LLVMRustFreeOperandBundleDef(Bundle: &'a mut OperandBundleDef<'a>);

    pub fn LLVMRustPositionBuilderAtStart(B: &Builder<'a>, BB: &'a BasicBlock);

    pub fn LLVMRustSetComdat(M: &'a Module, V: &'a Value, Name: *const c_char);
    pub fn LLVMRustUnsetComdat(V: &Value);
    pub fn LLVMRustSetModulePIELevel(M: &Module);
    pub fn LLVMRustModuleBufferCreate(M: &Module) -> &'static mut ModuleBuffer;
    pub fn LLVMRustModuleBufferPtr(p: &ModuleBuffer) -> *const u8;
    pub fn LLVMRustModuleBufferLen(p: &ModuleBuffer) -> usize;
    pub fn LLVMRustModuleBufferFree(p: &'static mut ModuleBuffer);
    pub fn LLVMRustModuleCost(M: &Module) -> u64;

    pub fn LLVMRustThinLTOBufferCreate(M: &Module) -> &'static mut ThinLTOBuffer;
    pub fn LLVMRustThinLTOBufferFree(M: &'static mut ThinLTOBuffer);
    pub fn LLVMRustThinLTOBufferPtr(M: &ThinLTOBuffer) -> *const c_char;
    pub fn LLVMRustThinLTOBufferLen(M: &ThinLTOBuffer) -> size_t;
    pub fn LLVMRustCreateThinLTOData(
        Modules: *const ThinLTOModule,
        NumModules: c_uint,
        PreservedSymbols: *const *const c_char,
        PreservedSymbolsLen: c_uint,
    ) -> Option<&'static mut ThinLTOData>;
    pub fn LLVMRustPrepareThinLTORename(
        Data: &ThinLTOData,
        Module: &Module,
    ) -> bool;
    pub fn LLVMRustPrepareThinLTOResolveWeak(
        Data: &ThinLTOData,
        Module: &Module,
    ) -> bool;
    pub fn LLVMRustPrepareThinLTOInternalize(
        Data: &ThinLTOData,
        Module: &Module,
    ) -> bool;
    pub fn LLVMRustPrepareThinLTOImport(
        Data: &ThinLTOData,
        Module: &Module,
    ) -> bool;
    pub fn LLVMRustGetThinLTOModuleImports(
        Data: *const ThinLTOData,
        ModuleNameCallback: ThinLTOModuleNameCallback,
        CallbackPayload: *mut c_void,
    );
    pub fn LLVMRustFreeThinLTOData(Data: &'static mut ThinLTOData);
    pub fn LLVMRustParseBitcodeForLTO(
        Context: &Context,
        Data: *const u8,
        len: usize,
        Identifier: *const c_char,
    ) -> Option<&Module>;
    pub fn LLVMRustThinLTOGetDICompileUnit(M: &Module,
                                           CU1: &mut *mut c_void,
                                           CU2: &mut *mut c_void);
    pub fn LLVMRustThinLTOPatchDICompileUnit(M: &Module, CU: *mut c_void);

    pub fn LLVMRustLinkerNew(M: &'a Module) -> &'a mut Linker<'a>;
    pub fn LLVMRustLinkerAdd(linker: &Linker<'_>,
                             bytecode: *const c_char,
                             bytecode_len: usize) -> bool;
    pub fn LLVMRustLinkerFree(linker: &'a mut Linker<'a>);
}
