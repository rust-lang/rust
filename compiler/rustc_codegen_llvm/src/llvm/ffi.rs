#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]

use std::fmt::Debug;
use std::marker::PhantomData;
use std::ptr;

use libc::{c_char, c_int, c_uint, c_ulonglong, c_void, size_t};
use rustc_macros::TryFromU32;
use rustc_target::spec::SymbolVisibility;

use super::RustString;
use super::debuginfo::{
    DIArray, DIBasicType, DIBuilder, DICompositeType, DIDerivedType, DIDescriptor, DIEnumerator,
    DIFile, DIFlags, DIGlobalVariableExpression, DILexicalBlock, DILocation, DINameSpace,
    DISPFlags, DIScope, DISubprogram, DISubrange, DITemplateTypeParameter, DIType, DIVariable,
    DebugEmissionKind, DebugNameTableKind,
};

pub type Bool = c_uint;

pub const True: Bool = 1 as Bool;
pub const False: Bool = 0 as Bool;

/// Wrapper for a raw enum value returned from LLVM's C APIs.
///
/// For C enums returned by LLVM, it's risky to use a Rust enum as the return
/// type, because it would be UB if a later version of LLVM adds a new enum
/// value and returns it. Instead, return this raw wrapper, then convert to the
/// Rust-side enum explicitly.
#[repr(transparent)]
pub struct RawEnum<T> {
    value: u32,
    /// We don't own or consume a `T`, but we can produce one.
    _rust_side_type: PhantomData<fn() -> T>,
}

impl<T: TryFrom<u32>> RawEnum<T> {
    #[track_caller]
    pub(crate) fn to_rust(self) -> T
    where
        T::Error: Debug,
    {
        // If this fails, the Rust-side enum is out of sync with LLVM's enum.
        T::try_from(self.value).expect("enum value returned by LLVM should be known")
    }
}

#[derive(Copy, Clone, PartialEq)]
#[repr(C)]
#[allow(dead_code)] // Variants constructed by C++.
pub enum LLVMRustResult {
    Success,
    Failure,
}

// Rust version of the C struct with the same name in rustc_llvm/llvm-wrapper/RustWrapper.cpp.
#[repr(C)]
pub struct LLVMRustCOFFShortExport {
    pub name: *const c_char,
    pub ordinal_present: bool,
    /// value of `ordinal` only important when `ordinal_present` is true
    pub ordinal: u16,
}

impl LLVMRustCOFFShortExport {
    pub fn new(name: *const c_char, ordinal: Option<u16>) -> LLVMRustCOFFShortExport {
        LLVMRustCOFFShortExport {
            name,
            ordinal_present: ordinal.is_some(),
            ordinal: ordinal.unwrap_or(0),
        }
    }
}

/// Translation of LLVM's MachineTypes enum, defined in llvm\include\llvm\BinaryFormat\COFF.h.
///
/// We include only architectures supported on Windows.
#[derive(Copy, Clone, PartialEq)]
#[repr(C)]
pub enum LLVMMachineType {
    AMD64 = 0x8664,
    I386 = 0x14c,
    ARM64 = 0xaa64,
    ARM64EC = 0xa641,
    ARM = 0x01c0,
}

/// Must match the layout of `LLVMRustModuleFlagMergeBehavior`.
///
/// When merging modules (e.g. during LTO), their metadata flags are combined. Conflicts are
/// resolved according to the merge behaviors specified here. Flags differing only in merge
/// behavior are still considered to be in conflict.
///
/// In order for Rust-C LTO to work, we must specify behaviors compatible with Clang. Notably,
/// 'Error' and 'Warning' cannot be mixed for a given flag.
///
/// There is a stable LLVM-C version of this enum (`LLVMModuleFlagBehavior`),
/// but as of LLVM 19 it does not support all of the enum values in the unstable
/// C++ API.
#[derive(Copy, Clone, PartialEq)]
#[repr(C)]
pub enum ModuleFlagMergeBehavior {
    Error = 1,
    Warning = 2,
    Require = 3,
    Override = 4,
    Append = 5,
    AppendUnique = 6,
    Max = 7,
    Min = 8,
}

// Consts for the LLVM CallConv type, pre-cast to usize.

/// LLVM CallingConv::ID. Should we wrap this?
///
/// See <https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/IR/CallingConv.h>
#[derive(Copy, Clone, PartialEq, Debug)]
#[repr(C)]
pub enum CallConv {
    CCallConv = 0,
    FastCallConv = 8,
    ColdCallConv = 9,
    PreserveMost = 14,
    PreserveAll = 15,
    Tail = 18,
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
    AvrNonBlockingInterrupt = 84,
    AvrInterrupt = 85,
}

/// Must match the layout of `LLVMLinkage`.
#[derive(Copy, Clone, PartialEq, TryFromU32)]
#[repr(C)]
pub enum Linkage {
    ExternalLinkage = 0,
    AvailableExternallyLinkage = 1,
    LinkOnceAnyLinkage = 2,
    LinkOnceODRLinkage = 3,
    #[deprecated = "marked obsolete by LLVM"]
    LinkOnceODRAutoHideLinkage = 4,
    WeakAnyLinkage = 5,
    WeakODRLinkage = 6,
    AppendingLinkage = 7,
    InternalLinkage = 8,
    PrivateLinkage = 9,
    #[deprecated = "marked obsolete by LLVM"]
    DLLImportLinkage = 10,
    #[deprecated = "marked obsolete by LLVM"]
    DLLExportLinkage = 11,
    ExternalWeakLinkage = 12,
    #[deprecated = "marked obsolete by LLVM"]
    GhostLinkage = 13,
    CommonLinkage = 14,
    LinkerPrivateLinkage = 15,
    LinkerPrivateWeakLinkage = 16,
}

/// Must match the layout of `LLVMVisibility`.
#[repr(C)]
#[derive(Copy, Clone, PartialEq, TryFromU32)]
pub enum Visibility {
    Default = 0,
    Hidden = 1,
    Protected = 2,
}

impl Visibility {
    pub fn from_generic(visibility: SymbolVisibility) -> Self {
        match visibility {
            SymbolVisibility::Hidden => Visibility::Hidden,
            SymbolVisibility::Protected => Visibility::Protected,
            SymbolVisibility::Interposable => Visibility::Default,
        }
    }
}

/// LLVMUnnamedAddr
#[repr(C)]
pub enum UnnamedAddr {
    No,
    Local,
    Global,
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

/// Must match the layout of `LLVMRustAttributeKind`.
/// Semantically a subset of the C++ enum llvm::Attribute::AttrKind,
/// though it is not ABI compatible (since it's a C++ enum)
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub enum AttributeKind {
    AlwaysInline = 0,
    ByVal = 1,
    Cold = 2,
    InlineHint = 3,
    MinSize = 4,
    Naked = 5,
    NoAlias = 6,
    NoCapture = 7,
    NoInline = 8,
    NonNull = 9,
    NoRedZone = 10,
    NoReturn = 11,
    NoUnwind = 12,
    OptimizeForSize = 13,
    ReadOnly = 14,
    SExt = 15,
    StructRet = 16,
    UWTable = 17,
    ZExt = 18,
    InReg = 19,
    SanitizeThread = 20,
    SanitizeAddress = 21,
    SanitizeMemory = 22,
    NonLazyBind = 23,
    OptimizeNone = 24,
    ReadNone = 26,
    SanitizeHWAddress = 28,
    WillReturn = 29,
    StackProtectReq = 30,
    StackProtectStrong = 31,
    StackProtect = 32,
    NoUndef = 33,
    SanitizeMemTag = 34,
    NoCfCheck = 35,
    ShadowCallStack = 36,
    AllocSize = 37,
    AllocatedPointer = 38,
    AllocAlign = 39,
    SanitizeSafeStack = 40,
    FnRetThunkExtern = 41,
    Writable = 42,
    DeadOnUnwind = 43,
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
        use rustc_codegen_ssa::common::IntPredicate as Common;
        match intpre {
            Common::IntEQ => Self::IntEQ,
            Common::IntNE => Self::IntNE,
            Common::IntUGT => Self::IntUGT,
            Common::IntUGE => Self::IntUGE,
            Common::IntULT => Self::IntULT,
            Common::IntULE => Self::IntULE,
            Common::IntSGT => Self::IntSGT,
            Common::IntSGE => Self::IntSGE,
            Common::IntSLT => Self::IntSLT,
            Common::IntSLE => Self::IntSLE,
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
    pub fn from_generic(realp: rustc_codegen_ssa::common::RealPredicate) -> Self {
        use rustc_codegen_ssa::common::RealPredicate as Common;
        match realp {
            Common::RealPredicateFalse => Self::RealPredicateFalse,
            Common::RealOEQ => Self::RealOEQ,
            Common::RealOGT => Self::RealOGT,
            Common::RealOGE => Self::RealOGE,
            Common::RealOLT => Self::RealOLT,
            Common::RealOLE => Self::RealOLE,
            Common::RealONE => Self::RealONE,
            Common::RealORD => Self::RealORD,
            Common::RealUNO => Self::RealUNO,
            Common::RealUEQ => Self::RealUEQ,
            Common::RealUGT => Self::RealUGT,
            Common::RealUGE => Self::RealUGE,
            Common::RealULT => Self::RealULT,
            Common::RealULE => Self::RealULE,
            Common::RealUNE => Self::RealUNE,
            Common::RealPredicateTrue => Self::RealPredicateTrue,
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
    Token = 16,
    ScalableVector = 17,
    BFloat = 18,
    X86_AMX = 19,
}

impl TypeKind {
    pub fn to_generic(self) -> rustc_codegen_ssa::common::TypeKind {
        use rustc_codegen_ssa::common::TypeKind as Common;
        match self {
            Self::Void => Common::Void,
            Self::Half => Common::Half,
            Self::Float => Common::Float,
            Self::Double => Common::Double,
            Self::X86_FP80 => Common::X86_FP80,
            Self::FP128 => Common::FP128,
            Self::PPC_FP128 => Common::PPC_FP128,
            Self::Label => Common::Label,
            Self::Integer => Common::Integer,
            Self::Function => Common::Function,
            Self::Struct => Common::Struct,
            Self::Array => Common::Array,
            Self::Pointer => Common::Pointer,
            Self::Vector => Common::Vector,
            Self::Metadata => Common::Metadata,
            Self::Token => Common::Token,
            Self::ScalableVector => Common::ScalableVector,
            Self::BFloat => Common::BFloat,
            Self::X86_AMX => Common::X86_AMX,
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
        use rustc_codegen_ssa::common::AtomicRmwBinOp as Common;
        match op {
            Common::AtomicXchg => Self::AtomicXchg,
            Common::AtomicAdd => Self::AtomicAdd,
            Common::AtomicSub => Self::AtomicSub,
            Common::AtomicAnd => Self::AtomicAnd,
            Common::AtomicNand => Self::AtomicNand,
            Common::AtomicOr => Self::AtomicOr,
            Common::AtomicXor => Self::AtomicXor,
            Common::AtomicMax => Self::AtomicMax,
            Common::AtomicMin => Self::AtomicMin,
            Common::AtomicUMax => Self::AtomicUMax,
            Common::AtomicUMin => Self::AtomicUMin,
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
        use rustc_codegen_ssa::common::AtomicOrdering as Common;
        match ao {
            Common::Unordered => Self::Unordered,
            Common::Relaxed => Self::Monotonic,
            Common::Acquire => Self::Acquire,
            Common::Release => Self::Release,
            Common::AcquireRelease => Self::AcquireRelease,
            Common::SequentiallyConsistent => Self::SequentiallyConsistent,
        }
    }
}

/// LLVMRustFileType
#[derive(Copy, Clone)]
#[repr(C)]
pub enum FileType {
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
    MD_unpredictable = 15,
    MD_align = 17,
    MD_type = 19,
    MD_vcall_visibility = 28,
    MD_noundef = 29,
    MD_kcfi_type = 36,
}

/// LLVMRustAsmDialect
#[derive(Copy, Clone, PartialEq)]
#[repr(C)]
pub enum AsmDialect {
    Att,
    Intel,
}

/// LLVMRustCodeGenOptLevel
#[derive(Copy, Clone, PartialEq)]
#[repr(C)]
pub enum CodeGenOptLevel {
    None,
    Less,
    Default,
    Aggressive,
}

/// LLVMRustPassBuilderOptLevel
#[repr(C)]
pub enum PassBuilderOptLevel {
    O0,
    O1,
    O2,
    O3,
    Os,
    Oz,
}

/// LLVMRustOptStage
#[derive(PartialEq)]
#[repr(C)]
pub enum OptStage {
    PreLinkNoLTO,
    PreLinkThinLTO,
    PreLinkFatLTO,
    ThinLTO,
    FatLTO,
}

/// LLVMRustSanitizerOptions
#[repr(C)]
pub struct SanitizerOptions {
    pub sanitize_address: bool,
    pub sanitize_address_recover: bool,
    pub sanitize_cfi: bool,
    pub sanitize_dataflow: bool,
    pub sanitize_dataflow_abilist: *const *const c_char,
    pub sanitize_dataflow_abilist_len: size_t,
    pub sanitize_kcfi: bool,
    pub sanitize_memory: bool,
    pub sanitize_memory_recover: bool,
    pub sanitize_memory_track_origins: c_int,
    pub sanitize_thread: bool,
    pub sanitize_hwaddress: bool,
    pub sanitize_hwaddress_recover: bool,
    pub sanitize_kernel_address: bool,
    pub sanitize_kernel_address_recover: bool,
}

/// LLVMRelocMode
#[derive(Copy, Clone, PartialEq)]
#[repr(C)]
pub enum RelocModel {
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
    Tiny,
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
    Unsupported,
    SrcMgr,
}

/// LLVMRustDiagnosticLevel
#[derive(Copy, Clone)]
#[repr(C)]
#[allow(dead_code)] // Variants constructed by C++.
pub enum DiagnosticLevel {
    Error,
    Warning,
    Note,
    Remark,
}

/// LLVMRustArchiveKind
#[derive(Copy, Clone)]
#[repr(C)]
pub enum ArchiveKind {
    K_GNU,
    K_BSD,
    K_DARWIN,
    K_COFF,
    K_AIXBIG,
}

unsafe extern "C" {
    // LLVMRustThinLTOData
    pub type ThinLTOData;

    // LLVMRustThinLTOBuffer
    pub type ThinLTOBuffer;
}

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
    LocalExec,
}

/// LLVMRustTailCallKind
#[derive(Copy, Clone)]
#[repr(C)]
pub enum TailCallKind {
    None,
    Tail,
    MustTail,
    NoTail,
}

/// LLVMRustChecksumKind
#[derive(Copy, Clone)]
#[repr(C)]
pub enum ChecksumKind {
    None,
    MD5,
    SHA1,
    SHA256,
}

/// LLVMRustMemoryEffects
#[derive(Copy, Clone)]
#[repr(C)]
pub enum MemoryEffects {
    None,
    ReadOnly,
    InaccessibleMemOnly,
}

unsafe extern "C" {
    type Opaque;
}
#[repr(C)]
struct InvariantOpaque<'a> {
    _marker: PhantomData<&'a mut &'a ()>,
    _opaque: Opaque,
}

// Opaque pointer types
unsafe extern "C" {
    pub type Module;
    pub type Context;
    pub type Type;
    pub type Value;
    pub type ConstantInt;
    pub type Attribute;
    pub type Metadata;
    pub type BasicBlock;
    pub type Comdat;
}
#[repr(C)]
pub struct Builder<'a>(InvariantOpaque<'a>);
#[repr(C)]
pub struct PassManager<'a>(InvariantOpaque<'a>);
unsafe extern "C" {
    pub type Pass;
    pub type TargetMachine;
    pub type Archive;
}
#[repr(C)]
pub struct ArchiveIterator<'a>(InvariantOpaque<'a>);
#[repr(C)]
pub struct ArchiveChild<'a>(InvariantOpaque<'a>);
unsafe extern "C" {
    pub type Twine;
    pub type DiagnosticInfo;
    pub type SMDiagnostic;
}
#[repr(C)]
pub struct RustArchiveMember<'a>(InvariantOpaque<'a>);
/// Opaque pointee of `LLVMOperandBundleRef`.
#[repr(C)]
pub(crate) struct OperandBundle<'a>(InvariantOpaque<'a>);
#[repr(C)]
pub struct Linker<'a>(InvariantOpaque<'a>);

unsafe extern "C" {
    pub type DiagnosticHandler;
}

pub type DiagnosticHandlerTy = unsafe extern "C" fn(&DiagnosticInfo, *mut c_void);
pub type InlineAsmDiagHandlerTy = unsafe extern "C" fn(&SMDiagnostic, *const c_void, c_uint);

pub mod debuginfo {
    use bitflags::bitflags;

    use super::{InvariantOpaque, Metadata};

    #[repr(C)]
    pub struct DIBuilder<'a>(InvariantOpaque<'a>);

    pub type DIDescriptor = Metadata;
    pub type DILocation = Metadata;
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
        #[derive(Clone, Copy, Default)]
        pub struct DIFlags: u32 {
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
        #[derive(Clone, Copy, Default)]
        pub struct DISPFlags: u32 {
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
        DebugDirectivesOnly,
    }

    impl DebugEmissionKind {
        pub fn from_generic(kind: rustc_session::config::DebugInfo) -> Self {
            // We should be setting LLVM's emission kind to `LineTablesOnly` if
            // we are compiling with "limited" debuginfo. However, some of the
            // existing tools relied on slightly more debuginfo being generated than
            // would be the case with `LineTablesOnly`, and we did not want to break
            // these tools in a "drive-by fix", without a good idea or plan about
            // what limited debuginfo should exactly look like. So for now we are
            // instead adding a new debuginfo option "line-tables-only" so as to
            // not break anything and to allow users to have 'limited' debug info.
            //
            // See https://github.com/rust-lang/rust/issues/60020 for details.
            use rustc_session::config::DebugInfo;
            match kind {
                DebugInfo::None => DebugEmissionKind::NoDebug,
                DebugInfo::LineDirectivesOnly => DebugEmissionKind::DebugDirectivesOnly,
                DebugInfo::LineTablesOnly => DebugEmissionKind::LineTablesOnly,
                DebugInfo::Limited | DebugInfo::Full => DebugEmissionKind::FullDebug,
            }
        }
    }

    /// LLVMRustDebugNameTableKind
    #[derive(Clone, Copy)]
    #[repr(C)]
    pub enum DebugNameTableKind {
        Default,
        Gnu,
        None,
    }
}

use bitflags::bitflags;
// These values **must** match with LLVMRustAllocKindFlags
bitflags! {
    #[repr(transparent)]
    #[derive(Default)]
    pub struct AllocKindFlags : u64 {
        const Unknown = 0;
        const Alloc = 1;
        const Realloc = 1 << 1;
        const Free = 1 << 2;
        const Uninitialized = 1 << 3;
        const Zeroed = 1 << 4;
        const Aligned = 1 << 5;
    }
}

unsafe extern "C" {
    pub type ModuleBuffer;
}

pub type SelfProfileBeforePassCallback =
    unsafe extern "C" fn(*mut c_void, *const c_char, *const c_char);
pub type SelfProfileAfterPassCallback = unsafe extern "C" fn(*mut c_void);

pub type GetSymbolsCallback = unsafe extern "C" fn(*mut c_void, *const c_char) -> *mut c_void;
pub type GetSymbolsErrorCallback = unsafe extern "C" fn(*const c_char) -> *mut c_void;

unsafe extern "C" {
    // Create and destroy contexts.
    pub fn LLVMContextDispose(C: &'static mut Context);
    pub fn LLVMGetMDKindIDInContext(C: &Context, Name: *const c_char, SLen: c_uint) -> c_uint;

    // Create modules.
    pub fn LLVMModuleCreateWithNameInContext(ModuleID: *const c_char, C: &Context) -> &Module;
    pub fn LLVMGetModuleContext(M: &Module) -> &Context;
    pub fn LLVMCloneModule(M: &Module) -> &Module;

    /// Data layout. See Module::getDataLayout.
    pub fn LLVMGetDataLayoutStr(M: &Module) -> *const c_char;
    pub fn LLVMSetDataLayout(M: &Module, Triple: *const c_char);

    /// See Module::setModuleInlineAsm.
    pub fn LLVMAppendModuleInlineAsm(M: &Module, Asm: *const c_char, Len: size_t);

    // Operations on integer types
    pub fn LLVMInt1TypeInContext(C: &Context) -> &Type;
    pub fn LLVMInt8TypeInContext(C: &Context) -> &Type;
    pub fn LLVMInt16TypeInContext(C: &Context) -> &Type;
    pub fn LLVMInt32TypeInContext(C: &Context) -> &Type;
    pub fn LLVMInt64TypeInContext(C: &Context) -> &Type;
    pub fn LLVMIntTypeInContext(C: &Context, NumBits: c_uint) -> &Type;

    pub fn LLVMGetIntTypeWidth(IntegerTy: &Type) -> c_uint;

    // Operations on real types
    pub fn LLVMHalfTypeInContext(C: &Context) -> &Type;
    pub fn LLVMFloatTypeInContext(C: &Context) -> &Type;
    pub fn LLVMDoubleTypeInContext(C: &Context) -> &Type;
    pub fn LLVMFP128TypeInContext(C: &Context) -> &Type;

    // Operations on function types
    pub fn LLVMFunctionType<'a>(
        ReturnType: &'a Type,
        ParamTypes: *const &'a Type,
        ParamCount: c_uint,
        IsVarArg: Bool,
    ) -> &'a Type;
    pub fn LLVMCountParamTypes(FunctionTy: &Type) -> c_uint;
    pub fn LLVMGetParamTypes<'a>(FunctionTy: &'a Type, Dest: *mut &'a Type);

    // Operations on struct types
    pub fn LLVMStructTypeInContext<'a>(
        C: &'a Context,
        ElementTypes: *const &'a Type,
        ElementCount: c_uint,
        Packed: Bool,
    ) -> &'a Type;

    // Operations on array, pointer, and vector types (sequence types)
    pub fn LLVMPointerTypeInContext(C: &Context, AddressSpace: c_uint) -> &Type;
    pub fn LLVMVectorType(ElementType: &Type, ElementCount: c_uint) -> &Type;

    pub fn LLVMGetElementType(Ty: &Type) -> &Type;
    pub fn LLVMGetVectorSize(VectorTy: &Type) -> c_uint;

    // Operations on other types
    pub fn LLVMVoidTypeInContext(C: &Context) -> &Type;
    pub fn LLVMTokenTypeInContext(C: &Context) -> &Type;
    pub fn LLVMMetadataTypeInContext(C: &Context) -> &Type;

    // Operations on all values
    pub fn LLVMTypeOf(Val: &Value) -> &Type;
    pub fn LLVMGetValueName2(Val: &Value, Length: *mut size_t) -> *const c_char;
    pub fn LLVMSetValueName2(Val: &Value, Name: *const c_char, NameLen: size_t);
    pub fn LLVMReplaceAllUsesWith<'a>(OldVal: &'a Value, NewVal: &'a Value);
    pub fn LLVMSetMetadata<'a>(Val: &'a Value, KindID: c_uint, Node: &'a Value);
    pub fn LLVMGlobalSetMetadata<'a>(Val: &'a Value, KindID: c_uint, Metadata: &'a Metadata);
    pub fn LLVMValueAsMetadata(Node: &Value) -> &Metadata;

    // Operations on constants of any type
    pub fn LLVMConstNull(Ty: &Type) -> &Value;
    pub fn LLVMGetUndef(Ty: &Type) -> &Value;
    pub fn LLVMGetPoison(Ty: &Type) -> &Value;

    // Operations on metadata
    pub fn LLVMMDStringInContext2(C: &Context, Str: *const c_char, SLen: size_t) -> &Metadata;
    pub fn LLVMMDNodeInContext2<'a>(
        C: &'a Context,
        Vals: *const &'a Metadata,
        Count: size_t,
    ) -> &'a Metadata;
    pub fn LLVMAddNamedMetadataOperand<'a>(M: &'a Module, Name: *const c_char, Val: &'a Value);

    // Operations on scalar constants
    pub fn LLVMConstInt(IntTy: &Type, N: c_ulonglong, SignExtend: Bool) -> &Value;
    pub fn LLVMConstIntOfArbitraryPrecision(IntTy: &Type, Wn: c_uint, Ws: *const u64) -> &Value;
    pub fn LLVMConstReal(RealTy: &Type, N: f64) -> &Value;

    // Operations on composite constants
    pub fn LLVMConstArray2<'a>(
        ElementTy: &'a Type,
        ConstantVals: *const &'a Value,
        Length: u64,
    ) -> &'a Value;
    pub fn LLVMArrayType2(ElementType: &Type, ElementCount: u64) -> &Type;
    pub fn LLVMConstStringInContext2(
        C: &Context,
        Str: *const c_char,
        Length: size_t,
        DontNullTerminate: Bool,
    ) -> &Value;
    pub fn LLVMConstStructInContext<'a>(
        C: &'a Context,
        ConstantVals: *const &'a Value,
        Count: c_uint,
        Packed: Bool,
    ) -> &'a Value;
    pub fn LLVMConstVector(ScalarConstantVals: *const &Value, Size: c_uint) -> &Value;

    // Constant expressions
    pub fn LLVMConstInBoundsGEP2<'a>(
        ty: &'a Type,
        ConstantVal: &'a Value,
        ConstantIndices: *const &'a Value,
        NumIndices: c_uint,
    ) -> &'a Value;
    pub fn LLVMConstPtrToInt<'a>(ConstantVal: &'a Value, ToType: &'a Type) -> &'a Value;
    pub fn LLVMConstIntToPtr<'a>(ConstantVal: &'a Value, ToType: &'a Type) -> &'a Value;
    pub fn LLVMConstBitCast<'a>(ConstantVal: &'a Value, ToType: &'a Type) -> &'a Value;
    pub fn LLVMGetAggregateElement(ConstantVal: &Value, Idx: c_uint) -> Option<&Value>;

    // Operations on global variables, functions, and aliases (globals)
    pub fn LLVMIsDeclaration(Global: &Value) -> Bool;
    pub fn LLVMGetLinkage(Global: &Value) -> RawEnum<Linkage>;
    pub fn LLVMSetLinkage(Global: &Value, RustLinkage: Linkage);
    pub fn LLVMSetSection(Global: &Value, Section: *const c_char);
    pub fn LLVMGetVisibility(Global: &Value) -> RawEnum<Visibility>;
    pub fn LLVMSetVisibility(Global: &Value, Viz: Visibility);
    pub fn LLVMGetAlignment(Global: &Value) -> c_uint;
    pub fn LLVMSetAlignment(Global: &Value, Bytes: c_uint);
    pub fn LLVMSetDLLStorageClass(V: &Value, C: DLLStorageClass);
    pub fn LLVMGlobalGetValueType(Global: &Value) -> &Type;

    // Operations on global variables
    pub fn LLVMIsAGlobalVariable(GlobalVar: &Value) -> Option<&Value>;
    pub fn LLVMAddGlobal<'a>(M: &'a Module, Ty: &'a Type, Name: *const c_char) -> &'a Value;
    pub fn LLVMGetNamedGlobal(M: &Module, Name: *const c_char) -> Option<&Value>;
    pub fn LLVMGetFirstGlobal(M: &Module) -> Option<&Value>;
    pub fn LLVMGetNextGlobal(GlobalVar: &Value) -> Option<&Value>;
    pub fn LLVMDeleteGlobal(GlobalVar: &Value);
    pub fn LLVMGetInitializer(GlobalVar: &Value) -> Option<&Value>;
    pub fn LLVMSetInitializer<'a>(GlobalVar: &'a Value, ConstantVal: &'a Value);
    pub fn LLVMIsThreadLocal(GlobalVar: &Value) -> Bool;
    pub fn LLVMSetThreadLocalMode(GlobalVar: &Value, Mode: ThreadLocalMode);
    pub fn LLVMIsGlobalConstant(GlobalVar: &Value) -> Bool;
    pub fn LLVMSetGlobalConstant(GlobalVar: &Value, IsConstant: Bool);
    pub fn LLVMSetTailCall(CallInst: &Value, IsTailCall: Bool);

    // Operations on attributes
    pub fn LLVMCreateStringAttribute(
        C: &Context,
        Name: *const c_char,
        NameLen: c_uint,
        Value: *const c_char,
        ValueLen: c_uint,
    ) -> &Attribute;

    // Operations on functions
    pub fn LLVMSetFunctionCallConv(Fn: &Value, CC: c_uint);

    // Operations on parameters
    pub fn LLVMIsAArgument(Val: &Value) -> Option<&Value>;
    pub fn LLVMCountParams(Fn: &Value) -> c_uint;
    pub fn LLVMGetParam(Fn: &Value, Index: c_uint) -> &Value;

    // Operations on basic blocks
    pub fn LLVMGetBasicBlockParent(BB: &BasicBlock) -> &Value;
    pub fn LLVMAppendBasicBlockInContext<'a>(
        C: &'a Context,
        Fn: &'a Value,
        Name: *const c_char,
    ) -> &'a BasicBlock;

    // Operations on instructions
    pub fn LLVMIsAInstruction(Val: &Value) -> Option<&Value>;
    pub fn LLVMGetFirstBasicBlock(Fn: &Value) -> &BasicBlock;

    // Operations on call sites
    pub fn LLVMSetInstructionCallConv(Instr: &Value, CC: c_uint);

    // Operations on load/store instructions (only)
    pub fn LLVMSetVolatile(MemoryAccessInst: &Value, volatile: Bool);

    // Operations on phi nodes
    pub fn LLVMAddIncoming<'a>(
        PhiNode: &'a Value,
        IncomingValues: *const &'a Value,
        IncomingBlocks: *const &'a BasicBlock,
        Count: c_uint,
    );

    // Instruction builders
    pub fn LLVMCreateBuilderInContext(C: &Context) -> &mut Builder<'_>;
    pub fn LLVMPositionBuilderAtEnd<'a>(Builder: &Builder<'a>, Block: &'a BasicBlock);
    pub fn LLVMGetInsertBlock<'a>(Builder: &Builder<'a>) -> &'a BasicBlock;
    pub fn LLVMDisposeBuilder<'a>(Builder: &'a mut Builder<'a>);

    // Metadata
    pub fn LLVMSetCurrentDebugLocation2<'a>(Builder: &Builder<'a>, Loc: *const Metadata);
    pub fn LLVMGetCurrentDebugLocation2<'a>(Builder: &Builder<'a>) -> Option<&'a Metadata>;

    // Terminators
    pub fn LLVMBuildRetVoid<'a>(B: &Builder<'a>) -> &'a Value;
    pub fn LLVMBuildRet<'a>(B: &Builder<'a>, V: &'a Value) -> &'a Value;
    pub fn LLVMBuildBr<'a>(B: &Builder<'a>, Dest: &'a BasicBlock) -> &'a Value;
    pub fn LLVMBuildCondBr<'a>(
        B: &Builder<'a>,
        If: &'a Value,
        Then: &'a BasicBlock,
        Else: &'a BasicBlock,
    ) -> &'a Value;
    pub fn LLVMBuildSwitch<'a>(
        B: &Builder<'a>,
        V: &'a Value,
        Else: &'a BasicBlock,
        NumCases: c_uint,
    ) -> &'a Value;
    pub fn LLVMBuildLandingPad<'a>(
        B: &Builder<'a>,
        Ty: &'a Type,
        PersFn: Option<&'a Value>,
        NumClauses: c_uint,
        Name: *const c_char,
    ) -> &'a Value;
    pub fn LLVMBuildResume<'a>(B: &Builder<'a>, Exn: &'a Value) -> &'a Value;
    pub fn LLVMBuildUnreachable<'a>(B: &Builder<'a>) -> &'a Value;

    pub fn LLVMBuildCleanupPad<'a>(
        B: &Builder<'a>,
        ParentPad: Option<&'a Value>,
        Args: *const &'a Value,
        NumArgs: c_uint,
        Name: *const c_char,
    ) -> Option<&'a Value>;
    pub fn LLVMBuildCleanupRet<'a>(
        B: &Builder<'a>,
        CleanupPad: &'a Value,
        BB: Option<&'a BasicBlock>,
    ) -> Option<&'a Value>;
    pub fn LLVMBuildCatchPad<'a>(
        B: &Builder<'a>,
        ParentPad: &'a Value,
        Args: *const &'a Value,
        NumArgs: c_uint,
        Name: *const c_char,
    ) -> Option<&'a Value>;
    pub fn LLVMBuildCatchRet<'a>(
        B: &Builder<'a>,
        CatchPad: &'a Value,
        BB: &'a BasicBlock,
    ) -> Option<&'a Value>;
    pub fn LLVMBuildCatchSwitch<'a>(
        Builder: &Builder<'a>,
        ParentPad: Option<&'a Value>,
        UnwindBB: Option<&'a BasicBlock>,
        NumHandlers: c_uint,
        Name: *const c_char,
    ) -> Option<&'a Value>;
    pub fn LLVMAddHandler<'a>(CatchSwitch: &'a Value, Dest: &'a BasicBlock);
    pub fn LLVMSetPersonalityFn<'a>(Func: &'a Value, Pers: &'a Value);

    // Add a case to the switch instruction
    pub fn LLVMAddCase<'a>(Switch: &'a Value, OnVal: &'a Value, Dest: &'a BasicBlock);

    // Add a clause to the landing pad instruction
    pub fn LLVMAddClause<'a>(LandingPad: &'a Value, ClauseVal: &'a Value);

    // Set the cleanup on a landing pad instruction
    pub fn LLVMSetCleanup(LandingPad: &Value, Val: Bool);

    // Arithmetic
    pub fn LLVMBuildAdd<'a>(
        B: &Builder<'a>,
        LHS: &'a Value,
        RHS: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;
    pub fn LLVMBuildFAdd<'a>(
        B: &Builder<'a>,
        LHS: &'a Value,
        RHS: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;
    pub fn LLVMBuildSub<'a>(
        B: &Builder<'a>,
        LHS: &'a Value,
        RHS: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;
    pub fn LLVMBuildFSub<'a>(
        B: &Builder<'a>,
        LHS: &'a Value,
        RHS: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;
    pub fn LLVMBuildMul<'a>(
        B: &Builder<'a>,
        LHS: &'a Value,
        RHS: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;
    pub fn LLVMBuildFMul<'a>(
        B: &Builder<'a>,
        LHS: &'a Value,
        RHS: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;
    pub fn LLVMBuildUDiv<'a>(
        B: &Builder<'a>,
        LHS: &'a Value,
        RHS: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;
    pub fn LLVMBuildExactUDiv<'a>(
        B: &Builder<'a>,
        LHS: &'a Value,
        RHS: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;
    pub fn LLVMBuildSDiv<'a>(
        B: &Builder<'a>,
        LHS: &'a Value,
        RHS: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;
    pub fn LLVMBuildExactSDiv<'a>(
        B: &Builder<'a>,
        LHS: &'a Value,
        RHS: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;
    pub fn LLVMBuildFDiv<'a>(
        B: &Builder<'a>,
        LHS: &'a Value,
        RHS: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;
    pub fn LLVMBuildURem<'a>(
        B: &Builder<'a>,
        LHS: &'a Value,
        RHS: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;
    pub fn LLVMBuildSRem<'a>(
        B: &Builder<'a>,
        LHS: &'a Value,
        RHS: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;
    pub fn LLVMBuildFRem<'a>(
        B: &Builder<'a>,
        LHS: &'a Value,
        RHS: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;
    pub fn LLVMBuildShl<'a>(
        B: &Builder<'a>,
        LHS: &'a Value,
        RHS: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;
    pub fn LLVMBuildLShr<'a>(
        B: &Builder<'a>,
        LHS: &'a Value,
        RHS: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;
    pub fn LLVMBuildAShr<'a>(
        B: &Builder<'a>,
        LHS: &'a Value,
        RHS: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;
    pub fn LLVMBuildNSWAdd<'a>(
        B: &Builder<'a>,
        LHS: &'a Value,
        RHS: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;
    pub fn LLVMBuildNUWAdd<'a>(
        B: &Builder<'a>,
        LHS: &'a Value,
        RHS: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;
    pub fn LLVMBuildNSWSub<'a>(
        B: &Builder<'a>,
        LHS: &'a Value,
        RHS: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;
    pub fn LLVMBuildNUWSub<'a>(
        B: &Builder<'a>,
        LHS: &'a Value,
        RHS: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;
    pub fn LLVMBuildNSWMul<'a>(
        B: &Builder<'a>,
        LHS: &'a Value,
        RHS: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;
    pub fn LLVMBuildNUWMul<'a>(
        B: &Builder<'a>,
        LHS: &'a Value,
        RHS: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;
    pub fn LLVMBuildAnd<'a>(
        B: &Builder<'a>,
        LHS: &'a Value,
        RHS: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;
    pub fn LLVMBuildOr<'a>(
        B: &Builder<'a>,
        LHS: &'a Value,
        RHS: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;
    pub fn LLVMBuildXor<'a>(
        B: &Builder<'a>,
        LHS: &'a Value,
        RHS: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;
    pub fn LLVMBuildNeg<'a>(B: &Builder<'a>, V: &'a Value, Name: *const c_char) -> &'a Value;
    pub fn LLVMBuildFNeg<'a>(B: &Builder<'a>, V: &'a Value, Name: *const c_char) -> &'a Value;
    pub fn LLVMBuildNot<'a>(B: &Builder<'a>, V: &'a Value, Name: *const c_char) -> &'a Value;

    // Memory
    pub fn LLVMBuildAlloca<'a>(B: &Builder<'a>, Ty: &'a Type, Name: *const c_char) -> &'a Value;
    pub fn LLVMBuildArrayAlloca<'a>(
        B: &Builder<'a>,
        Ty: &'a Type,
        Val: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;
    pub fn LLVMBuildLoad2<'a>(
        B: &Builder<'a>,
        Ty: &'a Type,
        PointerVal: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;

    pub fn LLVMBuildStore<'a>(B: &Builder<'a>, Val: &'a Value, Ptr: &'a Value) -> &'a Value;

    pub fn LLVMBuildGEP2<'a>(
        B: &Builder<'a>,
        Ty: &'a Type,
        Pointer: &'a Value,
        Indices: *const &'a Value,
        NumIndices: c_uint,
        Name: *const c_char,
    ) -> &'a Value;
    pub fn LLVMBuildInBoundsGEP2<'a>(
        B: &Builder<'a>,
        Ty: &'a Type,
        Pointer: &'a Value,
        Indices: *const &'a Value,
        NumIndices: c_uint,
        Name: *const c_char,
    ) -> &'a Value;

    // Casts
    pub fn LLVMBuildTrunc<'a>(
        B: &Builder<'a>,
        Val: &'a Value,
        DestTy: &'a Type,
        Name: *const c_char,
    ) -> &'a Value;
    pub fn LLVMBuildZExt<'a>(
        B: &Builder<'a>,
        Val: &'a Value,
        DestTy: &'a Type,
        Name: *const c_char,
    ) -> &'a Value;
    pub fn LLVMBuildSExt<'a>(
        B: &Builder<'a>,
        Val: &'a Value,
        DestTy: &'a Type,
        Name: *const c_char,
    ) -> &'a Value;
    pub fn LLVMBuildFPToUI<'a>(
        B: &Builder<'a>,
        Val: &'a Value,
        DestTy: &'a Type,
        Name: *const c_char,
    ) -> &'a Value;
    pub fn LLVMBuildFPToSI<'a>(
        B: &Builder<'a>,
        Val: &'a Value,
        DestTy: &'a Type,
        Name: *const c_char,
    ) -> &'a Value;
    pub fn LLVMBuildUIToFP<'a>(
        B: &Builder<'a>,
        Val: &'a Value,
        DestTy: &'a Type,
        Name: *const c_char,
    ) -> &'a Value;
    pub fn LLVMBuildSIToFP<'a>(
        B: &Builder<'a>,
        Val: &'a Value,
        DestTy: &'a Type,
        Name: *const c_char,
    ) -> &'a Value;
    pub fn LLVMBuildFPTrunc<'a>(
        B: &Builder<'a>,
        Val: &'a Value,
        DestTy: &'a Type,
        Name: *const c_char,
    ) -> &'a Value;
    pub fn LLVMBuildFPExt<'a>(
        B: &Builder<'a>,
        Val: &'a Value,
        DestTy: &'a Type,
        Name: *const c_char,
    ) -> &'a Value;
    pub fn LLVMBuildPtrToInt<'a>(
        B: &Builder<'a>,
        Val: &'a Value,
        DestTy: &'a Type,
        Name: *const c_char,
    ) -> &'a Value;
    pub fn LLVMBuildIntToPtr<'a>(
        B: &Builder<'a>,
        Val: &'a Value,
        DestTy: &'a Type,
        Name: *const c_char,
    ) -> &'a Value;
    pub fn LLVMBuildBitCast<'a>(
        B: &Builder<'a>,
        Val: &'a Value,
        DestTy: &'a Type,
        Name: *const c_char,
    ) -> &'a Value;
    pub fn LLVMBuildPointerCast<'a>(
        B: &Builder<'a>,
        Val: &'a Value,
        DestTy: &'a Type,
        Name: *const c_char,
    ) -> &'a Value;
    pub fn LLVMBuildIntCast2<'a>(
        B: &Builder<'a>,
        Val: &'a Value,
        DestTy: &'a Type,
        IsSigned: Bool,
        Name: *const c_char,
    ) -> &'a Value;

    // Comparisons
    pub fn LLVMBuildICmp<'a>(
        B: &Builder<'a>,
        Op: c_uint,
        LHS: &'a Value,
        RHS: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;
    pub fn LLVMBuildFCmp<'a>(
        B: &Builder<'a>,
        Op: c_uint,
        LHS: &'a Value,
        RHS: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;

    // Miscellaneous instructions
    pub fn LLVMBuildPhi<'a>(B: &Builder<'a>, Ty: &'a Type, Name: *const c_char) -> &'a Value;
    pub fn LLVMBuildSelect<'a>(
        B: &Builder<'a>,
        If: &'a Value,
        Then: &'a Value,
        Else: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;
    pub fn LLVMBuildVAArg<'a>(
        B: &Builder<'a>,
        list: &'a Value,
        Ty: &'a Type,
        Name: *const c_char,
    ) -> &'a Value;
    pub fn LLVMBuildExtractElement<'a>(
        B: &Builder<'a>,
        VecVal: &'a Value,
        Index: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;
    pub fn LLVMBuildInsertElement<'a>(
        B: &Builder<'a>,
        VecVal: &'a Value,
        EltVal: &'a Value,
        Index: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;
    pub fn LLVMBuildShuffleVector<'a>(
        B: &Builder<'a>,
        V1: &'a Value,
        V2: &'a Value,
        Mask: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;
    pub fn LLVMBuildExtractValue<'a>(
        B: &Builder<'a>,
        AggVal: &'a Value,
        Index: c_uint,
        Name: *const c_char,
    ) -> &'a Value;
    pub fn LLVMBuildInsertValue<'a>(
        B: &Builder<'a>,
        AggVal: &'a Value,
        EltVal: &'a Value,
        Index: c_uint,
        Name: *const c_char,
    ) -> &'a Value;

    // Atomic Operations
    pub fn LLVMBuildAtomicCmpXchg<'a>(
        B: &Builder<'a>,
        LHS: &'a Value,
        CMP: &'a Value,
        RHS: &'a Value,
        Order: AtomicOrdering,
        FailureOrder: AtomicOrdering,
        SingleThreaded: Bool,
    ) -> &'a Value;

    pub fn LLVMSetWeak(CmpXchgInst: &Value, IsWeak: Bool);

    pub fn LLVMBuildAtomicRMW<'a>(
        B: &Builder<'a>,
        Op: AtomicRmwBinOp,
        LHS: &'a Value,
        RHS: &'a Value,
        Order: AtomicOrdering,
        SingleThreaded: Bool,
    ) -> &'a Value;

    pub fn LLVMBuildFence<'a>(
        B: &Builder<'a>,
        Order: AtomicOrdering,
        SingleThreaded: Bool,
        Name: *const c_char,
    ) -> &'a Value;

    /// Writes a module to the specified path. Returns 0 on success.
    pub fn LLVMWriteBitcodeToFile(M: &Module, Path: *const c_char) -> c_int;

    /// Creates a legacy pass manager -- only used for final codegen.
    pub fn LLVMCreatePassManager<'a>() -> &'a mut PassManager<'a>;

    pub fn LLVMAddAnalysisPasses<'a>(T: &'a TargetMachine, PM: &PassManager<'a>);

    pub fn LLVMGetHostCPUFeatures() -> *mut c_char;

    pub fn LLVMDisposeMessage(message: *mut c_char);

    pub fn LLVMIsMultithreaded() -> Bool;

    pub fn LLVMStructCreateNamed(C: &Context, Name: *const c_char) -> &Type;

    pub fn LLVMStructSetBody<'a>(
        StructTy: &'a Type,
        ElementTypes: *const &'a Type,
        ElementCount: c_uint,
        Packed: Bool,
    );

    pub fn LLVMMetadataAsValue<'a>(C: &'a Context, MD: &'a Metadata) -> &'a Value;

    pub fn LLVMSetUnnamedAddress(Global: &Value, UnnamedAddr: UnnamedAddr);

    pub fn LLVMIsAConstantInt(value_ref: &Value) -> Option<&ConstantInt>;

    pub fn LLVMGetOrInsertComdat(M: &Module, Name: *const c_char) -> &Comdat;
    pub fn LLVMSetComdat(V: &Value, C: &Comdat);

    pub(crate) fn LLVMCreateOperandBundle(
        Tag: *const c_char,
        TagLen: size_t,
        Args: *const &'_ Value,
        NumArgs: c_uint,
    ) -> *mut OperandBundle<'_>;
    pub(crate) fn LLVMDisposeOperandBundle(Bundle: ptr::NonNull<OperandBundle<'_>>);

    pub(crate) fn LLVMBuildCallWithOperandBundles<'a>(
        B: &Builder<'a>,
        Ty: &'a Type,
        Fn: &'a Value,
        Args: *const &'a Value,
        NumArgs: c_uint,
        Bundles: *const &OperandBundle<'a>,
        NumBundles: c_uint,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildInvokeWithOperandBundles<'a>(
        B: &Builder<'a>,
        Ty: &'a Type,
        Fn: &'a Value,
        Args: *const &'a Value,
        NumArgs: c_uint,
        Then: &'a BasicBlock,
        Catch: &'a BasicBlock,
        Bundles: *const &OperandBundle<'a>,
        NumBundles: c_uint,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildCallBr<'a>(
        B: &Builder<'a>,
        Ty: &'a Type,
        Fn: &'a Value,
        DefaultDest: &'a BasicBlock,
        IndirectDests: *const &'a BasicBlock,
        NumIndirectDests: c_uint,
        Args: *const &'a Value,
        NumArgs: c_uint,
        Bundles: *const &OperandBundle<'a>,
        NumBundles: c_uint,
        Name: *const c_char,
    ) -> &'a Value;
}

#[link(name = "llvm-wrapper", kind = "static")]
unsafe extern "C" {
    pub fn LLVMRustInstallErrorHandlers();
    pub fn LLVMRustDisableSystemDialogsOnCrash();

    // Create and destroy contexts.
    pub fn LLVMRustContextCreate(shouldDiscardNames: bool) -> &'static mut Context;

    /// See llvm::LLVMTypeKind::getTypeID.
    pub fn LLVMRustGetTypeKind(Ty: &Type) -> TypeKind;

    // Operations on all values
    pub fn LLVMRustGlobalAddMetadata<'a>(Val: &'a Value, KindID: c_uint, Metadata: &'a Metadata);
    pub fn LLVMRustIsNonGVFunctionPointerTy(Val: &Value) -> bool;

    // Operations on scalar constants
    pub fn LLVMRustConstIntGetZExtValue(ConstantVal: &ConstantInt, Value: &mut u64) -> bool;
    pub fn LLVMRustConstInt128Get(
        ConstantVal: &ConstantInt,
        SExt: bool,
        high: &mut u64,
        low: &mut u64,
    ) -> bool;

    // Operations on global variables, functions, and aliases (globals)
    pub fn LLVMRustSetDSOLocal(Global: &Value, is_dso_local: bool);

    // Operations on global variables
    pub fn LLVMRustGetOrInsertGlobal<'a>(
        M: &'a Module,
        Name: *const c_char,
        NameLen: size_t,
        T: &'a Type,
    ) -> &'a Value;
    pub fn LLVMRustInsertPrivateGlobal<'a>(M: &'a Module, T: &'a Type) -> &'a Value;
    pub fn LLVMRustGetNamedValue(
        M: &Module,
        Name: *const c_char,
        NameLen: size_t,
    ) -> Option<&Value>;
    pub fn LLVMRustSetTailCallKind(CallInst: &Value, TKC: TailCallKind);

    // Operations on attributes
    pub fn LLVMRustCreateAttrNoValue(C: &Context, attr: AttributeKind) -> &Attribute;
    pub fn LLVMRustCreateAlignmentAttr(C: &Context, bytes: u64) -> &Attribute;
    pub fn LLVMRustCreateDereferenceableAttr(C: &Context, bytes: u64) -> &Attribute;
    pub fn LLVMRustCreateDereferenceableOrNullAttr(C: &Context, bytes: u64) -> &Attribute;
    pub fn LLVMRustCreateByValAttr<'a>(C: &'a Context, ty: &'a Type) -> &'a Attribute;
    pub fn LLVMRustCreateStructRetAttr<'a>(C: &'a Context, ty: &'a Type) -> &'a Attribute;
    pub fn LLVMRustCreateElementTypeAttr<'a>(C: &'a Context, ty: &'a Type) -> &'a Attribute;
    pub fn LLVMRustCreateUWTableAttr(C: &Context, async_: bool) -> &Attribute;
    pub fn LLVMRustCreateAllocSizeAttr(C: &Context, size_arg: u32) -> &Attribute;
    pub fn LLVMRustCreateAllocKindAttr(C: &Context, size_arg: u64) -> &Attribute;
    pub fn LLVMRustCreateMemoryEffectsAttr(C: &Context, effects: MemoryEffects) -> &Attribute;
    pub fn LLVMRustCreateRangeAttribute(
        C: &Context,
        num_bits: c_uint,
        lower_words: *const u64,
        upper_words: *const u64,
    ) -> &Attribute;

    // Operations on functions
    pub fn LLVMRustGetOrInsertFunction<'a>(
        M: &'a Module,
        Name: *const c_char,
        NameLen: size_t,
        FunctionTy: &'a Type,
    ) -> &'a Value;
    pub fn LLVMRustAddFunctionAttributes<'a>(
        Fn: &'a Value,
        index: c_uint,
        Attrs: *const &'a Attribute,
        AttrsLen: size_t,
    );

    // Operations on call sites
    pub fn LLVMRustAddCallSiteAttributes<'a>(
        Instr: &'a Value,
        index: c_uint,
        Attrs: *const &'a Attribute,
        AttrsLen: size_t,
    );

    pub fn LLVMRustSetFastMath(Instr: &Value);
    pub fn LLVMRustSetAlgebraicMath(Instr: &Value);
    pub fn LLVMRustSetAllowReassoc(Instr: &Value);

    // Miscellaneous instructions
    pub fn LLVMRustBuildMemCpy<'a>(
        B: &Builder<'a>,
        Dst: &'a Value,
        DstAlign: c_uint,
        Src: &'a Value,
        SrcAlign: c_uint,
        Size: &'a Value,
        IsVolatile: bool,
    ) -> &'a Value;
    pub fn LLVMRustBuildMemMove<'a>(
        B: &Builder<'a>,
        Dst: &'a Value,
        DstAlign: c_uint,
        Src: &'a Value,
        SrcAlign: c_uint,
        Size: &'a Value,
        IsVolatile: bool,
    ) -> &'a Value;
    pub fn LLVMRustBuildMemSet<'a>(
        B: &Builder<'a>,
        Dst: &'a Value,
        DstAlign: c_uint,
        Val: &'a Value,
        Size: &'a Value,
        IsVolatile: bool,
    ) -> &'a Value;

    pub fn LLVMRustBuildVectorReduceFAdd<'a>(
        B: &Builder<'a>,
        Acc: &'a Value,
        Src: &'a Value,
    ) -> &'a Value;
    pub fn LLVMRustBuildVectorReduceFMul<'a>(
        B: &Builder<'a>,
        Acc: &'a Value,
        Src: &'a Value,
    ) -> &'a Value;
    pub fn LLVMRustBuildVectorReduceAdd<'a>(B: &Builder<'a>, Src: &'a Value) -> &'a Value;
    pub fn LLVMRustBuildVectorReduceMul<'a>(B: &Builder<'a>, Src: &'a Value) -> &'a Value;
    pub fn LLVMRustBuildVectorReduceAnd<'a>(B: &Builder<'a>, Src: &'a Value) -> &'a Value;
    pub fn LLVMRustBuildVectorReduceOr<'a>(B: &Builder<'a>, Src: &'a Value) -> &'a Value;
    pub fn LLVMRustBuildVectorReduceXor<'a>(B: &Builder<'a>, Src: &'a Value) -> &'a Value;
    pub fn LLVMRustBuildVectorReduceMin<'a>(
        B: &Builder<'a>,
        Src: &'a Value,
        IsSigned: bool,
    ) -> &'a Value;
    pub fn LLVMRustBuildVectorReduceMax<'a>(
        B: &Builder<'a>,
        Src: &'a Value,
        IsSigned: bool,
    ) -> &'a Value;
    pub fn LLVMRustBuildVectorReduceFMin<'a>(
        B: &Builder<'a>,
        Src: &'a Value,
        IsNaN: bool,
    ) -> &'a Value;
    pub fn LLVMRustBuildVectorReduceFMax<'a>(
        B: &Builder<'a>,
        Src: &'a Value,
        IsNaN: bool,
    ) -> &'a Value;

    pub fn LLVMRustBuildMinNum<'a>(B: &Builder<'a>, LHS: &'a Value, LHS: &'a Value) -> &'a Value;
    pub fn LLVMRustBuildMaxNum<'a>(B: &Builder<'a>, LHS: &'a Value, LHS: &'a Value) -> &'a Value;

    // Atomic Operations
    pub fn LLVMRustBuildAtomicLoad<'a>(
        B: &Builder<'a>,
        ElementType: &'a Type,
        PointerVal: &'a Value,
        Name: *const c_char,
        Order: AtomicOrdering,
    ) -> &'a Value;

    pub fn LLVMRustBuildAtomicStore<'a>(
        B: &Builder<'a>,
        Val: &'a Value,
        Ptr: &'a Value,
        Order: AtomicOrdering,
    ) -> &'a Value;

    pub fn LLVMRustTimeTraceProfilerInitialize();

    pub fn LLVMRustTimeTraceProfilerFinishThread();

    pub fn LLVMRustTimeTraceProfilerFinish(FileName: *const c_char);

    /// Returns a string describing the last error caused by an LLVMRust* call.
    pub fn LLVMRustGetLastError() -> *const c_char;

    /// Prints the timing information collected by `-Ztime-llvm-passes`.
    pub(crate) fn LLVMRustPrintPassTimings(OutStr: &RustString);

    /// Prints the statistics collected by `-Zprint-codegen-stats`.
    pub(crate) fn LLVMRustPrintStatistics(OutStr: &RustString);

    /// Prepares inline assembly.
    pub fn LLVMRustInlineAsm(
        Ty: &Type,
        AsmString: *const c_char,
        AsmStringLen: size_t,
        Constraints: *const c_char,
        ConstraintsLen: size_t,
        SideEffects: Bool,
        AlignStack: Bool,
        Dialect: AsmDialect,
        CanThrow: Bool,
    ) -> &Value;
    pub fn LLVMRustInlineAsmVerify(
        Ty: &Type,
        Constraints: *const c_char,
        ConstraintsLen: size_t,
    ) -> bool;

    pub(crate) fn LLVMRustCoverageWriteFilenamesToBuffer(
        Filenames: *const *const c_char,
        FilenamesLen: size_t,
        Lengths: *const size_t,
        LengthsLen: size_t,
        BufferOut: &RustString,
    );

    pub(crate) fn LLVMRustCoverageWriteFunctionMappingsToBuffer(
        VirtualFileMappingIDs: *const c_uint,
        NumVirtualFileMappingIDs: size_t,
        Expressions: *const crate::coverageinfo::ffi::CounterExpression,
        NumExpressions: size_t,
        CodeRegions: *const crate::coverageinfo::ffi::CodeRegion,
        NumCodeRegions: size_t,
        BranchRegions: *const crate::coverageinfo::ffi::BranchRegion,
        NumBranchRegions: size_t,
        MCDCBranchRegions: *const crate::coverageinfo::ffi::MCDCBranchRegion,
        NumMCDCBranchRegions: size_t,
        MCDCDecisionRegions: *const crate::coverageinfo::ffi::MCDCDecisionRegion,
        NumMCDCDecisionRegions: size_t,
        BufferOut: &RustString,
    );

    pub(crate) fn LLVMRustCoverageCreatePGOFuncNameVar(
        F: &Value,
        FuncName: *const c_char,
        FuncNameLen: size_t,
    ) -> &Value;
    pub(crate) fn LLVMRustCoverageHashBytes(Bytes: *const c_char, NumBytes: size_t) -> u64;

    pub(crate) fn LLVMRustCoverageWriteCovmapSectionNameToString(M: &Module, OutStr: &RustString);

    pub(crate) fn LLVMRustCoverageWriteCovfunSectionNameToString(M: &Module, OutStr: &RustString);

    pub(crate) fn LLVMRustCoverageWriteCovmapVarNameToString(OutStr: &RustString);

    pub(crate) fn LLVMRustCoverageMappingVersion() -> u32;
    pub fn LLVMRustDebugMetadataVersion() -> u32;
    pub fn LLVMRustVersionMajor() -> u32;
    pub fn LLVMRustVersionMinor() -> u32;
    pub fn LLVMRustVersionPatch() -> u32;

    /// Add LLVM module flags.
    ///
    /// In order for Rust-C LTO to work, module flags must be compatible with Clang. What
    /// "compatible" means depends on the merge behaviors involved.
    pub fn LLVMRustAddModuleFlagU32(
        M: &Module,
        MergeBehavior: ModuleFlagMergeBehavior,
        Name: *const c_char,
        NameLen: size_t,
        Value: u32,
    );

    pub fn LLVMRustAddModuleFlagString(
        M: &Module,
        MergeBehavior: ModuleFlagMergeBehavior,
        Name: *const c_char,
        NameLen: size_t,
        Value: *const c_char,
        ValueLen: size_t,
    );

    pub fn LLVMRustDIBuilderCreate(M: &Module) -> &mut DIBuilder<'_>;

    pub fn LLVMRustDIBuilderDispose<'a>(Builder: &'a mut DIBuilder<'a>);

    pub fn LLVMRustDIBuilderFinalize(Builder: &DIBuilder<'_>);

    pub fn LLVMRustDIBuilderCreateCompileUnit<'a>(
        Builder: &DIBuilder<'a>,
        Lang: c_uint,
        File: &'a DIFile,
        Producer: *const c_char,
        ProducerLen: size_t,
        isOptimized: bool,
        Flags: *const c_char,
        RuntimeVer: c_uint,
        SplitName: *const c_char,
        SplitNameLen: size_t,
        kind: DebugEmissionKind,
        DWOId: u64,
        SplitDebugInlining: bool,
        DebugNameTableKind: DebugNameTableKind,
    ) -> &'a DIDescriptor;

    pub fn LLVMRustDIBuilderCreateFile<'a>(
        Builder: &DIBuilder<'a>,
        Filename: *const c_char,
        FilenameLen: size_t,
        Directory: *const c_char,
        DirectoryLen: size_t,
        CSKind: ChecksumKind,
        Checksum: *const c_char,
        ChecksumLen: size_t,
        Source: *const c_char,
        SourceLen: size_t,
    ) -> &'a DIFile;

    pub fn LLVMRustDIBuilderCreateSubroutineType<'a>(
        Builder: &DIBuilder<'a>,
        ParameterTypes: &'a DIArray,
    ) -> &'a DICompositeType;

    pub fn LLVMRustDIBuilderCreateFunction<'a>(
        Builder: &DIBuilder<'a>,
        Scope: &'a DIDescriptor,
        Name: *const c_char,
        NameLen: size_t,
        LinkageName: *const c_char,
        LinkageNameLen: size_t,
        File: &'a DIFile,
        LineNo: c_uint,
        Ty: &'a DIType,
        ScopeLine: c_uint,
        Flags: DIFlags,
        SPFlags: DISPFlags,
        MaybeFn: Option<&'a Value>,
        TParam: &'a DIArray,
        Decl: Option<&'a DIDescriptor>,
    ) -> &'a DISubprogram;

    pub fn LLVMRustDIBuilderCreateMethod<'a>(
        Builder: &DIBuilder<'a>,
        Scope: &'a DIDescriptor,
        Name: *const c_char,
        NameLen: size_t,
        LinkageName: *const c_char,
        LinkageNameLen: size_t,
        File: &'a DIFile,
        LineNo: c_uint,
        Ty: &'a DIType,
        Flags: DIFlags,
        SPFlags: DISPFlags,
        TParam: &'a DIArray,
    ) -> &'a DISubprogram;

    pub fn LLVMRustDIBuilderCreateBasicType<'a>(
        Builder: &DIBuilder<'a>,
        Name: *const c_char,
        NameLen: size_t,
        SizeInBits: u64,
        Encoding: c_uint,
    ) -> &'a DIBasicType;

    pub fn LLVMRustDIBuilderCreateTypedef<'a>(
        Builder: &DIBuilder<'a>,
        Type: &'a DIBasicType,
        Name: *const c_char,
        NameLen: size_t,
        File: &'a DIFile,
        LineNo: c_uint,
        Scope: Option<&'a DIScope>,
    ) -> &'a DIDerivedType;

    pub fn LLVMRustDIBuilderCreatePointerType<'a>(
        Builder: &DIBuilder<'a>,
        PointeeTy: &'a DIType,
        SizeInBits: u64,
        AlignInBits: u32,
        AddressSpace: c_uint,
        Name: *const c_char,
        NameLen: size_t,
    ) -> &'a DIDerivedType;

    pub fn LLVMRustDIBuilderCreateStructType<'a>(
        Builder: &DIBuilder<'a>,
        Scope: Option<&'a DIDescriptor>,
        Name: *const c_char,
        NameLen: size_t,
        File: &'a DIFile,
        LineNumber: c_uint,
        SizeInBits: u64,
        AlignInBits: u32,
        Flags: DIFlags,
        DerivedFrom: Option<&'a DIType>,
        Elements: &'a DIArray,
        RunTimeLang: c_uint,
        VTableHolder: Option<&'a DIType>,
        UniqueId: *const c_char,
        UniqueIdLen: size_t,
    ) -> &'a DICompositeType;

    pub fn LLVMRustDIBuilderCreateMemberType<'a>(
        Builder: &DIBuilder<'a>,
        Scope: &'a DIDescriptor,
        Name: *const c_char,
        NameLen: size_t,
        File: &'a DIFile,
        LineNo: c_uint,
        SizeInBits: u64,
        AlignInBits: u32,
        OffsetInBits: u64,
        Flags: DIFlags,
        Ty: &'a DIType,
    ) -> &'a DIDerivedType;

    pub fn LLVMRustDIBuilderCreateVariantMemberType<'a>(
        Builder: &DIBuilder<'a>,
        Scope: &'a DIScope,
        Name: *const c_char,
        NameLen: size_t,
        File: &'a DIFile,
        LineNumber: c_uint,
        SizeInBits: u64,
        AlignInBits: u32,
        OffsetInBits: u64,
        Discriminant: Option<&'a Value>,
        Flags: DIFlags,
        Ty: &'a DIType,
    ) -> &'a DIType;

    pub fn LLVMRustDIBuilderCreateStaticMemberType<'a>(
        Builder: &DIBuilder<'a>,
        Scope: &'a DIDescriptor,
        Name: *const c_char,
        NameLen: size_t,
        File: &'a DIFile,
        LineNo: c_uint,
        Ty: &'a DIType,
        Flags: DIFlags,
        val: Option<&'a Value>,
        AlignInBits: u32,
    ) -> &'a DIDerivedType;

    pub fn LLVMRustDIBuilderCreateLexicalBlock<'a>(
        Builder: &DIBuilder<'a>,
        Scope: &'a DIScope,
        File: &'a DIFile,
        Line: c_uint,
        Col: c_uint,
    ) -> &'a DILexicalBlock;

    pub fn LLVMRustDIBuilderCreateLexicalBlockFile<'a>(
        Builder: &DIBuilder<'a>,
        Scope: &'a DIScope,
        File: &'a DIFile,
    ) -> &'a DILexicalBlock;

    pub fn LLVMRustDIBuilderCreateStaticVariable<'a>(
        Builder: &DIBuilder<'a>,
        Context: Option<&'a DIScope>,
        Name: *const c_char,
        NameLen: size_t,
        LinkageName: *const c_char,
        LinkageNameLen: size_t,
        File: &'a DIFile,
        LineNo: c_uint,
        Ty: &'a DIType,
        isLocalToUnit: bool,
        Val: &'a Value,
        Decl: Option<&'a DIDescriptor>,
        AlignInBits: u32,
    ) -> &'a DIGlobalVariableExpression;

    pub fn LLVMRustDIBuilderCreateVariable<'a>(
        Builder: &DIBuilder<'a>,
        Tag: c_uint,
        Scope: &'a DIDescriptor,
        Name: *const c_char,
        NameLen: size_t,
        File: &'a DIFile,
        LineNo: c_uint,
        Ty: &'a DIType,
        AlwaysPreserve: bool,
        Flags: DIFlags,
        ArgNo: c_uint,
        AlignInBits: u32,
    ) -> &'a DIVariable;

    pub fn LLVMRustDIBuilderCreateArrayType<'a>(
        Builder: &DIBuilder<'a>,
        Size: u64,
        AlignInBits: u32,
        Ty: &'a DIType,
        Subscripts: &'a DIArray,
    ) -> &'a DIType;

    pub fn LLVMRustDIBuilderGetOrCreateSubrange<'a>(
        Builder: &DIBuilder<'a>,
        Lo: i64,
        Count: i64,
    ) -> &'a DISubrange;

    pub fn LLVMRustDIBuilderGetOrCreateArray<'a>(
        Builder: &DIBuilder<'a>,
        Ptr: *const Option<&'a DIDescriptor>,
        Count: c_uint,
    ) -> &'a DIArray;

    pub fn LLVMRustDIBuilderInsertDeclareAtEnd<'a>(
        Builder: &DIBuilder<'a>,
        Val: &'a Value,
        VarInfo: &'a DIVariable,
        AddrOps: *const u64,
        AddrOpsCount: c_uint,
        DL: &'a DILocation,
        InsertAtEnd: &'a BasicBlock,
    );

    pub fn LLVMRustDIBuilderCreateEnumerator<'a>(
        Builder: &DIBuilder<'a>,
        Name: *const c_char,
        NameLen: size_t,
        Value: *const u64,
        SizeInBits: c_uint,
        IsUnsigned: bool,
    ) -> &'a DIEnumerator;

    pub fn LLVMRustDIBuilderCreateEnumerationType<'a>(
        Builder: &DIBuilder<'a>,
        Scope: &'a DIScope,
        Name: *const c_char,
        NameLen: size_t,
        File: &'a DIFile,
        LineNumber: c_uint,
        SizeInBits: u64,
        AlignInBits: u32,
        Elements: &'a DIArray,
        ClassType: &'a DIType,
        IsScoped: bool,
    ) -> &'a DIType;

    pub fn LLVMRustDIBuilderCreateUnionType<'a>(
        Builder: &DIBuilder<'a>,
        Scope: Option<&'a DIScope>,
        Name: *const c_char,
        NameLen: size_t,
        File: &'a DIFile,
        LineNumber: c_uint,
        SizeInBits: u64,
        AlignInBits: u32,
        Flags: DIFlags,
        Elements: Option<&'a DIArray>,
        RunTimeLang: c_uint,
        UniqueId: *const c_char,
        UniqueIdLen: size_t,
    ) -> &'a DIType;

    pub fn LLVMRustDIBuilderCreateVariantPart<'a>(
        Builder: &DIBuilder<'a>,
        Scope: &'a DIScope,
        Name: *const c_char,
        NameLen: size_t,
        File: &'a DIFile,
        LineNo: c_uint,
        SizeInBits: u64,
        AlignInBits: u32,
        Flags: DIFlags,
        Discriminator: Option<&'a DIDerivedType>,
        Elements: &'a DIArray,
        UniqueId: *const c_char,
        UniqueIdLen: size_t,
    ) -> &'a DIDerivedType;

    pub fn LLVMRustDIBuilderCreateTemplateTypeParameter<'a>(
        Builder: &DIBuilder<'a>,
        Scope: Option<&'a DIScope>,
        Name: *const c_char,
        NameLen: size_t,
        Ty: &'a DIType,
    ) -> &'a DITemplateTypeParameter;

    pub fn LLVMRustDIBuilderCreateNameSpace<'a>(
        Builder: &DIBuilder<'a>,
        Scope: Option<&'a DIScope>,
        Name: *const c_char,
        NameLen: size_t,
        ExportSymbols: bool,
    ) -> &'a DINameSpace;

    pub fn LLVMRustDICompositeTypeReplaceArrays<'a>(
        Builder: &DIBuilder<'a>,
        CompositeType: &'a DIType,
        Elements: Option<&'a DIArray>,
        Params: Option<&'a DIArray>,
    );

    pub fn LLVMRustDIBuilderCreateDebugLocation<'a>(
        Line: c_uint,
        Column: c_uint,
        Scope: &'a DIScope,
        InlinedAt: Option<&'a DILocation>,
    ) -> &'a DILocation;
    pub fn LLVMRustDILocationCloneWithBaseDiscriminator<'a>(
        Location: &'a DILocation,
        BD: c_uint,
    ) -> Option<&'a DILocation>;
    pub fn LLVMRustDIBuilderCreateOpDeref() -> u64;
    pub fn LLVMRustDIBuilderCreateOpPlusUconst() -> u64;
    pub fn LLVMRustDIBuilderCreateOpLLVMFragment() -> u64;

    pub fn LLVMRustWriteTypeToString(Type: &Type, s: &RustString);
    pub fn LLVMRustWriteValueToString(value_ref: &Value, s: &RustString);

    pub fn LLVMRustHasFeature(T: &TargetMachine, s: *const c_char) -> bool;

    pub(crate) fn LLVMRustPrintTargetCPUs(TM: &TargetMachine, OutStr: &RustString);
    pub fn LLVMRustGetTargetFeaturesCount(T: &TargetMachine) -> size_t;
    pub fn LLVMRustGetTargetFeature(
        T: &TargetMachine,
        Index: size_t,
        Feature: &mut *const c_char,
        Desc: &mut *const c_char,
    );

    pub fn LLVMRustGetHostCPUName(LenOut: &mut size_t) -> *const u8;

    // This function makes copies of pointed to data, so the data's lifetime may end after this
    // function returns.
    pub fn LLVMRustCreateTargetMachine(
        Triple: *const c_char,
        CPU: *const c_char,
        Features: *const c_char,
        Abi: *const c_char,
        Model: CodeModel,
        Reloc: RelocModel,
        Level: CodeGenOptLevel,
        UseSoftFP: bool,
        FunctionSections: bool,
        DataSections: bool,
        UniqueSectionNames: bool,
        TrapUnreachable: bool,
        Singlethread: bool,
        VerboseAsm: bool,
        EmitStackSizeSection: bool,
        RelaxELFRelocations: bool,
        UseInitArray: bool,
        SplitDwarfFile: *const c_char,
        OutputObjFile: *const c_char,
        DebugInfoCompression: *const c_char,
        UseEmulatedTls: bool,
        ArgsCstrBuff: *const c_char,
        ArgsCstrBuffLen: usize,
    ) -> *mut TargetMachine;

    pub fn LLVMRustDisposeTargetMachine(T: *mut TargetMachine);
    pub fn LLVMRustAddLibraryInfo<'a>(
        PM: &PassManager<'a>,
        M: &'a Module,
        DisableSimplifyLibCalls: bool,
    );
    pub fn LLVMRustWriteOutputFile<'a>(
        T: &'a TargetMachine,
        PM: &PassManager<'a>,
        M: &'a Module,
        Output: *const c_char,
        DwoOutput: *const c_char,
        FileType: FileType,
        VerifyIR: bool,
    ) -> LLVMRustResult;
    pub fn LLVMRustOptimize<'a>(
        M: &'a Module,
        TM: &'a TargetMachine,
        OptLevel: PassBuilderOptLevel,
        OptStage: OptStage,
        IsLinkerPluginLTO: bool,
        NoPrepopulatePasses: bool,
        VerifyIR: bool,
        LintIR: bool,
        UseThinLTOBuffers: bool,
        MergeFunctions: bool,
        UnrollLoops: bool,
        SLPVectorize: bool,
        LoopVectorize: bool,
        DisableSimplifyLibCalls: bool,
        EmitLifetimeMarkers: bool,
        SanitizerOptions: Option<&SanitizerOptions>,
        PGOGenPath: *const c_char,
        PGOUsePath: *const c_char,
        InstrumentCoverage: bool,
        InstrProfileOutput: *const c_char,
        PGOSampleUsePath: *const c_char,
        DebugInfoForProfiling: bool,
        llvm_selfprofiler: *mut c_void,
        begin_callback: SelfProfileBeforePassCallback,
        end_callback: SelfProfileAfterPassCallback,
        ExtraPasses: *const c_char,
        ExtraPassesLen: size_t,
        LLVMPlugins: *const c_char,
        LLVMPluginsLen: size_t,
    ) -> LLVMRustResult;
    pub fn LLVMRustPrintModule(
        M: &Module,
        Output: *const c_char,
        Demangle: extern "C" fn(*const c_char, size_t, *mut c_char, size_t) -> size_t,
    ) -> LLVMRustResult;
    pub fn LLVMRustSetLLVMOptions(Argc: c_int, Argv: *const *const c_char);
    pub fn LLVMRustPrintPasses();
    pub fn LLVMRustSetNormalizedTarget(M: &Module, triple: *const c_char);
    pub fn LLVMRustRunRestrictionPass(M: &Module, syms: *const *const c_char, len: size_t);

    pub fn LLVMRustOpenArchive(path: *const c_char) -> Option<&'static mut Archive>;
    pub fn LLVMRustArchiveIteratorNew(AR: &Archive) -> &mut ArchiveIterator<'_>;
    pub fn LLVMRustArchiveIteratorNext<'a>(
        AIR: &ArchiveIterator<'a>,
    ) -> Option<&'a mut ArchiveChild<'a>>;
    pub fn LLVMRustArchiveChildName(ACR: &ArchiveChild<'_>, size: &mut size_t) -> *const c_char;
    pub fn LLVMRustArchiveChildFree<'a>(ACR: &'a mut ArchiveChild<'a>);
    pub fn LLVMRustArchiveIteratorFree<'a>(AIR: &'a mut ArchiveIterator<'a>);
    pub fn LLVMRustDestroyArchive(AR: &'static mut Archive);

    pub fn LLVMRustWriteTwineToString(T: &Twine, s: &RustString);

    pub fn LLVMRustUnpackOptimizationDiagnostic<'a>(
        DI: &'a DiagnosticInfo,
        pass_name_out: &RustString,
        function_out: &mut Option<&'a Value>,
        loc_line_out: &mut c_uint,
        loc_column_out: &mut c_uint,
        loc_filename_out: &RustString,
        message_out: &RustString,
    );

    pub fn LLVMRustUnpackInlineAsmDiagnostic<'a>(
        DI: &'a DiagnosticInfo,
        level_out: &mut DiagnosticLevel,
        cookie_out: &mut u64,
        message_out: &mut Option<&'a Twine>,
    );

    pub fn LLVMRustWriteDiagnosticInfoToString(DI: &DiagnosticInfo, s: &RustString);
    pub fn LLVMRustGetDiagInfoKind(DI: &DiagnosticInfo) -> DiagnosticKind;

    pub fn LLVMRustGetSMDiagnostic<'a>(
        DI: &'a DiagnosticInfo,
        cookie_out: &mut c_uint,
    ) -> &'a SMDiagnostic;

    pub fn LLVMRustUnpackSMDiagnostic(
        d: &SMDiagnostic,
        message_out: &RustString,
        buffer_out: &RustString,
        level_out: &mut DiagnosticLevel,
        loc_out: &mut c_uint,
        ranges_out: *mut c_uint,
        num_ranges: &mut usize,
    ) -> bool;

    pub fn LLVMRustWriteArchive(
        Dst: *const c_char,
        NumMembers: size_t,
        Members: *const &RustArchiveMember<'_>,
        WriteSymbtab: bool,
        Kind: ArchiveKind,
        isEC: bool,
    ) -> LLVMRustResult;
    pub fn LLVMRustArchiveMemberNew<'a>(
        Filename: *const c_char,
        Name: *const c_char,
        Child: Option<&ArchiveChild<'a>>,
    ) -> &'a mut RustArchiveMember<'a>;
    pub fn LLVMRustArchiveMemberFree<'a>(Member: &'a mut RustArchiveMember<'a>);

    pub fn LLVMRustWriteImportLibrary(
        ImportName: *const c_char,
        Path: *const c_char,
        Exports: *const LLVMRustCOFFShortExport,
        NumExports: usize,
        Machine: u16,
        MinGW: bool,
    ) -> LLVMRustResult;

    pub fn LLVMRustSetDataLayoutFromTargetMachine<'a>(M: &'a Module, TM: &'a TargetMachine);

    pub fn LLVMRustPositionBuilderAtStart<'a>(B: &Builder<'a>, BB: &'a BasicBlock);

    pub fn LLVMRustSetModulePICLevel(M: &Module);
    pub fn LLVMRustSetModulePIELevel(M: &Module);
    pub fn LLVMRustSetModuleCodeModel(M: &Module, Model: CodeModel);
    pub fn LLVMRustModuleBufferCreate(M: &Module) -> &'static mut ModuleBuffer;
    pub fn LLVMRustModuleBufferPtr(p: &ModuleBuffer) -> *const u8;
    pub fn LLVMRustModuleBufferLen(p: &ModuleBuffer) -> usize;
    pub fn LLVMRustModuleBufferFree(p: &'static mut ModuleBuffer);
    pub fn LLVMRustModuleCost(M: &Module) -> u64;
    pub fn LLVMRustModuleInstructionStats(M: &Module, Str: &RustString);

    pub fn LLVMRustThinLTOBufferCreate(
        M: &Module,
        is_thin: bool,
        emit_summary: bool,
    ) -> &'static mut ThinLTOBuffer;
    pub fn LLVMRustThinLTOBufferFree(M: &'static mut ThinLTOBuffer);
    pub fn LLVMRustThinLTOBufferPtr(M: &ThinLTOBuffer) -> *const c_char;
    pub fn LLVMRustThinLTOBufferLen(M: &ThinLTOBuffer) -> size_t;
    pub fn LLVMRustThinLTOBufferThinLinkDataPtr(M: &ThinLTOBuffer) -> *const c_char;
    pub fn LLVMRustThinLTOBufferThinLinkDataLen(M: &ThinLTOBuffer) -> size_t;
    pub fn LLVMRustCreateThinLTOData(
        Modules: *const ThinLTOModule,
        NumModules: size_t,
        PreservedSymbols: *const *const c_char,
        PreservedSymbolsLen: size_t,
    ) -> Option<&'static mut ThinLTOData>;
    pub fn LLVMRustPrepareThinLTORename(
        Data: &ThinLTOData,
        Module: &Module,
        Target: &TargetMachine,
    ) -> bool;
    pub fn LLVMRustPrepareThinLTOResolveWeak(Data: &ThinLTOData, Module: &Module) -> bool;
    pub fn LLVMRustPrepareThinLTOInternalize(Data: &ThinLTOData, Module: &Module) -> bool;
    pub fn LLVMRustPrepareThinLTOImport(
        Data: &ThinLTOData,
        Module: &Module,
        Target: &TargetMachine,
    ) -> bool;
    pub fn LLVMRustFreeThinLTOData(Data: &'static mut ThinLTOData);
    pub fn LLVMRustParseBitcodeForLTO(
        Context: &Context,
        Data: *const u8,
        len: usize,
        Identifier: *const c_char,
    ) -> Option<&Module>;
    pub fn LLVMRustGetSliceFromObjectDataByName(
        data: *const u8,
        len: usize,
        name: *const u8,
        name_len: usize,
        out_len: &mut usize,
    ) -> *const u8;

    pub fn LLVMRustLinkerNew(M: &Module) -> &mut Linker<'_>;
    pub fn LLVMRustLinkerAdd(
        linker: &Linker<'_>,
        bytecode: *const c_char,
        bytecode_len: usize,
    ) -> bool;
    pub fn LLVMRustLinkerFree<'a>(linker: &'a mut Linker<'a>);
    pub fn LLVMRustComputeLTOCacheKey(
        key_out: &RustString,
        mod_id: *const c_char,
        data: &ThinLTOData,
    );

    pub fn LLVMRustContextGetDiagnosticHandler(Context: &Context) -> Option<&DiagnosticHandler>;
    pub fn LLVMRustContextSetDiagnosticHandler(
        context: &Context,
        diagnostic_handler: Option<&DiagnosticHandler>,
    );
    pub fn LLVMRustContextConfigureDiagnosticHandler(
        context: &Context,
        diagnostic_handler_callback: DiagnosticHandlerTy,
        diagnostic_handler_context: *mut c_void,
        remark_all_passes: bool,
        remark_passes: *const *const c_char,
        remark_passes_len: usize,
        remark_file: *const c_char,
        pgo_available: bool,
    );

    pub fn LLVMRustGetMangledName(V: &Value, out: &RustString);

    pub fn LLVMRustGetElementTypeArgIndex(CallSite: &Value) -> i32;

    pub fn LLVMRustIsBitcode(ptr: *const u8, len: usize) -> bool;

    pub fn LLVMRustLLVMHasZlibCompressionForDebugSymbols() -> bool;

    pub fn LLVMRustLLVMHasZstdCompressionForDebugSymbols() -> bool;

    pub fn LLVMRustGetSymbols(
        buf_ptr: *const u8,
        buf_len: usize,
        state: *mut c_void,
        callback: GetSymbolsCallback,
        error_callback: GetSymbolsErrorCallback,
    ) -> *mut c_void;

    pub fn LLVMRustIs64BitSymbolicFile(buf_ptr: *const u8, buf_len: usize) -> bool;

    pub fn LLVMRustIsECObject(buf_ptr: *const u8, buf_len: usize) -> bool;

    pub fn LLVMRustSetNoSanitizeAddress(Global: &Value);
    pub fn LLVMRustSetNoSanitizeHWAddress(Global: &Value);
}
