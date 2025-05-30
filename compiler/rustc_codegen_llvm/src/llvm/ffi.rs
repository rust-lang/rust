//! Bindings to the LLVM-C API (`LLVM*`), and to our own `extern "C"` wrapper
//! functions around the unstable LLVM C++ API (`LLVMRust*`).
//!
//! ## Passing pointer/length strings as `*const c_uchar` (PTR_LEN_STR)
//!
//! Normally it's a good idea for Rust-side bindings to match the corresponding
//! C-side function declarations as closely as possible. But when passing `&str`
//! or `&[u8]` data as a pointer/length pair, it's more convenient to declare
//! the Rust-side pointer as `*const c_uchar` instead of `*const c_char`.
//! Both pointer types have the same ABI, and using `*const c_uchar` avoids
//! the need for an extra cast from `*const u8` on the Rust side.

#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]

use std::fmt::Debug;
use std::marker::PhantomData;
use std::num::NonZero;
use std::ptr;

use bitflags::bitflags;
use libc::{c_char, c_int, c_uchar, c_uint, c_ulonglong, c_void, size_t};
use rustc_macros::TryFromU32;
use rustc_target::spec::SymbolVisibility;

use super::RustString;
use super::debuginfo::{
    DIArray, DIBasicType, DIBuilder, DICompositeType, DIDerivedType, DIDescriptor, DIEnumerator,
    DIFile, DIFlags, DIGlobalVariableExpression, DILocation, DISPFlags, DIScope, DISubprogram,
    DISubrange, DITemplateTypeParameter, DIType, DIVariable, DebugEmissionKind, DebugNameTableKind,
};
use crate::llvm;

/// In the LLVM-C API, boolean values are passed as `typedef int LLVMBool`,
/// which has a different ABI from Rust or C++ `bool`.
pub(crate) type Bool = c_int;

pub(crate) const True: Bool = 1 as Bool;
pub(crate) const False: Bool = 0 as Bool;

/// Wrapper for a raw enum value returned from LLVM's C APIs.
///
/// For C enums returned by LLVM, it's risky to use a Rust enum as the return
/// type, because it would be UB if a later version of LLVM adds a new enum
/// value and returns it. Instead, return this raw wrapper, then convert to the
/// Rust-side enum explicitly.
#[repr(transparent)]
pub(crate) struct RawEnum<T> {
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
pub(crate) enum LLVMRustResult {
    Success,
    Failure,
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
pub(crate) enum ModuleFlagMergeBehavior {
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
#[derive(Copy, Clone, PartialEq, Debug, TryFromU32)]
#[repr(C)]
pub(crate) enum CallConv {
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
    AmdgpuKernel = 91,
}

/// Must match the layout of `LLVMLinkage`.
#[derive(Copy, Clone, PartialEq, TryFromU32)]
#[repr(C)]
pub(crate) enum Linkage {
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
pub(crate) enum Visibility {
    Default = 0,
    Hidden = 1,
    Protected = 2,
}

impl Visibility {
    pub(crate) fn from_generic(visibility: SymbolVisibility) -> Self {
        match visibility {
            SymbolVisibility::Hidden => Visibility::Hidden,
            SymbolVisibility::Protected => Visibility::Protected,
            SymbolVisibility::Interposable => Visibility::Default,
        }
    }
}

/// LLVMUnnamedAddr
#[repr(C)]
pub(crate) enum UnnamedAddr {
    No,
    #[expect(dead_code)]
    Local,
    Global,
}

/// LLVMDLLStorageClass
#[derive(Copy, Clone)]
#[repr(C)]
pub(crate) enum DLLStorageClass {
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
#[expect(dead_code, reason = "Some variants are unused, but are kept to match the C++")]
pub(crate) enum AttributeKind {
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
pub(crate) enum IntPredicate {
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
    pub(crate) fn from_generic(intpre: rustc_codegen_ssa::common::IntPredicate) -> Self {
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
pub(crate) enum RealPredicate {
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
    pub(crate) fn from_generic(realp: rustc_codegen_ssa::common::RealPredicate) -> Self {
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
#[expect(dead_code, reason = "Some variants are unused, but are kept to match LLVM-C")]
pub(crate) enum TypeKind {
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
    pub(crate) fn to_generic(self) -> rustc_codegen_ssa::common::TypeKind {
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
pub(crate) enum AtomicRmwBinOp {
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
    pub(crate) fn from_generic(op: rustc_codegen_ssa::common::AtomicRmwBinOp) -> Self {
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
pub(crate) enum AtomicOrdering {
    #[allow(dead_code)]
    NotAtomic = 0,
    #[allow(dead_code)]
    Unordered = 1,
    Monotonic = 2,
    // Consume = 3,  // Not specified yet.
    Acquire = 4,
    Release = 5,
    AcquireRelease = 6,
    SequentiallyConsistent = 7,
}

impl AtomicOrdering {
    pub(crate) fn from_generic(ao: rustc_middle::ty::AtomicOrdering) -> Self {
        use rustc_middle::ty::AtomicOrdering as Common;
        match ao {
            Common::Relaxed => Self::Monotonic,
            Common::Acquire => Self::Acquire,
            Common::Release => Self::Release,
            Common::AcqRel => Self::AcquireRelease,
            Common::SeqCst => Self::SequentiallyConsistent,
        }
    }
}

/// LLVMRustFileType
#[derive(Copy, Clone)]
#[repr(C)]
pub(crate) enum FileType {
    AssemblyFile,
    ObjectFile,
}

/// LLVMMetadataType
#[derive(Copy, Clone)]
#[repr(C)]
#[expect(dead_code, reason = "Some variants are unused, but are kept to match LLVM-C")]
pub(crate) enum MetadataType {
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

/// Must match the layout of `LLVMInlineAsmDialect`.
#[derive(Copy, Clone, PartialEq)]
#[repr(C)]
pub(crate) enum AsmDialect {
    Att,
    Intel,
}

/// LLVMRustCodeGenOptLevel
#[derive(Copy, Clone, PartialEq)]
#[repr(C)]
pub(crate) enum CodeGenOptLevel {
    None,
    Less,
    Default,
    Aggressive,
}

/// LLVMRustPassBuilderOptLevel
#[repr(C)]
pub(crate) enum PassBuilderOptLevel {
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
pub(crate) enum OptStage {
    PreLinkNoLTO,
    PreLinkThinLTO,
    PreLinkFatLTO,
    ThinLTO,
    FatLTO,
}

/// LLVMRustSanitizerOptions
#[repr(C)]
pub(crate) struct SanitizerOptions {
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

/// LLVMRustRelocModel
#[derive(Copy, Clone, PartialEq)]
#[repr(C)]
pub(crate) enum RelocModel {
    Static,
    PIC,
    DynamicNoPic,
    ROPI,
    RWPI,
    ROPI_RWPI,
}

/// LLVMRustFloatABI
#[derive(Copy, Clone, PartialEq)]
#[repr(C)]
pub(crate) enum FloatAbi {
    Default,
    Soft,
    Hard,
}

/// LLVMRustCodeModel
#[derive(Copy, Clone)]
#[repr(C)]
pub(crate) enum CodeModel {
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
pub(crate) enum DiagnosticKind {
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
pub(crate) enum DiagnosticLevel {
    Error,
    Warning,
    Note,
    Remark,
}

/// LLVMRustArchiveKind
#[derive(Copy, Clone)]
#[repr(C)]
pub(crate) enum ArchiveKind {
    K_GNU,
    K_BSD,
    K_DARWIN,
    K_COFF,
    K_AIXBIG,
}

unsafe extern "C" {
    // LLVMRustThinLTOData
    pub(crate) type ThinLTOData;

    // LLVMRustThinLTOBuffer
    pub(crate) type ThinLTOBuffer;
}

/// LLVMRustThinLTOModule
#[repr(C)]
pub(crate) struct ThinLTOModule {
    pub identifier: *const c_char,
    pub data: *const u8,
    pub len: usize,
}

/// LLVMThreadLocalMode
#[derive(Copy, Clone)]
#[repr(C)]
pub(crate) enum ThreadLocalMode {
    #[expect(dead_code)]
    NotThreadLocal,
    GeneralDynamic,
    LocalDynamic,
    InitialExec,
    LocalExec,
}

/// LLVMRustChecksumKind
#[derive(Copy, Clone)]
#[repr(C)]
pub(crate) enum ChecksumKind {
    None,
    MD5,
    SHA1,
    SHA256,
}

/// LLVMRustMemoryEffects
#[derive(Copy, Clone)]
#[repr(C)]
pub(crate) enum MemoryEffects {
    None,
    ReadOnly,
    InaccessibleMemOnly,
}

/// LLVMOpcode
#[derive(Copy, Clone, PartialEq, Eq)]
#[repr(C)]
#[expect(dead_code, reason = "Some variants are unused, but are kept to match LLVM-C")]
pub(crate) enum Opcode {
    Ret = 1,
    Br = 2,
    Switch = 3,
    IndirectBr = 4,
    Invoke = 5,
    Unreachable = 7,
    CallBr = 67,
    FNeg = 66,
    Add = 8,
    FAdd = 9,
    Sub = 10,
    FSub = 11,
    Mul = 12,
    FMul = 13,
    UDiv = 14,
    SDiv = 15,
    FDiv = 16,
    URem = 17,
    SRem = 18,
    FRem = 19,
    Shl = 20,
    LShr = 21,
    AShr = 22,
    And = 23,
    Or = 24,
    Xor = 25,
    Alloca = 26,
    Load = 27,
    Store = 28,
    GetElementPtr = 29,
    Trunc = 30,
    ZExt = 31,
    SExt = 32,
    FPToUI = 33,
    FPToSI = 34,
    UIToFP = 35,
    SIToFP = 36,
    FPTrunc = 37,
    FPExt = 38,
    PtrToInt = 39,
    IntToPtr = 40,
    BitCast = 41,
    AddrSpaceCast = 60,
    ICmp = 42,
    FCmp = 43,
    PHI = 44,
    Call = 45,
    Select = 46,
    UserOp1 = 47,
    UserOp2 = 48,
    VAArg = 49,
    ExtractElement = 50,
    InsertElement = 51,
    ShuffleVector = 52,
    ExtractValue = 53,
    InsertValue = 54,
    Freeze = 68,
    Fence = 55,
    AtomicCmpXchg = 56,
    AtomicRMW = 57,
    Resume = 58,
    LandingPad = 59,
    CleanupRet = 61,
    CatchRet = 62,
    CatchPad = 63,
    CleanupPad = 64,
    CatchSwitch = 65,
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
    pub(crate) type Module;
    pub(crate) type Context;
    pub(crate) type Type;
    pub(crate) type Value;
    pub(crate) type ConstantInt;
    pub(crate) type Attribute;
    pub(crate) type Metadata;
    pub(crate) type BasicBlock;
    pub(crate) type Comdat;
}
#[repr(C)]
pub(crate) struct Builder<'a>(InvariantOpaque<'a>);
#[repr(C)]
pub(crate) struct PassManager<'a>(InvariantOpaque<'a>);
unsafe extern "C" {
    pub type TargetMachine;
    pub(crate) type Archive;
}
#[repr(C)]
pub(crate) struct ArchiveIterator<'a>(InvariantOpaque<'a>);
#[repr(C)]
pub(crate) struct ArchiveChild<'a>(InvariantOpaque<'a>);
unsafe extern "C" {
    pub(crate) type Twine;
    pub(crate) type DiagnosticInfo;
    pub(crate) type SMDiagnostic;
}
#[repr(C)]
pub(crate) struct RustArchiveMember<'a>(InvariantOpaque<'a>);
/// Opaque pointee of `LLVMOperandBundleRef`.
#[repr(C)]
pub(crate) struct OperandBundle<'a>(InvariantOpaque<'a>);
#[repr(C)]
pub(crate) struct Linker<'a>(InvariantOpaque<'a>);

unsafe extern "C" {
    pub(crate) type DiagnosticHandler;
}

pub(crate) type DiagnosticHandlerTy = unsafe extern "C" fn(&DiagnosticInfo, *mut c_void);

pub(crate) mod debuginfo {
    use std::ptr;

    use bitflags::bitflags;

    use super::{InvariantOpaque, Metadata};
    use crate::llvm::{self, Module};

    /// Opaque target type for references to an LLVM debuginfo builder.
    ///
    /// `&'_ DIBuilder<'ll>` corresponds to `LLVMDIBuilderRef`, which is the
    /// LLVM-C wrapper for `DIBuilder *`.
    ///
    /// Debuginfo builders are created and destroyed during codegen, so the
    /// builder reference typically has a shorter lifetime than the LLVM
    /// session (`'ll`) that it participates in.
    #[repr(C)]
    pub(crate) struct DIBuilder<'ll>(InvariantOpaque<'ll>);

    /// Owning pointer to a `DIBuilder<'ll>` that will dispose of the builder
    /// when dropped. Use `.as_ref()` to get the underlying `&DIBuilder`
    /// needed for debuginfo FFI calls.
    pub(crate) struct DIBuilderBox<'ll> {
        raw: ptr::NonNull<DIBuilder<'ll>>,
    }

    impl<'ll> DIBuilderBox<'ll> {
        pub(crate) fn new(llmod: &'ll Module) -> Self {
            let raw = unsafe { llvm::LLVMCreateDIBuilder(llmod) };
            let raw = ptr::NonNull::new(raw).unwrap();
            Self { raw }
        }

        pub(crate) fn as_ref(&self) -> &DIBuilder<'ll> {
            // SAFETY: This is an owning pointer, so `&DIBuilder` is valid
            // for as long as `&self` is.
            unsafe { self.raw.as_ref() }
        }
    }

    impl<'ll> Drop for DIBuilderBox<'ll> {
        fn drop(&mut self) {
            unsafe { llvm::LLVMDisposeDIBuilder(self.raw) };
        }
    }

    pub(crate) type DIDescriptor = Metadata;
    pub(crate) type DILocation = Metadata;
    pub(crate) type DIScope = DIDescriptor;
    pub(crate) type DIFile = DIScope;
    pub(crate) type DILexicalBlock = DIScope;
    pub(crate) type DISubprogram = DIScope;
    pub(crate) type DIType = DIDescriptor;
    pub(crate) type DIBasicType = DIType;
    pub(crate) type DIDerivedType = DIType;
    pub(crate) type DICompositeType = DIDerivedType;
    pub(crate) type DIVariable = DIDescriptor;
    pub(crate) type DIGlobalVariableExpression = DIDescriptor;
    pub(crate) type DIArray = DIDescriptor;
    pub(crate) type DISubrange = DIDescriptor;
    pub(crate) type DIEnumerator = DIDescriptor;
    pub(crate) type DITemplateTypeParameter = DIDescriptor;

    bitflags! {
        /// Must match the layout of `LLVMDIFlags` in the LLVM-C API.
        ///
        /// Each value declared here must also be covered by the static
        /// assertions in `RustWrapper.cpp` used by `fromRust(LLVMDIFlags)`.
        #[repr(transparent)]
        #[derive(Clone, Copy, Default)]
        pub(crate) struct DIFlags: u32 {
            const FlagZero                = 0;
            const FlagPrivate             = 1;
            const FlagProtected           = 2;
            const FlagPublic              = 3;
            const FlagFwdDecl             = (1 << 2);
            const FlagAppleBlock          = (1 << 3);
            const FlagReservedBit4        = (1 << 4);
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
            const FlagReserved            = (1 << 15);
            const FlagSingleInheritance   = (1 << 16);
            const FlagMultipleInheritance = (2 << 16);
            const FlagVirtualInheritance  = (3 << 16);
            const FlagIntroducedVirtual   = (1 << 18);
            const FlagBitField            = (1 << 19);
            const FlagNoReturn            = (1 << 20);
            // The bit at (1 << 21) is unused, but was `LLVMDIFlagMainSubprogram`.
            const FlagTypePassByValue     = (1 << 22);
            const FlagTypePassByReference = (1 << 23);
            const FlagEnumClass           = (1 << 24);
            const FlagThunk               = (1 << 25);
            const FlagNonTrivial          = (1 << 26);
            const FlagBigEndian           = (1 << 27);
            const FlagLittleEndian        = (1 << 28);
        }
    }

    // These values **must** match with LLVMRustDISPFlags!!
    bitflags! {
        #[repr(transparent)]
        #[derive(Clone, Copy, Default)]
        pub(crate) struct DISPFlags: u32 {
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
    pub(crate) enum DebugEmissionKind {
        NoDebug,
        FullDebug,
        LineTablesOnly,
        DebugDirectivesOnly,
    }

    impl DebugEmissionKind {
        pub(crate) fn from_generic(kind: rustc_session::config::DebugInfo) -> Self {
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
    pub(crate) enum DebugNameTableKind {
        Default,
        #[expect(dead_code)]
        Gnu,
        None,
    }
}

// These values **must** match with LLVMRustAllocKindFlags
bitflags! {
    #[repr(transparent)]
    #[derive(Default)]
    pub(crate) struct AllocKindFlags : u64 {
        const Unknown = 0;
        const Alloc = 1;
        const Realloc = 1 << 1;
        const Free = 1 << 2;
        const Uninitialized = 1 << 3;
        const Zeroed = 1 << 4;
        const Aligned = 1 << 5;
    }
}

// These values **must** match with LLVMGEPNoWrapFlags
bitflags! {
    #[repr(transparent)]
    #[derive(Default)]
    pub struct GEPNoWrapFlags : c_uint {
        const InBounds = 1 << 0;
        const NUSW = 1 << 1;
        const NUW = 1 << 2;
    }
}

unsafe extern "C" {
    pub(crate) type ModuleBuffer;
}

pub(crate) type SelfProfileBeforePassCallback =
    unsafe extern "C" fn(*mut c_void, *const c_char, *const c_char);
pub(crate) type SelfProfileAfterPassCallback = unsafe extern "C" fn(*mut c_void);

pub(crate) type GetSymbolsCallback =
    unsafe extern "C" fn(*mut c_void, *const c_char) -> *mut c_void;
pub(crate) type GetSymbolsErrorCallback = unsafe extern "C" fn(*const c_char) -> *mut c_void;

#[derive(Copy, Clone)]
#[repr(transparent)]
pub(crate) struct MetadataKindId(c_uint);

impl From<MetadataType> for MetadataKindId {
    fn from(value: MetadataType) -> Self {
        Self(value as c_uint)
    }
}

unsafe extern "C" {
    // Create and destroy contexts.
    pub(crate) fn LLVMContextDispose(C: &'static mut Context);
    pub(crate) fn LLVMGetMDKindIDInContext(
        C: &Context,
        Name: *const c_char,
        SLen: c_uint,
    ) -> MetadataKindId;

    // Create modules.
    pub(crate) fn LLVMModuleCreateWithNameInContext(
        ModuleID: *const c_char,
        C: &Context,
    ) -> &Module;
    pub(crate) fn LLVMCloneModule(M: &Module) -> &Module;

    /// Data layout. See Module::getDataLayout.
    pub(crate) fn LLVMGetDataLayoutStr(M: &Module) -> *const c_char;
    pub(crate) fn LLVMSetDataLayout(M: &Module, Triple: *const c_char);

    /// Append inline assembly to a module. See `Module::appendModuleInlineAsm`.
    pub(crate) fn LLVMAppendModuleInlineAsm(
        M: &Module,
        Asm: *const c_uchar, // See "PTR_LEN_STR".
        Len: size_t,
    );

    /// Create the specified uniqued inline asm string. See `InlineAsm::get()`.
    pub(crate) fn LLVMGetInlineAsm<'ll>(
        Ty: &'ll Type,
        AsmString: *const c_uchar, // See "PTR_LEN_STR".
        AsmStringSize: size_t,
        Constraints: *const c_uchar, // See "PTR_LEN_STR".
        ConstraintsSize: size_t,
        HasSideEffects: llvm::Bool,
        IsAlignStack: llvm::Bool,
        Dialect: AsmDialect,
        CanThrow: llvm::Bool,
    ) -> &'ll Value;

    // Operations on integer types
    pub(crate) fn LLVMInt1TypeInContext(C: &Context) -> &Type;
    pub(crate) fn LLVMInt8TypeInContext(C: &Context) -> &Type;
    pub(crate) fn LLVMInt16TypeInContext(C: &Context) -> &Type;
    pub(crate) fn LLVMInt32TypeInContext(C: &Context) -> &Type;
    pub(crate) fn LLVMInt64TypeInContext(C: &Context) -> &Type;
    pub(crate) fn LLVMIntTypeInContext(C: &Context, NumBits: c_uint) -> &Type;

    pub(crate) fn LLVMGetIntTypeWidth(IntegerTy: &Type) -> c_uint;

    // Operations on real types
    pub(crate) fn LLVMHalfTypeInContext(C: &Context) -> &Type;
    pub(crate) fn LLVMFloatTypeInContext(C: &Context) -> &Type;
    pub(crate) fn LLVMDoubleTypeInContext(C: &Context) -> &Type;
    pub(crate) fn LLVMFP128TypeInContext(C: &Context) -> &Type;

    // Operations on function types
    pub(crate) fn LLVMFunctionType<'a>(
        ReturnType: &'a Type,
        ParamTypes: *const &'a Type,
        ParamCount: c_uint,
        IsVarArg: Bool,
    ) -> &'a Type;
    pub(crate) fn LLVMCountParamTypes(FunctionTy: &Type) -> c_uint;
    pub(crate) fn LLVMGetParamTypes<'a>(FunctionTy: &'a Type, Dest: *mut &'a Type);

    // Operations on struct types
    pub(crate) fn LLVMStructTypeInContext<'a>(
        C: &'a Context,
        ElementTypes: *const &'a Type,
        ElementCount: c_uint,
        Packed: Bool,
    ) -> &'a Type;

    // Operations on array, pointer, and vector types (sequence types)
    pub(crate) fn LLVMPointerTypeInContext(C: &Context, AddressSpace: c_uint) -> &Type;
    pub(crate) fn LLVMVectorType(ElementType: &Type, ElementCount: c_uint) -> &Type;

    pub(crate) fn LLVMGetElementType(Ty: &Type) -> &Type;
    pub(crate) fn LLVMGetVectorSize(VectorTy: &Type) -> c_uint;

    // Operations on other types
    pub(crate) fn LLVMVoidTypeInContext(C: &Context) -> &Type;
    pub(crate) fn LLVMTokenTypeInContext(C: &Context) -> &Type;
    pub(crate) fn LLVMMetadataTypeInContext(C: &Context) -> &Type;

    // Operations on all values
    pub(crate) fn LLVMTypeOf(Val: &Value) -> &Type;
    pub(crate) fn LLVMGetValueName2(Val: &Value, Length: *mut size_t) -> *const c_char;
    pub(crate) fn LLVMSetValueName2(Val: &Value, Name: *const c_char, NameLen: size_t);
    pub(crate) fn LLVMReplaceAllUsesWith<'a>(OldVal: &'a Value, NewVal: &'a Value);
    pub(crate) safe fn LLVMSetMetadata<'a>(Val: &'a Value, KindID: MetadataKindId, Node: &'a Value);
    pub(crate) fn LLVMGlobalSetMetadata<'a>(Val: &'a Value, KindID: c_uint, Metadata: &'a Metadata);
    pub(crate) safe fn LLVMValueAsMetadata(Node: &Value) -> &Metadata;

    // Operations on constants of any type
    pub(crate) fn LLVMConstNull(Ty: &Type) -> &Value;
    pub(crate) fn LLVMGetUndef(Ty: &Type) -> &Value;
    pub(crate) fn LLVMGetPoison(Ty: &Type) -> &Value;

    // Operations on metadata
    pub(crate) fn LLVMMDStringInContext2(
        C: &Context,
        Str: *const c_char,
        SLen: size_t,
    ) -> &Metadata;
    pub(crate) fn LLVMMDNodeInContext2<'a>(
        C: &'a Context,
        Vals: *const &'a Metadata,
        Count: size_t,
    ) -> &'a Metadata;
    pub(crate) fn LLVMAddNamedMetadataOperand<'a>(
        M: &'a Module,
        Name: *const c_char,
        Val: &'a Value,
    );

    // Operations on scalar constants
    pub(crate) fn LLVMConstInt(IntTy: &Type, N: c_ulonglong, SignExtend: Bool) -> &Value;
    pub(crate) fn LLVMConstIntOfArbitraryPrecision(
        IntTy: &Type,
        Wn: c_uint,
        Ws: *const u64,
    ) -> &Value;
    pub(crate) fn LLVMConstReal(RealTy: &Type, N: f64) -> &Value;

    // Operations on composite constants
    pub(crate) fn LLVMConstArray2<'a>(
        ElementTy: &'a Type,
        ConstantVals: *const &'a Value,
        Length: u64,
    ) -> &'a Value;
    pub(crate) fn LLVMArrayType2(ElementType: &Type, ElementCount: u64) -> &Type;
    pub(crate) fn LLVMConstStringInContext2(
        C: &Context,
        Str: *const c_char,
        Length: size_t,
        DontNullTerminate: Bool,
    ) -> &Value;
    pub(crate) fn LLVMConstStructInContext<'a>(
        C: &'a Context,
        ConstantVals: *const &'a Value,
        Count: c_uint,
        Packed: Bool,
    ) -> &'a Value;
    pub(crate) fn LLVMConstVector(ScalarConstantVals: *const &Value, Size: c_uint) -> &Value;

    // Constant expressions
    pub(crate) fn LLVMConstInBoundsGEP2<'a>(
        ty: &'a Type,
        ConstantVal: &'a Value,
        ConstantIndices: *const &'a Value,
        NumIndices: c_uint,
    ) -> &'a Value;
    pub(crate) fn LLVMConstPtrToInt<'a>(ConstantVal: &'a Value, ToType: &'a Type) -> &'a Value;
    pub(crate) fn LLVMConstIntToPtr<'a>(ConstantVal: &'a Value, ToType: &'a Type) -> &'a Value;
    pub(crate) fn LLVMConstBitCast<'a>(ConstantVal: &'a Value, ToType: &'a Type) -> &'a Value;
    pub(crate) fn LLVMConstPointerCast<'a>(ConstantVal: &'a Value, ToType: &'a Type) -> &'a Value;
    pub(crate) fn LLVMGetAggregateElement(ConstantVal: &Value, Idx: c_uint) -> Option<&Value>;
    pub(crate) fn LLVMGetConstOpcode(ConstantVal: &Value) -> Opcode;
    pub(crate) fn LLVMIsAConstantExpr(Val: &Value) -> Option<&Value>;

    // Operations on global variables, functions, and aliases (globals)
    pub(crate) fn LLVMIsDeclaration(Global: &Value) -> Bool;
    pub(crate) fn LLVMGetLinkage(Global: &Value) -> RawEnum<Linkage>;
    pub(crate) fn LLVMSetLinkage(Global: &Value, RustLinkage: Linkage);
    pub(crate) fn LLVMSetSection(Global: &Value, Section: *const c_char);
    pub(crate) fn LLVMGetVisibility(Global: &Value) -> RawEnum<Visibility>;
    pub(crate) fn LLVMSetVisibility(Global: &Value, Viz: Visibility);
    pub(crate) fn LLVMGetAlignment(Global: &Value) -> c_uint;
    pub(crate) fn LLVMSetAlignment(Global: &Value, Bytes: c_uint);
    pub(crate) fn LLVMSetDLLStorageClass(V: &Value, C: DLLStorageClass);
    pub(crate) fn LLVMGlobalGetValueType(Global: &Value) -> &Type;

    // Operations on global variables
    pub(crate) fn LLVMIsAGlobalVariable(GlobalVar: &Value) -> Option<&Value>;
    pub(crate) fn LLVMAddGlobal<'a>(M: &'a Module, Ty: &'a Type, Name: *const c_char) -> &'a Value;
    pub(crate) fn LLVMGetNamedGlobal(M: &Module, Name: *const c_char) -> Option<&Value>;
    pub(crate) fn LLVMGetFirstGlobal(M: &Module) -> Option<&Value>;
    pub(crate) fn LLVMGetNextGlobal(GlobalVar: &Value) -> Option<&Value>;
    pub(crate) fn LLVMDeleteGlobal(GlobalVar: &Value);
    pub(crate) fn LLVMGetInitializer(GlobalVar: &Value) -> Option<&Value>;
    pub(crate) fn LLVMSetInitializer<'a>(GlobalVar: &'a Value, ConstantVal: &'a Value);
    pub(crate) fn LLVMIsThreadLocal(GlobalVar: &Value) -> Bool;
    pub(crate) fn LLVMSetThreadLocalMode(GlobalVar: &Value, Mode: ThreadLocalMode);
    pub(crate) fn LLVMIsGlobalConstant(GlobalVar: &Value) -> Bool;
    pub(crate) fn LLVMSetGlobalConstant(GlobalVar: &Value, IsConstant: Bool);
    pub(crate) safe fn LLVMSetTailCall(CallInst: &Value, IsTailCall: Bool);

    // Operations on attributes
    pub(crate) fn LLVMCreateStringAttribute(
        C: &Context,
        Name: *const c_char,
        NameLen: c_uint,
        Value: *const c_char,
        ValueLen: c_uint,
    ) -> &Attribute;

    // Operations on functions
    pub(crate) fn LLVMSetFunctionCallConv(Fn: &Value, CC: c_uint);

    // Operations about llvm intrinsics
    pub(crate) fn LLVMLookupIntrinsicID(Name: *const c_char, NameLen: size_t) -> c_uint;
    pub(crate) fn LLVMIntrinsicIsOverloaded(ID: NonZero<c_uint>) -> Bool;
    pub(crate) fn LLVMIntrinsicCopyOverloadedName2<'a>(
        Mod: &'a Module,
        ID: NonZero<c_uint>,
        ParamTypes: *const &'a Type,
        ParamCount: size_t,
        NameLength: *mut size_t,
    ) -> *mut c_char;

    // Operations on parameters
    pub(crate) fn LLVMIsAArgument(Val: &Value) -> Option<&Value>;
    pub(crate) safe fn LLVMCountParams(Fn: &Value) -> c_uint;
    pub(crate) fn LLVMGetParam(Fn: &Value, Index: c_uint) -> &Value;

    // Operations on basic blocks
    pub(crate) fn LLVMGetBasicBlockParent(BB: &BasicBlock) -> &Value;
    pub(crate) fn LLVMAppendBasicBlockInContext<'a>(
        C: &'a Context,
        Fn: &'a Value,
        Name: *const c_char,
    ) -> &'a BasicBlock;

    // Operations on instructions
    pub(crate) fn LLVMIsAInstruction(Val: &Value) -> Option<&Value>;
    pub(crate) fn LLVMGetFirstBasicBlock(Fn: &Value) -> &BasicBlock;
    pub(crate) fn LLVMGetOperand(Val: &Value, Index: c_uint) -> Option<&Value>;

    // Operations on call sites
    pub(crate) fn LLVMSetInstructionCallConv(Instr: &Value, CC: c_uint);

    // Operations on load/store instructions (only)
    pub(crate) fn LLVMSetVolatile(MemoryAccessInst: &Value, volatile: Bool);

    // Operations on phi nodes
    pub(crate) fn LLVMAddIncoming<'a>(
        PhiNode: &'a Value,
        IncomingValues: *const &'a Value,
        IncomingBlocks: *const &'a BasicBlock,
        Count: c_uint,
    );

    // Instruction builders
    pub(crate) fn LLVMCreateBuilderInContext(C: &Context) -> &mut Builder<'_>;
    pub(crate) fn LLVMPositionBuilderAtEnd<'a>(Builder: &Builder<'a>, Block: &'a BasicBlock);
    pub(crate) fn LLVMGetInsertBlock<'a>(Builder: &Builder<'a>) -> &'a BasicBlock;
    pub(crate) fn LLVMDisposeBuilder<'a>(Builder: &'a mut Builder<'a>);

    // Metadata
    pub(crate) fn LLVMSetCurrentDebugLocation2<'a>(Builder: &Builder<'a>, Loc: *const Metadata);
    pub(crate) fn LLVMGetCurrentDebugLocation2<'a>(Builder: &Builder<'a>) -> Option<&'a Metadata>;

    // Terminators
    pub(crate) safe fn LLVMBuildRetVoid<'a>(B: &Builder<'a>) -> &'a Value;
    pub(crate) fn LLVMBuildRet<'a>(B: &Builder<'a>, V: &'a Value) -> &'a Value;
    pub(crate) fn LLVMBuildBr<'a>(B: &Builder<'a>, Dest: &'a BasicBlock) -> &'a Value;
    pub(crate) fn LLVMBuildCondBr<'a>(
        B: &Builder<'a>,
        If: &'a Value,
        Then: &'a BasicBlock,
        Else: &'a BasicBlock,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildSwitch<'a>(
        B: &Builder<'a>,
        V: &'a Value,
        Else: &'a BasicBlock,
        NumCases: c_uint,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildLandingPad<'a>(
        B: &Builder<'a>,
        Ty: &'a Type,
        PersFn: Option<&'a Value>,
        NumClauses: c_uint,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildResume<'a>(B: &Builder<'a>, Exn: &'a Value) -> &'a Value;
    pub(crate) fn LLVMBuildUnreachable<'a>(B: &Builder<'a>) -> &'a Value;

    pub(crate) fn LLVMBuildCleanupPad<'a>(
        B: &Builder<'a>,
        ParentPad: Option<&'a Value>,
        Args: *const &'a Value,
        NumArgs: c_uint,
        Name: *const c_char,
    ) -> Option<&'a Value>;
    pub(crate) fn LLVMBuildCleanupRet<'a>(
        B: &Builder<'a>,
        CleanupPad: &'a Value,
        BB: Option<&'a BasicBlock>,
    ) -> Option<&'a Value>;
    pub(crate) fn LLVMBuildCatchPad<'a>(
        B: &Builder<'a>,
        ParentPad: &'a Value,
        Args: *const &'a Value,
        NumArgs: c_uint,
        Name: *const c_char,
    ) -> Option<&'a Value>;
    pub(crate) fn LLVMBuildCatchRet<'a>(
        B: &Builder<'a>,
        CatchPad: &'a Value,
        BB: &'a BasicBlock,
    ) -> Option<&'a Value>;
    pub(crate) fn LLVMBuildCatchSwitch<'a>(
        Builder: &Builder<'a>,
        ParentPad: Option<&'a Value>,
        UnwindBB: Option<&'a BasicBlock>,
        NumHandlers: c_uint,
        Name: *const c_char,
    ) -> Option<&'a Value>;
    pub(crate) fn LLVMAddHandler<'a>(CatchSwitch: &'a Value, Dest: &'a BasicBlock);
    pub(crate) fn LLVMSetPersonalityFn<'a>(Func: &'a Value, Pers: &'a Value);

    // Add a case to the switch instruction
    pub(crate) fn LLVMAddCase<'a>(Switch: &'a Value, OnVal: &'a Value, Dest: &'a BasicBlock);

    // Add a clause to the landing pad instruction
    pub(crate) fn LLVMAddClause<'a>(LandingPad: &'a Value, ClauseVal: &'a Value);

    // Set the cleanup on a landing pad instruction
    pub(crate) fn LLVMSetCleanup(LandingPad: &Value, Val: Bool);

    // Arithmetic
    pub(crate) fn LLVMBuildAdd<'a>(
        B: &Builder<'a>,
        LHS: &'a Value,
        RHS: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildFAdd<'a>(
        B: &Builder<'a>,
        LHS: &'a Value,
        RHS: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildSub<'a>(
        B: &Builder<'a>,
        LHS: &'a Value,
        RHS: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildFSub<'a>(
        B: &Builder<'a>,
        LHS: &'a Value,
        RHS: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildMul<'a>(
        B: &Builder<'a>,
        LHS: &'a Value,
        RHS: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildFMul<'a>(
        B: &Builder<'a>,
        LHS: &'a Value,
        RHS: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildUDiv<'a>(
        B: &Builder<'a>,
        LHS: &'a Value,
        RHS: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildExactUDiv<'a>(
        B: &Builder<'a>,
        LHS: &'a Value,
        RHS: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildSDiv<'a>(
        B: &Builder<'a>,
        LHS: &'a Value,
        RHS: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildExactSDiv<'a>(
        B: &Builder<'a>,
        LHS: &'a Value,
        RHS: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildFDiv<'a>(
        B: &Builder<'a>,
        LHS: &'a Value,
        RHS: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildURem<'a>(
        B: &Builder<'a>,
        LHS: &'a Value,
        RHS: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildSRem<'a>(
        B: &Builder<'a>,
        LHS: &'a Value,
        RHS: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildFRem<'a>(
        B: &Builder<'a>,
        LHS: &'a Value,
        RHS: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildShl<'a>(
        B: &Builder<'a>,
        LHS: &'a Value,
        RHS: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildLShr<'a>(
        B: &Builder<'a>,
        LHS: &'a Value,
        RHS: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildAShr<'a>(
        B: &Builder<'a>,
        LHS: &'a Value,
        RHS: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildNSWAdd<'a>(
        B: &Builder<'a>,
        LHS: &'a Value,
        RHS: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildNUWAdd<'a>(
        B: &Builder<'a>,
        LHS: &'a Value,
        RHS: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildNSWSub<'a>(
        B: &Builder<'a>,
        LHS: &'a Value,
        RHS: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildNUWSub<'a>(
        B: &Builder<'a>,
        LHS: &'a Value,
        RHS: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildNSWMul<'a>(
        B: &Builder<'a>,
        LHS: &'a Value,
        RHS: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildNUWMul<'a>(
        B: &Builder<'a>,
        LHS: &'a Value,
        RHS: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildAnd<'a>(
        B: &Builder<'a>,
        LHS: &'a Value,
        RHS: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildOr<'a>(
        B: &Builder<'a>,
        LHS: &'a Value,
        RHS: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildXor<'a>(
        B: &Builder<'a>,
        LHS: &'a Value,
        RHS: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildNeg<'a>(B: &Builder<'a>, V: &'a Value, Name: *const c_char)
    -> &'a Value;
    pub(crate) fn LLVMBuildFNeg<'a>(
        B: &Builder<'a>,
        V: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildNot<'a>(B: &Builder<'a>, V: &'a Value, Name: *const c_char)
    -> &'a Value;

    // Extra flags on arithmetic
    pub(crate) fn LLVMSetIsDisjoint(Instr: &Value, IsDisjoint: Bool);
    pub(crate) fn LLVMSetNUW(ArithInst: &Value, HasNUW: Bool);
    pub(crate) fn LLVMSetNSW(ArithInst: &Value, HasNSW: Bool);

    // Memory
    pub(crate) fn LLVMBuildAlloca<'a>(
        B: &Builder<'a>,
        Ty: &'a Type,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildArrayAlloca<'a>(
        B: &Builder<'a>,
        Ty: &'a Type,
        Val: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildLoad2<'a>(
        B: &Builder<'a>,
        Ty: &'a Type,
        PointerVal: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;

    pub(crate) fn LLVMBuildStore<'a>(B: &Builder<'a>, Val: &'a Value, Ptr: &'a Value) -> &'a Value;

    pub(crate) fn LLVMBuildGEPWithNoWrapFlags<'a>(
        B: &Builder<'a>,
        Ty: &'a Type,
        Pointer: &'a Value,
        Indices: *const &'a Value,
        NumIndices: c_uint,
        Name: *const c_char,
        Flags: GEPNoWrapFlags,
    ) -> &'a Value;

    // Casts
    pub(crate) fn LLVMBuildTrunc<'a>(
        B: &Builder<'a>,
        Val: &'a Value,
        DestTy: &'a Type,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildZExt<'a>(
        B: &Builder<'a>,
        Val: &'a Value,
        DestTy: &'a Type,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildSExt<'a>(
        B: &Builder<'a>,
        Val: &'a Value,
        DestTy: &'a Type,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildFPToUI<'a>(
        B: &Builder<'a>,
        Val: &'a Value,
        DestTy: &'a Type,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildFPToSI<'a>(
        B: &Builder<'a>,
        Val: &'a Value,
        DestTy: &'a Type,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildUIToFP<'a>(
        B: &Builder<'a>,
        Val: &'a Value,
        DestTy: &'a Type,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildSIToFP<'a>(
        B: &Builder<'a>,
        Val: &'a Value,
        DestTy: &'a Type,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildFPTrunc<'a>(
        B: &Builder<'a>,
        Val: &'a Value,
        DestTy: &'a Type,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildFPExt<'a>(
        B: &Builder<'a>,
        Val: &'a Value,
        DestTy: &'a Type,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildPtrToInt<'a>(
        B: &Builder<'a>,
        Val: &'a Value,
        DestTy: &'a Type,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildIntToPtr<'a>(
        B: &Builder<'a>,
        Val: &'a Value,
        DestTy: &'a Type,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildBitCast<'a>(
        B: &Builder<'a>,
        Val: &'a Value,
        DestTy: &'a Type,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildPointerCast<'a>(
        B: &Builder<'a>,
        Val: &'a Value,
        DestTy: &'a Type,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildIntCast2<'a>(
        B: &Builder<'a>,
        Val: &'a Value,
        DestTy: &'a Type,
        IsSigned: Bool,
        Name: *const c_char,
    ) -> &'a Value;

    // Comparisons
    pub(crate) fn LLVMBuildICmp<'a>(
        B: &Builder<'a>,
        Op: c_uint,
        LHS: &'a Value,
        RHS: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildFCmp<'a>(
        B: &Builder<'a>,
        Op: c_uint,
        LHS: &'a Value,
        RHS: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;

    // Miscellaneous instructions
    pub(crate) fn LLVMBuildPhi<'a>(B: &Builder<'a>, Ty: &'a Type, Name: *const c_char)
    -> &'a Value;
    pub(crate) fn LLVMBuildSelect<'a>(
        B: &Builder<'a>,
        If: &'a Value,
        Then: &'a Value,
        Else: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildVAArg<'a>(
        B: &Builder<'a>,
        list: &'a Value,
        Ty: &'a Type,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildExtractElement<'a>(
        B: &Builder<'a>,
        VecVal: &'a Value,
        Index: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildInsertElement<'a>(
        B: &Builder<'a>,
        VecVal: &'a Value,
        EltVal: &'a Value,
        Index: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildShuffleVector<'a>(
        B: &Builder<'a>,
        V1: &'a Value,
        V2: &'a Value,
        Mask: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildExtractValue<'a>(
        B: &Builder<'a>,
        AggVal: &'a Value,
        Index: c_uint,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildInsertValue<'a>(
        B: &Builder<'a>,
        AggVal: &'a Value,
        EltVal: &'a Value,
        Index: c_uint,
        Name: *const c_char,
    ) -> &'a Value;

    // Atomic Operations
    pub(crate) fn LLVMBuildAtomicCmpXchg<'a>(
        B: &Builder<'a>,
        LHS: &'a Value,
        CMP: &'a Value,
        RHS: &'a Value,
        Order: AtomicOrdering,
        FailureOrder: AtomicOrdering,
        SingleThreaded: Bool,
    ) -> &'a Value;

    pub(crate) fn LLVMSetWeak(CmpXchgInst: &Value, IsWeak: Bool);

    pub(crate) fn LLVMBuildAtomicRMW<'a>(
        B: &Builder<'a>,
        Op: AtomicRmwBinOp,
        LHS: &'a Value,
        RHS: &'a Value,
        Order: AtomicOrdering,
        SingleThreaded: Bool,
    ) -> &'a Value;

    pub(crate) fn LLVMBuildFence<'a>(
        B: &Builder<'a>,
        Order: AtomicOrdering,
        SingleThreaded: Bool,
        Name: *const c_char,
    ) -> &'a Value;

    /// Writes a module to the specified path. Returns 0 on success.
    pub(crate) fn LLVMWriteBitcodeToFile(M: &Module, Path: *const c_char) -> c_int;

    /// Creates a legacy pass manager -- only used for final codegen.
    pub(crate) fn LLVMCreatePassManager<'a>() -> &'a mut PassManager<'a>;

    pub(crate) fn LLVMAddAnalysisPasses<'a>(T: &'a TargetMachine, PM: &PassManager<'a>);

    pub(crate) fn LLVMGetHostCPUFeatures() -> *mut c_char;

    pub(crate) fn LLVMDisposeMessage(message: *mut c_char);

    pub(crate) fn LLVMIsMultithreaded() -> Bool;

    pub(crate) fn LLVMStructCreateNamed(C: &Context, Name: *const c_char) -> &Type;

    pub(crate) fn LLVMStructSetBody<'a>(
        StructTy: &'a Type,
        ElementTypes: *const &'a Type,
        ElementCount: c_uint,
        Packed: Bool,
    );

    pub(crate) safe fn LLVMMetadataAsValue<'a>(C: &'a Context, MD: &'a Metadata) -> &'a Value;

    pub(crate) fn LLVMSetUnnamedAddress(Global: &Value, UnnamedAddr: UnnamedAddr);

    pub(crate) fn LLVMIsAConstantInt(value_ref: &Value) -> Option<&ConstantInt>;

    pub(crate) fn LLVMGetOrInsertComdat(M: &Module, Name: *const c_char) -> &Comdat;
    pub(crate) fn LLVMSetComdat(V: &Value, C: &Comdat);

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

// FFI bindings for `DIBuilder` functions in the LLVM-C API.
// Try to keep these in the same order as in `llvm/include/llvm-c/DebugInfo.h`.
//
// FIXME(#134001): Audit all `Option` parameters, especially in lists, to check
// that they really are nullable on the C/C++ side. LLVM doesn't appear to
// actually document which ones are nullable.
unsafe extern "C" {
    pub(crate) fn LLVMCreateDIBuilder<'ll>(M: &'ll Module) -> *mut DIBuilder<'ll>;
    pub(crate) fn LLVMDisposeDIBuilder<'ll>(Builder: ptr::NonNull<DIBuilder<'ll>>);

    pub(crate) fn LLVMDIBuilderFinalize<'ll>(Builder: &DIBuilder<'ll>);

    pub(crate) fn LLVMDIBuilderCreateNameSpace<'ll>(
        Builder: &DIBuilder<'ll>,
        ParentScope: Option<&'ll Metadata>,
        Name: *const c_uchar, // See "PTR_LEN_STR".
        NameLen: size_t,
        ExportSymbols: llvm::Bool,
    ) -> &'ll Metadata;

    pub(crate) fn LLVMDIBuilderCreateLexicalBlock<'ll>(
        Builder: &DIBuilder<'ll>,
        Scope: &'ll Metadata,
        File: &'ll Metadata,
        Line: c_uint,
        Column: c_uint,
    ) -> &'ll Metadata;

    pub(crate) fn LLVMDIBuilderCreateLexicalBlockFile<'ll>(
        Builder: &DIBuilder<'ll>,
        Scope: &'ll Metadata,
        File: &'ll Metadata,
        Discriminator: c_uint, // (optional "DWARF path discriminator"; default is 0)
    ) -> &'ll Metadata;

    pub(crate) fn LLVMDIBuilderCreateDebugLocation<'ll>(
        Ctx: &'ll Context,
        Line: c_uint,
        Column: c_uint,
        Scope: &'ll Metadata,
        InlinedAt: Option<&'ll Metadata>,
    ) -> &'ll Metadata;
}

#[link(name = "llvm-wrapper", kind = "static")]
unsafe extern "C" {
    pub(crate) fn LLVMRustInstallErrorHandlers();
    pub(crate) fn LLVMRustDisableSystemDialogsOnCrash();

    // Create and destroy contexts.
    pub(crate) fn LLVMRustContextCreate(shouldDiscardNames: bool) -> &'static mut Context;

    /// See llvm::LLVMTypeKind::getTypeID.
    pub(crate) fn LLVMRustGetTypeKind(Ty: &Type) -> TypeKind;

    // Operations on all values
    pub(crate) fn LLVMRustGlobalAddMetadata<'a>(
        Val: &'a Value,
        KindID: c_uint,
        Metadata: &'a Metadata,
    );
    pub(crate) fn LLVMRustIsNonGVFunctionPointerTy(Val: &Value) -> bool;

    // Operations on scalar constants
    pub(crate) fn LLVMRustConstIntGetZExtValue(ConstantVal: &ConstantInt, Value: &mut u64) -> bool;
    pub(crate) fn LLVMRustConstInt128Get(
        ConstantVal: &ConstantInt,
        SExt: bool,
        high: &mut u64,
        low: &mut u64,
    ) -> bool;

    // Operations on global variables, functions, and aliases (globals)
    pub(crate) fn LLVMRustSetDSOLocal(Global: &Value, is_dso_local: bool);

    // Operations on global variables
    pub(crate) fn LLVMRustGetOrInsertGlobal<'a>(
        M: &'a Module,
        Name: *const c_char,
        NameLen: size_t,
        T: &'a Type,
    ) -> &'a Value;
    pub(crate) fn LLVMRustInsertPrivateGlobal<'a>(M: &'a Module, T: &'a Type) -> &'a Value;
    pub(crate) fn LLVMRustGetNamedValue(
        M: &Module,
        Name: *const c_char,
        NameLen: size_t,
    ) -> Option<&Value>;

    // Operations on attributes
    pub(crate) fn LLVMRustCreateAttrNoValue(C: &Context, attr: AttributeKind) -> &Attribute;
    pub(crate) fn LLVMRustCreateAlignmentAttr(C: &Context, bytes: u64) -> &Attribute;
    pub(crate) fn LLVMRustCreateDereferenceableAttr(C: &Context, bytes: u64) -> &Attribute;
    pub(crate) fn LLVMRustCreateDereferenceableOrNullAttr(C: &Context, bytes: u64) -> &Attribute;
    pub(crate) fn LLVMRustCreateByValAttr<'a>(C: &'a Context, ty: &'a Type) -> &'a Attribute;
    pub(crate) fn LLVMRustCreateStructRetAttr<'a>(C: &'a Context, ty: &'a Type) -> &'a Attribute;
    pub(crate) fn LLVMRustCreateElementTypeAttr<'a>(C: &'a Context, ty: &'a Type) -> &'a Attribute;
    pub(crate) fn LLVMRustCreateUWTableAttr(C: &Context, async_: bool) -> &Attribute;
    pub(crate) fn LLVMRustCreateAllocSizeAttr(C: &Context, size_arg: u32) -> &Attribute;
    pub(crate) fn LLVMRustCreateAllocKindAttr(C: &Context, size_arg: u64) -> &Attribute;
    pub(crate) fn LLVMRustCreateMemoryEffectsAttr(
        C: &Context,
        effects: MemoryEffects,
    ) -> &Attribute;
    pub(crate) fn LLVMRustCreateRangeAttribute(
        C: &Context,
        num_bits: c_uint,
        lower_words: *const u64,
        upper_words: *const u64,
    ) -> &Attribute;

    // Operations on functions
    pub(crate) fn LLVMRustGetOrInsertFunction<'a>(
        M: &'a Module,
        Name: *const c_char,
        NameLen: size_t,
        FunctionTy: &'a Type,
    ) -> &'a Value;
    pub(crate) fn LLVMRustAddFunctionAttributes<'a>(
        Fn: &'a Value,
        index: c_uint,
        Attrs: *const &'a Attribute,
        AttrsLen: size_t,
    );

    // Operations on call sites
    pub(crate) fn LLVMRustAddCallSiteAttributes<'a>(
        Instr: &'a Value,
        index: c_uint,
        Attrs: *const &'a Attribute,
        AttrsLen: size_t,
    );

    pub(crate) fn LLVMRustSetFastMath(Instr: &Value);
    pub(crate) fn LLVMRustSetAlgebraicMath(Instr: &Value);
    pub(crate) fn LLVMRustSetAllowReassoc(Instr: &Value);

    // Miscellaneous instructions
    pub(crate) fn LLVMRustBuildMemCpy<'a>(
        B: &Builder<'a>,
        Dst: &'a Value,
        DstAlign: c_uint,
        Src: &'a Value,
        SrcAlign: c_uint,
        Size: &'a Value,
        IsVolatile: bool,
    ) -> &'a Value;
    pub(crate) fn LLVMRustBuildMemMove<'a>(
        B: &Builder<'a>,
        Dst: &'a Value,
        DstAlign: c_uint,
        Src: &'a Value,
        SrcAlign: c_uint,
        Size: &'a Value,
        IsVolatile: bool,
    ) -> &'a Value;
    pub(crate) fn LLVMRustBuildMemSet<'a>(
        B: &Builder<'a>,
        Dst: &'a Value,
        DstAlign: c_uint,
        Val: &'a Value,
        Size: &'a Value,
        IsVolatile: bool,
    ) -> &'a Value;

    pub(crate) fn LLVMRustBuildVectorReduceFAdd<'a>(
        B: &Builder<'a>,
        Acc: &'a Value,
        Src: &'a Value,
    ) -> &'a Value;
    pub(crate) fn LLVMRustBuildVectorReduceFMul<'a>(
        B: &Builder<'a>,
        Acc: &'a Value,
        Src: &'a Value,
    ) -> &'a Value;
    pub(crate) fn LLVMRustBuildVectorReduceAdd<'a>(B: &Builder<'a>, Src: &'a Value) -> &'a Value;
    pub(crate) fn LLVMRustBuildVectorReduceMul<'a>(B: &Builder<'a>, Src: &'a Value) -> &'a Value;
    pub(crate) fn LLVMRustBuildVectorReduceAnd<'a>(B: &Builder<'a>, Src: &'a Value) -> &'a Value;
    pub(crate) fn LLVMRustBuildVectorReduceOr<'a>(B: &Builder<'a>, Src: &'a Value) -> &'a Value;
    pub(crate) fn LLVMRustBuildVectorReduceXor<'a>(B: &Builder<'a>, Src: &'a Value) -> &'a Value;
    pub(crate) fn LLVMRustBuildVectorReduceMin<'a>(
        B: &Builder<'a>,
        Src: &'a Value,
        IsSigned: bool,
    ) -> &'a Value;
    pub(crate) fn LLVMRustBuildVectorReduceMax<'a>(
        B: &Builder<'a>,
        Src: &'a Value,
        IsSigned: bool,
    ) -> &'a Value;
    pub(crate) fn LLVMRustBuildVectorReduceFMin<'a>(
        B: &Builder<'a>,
        Src: &'a Value,
        IsNaN: bool,
    ) -> &'a Value;
    pub(crate) fn LLVMRustBuildVectorReduceFMax<'a>(
        B: &Builder<'a>,
        Src: &'a Value,
        IsNaN: bool,
    ) -> &'a Value;

    pub(crate) fn LLVMRustBuildMinNum<'a>(
        B: &Builder<'a>,
        LHS: &'a Value,
        LHS: &'a Value,
    ) -> &'a Value;
    pub(crate) fn LLVMRustBuildMaxNum<'a>(
        B: &Builder<'a>,
        LHS: &'a Value,
        LHS: &'a Value,
    ) -> &'a Value;

    // Atomic Operations
    pub(crate) fn LLVMRustBuildAtomicLoad<'a>(
        B: &Builder<'a>,
        ElementType: &'a Type,
        PointerVal: &'a Value,
        Name: *const c_char,
        Order: AtomicOrdering,
    ) -> &'a Value;

    pub(crate) fn LLVMRustBuildAtomicStore<'a>(
        B: &Builder<'a>,
        Val: &'a Value,
        Ptr: &'a Value,
        Order: AtomicOrdering,
    ) -> &'a Value;

    pub(crate) fn LLVMRustTimeTraceProfilerInitialize();

    pub(crate) fn LLVMRustTimeTraceProfilerFinishThread();

    pub(crate) fn LLVMRustTimeTraceProfilerFinish(FileName: *const c_char);

    /// Returns a string describing the last error caused by an LLVMRust* call.
    pub(crate) fn LLVMRustGetLastError() -> *const c_char;

    /// Prints the timing information collected by `-Ztime-llvm-passes`.
    pub(crate) fn LLVMRustPrintPassTimings(OutStr: &RustString);

    /// Prints the statistics collected by `-Zprint-codegen-stats`.
    pub(crate) fn LLVMRustPrintStatistics(OutStr: &RustString);

    pub(crate) fn LLVMRustInlineAsmVerify(
        Ty: &Type,
        Constraints: *const c_uchar, // See "PTR_LEN_STR".
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
        ExpansionRegions: *const crate::coverageinfo::ffi::ExpansionRegion,
        NumExpansionRegions: size_t,
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
    pub(crate) fn LLVMRustDebugMetadataVersion() -> u32;
    pub(crate) fn LLVMRustVersionMajor() -> u32;
    pub(crate) fn LLVMRustVersionMinor() -> u32;
    pub(crate) fn LLVMRustVersionPatch() -> u32;

    /// Add LLVM module flags.
    ///
    /// In order for Rust-C LTO to work, module flags must be compatible with Clang. What
    /// "compatible" means depends on the merge behaviors involved.
    pub(crate) fn LLVMRustAddModuleFlagU32(
        M: &Module,
        MergeBehavior: ModuleFlagMergeBehavior,
        Name: *const c_char,
        NameLen: size_t,
        Value: u32,
    );

    pub(crate) fn LLVMRustAddModuleFlagString(
        M: &Module,
        MergeBehavior: ModuleFlagMergeBehavior,
        Name: *const c_char,
        NameLen: size_t,
        Value: *const c_char,
        ValueLen: size_t,
    );

    pub(crate) fn LLVMRustDIBuilderCreateCompileUnit<'a>(
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

    pub(crate) fn LLVMRustDIBuilderCreateFile<'a>(
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

    pub(crate) fn LLVMRustDIBuilderCreateSubroutineType<'a>(
        Builder: &DIBuilder<'a>,
        ParameterTypes: &'a DIArray,
    ) -> &'a DICompositeType;

    pub(crate) fn LLVMRustDIBuilderCreateFunction<'a>(
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

    pub(crate) fn LLVMRustDIBuilderCreateMethod<'a>(
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

    pub(crate) fn LLVMRustDIBuilderCreateBasicType<'a>(
        Builder: &DIBuilder<'a>,
        Name: *const c_char,
        NameLen: size_t,
        SizeInBits: u64,
        Encoding: c_uint,
    ) -> &'a DIBasicType;

    pub(crate) fn LLVMRustDIBuilderCreateTypedef<'a>(
        Builder: &DIBuilder<'a>,
        Type: &'a DIBasicType,
        Name: *const c_char,
        NameLen: size_t,
        File: &'a DIFile,
        LineNo: c_uint,
        Scope: Option<&'a DIScope>,
    ) -> &'a DIDerivedType;

    pub(crate) fn LLVMRustDIBuilderCreatePointerType<'a>(
        Builder: &DIBuilder<'a>,
        PointeeTy: &'a DIType,
        SizeInBits: u64,
        AlignInBits: u32,
        AddressSpace: c_uint,
        Name: *const c_char,
        NameLen: size_t,
    ) -> &'a DIDerivedType;

    pub(crate) fn LLVMRustDIBuilderCreateStructType<'a>(
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

    pub(crate) fn LLVMRustDIBuilderCreateMemberType<'a>(
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

    pub(crate) fn LLVMRustDIBuilderCreateVariantMemberType<'a>(
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

    pub(crate) fn LLVMRustDIBuilderCreateStaticMemberType<'a>(
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

    pub(crate) fn LLVMRustDIBuilderCreateQualifiedType<'a>(
        Builder: &DIBuilder<'a>,
        Tag: c_uint,
        Type: &'a DIType,
    ) -> &'a DIDerivedType;

    pub(crate) fn LLVMRustDIBuilderCreateStaticVariable<'a>(
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

    pub(crate) fn LLVMRustDIBuilderCreateVariable<'a>(
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

    pub(crate) fn LLVMRustDIBuilderCreateArrayType<'a>(
        Builder: &DIBuilder<'a>,
        Size: u64,
        AlignInBits: u32,
        Ty: &'a DIType,
        Subscripts: &'a DIArray,
    ) -> &'a DIType;

    pub(crate) fn LLVMRustDIBuilderGetOrCreateSubrange<'a>(
        Builder: &DIBuilder<'a>,
        Lo: i64,
        Count: i64,
    ) -> &'a DISubrange;

    pub(crate) fn LLVMRustDIBuilderGetOrCreateArray<'a>(
        Builder: &DIBuilder<'a>,
        Ptr: *const Option<&'a DIDescriptor>,
        Count: c_uint,
    ) -> &'a DIArray;

    pub(crate) fn LLVMRustDIBuilderInsertDeclareAtEnd<'a>(
        Builder: &DIBuilder<'a>,
        Val: &'a Value,
        VarInfo: &'a DIVariable,
        AddrOps: *const u64,
        AddrOpsCount: c_uint,
        DL: &'a DILocation,
        InsertAtEnd: &'a BasicBlock,
    );

    pub(crate) fn LLVMRustDIBuilderCreateEnumerator<'a>(
        Builder: &DIBuilder<'a>,
        Name: *const c_char,
        NameLen: size_t,
        Value: *const u64,
        SizeInBits: c_uint,
        IsUnsigned: bool,
    ) -> &'a DIEnumerator;

    pub(crate) fn LLVMRustDIBuilderCreateEnumerationType<'a>(
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

    pub(crate) fn LLVMRustDIBuilderCreateUnionType<'a>(
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

    pub(crate) fn LLVMRustDIBuilderCreateVariantPart<'a>(
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

    pub(crate) fn LLVMRustDIBuilderCreateTemplateTypeParameter<'a>(
        Builder: &DIBuilder<'a>,
        Scope: Option<&'a DIScope>,
        Name: *const c_char,
        NameLen: size_t,
        Ty: &'a DIType,
    ) -> &'a DITemplateTypeParameter;

    pub(crate) fn LLVMRustDICompositeTypeReplaceArrays<'a>(
        Builder: &DIBuilder<'a>,
        CompositeType: &'a DIType,
        Elements: Option<&'a DIArray>,
        Params: Option<&'a DIArray>,
    );

    pub(crate) fn LLVMRustDILocationCloneWithBaseDiscriminator<'a>(
        Location: &'a DILocation,
        BD: c_uint,
    ) -> Option<&'a DILocation>;

    pub(crate) fn LLVMRustWriteTypeToString(Type: &Type, s: &RustString);
    pub(crate) fn LLVMRustWriteValueToString(value_ref: &Value, s: &RustString);

    pub(crate) fn LLVMRustHasFeature(T: &TargetMachine, s: *const c_char) -> bool;

    pub(crate) fn LLVMRustPrintTargetCPUs(TM: &TargetMachine, OutStr: &RustString);
    pub(crate) fn LLVMRustGetTargetFeaturesCount(T: &TargetMachine) -> size_t;
    pub(crate) fn LLVMRustGetTargetFeature(
        T: &TargetMachine,
        Index: size_t,
        Feature: &mut *const c_char,
        Desc: &mut *const c_char,
    );

    pub(crate) fn LLVMRustGetHostCPUName(LenOut: &mut size_t) -> *const u8;

    // This function makes copies of pointed to data, so the data's lifetime may end after this
    // function returns.
    pub(crate) fn LLVMRustCreateTargetMachine(
        Triple: *const c_char,
        CPU: *const c_char,
        Features: *const c_char,
        Abi: *const c_char,
        Model: CodeModel,
        Reloc: RelocModel,
        Level: CodeGenOptLevel,
        FloatABIType: FloatAbi,
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

    pub(crate) fn LLVMRustDisposeTargetMachine(T: *mut TargetMachine);
    pub(crate) fn LLVMRustAddLibraryInfo<'a>(
        PM: &PassManager<'a>,
        M: &'a Module,
        DisableSimplifyLibCalls: bool,
    );
    pub(crate) fn LLVMRustWriteOutputFile<'a>(
        T: &'a TargetMachine,
        PM: *mut PassManager<'a>,
        M: &'a Module,
        Output: *const c_char,
        DwoOutput: *const c_char,
        FileType: FileType,
        VerifyIR: bool,
    ) -> LLVMRustResult;
    pub(crate) fn LLVMRustOptimize<'a>(
        M: &'a Module,
        TM: &'a TargetMachine,
        OptLevel: PassBuilderOptLevel,
        OptStage: OptStage,
        IsLinkerPluginLTO: bool,
        NoPrepopulatePasses: bool,
        VerifyIR: bool,
        LintIR: bool,
        ThinLTOBuffer: Option<&mut *mut ThinLTOBuffer>,
        EmitThinLTO: bool,
        EmitThinLTOSummary: bool,
        MergeFunctions: bool,
        UnrollLoops: bool,
        SLPVectorize: bool,
        LoopVectorize: bool,
        DisableSimplifyLibCalls: bool,
        EmitLifetimeMarkers: bool,
        RunEnzyme: bool,
        PrintBeforeEnzyme: bool,
        PrintAfterEnzyme: bool,
        PrintPasses: bool,
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
    pub(crate) fn LLVMRustPrintModule(
        M: &Module,
        Output: *const c_char,
        Demangle: extern "C" fn(*const c_char, size_t, *mut c_char, size_t) -> size_t,
    ) -> LLVMRustResult;
    pub(crate) fn LLVMRustSetLLVMOptions(Argc: c_int, Argv: *const *const c_char);
    pub(crate) fn LLVMRustPrintPasses();
    pub(crate) fn LLVMRustSetNormalizedTarget(M: &Module, triple: *const c_char);
    pub(crate) fn LLVMRustRunRestrictionPass(M: &Module, syms: *const *const c_char, len: size_t);

    pub(crate) fn LLVMRustOpenArchive(path: *const c_char) -> Option<&'static mut Archive>;
    pub(crate) fn LLVMRustArchiveIteratorNew(AR: &Archive) -> &mut ArchiveIterator<'_>;
    pub(crate) fn LLVMRustArchiveIteratorNext<'a>(
        AIR: &ArchiveIterator<'a>,
    ) -> Option<&'a mut ArchiveChild<'a>>;
    pub(crate) fn LLVMRustArchiveChildName(
        ACR: &ArchiveChild<'_>,
        size: &mut size_t,
    ) -> *const c_char;
    pub(crate) fn LLVMRustArchiveChildFree<'a>(ACR: &'a mut ArchiveChild<'a>);
    pub(crate) fn LLVMRustArchiveIteratorFree<'a>(AIR: &'a mut ArchiveIterator<'a>);
    pub(crate) fn LLVMRustDestroyArchive(AR: &'static mut Archive);

    pub(crate) fn LLVMRustWriteTwineToString(T: &Twine, s: &RustString);

    pub(crate) fn LLVMRustUnpackOptimizationDiagnostic<'a>(
        DI: &'a DiagnosticInfo,
        pass_name_out: &RustString,
        function_out: &mut Option<&'a Value>,
        loc_line_out: &mut c_uint,
        loc_column_out: &mut c_uint,
        loc_filename_out: &RustString,
        message_out: &RustString,
    );

    pub(crate) fn LLVMRustUnpackInlineAsmDiagnostic<'a>(
        DI: &'a DiagnosticInfo,
        level_out: &mut DiagnosticLevel,
        cookie_out: &mut u64,
        message_out: &mut Option<&'a Twine>,
    );

    pub(crate) fn LLVMRustWriteDiagnosticInfoToString(DI: &DiagnosticInfo, s: &RustString);
    pub(crate) fn LLVMRustGetDiagInfoKind(DI: &DiagnosticInfo) -> DiagnosticKind;

    pub(crate) fn LLVMRustGetSMDiagnostic<'a>(
        DI: &'a DiagnosticInfo,
        cookie_out: &mut u64,
    ) -> &'a SMDiagnostic;

    pub(crate) fn LLVMRustUnpackSMDiagnostic(
        d: &SMDiagnostic,
        message_out: &RustString,
        buffer_out: &RustString,
        level_out: &mut DiagnosticLevel,
        loc_out: &mut c_uint,
        ranges_out: *mut c_uint,
        num_ranges: &mut usize,
    ) -> bool;

    pub(crate) fn LLVMRustWriteArchive(
        Dst: *const c_char,
        NumMembers: size_t,
        Members: *const &RustArchiveMember<'_>,
        WriteSymbtab: bool,
        Kind: ArchiveKind,
        isEC: bool,
    ) -> LLVMRustResult;
    pub(crate) fn LLVMRustArchiveMemberNew<'a>(
        Filename: *const c_char,
        Name: *const c_char,
        Child: Option<&ArchiveChild<'a>>,
    ) -> &'a mut RustArchiveMember<'a>;
    pub(crate) fn LLVMRustArchiveMemberFree<'a>(Member: &'a mut RustArchiveMember<'a>);

    pub(crate) fn LLVMRustSetDataLayoutFromTargetMachine<'a>(M: &'a Module, TM: &'a TargetMachine);

    pub(crate) fn LLVMRustPositionBuilderAtStart<'a>(B: &Builder<'a>, BB: &'a BasicBlock);

    pub(crate) fn LLVMRustSetModulePICLevel(M: &Module);
    pub(crate) fn LLVMRustSetModulePIELevel(M: &Module);
    pub(crate) fn LLVMRustSetModuleCodeModel(M: &Module, Model: CodeModel);
    pub(crate) fn LLVMRustModuleBufferCreate(M: &Module) -> &'static mut ModuleBuffer;
    pub(crate) fn LLVMRustModuleBufferPtr(p: &ModuleBuffer) -> *const u8;
    pub(crate) fn LLVMRustModuleBufferLen(p: &ModuleBuffer) -> usize;
    pub(crate) fn LLVMRustModuleBufferFree(p: &'static mut ModuleBuffer);
    pub(crate) fn LLVMRustModuleCost(M: &Module) -> u64;
    pub(crate) fn LLVMRustModuleInstructionStats(M: &Module, Str: &RustString);

    pub(crate) fn LLVMRustThinLTOBufferCreate(
        M: &Module,
        is_thin: bool,
        emit_summary: bool,
    ) -> &'static mut ThinLTOBuffer;
    pub(crate) fn LLVMRustThinLTOBufferFree(M: &'static mut ThinLTOBuffer);
    pub(crate) fn LLVMRustThinLTOBufferPtr(M: &ThinLTOBuffer) -> *const c_char;
    pub(crate) fn LLVMRustThinLTOBufferLen(M: &ThinLTOBuffer) -> size_t;
    pub(crate) fn LLVMRustThinLTOBufferThinLinkDataPtr(M: &ThinLTOBuffer) -> *const c_char;
    pub(crate) fn LLVMRustThinLTOBufferThinLinkDataLen(M: &ThinLTOBuffer) -> size_t;
    pub(crate) fn LLVMRustCreateThinLTOData(
        Modules: *const ThinLTOModule,
        NumModules: size_t,
        PreservedSymbols: *const *const c_char,
        PreservedSymbolsLen: size_t,
    ) -> Option<&'static mut ThinLTOData>;
    pub(crate) fn LLVMRustPrepareThinLTORename(
        Data: &ThinLTOData,
        Module: &Module,
        Target: &TargetMachine,
    );
    pub(crate) fn LLVMRustPrepareThinLTOResolveWeak(Data: &ThinLTOData, Module: &Module) -> bool;
    pub(crate) fn LLVMRustPrepareThinLTOInternalize(Data: &ThinLTOData, Module: &Module) -> bool;
    pub(crate) fn LLVMRustPrepareThinLTOImport(
        Data: &ThinLTOData,
        Module: &Module,
        Target: &TargetMachine,
    ) -> bool;
    pub(crate) fn LLVMRustFreeThinLTOData(Data: &'static mut ThinLTOData);
    pub(crate) fn LLVMRustParseBitcodeForLTO(
        Context: &Context,
        Data: *const u8,
        len: usize,
        Identifier: *const c_char,
    ) -> Option<&Module>;
    pub(crate) fn LLVMRustGetSliceFromObjectDataByName(
        data: *const u8,
        len: usize,
        name: *const u8,
        name_len: usize,
        out_len: &mut usize,
    ) -> *const u8;

    pub(crate) fn LLVMRustLinkerNew(M: &Module) -> &mut Linker<'_>;
    pub(crate) fn LLVMRustLinkerAdd(
        linker: &Linker<'_>,
        bytecode: *const c_char,
        bytecode_len: usize,
    ) -> bool;
    pub(crate) fn LLVMRustLinkerFree<'a>(linker: &'a mut Linker<'a>);
    pub(crate) fn LLVMRustComputeLTOCacheKey(
        key_out: &RustString,
        mod_id: *const c_char,
        data: &ThinLTOData,
    );

    pub(crate) fn LLVMRustContextGetDiagnosticHandler(
        Context: &Context,
    ) -> Option<&DiagnosticHandler>;
    pub(crate) fn LLVMRustContextSetDiagnosticHandler(
        context: &Context,
        diagnostic_handler: Option<&DiagnosticHandler>,
    );
    pub(crate) fn LLVMRustContextConfigureDiagnosticHandler(
        context: &Context,
        diagnostic_handler_callback: DiagnosticHandlerTy,
        diagnostic_handler_context: *mut c_void,
        remark_all_passes: bool,
        remark_passes: *const *const c_char,
        remark_passes_len: usize,
        remark_file: *const c_char,
        pgo_available: bool,
    );

    pub(crate) fn LLVMRustGetMangledName(V: &Value, out: &RustString);

    pub(crate) fn LLVMRustGetElementTypeArgIndex(CallSite: &Value) -> i32;

    pub(crate) fn LLVMRustLLVMHasZlibCompressionForDebugSymbols() -> bool;

    pub(crate) fn LLVMRustLLVMHasZstdCompressionForDebugSymbols() -> bool;

    pub(crate) fn LLVMRustGetSymbols(
        buf_ptr: *const u8,
        buf_len: usize,
        state: *mut c_void,
        callback: GetSymbolsCallback,
        error_callback: GetSymbolsErrorCallback,
    ) -> *mut c_void;

    pub(crate) fn LLVMRustIs64BitSymbolicFile(buf_ptr: *const u8, buf_len: usize) -> bool;

    pub(crate) fn LLVMRustIsECObject(buf_ptr: *const u8, buf_len: usize) -> bool;

    pub(crate) fn LLVMRustSetNoSanitizeAddress(Global: &Value);
    pub(crate) fn LLVMRustSetNoSanitizeHWAddress(Global: &Value);
}
