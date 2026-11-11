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

use std::fmt::{self, Debug};
use std::marker::PhantomData;
use std::num::NonZero;
use std::ptr;

use bitflags::bitflags;
use libc::{c_char, c_int, c_uchar, c_uint, c_ulonglong, c_void, size_t};

pub(crate) use self::branded::{
    Context, ContextRaw, Module, ModuleRaw, Type, Value, wrap_llcx, wrap_llmod,
};
use super::RustString;
use super::debuginfo::{
    DIArray, DIBuilder, DIDerivedType, DIDescriptor, DIFile, DIFlags, DILocation, DISPFlags,
    DIScope, DISubprogram, DITemplateTypeParameter, DIType, DebugEmissionKind, DebugNameTableKind,
};
use crate::llvm;
use crate::llvm::MetadataKindId;
use crate::macros::TryFromU32;

mod branded;

/// In the LLVM-C API, boolean values are passed as `typedef int LLVMBool`,
/// which has a different ABI from Rust or C++ `bool`.
///
/// This wrapper does not implement `PartialEq`.
/// To test the underlying boolean value, use [`Self::is_true`].
#[derive(Clone, Copy)]
#[repr(transparent)]
pub(crate) struct Bool {
    value: c_int,
}

pub(crate) const TRUE: Bool = Bool::TRUE;
pub(crate) const FALSE: Bool = Bool::FALSE;

impl Bool {
    pub(crate) const TRUE: Self = Self { value: 1 };
    pub(crate) const FALSE: Self = Self { value: 0 };

    pub(crate) const fn from_bool(rust_bool: bool) -> Self {
        if rust_bool { Self::TRUE } else { Self::FALSE }
    }

    /// Converts this LLVM-C boolean to a Rust `bool`
    pub(crate) fn is_true(self) -> bool {
        // Since we're interacting with a C API, follow the C convention of
        // treating any nonzero value as true.
        self.value != Self::FALSE.value
    }
}

impl Debug for Bool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.value {
            0 => f.write_str("FALSE"),
            1 => f.write_str("TRUE"),
            // As with `Self::is_true`, treat any nonzero value as true.
            v => write!(f, "TRUE ({v})"),
        }
    }
}

/// Convenience trait to convert `bool` to `llvm::Bool` with an explicit method call.
///
/// Being able to write `b.to_llvm_bool()` is less noisy than `llvm::Bool::from(b)`,
/// while being more explicit and less mistake-prone than something like `b.into()`.
pub(crate) trait ToLlvmBool: Copy {
    fn to_llvm_bool(self) -> llvm::Bool;
}

impl ToLlvmBool for bool {
    #[inline(always)]
    fn to_llvm_bool(self) -> llvm::Bool {
        llvm::Bool::from_bool(self)
    }
}

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

/// Must match the layout of `LLVMTailCallKind`.
#[derive(Copy, Clone, PartialEq, Debug)]
#[repr(C)]
#[allow(dead_code)]
pub(crate) enum TailCallKind {
    None = 0,
    Tail = 1,
    MustTail = 2,
    NoTail = 3,
}

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
    SwiftCallConv = 16,
    Tail = 18,
    PreserveNone = 21,
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

/// Must match the layout of `llvm::UWTableKind`.
#[derive(Copy, Clone)]
#[repr(C)]
pub(crate) enum UWTableKind {
    /// No unwind table requested
    None = 0,
    /// "Synchronous" unwind tables
    Sync = 1,
    /// "Asynchronous" unwind tables (instr precise)
    Async = 2,
}

/// Must match the layout of `llvm::FramePointerKind`.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[expect(dead_code, reason = "Some variants are unused, but are kept to match LLVM")]
pub(crate) enum FramePointerKind {
    None = 0,
    NonLeaf = 1,
    All = 2,
    Reserved = 3,
    NonLeafNoReserve = 4,
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
    CapturesAddress = 7,
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
    DeadOnReturn = 44,
    CapturesReadOnly = 45,
    CapturesNone = 46,
    SanitizeRealtimeNonblocking = 47,
    SanitizeRealtimeBlocking = 48,
    Convergent = 49,
    NoFree = 50,
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

/// Must match the layout of `LLVMTypeKind`.
///
/// Use [`RawEnum<TypeKind>`] for values of `LLVMTypeKind` returned from LLVM,
/// to avoid risk of UB if LLVM adds new enum values.
///
/// All of LLVM's variants should be declared here, even if no Rust-side code refers
/// to them, because unknown variants will cause [`RawEnum::to_rust`] to panic.
#[derive(Copy, Clone, PartialEq, Debug, TryFromU32)]
#[repr(C)]
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

/// LLVMRustFileType
#[derive(Copy, Clone)]
#[repr(C)]
pub(crate) enum FileType {
    AssemblyFile,
    ObjectFile,
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
    pub sanitize_realtime: bool,
    pub sanitize_thread: bool,
    pub sanitize_hwaddress: bool,
    pub sanitize_hwaddress_recover: bool,
    pub sanitize_kernel_address: bool,
    pub sanitize_kernel_address_recover: bool,
    pub sanitize_kernel_hwaddress: bool,
    pub sanitize_kernel_hwaddress_recover: bool,
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

/// LLVMDiagnosticSeverity
#[derive(Copy, Clone)]
#[repr(C)]
#[allow(dead_code)] // Variants constructed by C++.
pub(crate) enum DiagnosticSeverity {
    Error,
    Warning,
    Remark,
    Note,
}

unsafe extern "C" {
    // LLVMRustThinLTOData
    pub(crate) type ThinLTOData;
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
    ReadOnlyNotPure,
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

/// Must match the layout of `LLVMRustCompressionKind`.
#[derive(Copy, Clone)]
#[repr(C)]
pub(crate) enum CompressionKind {
    None = 0,
    Zlib = 1,
    Zstd = 2,
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
    pub(crate) type ConstantInt;
    pub(crate) type Attribute;
    pub(crate) type Metadata;
    pub(crate) type BasicBlock;
    pub(crate) type Comdat;
    /// `&'ll DbgRecord` represents `LLVMDbgRecordRef`.
    pub(crate) type DbgRecord;
}
#[repr(C)]
pub(crate) struct Builder<'a>(InvariantOpaque<'a>);
unsafe extern "C" {
    pub type TargetMachine;
}
unsafe extern "C" {
    pub(crate) type MCSubtargetInfo;
    pub(crate) type Twine;
    pub(crate) type DiagnosticInfo;
    pub(crate) type SMDiagnostic;
}
/// Opaque pointee of `LLVMOperandBundleRef`.
#[repr(C)]
pub(crate) struct OperandBundle<'a>(InvariantOpaque<'a>);
#[repr(C)]
pub(crate) struct Linker<'a>(InvariantOpaque<'a>);

unsafe extern "C" {
    /// Corresponds to LLVM's `DiagnosticHandler`.
    pub(crate) type DiagnosticHandler;
}

/// Must match the declaration of `LLVMRustDiagnosticHandlerCallback`.
pub(crate) type DiagnosticHandlerCallback = unsafe extern "C" fn(&DiagnosticInfo, *mut c_void);

pub(crate) mod debuginfo {
    use bitflags::bitflags;

    use super::{InvariantOpaque, Metadata};

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
    pub(crate) type DIArray = DIDescriptor;
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
    pub(crate) type Buffer;
}

pub(crate) type SelfProfileBeforePassCallback =
    unsafe extern "C" fn(*mut c_void, *const c_char, *const c_char);
pub(crate) type SelfProfileAfterPassCallback = unsafe extern "C" fn(*mut c_void);

pub(crate) type GetSymbolsCallback =
    unsafe extern "C" fn(*mut c_void, *const c_char) -> *mut c_void;
pub(crate) type GetSymbolsErrorCallback = unsafe extern "C" fn(*const c_char) -> *mut c_void;

unsafe extern "C" {
    // Create and destroy contexts.
    pub(crate) fn LLVMContextCreate() -> ptr::NonNull<ContextRaw>;
    pub(crate) fn LLVMContextDispose(C: ptr::NonNull<ContextRaw>);
    pub(crate) fn LLVMContextSetDiscardValueNames<'ll>(C: Context<'ll>, Discard: Bool);
    pub(crate) fn LLVMGetMDKindIDInContext<'ll>(
        C: Context<'ll>,
        Name: *const c_uchar, // See "PTR_LEN_STR".
        SLen: c_uint,
    ) -> MetadataKindId;

    /// Gets the actual version of LLVM that we are linked to at runtime.
    ///
    /// # Safety
    /// Can be called without initializing LLVM.
    pub(crate) safe fn LLVMGetVersion(major: &mut c_uint, minor: &mut c_uint, patch: &mut c_uint);

    pub(crate) fn LLVMDisposeTargetMachine(T: ptr::NonNull<TargetMachine>);

    // Create modules.
    pub(crate) fn LLVMModuleCreateWithNameInContext<'ll>(
        ModuleID: *const c_char,
        C: Context<'ll>,
    ) -> Module<'ll>;
    pub(crate) safe fn LLVMCloneModule<'ll>(M: Module<'ll>) -> Module<'ll>;

    /// Data layout. See Module::getDataLayout.
    pub(crate) fn LLVMGetDataLayoutStr<'ll>(M: Module<'ll>) -> *const c_char;
    pub(crate) fn LLVMSetDataLayout<'ll>(M: Module<'ll>, Triple: *const c_char);

    /// Create the specified uniqued inline asm string. See `InlineAsm::get()`.
    pub(crate) fn LLVMGetInlineAsm<'ll>(
        Ty: Type<'ll>,
        AsmString: *const c_uchar, // See "PTR_LEN_STR".
        AsmStringSize: size_t,
        Constraints: *const c_uchar, // See "PTR_LEN_STR".
        ConstraintsSize: size_t,
        HasSideEffects: llvm::Bool,
        IsAlignStack: llvm::Bool,
        Dialect: AsmDialect,
        CanThrow: llvm::Bool,
    ) -> Value<'ll>;

    pub(crate) safe fn LLVMGetTypeKind<'ll>(Ty: Type<'ll>) -> RawEnum<TypeKind>;

    // Operations on integer types
    pub(crate) fn LLVMInt1TypeInContext<'ll>(C: Context<'ll>) -> Type<'ll>;
    pub(crate) fn LLVMInt8TypeInContext<'ll>(C: Context<'ll>) -> Type<'ll>;
    pub(crate) fn LLVMInt16TypeInContext<'ll>(C: Context<'ll>) -> Type<'ll>;
    pub(crate) fn LLVMInt32TypeInContext<'ll>(C: Context<'ll>) -> Type<'ll>;
    pub(crate) fn LLVMInt64TypeInContext<'ll>(C: Context<'ll>) -> Type<'ll>;
    pub(crate) safe fn LLVMIntTypeInContext<'ll>(C: Context<'ll>, NumBits: c_uint) -> Type<'ll>;

    pub(crate) fn LLVMGetIntTypeWidth<'ll>(IntegerTy: Type<'ll>) -> c_uint;

    // Operations on real types
    pub(crate) fn LLVMHalfTypeInContext<'ll>(C: Context<'ll>) -> Type<'ll>;
    pub(crate) fn LLVMFloatTypeInContext<'ll>(C: Context<'ll>) -> Type<'ll>;
    pub(crate) fn LLVMDoubleTypeInContext<'ll>(C: Context<'ll>) -> Type<'ll>;
    pub(crate) fn LLVMFP128TypeInContext<'ll>(C: Context<'ll>) -> Type<'ll>;

    // Operations on non-IEEE real types
    pub(crate) fn LLVMBFloatTypeInContext<'ll>(C: Context<'ll>) -> Type<'ll>;
    pub(crate) fn LLVMX86FP80TypeInContext<'ll>(C: Context<'ll>) -> Type<'ll>;
    pub(crate) fn LLVMPPCFP128TypeInContext<'ll>(C: Context<'ll>) -> Type<'ll>;

    // Operations on function types
    pub(crate) fn LLVMFunctionType<'ll>(
        ReturnType: Type<'ll>,
        ParamTypes: *const Type<'ll>,
        ParamCount: c_uint,
        IsVarArg: Bool,
    ) -> Type<'ll>;
    pub(crate) fn LLVMCountParamTypes<'ll>(FunctionTy: Type<'ll>) -> c_uint;
    pub(crate) fn LLVMGetParamTypes<'ll>(FunctionTy: Type<'ll>, Dest: *mut Type<'ll>);
    pub(crate) fn LLVMGetReturnType<'ll>(FunctionTy: Type<'ll>) -> Type<'ll>;
    pub(crate) fn LLVMIsFunctionVarArg<'ll>(FunctionTy: Type<'ll>) -> Bool;

    // Operations on struct types
    pub(crate) fn LLVMStructTypeInContext<'ll>(
        C: Context<'ll>,
        ElementTypes: *const Type<'ll>,
        ElementCount: c_uint,
        Packed: Bool,
    ) -> Type<'ll>;

    // Operations on array, pointer, and vector types (sequence types)
    pub(crate) safe fn LLVMPointerTypeInContext<'ll>(
        C: Context<'ll>,
        AddressSpace: c_uint,
    ) -> Type<'ll>;
    pub(crate) fn LLVMVectorType<'ll>(ElementType: Type<'ll>, ElementCount: c_uint) -> Type<'ll>;
    pub(crate) fn LLVMScalableVectorType<'ll>(
        ElementType: Type<'ll>,
        ElementCount: c_uint,
    ) -> Type<'ll>;

    pub(crate) fn LLVMGetElementType<'ll>(Ty: Type<'ll>) -> Type<'ll>;
    pub(crate) fn LLVMGetVectorSize<'ll>(VectorTy: Type<'ll>) -> c_uint;

    // Operations on other types
    pub(crate) fn LLVMVoidTypeInContext<'ll>(C: Context<'ll>) -> Type<'ll>;

    // Operations on all values
    pub(crate) fn LLVMTypeOf<'ll>(Val: Value<'ll>) -> Type<'ll>;
    pub(crate) fn LLVMGetValueName2<'ll>(Val: Value<'ll>, Length: *mut size_t) -> *const c_char;
    pub(crate) fn LLVMSetValueName2<'ll>(
        Val: Value<'ll>,
        Name: *const c_uchar, // See "PTR_LEN_STR".
        NameLen: size_t,
    );
    pub(crate) fn LLVMReplaceAllUsesWith<'ll>(OldVal: Value<'ll>, NewVal: Value<'ll>);
    pub(crate) safe fn LLVMGetMetadata<'ll>(
        Val: Value<'ll>,
        KindID: MetadataKindId,
    ) -> Option<Value<'ll>>;
    pub(crate) safe fn LLVMSetMetadata<'ll>(
        Val: Value<'ll>,
        KindID: MetadataKindId,
        Node: Value<'ll>,
    );
    pub(crate) fn LLVMGlobalSetMetadata<'ll>(
        Val: Value<'ll>,
        KindID: MetadataKindId,
        Metadata: &'ll Metadata,
    );
    pub(crate) fn LLVMGlobalAddMetadata<'ll>(
        Global: Value<'ll>,
        Kind: MetadataKindId,
        MD: &'ll Metadata,
    );
    pub(crate) safe fn LLVMValueAsMetadata<'ll>(Node: Value<'ll>) -> &'ll Metadata;

    // Operations on constants of any type
    pub(crate) fn LLVMConstNull<'ll>(Ty: Type<'ll>) -> Value<'ll>;
    pub(crate) fn LLVMGetUndef<'ll>(Ty: Type<'ll>) -> Value<'ll>;
    pub(crate) fn LLVMGetPoison<'ll>(Ty: Type<'ll>) -> Value<'ll>;

    // Operations on metadata
    pub(crate) fn LLVMMDStringInContext2<'ll>(
        C: Context<'ll>,
        Str: *const c_uchar, // See "PTR_LEN_STR".
        SLen: size_t,
    ) -> &'ll Metadata;
    pub(crate) fn LLVMMDNodeInContext2<'ll>(
        C: Context<'ll>,
        Vals: *const &'ll Metadata,
        Count: size_t,
    ) -> &'ll Metadata;
    pub(crate) fn LLVMAddNamedMetadataOperand<'ll>(
        M: Module<'ll>,
        Name: *const c_char,
        Val: Value<'ll>,
    );
    pub(crate) fn LLVMReplaceMDNodeOperandWith<'ll>(
        Val: Value<'ll>,
        index: u32,
        replacement: &'ll Metadata,
    );

    // Operations on scalar constants
    pub(crate) fn LLVMConstInt<'ll>(
        IntTy: Type<'ll>,
        N: c_ulonglong,
        SignExtend: Bool,
    ) -> Value<'ll>;
    pub(crate) fn LLVMConstIntOfArbitraryPrecision<'ll>(
        IntTy: Type<'ll>,
        Wn: c_uint,
        Ws: *const u64,
    ) -> Value<'ll>;
    pub(crate) fn LLVMConstReal<'ll>(RealTy: Type<'ll>, N: f64) -> Value<'ll>;

    // Operations on composite constants
    pub(crate) fn LLVMConstArray2<'ll>(
        ElementTy: Type<'ll>,
        ConstantVals: *const Value<'ll>,
        Length: u64,
    ) -> Value<'ll>;
    pub(crate) fn LLVMArrayType2<'ll>(ElementType: Type<'ll>, ElementCount: u64) -> Type<'ll>;
    pub(crate) fn LLVMConstStringInContext2<'ll>(
        C: Context<'ll>,
        Str: *const c_uchar, // See "PTR_LEN_STR".
        Length: size_t,
        DontNullTerminate: Bool,
    ) -> Value<'ll>;
    pub(crate) fn LLVMConstStructInContext<'ll>(
        C: Context<'ll>,
        ConstantVals: *const Value<'ll>,
        Count: c_uint,
        Packed: Bool,
    ) -> Value<'ll>;
    pub(crate) fn LLVMConstNamedStruct<'ll>(
        StructTy: Type<'ll>,
        ConstantVals: *const Value<'ll>,
        Count: c_uint,
    ) -> Value<'ll>;
    pub(crate) fn LLVMConstVector<'ll>(
        ScalarConstantVals: *const Value<'ll>,
        Size: c_uint,
    ) -> Value<'ll>;

    // Constant expressions
    pub(crate) fn LLVMConstInBoundsGEP2<'ll>(
        ty: Type<'ll>,
        ConstantVal: Value<'ll>,
        ConstantIndices: *const Value<'ll>,
        NumIndices: c_uint,
    ) -> Value<'ll>;
    pub(crate) fn LLVMConstPtrToInt<'ll>(ConstantVal: Value<'ll>, ToType: Type<'ll>) -> Value<'ll>;
    pub(crate) fn LLVMConstIntToPtr<'ll>(ConstantVal: Value<'ll>, ToType: Type<'ll>) -> Value<'ll>;
    pub(crate) fn LLVMConstBitCast<'ll>(ConstantVal: Value<'ll>, ToType: Type<'ll>) -> Value<'ll>;
    pub(crate) fn LLVMConstPointerCast<'ll>(
        ConstantVal: Value<'ll>,
        ToType: Type<'ll>,
    ) -> Value<'ll>;
    pub(crate) fn LLVMGetAggregateElement<'ll>(
        ConstantVal: Value<'ll>,
        Idx: c_uint,
    ) -> Option<Value<'ll>>;
    pub(crate) fn LLVMGetConstOpcode<'ll>(ConstantVal: Value<'ll>) -> Opcode;
    pub(crate) fn LLVMIsAConstantExpr<'ll>(Val: Value<'ll>) -> Option<Value<'ll>>;

    // Operations on global variables, functions, and aliases (globals)
    pub(crate) fn LLVMIsDeclaration<'ll>(Global: Value<'ll>) -> Bool;
    pub(crate) fn LLVMGetLinkage<'ll>(Global: Value<'ll>) -> RawEnum<Linkage>;
    pub(crate) fn LLVMSetLinkage<'ll>(Global: Value<'ll>, RustLinkage: Linkage);
    pub(crate) fn LLVMSetSection<'ll>(Global: Value<'ll>, Section: *const c_char);
    pub(crate) fn LLVMGetVisibility<'ll>(Global: Value<'ll>) -> RawEnum<Visibility>;
    pub(crate) fn LLVMSetVisibility<'ll>(Global: Value<'ll>, Viz: Visibility);
    pub(crate) fn LLVMGetAlignment<'ll>(Global: Value<'ll>) -> c_uint;
    pub(crate) fn LLVMSetAlignment<'ll>(Global: Value<'ll>, Bytes: c_uint);
    pub(crate) fn LLVMSetDLLStorageClass<'ll>(V: Value<'ll>, C: DLLStorageClass);
    pub(crate) fn LLVMGlobalGetValueType<'ll>(Global: Value<'ll>) -> Type<'ll>;

    // Operations on global variables
    pub(crate) safe fn LLVMIsAGlobalVariable<'ll>(GlobalVar: Value<'ll>) -> Option<Value<'ll>>;
    pub(crate) fn LLVMAddGlobal<'ll>(
        M: Module<'ll>,
        Ty: Type<'ll>,
        Name: *const c_char,
    ) -> Value<'ll>;
    pub(crate) fn LLVMGetNamedGlobal<'ll>(
        M: Module<'ll>,
        Name: *const c_char,
    ) -> Option<Value<'ll>>;
    pub(crate) fn LLVMGetFirstGlobal<'ll>(M: Module<'ll>) -> Option<Value<'ll>>;
    pub(crate) fn LLVMGetNextGlobal<'ll>(GlobalVar: Value<'ll>) -> Option<Value<'ll>>;
    pub(crate) fn LLVMDeleteGlobal<'ll>(GlobalVar: Value<'ll>);
    pub(crate) safe fn LLVMGetInitializer<'ll>(GlobalVar: Value<'ll>) -> Option<Value<'ll>>;
    pub(crate) fn LLVMSetInitializer<'ll>(GlobalVar: Value<'ll>, ConstantVal: Value<'ll>);
    pub(crate) safe fn LLVMIsThreadLocal<'ll>(GlobalVar: Value<'ll>) -> Bool;
    pub(crate) fn LLVMSetThreadLocalMode<'ll>(GlobalVar: Value<'ll>, Mode: ThreadLocalMode);
    pub(crate) safe fn LLVMIsGlobalConstant<'ll>(GlobalVar: Value<'ll>) -> Bool;
    pub(crate) safe fn LLVMSetGlobalConstant<'ll>(GlobalVar: Value<'ll>, IsConstant: Bool);
    pub(crate) safe fn LLVMSetTailCall<'ll>(CallInst: Value<'ll>, IsTailCall: Bool);
    pub(crate) safe fn LLVMSetTailCallKind<'ll>(CallInst: Value<'ll>, kind: TailCallKind);
    pub(crate) safe fn LLVMSetExternallyInitialized<'ll>(GlobalVar: Value<'ll>, IsExtInit: Bool);

    // Operations on global aliases
    pub(crate) fn LLVMAddAlias2<'ll>(
        M: Module<'ll>,
        ValueTy: Type<'ll>,
        AddressSpace: c_uint,
        Aliasee: Value<'ll>,
        Name: *const c_char,
    ) -> Value<'ll>;
    pub(crate) fn LLVMGetFirstGlobalAlias<'ll>(M: Module<'ll>) -> Option<Value<'ll>>;
    pub(crate) fn LLVMGetNextGlobalAlias<'ll>(GlobalAlias: Value<'ll>) -> Option<Value<'ll>>;

    // Operations on attributes
    pub(crate) fn LLVMCreateStringAttribute<'ll>(
        C: Context<'ll>,
        Name: *const c_uchar, // See "PTR_LEN_STR".
        NameLen: c_uint,
        Value: *const c_uchar, // See "PTR_LEN_STR".
        ValueLen: c_uint,
    ) -> &'ll Attribute;

    // Operations on functions
    pub(crate) fn LLVMSetFunctionCallConv<'ll>(Fn: Value<'ll>, CC: c_uint);
    pub(crate) fn LLVMAddFunction<'ll>(
        Mod: Module<'ll>,
        Name: *const c_char,
        FunctionTy: Type<'ll>,
    ) -> Value<'ll>;
    pub(crate) fn LLVMDeleteFunction<'ll>(Fn: Value<'ll>);
    pub(crate) fn LLVMGetOrInsertFunction<'ll>(
        M: Module<'ll>,
        Name: *const c_uchar, // See "PTR_LEN_STR".
        NameLen: size_t,
        FunctionTy: Type<'ll>,
    ) -> Value<'ll>;

    // Operations about llvm intrinsics
    pub(crate) fn LLVMLookupIntrinsicID(
        Name: *const c_uchar, // See "PTR_LEN_STR".
        NameLen: size_t,
    ) -> c_uint;
    pub(crate) fn LLVMIntrinsicIsOverloaded(ID: NonZero<c_uint>) -> Bool;
    pub(crate) fn LLVMGetIntrinsicDeclaration<'ll>(
        Mod: Module<'ll>,
        ID: NonZero<c_uint>,
        ParamTypes: *const Type<'ll>,
        ParamCount: size_t,
    ) -> Value<'ll>;
    pub(crate) fn LLVMRustUpgradeIntrinsicFunction<'ll>(
        Fn: Value<'ll>,
        NewFn: &mut Option<Value<'ll>>,
    ) -> bool;
    pub(crate) fn LLVMRustIsTargetIntrinsic(ID: NonZero<c_uint>) -> bool;

    // Operations on parameters
    pub(crate) fn LLVMIsAArgument<'ll>(Val: Value<'ll>) -> Option<Value<'ll>>;
    pub(crate) safe fn LLVMCountParams<'ll>(Fn: Value<'ll>) -> c_uint;
    pub(crate) fn LLVMGetParam<'ll>(Fn: Value<'ll>, Index: c_uint) -> Value<'ll>;

    // Operations on basic blocks
    pub(crate) fn LLVMGetBasicBlockParent<'ll>(BB: &'ll BasicBlock) -> Value<'ll>;
    pub(crate) fn LLVMAppendBasicBlockInContext<'ll>(
        C: Context<'ll>,
        Fn: Value<'ll>,
        Name: *const c_char,
    ) -> &'ll BasicBlock;

    // Operations on instructions
    pub(crate) fn LLVMIsAInstruction<'ll>(Val: Value<'ll>) -> Option<Value<'ll>>;
    pub(crate) fn LLVMGetFirstBasicBlock<'ll>(Fn: Value<'ll>) -> &'ll BasicBlock;
    pub(crate) fn LLVMGetOperand<'ll>(Val: Value<'ll>, Index: c_uint) -> Option<Value<'ll>>;

    // Operations on call sites
    pub(crate) fn LLVMSetInstructionCallConv<'ll>(Instr: Value<'ll>, CC: c_uint);

    // Operations on load/store instructions (only)
    pub(crate) fn LLVMSetVolatile<'ll>(MemoryAccessInst: Value<'ll>, volatile: Bool);
    pub(crate) fn LLVMSetOrdering<'ll>(MemoryAccessInst: Value<'ll>, Ordering: AtomicOrdering);

    // Operations on phi nodes
    pub(crate) fn LLVMAddIncoming<'ll>(
        PhiNode: Value<'ll>,
        IncomingValues: *const Value<'ll>,
        IncomingBlocks: *const &'ll BasicBlock,
        Count: c_uint,
    );

    // Instruction builders
    pub(crate) fn LLVMCreateBuilderForModule<'ll>(M: Module<'ll>) -> ptr::NonNull<Builder<'ll>>;
    pub(crate) fn LLVMPositionBuilderAtEnd<'a>(Builder: &Builder<'a>, Block: &'a BasicBlock);
    pub(crate) fn LLVMGetInsertBlock<'a>(Builder: &Builder<'a>) -> &'a BasicBlock;
    pub(crate) fn LLVMDisposeBuilder<'ll>(Builder: ptr::NonNull<Builder<'ll>>);

    // Metadata
    pub(crate) fn LLVMSetCurrentDebugLocation2<'a>(Builder: &Builder<'a>, Loc: *const Metadata);
    pub(crate) fn LLVMGetCurrentDebugLocation2<'a>(Builder: &Builder<'a>) -> Option<&'a Metadata>;

    // Terminators
    pub(crate) safe fn LLVMBuildRetVoid<'ll>(B: &Builder<'ll>) -> Value<'ll>;
    pub(crate) fn LLVMBuildRet<'ll>(B: &Builder<'ll>, V: Value<'ll>) -> Value<'ll>;
    pub(crate) fn LLVMBuildBr<'ll>(B: &Builder<'ll>, Dest: &'ll BasicBlock) -> Value<'ll>;
    pub(crate) fn LLVMBuildCondBr<'ll>(
        B: &Builder<'ll>,
        If: Value<'ll>,
        Then: &'ll BasicBlock,
        Else: &'ll BasicBlock,
    ) -> Value<'ll>;
    pub(crate) fn LLVMBuildSwitch<'ll>(
        B: &Builder<'ll>,
        V: Value<'ll>,
        Else: &'ll BasicBlock,
        NumCases: c_uint,
    ) -> Value<'ll>;
    pub(crate) fn LLVMBuildLandingPad<'ll>(
        B: &Builder<'ll>,
        Ty: Type<'ll>,
        PersFn: Option<Value<'ll>>,
        NumClauses: c_uint,
        Name: *const c_char,
    ) -> Value<'ll>;
    pub(crate) fn LLVMBuildResume<'ll>(B: &Builder<'ll>, Exn: Value<'ll>) -> Value<'ll>;
    pub(crate) fn LLVMBuildUnreachable<'ll>(B: &Builder<'ll>) -> Value<'ll>;

    pub(crate) fn LLVMBuildCleanupPad<'ll>(
        B: &Builder<'ll>,
        ParentPad: Option<Value<'ll>>,
        Args: *const Value<'ll>,
        NumArgs: c_uint,
        Name: *const c_char,
    ) -> Option<Value<'ll>>;
    pub(crate) fn LLVMBuildCleanupRet<'ll>(
        B: &Builder<'ll>,
        CleanupPad: Value<'ll>,
        BB: Option<&'ll BasicBlock>,
    ) -> Option<Value<'ll>>;
    pub(crate) fn LLVMBuildCatchPad<'ll>(
        B: &Builder<'ll>,
        ParentPad: Value<'ll>,
        Args: *const Value<'ll>,
        NumArgs: c_uint,
        Name: *const c_char,
    ) -> Option<Value<'ll>>;
    pub(crate) fn LLVMBuildCatchRet<'ll>(
        B: &Builder<'ll>,
        CatchPad: Value<'ll>,
        BB: &'ll BasicBlock,
    ) -> Option<Value<'ll>>;
    pub(crate) fn LLVMBuildCatchSwitch<'ll>(
        Builder: &Builder<'ll>,
        ParentPad: Option<Value<'ll>>,
        UnwindBB: Option<&'ll BasicBlock>,
        NumHandlers: c_uint,
        Name: *const c_char,
    ) -> Option<Value<'ll>>;
    pub(crate) fn LLVMAddHandler<'ll>(CatchSwitch: Value<'ll>, Dest: &'ll BasicBlock);
    pub(crate) fn LLVMSetPersonalityFn<'ll>(Func: Value<'ll>, Pers: Value<'ll>);

    // Add a case to the switch instruction
    pub(crate) fn LLVMAddCase<'ll>(Switch: Value<'ll>, OnVal: Value<'ll>, Dest: &'ll BasicBlock);

    // Add a clause to the landing pad instruction
    pub(crate) fn LLVMAddClause<'ll>(LandingPad: Value<'ll>, ClauseVal: Value<'ll>);

    // Set the cleanup on a landing pad instruction
    pub(crate) fn LLVMSetCleanup<'ll>(LandingPad: Value<'ll>, Val: Bool);

    // Arithmetic
    pub(crate) fn LLVMBuildAdd<'ll>(
        B: &Builder<'ll>,
        LHS: Value<'ll>,
        RHS: Value<'ll>,
        Name: *const c_char,
    ) -> Value<'ll>;
    pub(crate) fn LLVMBuildFAdd<'ll>(
        B: &Builder<'ll>,
        LHS: Value<'ll>,
        RHS: Value<'ll>,
        Name: *const c_char,
    ) -> Value<'ll>;
    pub(crate) fn LLVMBuildSub<'ll>(
        B: &Builder<'ll>,
        LHS: Value<'ll>,
        RHS: Value<'ll>,
        Name: *const c_char,
    ) -> Value<'ll>;
    pub(crate) fn LLVMBuildFSub<'ll>(
        B: &Builder<'ll>,
        LHS: Value<'ll>,
        RHS: Value<'ll>,
        Name: *const c_char,
    ) -> Value<'ll>;
    pub(crate) fn LLVMBuildMul<'ll>(
        B: &Builder<'ll>,
        LHS: Value<'ll>,
        RHS: Value<'ll>,
        Name: *const c_char,
    ) -> Value<'ll>;
    pub(crate) fn LLVMBuildFMul<'ll>(
        B: &Builder<'ll>,
        LHS: Value<'ll>,
        RHS: Value<'ll>,
        Name: *const c_char,
    ) -> Value<'ll>;
    pub(crate) fn LLVMBuildUDiv<'ll>(
        B: &Builder<'ll>,
        LHS: Value<'ll>,
        RHS: Value<'ll>,
        Name: *const c_char,
    ) -> Value<'ll>;
    pub(crate) fn LLVMBuildExactUDiv<'ll>(
        B: &Builder<'ll>,
        LHS: Value<'ll>,
        RHS: Value<'ll>,
        Name: *const c_char,
    ) -> Value<'ll>;
    pub(crate) fn LLVMBuildSDiv<'ll>(
        B: &Builder<'ll>,
        LHS: Value<'ll>,
        RHS: Value<'ll>,
        Name: *const c_char,
    ) -> Value<'ll>;
    pub(crate) fn LLVMBuildExactSDiv<'ll>(
        B: &Builder<'ll>,
        LHS: Value<'ll>,
        RHS: Value<'ll>,
        Name: *const c_char,
    ) -> Value<'ll>;
    pub(crate) fn LLVMBuildFDiv<'ll>(
        B: &Builder<'ll>,
        LHS: Value<'ll>,
        RHS: Value<'ll>,
        Name: *const c_char,
    ) -> Value<'ll>;
    pub(crate) fn LLVMBuildURem<'ll>(
        B: &Builder<'ll>,
        LHS: Value<'ll>,
        RHS: Value<'ll>,
        Name: *const c_char,
    ) -> Value<'ll>;
    pub(crate) fn LLVMBuildSRem<'ll>(
        B: &Builder<'ll>,
        LHS: Value<'ll>,
        RHS: Value<'ll>,
        Name: *const c_char,
    ) -> Value<'ll>;
    pub(crate) fn LLVMBuildFRem<'ll>(
        B: &Builder<'ll>,
        LHS: Value<'ll>,
        RHS: Value<'ll>,
        Name: *const c_char,
    ) -> Value<'ll>;
    pub(crate) fn LLVMBuildShl<'ll>(
        B: &Builder<'ll>,
        LHS: Value<'ll>,
        RHS: Value<'ll>,
        Name: *const c_char,
    ) -> Value<'ll>;
    pub(crate) fn LLVMBuildLShr<'ll>(
        B: &Builder<'ll>,
        LHS: Value<'ll>,
        RHS: Value<'ll>,
        Name: *const c_char,
    ) -> Value<'ll>;
    pub(crate) fn LLVMBuildAShr<'ll>(
        B: &Builder<'ll>,
        LHS: Value<'ll>,
        RHS: Value<'ll>,
        Name: *const c_char,
    ) -> Value<'ll>;
    pub(crate) fn LLVMBuildNSWAdd<'ll>(
        B: &Builder<'ll>,
        LHS: Value<'ll>,
        RHS: Value<'ll>,
        Name: *const c_char,
    ) -> Value<'ll>;
    pub(crate) fn LLVMBuildNUWAdd<'ll>(
        B: &Builder<'ll>,
        LHS: Value<'ll>,
        RHS: Value<'ll>,
        Name: *const c_char,
    ) -> Value<'ll>;
    pub(crate) fn LLVMBuildNSWSub<'ll>(
        B: &Builder<'ll>,
        LHS: Value<'ll>,
        RHS: Value<'ll>,
        Name: *const c_char,
    ) -> Value<'ll>;
    pub(crate) fn LLVMBuildNUWSub<'ll>(
        B: &Builder<'ll>,
        LHS: Value<'ll>,
        RHS: Value<'ll>,
        Name: *const c_char,
    ) -> Value<'ll>;
    pub(crate) fn LLVMBuildNSWMul<'ll>(
        B: &Builder<'ll>,
        LHS: Value<'ll>,
        RHS: Value<'ll>,
        Name: *const c_char,
    ) -> Value<'ll>;
    pub(crate) fn LLVMBuildNUWMul<'ll>(
        B: &Builder<'ll>,
        LHS: Value<'ll>,
        RHS: Value<'ll>,
        Name: *const c_char,
    ) -> Value<'ll>;
    pub(crate) fn LLVMBuildAnd<'ll>(
        B: &Builder<'ll>,
        LHS: Value<'ll>,
        RHS: Value<'ll>,
        Name: *const c_char,
    ) -> Value<'ll>;
    pub(crate) fn LLVMBuildOr<'ll>(
        B: &Builder<'ll>,
        LHS: Value<'ll>,
        RHS: Value<'ll>,
        Name: *const c_char,
    ) -> Value<'ll>;
    pub(crate) fn LLVMBuildXor<'ll>(
        B: &Builder<'ll>,
        LHS: Value<'ll>,
        RHS: Value<'ll>,
        Name: *const c_char,
    ) -> Value<'ll>;
    pub(crate) fn LLVMBuildNeg<'ll>(
        B: &Builder<'ll>,
        V: Value<'ll>,
        Name: *const c_char,
    ) -> Value<'ll>;
    pub(crate) fn LLVMBuildFNeg<'ll>(
        B: &Builder<'ll>,
        V: Value<'ll>,
        Name: *const c_char,
    ) -> Value<'ll>;
    pub(crate) fn LLVMBuildNot<'ll>(
        B: &Builder<'ll>,
        V: Value<'ll>,
        Name: *const c_char,
    ) -> Value<'ll>;

    // Extra flags on arithmetic
    pub(crate) fn LLVMSetIsDisjoint<'ll>(Instr: Value<'ll>, IsDisjoint: Bool);
    pub(crate) fn LLVMSetNUW<'ll>(ArithInst: Value<'ll>, HasNUW: Bool);
    pub(crate) fn LLVMSetNSW<'ll>(ArithInst: Value<'ll>, HasNSW: Bool);

    // Memory
    pub(crate) fn LLVMBuildAlloca<'ll>(
        B: &Builder<'ll>,
        Ty: Type<'ll>,
        Name: *const c_char,
    ) -> Value<'ll>;
    pub(crate) fn LLVMBuildLoad2<'ll>(
        B: &Builder<'ll>,
        Ty: Type<'ll>,
        PointerVal: Value<'ll>,
        Name: *const c_char,
    ) -> Value<'ll>;

    pub(crate) fn LLVMBuildStore<'ll>(
        B: &Builder<'ll>,
        Val: Value<'ll>,
        Ptr: Value<'ll>,
    ) -> Value<'ll>;

    pub(crate) fn LLVMBuildGEPWithNoWrapFlags<'ll>(
        B: &Builder<'ll>,
        Ty: Type<'ll>,
        Pointer: Value<'ll>,
        Indices: *const Value<'ll>,
        NumIndices: c_uint,
        Name: *const c_char,
        Flags: GEPNoWrapFlags,
    ) -> Value<'ll>;

    // Casts
    pub(crate) fn LLVMBuildTrunc<'ll>(
        B: &Builder<'ll>,
        Val: Value<'ll>,
        DestTy: Type<'ll>,
        Name: *const c_char,
    ) -> Value<'ll>;
    pub(crate) fn LLVMBuildZExt<'ll>(
        B: &Builder<'ll>,
        Val: Value<'ll>,
        DestTy: Type<'ll>,
        Name: *const c_char,
    ) -> Value<'ll>;
    pub(crate) fn LLVMBuildSExt<'ll>(
        B: &Builder<'ll>,
        Val: Value<'ll>,
        DestTy: Type<'ll>,
        Name: *const c_char,
    ) -> Value<'ll>;
    pub(crate) fn LLVMBuildFPToUI<'ll>(
        B: &Builder<'ll>,
        Val: Value<'ll>,
        DestTy: Type<'ll>,
        Name: *const c_char,
    ) -> Value<'ll>;
    pub(crate) fn LLVMBuildFPToSI<'ll>(
        B: &Builder<'ll>,
        Val: Value<'ll>,
        DestTy: Type<'ll>,
        Name: *const c_char,
    ) -> Value<'ll>;
    pub(crate) fn LLVMBuildUIToFP<'ll>(
        B: &Builder<'ll>,
        Val: Value<'ll>,
        DestTy: Type<'ll>,
        Name: *const c_char,
    ) -> Value<'ll>;
    pub(crate) fn LLVMBuildSIToFP<'ll>(
        B: &Builder<'ll>,
        Val: Value<'ll>,
        DestTy: Type<'ll>,
        Name: *const c_char,
    ) -> Value<'ll>;
    pub(crate) fn LLVMBuildFPTrunc<'ll>(
        B: &Builder<'ll>,
        Val: Value<'ll>,
        DestTy: Type<'ll>,
        Name: *const c_char,
    ) -> Value<'ll>;
    pub(crate) fn LLVMBuildFPExt<'ll>(
        B: &Builder<'ll>,
        Val: Value<'ll>,
        DestTy: Type<'ll>,
        Name: *const c_char,
    ) -> Value<'ll>;
    pub(crate) fn LLVMBuildPtrToInt<'ll>(
        B: &Builder<'ll>,
        Val: Value<'ll>,
        DestTy: Type<'ll>,
        Name: *const c_char,
    ) -> Value<'ll>;
    pub(crate) fn LLVMBuildIntToPtr<'ll>(
        B: &Builder<'ll>,
        Val: Value<'ll>,
        DestTy: Type<'ll>,
        Name: *const c_char,
    ) -> Value<'ll>;
    pub(crate) fn LLVMBuildBitCast<'ll>(
        B: &Builder<'ll>,
        Val: Value<'ll>,
        DestTy: Type<'ll>,
        Name: *const c_char,
    ) -> Value<'ll>;
    pub(crate) fn LLVMBuildPointerCast<'ll>(
        B: &Builder<'ll>,
        Val: Value<'ll>,
        DestTy: Type<'ll>,
        Name: *const c_char,
    ) -> Value<'ll>;
    pub(crate) fn LLVMBuildIntCast2<'ll>(
        B: &Builder<'ll>,
        Val: Value<'ll>,
        DestTy: Type<'ll>,
        IsSigned: Bool,
        Name: *const c_char,
    ) -> Value<'ll>;

    // Comparisons
    pub(crate) fn LLVMBuildICmp<'ll>(
        B: &Builder<'ll>,
        Op: c_uint,
        LHS: Value<'ll>,
        RHS: Value<'ll>,
        Name: *const c_char,
    ) -> Value<'ll>;
    pub(crate) fn LLVMBuildFCmp<'ll>(
        B: &Builder<'ll>,
        Op: c_uint,
        LHS: Value<'ll>,
        RHS: Value<'ll>,
        Name: *const c_char,
    ) -> Value<'ll>;

    // Miscellaneous instructions
    pub(crate) fn LLVMBuildPhi<'ll>(
        B: &Builder<'ll>,
        Ty: Type<'ll>,
        Name: *const c_char,
    ) -> Value<'ll>;
    pub(crate) fn LLVMBuildSelect<'ll>(
        B: &Builder<'ll>,
        If: Value<'ll>,
        Then: Value<'ll>,
        Else: Value<'ll>,
        Name: *const c_char,
    ) -> Value<'ll>;
    pub(crate) fn LLVMBuildVAArg<'ll>(
        B: &Builder<'ll>,
        list: Value<'ll>,
        Ty: Type<'ll>,
        Name: *const c_char,
    ) -> Value<'ll>;
    pub(crate) fn LLVMBuildExtractElement<'ll>(
        B: &Builder<'ll>,
        VecVal: Value<'ll>,
        Index: Value<'ll>,
        Name: *const c_char,
    ) -> Value<'ll>;
    pub(crate) fn LLVMBuildInsertElement<'ll>(
        B: &Builder<'ll>,
        VecVal: Value<'ll>,
        EltVal: Value<'ll>,
        Index: Value<'ll>,
        Name: *const c_char,
    ) -> Value<'ll>;
    pub(crate) fn LLVMBuildShuffleVector<'ll>(
        B: &Builder<'ll>,
        V1: Value<'ll>,
        V2: Value<'ll>,
        Mask: Value<'ll>,
        Name: *const c_char,
    ) -> Value<'ll>;
    pub(crate) fn LLVMBuildExtractValue<'ll>(
        B: &Builder<'ll>,
        AggVal: Value<'ll>,
        Index: c_uint,
        Name: *const c_char,
    ) -> Value<'ll>;
    pub(crate) fn LLVMBuildInsertValue<'ll>(
        B: &Builder<'ll>,
        AggVal: Value<'ll>,
        EltVal: Value<'ll>,
        Index: c_uint,
        Name: *const c_char,
    ) -> Value<'ll>;

    // Atomic Operations
    pub(crate) fn LLVMBuildAtomicCmpXchg<'ll>(
        B: &Builder<'ll>,
        LHS: Value<'ll>,
        CMP: Value<'ll>,
        RHS: Value<'ll>,
        Order: AtomicOrdering,
        FailureOrder: AtomicOrdering,
        SingleThreaded: Bool,
    ) -> Value<'ll>;

    pub(crate) fn LLVMSetWeak<'ll>(CmpXchgInst: Value<'ll>, IsWeak: Bool);

    pub(crate) fn LLVMBuildAtomicRMW<'ll>(
        B: &Builder<'ll>,
        Op: AtomicRmwBinOp,
        LHS: Value<'ll>,
        RHS: Value<'ll>,
        Order: AtomicOrdering,
        SingleThreaded: Bool,
    ) -> Value<'ll>;

    pub(crate) fn LLVMBuildFence<'ll>(
        B: &Builder<'ll>,
        Order: AtomicOrdering,
        SingleThreaded: Bool,
        Name: *const c_char,
    ) -> Value<'ll>;

    /// Writes a module to the specified path. Returns 0 on success.
    pub(crate) fn LLVMWriteBitcodeToFile<'ll>(M: Module<'ll>, Path: *const c_char) -> c_int;

    pub(crate) fn LLVMGetHostCPUFeatures() -> *mut c_char;

    pub(crate) fn LLVMDisposeMessage(message: *mut c_char);

    pub(crate) fn LLVMIsMultithreaded() -> Bool;

    pub(crate) fn LLVMStructCreateNamed<'ll>(C: Context<'ll>, Name: *const c_char) -> Type<'ll>;

    pub(crate) fn LLVMStructSetBody<'ll>(
        StructTy: Type<'ll>,
        ElementTypes: *const Type<'ll>,
        ElementCount: c_uint,
        Packed: Bool,
    );

    pub(crate) fn LLVMCountStructElementTypes<'ll>(StructTy: Type<'ll>) -> c_uint;
    pub(crate) fn LLVMGetStructElementTypes<'ll>(StructTy: Type<'ll>, Dest: *mut Type<'ll>);

    pub(crate) safe fn LLVMMetadataAsValue<'ll>(C: Context<'ll>, MD: &'ll Metadata) -> Value<'ll>;

    pub(crate) safe fn LLVMSetUnnamedAddress<'ll>(Global: Value<'ll>, UnnamedAddr: UnnamedAddr);

    pub(crate) fn LLVMIsAConstantInt<'ll>(value_ref: Value<'ll>) -> Option<&'ll ConstantInt>;

    pub(crate) fn LLVMGetOrInsertComdat<'ll>(M: Module<'ll>, Name: *const c_char) -> &'ll Comdat;
    pub(crate) fn LLVMSetComdat<'ll>(V: Value<'ll>, C: &Comdat);

    pub(crate) fn LLVMCreateOperandBundle<'ll>(
        Tag: *const c_uchar, // See "PTR_LEN_STR".
        TagLen: size_t,
        Args: *const Value<'ll>,
        NumArgs: c_uint,
    ) -> *mut OperandBundle<'ll>;
    pub(crate) fn LLVMDisposeOperandBundle(Bundle: ptr::NonNull<OperandBundle<'_>>);

    pub(crate) fn LLVMBuildCallWithOperandBundles<'ll>(
        B: &Builder<'ll>,
        Ty: Type<'ll>,
        Fn: Value<'ll>,
        Args: *const Value<'ll>,
        NumArgs: c_uint,
        Bundles: *const &OperandBundle<'ll>,
        NumBundles: c_uint,
        Name: *const c_char,
    ) -> Value<'ll>;
    pub(crate) fn LLVMBuildInvokeWithOperandBundles<'ll>(
        B: &Builder<'ll>,
        Ty: Type<'ll>,
        Fn: Value<'ll>,
        Args: *const Value<'ll>,
        NumArgs: c_uint,
        Then: &'ll BasicBlock,
        Catch: &'ll BasicBlock,
        Bundles: *const &OperandBundle<'ll>,
        NumBundles: c_uint,
        Name: *const c_char,
    ) -> Value<'ll>;
    pub(crate) fn LLVMBuildCallBr<'ll>(
        B: &Builder<'ll>,
        Ty: Type<'ll>,
        Fn: Value<'ll>,
        DefaultDest: &'ll BasicBlock,
        IndirectDests: *const &'ll BasicBlock,
        NumIndirectDests: c_uint,
        Args: *const Value<'ll>,
        NumArgs: c_uint,
        Bundles: *const &OperandBundle<'ll>,
        NumBundles: c_uint,
        Name: *const c_char,
    ) -> Value<'ll>;
}

// FFI bindings for `DIBuilder` functions in the LLVM-C API.
// Try to keep these in the same order as in `llvm/include/llvm-c/DebugInfo.h`.
//
// FIXME(#134001): Audit all `Option` parameters, especially in lists, to check
// that they really are nullable on the C/C++ side. LLVM doesn't appear to
// actually document which ones are nullable.
unsafe extern "C" {
    pub(crate) fn LLVMCreateDIBuilder<'ll>(M: Module<'ll>) -> *mut DIBuilder<'ll>;
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
        Ctx: Context<'ll>,
        Line: c_uint,
        Column: c_uint,
        Scope: &'ll Metadata,
        InlinedAt: Option<&'ll Metadata>,
    ) -> &'ll Metadata;

    pub(crate) fn LLVMDIBuilderCreateSubroutineType<'ll>(
        Builder: &DIBuilder<'ll>,
        File: Option<&'ll Metadata>, // (ignored and has no effect)
        ParameterTypes: *const Option<&'ll Metadata>,
        NumParameterTypes: c_uint,
        Flags: DIFlags, // (default is `DIFlags::DIFlagZero`)
    ) -> &'ll Metadata;

    pub(crate) fn LLVMDIBuilderCreateEnumeratorOfArbitraryPrecision<'ll>(
        Builder: &DIBuilder<'ll>,
        Name: *const c_uchar, // See "PTR_LEN_STR".
        NameLen: size_t,
        SizeInBits: u64,
        Words: *const u64, // LLVM computes `NumWords = (SizeInBits + 63) / 64`.
        IsUnsigned: llvm::Bool,
    ) -> &'ll Metadata;

    pub(crate) fn LLVMDIBuilderCreateUnionType<'ll>(
        Builder: &DIBuilder<'ll>,
        Scope: Option<&'ll Metadata>,
        Name: *const c_uchar, // See "PTR_LEN_STR".
        NameLen: size_t,
        File: &'ll Metadata,
        LineNumber: c_uint,
        SizeInBits: u64,
        AlignInBits: u32,
        Flags: DIFlags,
        Elements: *const Option<&'ll Metadata>,
        NumElements: c_uint,
        RunTimeLang: c_uint, // (optional Objective-C runtime version; default is 0)
        UniqueId: *const c_uchar, // See "PTR_LEN_STR".
        UniqueIdLen: size_t,
    ) -> &'ll Metadata;

    pub(crate) fn LLVMDIBuilderCreateArrayType<'ll>(
        Builder: &DIBuilder<'ll>,
        Size: u64,
        Align: u32,
        Ty: &'ll Metadata,
        Subscripts: *const &'ll Metadata,
        NumSubscripts: c_uint,
    ) -> &'ll Metadata;

    pub(crate) fn LLVMDIBuilderCreateBasicType<'ll>(
        Builder: &DIBuilder<'ll>,
        Name: *const c_uchar, // See "PTR_LEN_STR".
        NameLen: size_t,
        SizeInBits: u64,
        Encoding: c_uint, // (`LLVMDWARFTypeEncoding`)
        Flags: DIFlags,   // (default is `DIFlags::DIFlagZero`)
    ) -> &'ll Metadata;

    pub(crate) fn LLVMDIBuilderCreatePointerType<'ll>(
        Builder: &DIBuilder<'ll>,
        PointeeTy: &'ll Metadata,
        SizeInBits: u64,
        AlignInBits: u32,
        AddressSpace: c_uint, // (optional DWARF address space; default is 0)
        Name: *const c_uchar, // See "PTR_LEN_STR".
        NameLen: size_t,
    ) -> &'ll Metadata;

    pub(crate) fn LLVMDIBuilderCreateStructType<'ll>(
        Builder: &DIBuilder<'ll>,
        Scope: Option<&'ll Metadata>,
        Name: *const c_uchar, // See "PTR_LEN_STR".
        NameLen: size_t,
        File: &'ll Metadata,
        LineNumber: c_uint,
        SizeInBits: u64,
        AlignInBits: u32,
        Flags: DIFlags,
        DerivedFrom: Option<&'ll Metadata>,
        Elements: *const Option<&'ll Metadata>,
        NumElements: c_uint,
        RunTimeLang: c_uint, // (optional Objective-C runtime version; default is 0)
        VTableHolder: Option<&'ll Metadata>,
        UniqueId: *const c_uchar, // See "PTR_LEN_STR".
        UniqueIdLen: size_t,
    ) -> &'ll Metadata;

    pub(crate) fn LLVMDIBuilderCreateMemberType<'ll>(
        Builder: &DIBuilder<'ll>,
        Scope: &'ll Metadata,
        Name: *const c_uchar, // See "PTR_LEN_STR".
        NameLen: size_t,
        File: &'ll Metadata,
        LineNo: c_uint,
        SizeInBits: u64,
        AlignInBits: u32,
        OffsetInBits: u64,
        Flags: DIFlags,
        Ty: &'ll Metadata,
    ) -> &'ll Metadata;

    pub(crate) fn LLVMDIBuilderCreateStaticMemberType<'ll>(
        Builder: &DIBuilder<'ll>,
        Scope: &'ll Metadata,
        Name: *const c_uchar, // See "PTR_LEN_STR".
        NameLen: size_t,
        File: &'ll Metadata,
        LineNumber: c_uint,
        Type: &'ll Metadata,
        Flags: DIFlags,
        ConstantVal: Option<Value<'ll>>,
        AlignInBits: u32,
    ) -> &'ll Metadata;

    /// Creates a "qualified type" in the C/C++ sense, by adding modifiers
    /// like `const` or `volatile`.
    pub(crate) fn LLVMDIBuilderCreateQualifiedType<'ll>(
        Builder: &DIBuilder<'ll>,
        Tag: c_uint, // (DWARF tag, e.g. `DW_TAG_const_type`)
        Type: &'ll Metadata,
    ) -> &'ll Metadata;

    pub(crate) fn LLVMDIBuilderCreateTypedef<'ll>(
        Builder: &DIBuilder<'ll>,
        Type: &'ll Metadata,
        Name: *const c_uchar, // See "PTR_LEN_STR".
        NameLen: size_t,
        File: &'ll Metadata,
        LineNo: c_uint,
        Scope: Option<&'ll Metadata>,
        AlignInBits: u32, // (optional; default is 0)
    ) -> &'ll Metadata;

    pub(crate) fn LLVMDIBuilderGetOrCreateSubrange<'ll>(
        Builder: &DIBuilder<'ll>,
        LowerBound: i64,
        Count: i64,
    ) -> &'ll Metadata;

    pub(crate) fn LLVMDIBuilderGetOrCreateArray<'ll>(
        Builder: &DIBuilder<'ll>,
        Data: *const Option<&'ll Metadata>,
        NumElements: size_t,
    ) -> &'ll Metadata;

    pub(crate) fn LLVMDIBuilderCreateExpression<'ll>(
        Builder: &DIBuilder<'ll>,
        Addr: *const u64,
        Length: size_t,
    ) -> &'ll Metadata;

    pub(crate) fn LLVMDIBuilderCreateGlobalVariableExpression<'ll>(
        Builder: &DIBuilder<'ll>,
        Scope: Option<&'ll Metadata>,
        Name: *const c_uchar, // See "PTR_LEN_STR".
        NameLen: size_t,
        Linkage: *const c_uchar, // See "PTR_LEN_STR".
        LinkLen: size_t,
        File: &'ll Metadata,
        LineNo: c_uint,
        Ty: &'ll Metadata,
        LocalToUnit: llvm::Bool,
        Expr: &'ll Metadata,
        Decl: Option<&'ll Metadata>,
        AlignInBits: u32,
    ) -> &'ll Metadata;

    pub(crate) fn LLVMDIBuilderInsertDeclareRecordAtEnd<'ll>(
        Builder: &DIBuilder<'ll>,
        Storage: Value<'ll>,
        VarInfo: &'ll Metadata,
        Expr: &'ll Metadata,
        DebugLoc: &'ll Metadata,
        Block: &'ll BasicBlock,
    ) -> &'ll DbgRecord;

    pub(crate) fn LLVMDIBuilderInsertDbgValueRecordAtEnd<'ll>(
        Builder: &DIBuilder<'ll>,
        Val: Value<'ll>,
        VarInfo: &'ll Metadata,
        Expr: &'ll Metadata,
        DebugLoc: &'ll Metadata,
        Block: &'ll BasicBlock,
    ) -> &'ll DbgRecord;

    pub(crate) fn LLVMDIBuilderCreateAutoVariable<'ll>(
        Builder: &DIBuilder<'ll>,
        Scope: &'ll Metadata,
        Name: *const c_uchar, // See "PTR_LEN_STR".
        NameLen: size_t,
        File: &'ll Metadata,
        LineNo: c_uint,
        Ty: &'ll Metadata,
        AlwaysPreserve: llvm::Bool, // "If true, this descriptor will survive optimizations."
        Flags: DIFlags,
        AlignInBits: u32,
    ) -> &'ll Metadata;

    pub(crate) fn LLVMDIBuilderCreateParameterVariable<'ll>(
        Builder: &DIBuilder<'ll>,
        Scope: &'ll Metadata,
        Name: *const c_uchar, // See "PTR_LEN_STR".
        NameLen: size_t,
        ArgNo: c_uint,
        File: &'ll Metadata,
        LineNo: c_uint,
        Ty: &'ll Metadata,
        AlwaysPreserve: llvm::Bool, // "If true, this descriptor will survive optimizations."
        Flags: DIFlags,
    ) -> &'ll Metadata;
}

#[link(name = "llvm-wrapper", kind = "static")]
unsafe extern "C" {
    pub(crate) fn LLVMRustInstallErrorHandlers();
    pub(crate) fn LLVMRustDisableSystemDialogsOnCrash();

    // Operations on all values
    pub(crate) fn LLVMRustIsNonGVFunctionPointerTy<'ll>(Val: Value<'ll>) -> bool;
    pub(crate) fn LLVMRustStripPointerCasts<'ll>(Val: Value<'ll>) -> Value<'ll>;

    // Operations on scalar constants
    pub(crate) fn LLVMRustConstIntGetZExtValue(ConstantVal: &ConstantInt, Value: &mut u64) -> bool;
    pub(crate) fn LLVMRustConstInt128Get(
        ConstantVal: &ConstantInt,
        SExt: bool,
        high: &mut u64,
        low: &mut u64,
    ) -> bool;

    // Operations on global variables, functions, and aliases (globals)
    pub(crate) fn LLVMRustSetDSOLocal<'ll>(Global: Value<'ll>, is_dso_local: bool);

    // Operations on global variables
    pub(crate) fn LLVMRustGetOrInsertGlobal<'ll>(
        M: Module<'ll>,
        Name: *const c_uchar, // See "PTR_LEN_STR".
        NameLen: size_t,
        T: Type<'ll>,
    ) -> Value<'ll>;
    pub(crate) fn LLVMRustGetOrInsertGlobalInAddrspace<'ll>(
        M: Module<'ll>,
        Name: *const c_uchar, // See "PTR_LEN_STR".
        NameLen: size_t,
        T: Type<'ll>,
        AddressSpace: c_uint,
    ) -> Value<'ll>;
    pub(crate) fn LLVMRustGetNamedValue<'ll>(
        M: Module<'ll>,
        Name: *const c_uchar, // See "PTR_LEN_STR".
        NameLen: size_t,
    ) -> Option<Value<'ll>>;

    // Operations on attributes
    pub(crate) fn LLVMRustCreateAttrNoValue<'ll>(
        C: Context<'ll>,
        attr: AttributeKind,
    ) -> &'ll Attribute;
    pub(crate) fn LLVMRustCreateAlignmentAttr<'ll>(C: Context<'ll>, bytes: u64) -> &'ll Attribute;
    pub(crate) fn LLVMRustCreateDereferenceableAttr<'ll>(
        C: Context<'ll>,
        bytes: u64,
    ) -> &'ll Attribute;
    pub(crate) fn LLVMRustCreateDereferenceableOrNullAttr<'ll>(
        C: Context<'ll>,
        bytes: u64,
    ) -> &'ll Attribute;
    pub(crate) fn LLVMRustCreateByValAttr<'ll>(C: Context<'ll>, ty: Type<'ll>) -> &'ll Attribute;
    pub(crate) fn LLVMRustCreateByRefAttr<'ll>(C: Context<'ll>, ty: Type<'ll>) -> &'ll Attribute;
    pub(crate) fn LLVMRustCreateStructRetAttr<'ll>(
        C: Context<'ll>,
        ty: Type<'ll>,
    ) -> &'ll Attribute;
    pub(crate) fn LLVMRustCreateElementTypeAttr<'ll>(
        C: Context<'ll>,
        ty: Type<'ll>,
    ) -> &'ll Attribute;
    pub(crate) fn LLVMRustCreateUWTableAttr<'ll>(C: Context<'ll>, async_: bool) -> &'ll Attribute;
    pub(crate) fn LLVMRustCreateAllocSizeAttr<'ll>(
        C: Context<'ll>,
        size_arg: u32,
    ) -> &'ll Attribute;
    pub(crate) fn LLVMRustCreateAllocKindAttr<'ll>(
        C: Context<'ll>,
        size_arg: u64,
    ) -> &'ll Attribute;
    pub(crate) fn LLVMRustCreateMemoryEffectsAttr<'ll>(
        C: Context<'ll>,
        effects: MemoryEffects,
    ) -> &'ll Attribute;
    /// ## Safety
    /// - Each of `LowerWords` and `UpperWords` must point to an array that is
    ///   long enough to fully define an integer of size `NumBits`, i.e. each
    ///   pointer must point to `NumBits.div_ceil(64)` elements or more.
    /// - The implementation will make its own copy of the pointed-to `u64`
    ///   values, so the pointers only need to outlive this function call.
    pub(crate) fn LLVMRustCreateRangeAttribute<'ll>(
        C: Context<'ll>,
        NumBits: c_uint,
        LowerWords: *const u64,
        UpperWords: *const u64,
    ) -> &'ll Attribute;

    // Operations on functions
    pub(crate) fn LLVMRustAddFunctionAttributes<'ll>(
        Fn: Value<'ll>,
        index: c_uint,
        Attrs: *const &'ll Attribute,
        AttrsLen: size_t,
    );

    // Operations on call sites
    pub(crate) fn LLVMRustAddCallSiteAttributes<'ll>(
        Instr: Value<'ll>,
        index: c_uint,
        Attrs: *const &'ll Attribute,
        AttrsLen: size_t,
    );

    pub(crate) fn LLVMRustSetFastMath<'ll>(Instr: Value<'ll>);
    pub(crate) fn LLVMRustSetAlgebraicMath<'ll>(Instr: Value<'ll>);
    pub(crate) fn LLVMRustSetAllowReassoc<'ll>(Instr: Value<'ll>);
    pub(crate) fn LLVMRustSetNoSignedZeros<'ll>(Instr: Value<'ll>);

    // Miscellaneous instructions
    pub(crate) fn LLVMRustBuildMemCpy<'ll>(
        B: &Builder<'ll>,
        Dst: Value<'ll>,
        DstAlign: c_uint,
        Src: Value<'ll>,
        SrcAlign: c_uint,
        Size: Value<'ll>,
        IsVolatile: bool,
    ) -> Value<'ll>;
    pub(crate) fn LLVMRustBuildMemMove<'ll>(
        B: &Builder<'ll>,
        Dst: Value<'ll>,
        DstAlign: c_uint,
        Src: Value<'ll>,
        SrcAlign: c_uint,
        Size: Value<'ll>,
        IsVolatile: bool,
    ) -> Value<'ll>;
    pub(crate) fn LLVMRustBuildMemSet<'ll>(
        B: &Builder<'ll>,
        Dst: Value<'ll>,
        DstAlign: c_uint,
        Val: Value<'ll>,
        Size: Value<'ll>,
        IsVolatile: bool,
    ) -> Value<'ll>;

    pub(crate) fn LLVMRustBuildVScale<'ll>(B: &Builder<'ll>, Ty: Type<'ll>) -> Value<'ll>;

    pub(crate) fn LLVMRustTimeTraceProfilerInitialize();

    pub(crate) fn LLVMRustTimeTraceProfilerFinishThread();

    pub(crate) fn LLVMRustTimeTraceProfilerFinish(FileName: *const c_char);

    /// Returns a string describing the last error caused by an LLVMRust* call.
    pub(crate) fn LLVMRustGetLastError() -> *const c_char;

    /// Prints the timing information collected by `-Ztime-llvm-passes`.
    pub(crate) fn LLVMRustPrintPassTimings(OutStr: &RustString);

    /// Prints the statistics collected by `-Zprint-codegen-stats`.
    pub(crate) fn LLVMRustPrintStatistics(OutStr: &RustString);

    /// Save the statistics collected by `-Zprint-codegen-stats-json`
    pub(crate) fn LLVMRustPrintStatisticsJSON(OutStr: &RustString);

    pub(crate) fn LLVMRustInlineAsmVerify<'ll>(
        Ty: Type<'ll>,
        Constraints: *const c_uchar, // See "PTR_LEN_STR".
        ConstraintsLen: size_t,
    ) -> bool;

    /// Append inline assembly to a module. See `Module::appendModuleInlineAsm`.
    pub(crate) fn LLVMRustAppendModuleInlineAsm<'ll>(
        M: Module<'ll>,
        Asm: *const c_uchar, // See "PTR_LEN_STR".
        AsmLen: size_t,
        TargetFeatures: *const c_uchar, // See "PTR_LEN_STR".
        TargetFeaturesLen: size_t,
        TargetCpu: *const c_uchar, // See "PTR_LEN_STR".
        TargetCpuLen: size_t,
    );

    /// A list of pointer-length strings is passed as two pointer-length slices,
    /// one slice containing pointers and one slice containing their corresponding
    /// lengths. The implementation will check that both slices have the same length.
    pub(crate) fn LLVMRustCoverageWriteFilenamesToBuffer(
        Filenames: *const *const c_uchar, // See "PTR_LEN_STR".
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
        BufferOut: &RustString,
    );

    pub(crate) fn LLVMRustCoverageCreatePGOFuncNameVar<'ll>(
        F: Value<'ll>,
        FuncName: *const c_uchar, // See "PTR_LEN_STR".
        FuncNameLen: size_t,
    ) -> Value<'ll>;
    pub(crate) fn LLVMRustCoverageHashBytes(
        Bytes: *const c_uchar, // See "PTR_LEN_STR".
        NumBytes: size_t,
    ) -> u64;

    pub(crate) safe fn LLVMRustCoverageWriteCovmapSectionNameToString<'ll>(
        M: Module<'ll>,
        OutStr: &RustString,
    );
    pub(crate) safe fn LLVMRustCoverageWriteCovfunSectionNameToString<'ll>(
        M: Module<'ll>,
        OutStr: &RustString,
    );
    pub(crate) safe fn LLVMRustCoverageWriteCovmapVarNameToString(OutStr: &RustString);

    pub(crate) safe fn LLVMRustCoverageMappingVersion() -> u32;
    pub(crate) fn LLVMRustDebugMetadataVersion() -> u32;

    /// Returns the LLVM major version that the compiler was built with.
    ///
    /// Note that this is hard-coded as `LLVM_VERSION_MAJOR` when `RustWrapper.cpp` is built. This
    /// could be different than what the runtime LLVM library reports in [`LLVMGetVersion`], so we
    /// assert their equality in `configure_llvm`.
    pub(crate) safe fn LLVMRustVersionMajor() -> u32;

    /// Add LLVM module flags.
    ///
    /// In order for Rust-C LTO to work, module flags must be compatible with Clang. What
    /// "compatible" means depends on the merge behaviors involved.
    pub(crate) fn LLVMRustAddModuleFlagU32<'ll>(
        M: Module<'ll>,
        MergeBehavior: ModuleFlagMergeBehavior,
        Name: *const c_uchar, // See "PTR_LEN_STR".
        NameLen: size_t,
        Value: u32,
    );

    pub(crate) fn LLVMRustAddModuleFlagString<'ll>(
        M: Module<'ll>,
        MergeBehavior: ModuleFlagMergeBehavior,
        Name: *const c_uchar, // See "PTR_LEN_STR".
        NameLen: size_t,
        Value: *const c_uchar, // See "PTR_LEN_STR".
        ValueLen: size_t,
    );

    pub(crate) fn LLVMRustAddModuleFlagMetadata<'ll>(
        M: Module<'ll>,
        MergeBehavior: ModuleFlagMergeBehavior,
        Name: *const c_uchar, // See "PTR_LEN_STR".
        NameLen: size_t,
        Value: &'ll Metadata,
    );

    /// We can't use LLVM-C's `LLVMDIBuilderCreateCompileUnit` because it hardcodes
    /// `DICompileUnit::DebugNameTableKind::Default`, but we want to be able to
    /// pass other values.
    pub(crate) fn LLVMRustDIBuilderCreateCompileUnit<'a>(
        Builder: &DIBuilder<'a>,
        Lang: c_uint,
        File: &'a DIFile,
        Producer: *const c_uchar, // See "PTR_LEN_STR".
        ProducerLen: size_t,
        isOptimized: bool,
        Flags: *const c_char,
        RuntimeVer: c_uint,
        SplitName: *const c_uchar, // See "PTR_LEN_STR".
        SplitNameLen: size_t,
        kind: DebugEmissionKind,
        DWOId: u64,
        SplitDebugInlining: bool,
        DebugInfoForProfiling: bool,
        DebugNameTableKind: DebugNameTableKind,
    ) -> &'a DIDescriptor;

    /// We can't use LLVM-C's `LLVMDIBuilderCreateFileWithChecksum` because it
    /// _requires_ a checksum, but we sometimes don't provide one.
    pub(crate) fn LLVMRustDIBuilderCreateFile<'a>(
        Builder: &DIBuilder<'a>,
        Filename: *const c_uchar, // See "PTR_LEN_STR".
        FilenameLen: size_t,
        Directory: *const c_uchar, // See "PTR_LEN_STR".
        DirectoryLen: size_t,
        CSKind: ChecksumKind,
        Checksum: *const c_uchar, // See "PTR_LEN_STR".
        ChecksumLen: size_t,
        Source: *const c_uchar, // See "PTR_LEN_STR".
        SourceLen: size_t,
    ) -> &'a DIFile;

    /// We can't use LLVM-C's `LLVMDIBuilderCreateFunction` because it only
    /// supports a subset of `DISubprogram::DISPFlags`.
    pub(crate) fn LLVMRustDIBuilderCreateFunction<'ll>(
        Builder: &DIBuilder<'ll>,
        Scope: &'ll DIDescriptor,
        Name: *const c_uchar, // See "PTR_LEN_STR".
        NameLen: size_t,
        LinkageName: *const c_uchar, // See "PTR_LEN_STR".
        LinkageNameLen: size_t,
        File: &'ll DIFile,
        LineNo: c_uint,
        Ty: &'ll DIType,
        ScopeLine: c_uint,
        Flags: DIFlags,
        SPFlags: DISPFlags,
        MaybeFn: Option<Value<'ll>>,
        TParam: &'ll DIArray,
        Decl: Option<&'ll DIDescriptor>,
    ) -> &'ll DISubprogram;

    /// As of LLVM 22 there is no corresponding LLVM-C function.
    pub(crate) fn LLVMRustDIBuilderCreateMethod<'a>(
        Builder: &DIBuilder<'a>,
        Scope: &'a DIDescriptor,
        Name: *const c_uchar, // See "PTR_LEN_STR".
        NameLen: size_t,
        LinkageName: *const c_uchar, // See "PTR_LEN_STR".
        LinkageNameLen: size_t,
        File: &'a DIFile,
        LineNo: c_uint,
        Ty: &'a DIType,
        Flags: DIFlags,
        SPFlags: DISPFlags,
        TParam: &'a DIArray,
    ) -> &'a DISubprogram;

    /// As of LLVM 22 there is no corresponding LLVM-C function.
    pub(crate) fn LLVMRustDIBuilderCreateVariantMemberType<'ll>(
        Builder: &DIBuilder<'ll>,
        Scope: &'ll DIScope,
        Name: *const c_uchar, // See "PTR_LEN_STR".
        NameLen: size_t,
        File: &'ll DIFile,
        LineNumber: c_uint,
        SizeInBits: u64,
        AlignInBits: u32,
        OffsetInBits: u64,
        Discriminant: Option<Value<'ll>>,
        Flags: DIFlags,
        Ty: &'ll DIType,
    ) -> &'ll DIType;

    /// As of LLVM 22 there is no corresponding LLVM-C function.
    pub(crate) fn LLVMRustDIBuilderCreateEnumerationType<'a>(
        Builder: &DIBuilder<'a>,
        Scope: &'a DIScope,
        Name: *const c_uchar, // See "PTR_LEN_STR".
        NameLen: size_t,
        File: &'a DIFile,
        LineNumber: c_uint,
        SizeInBits: u64,
        AlignInBits: u32,
        Elements: &'a DIArray,
        ClassType: &'a DIType,
        IsScoped: bool,
    ) -> &'a DIType;

    /// As of LLVM 22 there is no corresponding LLVM-C function.
    pub(crate) fn LLVMRustDIBuilderCreateVariantPart<'a>(
        Builder: &DIBuilder<'a>,
        Scope: &'a DIScope,
        Name: *const c_uchar, // See "PTR_LEN_STR".
        NameLen: size_t,
        File: &'a DIFile,
        LineNo: c_uint,
        SizeInBits: u64,
        AlignInBits: u32,
        Flags: DIFlags,
        Discriminator: Option<&'a DIDerivedType>,
        Elements: &'a DIArray,
        UniqueId: *const c_uchar, // See "PTR_LEN_STR".
        UniqueIdLen: size_t,
    ) -> &'a DIDerivedType;

    /// As of LLVM 22 there is no corresponding LLVM-C function.
    pub(crate) fn LLVMRustDIBuilderCreateTemplateTypeParameter<'a>(
        Builder: &DIBuilder<'a>,
        Scope: Option<&'a DIScope>,
        Name: *const c_uchar, // See "PTR_LEN_STR".
        NameLen: size_t,
        Ty: &'a DIType,
    ) -> &'a DITemplateTypeParameter;

    /// We can't use LLVM-C's `LLVMReplaceArrays` because it doesn't take a
    /// `Params` argument.
    pub(crate) fn LLVMRustDICompositeTypeReplaceArrays<'a>(
        Builder: &DIBuilder<'a>,
        CompositeType: &'a DIType,
        Elements: Option<&'a DIArray>,
        Params: Option<&'a DIArray>,
    );

    /// We can't use LLVM-C's `LLVMDIBuilderGetOrCreateSubrange` because it doesn't
    /// call the overload that takes a `Metadata` upper bound.
    pub(crate) fn LLVMRustDIGetOrCreateSubrange<'a>(
        Builder: &DIBuilder<'a>,
        CountNode: Option<&'a Metadata>,
        LB: &'a Metadata,
        UB: &'a Metadata,
        Stride: Option<&'a Metadata>,
    ) -> &'a Metadata;

    /// We can't use LLVM-C's `LLVMDIBuilderCreateVectorType` because it doesn't
    /// take a `BitStride` argument.
    pub(crate) fn LLVMRustDICreateVectorType<'a>(
        Builder: &DIBuilder<'a>,
        Size: u64,
        AlignInBits: u32,
        Type: &'a DIType,
        Subscripts: &'a DIArray,
        BitStride: Option<&'a Metadata>,
    ) -> &'a Metadata;

    /// As of LLVM 22 there is no corresponding LLVM-C function.
    pub(crate) fn LLVMRustDILocationCloneWithBaseDiscriminator<'a>(
        Location: &'a DILocation,
        BD: c_uint,
    ) -> Option<&'a DILocation>;

    pub(crate) fn LLVMRustWriteTypeToString<'ll>(Type: Type<'ll>, s: &RustString);
    pub(crate) fn LLVMRustWriteValueToString<'ll>(value_ref: Value<'ll>, s: &RustString);

    pub(crate) fn LLVMRustTargetHasMnemonic(T: &TargetMachine, s: *const c_char) -> bool;

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
        DebugInfoCompression: CompressionKind,
        UseEmulatedTls: bool,
        UseWasmEH: bool,
        LargeDataThreshold: u64,
    ) -> *mut TargetMachine;

    pub(crate) fn LLVMRustCreateMCSubtargetInfo(
        TripleStr: *const c_char,
        CPU: *const c_char,
        Features: *const c_char,
    ) -> *mut MCSubtargetInfo;

    pub(crate) fn LLVMRustMCSubtargetInfoCheckFeatures(
        MCInfo: &MCSubtargetInfo,
        Features: *const c_uchar, // See "PTR_LEN_STR".
        FeaturesLen: usize,
    ) -> bool;

    pub(crate) fn LLVMRustDisposeMCSubtargetInfo(MCInfo: ptr::NonNull<MCSubtargetInfo>);

    pub(crate) fn LLVMRustWriteOutputFile<'ll>(
        T: &'ll TargetMachine,
        M: Module<'ll>,
        Output: *const c_char,
        DwoOutput: *const c_char,
        FileType: FileType,
        VerifyIR: bool,
        DisableSimplifyLibCalls: bool,
    ) -> LLVMRustResult;
    pub(crate) fn LLVMRustOptimize<'ll>(
        M: Module<'ll>,
        TM: &'ll TargetMachine,
        OptLevel: PassBuilderOptLevel,
        OptStage: OptStage,
        IsLinkerPluginLTO: bool,
        NoPrepopulatePasses: bool,
        VerifyIR: bool,
        LintIR: bool,
        ThinLTOBuffer: Option<&mut Option<crate::back::lto::Buffer>>,
        ThinLTOSummaryBuffer: Option<&mut Option<crate::back::lto::Buffer>>,
        MergeFunctions: bool,
        UnrollLoops: bool,
        SLPVectorize: bool,
        LoopVectorize: bool,
        DisableSimplifyLibCalls: bool,
        EmitLifetimeMarkers: bool,
        RunEnzyme: *const c_void,
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
        PostEnzymePasses: *const c_uchar, // See "PTR_LEN_STR".
        PostEnzymePassesLen: size_t,
        ExtraPasses: *const c_uchar, // See "PTR_LEN_STR".
        ExtraPassesLen: size_t,
        LLVMPlugins: *const c_uchar, // See "PTR_LEN_STR".
        LLVMPluginsLen: size_t,
    ) -> LLVMRustResult;
    pub(crate) fn LLVMRustPrintModule<'ll>(
        M: Module<'ll>,
        Output: *const c_char,
        Demangle: extern "C" fn(*const c_char, size_t, *mut c_char, size_t) -> size_t,
    ) -> LLVMRustResult;
    pub(crate) fn LLVMRustSetLLVMOptions(Argc: c_int, Argv: *const *const c_char);
    pub(crate) fn LLVMRustPrintPasses();
    pub(crate) fn LLVMRustSetNormalizedTarget<'ll>(M: Module<'ll>, triple: *const c_char);
    pub(crate) fn LLVMRustRunRestrictionPass<'ll>(
        M: Module<'ll>,
        syms: *const *const c_char,
        len: size_t,
    );

    pub(crate) fn LLVMRustWriteTwineToString(T: &Twine, s: &RustString);

    pub(crate) fn LLVMRustUnpackOptimizationDiagnostic<'ll>(
        DI: &'ll DiagnosticInfo,
        pass_name_out: &RustString,
        function_out: &mut Option<Value<'ll>>,
        loc_line_out: &mut c_uint,
        loc_column_out: &mut c_uint,
        loc_filename_out: &RustString,
        message_out: &RustString,
    );

    pub(crate) fn LLVMRustUnpackInlineAsmDiagnostic<'a>(
        DI: &'a DiagnosticInfo,
        cookie_out: &mut u64,
        message_out: &mut Option<&'a Twine>,
    );

    pub(crate) fn LLVMRustWriteDiagnosticInfoToString(DI: &DiagnosticInfo, s: &RustString);
    pub(crate) fn LLVMRustGetDiagInfoKind(DI: &DiagnosticInfo) -> DiagnosticKind;
    pub(crate) fn LLVMGetDiagInfoSeverity(DI: &DiagnosticInfo) -> DiagnosticSeverity;

    pub(crate) fn LLVMRustGetSMDiagnostic<'a>(
        DI: &'a DiagnosticInfo,
        cookie_out: &mut u64,
    ) -> &'a SMDiagnostic;

    pub(crate) fn LLVMRustUnpackSMDiagnostic(
        d: &SMDiagnostic,
        message_out: &RustString,
        buffer_out: &RustString,
        loc_out: &mut c_uint,
        ranges_out: *mut c_uint,
        num_ranges: &mut usize,
    ) -> bool;

    pub(crate) fn LLVMRustSetDataLayoutFromTargetMachine<'ll>(M: Module<'ll>, TM: &TargetMachine);

    pub(crate) fn LLVMRustPositionBuilderPastAllocas<'ll>(B: &Builder<'ll>, Fn: Value<'ll>);
    pub(crate) fn LLVMRustPositionBuilderAtStart<'a>(B: &Builder<'a>, BB: &'a BasicBlock);

    pub(crate) fn LLVMRustSetModulePICLevel<'ll>(M: Module<'ll>);
    pub(crate) fn LLVMRustSetModulePIELevel<'ll>(M: Module<'ll>);
    pub(crate) fn LLVMRustSetModuleCodeModel<'ll>(M: Module<'ll>, Model: CodeModel);
    pub(crate) fn LLVMRustSetModuleLargeDataThreshold<'ll>(M: Module<'ll>, Threshold: u64);
    pub(crate) fn LLVMRustBufferPtr(p: &Buffer) -> *const u8;
    pub(crate) fn LLVMRustBufferLen(p: &Buffer) -> usize;
    pub(crate) fn LLVMRustBufferFree(p: &'static mut Buffer);
    pub(crate) fn LLVMRustModuleCost<'ll>(M: Module<'ll>) -> u64;
    pub(crate) fn LLVMRustModuleInstructionStats<'ll>(M: Module<'ll>) -> u64;

    pub(crate) fn LLVMRustModuleSerialize<'ll>(
        M: Module<'ll>,
        is_thin: bool,
    ) -> &'static mut Buffer;
    pub(crate) fn LLVMRustCreateThinLTOData(
        Modules: *const ThinLTOModule,
        NumModules: size_t,
        PreservedSymbols: *const *const c_char,
        PreservedSymbolsLen: size_t,
    ) -> Option<&'static mut ThinLTOData>;
    pub(crate) fn LLVMRustPrepareThinLTORename<'ll>(
        Data: &ThinLTOData,
        Module: Module<'ll>,
        Target: &TargetMachine,
    );
    pub(crate) fn LLVMRustPrepareThinLTOResolveWeak<'ll>(
        Data: &ThinLTOData,
        Module: Module<'ll>,
    ) -> bool;
    pub(crate) fn LLVMRustPrepareThinLTOInternalize<'ll>(
        Data: &ThinLTOData,
        Module: Module<'ll>,
    ) -> bool;
    pub(crate) fn LLVMRustPrepareThinLTOImport<'ll>(
        Data: &ThinLTOData,
        Module: Module<'ll>,
        Target: &TargetMachine,
    ) -> bool;
    pub(crate) fn LLVMRustFreeThinLTOData(Data: &'static mut ThinLTOData);
    pub(crate) fn LLVMRustParseBitcodeForLTO<'ll>(
        Context: Context<'ll>,
        Data: *const u8,
        len: usize,
        Identifier: *const c_char,
    ) -> Option<Module<'ll>>;

    pub(crate) fn LLVMRustLinkerNew<'ll>(M: Module<'ll>) -> &'ll mut Linker<'ll>;
    pub(crate) fn LLVMRustLinkerAdd(
        linker: &Linker<'_>,
        bytecode: *const c_uchar, // See "PTR_LEN_STR".
        bytecode_len: usize,
    ) -> bool;
    pub(crate) fn LLVMRustLinkerFree<'a>(linker: &'a mut Linker<'a>);
    pub(crate) fn LLVMRustComputeLTOCacheKey(
        key_out: &RustString,
        mod_id: *const c_char,
        data: &ThinLTOData,
    );

    /// Removes the currently-installed `DiagnosticHandler`, and returns an owning pointer to it.
    pub(crate) fn LLVMRustContextTakeDiagnosticHandler<'ll>(
        Context: Context<'ll>,
    ) -> Option<ptr::NonNull<DiagnosticHandler>>;

    /// Installs a diagnostic handler that was previously removed by
    /// [`LLVMRustContextTakeDiagnosticHandler`].
    pub(crate) fn LLVMRustContextSetDiagnosticHandler<'ll>(
        context: Context<'ll>,
        diagnostic_handler: Option<ptr::NonNull<DiagnosticHandler>>,
    );

    /// Installs a custom diagnostic handler.
    pub(crate) fn LLVMRustContextConfigureDiagnosticHandler<'ll>(
        context: Context<'ll>,
        diagnostic_handler_callback: DiagnosticHandlerCallback,
        diagnostic_handler_callback_data: *mut c_void,
        remark_all_passes: bool,
        remark_passes: *const *const c_char,
        remark_passes_len: usize,
        remark_file: *const c_char,
        pgo_available: bool,
    );

    pub(crate) fn LLVMRustGetMangledName<'ll>(V: Value<'ll>, out: &RustString);

    pub(crate) fn LLVMRustGetElementTypeArgIndex<'ll>(CallSite: Value<'ll>) -> i32;

    pub(crate) safe fn LLVMRustLLVMHasZlibCompression() -> bool;
    pub(crate) safe fn LLVMRustLLVMHasZstdCompression() -> bool;

    pub(crate) fn LLVMRustGetSymbols(
        buf_ptr: *const u8,
        buf_len: usize,
        state: *mut c_void,
        callback: GetSymbolsCallback,
        error_callback: GetSymbolsErrorCallback,
    ) -> *mut c_void;

    pub(crate) fn LLVMRustIs64BitSymbolicFile(buf_ptr: *const u8, buf_len: usize) -> bool;

    pub(crate) fn LLVMRustIsECObject(buf_ptr: *const u8, buf_len: usize) -> bool;

    pub(crate) fn LLVMRustIsAnyArm64Coff(buf_ptr: *const u8, buf_len: usize) -> bool;

    pub(crate) fn LLVMRustSetNoSanitizeAddress<'ll>(Global: Value<'ll>);
    pub(crate) fn LLVMRustSetNoSanitizeHWAddress<'ll>(Global: Value<'ll>);

    pub(crate) fn LLVMRustConstPtrAuth<'ll>(
        ptr: Value<'ll>,
        key: u32,
        disc: u64,
        addr_diversity: Option<Value<'ll>>,
        deactivation_symbol: Option<Value<'ll>>,
    ) -> Value<'ll>;
}
