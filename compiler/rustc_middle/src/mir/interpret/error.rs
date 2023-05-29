use super::{AllocId, AllocRange, ConstAlloc, Pointer, Scalar};

use crate::mir::interpret::ConstValue;
use crate::query::TyCtxtAt;
use crate::ty::{layout, tls, Ty, ValTree};

use rustc_data_structures::sync::Lock;
use rustc_errors::{
    struct_span_err, DiagnosticArgValue, DiagnosticBuilder, DiagnosticMessage, ErrorGuaranteed,
    IntoDiagnosticArg,
};
use rustc_macros::HashStable;
use rustc_session::CtfeBacktrace;
use rustc_span::def_id::DefId;
use rustc_target::abi::{call, Align, Size, WrappingRange};
use std::borrow::Cow;
use std::{any::Any, backtrace::Backtrace, fmt};

#[derive(Debug, Copy, Clone, PartialEq, Eq, HashStable, TyEncodable, TyDecodable)]
pub enum ErrorHandled {
    /// Already reported an error for this evaluation, and the compilation is
    /// *guaranteed* to fail. Warnings/lints *must not* produce `Reported`.
    Reported(ReportedErrorInfo),
    /// Don't emit an error, the evaluation failed because the MIR was generic
    /// and the substs didn't fully monomorphize it.
    TooGeneric,
}

impl From<ErrorGuaranteed> for ErrorHandled {
    #[inline]
    fn from(error: ErrorGuaranteed) -> ErrorHandled {
        ErrorHandled::Reported(error.into())
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, HashStable, TyEncodable, TyDecodable)]
pub struct ReportedErrorInfo {
    error: ErrorGuaranteed,
    is_tainted_by_errors: bool,
}

impl ReportedErrorInfo {
    #[inline]
    pub fn tainted_by_errors(error: ErrorGuaranteed) -> ReportedErrorInfo {
        ReportedErrorInfo { is_tainted_by_errors: true, error }
    }

    /// Returns true if evaluation failed because MIR was tainted by errors.
    #[inline]
    pub fn is_tainted_by_errors(self) -> bool {
        self.is_tainted_by_errors
    }
}

impl From<ErrorGuaranteed> for ReportedErrorInfo {
    #[inline]
    fn from(error: ErrorGuaranteed) -> ReportedErrorInfo {
        ReportedErrorInfo { is_tainted_by_errors: false, error }
    }
}

impl Into<ErrorGuaranteed> for ReportedErrorInfo {
    #[inline]
    fn into(self) -> ErrorGuaranteed {
        self.error
    }
}

TrivialTypeTraversalAndLiftImpls! {
    ErrorHandled,
}

pub type EvalToAllocationRawResult<'tcx> = Result<ConstAlloc<'tcx>, ErrorHandled>;
pub type EvalToConstValueResult<'tcx> = Result<ConstValue<'tcx>, ErrorHandled>;
pub type EvalToValTreeResult<'tcx> = Result<Option<ValTree<'tcx>>, ErrorHandled>;

pub fn struct_error<'tcx>(
    tcx: TyCtxtAt<'tcx>,
    msg: &str,
) -> DiagnosticBuilder<'tcx, ErrorGuaranteed> {
    struct_span_err!(tcx.sess, tcx.span, E0080, "{}", msg)
}

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
static_assert_size!(InterpErrorInfo<'_>, 8);

/// Packages the kind of error we got from the const code interpreter
/// up with a Rust-level backtrace of where the error occurred.
/// These should always be constructed by calling `.into()` on
/// an `InterpError`. In `rustc_mir::interpret`, we have `throw_err_*`
/// macros for this.
#[derive(Debug)]
pub struct InterpErrorInfo<'tcx>(Box<InterpErrorInfoInner<'tcx>>);

#[derive(Debug)]
struct InterpErrorInfoInner<'tcx> {
    kind: InterpError<'tcx>,
    backtrace: InterpErrorBacktrace,
}

#[derive(Debug)]
pub struct InterpErrorBacktrace {
    backtrace: Option<Box<Backtrace>>,
}

impl InterpErrorBacktrace {
    pub fn new() -> InterpErrorBacktrace {
        let capture_backtrace = tls::with_opt(|tcx| {
            if let Some(tcx) = tcx {
                *Lock::borrow(&tcx.sess.ctfe_backtrace)
            } else {
                CtfeBacktrace::Disabled
            }
        });

        let backtrace = match capture_backtrace {
            CtfeBacktrace::Disabled => None,
            CtfeBacktrace::Capture => Some(Box::new(Backtrace::force_capture())),
            CtfeBacktrace::Immediate => {
                // Print it now.
                let backtrace = Backtrace::force_capture();
                print_backtrace(&backtrace);
                None
            }
        };

        InterpErrorBacktrace { backtrace }
    }

    pub fn print_backtrace(&self) {
        if let Some(backtrace) = self.backtrace.as_ref() {
            print_backtrace(backtrace);
        }
    }
}

impl<'tcx> InterpErrorInfo<'tcx> {
    pub fn from_parts(kind: InterpError<'tcx>, backtrace: InterpErrorBacktrace) -> Self {
        Self(Box::new(InterpErrorInfoInner { kind, backtrace }))
    }

    pub fn into_parts(self) -> (InterpError<'tcx>, InterpErrorBacktrace) {
        let InterpErrorInfo(box InterpErrorInfoInner { kind, backtrace }) = self;
        (kind, backtrace)
    }

    pub fn into_kind(self) -> InterpError<'tcx> {
        let InterpErrorInfo(box InterpErrorInfoInner { kind, .. }) = self;
        kind
    }

    #[inline]
    pub fn kind(&self) -> &InterpError<'tcx> {
        &self.0.kind
    }
}

fn print_backtrace(backtrace: &Backtrace) {
    eprintln!("\n\nAn error occurred in miri:\n{}", backtrace);
}

impl From<ErrorGuaranteed> for InterpErrorInfo<'_> {
    fn from(err: ErrorGuaranteed) -> Self {
        InterpError::InvalidProgram(InvalidProgramInfo::AlreadyReported(err.into())).into()
    }
}

impl<'tcx> From<InterpError<'tcx>> for InterpErrorInfo<'tcx> {
    fn from(kind: InterpError<'tcx>) -> Self {
        InterpErrorInfo(Box::new(InterpErrorInfoInner {
            kind,
            backtrace: InterpErrorBacktrace::new(),
        }))
    }
}

/// Error information for when the program we executed turned out not to actually be a valid
/// program. This cannot happen in stand-alone Miri, but it can happen during CTFE/ConstProp
/// where we work on generic code or execution does not have all information available.
#[derive(Debug)]
pub enum InvalidProgramInfo<'tcx> {
    /// Resolution can fail if we are in a too generic context.
    TooGeneric,
    /// Abort in case errors are already reported.
    AlreadyReported(ReportedErrorInfo),
    /// An error occurred during layout computation.
    Layout(layout::LayoutError<'tcx>),
    /// An error occurred during FnAbi computation: the passed --target lacks FFI support
    /// (which unfortunately typeck does not reject).
    /// Not using `FnAbiError` as that contains a nested `LayoutError`.
    FnAbiAdjustForForeignAbi(call::AdjustForForeignAbiError),
    /// SizeOf of unsized type was requested.
    SizeOfUnsizedType(Ty<'tcx>),
    /// An unsized local was accessed without having been initialized.
    /// This is not meaningful as we can't even have backing memory for such locals.
    UninitUnsizedLocal,
}

/// Details of why a pointer had to be in-bounds.
#[derive(Debug, Copy, Clone, TyEncodable, TyDecodable, HashStable)]
pub enum CheckInAllocMsg {
    /// We are dereferencing a pointer (i.e., creating a place).
    DerefTest,
    /// We are access memory.
    MemoryAccessTest,
    /// We are doing pointer arithmetic.
    PointerArithmeticTest,
    /// We are doing pointer offset_from.
    OffsetFromTest,
    /// None of the above -- generic/unspecific inbounds test.
    InboundsTest,
}

#[derive(Debug, Copy, Clone, TyEncodable, TyDecodable, HashStable)]
pub enum InvalidMetaKind {
    /// Size of a `[T]` is too big
    SliceTooBig,
    /// Size of a DST is too big
    TooBig,
}

impl IntoDiagnosticArg for InvalidMetaKind {
    fn into_diagnostic_arg(self) -> DiagnosticArgValue<'static> {
        DiagnosticArgValue::Str(Cow::Borrowed(match self {
            InvalidMetaKind::SliceTooBig => "slice_too_big",
            InvalidMetaKind::TooBig => "too_big",
        }))
    }
}

/// Details of an access to uninitialized bytes where it is not allowed.
#[derive(Debug, Clone, Copy)]
pub struct UninitBytesAccess {
    /// Range of the original memory access.
    pub access: AllocRange,
    /// Range of the uninit memory that was encountered. (Might not be maximal.)
    pub uninit: AllocRange,
}

/// Information about a size mismatch.
#[derive(Debug)]
pub struct ScalarSizeMismatch {
    pub target_size: u64,
    pub data_size: u64,
}

macro_rules! impl_into_diagnostic_arg_through_debug {
    ($($ty:ty),*$(,)?) => {$(
        impl IntoDiagnosticArg for $ty {
            fn into_diagnostic_arg(self) -> DiagnosticArgValue<'static> {
                DiagnosticArgValue::Str(Cow::Owned(format!("{self:?}")))
            }
        }
    )*}
}

// These types have nice `Debug` output so we can just use them in diagnostics.
impl_into_diagnostic_arg_through_debug! {
    AllocId,
    Pointer,
    AllocRange,
}

/// Error information for when the program caused Undefined Behavior.
#[derive(Debug)]
pub enum UndefinedBehaviorInfo<'a> {
    /// Free-form case. Only for errors that are never caught! Used by miri
    Ub(String),
    /// Unreachable code was executed.
    Unreachable,
    /// A slice/array index projection went out-of-bounds.
    BoundsCheckFailed { len: u64, index: u64 },
    /// Something was divided by 0 (x / 0).
    DivisionByZero,
    /// Something was "remainded" by 0 (x % 0).
    RemainderByZero,
    /// Signed division overflowed (INT_MIN / -1).
    DivisionOverflow,
    /// Signed remainder overflowed (INT_MIN % -1).
    RemainderOverflow,
    /// Overflowing inbounds pointer arithmetic.
    PointerArithOverflow,
    /// Invalid metadata in a wide pointer
    InvalidMeta(InvalidMetaKind),
    /// Reading a C string that does not end within its allocation.
    UnterminatedCString(Pointer),
    /// Dereferencing a dangling pointer after it got freed.
    PointerUseAfterFree(AllocId),
    /// Used a pointer outside the bounds it is valid for.
    /// (If `ptr_size > 0`, determines the size of the memory range that was expected to be in-bounds.)
    PointerOutOfBounds {
        alloc_id: AllocId,
        alloc_size: Size,
        ptr_offset: i64,
        ptr_size: Size,
        msg: CheckInAllocMsg,
    },
    /// Using an integer as a pointer in the wrong way.
    DanglingIntPointer(u64, CheckInAllocMsg),
    /// Used a pointer with bad alignment.
    AlignmentCheckFailed { required: Align, has: Align },
    /// Writing to read-only memory.
    WriteToReadOnly(AllocId),
    /// Trying to access the data behind a function pointer.
    DerefFunctionPointer(AllocId),
    /// Trying to access the data behind a vtable pointer.
    DerefVTablePointer(AllocId),
    /// Using a non-boolean `u8` as bool.
    InvalidBool(u8),
    /// Using a non-character `u32` as character.
    InvalidChar(u32),
    /// The tag of an enum does not encode an actual discriminant.
    InvalidTag(Scalar),
    /// Using a pointer-not-to-a-function as function pointer.
    InvalidFunctionPointer(Pointer),
    /// Using a pointer-not-to-a-vtable as vtable pointer.
    InvalidVTablePointer(Pointer),
    /// Using a string that is not valid UTF-8,
    InvalidStr(std::str::Utf8Error),
    /// Using uninitialized data where it is not allowed.
    InvalidUninitBytes(Option<(AllocId, UninitBytesAccess)>),
    /// Working with a local that is not currently live.
    DeadLocal,
    /// Data size is not equal to target size.
    ScalarSizeMismatch(ScalarSizeMismatch),
    /// A discriminant of an uninhabited enum variant is written.
    UninhabitedEnumVariantWritten,
    /// Validation error.
    Validation(ValidationErrorInfo<'a>),
    // FIXME(fee1-dead) these should all be actual variants of the enum instead of dynamically
    // dispatched
    /// A custom (free-form) error, created by `err_ub_custom!`.
    Custom(crate::error::CustomSubdiagnostic<'a>),
}

#[derive(Debug, Clone, Copy)]
pub enum PointerKind {
    Ref,
    Box,
}

impl IntoDiagnosticArg for PointerKind {
    fn into_diagnostic_arg(self) -> DiagnosticArgValue<'static> {
        DiagnosticArgValue::Str(
            match self {
                Self::Ref => "ref",
                Self::Box => "box",
            }
            .into(),
        )
    }
}

#[derive(Debug)]
pub struct ValidationErrorInfo<'tcx> {
    pub path: Option<String>,
    pub kind: ValidationErrorKind<'tcx>,
}

#[derive(Debug)]
pub enum ExpectedKind {
    Reference,
    Box,
    RawPtr,
    InitScalar,
    Bool,
    Char,
    Float,
    Int,
    FnPtr,
}

impl From<PointerKind> for ExpectedKind {
    fn from(x: PointerKind) -> ExpectedKind {
        match x {
            PointerKind::Box => ExpectedKind::Box,
            PointerKind::Ref => ExpectedKind::Reference,
        }
    }
}

#[derive(Debug)]
pub enum ValidationErrorKind<'tcx> {
    PtrToUninhabited { ptr_kind: PointerKind, ty: Ty<'tcx> },
    PtrToStatic { ptr_kind: PointerKind },
    PtrToMut { ptr_kind: PointerKind },
    ExpectedNonPtr { value: String },
    MutableRefInConst,
    NullFnPtr,
    NeverVal,
    NullablePtrOutOfRange { range: WrappingRange, max_value: u128 },
    PtrOutOfRange { range: WrappingRange, max_value: u128 },
    OutOfRange { value: String, range: WrappingRange, max_value: u128 },
    UnsafeCell,
    UninhabitedVal { ty: Ty<'tcx> },
    InvalidEnumTag { value: String },
    UninitEnumTag,
    UninitStr,
    Uninit { expected: ExpectedKind },
    UninitVal,
    InvalidVTablePtr { value: String },
    InvalidMetaSliceTooLarge { ptr_kind: PointerKind },
    InvalidMetaTooLarge { ptr_kind: PointerKind },
    UnalignedPtr { ptr_kind: PointerKind, required_bytes: u64, found_bytes: u64 },
    NullPtr { ptr_kind: PointerKind },
    DanglingPtrNoProvenance { ptr_kind: PointerKind, pointer: String },
    DanglingPtrOutOfBounds { ptr_kind: PointerKind },
    DanglingPtrUseAfterFree { ptr_kind: PointerKind },
    InvalidBool { value: String },
    InvalidChar { value: String },
    InvalidFnPtr { value: String },
}

/// Error information for when the program did something that might (or might not) be correct
/// to do according to the Rust spec, but due to limitations in the interpreter, the
/// operation could not be carried out. These limitations can differ between CTFE and the
/// Miri engine, e.g., CTFE does not support dereferencing pointers at integral addresses.
#[derive(Debug)]
pub enum UnsupportedOpInfo {
    /// Free-form case. Only for errors that are never caught!
    // FIXME still use translatable diagnostics
    Unsupported(String),
    //
    // The variants below are only reachable from CTFE/const prop, miri will never emit them.
    //
    /// Overwriting parts of a pointer; without knowing absolute addresses, the resulting state
    /// cannot be represented by the CTFE interpreter.
    PartialPointerOverwrite(Pointer<AllocId>),
    /// Attempting to `copy` parts of a pointer to somewhere else; without knowing absolute
    /// addresses, the resulting state cannot be represented by the CTFE interpreter.
    PartialPointerCopy(Pointer<AllocId>),
    /// Encountered a pointer where we needed raw bytes.
    ReadPointerAsBytes,
    /// Accessing thread local statics
    ThreadLocalStatic(DefId),
    /// Accessing an unsupported extern static.
    ReadExternStatic(DefId),
}

/// Error information for when the program exhausted the resources granted to it
/// by the interpreter.
#[derive(Debug)]
pub enum ResourceExhaustionInfo {
    /// The stack grew too big.
    StackFrameLimitReached,
    /// There is not enough memory (on the host) to perform an allocation.
    MemoryExhausted,
    /// The address space (of the target) is full.
    AddressSpaceFull,
}

/// A trait for machine-specific errors (or other "machine stop" conditions).
pub trait MachineStopType: Any + fmt::Debug + Send {
    /// The diagnostic message for this error
    fn diagnostic_message(&self) -> DiagnosticMessage;
    /// Add diagnostic arguments by passing name and value pairs to `adder`, which are passed to
    /// fluent for formatting the translated diagnostic message.
    fn add_args(
        self: Box<Self>,
        adder: &mut dyn FnMut(Cow<'static, str>, DiagnosticArgValue<'static>),
    );
}

impl dyn MachineStopType {
    #[inline(always)]
    pub fn downcast_ref<T: Any>(&self) -> Option<&T> {
        let x: &dyn Any = self;
        x.downcast_ref()
    }
}

#[derive(Debug)]
pub enum InterpError<'tcx> {
    /// The program caused undefined behavior.
    UndefinedBehavior(UndefinedBehaviorInfo<'tcx>),
    /// The program did something the interpreter does not support (some of these *might* be UB
    /// but the interpreter is not sure).
    Unsupported(UnsupportedOpInfo),
    /// The program was invalid (ill-typed, bad MIR, not sufficiently monomorphized, ...).
    InvalidProgram(InvalidProgramInfo<'tcx>),
    /// The program exhausted the interpreter's resources (stack/heap too big,
    /// execution takes too long, ...).
    ResourceExhaustion(ResourceExhaustionInfo),
    /// Stop execution for a machine-controlled reason. This is never raised by
    /// the core engine itself.
    MachineStop(Box<dyn MachineStopType>),
}

pub type InterpResult<'tcx, T = ()> = Result<T, InterpErrorInfo<'tcx>>;

impl InterpError<'_> {
    /// Some errors do string formatting even if the error is never printed.
    /// To avoid performance issues, there are places where we want to be sure to never raise these formatting errors,
    /// so this method lets us detect them and `bug!` on unexpected errors.
    pub fn formatted_string(&self) -> bool {
        matches!(
            self,
            InterpError::Unsupported(UnsupportedOpInfo::Unsupported(_))
                | InterpError::UndefinedBehavior(UndefinedBehaviorInfo::Validation { .. })
                | InterpError::UndefinedBehavior(UndefinedBehaviorInfo::Ub(_))
        )
    }
}
