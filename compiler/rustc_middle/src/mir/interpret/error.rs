use std::any::Any;
use std::backtrace::Backtrace;
use std::borrow::Cow;
use std::{convert, fmt, mem, ops};

use either::Either;
use rustc_abi::{Align, Size, VariantIdx, WrappingRange};
use rustc_data_structures::sync::Lock;
use rustc_errors::{DiagArgName, DiagArgValue, DiagMessage, ErrorGuaranteed, IntoDiagArg};
use rustc_macros::{HashStable, TyDecodable, TyEncodable};
use rustc_session::CtfeBacktrace;
use rustc_span::def_id::DefId;
use rustc_span::{DUMMY_SP, Span, Symbol};

use super::{AllocId, AllocRange, ConstAllocation, Pointer, Scalar};
use crate::error;
use crate::mir::{ConstAlloc, ConstValue};
use crate::ty::{self, Mutability, Ty, TyCtxt, ValTree, layout, tls};

#[derive(Debug, Copy, Clone, PartialEq, Eq, HashStable, TyEncodable, TyDecodable)]
pub enum ErrorHandled {
    /// Already reported an error for this evaluation, and the compilation is
    /// *guaranteed* to fail. Warnings/lints *must not* produce `Reported`.
    Reported(ReportedErrorInfo, Span),
    /// Don't emit an error, the evaluation failed because the MIR was generic
    /// and the args didn't fully monomorphize it.
    TooGeneric(Span),
}

impl From<ReportedErrorInfo> for ErrorHandled {
    #[inline]
    fn from(error: ReportedErrorInfo) -> ErrorHandled {
        ErrorHandled::Reported(error, DUMMY_SP)
    }
}

impl ErrorHandled {
    pub(crate) fn with_span(self, span: Span) -> Self {
        match self {
            ErrorHandled::Reported(err, _span) => ErrorHandled::Reported(err, span),
            ErrorHandled::TooGeneric(_span) => ErrorHandled::TooGeneric(span),
        }
    }

    pub fn emit_note(&self, tcx: TyCtxt<'_>) {
        match self {
            &ErrorHandled::Reported(err, span) => {
                if !err.allowed_in_infallible && !span.is_dummy() {
                    tcx.dcx().emit_note(error::ErroneousConstant { span });
                }
            }
            &ErrorHandled::TooGeneric(_) => {}
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, HashStable, TyEncodable, TyDecodable)]
pub struct ReportedErrorInfo {
    error: ErrorGuaranteed,
    /// Whether this error is allowed to show up even in otherwise "infallible" promoteds.
    /// This is for things like overflows during size computation or resource exhaustion.
    allowed_in_infallible: bool,
}

impl ReportedErrorInfo {
    #[inline]
    pub fn const_eval_error(error: ErrorGuaranteed) -> ReportedErrorInfo {
        ReportedErrorInfo { allowed_in_infallible: false, error }
    }

    /// Use this when the error that led to this is *not* a const-eval error
    /// (e.g., a layout or type checking error).
    #[inline]
    pub fn non_const_eval_error(error: ErrorGuaranteed) -> ReportedErrorInfo {
        ReportedErrorInfo { allowed_in_infallible: true, error }
    }

    /// Use this when the error that led to this *is* a const-eval error, but
    /// we do allow it to occur in infallible constants (e.g., resource exhaustion).
    #[inline]
    pub fn allowed_in_infallible(error: ErrorGuaranteed) -> ReportedErrorInfo {
        ReportedErrorInfo { allowed_in_infallible: true, error }
    }

    pub fn is_allowed_in_infallible(&self) -> bool {
        self.allowed_in_infallible
    }
}

impl From<ReportedErrorInfo> for ErrorGuaranteed {
    #[inline]
    fn from(val: ReportedErrorInfo) -> Self {
        val.error
    }
}

/// An error type for the `const_to_valtree` query. Some error should be reported with a "use-site span",
/// which means the query cannot emit the error, so those errors are represented as dedicated variants here.
#[derive(Debug, Copy, Clone, PartialEq, Eq, HashStable, TyEncodable, TyDecodable)]
pub enum ValTreeCreationError<'tcx> {
    /// The constant is too big to be valtree'd.
    NodesOverflow,
    /// The constant references mutable or external memory, so it cannot be valtree'd.
    InvalidConst,
    /// Values of this type, or this particular value, are not supported as valtrees.
    NonSupportedType(Ty<'tcx>),
    /// The error has already been handled by const evaluation.
    ErrorHandled(ErrorHandled),
}

impl<'tcx> From<ErrorHandled> for ValTreeCreationError<'tcx> {
    fn from(err: ErrorHandled) -> Self {
        ValTreeCreationError::ErrorHandled(err)
    }
}

impl<'tcx> From<InterpErrorInfo<'tcx>> for ValTreeCreationError<'tcx> {
    fn from(err: InterpErrorInfo<'tcx>) -> Self {
        // An error occurred outside the const-eval query, as part of constructing the valtree. We
        // don't currently preserve the details of this error, since `InterpErrorInfo` cannot be put
        // into a query result and it can only be access of some mutable or external memory.
        let (_kind, backtrace) = err.into_parts();
        backtrace.print_backtrace();
        ValTreeCreationError::InvalidConst
    }
}

impl<'tcx> ValTreeCreationError<'tcx> {
    pub(crate) fn with_span(self, span: Span) -> Self {
        use ValTreeCreationError::*;
        match self {
            ErrorHandled(handled) => ErrorHandled(handled.with_span(span)),
            other => other,
        }
    }
}

pub type EvalToAllocationRawResult<'tcx> = Result<ConstAlloc<'tcx>, ErrorHandled>;
pub type EvalStaticInitializerRawResult<'tcx> = Result<ConstAllocation<'tcx>, ErrorHandled>;
pub type EvalToConstValueResult<'tcx> = Result<ConstValue, ErrorHandled>;
pub type EvalToValTreeResult<'tcx> = Result<ValTree<'tcx>, ValTreeCreationError<'tcx>>;

#[cfg(target_pointer_width = "64")]
rustc_data_structures::static_assert_size!(InterpErrorInfo<'_>, 8);

/// Packages the kind of error we got from the const code interpreter
/// up with a Rust-level backtrace of where the error occurred.
/// These should always be constructed by calling `.into()` on
/// an `InterpError`. In `rustc_mir::interpret`, we have `throw_err_*`
/// macros for this.
///
/// Interpreter errors must *not* be silently discarded (that will lead to a panic). Instead,
/// explicitly call `discard_err` if this is really the right thing to do. Note that if
/// this happens during const-eval or in Miri, it could lead to a UB error being lost!
#[derive(Debug)]
pub struct InterpErrorInfo<'tcx>(Box<InterpErrorInfoInner<'tcx>>);

#[derive(Debug)]
struct InterpErrorInfoInner<'tcx> {
    kind: InterpErrorKind<'tcx>,
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
    pub fn into_parts(self) -> (InterpErrorKind<'tcx>, InterpErrorBacktrace) {
        let InterpErrorInfo(box InterpErrorInfoInner { kind, backtrace }) = self;
        (kind, backtrace)
    }

    pub fn into_kind(self) -> InterpErrorKind<'tcx> {
        self.0.kind
    }

    pub fn from_parts(kind: InterpErrorKind<'tcx>, backtrace: InterpErrorBacktrace) -> Self {
        Self(Box::new(InterpErrorInfoInner { kind, backtrace }))
    }

    #[inline]
    pub fn kind(&self) -> &InterpErrorKind<'tcx> {
        &self.0.kind
    }
}

fn print_backtrace(backtrace: &Backtrace) {
    eprintln!("\n\nAn error occurred in the MIR interpreter:\n{backtrace}");
}

impl From<ErrorHandled> for InterpErrorInfo<'_> {
    fn from(err: ErrorHandled) -> Self {
        InterpErrorKind::InvalidProgram(match err {
            ErrorHandled::Reported(r, _span) => InvalidProgramInfo::AlreadyReported(r),
            ErrorHandled::TooGeneric(_span) => InvalidProgramInfo::TooGeneric,
        })
        .into()
    }
}

impl<'tcx> From<InterpErrorKind<'tcx>> for InterpErrorInfo<'tcx> {
    fn from(kind: InterpErrorKind<'tcx>) -> Self {
        InterpErrorInfo(Box::new(InterpErrorInfoInner {
            kind,
            backtrace: InterpErrorBacktrace::new(),
        }))
    }
}

/// Error information for when the program we executed turned out not to actually be a valid
/// program. This cannot happen in stand-alone Miri (except for layout errors that are only detect
/// during monomorphization), but it can happen during CTFE/ConstProp where we work on generic code
/// or execution does not have all information available.
#[derive(Debug)]
pub enum InvalidProgramInfo<'tcx> {
    /// Resolution can fail if we are in a too generic context.
    TooGeneric,
    /// Abort in case errors are already reported.
    AlreadyReported(ReportedErrorInfo),
    /// An error occurred during layout computation.
    Layout(layout::LayoutError<'tcx>),
}

/// Details of why a pointer had to be in-bounds.
#[derive(Debug, Copy, Clone)]
pub enum CheckInAllocMsg {
    /// We are accessing memory.
    MemoryAccess,
    /// We are doing pointer arithmetic.
    InboundsPointerArithmetic,
    /// None of the above -- generic/unspecific inbounds test.
    Dereferenceable,
}

/// Details of which pointer is not aligned.
#[derive(Debug, Copy, Clone)]
pub enum CheckAlignMsg {
    /// The accessed pointer did not have proper alignment.
    AccessedPtr,
    /// The access occurred with a place that was based on a misaligned pointer.
    BasedOn,
}

#[derive(Debug, Copy, Clone)]
pub enum InvalidMetaKind {
    /// Size of a `[T]` is too big
    SliceTooBig,
    /// Size of a DST is too big
    TooBig,
}

impl IntoDiagArg for InvalidMetaKind {
    fn into_diag_arg(self, _: &mut Option<std::path::PathBuf>) -> DiagArgValue {
        DiagArgValue::Str(Cow::Borrowed(match self {
            InvalidMetaKind::SliceTooBig => "slice_too_big",
            InvalidMetaKind::TooBig => "too_big",
        }))
    }
}

/// Details of an access to uninitialized bytes / bad pointer bytes where it is not allowed.
#[derive(Debug, Clone, Copy)]
pub struct BadBytesAccess {
    /// Range of the original memory access.
    pub access: AllocRange,
    /// Range of the bad memory that was encountered. (Might not be maximal.)
    pub bad: AllocRange,
}

/// Information about a size mismatch.
#[derive(Debug)]
pub struct ScalarSizeMismatch {
    pub target_size: u64,
    pub data_size: u64,
}

/// Information about a misaligned pointer.
#[derive(Copy, Clone, Hash, PartialEq, Eq, Debug)]
pub struct Misalignment {
    pub has: Align,
    pub required: Align,
}

macro_rules! impl_into_diag_arg_through_debug {
    ($($ty:ty),*$(,)?) => {$(
        impl IntoDiagArg for $ty {
            fn into_diag_arg(self, _: &mut Option<std::path::PathBuf>) -> DiagArgValue {
                DiagArgValue::Str(Cow::Owned(format!("{self:?}")))
            }
        }
    )*}
}

// These types have nice `Debug` output so we can just use them in diagnostics.
impl_into_diag_arg_through_debug! {
    AllocId,
    Pointer<AllocId>,
    AllocRange,
}

/// Error information for when the program caused Undefined Behavior.
#[derive(Debug)]
pub enum UndefinedBehaviorInfo<'tcx> {
    /// Free-form case. Only for errors that are never caught! Used by miri
    Ub(String),
    // FIXME(fee1-dead) these should all be actual variants of the enum instead of dynamically
    // dispatched
    /// A custom (free-form) fluent-translated error, created by `err_ub_custom!`.
    Custom(crate::error::CustomSubdiagnostic<'tcx>),
    /// Validation error.
    ValidationError(ValidationErrorInfo<'tcx>),

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
    /// Overflow in arithmetic that may not overflow.
    ArithOverflow { intrinsic: Symbol },
    /// Shift by too much.
    ShiftOverflow { intrinsic: Symbol, shift_amount: Either<u128, i128> },
    /// Invalid metadata in a wide pointer
    InvalidMeta(InvalidMetaKind),
    /// Reading a C string that does not end within its allocation.
    UnterminatedCString(Pointer<AllocId>),
    /// Using a pointer after it got freed.
    PointerUseAfterFree(AllocId, CheckInAllocMsg),
    /// Used a pointer outside the bounds it is valid for.
    PointerOutOfBounds {
        alloc_id: AllocId,
        alloc_size: Size,
        ptr_offset: i64,
        /// The size of the memory range that was expected to be in-bounds.
        inbounds_size: i64,
        msg: CheckInAllocMsg,
    },
    /// Using an integer as a pointer in the wrong way.
    DanglingIntPointer {
        addr: u64,
        /// The size of the memory range that was expected to be in-bounds (or 0 if we need an
        /// allocation but not any actual memory there, e.g. for function pointers).
        inbounds_size: i64,
        msg: CheckInAllocMsg,
    },
    /// Used a pointer with bad alignment.
    AlignmentCheckFailed(Misalignment, CheckAlignMsg),
    /// Writing to read-only memory.
    WriteToReadOnly(AllocId),
    /// Trying to access the data behind a function pointer.
    DerefFunctionPointer(AllocId),
    /// Trying to access the data behind a vtable pointer.
    DerefVTablePointer(AllocId),
    /// Trying to access the actual type id.
    DerefTypeIdPointer(AllocId),
    /// Using a non-boolean `u8` as bool.
    InvalidBool(u8),
    /// Using a non-character `u32` as character.
    InvalidChar(u32),
    /// The tag of an enum does not encode an actual discriminant.
    InvalidTag(Scalar<AllocId>),
    /// Using a pointer-not-to-a-function as function pointer.
    InvalidFunctionPointer(Pointer<AllocId>),
    /// Using a pointer-not-to-a-vtable as vtable pointer.
    InvalidVTablePointer(Pointer<AllocId>),
    /// Using a vtable for the wrong trait.
    InvalidVTableTrait {
        /// The vtable that was actually referenced by the wide pointer metadata.
        vtable_dyn_type: &'tcx ty::List<ty::PolyExistentialPredicate<'tcx>>,
        /// The vtable that was expected at the point in MIR that it was accessed.
        expected_dyn_type: &'tcx ty::List<ty::PolyExistentialPredicate<'tcx>>,
    },
    /// Using a string that is not valid UTF-8,
    InvalidStr(std::str::Utf8Error),
    /// Using uninitialized data where it is not allowed.
    InvalidUninitBytes(Option<(AllocId, BadBytesAccess)>),
    /// Working with a local that is not currently live.
    DeadLocal,
    /// Data size is not equal to target size.
    ScalarSizeMismatch(ScalarSizeMismatch),
    /// A discriminant of an uninhabited enum variant is written.
    UninhabitedEnumVariantWritten(VariantIdx),
    /// An uninhabited enum variant is projected.
    UninhabitedEnumVariantRead(Option<VariantIdx>),
    /// Trying to set discriminant to the niched variant, but the value does not match.
    InvalidNichedEnumVariantWritten { enum_ty: Ty<'tcx> },
    /// ABI-incompatible argument types.
    AbiMismatchArgument {
        /// The index of the argument whose type is wrong.
        arg_idx: usize,
        caller_ty: Ty<'tcx>,
        callee_ty: Ty<'tcx>,
    },
    /// ABI-incompatible return types.
    AbiMismatchReturn { caller_ty: Ty<'tcx>, callee_ty: Ty<'tcx> },
}

#[derive(Debug, Clone, Copy)]
pub enum PointerKind {
    Ref(Mutability),
    Box,
}

impl IntoDiagArg for PointerKind {
    fn into_diag_arg(self, _: &mut Option<std::path::PathBuf>) -> DiagArgValue {
        DiagArgValue::Str(
            match self {
                Self::Ref(_) => "ref",
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
    EnumTag,
    Str,
}

impl From<PointerKind> for ExpectedKind {
    fn from(x: PointerKind) -> ExpectedKind {
        match x {
            PointerKind::Box => ExpectedKind::Box,
            PointerKind::Ref(_) => ExpectedKind::Reference,
        }
    }
}

#[derive(Debug)]
pub enum ValidationErrorKind<'tcx> {
    PointerAsInt {
        expected: ExpectedKind,
    },
    PartialPointer,
    PtrToUninhabited {
        ptr_kind: PointerKind,
        ty: Ty<'tcx>,
    },
    MutableRefToImmutable,
    UnsafeCellInImmutable,
    MutableRefInConst,
    NullFnPtr,
    NeverVal,
    NonnullPtrMaybeNull,
    PtrOutOfRange {
        range: WrappingRange,
        max_value: u128,
    },
    OutOfRange {
        value: String,
        range: WrappingRange,
        max_value: u128,
    },
    UninhabitedVal {
        ty: Ty<'tcx>,
    },
    InvalidEnumTag {
        value: String,
    },
    UninhabitedEnumVariant,
    Uninit {
        expected: ExpectedKind,
    },
    InvalidVTablePtr {
        value: String,
    },
    InvalidMetaWrongTrait {
        /// The vtable that was actually referenced by the wide pointer metadata.
        vtable_dyn_type: &'tcx ty::List<ty::PolyExistentialPredicate<'tcx>>,
        /// The vtable that was expected at the point in MIR that it was accessed.
        expected_dyn_type: &'tcx ty::List<ty::PolyExistentialPredicate<'tcx>>,
    },
    InvalidMetaSliceTooLarge {
        ptr_kind: PointerKind,
    },
    InvalidMetaTooLarge {
        ptr_kind: PointerKind,
    },
    UnalignedPtr {
        ptr_kind: PointerKind,
        required_bytes: u64,
        found_bytes: u64,
    },
    NullPtr {
        ptr_kind: PointerKind,
        /// Records whether this pointer is definitely null or just may be null.
        maybe: bool,
    },
    DanglingPtrNoProvenance {
        ptr_kind: PointerKind,
        pointer: String,
    },
    DanglingPtrOutOfBounds {
        ptr_kind: PointerKind,
    },
    DanglingPtrUseAfterFree {
        ptr_kind: PointerKind,
    },
    InvalidBool {
        value: String,
    },
    InvalidChar {
        value: String,
    },
    InvalidFnPtr {
        value: String,
    },
}

/// Error information for when the program did something that might (or might not) be correct
/// to do according to the Rust spec, but due to limitations in the interpreter, the
/// operation could not be carried out. These limitations can differ between CTFE and the
/// Miri engine, e.g., CTFE does not support dereferencing pointers at integral addresses.
#[derive(Debug)]
pub enum UnsupportedOpInfo {
    /// Free-form case. Only for errors that are never caught! Used by Miri.
    // FIXME still use translatable diagnostics
    Unsupported(String),
    /// Unsized local variables.
    UnsizedLocal,
    /// Extern type field with an indeterminate offset.
    ExternTypeField,
    //
    // The variants below are only reachable from CTFE/const prop, miri will never emit them.
    //
    /// Attempting to read or copy parts of a pointer to somewhere else; without knowing absolute
    /// addresses, the resulting state cannot be represented by the CTFE interpreter.
    ReadPartialPointer(Pointer<AllocId>),
    /// Encountered a pointer where we needed an integer.
    ReadPointerAsInt(Option<(AllocId, BadBytesAccess)>),
    /// Accessing thread local statics
    ThreadLocalStatic(DefId),
    /// Accessing an unsupported extern static.
    ExternStatic(DefId),
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
    /// The compiler got an interrupt signal (a user ran out of patience).
    Interrupted,
}

/// A trait for machine-specific errors (or other "machine stop" conditions).
pub trait MachineStopType: Any + fmt::Debug + Send {
    /// The diagnostic message for this error
    fn diagnostic_message(&self) -> DiagMessage;
    /// Add diagnostic arguments by passing name and value pairs to `adder`, which are passed to
    /// fluent for formatting the translated diagnostic message.
    fn add_args(self: Box<Self>, adder: &mut dyn FnMut(DiagArgName, DiagArgValue));
}

impl dyn MachineStopType {
    #[inline(always)]
    pub fn downcast_ref<T: Any>(&self) -> Option<&T> {
        let x: &dyn Any = self;
        x.downcast_ref()
    }
}

#[derive(Debug)]
pub enum InterpErrorKind<'tcx> {
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

impl InterpErrorKind<'_> {
    /// Some errors do string formatting even if the error is never printed.
    /// To avoid performance issues, there are places where we want to be sure to never raise these formatting errors,
    /// so this method lets us detect them and `bug!` on unexpected errors.
    pub fn formatted_string(&self) -> bool {
        matches!(
            self,
            InterpErrorKind::Unsupported(UnsupportedOpInfo::Unsupported(_))
                | InterpErrorKind::UndefinedBehavior(UndefinedBehaviorInfo::ValidationError { .. })
                | InterpErrorKind::UndefinedBehavior(UndefinedBehaviorInfo::Ub(_))
        )
    }
}

// Macros for constructing / throwing `InterpErrorKind`
#[macro_export]
macro_rules! err_unsup {
    ($($tt:tt)*) => {
        $crate::mir::interpret::InterpErrorKind::Unsupported(
            $crate::mir::interpret::UnsupportedOpInfo::$($tt)*
        )
    };
}

#[macro_export]
macro_rules! err_unsup_format {
    ($($tt:tt)*) => { $crate::err_unsup!(Unsupported(format!($($tt)*))) };
}

#[macro_export]
macro_rules! err_inval {
    ($($tt:tt)*) => {
        $crate::mir::interpret::InterpErrorKind::InvalidProgram(
            $crate::mir::interpret::InvalidProgramInfo::$($tt)*
        )
    };
}

#[macro_export]
macro_rules! err_ub {
    ($($tt:tt)*) => {
        $crate::mir::interpret::InterpErrorKind::UndefinedBehavior(
            $crate::mir::interpret::UndefinedBehaviorInfo::$($tt)*
        )
    };
}

#[macro_export]
macro_rules! err_ub_format {
    ($($tt:tt)*) => { $crate::err_ub!(Ub(format!($($tt)*))) };
}

#[macro_export]
macro_rules! err_ub_custom {
    ($msg:expr $(, $($name:ident = $value:expr),* $(,)?)?) => {{
        $(
            let ($($name,)*) = ($($value,)*);
        )?
        $crate::err_ub!(Custom(
            $crate::error::CustomSubdiagnostic {
                msg: || $msg,
                add_args: Box::new(move |mut set_arg| {
                    $($(
                        set_arg(stringify!($name).into(), rustc_errors::IntoDiagArg::into_diag_arg($name, &mut None));
                    )*)?
                })
            }
        ))
    }};
}

#[macro_export]
macro_rules! err_exhaust {
    ($($tt:tt)*) => {
        $crate::mir::interpret::InterpErrorKind::ResourceExhaustion(
            $crate::mir::interpret::ResourceExhaustionInfo::$($tt)*
        )
    };
}

#[macro_export]
macro_rules! err_machine_stop {
    ($($tt:tt)*) => {
        $crate::mir::interpret::InterpErrorKind::MachineStop(Box::new($($tt)*))
    };
}

// In the `throw_*` macros, avoid `return` to make them work with `try {}`.
#[macro_export]
macro_rules! throw_unsup {
    ($($tt:tt)*) => { do yeet $crate::err_unsup!($($tt)*) };
}

#[macro_export]
macro_rules! throw_unsup_format {
    ($($tt:tt)*) => { do yeet $crate::err_unsup_format!($($tt)*) };
}

#[macro_export]
macro_rules! throw_inval {
    ($($tt:tt)*) => { do yeet $crate::err_inval!($($tt)*) };
}

#[macro_export]
macro_rules! throw_ub {
    ($($tt:tt)*) => { do yeet $crate::err_ub!($($tt)*) };
}

#[macro_export]
macro_rules! throw_ub_format {
    ($($tt:tt)*) => { do yeet $crate::err_ub_format!($($tt)*) };
}

#[macro_export]
macro_rules! throw_ub_custom {
    ($($tt:tt)*) => { do yeet $crate::err_ub_custom!($($tt)*) };
}

#[macro_export]
macro_rules! throw_exhaust {
    ($($tt:tt)*) => { do yeet $crate::err_exhaust!($($tt)*) };
}

#[macro_export]
macro_rules! throw_machine_stop {
    ($($tt:tt)*) => { do yeet $crate::err_machine_stop!($($tt)*) };
}

/// Guard type that panics on drop.
#[derive(Debug)]
struct Guard;

impl Drop for Guard {
    fn drop(&mut self) {
        // We silence the guard if we are already panicking, to avoid double-panics.
        if !std::thread::panicking() {
            panic!(
                "an interpreter error got improperly discarded; use `discard_err()` if this is intentional"
            );
        }
    }
}

/// The result type used by the interpreter. This is a newtype around `Result`
/// to block access to operations like `ok()` that discard UB errors.
///
/// We also make things panic if this type is ever implicitly dropped.
#[derive(Debug)]
#[must_use]
pub struct InterpResult<'tcx, T = ()> {
    res: Result<T, InterpErrorInfo<'tcx>>,
    guard: Guard,
}

impl<'tcx, T> ops::Try for InterpResult<'tcx, T> {
    type Output = T;
    type Residual = InterpResult<'tcx, convert::Infallible>;

    #[inline]
    fn from_output(output: Self::Output) -> Self {
        InterpResult::new(Ok(output))
    }

    #[inline]
    fn branch(self) -> ops::ControlFlow<Self::Residual, Self::Output> {
        match self.disarm() {
            Ok(v) => ops::ControlFlow::Continue(v),
            Err(e) => ops::ControlFlow::Break(InterpResult::new(Err(e))),
        }
    }
}

impl<'tcx, T> ops::Residual<T> for InterpResult<'tcx, convert::Infallible> {
    type TryType = InterpResult<'tcx, T>;
}

impl<'tcx, T> ops::FromResidual for InterpResult<'tcx, T> {
    #[inline]
    #[track_caller]
    fn from_residual(residual: InterpResult<'tcx, convert::Infallible>) -> Self {
        match residual.disarm() {
            Err(e) => Self::new(Err(e)),
        }
    }
}

// Allow `yeet`ing `InterpError` in functions returning `InterpResult_`.
impl<'tcx, T> ops::FromResidual<ops::Yeet<InterpErrorKind<'tcx>>> for InterpResult<'tcx, T> {
    #[inline]
    fn from_residual(ops::Yeet(e): ops::Yeet<InterpErrorKind<'tcx>>) -> Self {
        Self::new(Err(e.into()))
    }
}

// Allow `?` on `Result<_, InterpError>` in functions returning `InterpResult_`.
// This is useful e.g. for `option.ok_or_else(|| err_ub!(...))`.
impl<'tcx, T, E: Into<InterpErrorInfo<'tcx>>> ops::FromResidual<Result<convert::Infallible, E>>
    for InterpResult<'tcx, T>
{
    #[inline]
    fn from_residual(residual: Result<convert::Infallible, E>) -> Self {
        match residual {
            Err(e) => Self::new(Err(e.into())),
        }
    }
}

impl<'tcx, T, E: Into<InterpErrorInfo<'tcx>>> From<Result<T, E>> for InterpResult<'tcx, T> {
    #[inline]
    fn from(value: Result<T, E>) -> Self {
        Self::new(value.map_err(|e| e.into()))
    }
}

impl<'tcx, T, V: FromIterator<T>> FromIterator<InterpResult<'tcx, T>> for InterpResult<'tcx, V> {
    fn from_iter<I: IntoIterator<Item = InterpResult<'tcx, T>>>(iter: I) -> Self {
        Self::new(iter.into_iter().map(|x| x.disarm()).collect())
    }
}

impl<'tcx, T> InterpResult<'tcx, T> {
    #[inline(always)]
    fn new(res: Result<T, InterpErrorInfo<'tcx>>) -> Self {
        Self { res, guard: Guard }
    }

    #[inline(always)]
    fn disarm(self) -> Result<T, InterpErrorInfo<'tcx>> {
        mem::forget(self.guard);
        self.res
    }

    /// Discard the error information in this result. Only use this if ignoring Undefined Behavior is okay!
    #[inline]
    pub fn discard_err(self) -> Option<T> {
        self.disarm().ok()
    }

    /// Look at the `Result` wrapped inside of this.
    /// Must only be used to report the error!
    #[inline]
    pub fn report_err(self) -> Result<T, InterpErrorInfo<'tcx>> {
        self.disarm()
    }

    #[inline]
    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> InterpResult<'tcx, U> {
        InterpResult::new(self.disarm().map(f))
    }

    #[inline]
    pub fn map_err_info(
        self,
        f: impl FnOnce(InterpErrorInfo<'tcx>) -> InterpErrorInfo<'tcx>,
    ) -> InterpResult<'tcx, T> {
        InterpResult::new(self.disarm().map_err(f))
    }

    #[inline]
    pub fn map_err_kind(
        self,
        f: impl FnOnce(InterpErrorKind<'tcx>) -> InterpErrorKind<'tcx>,
    ) -> InterpResult<'tcx, T> {
        InterpResult::new(self.disarm().map_err(|mut e| {
            e.0.kind = f(e.0.kind);
            e
        }))
    }

    #[inline]
    pub fn inspect_err_kind(self, f: impl FnOnce(&InterpErrorKind<'tcx>)) -> InterpResult<'tcx, T> {
        InterpResult::new(self.disarm().inspect_err(|e| f(&e.0.kind)))
    }

    #[inline]
    #[track_caller]
    pub fn unwrap(self) -> T {
        self.disarm().unwrap()
    }

    #[inline]
    #[track_caller]
    pub fn unwrap_or_else(self, f: impl FnOnce(InterpErrorInfo<'tcx>) -> T) -> T {
        self.disarm().unwrap_or_else(f)
    }

    #[inline]
    #[track_caller]
    pub fn expect(self, msg: &str) -> T {
        self.disarm().expect(msg)
    }

    #[inline]
    pub fn and_then<U>(self, f: impl FnOnce(T) -> InterpResult<'tcx, U>) -> InterpResult<'tcx, U> {
        InterpResult::new(self.disarm().and_then(|t| f(t).disarm()))
    }

    /// Returns success if both `self` and `other` succeed, while ensuring we don't
    /// accidentally drop an error.
    ///
    /// If both are an error, `self` will be reported.
    #[inline]
    pub fn and<U>(self, other: InterpResult<'tcx, U>) -> InterpResult<'tcx, (T, U)> {
        match self.disarm() {
            Ok(t) => interp_ok((t, other?)),
            Err(e) => {
                // Discard the other error.
                drop(other.disarm());
                // Return `self`.
                InterpResult::new(Err(e))
            }
        }
    }
}

#[inline(always)]
pub fn interp_ok<'tcx, T>(x: T) -> InterpResult<'tcx, T> {
    InterpResult::new(Ok(x))
}
