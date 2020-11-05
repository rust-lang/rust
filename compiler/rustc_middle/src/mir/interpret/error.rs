use super::{AllocId, ConstAlloc, Pointer, Scalar};

use crate::mir::interpret::ConstValue;
use crate::ty::{layout, query::TyCtxtAt, tls, FnSig, Ty};

use rustc_data_structures::sync::Lock;
use rustc_errors::{pluralize, struct_span_err, DiagnosticBuilder, ErrorReported};
use rustc_macros::HashStable;
use rustc_session::CtfeBacktrace;
use rustc_span::def_id::DefId;
use rustc_target::abi::{Align, Size};
use std::{any::Any, backtrace::Backtrace, fmt, mem};

#[derive(Debug, Copy, Clone, PartialEq, Eq, HashStable, TyEncodable, TyDecodable)]
pub enum ErrorHandled {
    /// Already reported an error for this evaluation, and the compilation is
    /// *guaranteed* to fail. Warnings/lints *must not* produce `Reported`.
    Reported(ErrorReported),
    /// Already emitted a lint for this evaluation.
    Linted,
    /// Don't emit an error, the evaluation failed because the MIR was generic
    /// and the substs didn't fully monomorphize it.
    TooGeneric,
}

impl From<ErrorReported> for ErrorHandled {
    fn from(err: ErrorReported) -> ErrorHandled {
        ErrorHandled::Reported(err)
    }
}

CloneTypeFoldableAndLiftImpls! {
    ErrorHandled,
}

pub type EvalToAllocationRawResult<'tcx> = Result<ConstAlloc<'tcx>, ErrorHandled>;
pub type EvalToConstValueResult<'tcx> = Result<ConstValue<'tcx>, ErrorHandled>;

pub fn struct_error<'tcx>(tcx: TyCtxtAt<'tcx>, msg: &str) -> DiagnosticBuilder<'tcx> {
    struct_span_err!(tcx.sess, tcx.span, E0080, "{}", msg)
}

/// Packages the kind of error we got from the const code interpreter
/// up with a Rust-level backtrace of where the error occurred.
/// Thsese should always be constructed by calling `.into()` on
/// a `InterpError`. In `librustc_mir::interpret`, we have `throw_err_*`
/// macros for this.
#[derive(Debug)]
pub struct InterpErrorInfo<'tcx> {
    pub kind: InterpError<'tcx>,
    backtrace: Option<Box<Backtrace>>,
}

impl fmt::Display for InterpErrorInfo<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.kind)
    }
}

impl InterpErrorInfo<'_> {
    pub fn print_backtrace(&self) {
        if let Some(backtrace) = self.backtrace.as_ref() {
            print_backtrace(backtrace);
        }
    }
}

fn print_backtrace(backtrace: &Backtrace) {
    eprintln!("\n\nAn error occurred in miri:\n{}", backtrace);
}

impl From<ErrorHandled> for InterpErrorInfo<'_> {
    fn from(err: ErrorHandled) -> Self {
        match err {
            ErrorHandled::Reported(ErrorReported) | ErrorHandled::Linted => {
                err_inval!(ReferencedConstant)
            }
            ErrorHandled::TooGeneric => err_inval!(TooGeneric),
        }
        .into()
    }
}

impl From<ErrorReported> for InterpErrorInfo<'_> {
    fn from(err: ErrorReported) -> Self {
        InterpError::InvalidProgram(InvalidProgramInfo::AlreadyReported(err)).into()
    }
}

impl<'tcx> From<InterpError<'tcx>> for InterpErrorInfo<'tcx> {
    fn from(kind: InterpError<'tcx>) -> Self {
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

        InterpErrorInfo { kind, backtrace }
    }
}

/// Error information for when the program we executed turned out not to actually be a valid
/// program. This cannot happen in stand-alone Miri, but it can happen during CTFE/ConstProp
/// where we work on generic code or execution does not have all information available.
pub enum InvalidProgramInfo<'tcx> {
    /// Resolution can fail if we are in a too generic context.
    TooGeneric,
    /// Cannot compute this constant because it depends on another one
    /// which already produced an error.
    ReferencedConstant,
    /// Abort in case errors are already reported.
    AlreadyReported(ErrorReported),
    /// An error occurred during layout computation.
    Layout(layout::LayoutError<'tcx>),
    /// An invalid transmute happened.
    TransmuteSizeDiff(Ty<'tcx>, Ty<'tcx>),
}

impl fmt::Display for InvalidProgramInfo<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use InvalidProgramInfo::*;
        match self {
            TooGeneric => write!(f, "encountered overly generic constant"),
            ReferencedConstant => write!(f, "referenced constant has errors"),
            AlreadyReported(ErrorReported) => {
                write!(f, "encountered constants with type errors, stopping evaluation")
            }
            Layout(ref err) => write!(f, "{}", err),
            TransmuteSizeDiff(from_ty, to_ty) => write!(
                f,
                "transmuting `{}` to `{}` is not possible, because these types do not have the same size",
                from_ty, to_ty
            ),
        }
    }
}

/// Details of why a pointer had to be in-bounds.
#[derive(Debug, Copy, Clone, TyEncodable, TyDecodable, HashStable)]
pub enum CheckInAllocMsg {
    MemoryAccessTest,
    NullPointerTest,
    PointerArithmeticTest,
    InboundsTest,
}

impl fmt::Display for CheckInAllocMsg {
    /// When this is printed as an error the context looks like this
    /// "{test name} failed: pointer must be in-bounds at offset..."
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match *self {
                CheckInAllocMsg::MemoryAccessTest => "memory access",
                CheckInAllocMsg::NullPointerTest => "NULL pointer test",
                CheckInAllocMsg::PointerArithmeticTest => "pointer arithmetic",
                CheckInAllocMsg::InboundsTest => "inbounds test",
            }
        )
    }
}

/// Details of an access to uninitialized bytes where it is not allowed.
#[derive(Debug)]
pub struct UninitBytesAccess {
    /// Location of the original memory access.
    pub access_ptr: Pointer,
    /// Size of the original memory access.
    pub access_size: Size,
    /// Location of the first uninitialized byte that was accessed.
    pub uninit_ptr: Pointer,
    /// Number of consecutive uninitialized bytes that were accessed.
    pub uninit_size: Size,
}

/// Error information for when the program caused Undefined Behavior.
pub enum UndefinedBehaviorInfo<'tcx> {
    /// Free-form case. Only for errors that are never caught!
    Ub(String),
    /// Unreachable code was executed.
    Unreachable,
    /// A slice/array index projection went out-of-bounds.
    BoundsCheckFailed {
        len: u64,
        index: u64,
    },
    /// Something was divided by 0 (x / 0).
    DivisionByZero,
    /// Something was "remainded" by 0 (x % 0).
    RemainderByZero,
    /// Overflowing inbounds pointer arithmetic.
    PointerArithOverflow,
    /// Invalid metadata in a wide pointer (using `str` to avoid allocations).
    InvalidMeta(&'static str),
    /// Invalid drop function in vtable.
    InvalidDropFn(FnSig<'tcx>),
    /// Reading a C string that does not end within its allocation.
    UnterminatedCString(Pointer),
    /// Dereferencing a dangling pointer after it got freed.
    PointerUseAfterFree(AllocId),
    /// Used a pointer outside the bounds it is valid for.
    PointerOutOfBounds {
        ptr: Pointer,
        msg: CheckInAllocMsg,
        allocation_size: Size,
    },
    /// Using an integer as a pointer in the wrong way.
    DanglingIntPointer(u64, CheckInAllocMsg),
    /// Used a pointer with bad alignment.
    AlignmentCheckFailed {
        required: Align,
        has: Align,
    },
    /// Writing to read-only memory.
    WriteToReadOnly(AllocId),
    // Trying to access the data behind a function pointer.
    DerefFunctionPointer(AllocId),
    /// The value validity check found a problem.
    /// Should only be thrown by `validity.rs` and always point out which part of the value
    /// is the problem.
    ValidationFailure(String),
    /// Using a non-boolean `u8` as bool.
    InvalidBool(u8),
    /// Using a non-character `u32` as character.
    InvalidChar(u32),
    /// The tag of an enum does not encode an actual discriminant.
    InvalidTag(Scalar),
    /// Using a pointer-not-to-a-function as function pointer.
    InvalidFunctionPointer(Pointer),
    /// Using a string that is not valid UTF-8,
    InvalidStr(std::str::Utf8Error),
    /// Using uninitialized data where it is not allowed.
    InvalidUninitBytes(Option<Box<UninitBytesAccess>>),
    /// Working with a local that is not currently live.
    DeadLocal,
    /// Data size is not equal to target size.
    ScalarSizeMismatch {
        target_size: u64,
        data_size: u64,
    },
}

impl fmt::Display for UndefinedBehaviorInfo<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use UndefinedBehaviorInfo::*;
        match self {
            Ub(msg) => write!(f, "{}", msg),
            Unreachable => write!(f, "entering unreachable code"),
            BoundsCheckFailed { ref len, ref index } => {
                write!(f, "indexing out of bounds: the len is {} but the index is {}", len, index)
            }
            DivisionByZero => write!(f, "dividing by zero"),
            RemainderByZero => write!(f, "calculating the remainder with a divisor of zero"),
            PointerArithOverflow => write!(f, "overflowing in-bounds pointer arithmetic"),
            InvalidMeta(msg) => write!(f, "invalid metadata in wide pointer: {}", msg),
            InvalidDropFn(sig) => write!(
                f,
                "invalid drop function signature: got {}, expected exactly one argument which must be a pointer type",
                sig
            ),
            UnterminatedCString(p) => write!(
                f,
                "reading a null-terminated string starting at {} with no null found before end of allocation",
                p,
            ),
            PointerUseAfterFree(a) => {
                write!(f, "pointer to {} was dereferenced after this allocation got freed", a)
            }
            PointerOutOfBounds { ptr, msg, allocation_size } => write!(
                f,
                "{} failed: pointer must be in-bounds at offset {}, \
                           but is outside bounds of {} which has size {}",
                msg,
                ptr.offset.bytes(),
                ptr.alloc_id,
                allocation_size.bytes()
            ),
            DanglingIntPointer(_, CheckInAllocMsg::NullPointerTest) => {
                write!(f, "NULL pointer is not allowed for this operation")
            }
            DanglingIntPointer(i, msg) => {
                write!(f, "{} failed: 0x{:x} is not a valid pointer", msg, i)
            }
            AlignmentCheckFailed { required, has } => write!(
                f,
                "accessing memory with alignment {}, but alignment {} is required",
                has.bytes(),
                required.bytes()
            ),
            WriteToReadOnly(a) => write!(f, "writing to {} which is read-only", a),
            DerefFunctionPointer(a) => write!(f, "accessing {} which contains a function", a),
            ValidationFailure(ref err) => write!(f, "type validation failed: {}", err),
            InvalidBool(b) => {
                write!(f, "interpreting an invalid 8-bit value as a bool: 0x{:02x}", b)
            }
            InvalidChar(c) => {
                write!(f, "interpreting an invalid 32-bit value as a char: 0x{:08x}", c)
            }
            InvalidTag(val) => write!(f, "enum value has invalid tag: {}", val),
            InvalidFunctionPointer(p) => {
                write!(f, "using {} as function pointer but it does not point to a function", p)
            }
            InvalidStr(err) => write!(f, "this string is not valid UTF-8: {}", err),
            InvalidUninitBytes(Some(access)) => write!(
                f,
                "reading {} byte{} of memory starting at {}, \
                 but {} byte{} {} uninitialized starting at {}, \
                 and this operation requires initialized memory",
                access.access_size.bytes(),
                pluralize!(access.access_size.bytes()),
                access.access_ptr,
                access.uninit_size.bytes(),
                pluralize!(access.uninit_size.bytes()),
                if access.uninit_size.bytes() != 1 { "are" } else { "is" },
                access.uninit_ptr,
            ),
            InvalidUninitBytes(None) => write!(
                f,
                "using uninitialized data, but this operation requires initialized memory"
            ),
            DeadLocal => write!(f, "accessing a dead local variable"),
            ScalarSizeMismatch { target_size, data_size } => write!(
                f,
                "scalar size mismatch: expected {} bytes but got {} bytes instead",
                target_size, data_size
            ),
        }
    }
}

/// Error information for when the program did something that might (or might not) be correct
/// to do according to the Rust spec, but due to limitations in the interpreter, the
/// operation could not be carried out. These limitations can differ between CTFE and the
/// Miri engine, e.g., CTFE does not support dereferencing pointers at integral addresses.
pub enum UnsupportedOpInfo {
    /// Free-form case. Only for errors that are never caught!
    Unsupported(String),
    /// Could not find MIR for a function.
    NoMirFor(DefId),
    /// Encountered a pointer where we needed raw bytes.
    ReadPointerAsBytes,
    //
    // The variants below are only reachable from CTFE/const prop, miri will never emit them.
    //
    /// Encountered raw bytes where we needed a pointer.
    ReadBytesAsPointer,
    /// Accessing thread local statics
    ThreadLocalStatic(DefId),
    /// Accessing an unsupported extern static.
    ReadExternStatic(DefId),
}

impl fmt::Display for UnsupportedOpInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use UnsupportedOpInfo::*;
        match self {
            Unsupported(ref msg) => write!(f, "{}", msg),
            ReadExternStatic(did) => write!(f, "cannot read from extern static ({:?})", did),
            NoMirFor(did) => write!(f, "no MIR body is available for {:?}", did),
            ReadPointerAsBytes => write!(f, "unable to turn pointer into raw bytes",),
            ReadBytesAsPointer => write!(f, "unable to turn bytes into a pointer"),
            ThreadLocalStatic(did) => write!(f, "cannot access thread local static ({:?})", did),
        }
    }
}

/// Error information for when the program exhausted the resources granted to it
/// by the interpreter.
pub enum ResourceExhaustionInfo {
    /// The stack grew too big.
    StackFrameLimitReached,
    /// The program ran for too long.
    ///
    /// The exact limit is set by the `const_eval_limit` attribute.
    StepLimitReached,
}

impl fmt::Display for ResourceExhaustionInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use ResourceExhaustionInfo::*;
        match self {
            StackFrameLimitReached => {
                write!(f, "reached the configured maximum number of stack frames")
            }
            StepLimitReached => {
                write!(f, "exceeded interpreter step limit (see `#[const_eval_limit]`)")
            }
        }
    }
}

/// A trait to work around not having trait object upcasting.
pub trait AsAny: Any {
    fn as_any(&self) -> &dyn Any;
}
impl<T: Any> AsAny for T {
    #[inline(always)]
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// A trait for machine-specific errors (or other "machine stop" conditions).
pub trait MachineStopType: AsAny + fmt::Display + Send {}
impl MachineStopType for String {}

impl dyn MachineStopType {
    #[inline(always)]
    pub fn downcast_ref<T: Any>(&self) -> Option<&T> {
        self.as_any().downcast_ref()
    }
}

#[cfg(target_arch = "x86_64")]
static_assert_size!(InterpError<'_>, 40);

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

impl fmt::Display for InterpError<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use InterpError::*;
        match *self {
            Unsupported(ref msg) => write!(f, "{}", msg),
            InvalidProgram(ref msg) => write!(f, "{}", msg),
            UndefinedBehavior(ref msg) => write!(f, "{}", msg),
            ResourceExhaustion(ref msg) => write!(f, "{}", msg),
            MachineStop(ref msg) => write!(f, "{}", msg),
        }
    }
}

// Forward `Debug` to `Display`, so it does not look awful.
impl fmt::Debug for InterpError<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl InterpError<'_> {
    /// Some errors allocate to be created as they contain free-form strings.
    /// And sometimes we want to be sure that did not happen as it is a
    /// waste of resources.
    pub fn allocates(&self) -> bool {
        match self {
            // Zero-sized boxes do not allocate.
            InterpError::MachineStop(b) => mem::size_of_val::<dyn MachineStopType>(&**b) > 0,
            InterpError::Unsupported(UnsupportedOpInfo::Unsupported(_))
            | InterpError::UndefinedBehavior(UndefinedBehaviorInfo::ValidationFailure(_))
            | InterpError::UndefinedBehavior(UndefinedBehaviorInfo::Ub(_))
            | InterpError::UndefinedBehavior(UndefinedBehaviorInfo::InvalidUninitBytes(Some(_))) => {
                true
            }
            _ => false,
        }
    }
}
