use super::{AllocId, AllocRange, ConstAlloc, Pointer, Scalar};

use crate::mir::interpret::ConstValue;
use crate::ty::{layout, query::TyCtxtAt, tls, Ty, ValTree};

use rustc_data_structures::sync::Lock;
use rustc_errors::{pluralize, struct_span_err, DiagnosticBuilder, ErrorGuaranteed};
use rustc_macros::HashStable;
use rustc_session::CtfeBacktrace;
use rustc_span::def_id::DefId;
use rustc_target::abi::{call, Align, Size};
use std::{any::Any, backtrace::Backtrace, fmt};

#[derive(Debug, Copy, Clone, PartialEq, Eq, HashStable, TyEncodable, TyDecodable)]
pub enum ErrorHandled {
    /// Already reported an error for this evaluation, and the compilation is
    /// *guaranteed* to fail. Warnings/lints *must not* produce `Reported`.
    Reported(ErrorGuaranteed),
    /// Don't emit an error, the evaluation failed because the MIR was generic
    /// and the substs didn't fully monomorphize it.
    TooGeneric,
}

impl From<ErrorGuaranteed> for ErrorHandled {
    fn from(err: ErrorGuaranteed) -> ErrorHandled {
        ErrorHandled::Reported(err)
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
    backtrace: Option<Box<Backtrace>>,
}

impl fmt::Display for InterpErrorInfo<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0.kind)
    }
}

impl<'tcx> InterpErrorInfo<'tcx> {
    pub fn print_backtrace(&self) {
        if let Some(backtrace) = self.0.backtrace.as_ref() {
            print_backtrace(backtrace);
        }
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

        InterpErrorInfo(Box::new(InterpErrorInfoInner { kind, backtrace }))
    }
}

/// Error information for when the program we executed turned out not to actually be a valid
/// program. This cannot happen in stand-alone Miri, but it can happen during CTFE/ConstProp
/// where we work on generic code or execution does not have all information available.
pub enum InvalidProgramInfo<'tcx> {
    /// Resolution can fail if we are in a too generic context.
    TooGeneric,
    /// Abort in case errors are already reported.
    AlreadyReported(ErrorGuaranteed),
    /// An error occurred during layout computation.
    Layout(layout::LayoutError<'tcx>),
    /// An error occurred during FnAbi computation: the passed --target lacks FFI support
    /// (which unfortunately typeck does not reject).
    /// Not using `FnAbiError` as that contains a nested `LayoutError`.
    FnAbiAdjustForForeignAbi(call::AdjustForForeignAbiError),
    /// SizeOf of unsized type was requested.
    SizeOfUnsizedType(Ty<'tcx>),
}

impl fmt::Display for InvalidProgramInfo<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use InvalidProgramInfo::*;
        match self {
            TooGeneric => write!(f, "encountered overly generic constant"),
            AlreadyReported(ErrorGuaranteed { .. }) => {
                write!(
                    f,
                    "an error has already been reported elsewhere (this should not usually be printed)"
                )
            }
            Layout(ref err) => write!(f, "{err}"),
            FnAbiAdjustForForeignAbi(ref err) => write!(f, "{err}"),
            SizeOfUnsizedType(ty) => write!(f, "size_of called on unsized type `{ty}`"),
        }
    }
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

impl fmt::Display for CheckInAllocMsg {
    /// When this is printed as an error the context looks like this:
    /// "{msg}{pointer} is a dangling pointer".
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match *self {
                CheckInAllocMsg::DerefTest => "dereferencing pointer failed: ",
                CheckInAllocMsg::MemoryAccessTest => "memory access failed: ",
                CheckInAllocMsg::PointerArithmeticTest => "out-of-bounds pointer arithmetic: ",
                CheckInAllocMsg::OffsetFromTest => "out-of-bounds offset_from: ",
                CheckInAllocMsg::InboundsTest => "out-of-bounds pointer use: ",
            }
        )
    }
}

/// Details of an access to uninitialized bytes where it is not allowed.
#[derive(Debug)]
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

/// Error information for when the program caused Undefined Behavior.
pub enum UndefinedBehaviorInfo {
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
    /// Signed division overflowed (INT_MIN / -1).
    DivisionOverflow,
    /// Signed remainder overflowed (INT_MIN % -1).
    RemainderOverflow,
    /// Overflowing inbounds pointer arithmetic.
    PointerArithOverflow,
    /// Invalid metadata in a wide pointer (using `str` to avoid allocations).
    InvalidMeta(&'static str),
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
    AlignmentCheckFailed {
        required: Align,
        has: Align,
    },
    /// Writing to read-only memory.
    WriteToReadOnly(AllocId),
    // Trying to access the data behind a function pointer.
    DerefFunctionPointer(AllocId),
    // Trying to access the data behind a vtable pointer.
    DerefVTablePointer(AllocId),
    /// The value validity check found a problem.
    /// Should only be thrown by `validity.rs` and always point out which part of the value
    /// is the problem.
    ValidationFailure {
        /// The "path" to the value in question, e.g. `.0[5].field` for a struct
        /// field in the 6th element of an array that is the first element of a tuple.
        path: Option<String>,
        msg: String,
    },
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
}

impl fmt::Display for UndefinedBehaviorInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use UndefinedBehaviorInfo::*;
        match self {
            Ub(msg) => write!(f, "{msg}"),
            Unreachable => write!(f, "entering unreachable code"),
            BoundsCheckFailed { ref len, ref index } => {
                write!(f, "indexing out of bounds: the len is {len} but the index is {index}")
            }
            DivisionByZero => write!(f, "dividing by zero"),
            RemainderByZero => write!(f, "calculating the remainder with a divisor of zero"),
            DivisionOverflow => write!(f, "overflow in signed division (dividing MIN by -1)"),
            RemainderOverflow => write!(f, "overflow in signed remainder (dividing MIN by -1)"),
            PointerArithOverflow => write!(f, "overflowing in-bounds pointer arithmetic"),
            InvalidMeta(msg) => write!(f, "invalid metadata in wide pointer: {msg}"),
            UnterminatedCString(p) => write!(
                f,
                "reading a null-terminated string starting at {p:?} with no null found before end of allocation",
            ),
            PointerUseAfterFree(a) => {
                write!(f, "pointer to {a:?} was dereferenced after this allocation got freed")
            }
            PointerOutOfBounds { alloc_id, alloc_size, ptr_offset, ptr_size: Size::ZERO, msg } => {
                write!(
                    f,
                    "{msg}{alloc_id:?} has size {alloc_size}, so pointer at offset {ptr_offset} is out-of-bounds",
                    alloc_size = alloc_size.bytes(),
                )
            }
            PointerOutOfBounds { alloc_id, alloc_size, ptr_offset, ptr_size, msg } => write!(
                f,
                "{msg}{alloc_id:?} has size {alloc_size}, so pointer to {ptr_size} byte{ptr_size_p} starting at offset {ptr_offset} is out-of-bounds",
                alloc_size = alloc_size.bytes(),
                ptr_size = ptr_size.bytes(),
                ptr_size_p = pluralize!(ptr_size.bytes()),
            ),
            DanglingIntPointer(i, msg) => {
                write!(
                    f,
                    "{msg}{pointer} is a dangling pointer (it has no provenance)",
                    pointer = Pointer::<Option<AllocId>>::from_addr_invalid(*i),
                )
            }
            AlignmentCheckFailed { required, has } => write!(
                f,
                "accessing memory with alignment {has}, but alignment {required} is required",
                has = has.bytes(),
                required = required.bytes()
            ),
            WriteToReadOnly(a) => write!(f, "writing to {a:?} which is read-only"),
            DerefFunctionPointer(a) => write!(f, "accessing {a:?} which contains a function"),
            DerefVTablePointer(a) => write!(f, "accessing {a:?} which contains a vtable"),
            ValidationFailure { path: None, msg } => {
                write!(f, "constructing invalid value: {msg}")
            }
            ValidationFailure { path: Some(path), msg } => {
                write!(f, "constructing invalid value at {path}: {msg}")
            }
            InvalidBool(b) => {
                write!(f, "interpreting an invalid 8-bit value as a bool: 0x{b:02x}")
            }
            InvalidChar(c) => {
                write!(f, "interpreting an invalid 32-bit value as a char: 0x{c:08x}")
            }
            InvalidTag(val) => write!(f, "enum value has invalid tag: {val:x}"),
            InvalidFunctionPointer(p) => {
                write!(f, "using {p:?} as function pointer but it does not point to a function")
            }
            InvalidVTablePointer(p) => {
                write!(f, "using {p:?} as vtable pointer but it does not point to a vtable")
            }
            InvalidStr(err) => write!(f, "this string is not valid UTF-8: {err}"),
            InvalidUninitBytes(Some((alloc, info))) => write!(
                f,
                "reading memory at {alloc:?}{access:?}, \
                 but memory is uninitialized at {uninit:?}, \
                 and this operation requires initialized memory",
                access = info.access,
                uninit = info.uninit,
            ),
            InvalidUninitBytes(None) => write!(
                f,
                "using uninitialized data, but this operation requires initialized memory"
            ),
            DeadLocal => write!(f, "accessing a dead local variable"),
            ScalarSizeMismatch(self::ScalarSizeMismatch { target_size, data_size }) => write!(
                f,
                "scalar size mismatch: expected {target_size} bytes but got {data_size} bytes instead",
            ),
            UninhabitedEnumVariantWritten => {
                write!(f, "writing discriminant of an uninhabited enum")
            }
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

impl fmt::Display for UnsupportedOpInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use UnsupportedOpInfo::*;
        match self {
            Unsupported(ref msg) => write!(f, "{msg}"),
            PartialPointerOverwrite(ptr) => {
                write!(f, "unable to overwrite parts of a pointer in memory at {ptr:?}")
            }
            PartialPointerCopy(ptr) => {
                write!(f, "unable to copy parts of a pointer from memory at {ptr:?}")
            }
            ReadPointerAsBytes => write!(f, "unable to turn pointer into raw bytes"),
            ThreadLocalStatic(did) => write!(f, "cannot access thread local static ({did:?})"),
            ReadExternStatic(did) => write!(f, "cannot read from extern static ({did:?})"),
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
    /// There is not enough memory (on the host) to perform an allocation.
    MemoryExhausted,
    /// The address space (of the target) is full.
    AddressSpaceFull,
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
            MemoryExhausted => {
                write!(f, "tried to allocate more memory than available to compiler")
            }
            AddressSpaceFull => {
                write!(f, "there are no more free addresses in the address space")
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

impl dyn MachineStopType {
    #[inline(always)]
    pub fn downcast_ref<T: Any>(&self) -> Option<&T> {
        self.as_any().downcast_ref()
    }
}

pub enum InterpError<'tcx> {
    /// The program caused undefined behavior.
    UndefinedBehavior(UndefinedBehaviorInfo),
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
            Unsupported(ref msg) => write!(f, "{msg}"),
            InvalidProgram(ref msg) => write!(f, "{msg}"),
            UndefinedBehavior(ref msg) => write!(f, "{msg}"),
            ResourceExhaustion(ref msg) => write!(f, "{msg}"),
            MachineStop(ref msg) => write!(f, "{msg}"),
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
    /// Some errors do string formatting even if the error is never printed.
    /// To avoid performance issues, there are places where we want to be sure to never raise these formatting errors,
    /// so this method lets us detect them and `bug!` on unexpected errors.
    pub fn formatted_string(&self) -> bool {
        matches!(
            self,
            InterpError::Unsupported(UnsupportedOpInfo::Unsupported(_))
                | InterpError::UndefinedBehavior(UndefinedBehaviorInfo::ValidationFailure { .. })
                | InterpError::UndefinedBehavior(UndefinedBehaviorInfo::Ub(_))
        )
    }
}
