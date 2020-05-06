use super::{AllocId, Pointer, RawConst, ScalarMaybeUndef};

use crate::mir::interpret::ConstValue;
use crate::ty::layout::LayoutError;
use crate::ty::query::TyCtxtAt;
use crate::ty::{self, layout, tls, FnSig, Ty};

use rustc_data_structures::sync::Lock;
use rustc_errors::{struct_span_err, DiagnosticBuilder, ErrorReported};
use rustc_hir as hir;
use rustc_hir::definitions::DefPathData;
use rustc_macros::HashStable;
use rustc_session::CtfeBacktrace;
use rustc_span::{def_id::DefId, Pos, Span};
use rustc_target::abi::{Align, Size};
use std::{any::Any, backtrace::Backtrace, fmt, mem};

#[derive(Debug, Copy, Clone, PartialEq, Eq, HashStable, RustcEncodable, RustcDecodable)]
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

CloneTypeFoldableImpls! {
    ErrorHandled,
}

pub type ConstEvalRawResult<'tcx> = Result<RawConst<'tcx>, ErrorHandled>;
pub type ConstEvalResult<'tcx> = Result<ConstValue<'tcx>, ErrorHandled>;

#[derive(Debug)]
pub struct ConstEvalErr<'tcx> {
    pub span: Span,
    pub error: crate::mir::interpret::InterpError<'tcx>,
    pub stacktrace: Vec<FrameInfo<'tcx>>,
}

#[derive(Debug)]
pub struct FrameInfo<'tcx> {
    pub instance: ty::Instance<'tcx>,
    pub span: Span,
    pub lint_root: Option<hir::HirId>,
}

impl<'tcx> fmt::Display for FrameInfo<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        ty::tls::with(|tcx| {
            if tcx.def_key(self.instance.def_id()).disambiguated_data.data
                == DefPathData::ClosureExpr
            {
                write!(f, "inside closure")?;
            } else {
                write!(f, "inside `{}`", self.instance)?;
            }
            if !self.span.is_dummy() {
                let lo = tcx.sess.source_map().lookup_char_pos(self.span.lo());
                write!(f, " at {}:{}:{}", lo.file.name, lo.line, lo.col.to_usize() + 1)?;
            }
            Ok(())
        })
    }
}

impl<'tcx> ConstEvalErr<'tcx> {
    pub fn struct_error(
        &self,
        tcx: TyCtxtAt<'tcx>,
        message: &str,
        emit: impl FnOnce(DiagnosticBuilder<'_>),
    ) -> ErrorHandled {
        self.struct_generic(tcx, message, emit, None)
    }

    pub fn report_as_error(&self, tcx: TyCtxtAt<'tcx>, message: &str) -> ErrorHandled {
        self.struct_error(tcx, message, |mut e| e.emit())
    }

    pub fn report_as_lint(
        &self,
        tcx: TyCtxtAt<'tcx>,
        message: &str,
        lint_root: hir::HirId,
        span: Option<Span>,
    ) -> ErrorHandled {
        self.struct_generic(
            tcx,
            message,
            |mut lint: DiagnosticBuilder<'_>| {
                // Apply the span.
                if let Some(span) = span {
                    let primary_spans = lint.span.primary_spans().to_vec();
                    // point at the actual error as the primary span
                    lint.replace_span_with(span);
                    // point to the `const` statement as a secondary span
                    // they don't have any label
                    for sp in primary_spans {
                        if sp != span {
                            lint.span_label(sp, "");
                        }
                    }
                }
                lint.emit();
            },
            Some(lint_root),
        )
    }

    /// Create a diagnostic for this const eval error.
    ///
    /// Sets the message passed in via `message` and adds span labels with detailed error
    /// information before handing control back to `emit` to do any final processing.
    /// It's the caller's responsibility to call emit(), stash(), etc. within the `emit`
    /// function to dispose of the diagnostic properly.
    ///
    /// If `lint_root.is_some()` report it as a lint, else report it as a hard error.
    /// (Except that for some errors, we ignore all that -- see `must_error` below.)
    fn struct_generic(
        &self,
        tcx: TyCtxtAt<'tcx>,
        message: &str,
        emit: impl FnOnce(DiagnosticBuilder<'_>),
        lint_root: Option<hir::HirId>,
    ) -> ErrorHandled {
        let must_error = match self.error {
            err_inval!(Layout(LayoutError::Unknown(_))) | err_inval!(TooGeneric) => {
                return ErrorHandled::TooGeneric;
            }
            err_inval!(TypeckError(error_reported)) => {
                return ErrorHandled::Reported(error_reported);
            }
            // We must *always* hard error on these, even if the caller wants just a lint.
            err_inval!(Layout(LayoutError::SizeOverflow(_))) => true,
            _ => false,
        };
        trace!("reporting const eval failure at {:?}", self.span);

        let err_msg = match &self.error {
            InterpError::MachineStop(msg) => {
                // A custom error (`ConstEvalErrKind` in `librustc_mir/interp/const_eval/error.rs`).
                // Should be turned into a string by now.
                msg.downcast_ref::<String>().expect("invalid MachineStop payload").clone()
            }
            err => err.to_string(),
        };

        let finish = |mut err: DiagnosticBuilder<'_>, span_msg: Option<String>| {
            if let Some(span_msg) = span_msg {
                err.span_label(self.span, span_msg);
            }
            // Add spans for the stacktrace. Don't print a single-line backtrace though.
            if self.stacktrace.len() > 1 {
                for frame_info in &self.stacktrace {
                    err.span_label(frame_info.span, frame_info.to_string());
                }
            }
            // Let the caller finish the job.
            emit(err)
        };

        if must_error {
            // The `message` makes little sense here, this is a more serious error than the
            // caller thinks anyway.
            // See <https://github.com/rust-lang/rust/pull/63152>.
            finish(struct_error(tcx, &err_msg), None);
            ErrorHandled::Reported(ErrorReported)
        } else {
            // Regular case.
            if let Some(lint_root) = lint_root {
                // Report as lint.
                let hir_id = self
                    .stacktrace
                    .iter()
                    .rev()
                    .find_map(|frame| frame.lint_root)
                    .unwrap_or(lint_root);
                tcx.struct_span_lint_hir(
                    rustc_session::lint::builtin::CONST_ERR,
                    hir_id,
                    tcx.span,
                    |lint| finish(lint.build(message), Some(err_msg)),
                );
                ErrorHandled::Linted
            } else {
                // Report as hard error.
                finish(struct_error(tcx, message), Some(err_msg));
                ErrorHandled::Reported(ErrorReported)
            }
        }
    }
}

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

impl<'tcx> From<InterpError<'tcx>> for InterpErrorInfo<'tcx> {
    fn from(kind: InterpError<'tcx>) -> Self {
        let capture_backtrace = tls::with_context_opt(|ctxt| {
            if let Some(ctxt) = ctxt {
                *Lock::borrow(&ctxt.tcx.sess.ctfe_backtrace)
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
    /// Abort in case type errors are reached.
    TypeckError(ErrorReported),
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
            TypeckError(ErrorReported) => {
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
#[derive(Debug, Copy, Clone, RustcEncodable, RustcDecodable, HashStable)]
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
    /// An enum discriminant was set to a value which was outside the range of valid values.
    InvalidDiscriminant(ScalarMaybeUndef),
    /// Using a pointer-not-to-a-function as function pointer.
    InvalidFunctionPointer(Pointer),
    /// Using a string that is not valid UTF-8,
    InvalidStr(std::str::Utf8Error),
    /// Using uninitialized data where it is not allowed.
    InvalidUndefBytes(Option<Pointer>),
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
            InvalidDiscriminant(val) => write!(f, "enum value has invalid discriminant: {}", val),
            InvalidFunctionPointer(p) => {
                write!(f, "using {} as function pointer but it does not point to a function", p)
            }
            InvalidStr(err) => write!(f, "this string is not valid UTF-8: {}", err),
            InvalidUndefBytes(Some(p)) => write!(
                f,
                "reading uninitialized memory at {}, but this operation requires initialized memory",
                p
            ),
            InvalidUndefBytes(None) => write!(
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
    /// Accessing an unsupported foreign static.
    ReadForeignStatic(DefId),
    /// Could not find MIR for a function.
    NoMirFor(DefId),
    /// Encountered a pointer where we needed raw bytes.
    ReadPointerAsBytes,
    //
    // The variants below are only reachable from CTFE/const prop, miri will never emit them.
    //
    /// Encountered raw bytes where we needed a pointer.
    ReadBytesAsPointer,
}

impl fmt::Display for UnsupportedOpInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use UnsupportedOpInfo::*;
        match self {
            Unsupported(ref msg) => write!(f, "{}", msg),
            ReadForeignStatic(did) => {
                write!(f, "cannot read from foreign (extern) static {:?}", did)
            }
            NoMirFor(did) => write!(f, "no MIR body is available for {:?}", did),
            ReadPointerAsBytes => write!(f, "unable to turn pointer into raw bytes",),
            ReadBytesAsPointer => write!(f, "unable to turn bytes into a pointer"),
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
            | InterpError::UndefinedBehavior(UndefinedBehaviorInfo::Ub(_)) => true,
            _ => false,
        }
    }
}
