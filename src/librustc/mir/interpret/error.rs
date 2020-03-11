use super::{AllocId, CheckInAllocMsg, Pointer, RawConst, ScalarMaybeUndef};

use crate::hir::map::definitions::DefPathData;
use crate::mir::interpret::ConstValue;
use crate::ty::layout::{Align, LayoutError, Size};
use crate::ty::query::TyCtxtAt;
use crate::ty::tls;
use crate::ty::{self, layout, Ty};

use backtrace::Backtrace;
use rustc_data_structures::sync::Lock;
use rustc_errors::{struct_span_err, DiagnosticBuilder};
use rustc_hir as hir;
use rustc_macros::HashStable;
use rustc_session::CtfeBacktrace;
use rustc_span::{def_id::DefId, Pos, Span};
use std::{any::Any, fmt};

#[derive(Debug, Copy, Clone, PartialEq, Eq, HashStable, RustcEncodable, RustcDecodable)]
pub enum ErrorHandled {
    /// Already reported a lint or an error for this evaluation.
    Reported,
    /// Don't emit an error, the evaluation failed because the MIR was generic
    /// and the substs didn't fully monomorphize it.
    TooGeneric,
}

impl ErrorHandled {
    pub fn assert_reported(self) {
        match self {
            ErrorHandled::Reported => {}
            ErrorHandled::TooGeneric => bug!(
                "MIR interpretation failed without reporting an error \
                 even though it was fully monomorphized"
            ),
        }
    }
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
    /// This span is in the caller.
    pub call_site: Span,
    pub instance: ty::Instance<'tcx>,
    pub lint_root: Option<hir::HirId>,
}

impl<'tcx> fmt::Display for FrameInfo<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        ty::tls::with(|tcx| {
            if tcx.def_key(self.instance.def_id()).disambiguated_data.data
                == DefPathData::ClosureExpr
            {
                write!(f, "inside call to closure")?;
            } else {
                write!(f, "inside call to `{}`", self.instance)?;
            }
            if !self.call_site.is_dummy() {
                let lo = tcx.sess.source_map().lookup_char_pos(self.call_site.lo());
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
    ) -> Result<(), ErrorHandled> {
        self.struct_generic(tcx, message, emit, None)
    }

    pub fn report_as_error(&self, tcx: TyCtxtAt<'tcx>, message: &str) -> ErrorHandled {
        match self.struct_error(tcx, message, |mut e| e.emit()) {
            Ok(_) => ErrorHandled::Reported,
            Err(x) => x,
        }
    }

    pub fn report_as_lint(
        &self,
        tcx: TyCtxtAt<'tcx>,
        message: &str,
        lint_root: hir::HirId,
        span: Option<Span>,
    ) -> ErrorHandled {
        match self.struct_generic(
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
        ) {
            Ok(_) => ErrorHandled::Reported,
            Err(err) => err,
        }
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
    ) -> Result<(), ErrorHandled> {
        let must_error = match self.error {
            err_inval!(Layout(LayoutError::Unknown(_))) | err_inval!(TooGeneric) => {
                return Err(ErrorHandled::TooGeneric);
            }
            err_inval!(TypeckError) => return Err(ErrorHandled::Reported),
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
            // Add spans for the stacktrace.
            // Skip the last, which is just the environment of the constant.  The stacktrace
            // is sometimes empty because we create "fake" eval contexts in CTFE to do work
            // on constant values.
            if !self.stacktrace.is_empty() {
                for frame_info in &self.stacktrace[..self.stacktrace.len() - 1] {
                    err.span_label(frame_info.call_site, frame_info.to_string());
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
        } else {
            // Regular case.
            if let Some(lint_root) = lint_root {
                // Report as lint.
                let hir_id = self
                    .stacktrace
                    .iter()
                    .rev()
                    .filter_map(|frame| frame.lint_root)
                    .next()
                    .unwrap_or(lint_root);
                tcx.struct_span_lint_hir(
                    rustc_session::lint::builtin::CONST_ERR,
                    hir_id,
                    tcx.span,
                    |lint| finish(lint.build(message), Some(err_msg)),
                );
            } else {
                // Report as hard error.
                finish(struct_error(tcx, message), Some(err_msg));
            }
        }
        Ok(())
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
    pub fn print_backtrace(&mut self) {
        if let Some(ref mut backtrace) = self.backtrace {
            print_backtrace(&mut *backtrace);
        }
    }
}

fn print_backtrace(backtrace: &mut Backtrace) {
    backtrace.resolve();
    eprintln!("\n\nAn error occurred in miri:\n{:?}", backtrace);
}

impl From<ErrorHandled> for InterpErrorInfo<'_> {
    fn from(err: ErrorHandled) -> Self {
        match err {
            ErrorHandled::Reported => err_inval!(ReferencedConstant),
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
            CtfeBacktrace::Capture => Some(Box::new(Backtrace::new_unresolved())),
            CtfeBacktrace::Immediate => {
                // Print it now.
                let mut backtrace = Backtrace::new_unresolved();
                print_backtrace(&mut backtrace);
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
    TypeckError,
    /// An error occurred during layout computation.
    Layout(layout::LayoutError<'tcx>),
    /// An invalid transmute happened.
    TransmuteSizeDiff(Ty<'tcx>, Ty<'tcx>),
}

impl fmt::Debug for InvalidProgramInfo<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use InvalidProgramInfo::*;
        match self {
            TooGeneric => write!(f, "encountered overly generic constant"),
            ReferencedConstant => write!(f, "referenced constant has errors"),
            TypeckError => write!(f, "encountered constants with type errors, stopping evaluation"),
            Layout(ref err) => write!(f, "{}", err),
            TransmuteSizeDiff(from_ty, to_ty) => write!(
                f,
                "tried to transmute from {:?} to {:?}, but their sizes differed",
                from_ty, to_ty
            ),
        }
    }
}

/// Error information for when the program caused Undefined Behavior.
pub enum UndefinedBehaviorInfo {
    /// Free-form case. Only for errors that are never caught!
    Ub(String),
    /// Free-form case for experimental UB. Only for errors that are never caught!
    UbExperimental(String),
    /// Unreachable code was executed.
    Unreachable,
    /// An enum discriminant was set to a value which was outside the range of valid values.
    InvalidDiscriminant(ScalarMaybeUndef),
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
    /// Used a pointer with bad alignment.
    AlignmentCheckFailed {
        required: Align,
        has: Align,
    },
    /// Using an integer as a pointer in the wrong way.
    InvalidIntPointerUsage(u64),
    /// Writing to read-only memory.
    WriteToReadOnly(AllocId),
    /// Using a pointer-not-to-a-function as function pointer.
    InvalidFunctionPointer(Pointer),
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
    /// Using uninitialized data where it is not allowed.
    InvalidUndefBytes(Option<Pointer>),
    /// Working with a local that is not currently live.
    DeadLocal,
    /// Trying to read from the return place of a function.
    ReadFromReturnPlace,
}

impl fmt::Debug for UndefinedBehaviorInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use UndefinedBehaviorInfo::*;
        match self {
            Ub(msg) | UbExperimental(msg) => write!(f, "{}", msg),
            Unreachable => write!(f, "entering unreachable code"),
            InvalidDiscriminant(val) => write!(f, "encountering invalid enum discriminant {}", val),
            BoundsCheckFailed { ref len, ref index } => write!(
                f,
                "indexing out of bounds: the len is {:?} but the index is {:?}",
                len, index
            ),
            DivisionByZero => write!(f, "dividing by zero"),
            RemainderByZero => write!(f, "calculating the remainder with a divisor of zero"),
            PointerArithOverflow => write!(f, "overflowing in-bounds pointer arithmetic"),
            InvalidMeta(msg) => write!(f, "invalid metadata in wide pointer: {}", msg),
            UnterminatedCString(p) => write!(
                f,
                "reading a null-terminated string starting at {:?} with no null found before end of allocation",
                p,
            ),
            PointerUseAfterFree(a) => {
                write!(f, "pointer to {:?} was dereferenced after this allocation got freed", a)
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
            InvalidIntPointerUsage(0) => write!(f, "invalid use of NULL pointer"),
            InvalidIntPointerUsage(i) => write!(f, "invalid use of {} as a pointer", i),
            AlignmentCheckFailed { required, has } => write!(
                f,
                "accessing memory with alignment {}, but alignment {} is required",
                has.bytes(),
                required.bytes()
            ),
            WriteToReadOnly(a) => write!(f, "writing to {:?} which is read-only", a),
            InvalidFunctionPointer(p) => {
                write!(f, "using {:?} as function pointer but it does not point to a function", p)
            }
            DerefFunctionPointer(a) => write!(f, "accessing {:?} which contains a function", a),
            ValidationFailure(ref err) => write!(f, "type validation failed: {}", err),
            InvalidBool(b) => write!(f, "interpreting an invalid 8-bit value as a bool: {}", b),
            InvalidChar(c) => write!(f, "interpreting an invalid 32-bit value as a char: {}", c),
            InvalidUndefBytes(Some(p)) => write!(
                f,
                "reading uninitialized memory at {:?}, but this operation requires initialized memory",
                p
            ),
            InvalidUndefBytes(None) => write!(
                f,
                "using uninitialized data, but this operation requires initialized memory"
            ),
            DeadLocal => write!(f, "accessing a dead local variable"),
            ReadFromReturnPlace => write!(f, "tried to read from the return place"),
        }
    }
}

/// Error information for when the program did something that might (or might not) be correct
/// to do according to the Rust spec, but due to limitations in the interpreter, the
/// operation could not be carried out. These limitations can differ between CTFE and the
/// Miri engine, e.g., CTFE does not support casting pointers to "real" integers.
///
/// Currently, we also use this as fall-back error kind for errors that have not been
/// categorized yet.
pub enum UnsupportedOpInfo {
    /// Free-form case. Only for errors that are never caught!
    Unsupported(String),
    /// When const-prop encounters a situation it does not support, it raises this error.
    /// This must not allocate for performance reasons (hence `str`, not `String`).
    ConstPropUnsupported(&'static str),
    /// Accessing an unsupported foreign static.
    ReadForeignStatic(DefId),
    /// Could not find MIR for a function.
    NoMirFor(DefId),
    /// Modified a static during const-eval.
    /// FIXME: move this to `ConstEvalErrKind` through a machine hook.
    ModifiedStatic,
    /// Encountered a pointer where we needed raw bytes.
    ReadPointerAsBytes,
    /// Encountered raw bytes where we needed a pointer.
    ReadBytesAsPointer,
}

impl fmt::Debug for UnsupportedOpInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use UnsupportedOpInfo::*;
        match self {
            Unsupported(ref msg) => write!(f, "{}", msg),
            ConstPropUnsupported(ref msg) => {
                write!(f, "Constant propagation encountered an unsupported situation: {}", msg)
            }
            ReadForeignStatic(did) => {
                write!(f, "tried to read from foreign (extern) static {:?}", did)
            }
            NoMirFor(did) => write!(f, "could not load MIR for {:?}", did),
            ModifiedStatic => write!(
                f,
                "tried to modify a static's initial value from another static's \
                    initializer"
            ),

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
    /// The program ran into an infinite loop.
    InfiniteLoop,
}

impl fmt::Debug for ResourceExhaustionInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use ResourceExhaustionInfo::*;
        match self {
            StackFrameLimitReached => {
                write!(f, "reached the configured maximum number of stack frames")
            }
            InfiniteLoop => write!(
                f,
                "duplicate interpreter state observed here, const evaluation will never \
                    terminate"
            ),
        }
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
    MachineStop(Box<dyn Any + Send>),
}

pub type InterpResult<'tcx, T = ()> = Result<T, InterpErrorInfo<'tcx>>;

impl fmt::Display for InterpError<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Forward `Display` to `Debug`.
        fmt::Debug::fmt(self, f)
    }
}

impl fmt::Debug for InterpError<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use InterpError::*;
        match *self {
            Unsupported(ref msg) => write!(f, "{:?}", msg),
            InvalidProgram(ref msg) => write!(f, "{:?}", msg),
            UndefinedBehavior(ref msg) => write!(f, "{:?}", msg),
            ResourceExhaustion(ref msg) => write!(f, "{:?}", msg),
            MachineStop(_) => bug!("unhandled MachineStop"),
        }
    }
}

impl InterpError<'_> {
    /// Some errors allocate to be created as they contain free-form strings.
    /// And sometimes we want to be sure that did not happen as it is a
    /// waste of resources.
    pub fn allocates(&self) -> bool {
        match self {
            InterpError::MachineStop(_)
            | InterpError::Unsupported(UnsupportedOpInfo::Unsupported(_))
            | InterpError::UndefinedBehavior(UndefinedBehaviorInfo::ValidationFailure(_))
            | InterpError::UndefinedBehavior(UndefinedBehaviorInfo::Ub(_))
            | InterpError::UndefinedBehavior(UndefinedBehaviorInfo::UbExperimental(_)) => true,
            _ => false,
        }
    }
}
