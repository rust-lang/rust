use super::{CheckInAllocMsg, Pointer, RawConst, ScalarMaybeUndef};

use crate::hir::map::definitions::DefPathData;
use crate::mir;
use crate::ty::layout::{Align, LayoutError, Size};
use crate::ty::query::TyCtxtAt;
use crate::ty::{self, layout, Ty};

use backtrace::Backtrace;
use hir::GeneratorKind;
use rustc_errors::{struct_span_err, DiagnosticBuilder};
use rustc_hir as hir;
use rustc_macros::HashStable;
use rustc_span::symbol::Symbol;
use rustc_span::{Pos, Span};
use rustc_target::spec::abi::Abi;
use std::{any::Any, env, fmt};

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
pub type ConstEvalResult<'tcx> = Result<&'tcx ty::Const<'tcx>, ErrorHandled>;

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
            }
        , Some(lint_root)) {
            Ok(_) => {
                ErrorHandled::Reported
            }
            Err(err) => err,
        }
    }

   /// Sets the message passed in via `message`, then adds the span labels for you, before applying
   /// further modifications in `emit`. It's up to you to call emit(), stash(..), etc. within the
   /// `emit` method. If you don't need to do any additional processing, just use
   /// struct_generic.
    fn struct_generic(
        &self,
        tcx: TyCtxtAt<'tcx>,
        message: &str,
        emit: impl FnOnce(DiagnosticBuilder<'_>),
        lint_root: Option<hir::HirId>,
    ) -> Result<(), ErrorHandled> {
        let must_error = match self.error {
            InterpError::MachineStop(_) => bug!("CTFE does not stop"),
            err_inval!(Layout(LayoutError::Unknown(_))) | err_inval!(TooGeneric) => {
                return Err(ErrorHandled::TooGeneric);
            }
            err_inval!(TypeckError) => return Err(ErrorHandled::Reported),
            err_inval!(Layout(LayoutError::SizeOverflow(_))) => true,
            _ => false,
        };
        trace!("reporting const eval failure at {:?}", self.span);

        let add_span_labels = |err: &mut DiagnosticBuilder<'_>| {
            if !must_error {
                err.span_label(self.span, self.error.to_string());
            }
            // Skip the last, which is just the environment of the constant.  The stacktrace
            // is sometimes empty because we create "fake" eval contexts in CTFE to do work
            // on constant values.
            if self.stacktrace.len() > 0 {
                for frame_info in &self.stacktrace[..self.stacktrace.len() - 1] {
                    err.span_label(frame_info.call_site, frame_info.to_string());
                }
            }
        };

        if let (Some(lint_root), false) = (lint_root, must_error) {
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
                |lint| {
                    let mut err = lint.build(message);
                    add_span_labels(&mut err);
                    emit(err);
                },
            );
        } else {
            let mut err = if must_error {
                struct_error(tcx, &self.error.to_string())
            } else {
                struct_error(tcx, message)
            };
            add_span_labels(&mut err);
            emit(err);
        };
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

impl From<ErrorHandled> for InterpErrorInfo<'tcx> {
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
        let backtrace = match env::var("RUSTC_CTFE_BACKTRACE") {
            // Matching `RUST_BACKTRACE` -- we treat "0" the same as "not present".
            Ok(ref val) if val != "0" => {
                let mut backtrace = Backtrace::new_unresolved();

                if val == "immediate" {
                    // Print it now.
                    print_backtrace(&mut backtrace);
                    None
                } else {
                    Some(Box::new(backtrace))
                }
            }
            _ => None,
        };
        InterpErrorInfo { kind, backtrace }
    }
}

#[derive(Clone, RustcEncodable, RustcDecodable, HashStable, PartialEq)]
pub enum PanicInfo<O> {
    Panic { msg: Symbol, line: u32, col: u32, file: Symbol },
    BoundsCheck { len: O, index: O },
    Overflow(mir::BinOp),
    OverflowNeg,
    DivisionByZero,
    RemainderByZero,
    ResumedAfterReturn(GeneratorKind),
    ResumedAfterPanic(GeneratorKind),
}

/// Type for MIR `Assert` terminator error messages.
pub type AssertMessage<'tcx> = PanicInfo<mir::Operand<'tcx>>;

impl<O> PanicInfo<O> {
    /// Getting a description does not require `O` to be printable, and does not
    /// require allocation.
    /// The caller is expected to handle `Panic` and `BoundsCheck` separately.
    pub fn description(&self) -> &'static str {
        use PanicInfo::*;
        match self {
            Overflow(mir::BinOp::Add) => "attempt to add with overflow",
            Overflow(mir::BinOp::Sub) => "attempt to subtract with overflow",
            Overflow(mir::BinOp::Mul) => "attempt to multiply with overflow",
            Overflow(mir::BinOp::Div) => "attempt to divide with overflow",
            Overflow(mir::BinOp::Rem) => "attempt to calculate the remainder with overflow",
            OverflowNeg => "attempt to negate with overflow",
            Overflow(mir::BinOp::Shr) => "attempt to shift right with overflow",
            Overflow(mir::BinOp::Shl) => "attempt to shift left with overflow",
            Overflow(op) => bug!("{:?} cannot overflow", op),
            DivisionByZero => "attempt to divide by zero",
            RemainderByZero => "attempt to calculate the remainder with a divisor of zero",
            ResumedAfterReturn(GeneratorKind::Gen) => "generator resumed after completion",
            ResumedAfterReturn(GeneratorKind::Async(_)) => "`async fn` resumed after completion",
            ResumedAfterPanic(GeneratorKind::Gen) => "generator resumed after panicking",
            ResumedAfterPanic(GeneratorKind::Async(_)) => "`async fn` resumed after panicking",
            Panic { .. } | BoundsCheck { .. } => bug!("Unexpected PanicInfo"),
        }
    }
}

impl<O: fmt::Debug> fmt::Debug for PanicInfo<O> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use PanicInfo::*;
        match self {
            Panic { ref msg, line, col, ref file } => {
                write!(f, "the evaluated program panicked at '{}', {}:{}:{}", msg, file, line, col)
            }
            BoundsCheck { ref len, ref index } => {
                write!(f, "index out of bounds: the len is {:?} but the index is {:?}", len, index)
            }
            _ => write!(f, "{}", self.description()),
        }
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
}

impl fmt::Debug for InvalidProgramInfo<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use InvalidProgramInfo::*;
        match self {
            TooGeneric => write!(f, "encountered overly generic constant"),
            ReferencedConstant => write!(f, "referenced constant has errors"),
            TypeckError => write!(f, "encountered constants with type errors, stopping evaluation"),
            Layout(ref err) => write!(f, "{}", err),
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
    BoundsCheckFailed { len: u64, index: u64 },
    /// Something was divided by 0 (x / 0).
    DivisionByZero,
    /// Something was "remainded" by 0 (x % 0).
    RemainderByZero,
    /// Overflowing inbounds pointer arithmetic.
    PointerArithOverflow,
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
pub enum UnsupportedOpInfo<'tcx> {
    /// Free-form case. Only for errors that are never caught!
    Unsupported(String),

    /// When const-prop encounters a situation it does not support, it raises this error.
    /// This must not allocate for performance reasons.
    ConstPropUnsupported(&'tcx str),

    // -- Everything below is not categorized yet --
    FunctionAbiMismatch(Abi, Abi),
    FunctionArgMismatch(Ty<'tcx>, Ty<'tcx>),
    FunctionRetMismatch(Ty<'tcx>, Ty<'tcx>),
    FunctionArgCountMismatch,
    UnterminatedCString(Pointer),
    DanglingPointerDeref,
    DoubleFree,
    InvalidMemoryAccess,
    InvalidFunctionPointer,
    InvalidBool,
    PointerOutOfBounds {
        ptr: Pointer,
        msg: CheckInAllocMsg,
        allocation_size: Size,
    },
    InvalidNullPointerUsage,
    ReadPointerAsBytes,
    ReadBytesAsPointer,
    ReadForeignStatic,
    InvalidPointerMath,
    ReadUndefBytes(Size),
    DeadLocal,
    InvalidBoolOp(mir::BinOp),
    UnimplementedTraitSelection,
    CalledClosureAsFunction,
    NoMirFor(String),
    DerefFunctionPointer,
    ExecuteMemory,
    InvalidChar(u128),
    OutOfTls,
    TlsOutOfBounds,
    AlignmentCheckFailed {
        required: Align,
        has: Align,
    },
    ValidationFailure(String),
    VtableForArgumentlessMethod,
    ModifiedConstantMemory,
    ModifiedStatic,
    TypeNotPrimitive(Ty<'tcx>),
    ReallocatedWrongMemoryKind(String, String),
    DeallocatedWrongMemoryKind(String, String),
    ReallocateNonBasePtr,
    DeallocateNonBasePtr,
    IncorrectAllocationInformation(Size, Size, Align, Align),
    HeapAllocZeroBytes,
    HeapAllocNonPowerOfTwoAlignment(u64),
    ReadFromReturnPointer,
    PathNotFound(Vec<String>),
    TransmuteSizeDiff(Ty<'tcx>, Ty<'tcx>),
}

impl fmt::Debug for UnsupportedOpInfo<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use UnsupportedOpInfo::*;
        match self {
            PointerOutOfBounds { ptr, msg, allocation_size } => write!(
                f,
                "{} failed: pointer must be in-bounds at offset {}, \
                           but is outside bounds of allocation {} which has size {}",
                msg,
                ptr.offset.bytes(),
                ptr.alloc_id,
                allocation_size.bytes()
            ),
            ValidationFailure(ref err) => write!(f, "type validation failed: {}", err),
            NoMirFor(ref func) => write!(f, "no MIR for `{}`", func),
            FunctionAbiMismatch(caller_abi, callee_abi) => write!(
                f,
                "tried to call a function with ABI {:?} using caller ABI {:?}",
                callee_abi, caller_abi
            ),
            FunctionArgMismatch(caller_ty, callee_ty) => write!(
                f,
                "tried to call a function with argument of type {:?} \
                           passing data of type {:?}",
                callee_ty, caller_ty
            ),
            TransmuteSizeDiff(from_ty, to_ty) => write!(
                f,
                "tried to transmute from {:?} to {:?}, but their sizes differed",
                from_ty, to_ty
            ),
            FunctionRetMismatch(caller_ty, callee_ty) => write!(
                f,
                "tried to call a function with return type {:?} \
                           passing return place of type {:?}",
                callee_ty, caller_ty
            ),
            FunctionArgCountMismatch => {
                write!(f, "tried to call a function with incorrect number of arguments")
            }
            ReallocatedWrongMemoryKind(ref old, ref new) => {
                write!(f, "tried to reallocate memory from `{}` to `{}`", old, new)
            }
            DeallocatedWrongMemoryKind(ref old, ref new) => {
                write!(f, "tried to deallocate `{}` memory but gave `{}` as the kind", old, new)
            }
            InvalidChar(c) => {
                write!(f, "tried to interpret an invalid 32-bit value as a char: {}", c)
            }
            AlignmentCheckFailed { required, has } => write!(
                f,
                "tried to access memory with alignment {}, but alignment {} is required",
                has.bytes(),
                required.bytes()
            ),
            TypeNotPrimitive(ty) => write!(f, "expected primitive type, got {}", ty),
            PathNotFound(ref path) => write!(f, "cannot find path {:?}", path),
            IncorrectAllocationInformation(size, size2, align, align2) => write!(
                f,
                "incorrect alloc info: expected size {} and align {}, \
                           got size {} and align {}",
                size.bytes(),
                align.bytes(),
                size2.bytes(),
                align2.bytes()
            ),
            InvalidMemoryAccess => write!(f, "tried to access memory through an invalid pointer"),
            DanglingPointerDeref => write!(f, "dangling pointer was dereferenced"),
            DoubleFree => write!(f, "tried to deallocate dangling pointer"),
            InvalidFunctionPointer => {
                write!(f, "tried to use a function pointer after offsetting it")
            }
            InvalidBool => write!(f, "invalid boolean value read"),
            InvalidNullPointerUsage => write!(f, "invalid use of NULL pointer"),
            ReadPointerAsBytes => write!(
                f,
                "a raw memory access tried to access part of a pointer value as raw \
                    bytes"
            ),
            ReadBytesAsPointer => {
                write!(f, "a memory access tried to interpret some bytes as a pointer")
            }
            ReadForeignStatic => write!(f, "tried to read from foreign (extern) static"),
            InvalidPointerMath => write!(
                f,
                "attempted to do invalid arithmetic on pointers that would leak base \
                    addresses, e.g., comparing pointers into different allocations"
            ),
            DeadLocal => write!(f, "tried to access a dead local variable"),
            DerefFunctionPointer => write!(f, "tried to dereference a function pointer"),
            ExecuteMemory => write!(f, "tried to treat a memory pointer as a function pointer"),
            OutOfTls => write!(f, "reached the maximum number of representable TLS keys"),
            TlsOutOfBounds => write!(f, "accessed an invalid (unallocated) TLS key"),
            CalledClosureAsFunction => {
                write!(f, "tried to call a closure through a function pointer")
            }
            VtableForArgumentlessMethod => {
                write!(f, "tried to call a vtable function without arguments")
            }
            ModifiedConstantMemory => write!(f, "tried to modify constant memory"),
            ModifiedStatic => write!(
                f,
                "tried to modify a static's initial value from another static's \
                    initializer"
            ),
            ReallocateNonBasePtr => write!(
                f,
                "tried to reallocate with a pointer not to the beginning of an \
                    existing object"
            ),
            DeallocateNonBasePtr => write!(
                f,
                "tried to deallocate with a pointer not to the beginning of an \
                    existing object"
            ),
            HeapAllocZeroBytes => write!(f, "tried to re-, de- or allocate zero bytes on the heap"),
            ReadFromReturnPointer => write!(f, "tried to read from the return pointer"),
            UnimplementedTraitSelection => {
                write!(f, "there were unresolved type arguments during trait selection")
            }
            InvalidBoolOp(_) => write!(f, "invalid boolean operation"),
            UnterminatedCString(_) => write!(
                f,
                "attempted to get length of a null-terminated string, but no null \
                    found before end of allocation"
            ),
            ReadUndefBytes(_) => write!(f, "attempted to read undefined bytes"),
            HeapAllocNonPowerOfTwoAlignment(_) => write!(
                f,
                "tried to re-, de-, or allocate heap memory with alignment that is \
                    not a power of two"
            ),
            Unsupported(ref msg) => write!(f, "{}", msg),
            ConstPropUnsupported(ref msg) => {
                write!(f, "Constant propagation encountered an unsupported situation: {}", msg)
            }
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
    /// The program panicked.
    Panic(PanicInfo<u64>),
    /// The program caused undefined behavior.
    UndefinedBehavior(UndefinedBehaviorInfo),
    /// The program did something the interpreter does not support (some of these *might* be UB
    /// but the interpreter is not sure).
    Unsupported(UnsupportedOpInfo<'tcx>),
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
        write!(f, "{:?}", self)
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
            Panic(ref msg) => write!(f, "{:?}", msg),
            MachineStop(_) => write!(f, "machine caused execution to stop"),
        }
    }
}
