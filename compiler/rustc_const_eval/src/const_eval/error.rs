use std::{fmt, mem};

use rustc_errors::{Diag, E0080};
use rustc_middle::mir::AssertKind;
use rustc_middle::mir::interpret::{
    AllocId, Provenance, ReportedErrorInfo, UndefinedBehaviorInfo, UnsupportedOpInfo,
};
use rustc_middle::query::TyCtxtAt;
use rustc_middle::ty::ConstInt;
use rustc_middle::ty::layout::LayoutError;
use rustc_span::{DUMMY_SP, Span, Symbol};

use super::CompileTimeMachine;
use crate::errors::{self, FrameNote};
use crate::interpret::{
    CtfeProvenance, ErrorHandled, Frame, InterpCx, InterpErrorInfo, InterpErrorKind,
    MachineStopType, Pointer, err_inval, err_machine_stop,
};

/// The CTFE machine has some custom error kinds.
#[derive(Clone, Debug)]
pub enum ConstEvalErrKind {
    ConstAccessesMutGlobal,
    ModifiedGlobal,
    RecursiveStatic,
    AssertFailure(AssertKind<ConstInt>),
    Panic {
        msg: Symbol,
        line: u32,
        col: u32,
        file: Symbol,
    },
    WriteThroughImmutablePointer,
    /// Called `const_make_global` twice.
    ConstMakeGlobalPtrAlreadyMadeGlobal(AllocId),
    /// Called `const_make_global` on a non-heap pointer.
    ConstMakeGlobalPtrIsNonHeap(Pointer<Option<CtfeProvenance>>),
    /// Called `const_make_global` on a dangling pointer.
    ConstMakeGlobalWithDanglingPtr(Pointer<Option<CtfeProvenance>>),
    /// Called `const_make_global` on a pointer that does not start at the
    /// beginning of an object.
    ConstMakeGlobalWithOffset(Pointer<Option<CtfeProvenance>>),
}

impl fmt::Display for ConstEvalErrKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use ConstEvalErrKind::*;
        match self {
            ConstAccessesMutGlobal => write!(f, "constant accesses mutable global memory"),
            ModifiedGlobal => {
                write!(f, "modifying a static's initial value from another static's initializer")
            }
            Panic { msg, .. } => write!(f, "evaluation panicked: {msg}"),
            RecursiveStatic => {
                write!(f, "encountered static that tried to access itself during initialization")
            }
            AssertFailure(x) => write!(f, "{x}"),
            WriteThroughImmutablePointer => {
                write!(
                    f,
                    "writing through a pointer that was derived from a shared (immutable) reference"
                )
            }
            ConstMakeGlobalPtrAlreadyMadeGlobal(alloc) => {
                write!(
                    f,
                    "attempting to call `const_make_global` twice on the same allocation {alloc}"
                )
            }
            ConstMakeGlobalPtrIsNonHeap(ptr) => {
                write!(
                    f,
                    "pointer passed to `const_make_global` does not point to a heap allocation: {ptr}"
                )
            }
            ConstMakeGlobalWithDanglingPtr(ptr) => {
                write!(f, "pointer passed to `const_make_global` is dangling: {ptr}")
            }
            ConstMakeGlobalWithOffset(ptr) => {
                write!(f, "making {ptr} global which does not point to the beginning of an object")
            }
        }
    }
}

impl MachineStopType for ConstEvalErrKind {}

/// The errors become [`InterpErrorKind::MachineStop`] when being raised.
impl<'tcx> Into<InterpErrorInfo<'tcx>> for ConstEvalErrKind {
    fn into(self) -> InterpErrorInfo<'tcx> {
        err_machine_stop!(self).into()
    }
}

pub fn get_span_and_frames<'tcx>(
    tcx: TyCtxtAt<'tcx>,
    stack: &[Frame<'tcx, impl Provenance, impl Sized>],
) -> (Span, Vec<errors::FrameNote>) {
    let mut stacktrace = Frame::generate_stacktrace_from_stack(stack, *tcx);
    // Filter out `requires_caller_location` frames.
    stacktrace.retain(|frame| !frame.instance.def.requires_caller_location(*tcx));
    let span = stacktrace.last().map(|f| f.span).unwrap_or(tcx.span);

    let mut frames = Vec::new();

    // Add notes to the backtrace. Don't print a single-line backtrace though.
    if stacktrace.len() > 1 {
        // Helper closure to print duplicated lines.
        let mut add_frame = |mut frame: errors::FrameNote| {
            frames.push(errors::FrameNote { times: 0, ..frame.clone() });
            // Don't print [... additional calls ...] if the number of lines is small
            if frame.times < 3 {
                let times = frame.times;
                frame.times = 0;
                frames.extend(std::iter::repeat_n(frame, times as usize));
            } else {
                frames.push(frame);
            }
        };

        let mut last_frame: Option<errors::FrameNote> = None;
        for frame_info in &stacktrace {
            let frame = frame_info.as_note(*tcx);
            match last_frame.as_mut() {
                Some(last_frame)
                    if last_frame.span == frame.span
                        && last_frame.where_ == frame.where_
                        && last_frame.instance == frame.instance =>
                {
                    last_frame.times += 1;
                }
                Some(last_frame) => {
                    add_frame(mem::replace(last_frame, frame));
                }
                None => {
                    last_frame = Some(frame);
                }
            }
        }
        if let Some(frame) = last_frame {
            add_frame(frame);
        }
    }

    // In `rustc`, we present const-eval errors from the outer-most place first to the inner-most.
    // So we reverse the frames here. The first frame will be the same as the span from the current
    // `TyCtxtAt<'_>`, so we remove it as it would be redundant.
    frames.reverse();
    if frames.len() > 0 {
        frames.remove(0);
    }
    if let Some(last) = frames.last_mut()
        // If the span is not going to be printed, we don't want the span label for `is_last`.
        && tcx.sess.source_map().span_to_snippet(last.span.source_callsite()).is_ok()
    {
        last.has_label = true;
    }

    (span, frames)
}

/// Create a diagnostic for a const eval error.
///
/// This will use the `mk` function for adding more information to the error.
/// You can use it to add a stacktrace of current execution according to
/// `get_span_and_frames` or just give context on where the const eval error happened.
pub(super) fn report<'tcx>(
    ecx: &InterpCx<'tcx, CompileTimeMachine<'tcx>>,
    error: InterpErrorKind<'tcx>,
    mk: impl FnOnce(&mut Diag<'_>, Span, Vec<FrameNote>),
) -> ErrorHandled {
    let tcx = ecx.tcx.tcx;
    // Special handling for certain errors
    match error {
        // Don't emit a new diagnostic for these errors, they are already reported elsewhere or
        // should remain silent.
        err_inval!(AlreadyReported(info)) => ErrorHandled::Reported(info, DUMMY_SP),
        err_inval!(Layout(LayoutError::TooGeneric(_))) | err_inval!(TooGeneric) => {
            ErrorHandled::TooGeneric(DUMMY_SP)
        }
        err_inval!(Layout(LayoutError::ReferencesError(guar))) => {
            // This can occur in infallible promoteds e.g. when a non-existent type or field is
            // encountered.
            ErrorHandled::Reported(ReportedErrorInfo::allowed_in_infallible(guar), DUMMY_SP)
        }
        // Report remaining errors.
        _ => {
            let (span, frames) = super::get_span_and_frames(ecx.tcx, ecx.stack());
            let mut err = tcx.dcx().struct_span_err(span, error.to_string());
            err.code(E0080);
            if matches!(
                error,
                InterpErrorKind::UndefinedBehavior(UndefinedBehaviorInfo::ValidationError {
                    ptr_bytes_warning: true,
                    ..
                }) | InterpErrorKind::Unsupported(
                    UnsupportedOpInfo::ReadPointerAsInt(..)
                        | UnsupportedOpInfo::ReadPartialPointer(..)
                )
            ) {
                err.help("this code performed an operation that depends on the underlying bytes representing a pointer");
                err.help("the absolute address of a pointer is not known at compile-time, so such operations are not supported");
            }
            if let InterpErrorKind::UndefinedBehavior(UndefinedBehaviorInfo::InvalidUninitBytes(
                Some((alloc_id, _access)),
            )) = error
            {
                let bytes = ecx.print_alloc_bytes_for_diagnostics(alloc_id);
                let info = ecx.get_alloc_info(alloc_id);
                let raw_bytes = errors::RawBytesNote {
                    size: info.size.bytes(),
                    align: info.align.bytes(),
                    bytes,
                };
                err.subdiagnostic(raw_bytes);
            }

            // We allow invalid programs in infallible promoteds since invalid layouts can occur
            // anyway (e.g. due to size overflow). And we allow OOM as that can happen any time.
            let allowed_in_infallible = matches!(
                error,
                InterpErrorKind::ResourceExhaustion(_) | InterpErrorKind::InvalidProgram(_)
            );

            mk(&mut err, span, frames);
            let g = err.emit();
            let reported = if allowed_in_infallible {
                ReportedErrorInfo::allowed_in_infallible(g)
            } else {
                ReportedErrorInfo::const_eval_error(g)
            };
            ErrorHandled::Reported(reported, span)
        }
    }
}

/// Emit a lint from a const-eval situation, with a backtrace.
// Even if this is unused, please don't remove it -- chances are we will need to emit a lint during const-eval again in the future!
#[allow(unused)]
pub(super) fn lint<'tcx, L>(
    tcx: TyCtxtAt<'tcx>,
    machine: &CompileTimeMachine<'tcx>,
    lint: &'static rustc_session::lint::Lint,
    decorator: impl FnOnce(Vec<errors::FrameNote>) -> L,
) where
    L: for<'a> rustc_errors::Diagnostic<'a, ()>,
{
    let (span, frames) = get_span_and_frames(tcx, &machine.stack);

    tcx.emit_node_span_lint(lint, machine.best_lint_scope(*tcx), span, decorator(frames));
}
