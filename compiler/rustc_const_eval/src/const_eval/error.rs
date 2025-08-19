use std::mem;

use rustc_errors::{Diag, DiagArgName, DiagArgValue, DiagMessage, IntoDiagArg};
use rustc_middle::mir::AssertKind;
use rustc_middle::mir::interpret::{AllocId, Provenance, ReportedErrorInfo, UndefinedBehaviorInfo};
use rustc_middle::query::TyCtxtAt;
use rustc_middle::ty::ConstInt;
use rustc_middle::ty::layout::LayoutError;
use rustc_span::{Span, Symbol};

use super::CompileTimeMachine;
use crate::errors::{self, FrameNote, ReportErrorExt};
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

impl MachineStopType for ConstEvalErrKind {
    fn diagnostic_message(&self) -> DiagMessage {
        use ConstEvalErrKind::*;

        use crate::fluent_generated::*;
        match self {
            ConstAccessesMutGlobal => const_eval_const_accesses_mut_global,
            ModifiedGlobal => const_eval_modified_global,
            Panic { .. } => const_eval_panic,
            RecursiveStatic => const_eval_recursive_static,
            AssertFailure(x) => x.diagnostic_message(),
            WriteThroughImmutablePointer => const_eval_write_through_immutable_pointer,
            ConstMakeGlobalPtrAlreadyMadeGlobal { .. } => {
                const_eval_const_make_global_ptr_already_made_global
            }
            ConstMakeGlobalPtrIsNonHeap(_) => const_eval_const_make_global_ptr_is_non_heap,
            ConstMakeGlobalWithDanglingPtr(_) => const_eval_const_make_global_with_dangling_ptr,
            ConstMakeGlobalWithOffset(_) => const_eval_const_make_global_with_offset,
        }
    }
    fn add_args(self: Box<Self>, adder: &mut dyn FnMut(DiagArgName, DiagArgValue)) {
        use ConstEvalErrKind::*;
        match *self {
            RecursiveStatic
            | ConstAccessesMutGlobal
            | ModifiedGlobal
            | WriteThroughImmutablePointer => {}
            AssertFailure(kind) => kind.add_args(adder),
            Panic { msg, .. } => {
                adder("msg".into(), msg.into_diag_arg(&mut None));
            }
            ConstMakeGlobalPtrIsNonHeap(ptr)
            | ConstMakeGlobalWithOffset(ptr)
            | ConstMakeGlobalWithDanglingPtr(ptr) => {
                adder("ptr".into(), format!("{ptr:?}").into_diag_arg(&mut None));
            }
            ConstMakeGlobalPtrAlreadyMadeGlobal(alloc) => {
                adder("alloc".into(), alloc.into_diag_arg(&mut None));
            }
        }
    }
}

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
    let mut stacktrace = Frame::generate_stacktrace_from_stack(stack);
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
                frames.extend(std::iter::repeat(frame).take(times as usize));
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
pub(super) fn report<'tcx, C, F>(
    ecx: &InterpCx<'tcx, CompileTimeMachine<'tcx>>,
    error: InterpErrorKind<'tcx>,
    span: Span,
    get_span_and_frames: C,
    mk: F,
) -> ErrorHandled
where
    C: FnOnce() -> (Span, Vec<FrameNote>),
    F: FnOnce(&mut Diag<'_>, Span, Vec<FrameNote>),
{
    let tcx = ecx.tcx.tcx;
    // Special handling for certain errors
    match error {
        // Don't emit a new diagnostic for these errors, they are already reported elsewhere or
        // should remain silent.
        err_inval!(AlreadyReported(info)) => ErrorHandled::Reported(info, span),
        err_inval!(Layout(LayoutError::TooGeneric(_))) | err_inval!(TooGeneric) => {
            ErrorHandled::TooGeneric(span)
        }
        err_inval!(Layout(LayoutError::ReferencesError(guar))) => {
            // This can occur in infallible promoteds e.g. when a non-existent type or field is
            // encountered.
            ErrorHandled::Reported(ReportedErrorInfo::allowed_in_infallible(guar), span)
        }
        // Report remaining errors.
        _ => {
            let (our_span, frames) = get_span_and_frames();
            let span = span.substitute_dummy(our_span);
            let mut err = tcx.dcx().struct_span_err(our_span, error.diagnostic_message());
            // We allow invalid programs in infallible promoteds since invalid layouts can occur
            // anyway (e.g. due to size overflow). And we allow OOM as that can happen any time.
            let allowed_in_infallible = matches!(
                error,
                InterpErrorKind::ResourceExhaustion(_) | InterpErrorKind::InvalidProgram(_)
            );

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

            error.add_args(&mut err);

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
    L: for<'a> rustc_errors::LintDiagnostic<'a, ()>,
{
    let (span, frames) = get_span_and_frames(tcx, &machine.stack);

    tcx.emit_node_span_lint(lint, machine.best_lint_scope(*tcx), span, decorator(frames));
}
