use std::mem;

use rustc_errors::{DiagArgName, DiagArgValue, DiagMessage, Diagnostic, IntoDiagArg};
use rustc_middle::mir::AssertKind;
use rustc_middle::mir::interpret::{Provenance, ReportedErrorInfo};
use rustc_middle::query::TyCtxtAt;
use rustc_middle::ty::layout::LayoutError;
use rustc_middle::ty::{ConstInt, TyCtxt};
use rustc_span::{Span, Symbol};

use super::CompileTimeMachine;
use crate::errors::{self, FrameNote, ReportErrorExt};
use crate::interpret::{
    ErrorHandled, Frame, InterpErrorInfo, InterpErrorKind, MachineStopType, err_inval,
    err_machine_stop,
};

/// The CTFE machine has some custom error kinds.
#[derive(Clone, Debug)]
pub enum ConstEvalErrKind {
    ConstAccessesMutGlobal,
    ModifiedGlobal,
    RecursiveStatic,
    AssertFailure(AssertKind<ConstInt>),
    Panic { msg: Symbol, line: u32, col: u32, file: Symbol },
    WriteThroughImmutablePointer,
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
            Panic { msg, line, col, file } => {
                adder("msg".into(), msg.into_diag_arg());
                adder("file".into(), file.into_diag_arg());
                adder("line".into(), line.into_diag_arg());
                adder("col".into(), col.into_diag_arg());
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
    let span = stacktrace.first().map(|f| f.span).unwrap_or(tcx.span);

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

    (span, frames)
}

/// Create a diagnostic for a const eval error.
///
/// This will use the `mk` function for creating the error which will get passed labels according to
/// the `InterpError` and the span and a stacktrace of current execution according to
/// `get_span_and_frames`.
pub(super) fn report<'tcx, C, F, E>(
    tcx: TyCtxt<'tcx>,
    error: InterpErrorKind<'tcx>,
    span: Span,
    get_span_and_frames: C,
    mk: F,
) -> ErrorHandled
where
    C: FnOnce() -> (Span, Vec<FrameNote>),
    F: FnOnce(Span, Vec<FrameNote>) -> E,
    E: Diagnostic<'tcx>,
{
    // Special handling for certain errors
    match error {
        // Don't emit a new diagnostic for these errors, they are already reported elsewhere or
        // should remain silent.
        err_inval!(Layout(LayoutError::Unknown(_))) | err_inval!(TooGeneric) => {
            ErrorHandled::TooGeneric(span)
        }
        err_inval!(AlreadyReported(guar)) => ErrorHandled::Reported(guar, span),
        err_inval!(Layout(LayoutError::ReferencesError(guar))) => {
            ErrorHandled::Reported(ReportedErrorInfo::tainted_by_errors(guar), span)
        }
        // Report remaining errors.
        _ => {
            let (our_span, frames) = get_span_and_frames();
            let span = span.substitute_dummy(our_span);
            let err = mk(span, frames);
            let mut err = tcx.dcx().create_err(err);

            let msg = error.diagnostic_message();
            error.add_args(&mut err);

            // Use *our* span to label the interp error
            err.span_label(our_span, msg);
            ErrorHandled::Reported(err.emit().into(), span)
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
