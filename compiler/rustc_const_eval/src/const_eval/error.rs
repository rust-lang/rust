use std::mem;

use rustc_errors::{
    DiagnosticArgName, DiagnosticArgValue, DiagnosticMessage, IntoDiagnostic, IntoDiagnosticArg,
};
use rustc_hir::CRATE_HIR_ID;
use rustc_middle::mir::AssertKind;
use rustc_middle::query::TyCtxtAt;
use rustc_middle::ty::TyCtxt;
use rustc_middle::ty::{layout::LayoutError, ConstInt};
use rustc_span::{Span, Symbol, DUMMY_SP};

use super::{CompileTimeInterpreter, InterpCx};
use crate::errors::{self, FrameNote, ReportErrorExt};
use crate::interpret::{ErrorHandled, InterpError, InterpErrorInfo, MachineStopType};

/// The CTFE machine has some custom error kinds.
#[derive(Clone, Debug)]
pub enum ConstEvalErrKind {
    ConstAccessesMutGlobal,
    ModifiedGlobal,
    AssertFailure(AssertKind<ConstInt>),
    Panic { msg: Symbol, line: u32, col: u32, file: Symbol },
}

impl MachineStopType for ConstEvalErrKind {
    fn diagnostic_message(&self) -> DiagnosticMessage {
        use crate::fluent_generated::*;
        use ConstEvalErrKind::*;
        match self {
            ConstAccessesMutGlobal => const_eval_const_accesses_mut_global,
            ModifiedGlobal => const_eval_modified_global,
            Panic { .. } => const_eval_panic,
            AssertFailure(x) => x.diagnostic_message(),
        }
    }
    fn add_args(self: Box<Self>, adder: &mut dyn FnMut(DiagnosticArgName, DiagnosticArgValue)) {
        use ConstEvalErrKind::*;
        match *self {
            ConstAccessesMutGlobal | ModifiedGlobal => {}
            AssertFailure(kind) => kind.add_args(adder),
            Panic { msg, line, col, file } => {
                adder("msg".into(), msg.into_diagnostic_arg());
                adder("file".into(), file.into_diagnostic_arg());
                adder("line".into(), line.into_diagnostic_arg());
                adder("col".into(), col.into_diagnostic_arg());
            }
        }
    }
}

/// The errors become [`InterpError::MachineStop`] when being raised.
impl<'tcx> Into<InterpErrorInfo<'tcx>> for ConstEvalErrKind {
    fn into(self) -> InterpErrorInfo<'tcx> {
        err_machine_stop!(self).into()
    }
}

pub fn get_span_and_frames<'tcx, 'mir>(
    tcx: TyCtxtAt<'tcx>,
    machine: &CompileTimeInterpreter<'mir, 'tcx>,
) -> (Span, Vec<errors::FrameNote>)
where
    'tcx: 'mir,
{
    let mut stacktrace =
        InterpCx::<CompileTimeInterpreter<'mir, 'tcx>>::generate_stacktrace_from_stack(
            &machine.stack,
        );
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
    error: InterpError<'tcx>,
    span: Option<Span>,
    get_span_and_frames: C,
    mk: F,
) -> ErrorHandled
where
    C: FnOnce() -> (Span, Vec<FrameNote>),
    F: FnOnce(Span, Vec<FrameNote>) -> E,
    E: IntoDiagnostic<'tcx>,
{
    // Special handling for certain errors
    match error {
        // Don't emit a new diagnostic for these errors, they are already reported elsewhere or
        // should remain silent.
        err_inval!(Layout(LayoutError::Unknown(_))) | err_inval!(TooGeneric) => {
            ErrorHandled::TooGeneric(span.unwrap_or(DUMMY_SP))
        }
        err_inval!(AlreadyReported(guar)) => ErrorHandled::Reported(guar, span.unwrap_or(DUMMY_SP)),
        err_inval!(Layout(LayoutError::ReferencesError(guar))) => {
            ErrorHandled::Reported(guar.into(), span.unwrap_or(DUMMY_SP))
        }
        // Report remaining errors.
        _ => {
            let (our_span, frames) = get_span_and_frames();
            let span = span.unwrap_or(our_span);
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

/// Emit a lint from a const-eval situation.
// Even if this is unused, please don't remove it -- chances are we will need to emit a lint during const-eval again in the future!
pub(super) fn lint<'tcx, 'mir, L>(
    tcx: TyCtxtAt<'tcx>,
    machine: &CompileTimeInterpreter<'mir, 'tcx>,
    lint: &'static rustc_session::lint::Lint,
    decorator: impl FnOnce(Vec<errors::FrameNote>) -> L,
) where
    L: for<'a> rustc_errors::DecorateLint<'a, ()>,
{
    let (span, frames) = get_span_and_frames(tcx, machine);

    tcx.emit_node_span_lint(
        lint,
        // We use the root frame for this so the crate that defines the const defines whether the
        // lint is emitted.
        machine.stack.first().and_then(|frame| frame.lint_root()).unwrap_or(CRATE_HIR_ID),
        span,
        decorator(frames),
    );
}
