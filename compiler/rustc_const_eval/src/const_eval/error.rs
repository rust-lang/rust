use std::mem;

use rustc_errors::{DiagnosticArgValue, DiagnosticMessage, IntoDiagnostic, IntoDiagnosticArg};
use rustc_middle::mir::AssertKind;
use rustc_middle::ty::TyCtxt;
use rustc_middle::ty::{layout::LayoutError, ConstInt};
use rustc_span::source_map::Spanned;
use rustc_span::{ErrorGuaranteed, Span, Symbol};

use super::InterpCx;
use crate::errors::{self, FrameNote, ReportErrorExt};
use crate::interpret::{ErrorHandled, InterpError, InterpErrorInfo, Machine, MachineStopType};

/// The CTFE machine has some custom error kinds.
#[derive(Clone, Debug)]
pub enum ConstEvalErrKind {
    ConstAccessesStatic,
    ModifiedGlobal,
    AssertFailure(AssertKind<ConstInt>),
    Panic { msg: Symbol, line: u32, col: u32, file: Symbol },
    Abort(String),
}

impl MachineStopType for ConstEvalErrKind {
    fn diagnostic_message(&self) -> DiagnosticMessage {
        use crate::fluent_generated::*;
        use ConstEvalErrKind::*;
        match self {
            ConstAccessesStatic => const_eval_const_accesses_static,
            ModifiedGlobal => const_eval_modified_global,
            Panic { .. } => const_eval_panic,
            AssertFailure(x) => x.diagnostic_message(),
            Abort(msg) => msg.to_string().into(),
        }
    }
    fn add_args(
        self: Box<Self>,
        adder: &mut dyn FnMut(std::borrow::Cow<'static, str>, DiagnosticArgValue<'static>),
    ) {
        use ConstEvalErrKind::*;
        match *self {
            ConstAccessesStatic | ModifiedGlobal | Abort(_) => {}
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

// The errors become `MachineStop` with plain strings when being raised.
// `ConstEvalErr` (in `librustc_middle/mir/interpret/error.rs`) knows to
// handle these.
impl<'tcx> Into<InterpErrorInfo<'tcx>> for ConstEvalErrKind {
    fn into(self) -> InterpErrorInfo<'tcx> {
        err_machine_stop!(self).into()
    }
}

pub fn get_span_and_frames<'tcx, 'mir, M: Machine<'mir, 'tcx>>(
    ecx: &InterpCx<'mir, 'tcx, M>,
) -> (Span, Vec<errors::FrameNote>)
where
    'tcx: 'mir,
{
    let mut stacktrace = ecx.generate_stacktrace();
    // Filter out `requires_caller_location` frames.
    stacktrace.retain(|frame| !frame.instance.def.requires_caller_location(*ecx.tcx));
    let span = stacktrace.first().map(|f| f.span).unwrap_or(ecx.tcx.span);

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
            let frame = frame_info.as_note(*ecx.tcx);
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
    E: IntoDiagnostic<'tcx, ErrorGuaranteed>,
{
    // Special handling for certain errors
    match error {
        // Don't emit a new diagnostic for these errors
        err_inval!(Layout(LayoutError::Unknown(_))) | err_inval!(TooGeneric) => {
            ErrorHandled::TooGeneric
        }
        err_inval!(AlreadyReported(error_reported)) => ErrorHandled::Reported(error_reported),
        err_inval!(Layout(layout_error @ LayoutError::SizeOverflow(_))) => {
            // We must *always* hard error on these, even if the caller wants just a lint.
            // The `message` makes little sense here, this is a more serious error than the
            // caller thinks anyway.
            // See <https://github.com/rust-lang/rust/pull/63152>.
            let (our_span, frames) = get_span_and_frames();
            let span = span.unwrap_or(our_span);
            let mut err =
                tcx.sess.create_err(Spanned { span, node: layout_error.into_diagnostic() });
            err.code(rustc_errors::error_code!(E0080));
            let Some((mut err, handler)) = err.into_diagnostic() else {
                panic!("did not emit diag");
            };
            for frame in frames {
                err.eager_subdiagnostic(handler, frame);
            }

            ErrorHandled::Reported(handler.emit_diagnostic(&mut err).unwrap().into())
        }
        _ => {
            // Report as hard error.
            let (our_span, frames) = get_span_and_frames();
            let span = span.unwrap_or(our_span);
            let err = mk(span, frames);
            let mut err = tcx.sess.create_err(err);

            let msg = error.diagnostic_message();
            error.add_args(&tcx.sess.parse_sess.span_diagnostic, &mut err);

            // Use *our* span to label the interp error
            err.span_label(our_span, msg);
            ErrorHandled::Reported(err.emit().into())
        }
    }
}
