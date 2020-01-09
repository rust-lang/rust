use rustc_mir::interpret::InterpErrorInfo;
use std::cell::RefCell;

use crate::*;

/// Miri specific diagnostics
pub enum NonHaltingDiagnostic {
    PoppedTrackedPointerTag(Item),
}

/// Emit a custom diagnostic without going through the miri-engine machinery
pub fn report_diagnostic<'tcx, 'mir>(
    ecx: &InterpCx<'mir, 'tcx, Evaluator<'tcx>>,
    mut e: InterpErrorInfo<'tcx>,
) -> Option<i64> {
    // Special treatment for some error kinds
    let msg = match e.kind {
        InterpError::MachineStop(ref info) => {
            let info = info.downcast_ref::<TerminationInfo>().expect("invalid MachineStop payload");
            match info {
                TerminationInfo::Exit(code) => return Some(*code),
                TerminationInfo::Abort => format!("the evaluated program aborted execution"),
            }
        }
        err_unsup!(NoMirFor(..)) => format!(
            "{}. Did you set `MIRI_SYSROOT` to a Miri-enabled sysroot? You can prepare one with `cargo miri setup`.",
            e
        ),
        InterpError::InvalidProgram(_) => bug!("This error should be impossible in Miri: {}", e),
        _ => e.to_string(),
    };
    e.print_backtrace();
    report_msg(ecx, msg, true)
}

/// Report an error or note (depending on the `error` argument) at the current frame's current statement.
/// Also emits a full stacktrace of the interpreter stack.
pub fn report_msg<'tcx, 'mir>(
    ecx: &InterpCx<'mir, 'tcx, Evaluator<'tcx>>,
    msg: String,
    error: bool,
) -> Option<i64> {
    if let Some(frame) = ecx.stack().last() {
        let span = frame.current_source_info().unwrap().span;

        let mut err = if error {
            let msg = format!("Miri evaluation error: {}", msg);
            ecx.tcx.sess.struct_span_err(span, msg.as_str())
        } else {
            ecx.tcx.sess.diagnostic().span_note_diag(span, msg.as_str())
        };
        let frames = ecx.generate_stacktrace(None);
        err.span_label(span, msg);
        // We iterate with indices because we need to look at the next frame (the caller).
        for idx in 0..frames.len() {
            let frame_info = &frames[idx];
            let call_site_is_local = frames
                .get(idx + 1)
                .map_or(false, |caller_info| caller_info.instance.def_id().is_local());
            if call_site_is_local {
                err.span_note(frame_info.call_site, &frame_info.to_string());
            } else {
                err.note(&frame_info.to_string());
            }
        }
        err.emit();
    } else {
        ecx.tcx.sess.err(&msg);
    }

    for (i, frame) in ecx.stack().iter().enumerate() {
        trace!("-------------------");
        trace!("Frame {}", i);
        trace!("    return: {:?}", frame.return_place.map(|p| *p));
        for (i, local) in frame.locals.iter().enumerate() {
            trace!("    local {}: {:?}", i, local.value);
        }
    }
    // Let the reported error determine the return code.
    return None;
}

thread_local! {
    static DIAGNOSTICS: RefCell<Vec<NonHaltingDiagnostic>> = RefCell::new(Vec::new());
}

/// Schedule a diagnostic for emitting. This function works even if you have no `InterpCx` available.
/// The diagnostic will be emitted after the current interpreter step is finished.
pub fn register_diagnostic(e: NonHaltingDiagnostic) {
    DIAGNOSTICS.with(|diagnostics| diagnostics.borrow_mut().push(e));
}

impl<'mir, 'tcx> EvalContextExt<'mir, 'tcx> for crate::MiriEvalContext<'mir, 'tcx> {}
pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriEvalContextExt<'mir, 'tcx> {
    /// Emit all diagnostics that were registed with `register_diagnostics`
    fn process_diagnostics(&self) {
        let this = self.eval_context_ref();
        DIAGNOSTICS.with(|diagnostics| {
            for e in diagnostics.borrow_mut().drain(..) {
                let msg = match e {
                    NonHaltingDiagnostic::PoppedTrackedPointerTag(item) =>
                        format!("popped tracked tag for item {:?}", item),
                };
                report_msg(this, msg, false);
            }
        });
    }
}
