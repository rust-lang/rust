use std::cell::RefCell;
use std::fmt;

use log::trace;

use rustc_span::DUMMY_SP;

use crate::*;

/// Details of premature program termination.
pub enum TerminationInfo {
    Exit(i64),
    Abort(Option<String>),
    UnsupportedInIsolation(String),
    ExperimentalUb { msg: String, url: String },
    Deadlock,
}

impl fmt::Debug for TerminationInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use TerminationInfo::*;
        match self {
            Exit(code) =>
                write!(f, "the evaluated program completed with exit code {}", code),
            Abort(None) =>
                write!(f, "the evaluated program aborted execution"),
            Abort(Some(msg)) =>
                write!(f, "the evaluated program aborted execution: {}", msg),
            UnsupportedInIsolation(msg) =>
                write!(f, "{}", msg),
            ExperimentalUb { msg, .. } =>
                write!(f, "{}", msg),
            Deadlock =>
                write!(f, "the evaluated program deadlocked"),
        }
    }
}

impl MachineStopType for TerminationInfo {}

/// Miri specific diagnostics
pub enum NonHaltingDiagnostic {
    PoppedTrackedPointerTag(Item),
    CreatedAlloc(AllocId),
    FreedAlloc(AllocId),
}

/// Emit a custom diagnostic without going through the miri-engine machinery
pub fn report_error<'tcx, 'mir>(
    ecx: &InterpCx<'mir, 'tcx, Evaluator<'tcx>>,
    mut e: InterpErrorInfo<'tcx>,
) -> Option<i64> {
    use InterpError::*;

    let (title, helps) = match &e.kind {
        MachineStop(info) => {
            let info = info.downcast_ref::<TerminationInfo>().expect("invalid MachineStop payload");
            use TerminationInfo::*;
            let title = match info {
                Exit(code) => return Some(*code),
                Abort(_) =>
                    "abnormal termination",
                UnsupportedInIsolation(_) =>
                    "unsupported operation",
                ExperimentalUb { .. } =>
                    "Undefined Behavior",
                Deadlock => "deadlock",
            };
            let helps = match info {
                UnsupportedInIsolation(_) =>
                    vec![format!("pass the flag `-Zmiri-disable-isolation` to disable isolation")],
                ExperimentalUb { url, .. } =>
                    vec![
                        format!("this indicates a potential bug in the program: it performed an invalid operation, but the rules it violated are still experimental"),
                        format!("see {} for further information", url),
                    ],
                _ => vec![],
            };
            (title, helps)
        }
        _ => {
            let title = match e.kind {
                Unsupported(_) =>
                    "unsupported operation",
                UndefinedBehavior(_) =>
                    "Undefined Behavior",
                ResourceExhaustion(_) =>
                    "resource exhaustion",
                _ =>
                    bug!("This error should be impossible in Miri: {}", e),
            };
            let helps = match e.kind {
                Unsupported(UnsupportedOpInfo::NoMirFor(..)) =>
                    vec![format!("make sure to use a Miri sysroot, which you can prepare with `cargo miri setup`")],
                Unsupported(_) =>
                    vec![format!("this is likely not a bug in the program; it indicates that the program performed an operation that the interpreter does not support")],
                UndefinedBehavior(UndefinedBehaviorInfo::AlignmentCheckFailed { .. }) =>
                    vec![
                        format!("this usually indicates that your program performed an invalid operation and caused Undefined Behavior"),
                        format!("but alignment errors can also be false positives, see https://github.com/rust-lang/miri/issues/1074"),
                        format!("you can disable the alignment check with `-Zmiri-disable-alignment-check`, but that could hide true bugs")
                    ],
                UndefinedBehavior(_) =>
                    vec![
                        format!("this indicates a bug in the program: it performed an invalid operation, and caused Undefined Behavior"),
                        format!("see https://doc.rust-lang.org/nightly/reference/behavior-considered-undefined.html for further information"),
                    ],
                _ => vec![],
            };
            (title, helps)
        }
    };

    e.print_backtrace();
    let msg = e.to_string();
    report_msg(ecx, &format!("{}: {}", title, msg), msg, helps, true)
}

/// Report an error or note (depending on the `error` argument) at the current frame's current statement.
/// Also emits a full stacktrace of the interpreter stack.
fn report_msg<'tcx, 'mir>(
    ecx: &InterpCx<'mir, 'tcx, Evaluator<'tcx>>,
    title: &str,
    span_msg: String,
    mut helps: Vec<String>,
    error: bool,
) -> Option<i64> {
    let span = if let Some(frame) = ecx.stack().last() {
        frame.current_source_info().unwrap().span
    } else {
        DUMMY_SP
    };
    let mut err = if error {
        ecx.tcx.sess.struct_span_err(span, title)
    } else {
        ecx.tcx.sess.diagnostic().span_note_diag(span, title)
    };
    err.span_label(span, span_msg);
    if !helps.is_empty() {
        // Add visual separator before backtrace.
        helps.last_mut().unwrap().push_str("\n");
        for help in helps {
            err.help(&help);
        }
    }
    // Add backtrace
    let frames = ecx.generate_stacktrace();
    for (idx, frame_info) in frames.iter().enumerate() {
        let is_local = frame_info.instance.def_id().is_local();
        // No span for non-local frames and the first frame (which is the error site).
        if is_local && idx > 0 {
            err.span_note(frame_info.span, &frame_info.to_string());
        } else {
            err.note(&frame_info.to_string());
        }
    }

    err.emit();

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
                use NonHaltingDiagnostic::*;
                let msg = match e {
                    PoppedTrackedPointerTag(item) =>
                        format!("popped tracked tag for item {:?}", item),
                    CreatedAlloc(AllocId(id)) =>
                        format!("created allocation with id {}", id),
                    FreedAlloc(AllocId(id)) =>
                        format!("freed allocation with id {}", id),
                };
                report_msg(this, "tracking was triggered", msg, vec![], false);
            }
        });
    }
}
