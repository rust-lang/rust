use std::cell::RefCell;
use std::fmt;
use std::num::NonZeroU64;

use log::trace;

use rustc_middle::ty::{self, TyCtxt};
use rustc_span::{source_map::DUMMY_SP, Span, SpanData, Symbol};

use crate::*;

/// Details of premature program termination.
pub enum TerminationInfo {
    Exit(i64),
    Abort(String),
    UnsupportedInIsolation(String),
    ExperimentalUb {
        msg: String,
        url: String,
    },
    Deadlock,
    MultipleSymbolDefinitions {
        link_name: Symbol,
        first: SpanData,
        first_crate: Symbol,
        second: SpanData,
        second_crate: Symbol,
    },
}

impl fmt::Display for TerminationInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use TerminationInfo::*;
        match self {
            Exit(code) => write!(f, "the evaluated program completed with exit code {}", code),
            Abort(msg) => write!(f, "{}", msg),
            UnsupportedInIsolation(msg) => write!(f, "{}", msg),
            ExperimentalUb { msg, .. } => write!(f, "{}", msg),
            Deadlock => write!(f, "the evaluated program deadlocked"),
            MultipleSymbolDefinitions { link_name, .. } =>
                write!(f, "multiple definitions of symbol `{}`", link_name),
        }
    }
}

impl MachineStopType for TerminationInfo {}

/// Miri specific diagnostics
pub enum NonHaltingDiagnostic {
    CreatedPointerTag(NonZeroU64),
    PoppedPointerTag(Item),
    CreatedCallId(CallId),
    CreatedAlloc(AllocId),
    FreedAlloc(AllocId),
    RejectedIsolatedOp(String),
}

/// Level of Miri specific diagnostics
enum DiagLevel {
    Error,
    Warning,
    Note,
}

/// Emit a custom diagnostic without going through the miri-engine machinery
pub fn report_error<'tcx, 'mir>(
    ecx: &InterpCx<'mir, 'tcx, Evaluator<'mir, 'tcx>>,
    e: InterpErrorInfo<'tcx>,
) -> Option<i64> {
    use InterpError::*;

    let (title, helps) = match &e.kind() {
        MachineStop(info) => {
            let info = info.downcast_ref::<TerminationInfo>().expect("invalid MachineStop payload");
            use TerminationInfo::*;
            let title = match info {
                Exit(code) => return Some(*code),
                Abort(_) => Some("abnormal termination"),
                UnsupportedInIsolation(_) => Some("unsupported operation"),
                ExperimentalUb { .. } => Some("Undefined Behavior"),
                Deadlock => Some("deadlock"),
                MultipleSymbolDefinitions { .. } => None,
            };
            #[rustfmt::skip]
            let helps = match info {
                UnsupportedInIsolation(_) =>
                    vec![
                        (None, format!("pass the flag `-Zmiri-disable-isolation` to disable isolation;")),
                        (None, format!("or pass `-Zmiri-isolation-error=warn to configure Miri to return an error code from isolated operations (if supported for that operation) and continue with a warning")),
                    ],
                ExperimentalUb { url, .. } =>
                    vec![
                        (None, format!("this indicates a potential bug in the program: it performed an invalid operation, but the rules it violated are still experimental")),
                        (None, format!("see {} for further information", url)),
                    ],
                MultipleSymbolDefinitions { first, first_crate, second, second_crate, .. } =>
                    vec![
                        (Some(*first), format!("it's first defined here, in crate `{}`", first_crate)),
                        (Some(*second), format!("then it's defined here again, in crate `{}`", second_crate)),
                    ],
                _ => vec![],
            };
            (title, helps)
        }
        _ => {
            #[rustfmt::skip]
            let title = match e.kind() {
                Unsupported(_) =>
                    "unsupported operation",
                UndefinedBehavior(_) =>
                    "Undefined Behavior",
                ResourceExhaustion(_) =>
                    "resource exhaustion",
                InvalidProgram(InvalidProgramInfo::ReferencedConstant) =>
                    "post-monomorphization error",
                InvalidProgram(InvalidProgramInfo::AlreadyReported(_)) =>
                    "error occurred",
                kind =>
                    bug!("This error should be impossible in Miri: {:?}", kind),
            };
            #[rustfmt::skip]
            let helps = match e.kind() {
                Unsupported(UnsupportedOpInfo::NoMirFor(..)) =>
                    vec![(None, format!("make sure to use a Miri sysroot, which you can prepare with `cargo miri setup`"))],
                Unsupported(UnsupportedOpInfo::ReadBytesAsPointer | UnsupportedOpInfo::ThreadLocalStatic(_) | UnsupportedOpInfo::ReadExternStatic(_)) =>
                    panic!("Error should never be raised by Miri: {:?}", e.kind()),
                Unsupported(_) =>
                    vec![(None, format!("this is likely not a bug in the program; it indicates that the program performed an operation that the interpreter does not support"))],
                UndefinedBehavior(UndefinedBehaviorInfo::AlignmentCheckFailed { .. })
                    if ecx.memory.extra.check_alignment == AlignmentCheck::Symbolic
                =>
                    vec![
                        (None, format!("this usually indicates that your program performed an invalid operation and caused Undefined Behavior")),
                        (None, format!("but due to `-Zmiri-symbolic-alignment-check`, alignment errors can also be false positives")),
                    ],
                UndefinedBehavior(_) =>
                    vec![
                        (None, format!("this indicates a bug in the program: it performed an invalid operation, and caused Undefined Behavior")),
                        (None, format!("see https://doc.rust-lang.org/nightly/reference/behavior-considered-undefined.html for further information")),
                    ],
                _ => vec![],
            };
            (Some(title), helps)
        }
    };

    e.print_backtrace();
    let msg = e.to_string();
    report_msg(
        *ecx.tcx,
        DiagLevel::Error,
        &if let Some(title) = title { format!("{}: {}", title, msg) } else { msg.clone() },
        msg,
        helps,
        &ecx.generate_stacktrace(),
    );

    // Debug-dump all locals.
    for (i, frame) in ecx.active_thread_stack().iter().enumerate() {
        trace!("-------------------");
        trace!("Frame {}", i);
        trace!("    return: {:?}", frame.return_place.map(|p| *p));
        for (i, local) in frame.locals.iter().enumerate() {
            trace!("    local {}: {:?}", i, local.value);
        }
    }

    // Extra output to help debug specific issues.
    match e.kind() {
        UndefinedBehavior(UndefinedBehaviorInfo::InvalidUninitBytes(Some((alloc_id, access)))) => {
            eprintln!(
                "Uninitialized read occurred at offsets 0x{:x}..0x{:x} into this allocation:",
                access.uninit_offset.bytes(),
                access.uninit_offset.bytes() + access.uninit_size.bytes(),
            );
            eprintln!("{:?}", ecx.memory.dump_alloc(*alloc_id));
        }
        _ => {}
    }

    None
}

/// Report an error or note (depending on the `error` argument) with the given stacktrace.
/// Also emits a full stacktrace of the interpreter stack.
fn report_msg<'tcx>(
    tcx: TyCtxt<'tcx>,
    diag_level: DiagLevel,
    title: &str,
    span_msg: String,
    mut helps: Vec<(Option<SpanData>, String)>,
    stacktrace: &[FrameInfo<'tcx>],
) {
    let span = stacktrace.first().map_or(DUMMY_SP, |fi| fi.span);
    let mut err = match diag_level {
        DiagLevel::Error => tcx.sess.struct_span_err(span, title),
        DiagLevel::Warning => tcx.sess.struct_span_warn(span, title),
        DiagLevel::Note => tcx.sess.diagnostic().span_note_diag(span, title),
    };

    // Show main message.
    if span != DUMMY_SP {
        err.span_label(span, span_msg);
    } else {
        // Make sure we show the message even when it is a dummy span.
        err.note(&span_msg);
        err.note("(no span available)");
    }
    // Show help messages.
    if !helps.is_empty() {
        // Add visual separator before backtrace.
        helps.last_mut().unwrap().1.push_str("\n");
        for (span_data, help) in helps {
            if let Some(span_data) = span_data {
                err.span_help(span_data.span(), &help);
            } else {
                err.help(&help);
            }
        }
    }
    // Add backtrace
    for (idx, frame_info) in stacktrace.iter().enumerate() {
        let is_local = frame_info.instance.def_id().is_local();
        // No span for non-local frames and the first frame (which is the error site).
        if is_local && idx > 0 {
            err.span_note(frame_info.span, &frame_info.to_string());
        } else {
            err.note(&frame_info.to_string());
        }
    }

    err.emit();
}

thread_local! {
    static DIAGNOSTICS: RefCell<Vec<NonHaltingDiagnostic>> = RefCell::new(Vec::new());
}

/// Schedule a diagnostic for emitting. This function works even if you have no `InterpCx` available.
/// The diagnostic will be emitted after the current interpreter step is finished.
pub fn register_diagnostic(e: NonHaltingDiagnostic) {
    DIAGNOSTICS.with(|diagnostics| diagnostics.borrow_mut().push(e));
}

/// Remember enough about the topmost frame so that we can restore the stack
/// after a step was taken.
pub struct TopFrameInfo<'tcx> {
    stack_size: usize,
    instance: Option<ty::Instance<'tcx>>,
    span: Span,
}

impl<'mir, 'tcx: 'mir> EvalContextExt<'mir, 'tcx> for crate::MiriEvalContext<'mir, 'tcx> {}
pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriEvalContextExt<'mir, 'tcx> {
    fn preprocess_diagnostics(&self) -> TopFrameInfo<'tcx> {
        // Ensure we have no lingering diagnostics.
        DIAGNOSTICS.with(|diagnostics| assert!(diagnostics.borrow().is_empty()));

        let this = self.eval_context_ref();
        if this.active_thread_stack().is_empty() {
            // Diagnostics can happen even with the empty stack (e.g. deallocation of thread-local statics).
            return TopFrameInfo { stack_size: 0, instance: None, span: DUMMY_SP };
        }
        let frame = this.frame();

        TopFrameInfo {
            stack_size: this.active_thread_stack().len(),
            instance: Some(frame.instance),
            span: frame.current_span(),
        }
    }

    /// Emit all diagnostics that were registed with `register_diagnostics`
    fn process_diagnostics(&self, info: TopFrameInfo<'tcx>) {
        let this = self.eval_context_ref();
        DIAGNOSTICS.with(|diagnostics| {
            let mut diagnostics = diagnostics.borrow_mut();
            if diagnostics.is_empty() {
                return;
            }
            // We need to fix up the stack trace, because the machine has already
            // stepped to the next statement.
            let mut stacktrace = this.generate_stacktrace();
            // Remove newly pushed frames.
            while stacktrace.len() > info.stack_size {
                stacktrace.remove(0);
            }
            // Add popped frame back.
            if stacktrace.len() < info.stack_size {
                assert!(
                    stacktrace.len() == info.stack_size - 1,
                    "we should never pop more than one frame at once"
                );
                let frame_info = FrameInfo {
                    instance: info.instance.unwrap(),
                    span: info.span,
                    lint_root: None,
                };
                stacktrace.insert(0, frame_info);
            } else if let Some(instance) = info.instance {
                // Adjust topmost frame.
                stacktrace[0].span = info.span;
                assert_eq!(
                    stacktrace[0].instance, instance,
                    "we should not pop and push a frame in one step"
                );
            }

            // Show diagnostics.
            for e in diagnostics.drain(..) {
                use NonHaltingDiagnostic::*;
                let msg = match e {
                    CreatedPointerTag(tag) => format!("created tag {:?}", tag),
                    PoppedPointerTag(item) => format!("popped tracked tag for item {:?}", item),
                    CreatedCallId(id) => format!("function call with id {}", id),
                    CreatedAlloc(AllocId(id)) => format!("created allocation with id {}", id),
                    FreedAlloc(AllocId(id)) => format!("freed allocation with id {}", id),
                    RejectedIsolatedOp(ref op) =>
                        format!("`{}` was made to return an error due to isolation", op),
                };

                let (title, diag_level) = match e {
                    RejectedIsolatedOp(_) =>
                        ("operation rejected by isolation", DiagLevel::Warning),
                    _ => ("tracking was triggered", DiagLevel::Note),
                };

                report_msg(*this.tcx, diag_level, title, msg, vec![], &stacktrace);
            }
        });
    }
}
