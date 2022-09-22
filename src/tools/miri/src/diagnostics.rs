use std::fmt;
use std::num::NonZeroU64;

use log::trace;

use rustc_span::{source_map::DUMMY_SP, SpanData, Symbol};
use rustc_target::abi::{Align, Size};

use crate::stacked_borrows::{diagnostics::TagHistory, AccessKind};
use crate::*;

/// Details of premature program termination.
pub enum TerminationInfo {
    Exit(i64),
    Abort(String),
    UnsupportedInIsolation(String),
    StackedBorrowsUb {
        msg: String,
        help: Option<String>,
        history: Option<TagHistory>,
    },
    Int2PtrWithStrictProvenance,
    Deadlock,
    MultipleSymbolDefinitions {
        link_name: Symbol,
        first: SpanData,
        first_crate: Symbol,
        second: SpanData,
        second_crate: Symbol,
    },
    SymbolShimClashing {
        link_name: Symbol,
        span: SpanData,
    },
}

impl fmt::Display for TerminationInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use TerminationInfo::*;
        match self {
            Exit(code) => write!(f, "the evaluated program completed with exit code {code}"),
            Abort(msg) => write!(f, "{msg}"),
            UnsupportedInIsolation(msg) => write!(f, "{msg}"),
            Int2PtrWithStrictProvenance =>
                write!(
                    f,
                    "integer-to-pointer casts and `ptr::from_exposed_addr` are not supported with `-Zmiri-strict-provenance`"
                ),
            StackedBorrowsUb { msg, .. } => write!(f, "{msg}"),
            Deadlock => write!(f, "the evaluated program deadlocked"),
            MultipleSymbolDefinitions { link_name, .. } =>
                write!(f, "multiple definitions of symbol `{link_name}`"),
            SymbolShimClashing { link_name, .. } =>
                write!(f, "found `{link_name}` symbol definition that clashes with a built-in shim",),
        }
    }
}

impl MachineStopType for TerminationInfo {}

/// Miri specific diagnostics
pub enum NonHaltingDiagnostic {
    CreatedPointerTag(NonZeroU64, Option<(AllocId, AllocRange)>),
    /// This `Item` was popped from the borrow stack, either due to an access with the given tag or
    /// a deallocation when the second argument is `None`.
    PoppedPointerTag(Item, Option<(ProvenanceExtra, AccessKind)>),
    CreatedCallId(CallId),
    CreatedAlloc(AllocId, Size, Align, MemoryKind<MiriMemoryKind>),
    FreedAlloc(AllocId),
    RejectedIsolatedOp(String),
    ProgressReport {
        block_count: u64, // how many basic blocks have been run so far
    },
    Int2Ptr {
        details: bool,
    },
    WeakMemoryOutdatedLoad,
}

/// Level of Miri specific diagnostics
enum DiagLevel {
    Error,
    Warning,
    Note,
}

/// Attempts to prune a stacktrace to omit the Rust runtime, and returns a bool indicating if any
/// frames were pruned. If the stacktrace does not have any local frames, we conclude that it must
/// be pointing to a problem in the Rust runtime itself, and do not prune it at all.
fn prune_stacktrace<'tcx>(
    mut stacktrace: Vec<FrameInfo<'tcx>>,
    machine: &MiriMachine<'_, 'tcx>,
) -> (Vec<FrameInfo<'tcx>>, bool) {
    match machine.backtrace_style {
        BacktraceStyle::Off => {
            // Remove all frames marked with `caller_location` -- that attribute indicates we
            // usually want to point at the caller, not them.
            stacktrace.retain(|frame| !frame.instance.def.requires_caller_location(machine.tcx));
            // Retain one frame so that we can print a span for the error itself
            stacktrace.truncate(1);
            (stacktrace, false)
        }
        BacktraceStyle::Short => {
            let original_len = stacktrace.len();
            // Only prune frames if there is at least one local frame. This check ensures that if
            // we get a backtrace that never makes it to the user code because it has detected a
            // bug in the Rust runtime, we don't prune away every frame.
            let has_local_frame = stacktrace.iter().any(|frame| machine.is_local(frame));
            if has_local_frame {
                // Remove all frames marked with `caller_location` -- that attribute indicates we
                // usually want to point at the caller, not them.
                stacktrace
                    .retain(|frame| !frame.instance.def.requires_caller_location(machine.tcx));

                // This is part of the logic that `std` uses to select the relevant part of a
                // backtrace. But here, we only look for __rust_begin_short_backtrace, not
                // __rust_end_short_backtrace because the end symbol comes from a call to the default
                // panic handler.
                stacktrace = stacktrace
                    .into_iter()
                    .take_while(|frame| {
                        let def_id = frame.instance.def_id();
                        let path = machine.tcx.def_path_str(def_id);
                        !path.contains("__rust_begin_short_backtrace")
                    })
                    .collect::<Vec<_>>();

                // After we prune frames from the bottom, there are a few left that are part of the
                // Rust runtime. So we remove frames until we get to a local symbol, which should be
                // main or a test.
                // This len check ensures that we don't somehow remove every frame, as doing so breaks
                // the primary error message.
                while stacktrace.len() > 1
                    && stacktrace.last().map_or(false, |frame| !machine.is_local(frame))
                {
                    stacktrace.pop();
                }
            }
            let was_pruned = stacktrace.len() != original_len;
            (stacktrace, was_pruned)
        }
        BacktraceStyle::Full => (stacktrace, false),
    }
}

/// Emit a custom diagnostic without going through the miri-engine machinery
pub fn report_error<'tcx, 'mir>(
    ecx: &InterpCx<'mir, 'tcx, MiriMachine<'mir, 'tcx>>,
    e: InterpErrorInfo<'tcx>,
) -> Option<i64> {
    use InterpError::*;

    let mut msg = vec![];

    let (title, helps) = match &e.kind() {
        MachineStop(info) => {
            let info = info.downcast_ref::<TerminationInfo>().expect("invalid MachineStop payload");
            use TerminationInfo::*;
            let title = match info {
                Exit(code) => return Some(*code),
                Abort(_) => Some("abnormal termination"),
                UnsupportedInIsolation(_) | Int2PtrWithStrictProvenance =>
                    Some("unsupported operation"),
                StackedBorrowsUb { .. } => Some("Undefined Behavior"),
                Deadlock => Some("deadlock"),
                MultipleSymbolDefinitions { .. } | SymbolShimClashing { .. } => None,
            };
            #[rustfmt::skip]
            let helps = match info {
                UnsupportedInIsolation(_) =>
                    vec![
                        (None, format!("pass the flag `-Zmiri-disable-isolation` to disable isolation;")),
                        (None, format!("or pass `-Zmiri-isolation-error=warn` to configure Miri to return an error code from isolated operations (if supported for that operation) and continue with a warning")),
                    ],
                StackedBorrowsUb { help, history, .. } => {
                    let url = "https://github.com/rust-lang/unsafe-code-guidelines/blob/master/wip/stacked-borrows.md";
                    msg.extend(help.clone());
                    let mut helps = vec![
                        (None, format!("this indicates a potential bug in the program: it performed an invalid operation, but the Stacked Borrows rules it violated are still experimental")),
                        (None, format!("see {url} for further information")),
                    ];
                    if let Some(TagHistory {created, invalidated, protected}) = history.clone() {
                        helps.push((Some(created.1), created.0));
                        if let Some((msg, span)) = invalidated {
                            helps.push((Some(span), msg));
                        }
                        if let Some((protector_msg, protector_span)) = protected {
                            helps.push((Some(protector_span), protector_msg));
                        }
                    }
                    helps
                }
                MultipleSymbolDefinitions { first, first_crate, second, second_crate, .. } =>
                    vec![
                        (Some(*first), format!("it's first defined here, in crate `{first_crate}`")),
                        (Some(*second), format!("then it's defined here again, in crate `{second_crate}`")),
                    ],
                SymbolShimClashing { link_name, span } =>
                    vec![(Some(*span), format!("the `{link_name}` symbol is defined here"))],
                Int2PtrWithStrictProvenance =>
                    vec![(None, format!("use Strict Provenance APIs (https://doc.rust-lang.org/nightly/std/ptr/index.html#strict-provenance, https://crates.io/crates/sptr) instead"))],
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
                InvalidProgram(
                    InvalidProgramInfo::AlreadyReported(_) |
                    InvalidProgramInfo::Layout(..) |
                    InvalidProgramInfo::ReferencedConstant
                ) =>
                    "post-monomorphization error",
                kind =>
                    bug!("This error should be impossible in Miri: {kind:?}"),
            };
            #[rustfmt::skip]
            let helps = match e.kind() {
                Unsupported(
                    UnsupportedOpInfo::ThreadLocalStatic(_) |
                    UnsupportedOpInfo::ReadExternStatic(_) |
                    UnsupportedOpInfo::PartialPointerOverwrite(_) | // we make memory uninit instead
                    UnsupportedOpInfo::ReadPointerAsBytes
                ) =>
                    panic!("Error should never be raised by Miri: {kind:?}", kind = e.kind()),
                Unsupported(
                    UnsupportedOpInfo::Unsupported(_) |
                    UnsupportedOpInfo::PartialPointerCopy(_)
                ) =>
                    vec![(None, format!("this is likely not a bug in the program; it indicates that the program performed an operation that the interpreter does not support"))],
                UndefinedBehavior(UndefinedBehaviorInfo::AlignmentCheckFailed { .. })
                    if ecx.machine.check_alignment == AlignmentCheck::Symbolic
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
                InvalidProgram(_) | ResourceExhaustion(_) | MachineStop(_) =>
                    vec![],
            };
            (Some(title), helps)
        }
    };

    let stacktrace = ecx.generate_stacktrace();
    let (stacktrace, was_pruned) = prune_stacktrace(stacktrace, &ecx.machine);
    e.print_backtrace();
    msg.insert(0, e.to_string());
    report_msg(
        DiagLevel::Error,
        &if let Some(title) = title { format!("{}: {}", title, msg[0]) } else { msg[0].clone() },
        msg,
        vec![],
        helps,
        &stacktrace,
        &ecx.machine,
    );

    // Include a note like `std` does when we omit frames from a backtrace
    if was_pruned {
        ecx.tcx.sess.diagnostic().note_without_error(
            "some details are omitted, run with `MIRIFLAGS=-Zmiri-backtrace=full` for a verbose backtrace",
        );
    }

    // Debug-dump all locals.
    for (i, frame) in ecx.active_thread_stack().iter().enumerate() {
        trace!("-------------------");
        trace!("Frame {}", i);
        trace!("    return: {:?}", *frame.return_place);
        for (i, local) in frame.locals.iter().enumerate() {
            trace!("    local {}: {:?}", i, local.value);
        }
    }

    // Extra output to help debug specific issues.
    match e.kind() {
        UndefinedBehavior(UndefinedBehaviorInfo::InvalidUninitBytes(Some((alloc_id, access)))) => {
            eprintln!(
                "Uninitialized memory occurred at {alloc_id:?}{range:?}, in this allocation:",
                range = access.uninit,
            );
            eprintln!("{:?}", ecx.dump_alloc(*alloc_id));
        }
        _ => {}
    }

    None
}

/// Report an error or note (depending on the `error` argument) with the given stacktrace.
/// Also emits a full stacktrace of the interpreter stack.
/// We want to present a multi-line span message for some errors. Diagnostics do not support this
/// directly, so we pass the lines as a `Vec<String>` and display each line after the first with an
/// additional `span_label` or `note` call.
fn report_msg<'tcx>(
    diag_level: DiagLevel,
    title: &str,
    span_msg: Vec<String>,
    notes: Vec<(Option<SpanData>, String)>,
    helps: Vec<(Option<SpanData>, String)>,
    stacktrace: &[FrameInfo<'tcx>],
    machine: &MiriMachine<'_, 'tcx>,
) {
    let span = stacktrace.first().map_or(DUMMY_SP, |fi| fi.span);
    let sess = machine.tcx.sess;
    let mut err = match diag_level {
        DiagLevel::Error => sess.struct_span_err(span, title).forget_guarantee(),
        DiagLevel::Warning => sess.struct_span_warn(span, title),
        DiagLevel::Note => sess.diagnostic().span_note_diag(span, title),
    };

    // Show main message.
    if span != DUMMY_SP {
        for line in span_msg {
            err.span_label(span, line);
        }
    } else {
        // Make sure we show the message even when it is a dummy span.
        for line in span_msg {
            err.note(&line);
        }
        err.note("(no span available)");
    }

    // Show note and help messages.
    for (span_data, note) in &notes {
        if let Some(span_data) = span_data {
            err.span_note(span_data.span(), note);
        } else {
            err.note(note);
        }
    }
    for (span_data, help) in &helps {
        if let Some(span_data) = span_data {
            err.span_help(span_data.span(), help);
        } else {
            err.help(help);
        }
    }
    if notes.len() + helps.len() > 0 {
        // Add visual separator before backtrace.
        err.note("BACKTRACE:");
    }
    // Add backtrace
    for (idx, frame_info) in stacktrace.iter().enumerate() {
        let is_local = machine.is_local(frame_info);
        // No span for non-local frames and the first frame (which is the error site).
        if is_local && idx > 0 {
            err.span_note(frame_info.span, &frame_info.to_string());
        } else {
            err.note(&frame_info.to_string());
        }
    }

    err.emit();
}

impl<'mir, 'tcx> MiriMachine<'mir, 'tcx> {
    pub fn emit_diagnostic(&self, e: NonHaltingDiagnostic) {
        use NonHaltingDiagnostic::*;

        let stacktrace =
            MiriInterpCx::generate_stacktrace_from_stack(self.threads.active_thread_stack());
        let (stacktrace, _was_pruned) = prune_stacktrace(stacktrace, self);

        let (title, diag_level) = match e {
            RejectedIsolatedOp(_) => ("operation rejected by isolation", DiagLevel::Warning),
            Int2Ptr { .. } => ("integer-to-pointer cast", DiagLevel::Warning),
            CreatedPointerTag(..)
            | PoppedPointerTag(..)
            | CreatedCallId(..)
            | CreatedAlloc(..)
            | FreedAlloc(..)
            | ProgressReport { .. }
            | WeakMemoryOutdatedLoad => ("tracking was triggered", DiagLevel::Note),
        };

        let msg = match e {
            CreatedPointerTag(tag, None) => format!("created tag {tag:?}"),
            CreatedPointerTag(tag, Some((alloc_id, range))) =>
                format!("created tag {tag:?} at {alloc_id:?}{range:?}"),
            PoppedPointerTag(item, tag) =>
                match tag {
                    None => format!("popped tracked tag for item {item:?} due to deallocation",),
                    Some((tag, access)) => {
                        format!(
                            "popped tracked tag for item {item:?} due to {access:?} access for {tag:?}",
                        )
                    }
                },
            CreatedCallId(id) => format!("function call with id {id}"),
            CreatedAlloc(AllocId(id), size, align, kind) =>
                format!(
                    "created {kind} allocation of {size} bytes (alignment {align} bytes) with id {id}",
                    size = size.bytes(),
                    align = align.bytes(),
                ),
            FreedAlloc(AllocId(id)) => format!("freed allocation with id {id}"),
            RejectedIsolatedOp(ref op) =>
                format!("{op} was made to return an error due to isolation"),
            ProgressReport { .. } =>
                format!("progress report: current operation being executed is here"),
            Int2Ptr { .. } => format!("integer-to-pointer cast"),
            WeakMemoryOutdatedLoad =>
                format!("weak memory emulation: outdated value returned from load"),
        };

        let notes = match e {
            ProgressReport { block_count } => {
                // It is important that each progress report is slightly different, since
                // identical diagnostics are being deduplicated.
                vec![(None, format!("so far, {block_count} basic blocks have been executed"))]
            }
            _ => vec![],
        };

        let helps = match e {
            Int2Ptr { details: true } =>
                vec![
                    (
                        None,
                        format!(
                            "This program is using integer-to-pointer casts or (equivalently) `ptr::from_exposed_addr`,"
                        ),
                    ),
                    (
                        None,
                        format!("which means that Miri might miss pointer bugs in this program."),
                    ),
                    (
                        None,
                        format!(
                            "See https://doc.rust-lang.org/nightly/std/ptr/fn.from_exposed_addr.html for more details on that operation."
                        ),
                    ),
                    (
                        None,
                        format!(
                            "To ensure that Miri does not miss bugs in your program, use Strict Provenance APIs (https://doc.rust-lang.org/nightly/std/ptr/index.html#strict-provenance, https://crates.io/crates/sptr) instead."
                        ),
                    ),
                    (
                        None,
                        format!(
                            "You can then pass the `-Zmiri-strict-provenance` flag to Miri, to ensure you are not relying on `from_exposed_addr` semantics."
                        ),
                    ),
                    (
                        None,
                        format!(
                            "Alternatively, the `-Zmiri-permissive-provenance` flag disables this warning."
                        ),
                    ),
                ],
            _ => vec![],
        };

        report_msg(diag_level, title, vec![msg], notes, helps, &stacktrace, self);
    }
}

impl<'mir, 'tcx: 'mir> EvalContextExt<'mir, 'tcx> for crate::MiriInterpCx<'mir, 'tcx> {}
pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriInterpCxExt<'mir, 'tcx> {
    fn emit_diagnostic(&self, e: NonHaltingDiagnostic) {
        let this = self.eval_context_ref();
        this.machine.emit_diagnostic(e);
    }

    /// We had a panic in Miri itself, try to print something useful.
    fn handle_ice(&self) {
        eprintln!();
        eprintln!(
            "Miri caused an ICE during evaluation. Here's the interpreter backtrace at the time of the panic:"
        );
        let this = self.eval_context_ref();
        let stacktrace = this.generate_stacktrace();
        report_msg(
            DiagLevel::Note,
            "the place in the program where the ICE was triggered",
            vec![],
            vec![],
            vec![],
            &stacktrace,
            &this.machine,
        );
    }
}
