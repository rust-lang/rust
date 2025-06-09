use std::fmt::{self, Write};
use std::num::NonZero;

use rustc_abi::{Align, Size};
use rustc_errors::{Diag, DiagMessage, Level};
use rustc_span::{DUMMY_SP, SpanData, Symbol};

use crate::borrow_tracker::stacked_borrows::diagnostics::TagHistory;
use crate::borrow_tracker::tree_borrows::diagnostics as tree_diagnostics;
use crate::*;

/// Details of premature program termination.
pub enum TerminationInfo {
    Exit {
        code: i32,
        leak_check: bool,
    },
    Abort(String),
    /// Miri was interrupted by a Ctrl+C from the user
    Interrupted,
    UnsupportedInIsolation(String),
    StackedBorrowsUb {
        msg: String,
        help: Vec<String>,
        history: Option<TagHistory>,
    },
    TreeBorrowsUb {
        title: String,
        details: Vec<String>,
        history: tree_diagnostics::HistoryData,
    },
    Int2PtrWithStrictProvenance,
    Deadlock,
    /// In GenMC mode, an execution can get stuck in certain cases. This is not an error.
    GenmcStuckExecution,
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
    DataRace {
        involves_non_atomic: bool,
        ptr: interpret::Pointer<AllocId>,
        op1: RacingOp,
        op2: RacingOp,
        extra: Option<&'static str>,
        retag_explain: bool,
    },
    UnsupportedForeignItem(String),
}

pub struct RacingOp {
    pub action: String,
    pub thread_info: String,
    pub span: SpanData,
}

impl fmt::Display for TerminationInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use TerminationInfo::*;
        match self {
            Exit { code, .. } => write!(f, "the evaluated program completed with exit code {code}"),
            Abort(msg) => write!(f, "{msg}"),
            Interrupted => write!(f, "interpretation was interrupted"),
            UnsupportedInIsolation(msg) => write!(f, "{msg}"),
            Int2PtrWithStrictProvenance =>
                write!(
                    f,
                    "integer-to-pointer casts and `ptr::with_exposed_provenance` are not supported with `-Zmiri-strict-provenance`"
                ),
            StackedBorrowsUb { msg, .. } => write!(f, "{msg}"),
            TreeBorrowsUb { title, .. } => write!(f, "{title}"),
            Deadlock => write!(f, "the evaluated program deadlocked"),
            GenmcStuckExecution => write!(f, "GenMC determined that the execution got stuck"),
            MultipleSymbolDefinitions { link_name, .. } =>
                write!(f, "multiple definitions of symbol `{link_name}`"),
            SymbolShimClashing { link_name, .. } =>
                write!(f, "found `{link_name}` symbol definition that clashes with a built-in shim",),
            DataRace { involves_non_atomic, ptr, op1, op2, .. } =>
                write!(
                    f,
                    "{} detected between (1) {} on {} and (2) {} on {} at {ptr:?}. (2) just happened here",
                    if *involves_non_atomic { "Data race" } else { "Race condition" },
                    op1.action,
                    op1.thread_info,
                    op2.action,
                    op2.thread_info
                ),
            UnsupportedForeignItem(msg) => write!(f, "{msg}"),
        }
    }
}

impl fmt::Debug for TerminationInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl MachineStopType for TerminationInfo {
    fn diagnostic_message(&self) -> DiagMessage {
        self.to_string().into()
    }
    fn add_args(
        self: Box<Self>,
        _: &mut dyn FnMut(std::borrow::Cow<'static, str>, rustc_errors::DiagArgValue),
    ) {
    }
}

/// Miri specific diagnostics
pub enum NonHaltingDiagnostic {
    /// (new_tag, new_perm, (alloc_id, base_offset, orig_tag))
    ///
    /// new_perm is `None` for base tags.
    CreatedPointerTag(NonZero<u64>, Option<String>, Option<(AllocId, AllocRange, ProvenanceExtra)>),
    /// This `Item` was popped from the borrow stack. The string explains the reason.
    PoppedPointerTag(Item, String),
    CreatedAlloc(AllocId, Size, Align, MemoryKind),
    FreedAlloc(AllocId),
    AccessedAlloc(AllocId, AccessKind),
    RejectedIsolatedOp(String),
    ProgressReport {
        block_count: u64, // how many basic blocks have been run so far
    },
    Int2Ptr {
        details: bool,
    },
    NativeCallSharedMem,
    WeakMemoryOutdatedLoad {
        ptr: Pointer,
    },
    ExternTypeReborrow,
}

/// Level of Miri specific diagnostics
pub enum DiagLevel {
    Error,
    Warning,
    Note,
}

/// Generate a note/help text without a span.
macro_rules! note {
    ($($tt:tt)*) => { (None, format!($($tt)*)) };
}
/// Generate a note/help text with a span.
macro_rules! note_span {
    ($span:expr, $($tt:tt)*) => { (Some($span), format!($($tt)*)) };
}

/// Attempts to prune a stacktrace to omit the Rust runtime, and returns a bool indicating if any
/// frames were pruned. If the stacktrace does not have any local frames, we conclude that it must
/// be pointing to a problem in the Rust runtime itself, and do not prune it at all.
pub fn prune_stacktrace<'tcx>(
    mut stacktrace: Vec<FrameInfo<'tcx>>,
    machine: &MiriMachine<'tcx>,
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
                    && stacktrace.last().is_some_and(|frame| !machine.is_local(frame))
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

/// Emit a custom diagnostic without going through the miri-engine machinery.
///
/// Returns `Some` if this was regular program termination with a given exit code and a `bool` indicating whether a leak check should happen; `None` otherwise.
pub fn report_error<'tcx>(
    ecx: &InterpCx<'tcx, MiriMachine<'tcx>>,
    e: InterpErrorInfo<'tcx>,
) -> Option<(i32, bool)> {
    use InterpErrorKind::*;
    use UndefinedBehaviorInfo::*;

    let mut msg = vec![];

    let (title, helps) = if let MachineStop(info) = e.kind() {
        let info = info.downcast_ref::<TerminationInfo>().expect("invalid MachineStop payload");
        use TerminationInfo::*;
        let title = match info {
            &Exit { code, leak_check } => return Some((code, leak_check)),
            Abort(_) => Some("abnormal termination"),
            Interrupted => None,
            UnsupportedInIsolation(_) | Int2PtrWithStrictProvenance | UnsupportedForeignItem(_) =>
                Some("unsupported operation"),
            StackedBorrowsUb { .. } | TreeBorrowsUb { .. } | DataRace { .. } =>
                Some("Undefined Behavior"),
            Deadlock => Some("deadlock"),
            GenmcStuckExecution => {
                // This case should only happen in GenMC mode. We treat it like a normal program exit.
                assert!(ecx.machine.data_race.as_genmc_ref().is_some());
                tracing::info!("GenMC: found stuck execution");
                return Some((0, true));
            }
            MultipleSymbolDefinitions { .. } | SymbolShimClashing { .. } => None,
        };
        #[rustfmt::skip]
        let helps = match info {
            UnsupportedInIsolation(_) =>
                vec![
                    note!("set `MIRIFLAGS=-Zmiri-disable-isolation` to disable isolation;"),
                    note!("or set `MIRIFLAGS=-Zmiri-isolation-error=warn` to make Miri return an error code from isolated operations (if supported for that operation) and continue with a warning"),
                ],
            UnsupportedForeignItem(_) => {
                vec![
                    note!("this means the program tried to do something Miri does not support; it does not indicate a bug in the program"),
                ]
            }
            StackedBorrowsUb { help, history, .. } => {
                msg.extend(help.clone());
                let mut helps = vec![
                    note!("this indicates a potential bug in the program: it performed an invalid operation, but the Stacked Borrows rules it violated are still experimental"),
                    note!("see https://github.com/rust-lang/unsafe-code-guidelines/blob/master/wip/stacked-borrows.md for further information"),
                ];
                if let Some(TagHistory {created, invalidated, protected}) = history.clone() {
                    helps.push((Some(created.1), created.0));
                    if let Some((msg, span)) = invalidated {
                        helps.push(note_span!(span, "{msg}"));
                    }
                    if let Some((protector_msg, protector_span)) = protected {
                        helps.push(note_span!(protector_span, "{protector_msg}"));
                    }
                }
                helps
            },
            TreeBorrowsUb { title: _, details, history } => {
                let mut helps = vec![
                    note!("this indicates a potential bug in the program: it performed an invalid operation, but the Tree Borrows rules it violated are still experimental")
                ];
                for m in details {
                    helps.push(note!("{m}"));
                }
                for event in history.events.clone() {
                    helps.push(event);
                }
                helps
            }
            MultipleSymbolDefinitions { first, first_crate, second, second_crate, .. } =>
                vec![
                    note_span!(*first, "it's first defined here, in crate `{first_crate}`"),
                    note_span!(*second, "then it's defined here again, in crate `{second_crate}`"),
                ],
            SymbolShimClashing { link_name, span } =>
                vec![note_span!(*span, "the `{link_name}` symbol is defined here")],
            Int2PtrWithStrictProvenance =>
                vec![note!("use Strict Provenance APIs (https://doc.rust-lang.org/nightly/std/ptr/index.html#strict-provenance, https://crates.io/crates/sptr) instead")],
            DataRace { op1, extra, retag_explain, .. } => {
                let mut helps = vec![note_span!(op1.span, "and (1) occurred earlier here")];
                if let Some(extra) = extra {
                    helps.push(note!("{extra}"));
                    helps.push(note!("see https://doc.rust-lang.org/nightly/std/sync/atomic/index.html#memory-model-for-atomic-accesses for more information about the Rust memory model"));
                }
                if *retag_explain {
                    helps.push(note!("retags occur on all (re)borrows and as well as when references are copied or moved"));
                    helps.push(note!("retags permit optimizations that insert speculative reads or writes"));
                    helps.push(note!("therefore from the perspective of data races, a retag has the same implications as a read or write"));
                }
                helps.push(note!("this indicates a bug in the program: it performed an invalid operation, and caused Undefined Behavior"));
                helps.push(note!("see https://doc.rust-lang.org/nightly/reference/behavior-considered-undefined.html for further information"));
                helps
            }
                ,
            _ => vec![],
        };
        (title, helps)
    } else {
        let title = match e.kind() {
            UndefinedBehavior(ValidationError(validation_err))
                if matches!(
                    validation_err.kind,
                    ValidationErrorKind::PointerAsInt { .. } | ValidationErrorKind::PartialPointer
                ) =>
            {
                ecx.handle_ice(); // print interpreter backtrace (this is outside the eval `catch_unwind`)
                bug!(
                    "This validation error should be impossible in Miri: {}",
                    format_interp_error(ecx.tcx.dcx(), e)
                );
            }
            UndefinedBehavior(_) => "Undefined Behavior",
            ResourceExhaustion(_) => "resource exhaustion",
            Unsupported(
                // We list only the ones that can actually happen.
                UnsupportedOpInfo::Unsupported(_)
                | UnsupportedOpInfo::UnsizedLocal
                | UnsupportedOpInfo::ExternTypeField,
            ) => "unsupported operation",
            InvalidProgram(
                // We list only the ones that can actually happen.
                InvalidProgramInfo::AlreadyReported(_) | InvalidProgramInfo::Layout(..),
            ) => "post-monomorphization error",
            _ => {
                ecx.handle_ice(); // print interpreter backtrace (this is outside the eval `catch_unwind`)
                bug!(
                    "This error should be impossible in Miri: {}",
                    format_interp_error(ecx.tcx.dcx(), e)
                );
            }
        };
        #[rustfmt::skip]
        let helps = match e.kind() {
            Unsupported(_) =>
                vec![
                    note!("this is likely not a bug in the program; it indicates that the program performed an operation that Miri does not support"),
                ],
            UndefinedBehavior(AlignmentCheckFailed { .. })
                if ecx.machine.check_alignment == AlignmentCheck::Symbolic
            =>
                vec![
                    note!("this usually indicates that your program performed an invalid operation and caused Undefined Behavior"),
                    note!("but due to `-Zmiri-symbolic-alignment-check`, alignment errors can also be false positives"),
                ],
            UndefinedBehavior(info) => {
                let mut helps = vec![
                    note!("this indicates a bug in the program: it performed an invalid operation, and caused Undefined Behavior"),
                    note!("see https://doc.rust-lang.org/nightly/reference/behavior-considered-undefined.html for further information"),
                ];
                match info {
                    PointerUseAfterFree(alloc_id, _) | PointerOutOfBounds { alloc_id, .. } => {
                        if let Some(span) = ecx.machine.allocated_span(*alloc_id) {
                            helps.push(note_span!(span, "{:?} was allocated here:", alloc_id));
                        }
                        if let Some(span) = ecx.machine.deallocated_span(*alloc_id) {
                            helps.push(note_span!(span, "{:?} was deallocated here:", alloc_id));
                        }
                    }
                    AbiMismatchArgument { .. } | AbiMismatchReturn { .. } => {
                        helps.push(note!("this means these two types are not *guaranteed* to be ABI-compatible across all targets"));
                        helps.push(note!("if you think this code should be accepted anyway, please report an issue with Miri"));
                    }
                    _ => {},
                }
                helps
            }
            InvalidProgram(
                InvalidProgramInfo::AlreadyReported(_)
            ) => {
                // This got already reported. No point in reporting it again.
                return None;
            }
            _ =>
                vec![],
        };
        (Some(title), helps)
    };

    let stacktrace = ecx.generate_stacktrace();
    let (stacktrace, mut any_pruned) = prune_stacktrace(stacktrace, &ecx.machine);

    let mut show_all_threads = false;

    // We want to dump the allocation if this is `InvalidUninitBytes`.
    // Since `format_interp_error` consumes `e`, we compute the outut early.
    let mut extra = String::new();
    match e.kind() {
        UndefinedBehavior(InvalidUninitBytes(Some((alloc_id, access)))) => {
            writeln!(
                extra,
                "Uninitialized memory occurred at {alloc_id:?}{range:?}, in this allocation:",
                range = access.bad,
            )
            .unwrap();
            writeln!(extra, "{:?}", ecx.dump_alloc(*alloc_id)).unwrap();
        }
        MachineStop(info) => {
            let info = info.downcast_ref::<TerminationInfo>().expect("invalid MachineStop payload");
            match info {
                TerminationInfo::Deadlock => {
                    show_all_threads = true;
                }
                _ => {}
            }
        }
        _ => {}
    }

    msg.insert(0, format_interp_error(ecx.tcx.dcx(), e));

    report_msg(
        DiagLevel::Error,
        if let Some(title) = title { format!("{title}: {}", msg[0]) } else { msg[0].clone() },
        msg,
        vec![],
        helps,
        &stacktrace,
        Some(ecx.active_thread()),
        &ecx.machine,
    );

    eprint!("{extra}"); // newlines are already in the string

    if show_all_threads {
        for (thread, stack) in ecx.machine.threads.all_stacks() {
            if thread != ecx.active_thread() {
                let stacktrace = Frame::generate_stacktrace_from_stack(stack);
                let (stacktrace, was_pruned) = prune_stacktrace(stacktrace, &ecx.machine);
                any_pruned |= was_pruned;
                report_msg(
                    DiagLevel::Error,
                    format!("deadlock: the evaluated program deadlocked"),
                    vec![format!("the evaluated program deadlocked")],
                    vec![],
                    vec![],
                    &stacktrace,
                    Some(thread),
                    &ecx.machine,
                )
            }
        }
    }

    // Include a note like `std` does when we omit frames from a backtrace
    if any_pruned {
        ecx.tcx.dcx().note(
            "some details are omitted, run with `MIRIFLAGS=-Zmiri-backtrace=full` for a verbose backtrace",
        );
    }

    // Debug-dump all locals.
    for (i, frame) in ecx.active_thread_stack().iter().enumerate() {
        trace!("-------------------");
        trace!("Frame {}", i);
        trace!("    return: {:?}", frame.return_place);
        for (i, local) in frame.locals.iter().enumerate() {
            trace!("    local {}: {:?}", i, local);
        }
    }

    None
}

pub fn report_leaks<'tcx>(
    ecx: &InterpCx<'tcx, MiriMachine<'tcx>>,
    leaks: Vec<(AllocId, MemoryKind, Allocation<Provenance, AllocExtra<'tcx>, MiriAllocBytes>)>,
) {
    let mut any_pruned = false;
    for (id, kind, alloc) in leaks {
        let mut title = format!(
            "memory leaked: {id:?} ({}, size: {:?}, align: {:?})",
            kind,
            alloc.size().bytes(),
            alloc.align.bytes()
        );
        let Some(backtrace) = alloc.extra.backtrace else {
            ecx.tcx.dcx().err(title);
            continue;
        };
        title.push_str(", allocated here:");
        let (backtrace, pruned) = prune_stacktrace(backtrace, &ecx.machine);
        any_pruned |= pruned;
        report_msg(
            DiagLevel::Error,
            title,
            vec![],
            vec![],
            vec![],
            &backtrace,
            None, // we don't know the thread this is from
            &ecx.machine,
        );
    }
    if any_pruned {
        ecx.tcx.dcx().note(
            "some details are omitted, run with `MIRIFLAGS=-Zmiri-backtrace=full` for a verbose backtrace",
        );
    }
}

/// Report an error or note (depending on the `error` argument) with the given stacktrace.
/// Also emits a full stacktrace of the interpreter stack.
/// We want to present a multi-line span message for some errors. Diagnostics do not support this
/// directly, so we pass the lines as a `Vec<String>` and display each line after the first with an
/// additional `span_label` or `note` call.
pub fn report_msg<'tcx>(
    diag_level: DiagLevel,
    title: String,
    span_msg: Vec<String>,
    notes: Vec<(Option<SpanData>, String)>,
    helps: Vec<(Option<SpanData>, String)>,
    stacktrace: &[FrameInfo<'tcx>],
    thread: Option<ThreadId>,
    machine: &MiriMachine<'tcx>,
) {
    let span = stacktrace.first().map_or(DUMMY_SP, |fi| fi.span);
    let sess = machine.tcx.sess;
    let level = match diag_level {
        DiagLevel::Error => Level::Error,
        DiagLevel::Warning => Level::Warning,
        DiagLevel::Note => Level::Note,
    };
    let mut err = Diag::<()>::new(sess.dcx(), level, title);
    err.span(span);

    // Show main message.
    if span != DUMMY_SP {
        for line in span_msg {
            err.span_label(span, line);
        }
    } else {
        // Make sure we show the message even when it is a dummy span.
        for line in span_msg {
            err.note(line);
        }
        err.note("(no span available)");
    }

    // Show note and help messages.
    let mut extra_span = false;
    for (span_data, note) in notes {
        if let Some(span_data) = span_data {
            err.span_note(span_data.span(), note);
            extra_span = true;
        } else {
            err.note(note);
        }
    }
    for (span_data, help) in helps {
        if let Some(span_data) = span_data {
            err.span_help(span_data.span(), help);
            extra_span = true;
        } else {
            err.help(help);
        }
    }

    // Add backtrace
    let mut backtrace_title = String::from("BACKTRACE");
    if extra_span {
        write!(backtrace_title, " (of the first span)").unwrap();
    }
    if let Some(thread) = thread {
        let thread_name = machine.threads.get_thread_display_name(thread);
        if thread_name != "main" {
            // Only print thread name if it is not `main`.
            write!(backtrace_title, " on thread `{thread_name}`").unwrap();
        };
    }
    write!(backtrace_title, ":").unwrap();
    err.note(backtrace_title);
    for (idx, frame_info) in stacktrace.iter().enumerate() {
        let is_local = machine.is_local(frame_info);
        // No span for non-local frames and the first frame (which is the error site).
        if is_local && idx > 0 {
            err.subdiagnostic(frame_info.as_note(machine.tcx));
        } else {
            let sm = sess.source_map();
            let span = sm.span_to_embeddable_string(frame_info.span);
            err.note(format!("{frame_info} at {span}"));
        }
    }

    err.emit();
}

impl<'tcx> MiriMachine<'tcx> {
    pub fn emit_diagnostic(&self, e: NonHaltingDiagnostic) {
        use NonHaltingDiagnostic::*;

        let stacktrace = Frame::generate_stacktrace_from_stack(self.threads.active_thread_stack());
        let (stacktrace, _was_pruned) = prune_stacktrace(stacktrace, self);

        let (title, diag_level) = match &e {
            RejectedIsolatedOp(_) =>
                ("operation rejected by isolation".to_string(), DiagLevel::Warning),
            Int2Ptr { .. } => ("integer-to-pointer cast".to_string(), DiagLevel::Warning),
            NativeCallSharedMem =>
                ("sharing memory with a native function".to_string(), DiagLevel::Warning),
            ExternTypeReborrow =>
                ("reborrow of reference to `extern type`".to_string(), DiagLevel::Warning),
            CreatedPointerTag(..)
            | PoppedPointerTag(..)
            | CreatedAlloc(..)
            | AccessedAlloc(..)
            | FreedAlloc(..)
            | ProgressReport { .. }
            | WeakMemoryOutdatedLoad { .. } =>
                ("tracking was triggered".to_string(), DiagLevel::Note),
        };

        let msg = match &e {
            CreatedPointerTag(tag, None, _) => format!("created base tag {tag:?}"),
            CreatedPointerTag(tag, Some(perm), None) =>
                format!("created {tag:?} with {perm} derived from unknown tag"),
            CreatedPointerTag(tag, Some(perm), Some((alloc_id, range, orig_tag))) =>
                format!(
                    "created tag {tag:?} with {perm} at {alloc_id:?}{range:?} derived from {orig_tag:?}"
                ),
            PoppedPointerTag(item, cause) => format!("popped tracked tag for item {item:?}{cause}"),
            CreatedAlloc(AllocId(id), size, align, kind) =>
                format!(
                    "created {kind} allocation of {size} bytes (alignment {align} bytes) with id {id}",
                    size = size.bytes(),
                    align = align.bytes(),
                ),
            AccessedAlloc(AllocId(id), access_kind) =>
                format!("{access_kind} to allocation with id {id}"),
            FreedAlloc(AllocId(id)) => format!("freed allocation with id {id}"),
            RejectedIsolatedOp(op) => format!("{op} was made to return an error due to isolation"),
            ProgressReport { .. } =>
                format!("progress report: current operation being executed is here"),
            Int2Ptr { .. } => format!("integer-to-pointer cast"),
            NativeCallSharedMem => format!("sharing memory with a native function called via FFI"),
            WeakMemoryOutdatedLoad { ptr } =>
                format!("weak memory emulation: outdated value returned from load at {ptr}"),
            ExternTypeReborrow =>
                format!("reborrow of a reference to `extern type` is not properly supported"),
        };

        let notes = match &e {
            ProgressReport { block_count } => {
                vec![note!("so far, {block_count} basic blocks have been executed")]
            }
            _ => vec![],
        };

        let helps = match &e {
            Int2Ptr { details: true } => {
                let mut v = vec![
                    note!(
                        "this program is using integer-to-pointer casts or (equivalently) `ptr::with_exposed_provenance`, which means that Miri might miss pointer bugs in this program"
                    ),
                    note!(
                        "see https://doc.rust-lang.org/nightly/std/ptr/fn.with_exposed_provenance.html for more details on that operation"
                    ),
                    note!(
                        "to ensure that Miri does not miss bugs in your program, use Strict Provenance APIs (https://doc.rust-lang.org/nightly/std/ptr/index.html#strict-provenance, https://crates.io/crates/sptr) instead"
                    ),
                    note!(
                        "you can then set `MIRIFLAGS=-Zmiri-strict-provenance` to ensure you are not relying on `with_exposed_provenance` semantics"
                    ),
                ];
                if self.borrow_tracker.as_ref().is_some_and(|b| {
                    matches!(
                        b.borrow().borrow_tracker_method(),
                        BorrowTrackerMethod::TreeBorrows { .. }
                    )
                }) {
                    v.push(
                        note!("Tree Borrows does not support integer-to-pointer casts, so the program is likely to go wrong when this pointer gets used")
                    );
                } else {
                    v.push(
                        note!("alternatively, `MIRIFLAGS=-Zmiri-permissive-provenance` disables this warning")
                    );
                }
                v
            }
            NativeCallSharedMem => {
                vec![
                    note!(
                        "when memory is shared with a native function call, Miri stops tracking initialization and provenance for that memory"
                    ),
                    note!(
                        "in particular, Miri assumes that the native call initializes all memory it has access to"
                    ),
                    note!(
                        "Miri also assumes that any part of this memory may be a pointer that is permitted to point to arbitrary exposed memory"
                    ),
                    note!(
                        "what this means is that Miri will easily miss Undefined Behavior related to incorrect usage of this shared memory, so you should not take a clean Miri run as a signal that your FFI code is UB-free"
                    ),
                ]
            }
            ExternTypeReborrow => {
                assert!(self.borrow_tracker.as_ref().is_some_and(|b| {
                    matches!(
                        b.borrow().borrow_tracker_method(),
                        BorrowTrackerMethod::StackedBorrows
                    )
                }));
                vec![
                    note!(
                        "`extern type` are not compatible with the Stacked Borrows aliasing model implemented by Miri; Miri may miss bugs in this code"
                    ),
                    note!(
                        "try running with `MIRIFLAGS=-Zmiri-tree-borrows` to use the more permissive but also even more experimental Tree Borrows aliasing checks instead"
                    ),
                ]
            }
            _ => vec![],
        };

        report_msg(
            diag_level,
            title,
            vec![msg],
            notes,
            helps,
            &stacktrace,
            Some(self.threads.active_thread()),
            self,
        );
    }
}

impl<'tcx> EvalContextExt<'tcx> for crate::MiriInterpCx<'tcx> {}
pub trait EvalContextExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
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
            "the place in the program where the ICE was triggered".to_string(),
            vec![],
            vec![],
            vec![],
            &stacktrace,
            Some(this.active_thread()),
            &this.machine,
        );
    }
}
