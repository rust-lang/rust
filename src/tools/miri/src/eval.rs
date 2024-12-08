//! Main evaluator loop and setting up the initial stack frame.

use std::ffi::{OsStr, OsString};
use std::panic::{self, AssertUnwindSafe};
use std::path::PathBuf;
use std::task::Poll;
use std::{iter, thread};

use rustc_abi::ExternAbi;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_hir::def::Namespace;
use rustc_hir::def_id::DefId;
use rustc_middle::ty::layout::{LayoutCx, LayoutOf};
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_session::config::EntryFnType;

use crate::concurrency::thread::TlsAllocAction;
use crate::diagnostics::report_leaks;
use crate::shims::tls;
use crate::*;

/// When the main thread would exit, we will yield to any other thread that is ready to execute.
/// But we must only do that a finite number of times, or a background thread running `loop {}`
/// will hang the program.
const MAIN_THREAD_YIELDS_AT_SHUTDOWN: u32 = 256;

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum AlignmentCheck {
    /// Do not check alignment.
    None,
    /// Check alignment "symbolically", i.e., using only the requested alignment for an allocation and not its real base address.
    Symbolic,
    /// Check alignment on the actual physical integer address.
    Int,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum RejectOpWith {
    /// Isolated op is rejected with an abort of the machine.
    Abort,

    /// If not Abort, miri returns an error for an isolated op.
    /// Following options determine if user should be warned about such error.
    /// Do not print warning about rejected isolated op.
    NoWarning,

    /// Print a warning about rejected isolated op, with backtrace.
    Warning,

    /// Print a warning about rejected isolated op, without backtrace.
    WarningWithoutBacktrace,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum IsolatedOp {
    /// Reject an op requiring communication with the host. By
    /// default, miri rejects the op with an abort. If not, it returns
    /// an error code, and prints a warning about it. Warning levels
    /// are controlled by `RejectOpWith` enum.
    Reject(RejectOpWith),

    /// Execute op requiring communication with the host, i.e. disable isolation.
    Allow,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum BacktraceStyle {
    /// Prints a terser backtrace which ideally only contains relevant information.
    Short,
    /// Prints a backtrace with all possible information.
    Full,
    /// Prints only the frame that the error occurs in.
    Off,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ValidationMode {
    /// Do not perform any kind of validation.
    No,
    /// Validate the interior of the value, but not things behind references.
    Shallow,
    /// Fully recursively validate references.
    Deep,
}

/// Configuration needed to spawn a Miri instance.
#[derive(Clone)]
pub struct MiriConfig {
    /// The host environment snapshot to use as basis for what is provided to the interpreted program.
    /// (This is still subject to isolation as well as `forwarded_env_vars`.)
    pub env: Vec<(OsString, OsString)>,
    /// Determine if validity checking is enabled.
    pub validation: ValidationMode,
    /// Determines if Stacked Borrows or Tree Borrows is enabled.
    pub borrow_tracker: Option<BorrowTrackerMethod>,
    /// Whether `core::ptr::Unique` receives special treatment.
    /// If `true` then `Unique` is reborrowed with its own new tag and permission,
    /// otherwise `Unique` is just another raw pointer.
    pub unique_is_unique: bool,
    /// Controls alignment checking.
    pub check_alignment: AlignmentCheck,
    /// Action for an op requiring communication with the host.
    pub isolated_op: IsolatedOp,
    /// Determines if memory leaks should be ignored.
    pub ignore_leaks: bool,
    /// Environment variables that should always be forwarded from the host.
    pub forwarded_env_vars: Vec<String>,
    /// Additional environment variables that should be set in the interpreted program.
    pub set_env_vars: FxHashMap<String, String>,
    /// Command-line arguments passed to the interpreted program.
    pub args: Vec<String>,
    /// The seed to use when non-determinism or randomness are required (e.g. ptr-to-int cast, `getrandom()`).
    pub seed: Option<u64>,
    /// The stacked borrows pointer ids to report about
    pub tracked_pointer_tags: FxHashSet<BorTag>,
    /// The allocation ids to report about.
    pub tracked_alloc_ids: FxHashSet<AllocId>,
    /// For the tracked alloc ids, also report read/write accesses.
    pub track_alloc_accesses: bool,
    /// Determine if data race detection should be enabled
    pub data_race_detector: bool,
    /// Determine if weak memory emulation should be enabled. Requires data race detection to be enabled
    pub weak_memory_emulation: bool,
    /// Track when an outdated (weak memory) load happens.
    pub track_outdated_loads: bool,
    /// Rate of spurious failures for compare_exchange_weak atomic operations,
    /// between 0.0 and 1.0, defaulting to 0.8 (80% chance of failure).
    pub cmpxchg_weak_failure_rate: f64,
    /// If `Some`, enable the `measureme` profiler, writing results to a file
    /// with the specified prefix.
    pub measureme_out: Option<String>,
    /// Which style to use for printing backtraces.
    pub backtrace_style: BacktraceStyle,
    /// Which provenance to use for int2ptr casts
    pub provenance_mode: ProvenanceMode,
    /// Whether to ignore any output by the program. This is helpful when debugging miri
    /// as its messages don't get intermingled with the program messages.
    pub mute_stdout_stderr: bool,
    /// The probability of the active thread being preempted at the end of each basic block.
    pub preemption_rate: f64,
    /// Report the current instruction being executed every N basic blocks.
    pub report_progress: Option<u32>,
    /// Whether Stacked Borrows and Tree Borrows retagging should recurse into fields of datatypes.
    pub retag_fields: RetagFields,
    /// The location of a shared object file to load when calling external functions
    /// FIXME! consider allowing users to specify paths to multiple files, or to a directory
    pub native_lib: Option<PathBuf>,
    /// Run a garbage collector for BorTags every N basic blocks.
    pub gc_interval: u32,
    /// The number of CPUs to be reported by miri.
    pub num_cpus: u32,
    /// Requires Miri to emulate pages of a certain size
    pub page_size: Option<u64>,
    /// Whether to collect a backtrace when each allocation is created, just in case it leaks.
    pub collect_leak_backtraces: bool,
    /// Probability for address reuse.
    pub address_reuse_rate: f64,
    /// Probability for address reuse across threads.
    pub address_reuse_cross_thread_rate: f64,
}

impl Default for MiriConfig {
    fn default() -> MiriConfig {
        MiriConfig {
            env: vec![],
            validation: ValidationMode::Shallow,
            borrow_tracker: Some(BorrowTrackerMethod::StackedBorrows),
            unique_is_unique: false,
            check_alignment: AlignmentCheck::Int,
            isolated_op: IsolatedOp::Reject(RejectOpWith::Abort),
            ignore_leaks: false,
            forwarded_env_vars: vec![],
            set_env_vars: FxHashMap::default(),
            args: vec![],
            seed: None,
            tracked_pointer_tags: FxHashSet::default(),
            tracked_alloc_ids: FxHashSet::default(),
            track_alloc_accesses: false,
            data_race_detector: true,
            weak_memory_emulation: true,
            track_outdated_loads: false,
            cmpxchg_weak_failure_rate: 0.8, // 80%
            measureme_out: None,
            backtrace_style: BacktraceStyle::Short,
            provenance_mode: ProvenanceMode::Default,
            mute_stdout_stderr: false,
            preemption_rate: 0.01, // 1%
            report_progress: None,
            retag_fields: RetagFields::Yes,
            native_lib: None,
            gc_interval: 10_000,
            num_cpus: 1,
            page_size: None,
            collect_leak_backtraces: true,
            address_reuse_rate: 0.5,
            address_reuse_cross_thread_rate: 0.1,
        }
    }
}

/// The state of the main thread. Implementation detail of `on_main_stack_empty`.
#[derive(Default, Debug)]
enum MainThreadState<'tcx> {
    #[default]
    Running,
    TlsDtors(tls::TlsDtorsState<'tcx>),
    Yield {
        remaining: u32,
    },
    Done,
}

impl<'tcx> MainThreadState<'tcx> {
    fn on_main_stack_empty(
        &mut self,
        this: &mut MiriInterpCx<'tcx>,
    ) -> InterpResult<'tcx, Poll<()>> {
        use MainThreadState::*;
        match self {
            Running => {
                *self = TlsDtors(Default::default());
            }
            TlsDtors(state) =>
                match state.on_stack_empty(this)? {
                    Poll::Pending => {} // just keep going
                    Poll::Ready(()) => {
                        // Give background threads a chance to finish by yielding the main thread a
                        // couple of times -- but only if we would also preempt threads randomly.
                        if this.machine.preemption_rate > 0.0 {
                            // There is a non-zero chance they will yield back to us often enough to
                            // make Miri terminate eventually.
                            *self = Yield { remaining: MAIN_THREAD_YIELDS_AT_SHUTDOWN };
                        } else {
                            // The other threads did not get preempted, so no need to yield back to
                            // them.
                            *self = Done;
                        }
                    }
                },
            Yield { remaining } =>
                match remaining.checked_sub(1) {
                    None => *self = Done,
                    Some(new_remaining) => {
                        *remaining = new_remaining;
                        this.yield_active_thread();
                    }
                },
            Done => {
                // Figure out exit code.
                let ret_place = this.machine.main_fn_ret_place.clone().unwrap();
                let exit_code = this.read_target_isize(&ret_place)?;
                // Deal with our thread-local memory. We do *not* want to actually free it, instead we consider TLS
                // to be like a global `static`, so that all memory reached by it is considered to "not leak".
                this.terminate_active_thread(TlsAllocAction::Leak)?;
                // Stop interpreter loop.
                throw_machine_stop!(TerminationInfo::Exit { code: exit_code, leak_check: true });
            }
        }
        interp_ok(Poll::Pending)
    }
}

/// Returns a freshly created `InterpCx`.
/// Public because this is also used by `priroda`.
pub fn create_ecx<'tcx>(
    tcx: TyCtxt<'tcx>,
    entry_id: DefId,
    entry_type: EntryFnType,
    config: &MiriConfig,
) -> InterpResult<'tcx, InterpCx<'tcx, MiriMachine<'tcx>>> {
    let typing_env = ty::TypingEnv::fully_monomorphized();
    let layout_cx = LayoutCx::new(tcx, typing_env);
    let mut ecx =
        InterpCx::new(tcx, rustc_span::DUMMY_SP, typing_env, MiriMachine::new(config, layout_cx));

    // Some parts of initialization require a full `InterpCx`.
    MiriMachine::late_init(&mut ecx, config, {
        let mut state = MainThreadState::default();
        // Cannot capture anything GC-relevant here.
        Box::new(move |m| state.on_main_stack_empty(m))
    })?;

    // Make sure we have MIR. We check MIR for some stable monomorphic function in libcore.
    let sentinel =
        helpers::try_resolve_path(tcx, &["core", "ascii", "escape_default"], Namespace::ValueNS);
    if !matches!(sentinel, Some(s) if tcx.is_mir_available(s.def.def_id())) {
        tcx.dcx().fatal(
            "the current sysroot was built without `-Zalways-encode-mir`, or libcore seems missing. \
            Use `cargo miri setup` to prepare a sysroot that is suitable for Miri."
        );
    }

    // Setup first stack frame.
    let entry_instance = ty::Instance::mono(tcx, entry_id);

    // First argument is constructed later, because it's skipped if the entry function uses #[start].

    // Second argument (argc): length of `config.args`.
    let argc =
        ImmTy::from_int(i64::try_from(config.args.len()).unwrap(), ecx.machine.layouts.isize);
    // Third argument (`argv`): created from `config.args`.
    let argv = {
        // Put each argument in memory, collect pointers.
        let mut argvs = Vec::<Immediate<Provenance>>::with_capacity(config.args.len());
        for arg in config.args.iter() {
            // Make space for `0` terminator.
            let size = u64::try_from(arg.len()).unwrap().strict_add(1);
            let arg_type = Ty::new_array(tcx, tcx.types.u8, size);
            let arg_place =
                ecx.allocate(ecx.layout_of(arg_type)?, MiriMemoryKind::Machine.into())?;
            ecx.write_os_str_to_c_str(OsStr::new(arg), arg_place.ptr(), size)?;
            ecx.mark_immutable(&arg_place);
            argvs.push(arg_place.to_ref(&ecx));
        }
        // Make an array with all these pointers, in the Miri memory.
        let argvs_layout = ecx.layout_of(Ty::new_array(
            tcx,
            Ty::new_imm_ptr(tcx, tcx.types.u8),
            u64::try_from(argvs.len()).unwrap(),
        ))?;
        let argvs_place = ecx.allocate(argvs_layout, MiriMemoryKind::Machine.into())?;
        for (idx, arg) in argvs.into_iter().enumerate() {
            let place = ecx.project_field(&argvs_place, idx)?;
            ecx.write_immediate(arg, &place)?;
        }
        ecx.mark_immutable(&argvs_place);
        // Store `argc` and `argv` for macOS `_NSGetArg{c,v}`.
        {
            let argc_place =
                ecx.allocate(ecx.machine.layouts.isize, MiriMemoryKind::Machine.into())?;
            ecx.write_immediate(*argc, &argc_place)?;
            ecx.mark_immutable(&argc_place);
            ecx.machine.argc = Some(argc_place.ptr());

            let argv_place = ecx.allocate(
                ecx.layout_of(Ty::new_imm_ptr(tcx, tcx.types.unit))?,
                MiriMemoryKind::Machine.into(),
            )?;
            ecx.write_pointer(argvs_place.ptr(), &argv_place)?;
            ecx.mark_immutable(&argv_place);
            ecx.machine.argv = Some(argv_place.ptr());
        }
        // Store command line as UTF-16 for Windows `GetCommandLineW`.
        {
            // Construct a command string with all the arguments.
            let cmd_utf16: Vec<u16> = args_to_utf16_command_string(config.args.iter());

            let cmd_type =
                Ty::new_array(tcx, tcx.types.u16, u64::try_from(cmd_utf16.len()).unwrap());
            let cmd_place =
                ecx.allocate(ecx.layout_of(cmd_type)?, MiriMemoryKind::Machine.into())?;
            ecx.machine.cmd_line = Some(cmd_place.ptr());
            // Store the UTF-16 string. We just allocated so we know the bounds are fine.
            for (idx, &c) in cmd_utf16.iter().enumerate() {
                let place = ecx.project_field(&cmd_place, idx)?;
                ecx.write_scalar(Scalar::from_u16(c), &place)?;
            }
            ecx.mark_immutable(&cmd_place);
        }
        ecx.mplace_to_ref(&argvs_place)?
    };

    // Return place (in static memory so that it does not count as leak).
    let ret_place = ecx.allocate(ecx.machine.layouts.isize, MiriMemoryKind::Machine.into())?;
    ecx.machine.main_fn_ret_place = Some(ret_place.clone());
    // Call start function.

    match entry_type {
        EntryFnType::Main { .. } => {
            let start_id = tcx.lang_items().start_fn().unwrap_or_else(|| {
                tcx.dcx().fatal(
                    "could not find start function. Make sure the entry point is marked with `#[start]`."
                );
            });
            let main_ret_ty = tcx.fn_sig(entry_id).no_bound_vars().unwrap().output();
            let main_ret_ty = main_ret_ty.no_bound_vars().unwrap();
            let start_instance = ty::Instance::try_resolve(
                tcx,
                typing_env,
                start_id,
                tcx.mk_args(&[ty::GenericArg::from(main_ret_ty)]),
            )
            .unwrap()
            .unwrap();

            let main_ptr = ecx.fn_ptr(FnVal::Instance(entry_instance));

            // Always using DEFAULT is okay since we don't support signals in Miri anyway.
            // (This means we are effectively ignoring `-Zon-broken-pipe`.)
            let sigpipe = rustc_session::config::sigpipe::DEFAULT;

            ecx.call_function(
                start_instance,
                ExternAbi::Rust,
                &[
                    ImmTy::from_scalar(
                        Scalar::from_pointer(main_ptr, &ecx),
                        // FIXME use a proper fn ptr type
                        ecx.machine.layouts.const_raw_ptr,
                    ),
                    argc,
                    argv,
                    ImmTy::from_uint(sigpipe, ecx.machine.layouts.u8),
                ],
                Some(&ret_place),
                StackPopCleanup::Root { cleanup: true },
            )?;
        }
        EntryFnType::Start => {
            ecx.call_function(
                entry_instance,
                ExternAbi::Rust,
                &[argc, argv],
                Some(&ret_place),
                StackPopCleanup::Root { cleanup: true },
            )?;
        }
    }

    interp_ok(ecx)
}

/// Evaluates the entry function specified by `entry_id`.
/// Returns `Some(return_code)` if program executed completed.
/// Returns `None` if an evaluation error occurred.
#[expect(clippy::needless_lifetimes)]
pub fn eval_entry<'tcx>(
    tcx: TyCtxt<'tcx>,
    entry_id: DefId,
    entry_type: EntryFnType,
    config: MiriConfig,
) -> Option<i64> {
    // Copy setting before we move `config`.
    let ignore_leaks = config.ignore_leaks;

    let mut ecx = match create_ecx(tcx, entry_id, entry_type, &config).report_err() {
        Ok(v) => v,
        Err(err) => {
            let (kind, backtrace) = err.into_parts();
            backtrace.print_backtrace();
            panic!("Miri initialization error: {kind:?}")
        }
    };

    // Perform the main execution.
    let res: thread::Result<InterpResult<'_, !>> =
        panic::catch_unwind(AssertUnwindSafe(|| ecx.run_threads()));
    let res = res.unwrap_or_else(|panic_payload| {
        ecx.handle_ice();
        panic::resume_unwind(panic_payload)
    });
    // `Ok` can never happen.
    let Err(err) = res.report_err();

    // Machine cleanup. Only do this if all threads have terminated; threads that are still running
    // might cause Stacked Borrows errors (https://github.com/rust-lang/miri/issues/2396).
    if ecx.have_all_terminated() {
        // Even if all threads have terminated, we have to beware of data races since some threads
        // might not have joined the main thread (https://github.com/rust-lang/miri/issues/2020,
        // https://github.com/rust-lang/miri/issues/2508).
        ecx.allow_data_races_all_threads_done();
        EnvVars::cleanup(&mut ecx).expect("error during env var cleanup");
    }

    // Process the result.
    let (return_code, leak_check) = report_error(&ecx, err)?;
    if leak_check && !ignore_leaks {
        // Check for thread leaks.
        if !ecx.have_all_terminated() {
            tcx.dcx().err("the main thread terminated without waiting for all remaining threads");
            tcx.dcx().note("set `MIRIFLAGS=-Zmiri-ignore-leaks` to disable this check");
            return None;
        }
        // Check for memory leaks.
        info!("Additional static roots: {:?}", ecx.machine.static_roots);
        let leaks = ecx.take_leaked_allocations(|ecx| &ecx.machine.static_roots);
        if !leaks.is_empty() {
            report_leaks(&ecx, leaks);
            tcx.dcx().note("set `MIRIFLAGS=-Zmiri-ignore-leaks` to disable this check");
            // Ignore the provided return code - let the reported error
            // determine the return code.
            return None;
        }
    }
    Some(return_code)
}

/// Turns an array of arguments into a Windows command line string.
///
/// The string will be UTF-16 encoded and NUL terminated.
///
/// Panics if the zeroth argument contains the `"` character because doublequotes
/// in `argv[0]` cannot be encoded using the standard command line parsing rules.
///
/// Further reading:
/// * [Parsing C++ command-line arguments](https://docs.microsoft.com/en-us/cpp/cpp/main-function-command-line-args?view=msvc-160#parsing-c-command-line-arguments)
/// * [The C/C++ Parameter Parsing Rules](https://daviddeley.com/autohotkey/parameters/parameters.htm#WINCRULES)
fn args_to_utf16_command_string<I, T>(mut args: I) -> Vec<u16>
where
    I: Iterator<Item = T>,
    T: AsRef<str>,
{
    // Parse argv[0]. Slashes aren't escaped. Literal double quotes are not allowed.
    let mut cmd = {
        let arg0 = if let Some(arg0) = args.next() {
            arg0
        } else {
            return vec![0];
        };
        let arg0 = arg0.as_ref();
        if arg0.contains('"') {
            panic!("argv[0] cannot contain a doublequote (\") character");
        } else {
            // Always surround argv[0] with quotes.
            let mut s = String::new();
            s.push('"');
            s.push_str(arg0);
            s.push('"');
            s
        }
    };

    // Build the other arguments.
    for arg in args {
        let arg = arg.as_ref();
        cmd.push(' ');
        if arg.is_empty() {
            cmd.push_str("\"\"");
        } else if !arg.bytes().any(|c| matches!(c, b'"' | b'\t' | b' ')) {
            // No quote, tab, or space -- no escaping required.
            cmd.push_str(arg);
        } else {
            // Spaces and tabs are escaped by surrounding them in quotes.
            // Quotes are themselves escaped by using backslashes when in a
            // quoted block.
            // Backslashes only need to be escaped when one or more are directly
            // followed by a quote. Otherwise they are taken literally.

            cmd.push('"');
            let mut chars = arg.chars().peekable();
            loop {
                let mut nslashes = 0;
                while let Some(&'\\') = chars.peek() {
                    chars.next();
                    nslashes += 1;
                }

                match chars.next() {
                    Some('"') => {
                        cmd.extend(iter::repeat('\\').take(nslashes * 2 + 1));
                        cmd.push('"');
                    }
                    Some(c) => {
                        cmd.extend(iter::repeat('\\').take(nslashes));
                        cmd.push(c);
                    }
                    None => {
                        cmd.extend(iter::repeat('\\').take(nslashes * 2));
                        break;
                    }
                }
            }
            cmd.push('"');
        }
    }

    if cmd.contains('\0') {
        panic!("interior null in command line arguments");
    }
    cmd.encode_utf16().chain(iter::once(0)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[should_panic(expected = "argv[0] cannot contain a doublequote (\") character")]
    fn windows_argv0_panic_on_quote() {
        args_to_utf16_command_string(["\""].iter());
    }
    #[test]
    fn windows_argv0_no_escape() {
        // Ensure that a trailing backslash in argv[0] is not escaped.
        let cmd = String::from_utf16_lossy(&args_to_utf16_command_string(
            [r"C:\Program Files\", "arg1", "arg 2", "arg \" 3"].iter(),
        ));
        assert_eq!(cmd.trim_end_matches('\0'), r#""C:\Program Files\" arg1 "arg 2" "arg \" 3""#);
    }
}
