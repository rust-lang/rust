//! Main evaluator loop and setting up the initial stack frame.

use std::ffi::{OsStr, OsString};
use std::panic::{self, AssertUnwindSafe};
use std::path::PathBuf;
use std::rc::Rc;
use std::task::Poll;
use std::{iter, thread};

use rustc_abi::ExternAbi;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_hir::def::Namespace;
use rustc_hir::def_id::DefId;
use rustc_middle::ty::layout::{HasTyCtxt, HasTypingEnv, LayoutCx};
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_session::config::EntryFnType;

use crate::concurrency::GenmcCtx;
use crate::concurrency::thread::TlsAllocAction;
use crate::diagnostics::report_leaks;
use crate::shims::{global_ctor, tls};
use crate::*;

#[derive(Copy, Clone, Debug)]
pub enum MiriEntryFnType {
    MiriStart,
    Rustc(EntryFnType),
}

/// When the main thread would exit, we will yield to any other thread that is ready to execute.
/// But we must only do that a finite number of times, or a background thread running `loop {}`
/// will hang the program.
const MAIN_THREAD_YIELDS_AT_SHUTDOWN: u32 = 256;

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
    /// The stacked borrows pointer ids to report about.
    pub tracked_pointer_tags: FxHashSet<BorTag>,
    /// The allocation ids to report about.
    pub tracked_alloc_ids: FxHashSet<AllocId>,
    /// For the tracked alloc ids, also report read/write accesses.
    pub track_alloc_accesses: bool,
    /// Determine if data race detection should be enabled.
    pub data_race_detector: bool,
    /// Determine if weak memory emulation should be enabled. Requires data race detection to be enabled.
    pub weak_memory_emulation: bool,
    /// Determine if we are running in GenMC mode and with which settings. In GenMC mode, Miri will explore multiple concurrent executions of the given program.
    pub genmc_config: Option<GenmcConfig>,
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
    /// Which provenance to use for int2ptr casts.
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
    /// The location of the shared object files to load when calling external functions
    pub native_lib: Vec<PathBuf>,
    /// Whether to enable the new native lib tracing system.
    pub native_lib_enable_tracing: bool,
    /// Run a garbage collector for BorTags every N basic blocks.
    pub gc_interval: u32,
    /// The number of CPUs to be reported by miri.
    pub num_cpus: u32,
    /// Requires Miri to emulate pages of a certain size.
    pub page_size: Option<u64>,
    /// Whether to collect a backtrace when each allocation is created, just in case it leaks.
    pub collect_leak_backtraces: bool,
    /// Probability for address reuse.
    pub address_reuse_rate: f64,
    /// Probability for address reuse across threads.
    pub address_reuse_cross_thread_rate: f64,
    /// Round Robin scheduling with no preemption.
    pub fixed_scheduling: bool,
    /// Always prefer the intrinsic fallback body over the native Miri implementation.
    pub force_intrinsic_fallback: bool,
    /// Whether floating-point operations can behave non-deterministically.
    pub float_nondet: bool,
    /// Whether floating-point operations can have a non-deterministic rounding error.
    pub float_rounding_error: FloatRoundingErrorMode,
    /// Whether Miri artifically introduces short reads/writes on file descriptors.
    pub short_fd_operations: bool,
}

impl Default for MiriConfig {
    fn default() -> MiriConfig {
        MiriConfig {
            env: vec![],
            validation: ValidationMode::Shallow,
            borrow_tracker: Some(BorrowTrackerMethod::StackedBorrows),
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
            genmc_config: None,
            track_outdated_loads: false,
            cmpxchg_weak_failure_rate: 0.8, // 80%
            measureme_out: None,
            backtrace_style: BacktraceStyle::Short,
            provenance_mode: ProvenanceMode::Default,
            mute_stdout_stderr: false,
            preemption_rate: 0.01, // 1%
            report_progress: None,
            retag_fields: RetagFields::Yes,
            native_lib: vec![],
            native_lib_enable_tracing: false,
            gc_interval: 10_000,
            num_cpus: 1,
            page_size: None,
            collect_leak_backtraces: true,
            address_reuse_rate: 0.5,
            address_reuse_cross_thread_rate: 0.1,
            fixed_scheduling: false,
            force_intrinsic_fallback: false,
            float_nondet: true,
            float_rounding_error: FloatRoundingErrorMode::Random,
            short_fd_operations: true,
        }
    }
}

/// The state of the main thread. Implementation detail of `on_main_stack_empty`.
#[derive(Debug)]
enum MainThreadState<'tcx> {
    GlobalCtors {
        ctor_state: global_ctor::GlobalCtorState<'tcx>,
        /// The main function to call.
        entry_id: DefId,
        entry_type: MiriEntryFnType,
        /// Arguments passed to `main`.
        argc: ImmTy<'tcx>,
        argv: ImmTy<'tcx>,
    },
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
            GlobalCtors { ctor_state, entry_id, entry_type, argc, argv } => {
                match ctor_state.on_stack_empty(this)? {
                    Poll::Pending => {} // just keep going
                    Poll::Ready(()) => {
                        call_main(this, *entry_id, *entry_type, argc.clone(), argv.clone())?;
                        *self = Running;
                    }
                }
            }
            Running => {
                *self = TlsDtors(Default::default());
            }
            TlsDtors(state) =>
                match state.on_stack_empty(this)? {
                    Poll::Pending => {} // just keep going
                    Poll::Ready(()) => {
                        if this.machine.data_race.as_genmc_ref().is_some() {
                            // In GenMC mode, we don't yield at the end of the main thread.
                            // Instead, the `GenmcCtx` will ensure that unfinished threads get a chance to run at this point.
                            *self = Done;
                        } else {
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
                // Rust uses `isize` but the underlying type of an exit code is `i32`.
                // Do a saturating cast.
                let exit_code = i32::try_from(exit_code).unwrap_or(if exit_code >= 0 {
                    i32::MAX
                } else {
                    i32::MIN
                });
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
    entry_type: MiriEntryFnType,
    config: &MiriConfig,
    genmc_ctx: Option<Rc<GenmcCtx>>,
) -> InterpResult<'tcx, InterpCx<'tcx, MiriMachine<'tcx>>> {
    let typing_env = ty::TypingEnv::fully_monomorphized();
    let layout_cx = LayoutCx::new(tcx, typing_env);
    let mut ecx = InterpCx::new(
        tcx,
        rustc_span::DUMMY_SP,
        typing_env,
        MiriMachine::new(config, layout_cx, genmc_ctx),
    );

    // Make sure we have MIR. We check MIR for some stable monomorphic function in libcore.
    let sentinel =
        helpers::try_resolve_path(tcx, &["core", "ascii", "escape_default"], Namespace::ValueNS);
    if !matches!(sentinel, Some(s) if tcx.is_mir_available(s.def.def_id())) {
        tcx.dcx().fatal(
            "the current sysroot was built without `-Zalways-encode-mir`, or libcore seems missing.\n\
            Note that directly invoking the `miri` binary is not supported; please use `cargo miri` instead."
        );
    }

    // Compute argc and argv from `config.args`.
    let argc =
        ImmTy::from_int(i64::try_from(config.args.len()).unwrap(), ecx.machine.layouts.isize);
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
        let u8_ptr_type = Ty::new_imm_ptr(tcx, tcx.types.u8);
        let u8_ptr_ptr_type = Ty::new_imm_ptr(tcx, u8_ptr_type);
        let argvs_layout =
            ecx.layout_of(Ty::new_array(tcx, u8_ptr_type, u64::try_from(argvs.len()).unwrap()))?;
        let argvs_place = ecx.allocate(argvs_layout, MiriMemoryKind::Machine.into())?;
        for (arg, idx) in argvs.into_iter().zip(0..) {
            let place = ecx.project_index(&argvs_place, idx)?;
            ecx.write_immediate(arg, &place)?;
        }
        ecx.mark_immutable(&argvs_place);
        // Store `argc` and `argv` for macOS `_NSGetArg{c,v}`, and for the GC to see them.
        {
            let argc_place =
                ecx.allocate(ecx.machine.layouts.isize, MiriMemoryKind::Machine.into())?;
            ecx.write_immediate(*argc, &argc_place)?;
            ecx.mark_immutable(&argc_place);
            ecx.machine.argc = Some(argc_place.ptr());

            let argv_place =
                ecx.allocate(ecx.layout_of(u8_ptr_ptr_type)?, MiriMemoryKind::Machine.into())?;
            ecx.write_pointer(argvs_place.ptr(), &argv_place)?;
            ecx.mark_immutable(&argv_place);
            ecx.machine.argv = Some(argv_place.ptr());
        }
        // Store command line as UTF-16 for Windows `GetCommandLineW`.
        if tcx.sess.target.os == "windows" {
            // Construct a command string with all the arguments.
            let cmd_utf16: Vec<u16> = args_to_utf16_command_string(config.args.iter());

            let cmd_type =
                Ty::new_array(tcx, tcx.types.u16, u64::try_from(cmd_utf16.len()).unwrap());
            let cmd_place =
                ecx.allocate(ecx.layout_of(cmd_type)?, MiriMemoryKind::Machine.into())?;
            ecx.machine.cmd_line = Some(cmd_place.ptr());
            // Store the UTF-16 string. We just allocated so we know the bounds are fine.
            for (&c, idx) in cmd_utf16.iter().zip(0..) {
                let place = ecx.project_index(&cmd_place, idx)?;
                ecx.write_scalar(Scalar::from_u16(c), &place)?;
            }
            ecx.mark_immutable(&cmd_place);
        }
        let imm = argvs_place.to_ref(&ecx);
        let layout = ecx.layout_of(u8_ptr_ptr_type)?;
        ImmTy::from_immediate(imm, layout)
    };

    // Some parts of initialization require a full `InterpCx`.
    MiriMachine::late_init(&mut ecx, config, {
        let mut main_thread_state = MainThreadState::GlobalCtors {
            entry_id,
            entry_type,
            argc,
            argv,
            ctor_state: global_ctor::GlobalCtorState::default(),
        };

        // Cannot capture anything GC-relevant here.
        // `argc` and `argv` *are* GC_relevant, but they also get stored in `machine.argc` and
        // `machine.argv` so we are good.
        Box::new(move |m| main_thread_state.on_main_stack_empty(m))
    })?;

    interp_ok(ecx)
}

// Call the entry function.
fn call_main<'tcx>(
    ecx: &mut MiriInterpCx<'tcx>,
    entry_id: DefId,
    entry_type: MiriEntryFnType,
    argc: ImmTy<'tcx>,
    argv: ImmTy<'tcx>,
) -> InterpResult<'tcx, ()> {
    let tcx = ecx.tcx();

    // Setup first stack frame.
    let entry_instance = ty::Instance::mono(tcx, entry_id);

    // Return place (in static memory so that it does not count as leak).
    let ret_place = ecx.allocate(ecx.machine.layouts.isize, MiriMemoryKind::Machine.into())?;
    ecx.machine.main_fn_ret_place = Some(ret_place.clone());

    // Call start function.
    match entry_type {
        MiriEntryFnType::Rustc(EntryFnType::Main { .. }) => {
            let start_id = tcx.lang_items().start_fn().unwrap_or_else(|| {
                tcx.dcx().fatal("could not find start lang item");
            });
            let main_ret_ty = tcx.fn_sig(entry_id).no_bound_vars().unwrap().output();
            let main_ret_ty = main_ret_ty.no_bound_vars().unwrap();
            let start_instance = ty::Instance::try_resolve(
                tcx,
                ecx.typing_env(),
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
                        Scalar::from_pointer(main_ptr, ecx),
                        // FIXME use a proper fn ptr type
                        ecx.machine.layouts.const_raw_ptr,
                    ),
                    argc,
                    argv,
                    ImmTy::from_uint(sigpipe, ecx.machine.layouts.u8),
                ],
                Some(&ret_place),
                ReturnContinuation::Stop { cleanup: true },
            )?;
        }
        MiriEntryFnType::MiriStart => {
            ecx.call_function(
                entry_instance,
                ExternAbi::Rust,
                &[argc, argv],
                Some(&ret_place),
                ReturnContinuation::Stop { cleanup: true },
            )?;
        }
    }

    interp_ok(())
}

/// Evaluates the entry function specified by `entry_id`.
/// Returns `Some(return_code)` if program execution completed.
/// Returns `None` if an evaluation error occurred.
pub fn eval_entry<'tcx>(
    tcx: TyCtxt<'tcx>,
    entry_id: DefId,
    entry_type: MiriEntryFnType,
    config: &MiriConfig,
    genmc_ctx: Option<Rc<GenmcCtx>>,
) -> Option<i32> {
    // Copy setting before we move `config`.
    let ignore_leaks = config.ignore_leaks;

    if let Some(genmc_ctx) = &genmc_ctx {
        genmc_ctx.handle_execution_start();
    }

    let mut ecx = match create_ecx(tcx, entry_id, entry_type, config, genmc_ctx).report_err() {
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
    // `Ok` can never happen; the interpreter loop always exits with an "error"
    // (but that "error" might be just "regular program termination").
    let Err(err) = res.report_err();

    // Show diagnostic, if any.
    let (return_code, leak_check) = report_error(&ecx, err)?;

    // We inform GenMC that the execution is complete.
    if let Some(genmc_ctx) = ecx.machine.data_race.as_genmc_ref()
        && let Err(error) = genmc_ctx.handle_execution_end(&ecx)
    {
        // FIXME(GenMC): Improve error reporting.
        tcx.dcx().err(format!("GenMC returned an error: \"{error}\""));
        return None;
    }

    // If we get here there was no fatal error.

    // Possibly check for memory leaks.
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
                        cmd.extend(iter::repeat_n('\\', nslashes * 2 + 1));
                        cmd.push('"');
                    }
                    Some(c) => {
                        cmd.extend(iter::repeat_n('\\', nslashes));
                        cmd.push(c);
                    }
                    None => {
                        cmd.extend(iter::repeat_n('\\', nslashes * 2));
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
