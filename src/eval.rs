//! Main evaluator loop and setting up the initial stack frame.

use std::convert::TryFrom;
use std::ffi::OsStr;

use log::info;

use rustc_hir::def_id::DefId;
use rustc_middle::ty::{self, layout::LayoutCx, TyCtxt};
use rustc_target::abi::LayoutOf;
use rustc_target::spec::abi::Abi;

use crate::*;

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

/// Configuration needed to spawn a Miri instance.
#[derive(Clone)]
pub struct MiriConfig {
    /// Determine if validity checking is enabled.
    pub validate: bool,
    /// Determines if Stacked Borrows is enabled.
    pub stacked_borrows: bool,
    /// Controls alignment checking.
    pub check_alignment: AlignmentCheck,
    /// Controls function [ABI](Abi) checking.
    pub check_abi: bool,
    /// Action for an op requiring communication with the host.
    pub isolated_op: IsolatedOp,
    /// Determines if memory leaks should be ignored.
    pub ignore_leaks: bool,
    /// Environment variables that should always be isolated from the host.
    pub excluded_env_vars: Vec<String>,
    /// Command-line arguments passed to the interpreted program.
    pub args: Vec<String>,
    /// The seed to use when non-determinism or randomness are required (e.g. ptr-to-int cast, `getrandom()`).
    pub seed: Option<u64>,
    /// The stacked borrows pointer id to report about
    pub tracked_pointer_tag: Option<PtrId>,
    /// The stacked borrows call ID to report about
    pub tracked_call_id: Option<CallId>,
    /// The allocation id to report about.
    pub tracked_alloc_id: Option<AllocId>,
    /// Whether to track raw pointers in stacked borrows.
    pub track_raw: bool,
    /// Determine if data race detection should be enabled
    pub data_race_detector: bool,
    /// Rate of spurious failures for compare_exchange_weak atomic operations,
    /// between 0.0 and 1.0, defaulting to 0.8 (80% chance of failure).
    pub cmpxchg_weak_failure_rate: f64,
    /// If `Some`, enable the `measureme` profiler, writing results to a file
    /// with the specified prefix.
    pub measureme_out: Option<String>,
    /// Panic when unsupported functionality is encountered
    pub panic_on_unsupported: bool,
}

impl Default for MiriConfig {
    fn default() -> MiriConfig {
        MiriConfig {
            validate: true,
            stacked_borrows: true,
            check_alignment: AlignmentCheck::Int,
            check_abi: true,
            isolated_op: IsolatedOp::Reject(RejectOpWith::Abort),
            ignore_leaks: false,
            excluded_env_vars: vec![],
            args: vec![],
            seed: None,
            tracked_pointer_tag: None,
            tracked_call_id: None,
            tracked_alloc_id: None,
            track_raw: false,
            data_race_detector: true,
            cmpxchg_weak_failure_rate: 0.8,
            measureme_out: None,
            panic_on_unsupported: false,
        }
    }
}

/// Returns a freshly created `InterpCx`, along with an `MPlaceTy` representing
/// the location where the return value of the `start` lang item will be
/// written to.
/// Public because this is also used by `priroda`.
pub fn create_ecx<'mir, 'tcx: 'mir>(
    tcx: TyCtxt<'tcx>,
    main_id: DefId,
    config: MiriConfig,
) -> InterpResult<'tcx, (InterpCx<'mir, 'tcx, Evaluator<'mir, 'tcx>>, MPlaceTy<'tcx, Tag>)> {
    let param_env = ty::ParamEnv::reveal_all();
    let layout_cx = LayoutCx { tcx, param_env };
    let mut ecx = InterpCx::new(
        tcx,
        rustc_span::source_map::DUMMY_SP,
        param_env,
        Evaluator::new(&config, layout_cx),
        MemoryExtra::new(&config),
    );
    // Complete initialization.
    EnvVars::init(&mut ecx, config.excluded_env_vars)?;
    MemoryExtra::init_extern_statics(&mut ecx)?;

    // Make sure we have MIR. We check MIR for some stable monomorphic function in libcore.
    let sentinel = ecx.resolve_path(&["core", "ascii", "escape_default"]);
    if !tcx.is_mir_available(sentinel.def.def_id()) {
        tcx.sess.fatal("the current sysroot was built without `-Zalways-encode-mir`. Use `cargo miri setup` to prepare a sysroot that is suitable for Miri.");
    }

    // Setup first stack-frame
    let main_instance = ty::Instance::mono(tcx, main_id);
    let main_mir = ecx.load_mir(main_instance.def, None)?;
    if main_mir.arg_count != 0 {
        bug!("main function must not take any arguments");
    }

    let start_id = tcx.lang_items().start_fn().unwrap();
    let main_ret_ty = tcx.fn_sig(main_id).output();
    let main_ret_ty = main_ret_ty.no_bound_vars().unwrap();
    let start_instance = ty::Instance::resolve(
        tcx,
        ty::ParamEnv::reveal_all(),
        start_id,
        tcx.mk_substs(::std::iter::once(ty::subst::GenericArg::from(main_ret_ty))),
    )
    .unwrap()
    .unwrap();

    // First argument: pointer to `main()`.
    let main_ptr = ecx.memory.create_fn_alloc(FnVal::Instance(main_instance));
    // Second argument (argc): length of `config.args`.
    let argc = Scalar::from_machine_usize(u64::try_from(config.args.len()).unwrap(), &ecx);
    // Third argument (`argv`): created from `config.args`.
    let argv = {
        // Put each argument in memory, collect pointers.
        let mut argvs = Vec::<Immediate<Tag>>::new();
        for arg in config.args.iter() {
            // Make space for `0` terminator.
            let size = u64::try_from(arg.len()).unwrap().checked_add(1).unwrap();
            let arg_type = tcx.mk_array(tcx.types.u8, size);
            let arg_place =
                ecx.allocate(ecx.layout_of(arg_type)?, MiriMemoryKind::Machine.into())?;
            ecx.write_os_str_to_c_str(OsStr::new(arg), arg_place.ptr, size)?;
            ecx.mark_immutable(&*arg_place);
            argvs.push(arg_place.to_ref(&ecx));
        }
        // Make an array with all these pointers, in the Miri memory.
        let argvs_layout = ecx.layout_of(
            tcx.mk_array(tcx.mk_imm_ptr(tcx.types.u8), u64::try_from(argvs.len()).unwrap()),
        )?;
        let argvs_place = ecx.allocate(argvs_layout, MiriMemoryKind::Machine.into())?;
        for (idx, arg) in argvs.into_iter().enumerate() {
            let place = ecx.mplace_field(&argvs_place, idx)?;
            ecx.write_immediate(arg, &place.into())?;
        }
        ecx.mark_immutable(&*argvs_place);
        // A pointer to that place is the 3rd argument for main.
        let argv = argvs_place.to_ref(&ecx);
        // Store `argc` and `argv` for macOS `_NSGetArg{c,v}`.
        {
            let argc_place =
                ecx.allocate(ecx.machine.layouts.isize, MiriMemoryKind::Machine.into())?;
            ecx.write_scalar(argc, &argc_place.into())?;
            ecx.mark_immutable(&*argc_place);
            ecx.machine.argc = Some(*argc_place);

            let argv_place = ecx.allocate(
                ecx.layout_of(tcx.mk_imm_ptr(tcx.types.unit))?,
                MiriMemoryKind::Machine.into(),
            )?;
            ecx.write_immediate(argv, &argv_place.into())?;
            ecx.mark_immutable(&*argv_place);
            ecx.machine.argv = Some(*argv_place);
        }
        // Store command line as UTF-16 for Windows `GetCommandLineW`.
        {
            // Construct a command string with all the aguments.
            let mut cmd = String::new();
            for arg in config.args.iter() {
                if !cmd.is_empty() {
                    cmd.push(' ');
                }
                cmd.push_str(&*shell_escape::windows::escape(arg.as_str().into()));
            }
            // Don't forget `0` terminator.
            cmd.push(std::char::from_u32(0).unwrap());

            let cmd_utf16: Vec<u16> = cmd.encode_utf16().collect();
            let cmd_type = tcx.mk_array(tcx.types.u16, u64::try_from(cmd_utf16.len()).unwrap());
            let cmd_place =
                ecx.allocate(ecx.layout_of(cmd_type)?, MiriMemoryKind::Machine.into())?;
            ecx.machine.cmd_line = Some(*cmd_place);
            // Store the UTF-16 string. We just allocated so we know the bounds are fine.
            for (idx, &c) in cmd_utf16.iter().enumerate() {
                let place = ecx.mplace_field(&cmd_place, idx)?;
                ecx.write_scalar(Scalar::from_u16(c), &place.into())?;
            }
            ecx.mark_immutable(&*cmd_place);
        }
        argv
    };

    // Return place (in static memory so that it does not count as leak).
    let ret_place = ecx.allocate(ecx.machine.layouts.isize, MiriMemoryKind::Machine.into())?;
    // Call start function.
    ecx.call_function(
        start_instance,
        Abi::Rust,
        &[Scalar::from_pointer(main_ptr, &ecx).into(), argc.into(), argv],
        Some(&ret_place.into()),
        StackPopCleanup::None { cleanup: true },
    )?;

    Ok((ecx, ret_place))
}

/// Evaluates the main function specified by `main_id`.
/// Returns `Some(return_code)` if program executed completed.
/// Returns `None` if an evaluation error occured.
pub fn eval_main<'tcx>(tcx: TyCtxt<'tcx>, main_id: DefId, config: MiriConfig) -> Option<i64> {
    // Copy setting before we move `config`.
    let ignore_leaks = config.ignore_leaks;

    let (mut ecx, ret_place) = match create_ecx(tcx, main_id, config) {
        Ok(v) => v,
        Err(err) => {
            err.print_backtrace();
            panic!("Miri initialization error: {}", err.kind())
        }
    };

    // Perform the main execution.
    let res: InterpResult<'_, i64> = (|| {
        // Main loop.
        loop {
            let info = ecx.preprocess_diagnostics();
            match ecx.schedule()? {
                SchedulingAction::ExecuteStep => {
                    assert!(ecx.step()?, "a terminated thread was scheduled for execution");
                }
                SchedulingAction::ExecuteTimeoutCallback => {
                    assert!(
                        ecx.machine.communicate(),
                        "scheduler callbacks require disabled isolation, but the code \
                        that created the callback did not check it"
                    );
                    ecx.run_timeout_callback()?;
                }
                SchedulingAction::ExecuteDtors => {
                    // This will either enable the thread again (so we go back
                    // to `ExecuteStep`), or determine that this thread is done
                    // for good.
                    ecx.schedule_next_tls_dtor_for_active_thread()?;
                }
                SchedulingAction::Stop => {
                    break;
                }
            }
            ecx.process_diagnostics(info);
        }
        let return_code = ecx.read_scalar(&ret_place.into())?.to_machine_isize(&ecx)?;
        Ok(return_code)
    })();

    // Machine cleanup.
    EnvVars::cleanup(&mut ecx).unwrap();

    // Process the result.
    match res {
        Ok(return_code) => {
            if !ignore_leaks {
                // Check for thread leaks.
                if !ecx.have_all_terminated() {
                    tcx.sess.err(
                        "the main thread terminated without waiting for all remaining threads",
                    );
                    tcx.sess.note_without_error("pass `-Zmiri-ignore-leaks` to disable this check");
                    return None;
                }
                // Check for memory leaks.
                info!("Additonal static roots: {:?}", ecx.machine.static_roots);
                let leaks = ecx.memory.leak_report(&ecx.machine.static_roots);
                if leaks != 0 {
                    tcx.sess.err("the evaluated program leaked memory");
                    tcx.sess.note_without_error("pass `-Zmiri-ignore-leaks` to disable this check");
                    // Ignore the provided return code - let the reported error
                    // determine the return code.
                    return None;
                }
            }
            Some(return_code)
        }
        Err(e) => report_error(&ecx, e),
    }
}
