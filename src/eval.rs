//! Main evaluator loop and setting up the initial stack frame.

use std::convert::TryFrom;
use std::ffi::OsStr;
use std::iter;

use log::info;

use rustc_hir::def_id::DefId;
use rustc_middle::ty::{
    self,
    layout::{LayoutCx, LayoutOf},
    TyCtxt,
};
use rustc_target::spec::abi::Abi;

use rustc_session::config::EntryFnType;

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

#[derive(Copy, Clone, PartialEq, Eq)]
pub enum BacktraceStyle {
    /// Prints a terser backtrace which ideally only contains relevant information.
    Short,
    /// Prints a backtrace with all possible information.
    Full,
    /// Prints only the frame that the error occurs in.
    Off,
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
    /// Controls integer and float validity (e.g., initialization) checking.
    pub check_number_validity: bool,
    /// Controls function [ABI](Abi) checking.
    pub check_abi: bool,
    /// Action for an op requiring communication with the host.
    pub isolated_op: IsolatedOp,
    /// Determines if memory leaks should be ignored.
    pub ignore_leaks: bool,
    /// Environment variables that should always be isolated from the host.
    pub excluded_env_vars: Vec<String>,
    /// Environment variables that should always be forwarded from the host.
    pub forwarded_env_vars: Vec<String>,
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
    pub tag_raw: bool,
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
    pub backtrace_style: BacktraceStyle,
}

impl Default for MiriConfig {
    fn default() -> MiriConfig {
        MiriConfig {
            validate: true,
            stacked_borrows: true,
            check_alignment: AlignmentCheck::Int,
            check_number_validity: false,
            check_abi: true,
            isolated_op: IsolatedOp::Reject(RejectOpWith::Abort),
            ignore_leaks: false,
            excluded_env_vars: vec![],
            forwarded_env_vars: vec![],
            args: vec![],
            seed: None,
            tracked_pointer_tag: None,
            tracked_call_id: None,
            tracked_alloc_id: None,
            tag_raw: false,
            data_race_detector: true,
            cmpxchg_weak_failure_rate: 0.8,
            measureme_out: None,
            panic_on_unsupported: false,
            backtrace_style: BacktraceStyle::Short,
        }
    }
}

/// Returns a freshly created `InterpCx`, along with an `MPlaceTy` representing
/// the location where the return value of the `start` function will be
/// written to.
/// Public because this is also used by `priroda`.
pub fn create_ecx<'mir, 'tcx: 'mir>(
    tcx: TyCtxt<'tcx>,
    entry_id: DefId,
    entry_type: EntryFnType,
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
    EnvVars::init(&mut ecx, config.excluded_env_vars, config.forwarded_env_vars)?;
    MemoryExtra::init_extern_statics(&mut ecx)?;

    // Make sure we have MIR. We check MIR for some stable monomorphic function in libcore.
    let sentinel = ecx.resolve_path(&["core", "ascii", "escape_default"]);
    if !tcx.is_mir_available(sentinel.def.def_id()) {
        tcx.sess.fatal("the current sysroot was built without `-Zalways-encode-mir`. Use `cargo miri setup` to prepare a sysroot that is suitable for Miri.");
    }

    // Setup first stack-frame
    let entry_instance = ty::Instance::mono(tcx, entry_id);

    // First argument is constructed later, because its skipped if the entry function uses #[start]

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
            let cmd_utf16: Vec<u16> = args_to_utf16_command_string(config.args.iter());

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

    match entry_type {
        EntryFnType::Main => {
            let start_id = tcx.lang_items().start_fn().unwrap();
            let main_ret_ty = tcx.fn_sig(entry_id).output();
            let main_ret_ty = main_ret_ty.no_bound_vars().unwrap();
            let start_instance = ty::Instance::resolve(
                tcx,
                ty::ParamEnv::reveal_all(),
                start_id,
                tcx.mk_substs(::std::iter::once(ty::subst::GenericArg::from(main_ret_ty))),
            )
            .unwrap()
            .unwrap();

            let main_ptr = ecx.memory.create_fn_alloc(FnVal::Instance(entry_instance));

            ecx.call_function(
                start_instance,
                Abi::Rust,
                &[Scalar::from_pointer(main_ptr, &ecx).into(), argc.into(), argv],
                Some(&ret_place.into()),
                StackPopCleanup::Root { cleanup: true },
            )?;
        }
        EntryFnType::Start => {
            ecx.call_function(
                entry_instance,
                Abi::Rust,
                &[argc.into(), argv],
                Some(&ret_place.into()),
                StackPopCleanup::Root { cleanup: true },
            )?;
        }
    }

    Ok((ecx, ret_place))
}

/// Evaluates the entry function specified by `entry_id`.
/// Returns `Some(return_code)` if program executed completed.
/// Returns `None` if an evaluation error occured.
pub fn eval_entry<'tcx>(
    tcx: TyCtxt<'tcx>,
    entry_id: DefId,
    entry_type: EntryFnType,
    config: MiriConfig,
) -> Option<i64> {
    // Copy setting before we move `config`.
    let ignore_leaks = config.ignore_leaks;

    let (mut ecx, ret_place) = match create_ecx(tcx, entry_id, entry_type, config) {
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

/// Turns an array of arguments into a Windows command line string.
///
/// The string will be UTF-16 encoded and NUL terminated.
///
/// Panics if the zeroth argument contains the `"` character because doublequotes
/// in argv[0] cannot be encoded using the standard command line parsing rules.
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
        assert_eq!(cmd.trim_end_matches("\0"), r#""C:\Program Files\" arg1 "arg 2" "arg \" 3""#);
    }
}
