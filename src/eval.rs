//! Main evaluator loop and setting up the initial stack frame.

use std::ffi::OsStr;

use rand::rngs::StdRng;
use rand::SeedableRng;

use rustc::hir::def_id::DefId;
use rustc::ty::layout::{LayoutOf, Size};
use rustc::ty::{self, TyCtxt};

use crate::*;

/// Configuration needed to spawn a Miri instance.
#[derive(Clone)]
pub struct MiriConfig {
    /// Determine if validity checking and Stacked Borrows are enabled.
    pub validate: bool,
    /// Determines if communication with the host environment is enabled.
    pub communicate: bool,
    /// Determines if memory leaks should be ignored.
    pub ignore_leaks: bool,
    /// Environment variables that should always be isolated from the host.
    pub excluded_env_vars: Vec<String>,
    /// Command-line arguments passed to the interpreted program.
    pub args: Vec<String>,
    /// The seed to use when non-determinism or randomness are required (e.g. ptr-to-int cast, `getrandom()`).
    pub seed: Option<u64>,
}

/// Details of premature program termination.
pub enum TerminationInfo {
    Exit(i64),
    Abort,
}

/// Returns a freshly created `InterpCx`, along with an `MPlaceTy` representing
/// the location where the return value of the `start` lang item will be
/// written to.
/// Public because this is also used by `priroda`.
pub fn create_ecx<'mir, 'tcx: 'mir>(
    tcx: TyCtxt<'tcx>,
    main_id: DefId,
    config: MiriConfig,
) -> InterpResult<'tcx, (InterpCx<'mir, 'tcx, Evaluator<'tcx>>, MPlaceTy<'tcx, Tag>)> {
    let mut ecx = InterpCx::new(
        tcx.at(syntax::source_map::DUMMY_SP),
        ty::ParamEnv::reveal_all(),
        Evaluator::new(config.communicate),
        MemoryExtra::new(StdRng::seed_from_u64(config.seed.unwrap_or(0)), config.validate),
    );
    // Complete initialization.
    EnvVars::init(&mut ecx, config.excluded_env_vars);

    // Setup first stack-frame
    let main_instance = ty::Instance::mono(tcx, main_id);
    let main_mir = ecx.load_mir(main_instance.def, None)?;

    if !main_mir.return_ty().is_unit() || main_mir.arg_count != 0 {
        throw_unsup_format!("miri does not support main functions without `fn()` type signatures");
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
    .unwrap();

    // First argument: pointer to `main()`.
    let main_ptr = ecx
        .memory
        .create_fn_alloc(FnVal::Instance(main_instance));
    // Second argument (argc): length of `config.args`.
    let argc = Scalar::from_uint(config.args.len() as u128, ecx.pointer_size());
    // Third argument (`argv`): created from `config.args`.
    let argv = {
        // Put each argument in memory, collect pointers.
        let mut argvs = Vec::<Scalar<Tag>>::new();
        for arg in config.args.iter() {
            // Make space for `0` terminator.
            let size = arg.len() as u64 + 1;
            let arg_type = tcx.mk_array(tcx.types.u8, size);
            let arg_place = ecx.allocate(ecx.layout_of(arg_type)?, MiriMemoryKind::Env.into());
            ecx.write_os_str_to_c_str(OsStr::new(arg), arg_place.ptr, size)?;
            argvs.push(arg_place.ptr);
        }
        // Make an array with all these pointers, in the Miri memory.
        let argvs_layout = ecx.layout_of(
            tcx.mk_array(tcx.mk_imm_ptr(tcx.types.u8), argvs.len() as u64),
        )?;
        let argvs_place = ecx.allocate(argvs_layout, MiriMemoryKind::Env.into());
        for (idx, arg) in argvs.into_iter().enumerate() {
            let place = ecx.mplace_field(argvs_place, idx as u64)?;
            ecx.write_scalar(arg, place.into())?;
        }
        ecx.memory
            .mark_immutable(argvs_place.ptr.assert_ptr().alloc_id)?;
        // A pointer to that place is the 3rd argument for main.
        let argv = argvs_place.ptr;
        // Store `argc` and `argv` for macOS `_NSGetArg{c,v}`.
        {
            let argc_place = ecx.allocate(
                ecx.layout_of(tcx.types.isize)?,
                MiriMemoryKind::Env.into(),
            );
            ecx.write_scalar(argc, argc_place.into())?;
            ecx.machine.argc = Some(argc_place.ptr);

            let argv_place = ecx.allocate(
                ecx.layout_of(tcx.mk_imm_ptr(tcx.types.unit))?,
                MiriMemoryKind::Env.into(),
            );
            ecx.write_scalar(argv, argv_place.into())?;
            ecx.machine.argv = Some(argv_place.ptr);
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
            let cmd_type = tcx.mk_array(tcx.types.u16, cmd_utf16.len() as u64);
            let cmd_place = ecx.allocate(ecx.layout_of(cmd_type)?, MiriMemoryKind::Env.into());
            ecx.machine.cmd_line = Some(cmd_place.ptr);
            // Store the UTF-16 string. We just allocated so we know the bounds are fine.
            let char_size = Size::from_bytes(2);
            for (idx, &c) in cmd_utf16.iter().enumerate() {
                let place = ecx.mplace_field(cmd_place, idx as u64)?;
                ecx.write_scalar(Scalar::from_uint(c, char_size), place.into())?;
            }
        }
        argv
    };

    // Return place (in static memory so that it does not count as leak).
    let ret_place = ecx.allocate(
        ecx.layout_of(tcx.types.isize)?,
        MiriMemoryKind::Env.into(),
    );
    // Call start function.
    ecx.call_function(
        start_instance,
        &[main_ptr.into(), argc.into(), argv.into()],
        Some(ret_place.into()),
        StackPopCleanup::None { cleanup: true },
    )?;

    // Set the last_error to 0
    let errno_layout = ecx.layout_of(tcx.types.u32)?;
    let errno_place = ecx.allocate(errno_layout, MiriMemoryKind::Env.into());
    ecx.write_scalar(Scalar::from_u32(0), errno_place.into())?;
    ecx.machine.last_error = Some(errno_place);

    Ok((ecx, ret_place))
}

/// Evaluates the main function specified by `main_id`.
/// Returns `Some(return_code)` if program executed completed.
/// Returns `None` if an evaluation error occured.
pub fn eval_main<'tcx>(tcx: TyCtxt<'tcx>, main_id: DefId, config: MiriConfig) -> Option<i64> {
    // FIXME: We always ignore leaks on some platforms where we do not
    // correctly implement TLS destructors.
    let target_os = tcx.sess.target.target.target_os.to_lowercase();
    let ignore_leaks = config.ignore_leaks || target_os == "windows" || target_os == "macos";

    let (mut ecx, ret_place) = match create_ecx(tcx, main_id, config) {
        Ok(v) => v,
        Err(mut err) => {
            err.print_backtrace();
            panic!("Miri initialziation error: {}", err.kind)
        }
    };

    // Perform the main execution.
    let res: InterpResult<'_, i64> = (|| {
        ecx.run()?;
        // Read the return code pointer *before* we run TLS destructors, to assert
        // that it was written to by the time that `start` lang item returned.
        let return_code = ecx.read_scalar(ret_place.into())?.not_undef()?.to_machine_isize(&ecx)?;
        ecx.run_tls_dtors()?;
        Ok(return_code)
    })();

    // Process the result.
    match res {
        Ok(return_code) => {
            if !ignore_leaks {
                let leaks = ecx.memory.leak_report();
                if leaks != 0 {
                    tcx.sess.err("the evaluated program leaked memory");
                    // Ignore the provided return code - let the reported error
                    // determine the return code.
                    return None;
                }
            }
            return Some(return_code)
        }
        Err(mut e) => {
            // Special treatment for some error kinds
            let msg = match e.kind {
                InterpError::MachineStop(ref info) => {
                    let info = info.downcast_ref::<TerminationInfo>()
                        .expect("invalid MachineStop payload");
                    match info {
                        TerminationInfo::Exit(code) => return Some(*code),
                        TerminationInfo::Abort =>
                            format!("the evaluated program aborted execution")
                    }
                }
                err_unsup!(NoMirFor(..)) =>
                    format!("{}. Did you set `MIRI_SYSROOT` to a Miri-enabled sysroot? You can prepare one with `cargo miri setup`.", e),
                InterpError::InvalidProgram(_) =>
                    bug!("This error should be impossible in Miri: {}", e),
                _ => e.to_string()
            };
            e.print_backtrace();
            if let Some(frame) = ecx.stack().last() {
                let span = frame.current_source_info().unwrap().span;

                let msg = format!("Miri evaluation error: {}", msg);
                let mut err = ecx.tcx.sess.struct_span_err(span, msg.as_str());
                let frames = ecx.generate_stacktrace(None);
                err.span_label(span, msg);
                // We iterate with indices because we need to look at the next frame (the caller).
                for idx in 0..frames.len() {
                    let frame_info = &frames[idx];
                    let call_site_is_local = frames.get(idx + 1).map_or(false, |caller_info| {
                        caller_info.instance.def_id().is_local()
                    });
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
    }
}
