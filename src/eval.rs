//! Main evaluator loop and setting up the initial stack frame.

use rand::rngs::StdRng;
use rand::SeedableRng;

use syntax::source_map::DUMMY_SP;
use rustc::ty::{self, TyCtxt};
use rustc::ty::layout::{LayoutOf, Size, Align};
use rustc::hir::def_id::DefId;

use crate::{
    InterpResult, InterpError, InterpCx, StackPopCleanup, struct_error,
    Scalar, Tag, Pointer, FnVal,
    MemoryExtra, MiriMemoryKind, Evaluator, TlsEvalContextExt, HelpersEvalContextExt,
    EnvVars,
};

/// Configuration needed to spawn a Miri instance.
#[derive(Clone)]
pub struct MiriConfig {
    /// Determine if validity checking and Stacked Borrows are enabled.
    pub validate: bool,
    /// Determines if communication with the host environment is enabled.
    pub communicate: bool,
    /// Environment variables that should always be isolated from the host.
    pub excluded_env_vars: Vec<String>,
    /// Command-line arguments passed to the interpreted program.
    pub args: Vec<String>,
    /// The seed to use when non-determinism or randomness are required (e.g. ptr-to-int cast, `getrandom()`).
    pub seed: Option<u64>,
}

// Used by priroda.
pub fn create_ecx<'mir, 'tcx: 'mir>(
    tcx: TyCtxt<'tcx>,
    main_id: DefId,
    config: MiriConfig,
) -> InterpResult<'tcx, InterpCx<'mir, 'tcx, Evaluator<'tcx>>> {
    let mut ecx = InterpCx::new(
        tcx.at(syntax::source_map::DUMMY_SP),
        ty::ParamEnv::reveal_all(),
        Evaluator::new(config.communicate),
        MemoryExtra::new(StdRng::seed_from_u64(config.seed.unwrap_or(0)), config.validate),
    );
    // Complete initialization.
    EnvVars::init(&mut ecx, config.excluded_env_vars);

    // Setup first stack-frame
    let main_instance = ty::Instance::mono(ecx.tcx.tcx, main_id);
    let main_mir = ecx.load_mir(main_instance.def, None)?;

    if !main_mir.return_ty().is_unit() || main_mir.arg_count != 0 {
        throw_unsup_format!(
            "miri does not support main functions without `fn()` type signatures"
        );
    }

    let start_id = tcx.lang_items().start_fn().unwrap();
    let main_ret_ty = tcx.fn_sig(main_id).output();
    let main_ret_ty = main_ret_ty.no_bound_vars().unwrap();
    let start_instance = ty::Instance::resolve(
        ecx.tcx.tcx,
        ty::ParamEnv::reveal_all(),
        start_id,
        ecx.tcx.mk_substs(
            ::std::iter::once(ty::subst::Kind::from(main_ret_ty)))
        ).unwrap();
    let start_mir = ecx.load_mir(start_instance.def, None)?;

    if start_mir.arg_count != 3 {
        bug!(
            "'start' lang item should have three arguments, but has {}",
            start_mir.arg_count
        );
    }

    // Return value (in static memory so that it does not count as leak).
    let ret = ecx.layout_of(start_mir.return_ty())?;
    let ret_ptr = ecx.allocate(ret, MiriMemoryKind::Static.into());

    // Push our stack frame.
    ecx.push_stack_frame(
        start_instance,
        // There is no call site.
        DUMMY_SP,
        start_mir,
        Some(ret_ptr.into()),
        StackPopCleanup::None { cleanup: true },
    )?;

    let mut args = ecx.frame().body.args_iter();

    // First argument: pointer to `main()`.
    let main_ptr = ecx.memory_mut().create_fn_alloc(FnVal::Instance(main_instance));
    let dest = ecx.local_place(args.next().unwrap())?;
    ecx.write_scalar(Scalar::Ptr(main_ptr), dest)?;

    // Second argument (argc): `1`.
    let dest = ecx.local_place(args.next().unwrap())?;
    let argc = Scalar::from_uint(config.args.len() as u128, dest.layout.size);
    ecx.write_scalar(argc, dest)?;
    // Store argc for macOS's `_NSGetArgc`.
    {
        let argc_place = ecx.allocate(dest.layout, MiriMemoryKind::Env.into());
        ecx.write_scalar(argc, argc_place.into())?;
        ecx.machine.argc = Some(argc_place.ptr.to_ptr()?);
    }

    // Third argument (`argv`): created from `config.args`.
    let dest = ecx.local_place(args.next().unwrap())?;
    // For Windows, construct a command string with all the aguments.
    let mut cmd = String::new();
    for arg in config.args.iter() {
        if !cmd.is_empty() {
            cmd.push(' ');
        }
        cmd.push_str(&*shell_escape::windows::escape(arg.as_str().into()));
    }
    // Don't forget `0` terminator.
    cmd.push(std::char::from_u32(0).unwrap());
    // Collect the pointers to the individual strings.
    let mut argvs = Vec::<Pointer<Tag>>::new();
    for arg in config.args {
        // Add `0` terminator.
        let mut arg = arg.into_bytes();
        arg.push(0);
        argvs.push(ecx.memory_mut().allocate_static_bytes(arg.as_slice(), MiriMemoryKind::Static.into()));
    }
    // Make an array with all these pointers, in the Miri memory.
    let argvs_layout = ecx.layout_of(ecx.tcx.mk_array(ecx.tcx.mk_imm_ptr(ecx.tcx.types.u8), argvs.len() as u64))?;
    let argvs_place = ecx.allocate(argvs_layout, MiriMemoryKind::Env.into());
    for (idx, arg) in argvs.into_iter().enumerate() {
        let place = ecx.mplace_field(argvs_place, idx as u64)?;
        ecx.write_scalar(Scalar::Ptr(arg), place.into())?;
    }
    ecx.memory_mut().mark_immutable(argvs_place.ptr.assert_ptr().alloc_id)?;
    // Write a pointer to that place as the argument.
    let argv = argvs_place.ptr;
    ecx.write_scalar(argv, dest)?;
    // Store `argv` for macOS `_NSGetArgv`.
    {
        let argv_place = ecx.allocate(dest.layout, MiriMemoryKind::Env.into());
        ecx.write_scalar(argv, argv_place.into())?;
        ecx.machine.argv = Some(argv_place.ptr.to_ptr()?);
    }
    // Store command line as UTF-16 for Windows `GetCommandLineW`.
    {
        let tcx = &{ecx.tcx.tcx};
        let cmd_utf16: Vec<u16> = cmd.encode_utf16().collect();
        let cmd_ptr = ecx.memory_mut().allocate(
            Size::from_bytes(cmd_utf16.len() as u64 * 2),
            Align::from_bytes(2).unwrap(),
            MiriMemoryKind::Env.into(),
        );
        ecx.machine.cmd_line = Some(cmd_ptr);
        // Store the UTF-16 string.
        let char_size = Size::from_bytes(2);
        let cmd_alloc = ecx.memory_mut().get_mut(cmd_ptr.alloc_id)?;
        let mut cur_ptr = cmd_ptr;
        for &c in cmd_utf16.iter() {
            cmd_alloc.write_scalar(
                tcx,
                cur_ptr,
                Scalar::from_uint(c, char_size).into(),
                char_size,
            )?;
            cur_ptr = cur_ptr.offset(char_size, tcx)?;
        }
    }

    assert!(args.next().is_none(), "start lang item has more arguments than expected");

    Ok(ecx)
}

pub fn eval_main<'tcx>(
    tcx: TyCtxt<'tcx>,
    main_id: DefId,
    config: MiriConfig,
) {
    let mut ecx = match create_ecx(tcx, main_id, config) {
        Ok(ecx) => ecx,
        Err(mut err) => {
            err.print_backtrace();
            panic!("Miri initialziation error: {}", err.kind)
        }
    };

    // Perform the main execution.
    let res: InterpResult<'_> = (|| {
        ecx.run()?;
        ecx.run_tls_dtors()
    })();

    // Process the result.
    match res {
        Ok(()) => {
            let leaks = ecx.memory().leak_report();
            // Disable the leak test on some platforms where we do not
            // correctly implement TLS destructors.
            let target_os = ecx.tcx.tcx.sess.target.target.target_os.to_lowercase();
            let ignore_leaks = target_os == "windows" || target_os == "macos";
            if !ignore_leaks && leaks != 0 {
                tcx.sess.err("the evaluated program leaked memory");
            }
        }
        Err(mut e) => {
            // Special treatment for some error kinds
            let msg = match e.kind {
                InterpError::Exit(code) => std::process::exit(code),
                err_unsup!(NoMirFor(..)) =>
                    format!("{}. Did you set `MIRI_SYSROOT` to a Miri-enabled sysroot? You can prepare one with `cargo miri setup`.", e),
                _ => e.to_string()
            };
            e.print_backtrace();
            if let Some(frame) = ecx.stack().last() {
                let block = &frame.body.basic_blocks()[frame.block];
                let span = if frame.stmt < block.statements.len() {
                    block.statements[frame.stmt].source_info.span
                } else {
                    block.terminator().source_info.span
                };

                let msg = format!("Miri evaluation error: {}", msg);
                let mut err = struct_error(ecx.tcx.tcx.at(span), msg.as_str());
                let frames = ecx.generate_stacktrace(None);
                err.span_label(span, msg);
                // We iterate with indices because we need to look at the next frame (the caller).
                for idx in 0..frames.len() {
                    let frame_info = &frames[idx];
                    let call_site_is_local = frames.get(idx+1).map_or(false,
                        |caller_info| caller_info.instance.def_id().is_local());
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
        }
    }
}
