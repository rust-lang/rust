#![feature(rustc_private)]

#![allow(clippy::cast_lossless)]

#[macro_use]
extern crate log;
// From rustc.
extern crate syntax;
#[macro_use]
extern crate rustc;
extern crate rustc_data_structures;
extern crate rustc_mir;
extern crate rustc_target;

mod fn_call;
mod operator;
mod intrinsic;
mod helpers;
mod tls;
mod range_map;
mod mono_hash_map;
mod stacked_borrows;

use std::collections::HashMap;
use std::borrow::Cow;
use std::rc::Rc;

use rand::rngs::StdRng;
use rand::SeedableRng;

use rustc::ty::{self, TyCtxt, query::TyCtxtAt};
use rustc::ty::layout::{LayoutOf, Size, Align};
use rustc::hir::{self, def_id::DefId};
use rustc::mir;
pub use rustc_mir::interpret::*;
// Resolve ambiguity.
pub use rustc_mir::interpret::{self, AllocMap, PlaceTy};
use syntax::attr;
use syntax::source_map::DUMMY_SP;
use syntax::symbol::sym;

pub use crate::fn_call::EvalContextExt as MissingFnsEvalContextExt;
pub use crate::operator::EvalContextExt as OperatorEvalContextExt;
pub use crate::intrinsic::EvalContextExt as IntrinsicEvalContextExt;
pub use crate::tls::{EvalContextExt as TlsEvalContextExt, TlsData};
use crate::range_map::RangeMap;
#[allow(unused_imports)] // FIXME: rustc bug, issue <https://github.com/rust-lang/rust/issues/53682>.
pub use crate::helpers::{EvalContextExt as HelpersEvalContextExt};
use crate::mono_hash_map::MonoHashMap;
pub use crate::stacked_borrows::{EvalContextExt as StackedBorEvalContextExt};

// Used by priroda.
pub use crate::stacked_borrows::{Tag, Permission, Stack, Stacks, Item};

/// Insert rustc arguments at the beginning of the argument list that Miri wants to be
/// set per default, for maximal validation power.
pub fn miri_default_args() -> &'static [&'static str] {
    // The flags here should be kept in sync with what bootstrap adds when `test-miri` is
    // set, which happens in `bootstrap/bin/rustc.rs` in the rustc sources.
    &["-Zalways-encode-mir", "-Zmir-emit-retag", "-Zmir-opt-level=0", "--cfg=miri"]
}

/// Configuration needed to spawn a Miri instance.
#[derive(Clone)]
pub struct MiriConfig {
    pub validate: bool,
    pub args: Vec<String>,

    // The seed to use when non-determinism is required (e.g. getrandom())
    pub seed: Option<u64>
}

// Used by priroda.
pub fn create_ecx<'a, 'mir: 'a, 'tcx: 'mir>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    main_id: DefId,
    config: MiriConfig,
) -> EvalResult<'tcx, InterpretCx<'a, 'mir, 'tcx, Evaluator<'tcx>>> {
    let mut ecx = InterpretCx::new(
        tcx.at(syntax::source_map::DUMMY_SP),
        ty::ParamEnv::reveal_all(),
        Evaluator::new(config.validate, config.seed),
    );

    let main_instance = ty::Instance::mono(ecx.tcx.tcx, main_id);
    let main_mir = ecx.load_mir(main_instance.def)?;

    if !main_mir.return_ty().is_unit() || main_mir.arg_count != 0 {
        return err!(Unimplemented(
            "miri does not support main functions without `fn()` type signatures"
                .to_owned(),
        ));
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
    let start_mir = ecx.load_mir(start_instance.def)?;

    if start_mir.arg_count != 3 {
        return err!(AbiViolation(format!(
            "'start' lang item should have three arguments, but has {}",
            start_mir.arg_count
        )));
    }

    // Return value (in static memory so that it does not count as leak).
    let ret = ecx.layout_of(start_mir.return_ty())?;
    let ret_ptr = ecx.allocate(ret, MiriMemoryKind::MutStatic.into());

    // Push our stack frame.
    ecx.push_stack_frame(
        start_instance,
        // There is no call site.
        DUMMY_SP,
        start_mir,
        Some(ret_ptr.into()),
        StackPopCleanup::None { cleanup: true },
    )?;

    let mut args = ecx.frame().mir.args_iter();

    // First argument: pointer to `main()`.
    let main_ptr = ecx.memory_mut().create_fn_alloc(main_instance).with_default_tag();
    let dest = ecx.eval_place(&mir::Place::Base(mir::PlaceBase::Local(args.next().unwrap())))?;
    ecx.write_scalar(Scalar::Ptr(main_ptr), dest)?;

    // Second argument (argc): `1`.
    let dest = ecx.eval_place(&mir::Place::Base(mir::PlaceBase::Local(args.next().unwrap())))?;
    let argc = Scalar::from_uint(config.args.len() as u128, dest.layout.size);
    ecx.write_scalar(argc, dest)?;
    // Store argc for macOS's `_NSGetArgc`.
    {
        let argc_place = ecx.allocate(dest.layout, MiriMemoryKind::Env.into());
        ecx.write_scalar(argc, argc_place.into())?;
        ecx.machine.argc = Some(argc_place.ptr.to_ptr()?);
    }

    // FIXME: extract main source file path.
    // Third argument (`argv`): created from `config.args`.
    let dest = ecx.eval_place(&mir::Place::Base(mir::PlaceBase::Local(args.next().unwrap())))?;
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
        argvs.push(ecx.memory_mut().allocate_static_bytes(arg.as_slice()).with_default_tag());
    }
    // Make an array with all these pointers, in the Miri memory.
    let argvs_layout = ecx.layout_of(ecx.tcx.mk_array(ecx.tcx.mk_imm_ptr(ecx.tcx.types.u8), argvs.len() as u64))?;
    let argvs_place = ecx.allocate(argvs_layout, MiriMemoryKind::Env.into());
    for (idx, arg) in argvs.into_iter().enumerate() {
        let place = ecx.mplace_field(argvs_place, idx as u64)?;
        ecx.write_scalar(Scalar::Ptr(arg), place.into())?;
    }
    ecx.memory_mut().mark_immutable(argvs_place.to_ptr()?.alloc_id)?;
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

pub fn eval_main<'a, 'tcx: 'a>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
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
    let res: EvalResult = (|| {
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
                InterpError::NoMirFor(..) =>
                    format!("{}. Did you set `MIRI_SYSROOT` to a Miri-enabled sysroot? You can prepare one with `cargo miri setup`.", e),
                _ => e.to_string()
            };
            e.print_backtrace();
            if let Some(frame) = ecx.stack().last() {
                let block = &frame.mir.basic_blocks()[frame.block];
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

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum MiriMemoryKind {
    /// `__rust_alloc` memory.
    Rust,
    /// `malloc` memory.
    C,
    /// Part of env var emulation.
    Env,
    /// Mutable statics.
    MutStatic,
}

impl Into<MemoryKind<MiriMemoryKind>> for MiriMemoryKind {
    #[inline(always)]
    fn into(self) -> MemoryKind<MiriMemoryKind> {
        MemoryKind::Machine(self)
    }
}

impl MayLeak for MiriMemoryKind {
    #[inline(always)]
    fn may_leak(self) -> bool {
        use self::MiriMemoryKind::*;
        match self {
            Rust | C => false,
            Env | MutStatic => true,
        }
    }
}

pub struct Evaluator<'tcx> {
    /// Environment variables set by `setenv`.
    /// Miri does not expose env vars from the host to the emulated program.
    pub(crate) env_vars: HashMap<Vec<u8>, Pointer<Tag>>,

    /// Program arguments (`Option` because we can only initialize them after creating the ecx).
    /// These are *pointers* to argc/argv because macOS.
    /// We also need the full command line as one string because of Windows.
    pub(crate) argc: Option<Pointer<Tag>>,
    pub(crate) argv: Option<Pointer<Tag>>,
    pub(crate) cmd_line: Option<Pointer<Tag>>,

    /// Last OS error.
    pub(crate) last_error: u32,

    /// TLS state.
    pub(crate) tls: TlsData<'tcx>,

    /// Whether to enforce the validity invariant.
    pub(crate) validate: bool,

    /// The random number generator to use if Miri
    /// is running in non-deterministic mode
    pub(crate) rng: Option<StdRng>
}

impl<'tcx> Evaluator<'tcx> {
    fn new(validate: bool, seed: Option<u64>) -> Self {
        Evaluator {
            env_vars: HashMap::default(),
            argc: None,
            argv: None,
            cmd_line: None,
            last_error: 0,
            tls: TlsData::default(),
            validate,
            rng: seed.map(|s| StdRng::seed_from_u64(s))
        }
    }
}

// FIXME: rustc issue <https://github.com/rust-lang/rust/issues/47131>.
#[allow(dead_code)]
type MiriEvalContext<'a, 'mir, 'tcx> = InterpretCx<'a, 'mir, 'tcx, Evaluator<'tcx>>;

// A little trait that's useful to be inherited by extension traits.
pub trait MiriEvalContextExt<'a, 'mir, 'tcx> {
    fn eval_context_ref(&self) -> &MiriEvalContext<'a, 'mir, 'tcx>;
    fn eval_context_mut(&mut self) -> &mut MiriEvalContext<'a, 'mir, 'tcx>;
}
impl<'a, 'mir, 'tcx> MiriEvalContextExt<'a, 'mir, 'tcx> for MiriEvalContext<'a, 'mir, 'tcx> {
    #[inline(always)]
    fn eval_context_ref(&self) -> &MiriEvalContext<'a, 'mir, 'tcx> {
        self
    }
    #[inline(always)]
    fn eval_context_mut(&mut self) -> &mut MiriEvalContext<'a, 'mir, 'tcx> {
        self
    }
}

impl<'a, 'mir, 'tcx> Machine<'a, 'mir, 'tcx> for Evaluator<'tcx> {
    type MemoryKinds = MiriMemoryKind;

    type FrameExtra = stacked_borrows::CallId;
    type MemoryExtra = stacked_borrows::MemoryState;
    type AllocExtra = stacked_borrows::Stacks;
    type PointerTag = Tag;

    type MemoryMap = MonoHashMap<AllocId, (MemoryKind<MiriMemoryKind>, Allocation<Tag, Self::AllocExtra>)>;

    const STATIC_KIND: Option<MiriMemoryKind> = Some(MiriMemoryKind::MutStatic);

    #[inline(always)]
    fn enforce_validity(ecx: &InterpretCx<'a, 'mir, 'tcx, Self>) -> bool {
        ecx.machine.validate
    }

    /// Returns `Ok()` when the function was handled; fail otherwise.
    #[inline(always)]
    fn find_fn(
        ecx: &mut InterpretCx<'a, 'mir, 'tcx, Self>,
        instance: ty::Instance<'tcx>,
        args: &[OpTy<'tcx, Tag>],
        dest: Option<PlaceTy<'tcx, Tag>>,
        ret: Option<mir::BasicBlock>,
    ) -> EvalResult<'tcx, Option<&'mir mir::Mir<'tcx>>> {
        ecx.find_fn(instance, args, dest, ret)
    }

    #[inline(always)]
    fn call_intrinsic(
        ecx: &mut rustc_mir::interpret::InterpretCx<'a, 'mir, 'tcx, Self>,
        instance: ty::Instance<'tcx>,
        args: &[OpTy<'tcx, Tag>],
        dest: PlaceTy<'tcx, Tag>,
    ) -> EvalResult<'tcx> {
        ecx.call_intrinsic(instance, args, dest)
    }

    #[inline(always)]
    fn ptr_op(
        ecx: &rustc_mir::interpret::InterpretCx<'a, 'mir, 'tcx, Self>,
        bin_op: mir::BinOp,
        left: ImmTy<'tcx, Tag>,
        right: ImmTy<'tcx, Tag>,
    ) -> EvalResult<'tcx, (Scalar<Tag>, bool)> {
        ecx.ptr_op(bin_op, left, right)
    }

    fn box_alloc(
        ecx: &mut InterpretCx<'a, 'mir, 'tcx, Self>,
        dest: PlaceTy<'tcx, Tag>,
    ) -> EvalResult<'tcx> {
        trace!("box_alloc for {:?}", dest.layout.ty);
        // Call the `exchange_malloc` lang item.
        let malloc = ecx.tcx.lang_items().exchange_malloc_fn().unwrap();
        let malloc = ty::Instance::mono(ecx.tcx.tcx, malloc);
        let malloc_mir = ecx.load_mir(malloc.def)?;
        ecx.push_stack_frame(
            malloc,
            malloc_mir.span,
            malloc_mir,
            Some(dest),
            // Don't do anything when we are done. The `statement()` function will increment
            // the old stack frame's stmt counter to the next statement, which means that when
            // `exchange_malloc` returns, we go on evaluating exactly where we want to be.
            StackPopCleanup::None { cleanup: true },
        )?;

        let mut args = ecx.frame().mir.args_iter();
        let layout = ecx.layout_of(dest.layout.ty.builtin_deref(false).unwrap().ty)?;

        // First argument: `size`.
        // (`0` is allowed here -- this is expected to be handled by the lang item).
        let arg = ecx.eval_place(&mir::Place::Base(mir::PlaceBase::Local(args.next().unwrap())))?;
        let size = layout.size.bytes();
        ecx.write_scalar(Scalar::from_uint(size, arg.layout.size), arg)?;

        // Second argument: `align`.
        let arg = ecx.eval_place(&mir::Place::Base(mir::PlaceBase::Local(args.next().unwrap())))?;
        let align = layout.align.abi.bytes();
        ecx.write_scalar(Scalar::from_uint(align, arg.layout.size), arg)?;

        // No more arguments.
        assert!(
            args.next().is_none(),
            "`exchange_malloc` lang item has more arguments than expected"
        );
        Ok(())
    }

    fn find_foreign_static(
        def_id: DefId,
        tcx: TyCtxtAt<'a, 'tcx, 'tcx>,
        memory_extra: &Self::MemoryExtra,
    ) -> EvalResult<'tcx, Cow<'tcx, Allocation<Tag, Self::AllocExtra>>> {
        let attrs = tcx.get_attrs(def_id);
        let link_name = match attr::first_attr_value_str_by_name(&attrs, sym::link_name) {
            Some(name) => name.as_str(),
            None => tcx.item_name(def_id).as_str(),
        };

        let alloc = match link_name.get() {
            "__cxa_thread_atexit_impl" => {
                // This should be all-zero, pointer-sized.
                let size = tcx.data_layout.pointer_size;
                let data = vec![0; size.bytes() as usize];
                let extra = Stacks::new(size, Tag::default(), Rc::clone(memory_extra));
                Allocation::from_bytes(&data, tcx.data_layout.pointer_align.abi, extra)
            }
            _ => return err!(Unimplemented(
                    format!("can't access foreign static: {}", link_name),
                )),
        };
        Ok(Cow::Owned(alloc))
    }

    #[inline(always)]
    fn before_terminator(_ecx: &mut InterpretCx<'a, 'mir, 'tcx, Self>) -> EvalResult<'tcx>
    {
        // We are not interested in detecting loops.
        Ok(())
    }

    fn adjust_static_allocation<'b>(
        alloc: &'b Allocation,
        memory_extra: &Self::MemoryExtra,
    ) -> Cow<'b, Allocation<Tag, Self::AllocExtra>> {
        let extra = Stacks::new(
            Size::from_bytes(alloc.bytes.len() as u64),
            Tag::default(),
            Rc::clone(memory_extra),
        );
        let alloc: Allocation<Tag, Self::AllocExtra> = Allocation {
            bytes: alloc.bytes.clone(),
            relocations: Relocations::from_presorted(
                alloc.relocations.iter()
                    .map(|&(offset, ((), alloc))| (offset, (Tag::default(), alloc)))
                    .collect()
            ),
            undef_mask: alloc.undef_mask.clone(),
            align: alloc.align,
            mutability: alloc.mutability,
            extra,
        };
        Cow::Owned(alloc)
    }

    #[inline(always)]
    fn new_allocation(
        size: Size,
        extra: &Self::MemoryExtra,
        kind: MemoryKind<MiriMemoryKind>,
    ) -> (Self::AllocExtra, Self::PointerTag) {
        Stacks::new_allocation(size, extra, kind)
    }

    #[inline(always)]
    fn tag_dereference(
        _ecx: &InterpretCx<'a, 'mir, 'tcx, Self>,
        place: MPlaceTy<'tcx, Tag>,
        _mutability: Option<hir::Mutability>,
    ) -> EvalResult<'tcx, Scalar<Tag>> {
        // Nothing happens.
        Ok(place.ptr)
    }

    #[inline(always)]
    fn retag(
        ecx: &mut InterpretCx<'a, 'mir, 'tcx, Self>,
        kind: mir::RetagKind,
        place: PlaceTy<'tcx, Tag>,
    ) -> EvalResult<'tcx> {
        if !ecx.tcx.sess.opts.debugging_opts.mir_emit_retag || !Self::enforce_validity(ecx) {
            // No tracking, or no retagging. The latter is possible because a dependency of ours
            // might be called with different flags than we are, so there are `Retag`
            // statements but we do not want to execute them.
            // Also, honor the whitelist in `enforce_validity` because otherwise we might retag
            // uninitialized data.
             Ok(())
        } else {
            ecx.retag(kind, place)
        }
    }

    #[inline(always)]
    fn stack_push(
        ecx: &mut InterpretCx<'a, 'mir, 'tcx, Self>,
    ) -> EvalResult<'tcx, stacked_borrows::CallId> {
        Ok(ecx.memory().extra.borrow_mut().new_call())
    }

    #[inline(always)]
    fn stack_pop(
        ecx: &mut InterpretCx<'a, 'mir, 'tcx, Self>,
        extra: stacked_borrows::CallId,
    ) -> EvalResult<'tcx> {
        Ok(ecx.memory().extra.borrow_mut().end_call(extra))
    }
}
