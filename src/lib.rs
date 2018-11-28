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

use std::collections::HashMap;
use std::borrow::Cow;
use std::env;

use rustc::ty::{self, TyCtxt, query::TyCtxtAt};
use rustc::ty::layout::{TyLayout, LayoutOf, Size};
use rustc::hir::{self, def_id::DefId};
use rustc::mir;

use syntax::attr;
use syntax::source_map::DUMMY_SP;

pub use rustc_mir::interpret::*;
pub use rustc_mir::interpret::{self, AllocMap, PlaceTy}; // resolve ambiguity

mod fn_call;
mod operator;
mod intrinsic;
mod helpers;
mod tls;
mod range_map;
mod mono_hash_map;
mod stacked_borrows;

pub use crate::fn_call::EvalContextExt as MissingFnsEvalContextExt;
pub use crate::operator::EvalContextExt as OperatorEvalContextExt;
pub use crate::intrinsic::EvalContextExt as IntrinsicEvalContextExt;
pub use crate::tls::{EvalContextExt as TlsEvalContextExt, TlsData};
use crate::range_map::RangeMap;
#[allow(unused_imports)] // FIXME rustc bug https://github.com/rust-lang/rust/issues/53682
pub use crate::helpers::{ScalarExt, EvalContextExt as HelpersEvalContextExt};
use crate::mono_hash_map::MonoHashMap;
pub use crate::stacked_borrows::{EvalContextExt as StackedBorEvalContextExt};

// Used by priroda
pub use crate::stacked_borrows::{Borrow, Stack, Stacks, BorStackItem};

/// Insert rustc arguments at the beginning of the argument list that miri wants to be
/// set per default, for maximal validation power.
pub fn miri_default_args() -> &'static [&'static str] {
    // The flags here should be kept in sync with what bootstrap adds when `test-miri` is
    // set, which happens in `bootstrap/bin/rustc.rs` in the rustc sources.
    &["-Zalways-encode-mir", "-Zmir-emit-retag", "-Zmir-opt-level=0"]
}

// Used by priroda
pub fn create_ecx<'a, 'mir: 'a, 'tcx: 'mir>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    main_id: DefId,
    validate: bool,
) -> EvalResult<'tcx, EvalContext<'a, 'mir, 'tcx, Evaluator<'tcx>>> {
    let mut ecx = EvalContext::new(
        tcx.at(syntax::source_map::DUMMY_SP),
        ty::ParamEnv::reveal_all(),
        Evaluator::new(validate),
    );

    let main_instance = ty::Instance::mono(ecx.tcx.tcx, main_id);
    let main_mir = ecx.load_mir(main_instance.def)?;

    if !main_mir.return_ty().is_unit() || main_mir.arg_count != 0 {
        return err!(Unimplemented(
            "miri does not support main functions without `fn()` type signatures"
                .to_owned(),
        ));
    }

    let libstd_has_mir = {
        let rustc_panic = ecx.resolve_path(&["std", "panicking", "rust_panic"])?;
        ecx.load_mir(rustc_panic.def).is_ok()
    };

    if libstd_has_mir {
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

        // Return value (in static memory so that it does not count as leak)
        let ret = ecx.layout_of(start_mir.return_ty())?;
        let ret_ptr = ecx.allocate(ret, MiriMemoryKind::MutStatic.into())?;

        // Push our stack frame
        ecx.push_stack_frame(
            start_instance,
            DUMMY_SP, // there is no call site, we want no span
            start_mir,
            Some(ret_ptr.into()),
            StackPopCleanup::None { cleanup: true },
        )?;

        let mut args = ecx.frame().mir.args_iter();

        // First argument: pointer to main()
        let main_ptr = ecx.memory_mut().create_fn_alloc(main_instance).with_default_tag();
        let dest = ecx.eval_place(&mir::Place::Local(args.next().unwrap()))?;
        ecx.write_scalar(Scalar::Ptr(main_ptr), dest)?;

        // Second argument (argc): 1
        let dest = ecx.eval_place(&mir::Place::Local(args.next().unwrap()))?;
        ecx.write_scalar(Scalar::from_int(1, dest.layout.size), dest)?;

        // FIXME: extract main source file path
        // Third argument (argv): &[b"foo"]
        let dest = ecx.eval_place(&mir::Place::Local(args.next().unwrap()))?;
        let foo = ecx.memory_mut().allocate_static_bytes(b"foo\0").with_default_tag();
        let foo_ty = ecx.tcx.mk_imm_ptr(ecx.tcx.types.u8);
        let foo_layout = ecx.layout_of(foo_ty)?;
        let foo_place = ecx.allocate(foo_layout, MiriMemoryKind::Env.into())?;
        ecx.write_scalar(Scalar::Ptr(foo), foo_place.into())?;
        ecx.memory_mut().mark_immutable(foo_place.to_ptr()?.alloc_id)?;
        ecx.write_scalar(foo_place.ptr, dest)?;

        assert!(args.next().is_none(), "start lang item has more arguments than expected");
    } else {
        let ret_place = MPlaceTy::dangling(ecx.layout_of(tcx.mk_unit())?, &ecx).into();
        ecx.push_stack_frame(
            main_instance,
            DUMMY_SP, // there is no call site, we want no span
            main_mir,
            Some(ret_place),
            StackPopCleanup::None { cleanup: true },
        )?;

        // No arguments
        let mut args = ecx.frame().mir.args_iter();
        assert!(args.next().is_none(), "main function must not have arguments");
    }

    Ok(ecx)
}

pub fn eval_main<'a, 'tcx: 'a>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    main_id: DefId,
    validate: bool,
) {
    let mut ecx = create_ecx(tcx, main_id, validate).expect("Couldn't create ecx");

    // If MIRI_BACKTRACE is set and RUST_CTFE_BACKTRACE is not, set RUST_CTFE_BACKTRACE.
    // Do this late, so we really only apply this to miri's errors.
    if let Ok(var) = env::var("MIRI_BACKTRACE") {
        if env::var("RUST_CTFE_BACKTRACE") == Err(env::VarError::NotPresent) {
            env::set_var("RUST_CTFE_BACKTRACE", &var);
        }
    }

    // Run! The main execution.
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
            e.print_backtrace();
            if let Some(frame) = ecx.stack().last() {
                let block = &frame.mir.basic_blocks()[frame.block];
                let span = if frame.stmt < block.statements.len() {
                    block.statements[frame.stmt].source_info.span
                } else {
                    block.terminator().source_info.span
                };

                let e = e.to_string();
                let msg = format!("constant evaluation error: {}", e);
                let mut err = struct_error(ecx.tcx.tcx.at(span), msg.as_str());
                let frames = ecx.generate_stacktrace(None);
                err.span_label(span, e);
                // we iterate with indices because we need to look at the next frame (the caller)
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
                ecx.tcx.sess.err(&e.to_string());
            }

            for (i, frame) in ecx.stack().iter().enumerate() {
                trace!("-------------------");
                trace!("Frame {}", i);
                trace!("    return: {:#?}", frame.return_place);
                for (i, local) in frame.locals.iter().enumerate() {
                    if let Ok(local) = local.access() {
                        trace!("    local {}: {:?}", i, local);
                    }
                }
            }
        }
    }
}


#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum MiriMemoryKind {
    /// `__rust_alloc` memory
    Rust,
    /// `malloc` memory
    C,
    /// Part of env var emulation
    Env,
    /// mutable statics
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
    /// Environment variables set by `setenv`
    /// Miri does not expose env vars from the host to the emulated program
    pub(crate) env_vars: HashMap<Vec<u8>, Pointer<Borrow>>,

    /// TLS state
    pub(crate) tls: TlsData<'tcx>,

    /// Whether to enforce the validity invariant
    pub(crate) validate: bool,

    /// Stacked Borrows state
    pub(crate) stacked_borrows: stacked_borrows::State,
}

impl<'tcx> Evaluator<'tcx> {
    fn new(validate: bool) -> Self {
        Evaluator {
            env_vars: HashMap::default(),
            tls: TlsData::default(),
            validate,
            stacked_borrows: stacked_borrows::State::default(),
        }
    }
}

#[allow(dead_code)] // FIXME https://github.com/rust-lang/rust/issues/47131
type MiriEvalContext<'a, 'mir, 'tcx> = EvalContext<'a, 'mir, 'tcx, Evaluator<'tcx>>;


impl<'a, 'mir, 'tcx> Machine<'a, 'mir, 'tcx> for Evaluator<'tcx> {
    type MemoryKinds = MiriMemoryKind;

    type FrameExtra = stacked_borrows::CallId;
    type MemoryExtra = stacked_borrows::MemoryState;
    type AllocExtra = stacked_borrows::Stacks;
    type PointerTag = Borrow;

    type MemoryMap = MonoHashMap<AllocId, (MemoryKind<MiriMemoryKind>, Allocation<Borrow, Self::AllocExtra>)>;

    const STATIC_KIND: Option<MiriMemoryKind> = Some(MiriMemoryKind::MutStatic);

    fn enforce_validity(ecx: &EvalContext<'a, 'mir, 'tcx, Self>) -> bool {
        if !ecx.machine.validate {
            return false;
        }

        // Some functions are whitelisted until we figure out how to fix them.
        // We walk up the stack a few frames to also cover their callees.
        const WHITELIST: &[(&str, &str)] = &[
            // Uses mem::uninitialized
            ("std::sys::windows::mutex::Mutex::", ""),
        ];
        for frame in ecx.stack().iter()
            .rev().take(3)
        {
            let name = frame.instance.to_string();
            if WHITELIST.iter().any(|(prefix, suffix)| name.starts_with(prefix) && name.ends_with(suffix)) {
                return false;
            }
        }
        true
    }

    /// Returns Ok() when the function was handled, fail otherwise
    #[inline(always)]
    fn find_fn(
        ecx: &mut EvalContext<'a, 'mir, 'tcx, Self>,
        instance: ty::Instance<'tcx>,
        args: &[OpTy<'tcx, Borrow>],
        dest: Option<PlaceTy<'tcx, Borrow>>,
        ret: Option<mir::BasicBlock>,
    ) -> EvalResult<'tcx, Option<&'mir mir::Mir<'tcx>>> {
        ecx.find_fn(instance, args, dest, ret)
    }

    #[inline(always)]
    fn call_intrinsic(
        ecx: &mut rustc_mir::interpret::EvalContext<'a, 'mir, 'tcx, Self>,
        instance: ty::Instance<'tcx>,
        args: &[OpTy<'tcx, Borrow>],
        dest: PlaceTy<'tcx, Borrow>,
    ) -> EvalResult<'tcx> {
        ecx.call_intrinsic(instance, args, dest)
    }

    #[inline(always)]
    fn ptr_op(
        ecx: &rustc_mir::interpret::EvalContext<'a, 'mir, 'tcx, Self>,
        bin_op: mir::BinOp,
        left: Scalar<Borrow>,
        left_layout: TyLayout<'tcx>,
        right: Scalar<Borrow>,
        right_layout: TyLayout<'tcx>,
    ) -> EvalResult<'tcx, (Scalar<Borrow>, bool)> {
        ecx.ptr_op(bin_op, left, left_layout, right, right_layout)
    }

    fn box_alloc(
        ecx: &mut EvalContext<'a, 'mir, 'tcx, Self>,
        dest: PlaceTy<'tcx, Borrow>,
    ) -> EvalResult<'tcx> {
        trace!("box_alloc for {:?}", dest.layout.ty);
        // Call the `exchange_malloc` lang item
        let malloc = ecx.tcx.lang_items().exchange_malloc_fn().unwrap();
        let malloc = ty::Instance::mono(ecx.tcx.tcx, malloc);
        let malloc_mir = ecx.load_mir(malloc.def)?;
        ecx.push_stack_frame(
            malloc,
            malloc_mir.span,
            malloc_mir,
            Some(dest),
            // Don't do anything when we are done.  The statement() function will increment
            // the old stack frame's stmt counter to the next statement, which means that when
            // exchange_malloc returns, we go on evaluating exactly where we want to be.
            StackPopCleanup::None { cleanup: true },
        )?;

        let mut args = ecx.frame().mir.args_iter();
        let layout = ecx.layout_of(dest.layout.ty.builtin_deref(false).unwrap().ty)?;

        // First argument: size
        // (0 is allowed here, this is expected to be handled by the lang item)
        let arg = ecx.eval_place(&mir::Place::Local(args.next().unwrap()))?;
        let size = layout.size.bytes();
        ecx.write_scalar(Scalar::from_uint(size, arg.layout.size), arg)?;

        // Second argument: align
        let arg = ecx.eval_place(&mir::Place::Local(args.next().unwrap()))?;
        let align = layout.align.abi.bytes();
        ecx.write_scalar(Scalar::from_uint(align, arg.layout.size), arg)?;

        // No more arguments
        assert!(args.next().is_none(), "exchange_malloc lang item has more arguments than expected");
        Ok(())
    }

    fn find_foreign_static(
        def_id: DefId,
        tcx: TyCtxtAt<'a, 'tcx, 'tcx>,
        memory_extra: &Self::MemoryExtra,
    ) -> EvalResult<'tcx, Cow<'tcx, Allocation<Borrow, Self::AllocExtra>>> {
        let attrs = tcx.get_attrs(def_id);
        let link_name = match attr::first_attr_value_str_by_name(&attrs, "link_name") {
            Some(name) => name.as_str(),
            None => tcx.item_name(def_id).as_str(),
        };

        let alloc = match &link_name[..] {
            "__cxa_thread_atexit_impl" => {
                // This should be all-zero, pointer-sized
                let size = tcx.data_layout.pointer_size;
                let data = vec![0; size.bytes() as usize];
                let extra = AllocationExtra::memory_allocated(size, memory_extra);
                Allocation::from_bytes(&data[..], tcx.data_layout.pointer_align.abi, extra)
            }
            _ => return err!(Unimplemented(
                    format!("can't access foreign static: {}", link_name),
                )),
        };
        Ok(Cow::Owned(alloc))
    }

    #[inline(always)]
    fn before_terminator(_ecx: &mut EvalContext<'a, 'mir, 'tcx, Self>) -> EvalResult<'tcx>
    {
        // We are not interested in detecting loops
        Ok(())
    }

    fn adjust_static_allocation<'b>(
        alloc: &'b Allocation,
        memory_extra: &Self::MemoryExtra,
    ) -> Cow<'b, Allocation<Borrow, Self::AllocExtra>> {
        let extra = AllocationExtra::memory_allocated(
            Size::from_bytes(alloc.bytes.len() as u64),
            memory_extra,
        );
        let alloc: Allocation<Borrow, Self::AllocExtra> = Allocation {
            bytes: alloc.bytes.clone(),
            relocations: Relocations::from_presorted(
                alloc.relocations.iter()
                    .map(|&(offset, ((), alloc))| (offset, (Borrow::default(), alloc)))
                    .collect()
            ),
            undef_mask: alloc.undef_mask.clone(),
            align: alloc.align,
            mutability: alloc.mutability,
            extra,
        };
        Cow::Owned(alloc)
    }

    fn tag_dereference(
        ecx: &EvalContext<'a, 'mir, 'tcx, Self>,
        place: MPlaceTy<'tcx, Borrow>,
        mutability: Option<hir::Mutability>,
    ) -> EvalResult<'tcx, Scalar<Borrow>> {
        let size = ecx.size_and_align_of_mplace(place)?.map(|(size, _)| size)
            // for extern types, just cover what we can
            .unwrap_or_else(|| place.layout.size);
        if !ecx.tcx.sess.opts.debugging_opts.mir_emit_retag ||
            !Self::enforce_validity(ecx) || size == Size::ZERO
        {
            // No tracking
            Ok(place.ptr)
        } else {
            ecx.ptr_dereference(place, size, mutability.into())?;
            // We never change the pointer
            Ok(place.ptr)
        }
    }

    #[inline(always)]
    fn tag_new_allocation(
        ecx: &mut EvalContext<'a, 'mir, 'tcx, Self>,
        ptr: Pointer,
        kind: MemoryKind<Self::MemoryKinds>,
    ) -> EvalResult<'tcx, Pointer<Borrow>> {
        if !ecx.machine.validate {
            // No tracking
            Ok(ptr.with_default_tag())
        } else {
            let tag = ecx.tag_new_allocation(ptr.alloc_id, kind);
            Ok(Pointer::new_with_tag(ptr.alloc_id, ptr.offset, tag))
        }
    }

    #[inline]
    fn escape_to_raw(
        ecx: &mut EvalContext<'a, 'mir, 'tcx, Self>,
        ptr: OpTy<'tcx, Self::PointerTag>,
    ) -> EvalResult<'tcx> {
        // It is tempting to check the type here, but drop glue does EscapeToRaw
        // on a raw pointer.
        // This is deliberately NOT `deref_operand` as we do not want `tag_dereference`
        // to be called!  That would kill the original tag if we got a raw ptr.
        let place = ecx.ref_to_mplace(ecx.read_immediate(ptr)?)?;
        let size = ecx.size_and_align_of_mplace(place)?.map(|(size, _)| size)
            // for extern types, just cover what we can
            .unwrap_or_else(|| place.layout.size);
        if !ecx.tcx.sess.opts.debugging_opts.mir_emit_retag ||
            !ecx.machine.validate || size == Size::ZERO
        {
            // No tracking, or no retagging. The latter is possible because a dependency of ours
            // might be called with different flags than we are, so there are `Retag`
            // statements but we do not want to execute them.
            Ok(())
        } else {
            ecx.escape_to_raw(place, size)
        }
    }

    #[inline(always)]
    fn retag(
        ecx: &mut EvalContext<'a, 'mir, 'tcx, Self>,
        fn_entry: bool,
        place: PlaceTy<'tcx, Borrow>,
    ) -> EvalResult<'tcx> {
        if !ecx.tcx.sess.opts.debugging_opts.mir_emit_retag || !Self::enforce_validity(ecx) {
            // No tracking, or no retagging. The latter is possible because a dependency of ours
            // might be called with different flags than we are, so there are `Retag`
            // statements but we do not want to execute them.
            // Also, honor the whitelist in `enforce_validity` because otherwise we might retag
            // uninitialized data.
             Ok(())
        } else {
            ecx.retag(fn_entry, place)
        }
    }

    #[inline(always)]
    fn stack_push(
        ecx: &mut EvalContext<'a, 'mir, 'tcx, Self>,
    ) -> EvalResult<'tcx, stacked_borrows::CallId> {
        Ok(ecx.memory().extra.borrow_mut().new_call())
    }

    #[inline(always)]
    fn stack_pop(
        ecx: &mut EvalContext<'a, 'mir, 'tcx, Self>,
        extra: stacked_borrows::CallId,
    ) -> EvalResult<'tcx> {
        Ok(ecx.memory().extra.borrow_mut().end_call(extra))
    }
}
