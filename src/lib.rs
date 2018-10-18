#![feature(rustc_private)]

#![cfg_attr(feature = "cargo-clippy", allow(cast_lossless))]

#[macro_use]
extern crate log;

// From rustc.
#[macro_use]
extern crate rustc;
extern crate rustc_data_structures;
extern crate rustc_mir;
extern crate rustc_target;
extern crate syntax;

use std::collections::HashMap;
use std::borrow::Cow;

use rustc::ty::{self, Ty, TyCtxt, query::TyCtxtAt};
use rustc::ty::layout::{TyLayout, LayoutOf, Size};
use rustc::hir::{self, def_id::DefId};
use rustc::mir;

use syntax::attr;


pub use rustc_mir::interpret::*;
pub use rustc_mir::interpret::{self, AllocMap}; // resolve ambiguity

mod fn_call;
mod operator;
mod intrinsic;
mod helpers;
mod tls;
mod range_map;
mod mono_hash_map;
mod stacked_borrows;

use fn_call::EvalContextExt as MissingFnsEvalContextExt;
use operator::EvalContextExt as OperatorEvalContextExt;
use intrinsic::EvalContextExt as IntrinsicEvalContextExt;
use tls::{EvalContextExt as TlsEvalContextExt, TlsData};
use range_map::RangeMap;
#[allow(unused_imports)] // FIXME rustc bug https://github.com/rust-lang/rust/issues/53682
use helpers::{ScalarExt, EvalContextExt as HelpersEvalContextExt};
use mono_hash_map::MonoHashMap;
use stacked_borrows::{EvalContextExt as StackedBorEvalContextExt, Borrow};

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
        let main_ret_ty = main_ret_ty.no_late_bound_regions().unwrap();
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
            start_mir.span,
            start_mir,
            Some(ret_ptr.into()),
            StackPopCleanup::None { cleanup: true },
        )?;

        let mut args = ecx.frame().mir.args_iter();

        // First argument: pointer to main()
        let main_ptr = ecx.memory_mut().create_fn_alloc(main_instance);
        let dest = ecx.eval_place(&mir::Place::Local(args.next().unwrap()))?;
        ecx.write_scalar(Scalar::Ptr(main_ptr), dest)?;

        // Second argument (argc): 1
        let dest = ecx.eval_place(&mir::Place::Local(args.next().unwrap()))?;
        ecx.write_scalar(Scalar::from_int(1, dest.layout.size), dest)?;

        // FIXME: extract main source file path
        // Third argument (argv): &[b"foo"]
        let dest = ecx.eval_place(&mir::Place::Local(args.next().unwrap()))?;
        let foo = ecx.memory.allocate_static_bytes(b"foo\0");
        let foo_ty = ecx.tcx.mk_imm_ptr(ecx.tcx.types.u8);
        let foo_layout = ecx.layout_of(foo_ty)?;
        let foo_place = ecx.allocate(foo_layout, MiriMemoryKind::Env.into())?;
        ecx.write_scalar(Scalar::Ptr(foo), foo_place.into())?;
        ecx.memory.mark_immutable(foo_place.to_ptr()?.alloc_id)?;
        ecx.write_scalar(foo_place.ptr, dest)?;

        assert!(args.next().is_none(), "start lang item has more arguments than expected");
    } else {
        let ret_place = MPlaceTy::dangling(ecx.layout_of(tcx.mk_unit())?, &ecx).into();
        ecx.push_stack_frame(
            main_instance,
            main_mir.span,
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

    let res: EvalResult = (|| {
        ecx.run()?;
        ecx.run_tls_dtors()
    })();

    match res {
        Ok(()) => {
            let leaks = ecx.memory().leak_report();
            // Disable the leak test on some platforms where we likely do not
            // correctly implement TLS destructors.
            let target_os = ecx.tcx.tcx.sess.target.target.target_os.to_lowercase();
            let ignore_leaks = target_os == "windows" || target_os == "macos";
            if !ignore_leaks && leaks != 0 {
                tcx.sess.err("the evaluated program leaked memory");
            }
        }
        Err(e) => {
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
                let (frames, span) = ecx.generate_stacktrace(None);
                err.span_label(span, e);
                for FrameInfo { span, location, .. } in frames {
                    err.span_note(span, &format!("inside call to `{}`", location));
                }
                err.emit();
            } else {
                ecx.tcx.sess.err(&e.to_string());
            }

            /* Nice try, but with MIRI_BACKTRACE this shows 100s of backtraces.
            for (i, frame) in ecx.stack().iter().enumerate() {
                trace!("-------------------");
                trace!("Frame {}", i);
                trace!("    return: {:#?}", frame.return_place);
                for (i, local) in frame.locals.iter().enumerate() {
                    if let Ok(local) = local.access() {
                        trace!("    local {}: {:?}", i, local);
                    }
                }
            }*/
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
        use MiriMemoryKind::*;
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
            stacked_borrows: stacked_borrows::State::new(),
        }
    }
}

#[allow(dead_code)] // FIXME https://github.com/rust-lang/rust/issues/47131
type MiriEvalContext<'a, 'mir, 'tcx> = EvalContext<'a, 'mir, 'tcx, Evaluator<'tcx>>;


impl<'a, 'mir, 'tcx> Machine<'a, 'mir, 'tcx> for Evaluator<'tcx> {
    type MemoryKinds = MiriMemoryKind;

    type AllocExtra = stacked_borrows::Stacks;
    type PointerTag = Borrow;
    const ENABLE_PTR_TRACKING_HOOKS: bool = true;

    type MemoryMap = MonoHashMap<AllocId, (MemoryKind<MiriMemoryKind>, Allocation<Borrow, Self::AllocExtra>)>;

    const STATIC_KIND: Option<MiriMemoryKind> = Some(MiriMemoryKind::MutStatic);

    fn enforce_validity(ecx: &EvalContext<'a, 'mir, 'tcx, Self>) -> bool {
        if !ecx.machine.validate {
            return false;
        }

        // Some functions are whitelisted until we figure out how to fix them.
        // We walk up the stack a few frames to also cover their callees.
        const WHITELIST: &[&str] = &[
            // Uses mem::uninitialized
            "std::ptr::read",
            "std::sys::windows::mutex::Mutex::",
        ];
        for frame in ecx.stack().iter()
            .rev().take(3)
        {
            let name = frame.instance.to_string();
            if WHITELIST.iter().any(|white| name.starts_with(white)) {
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
        let align = layout.align.abi();
        ecx.write_scalar(Scalar::from_uint(align, arg.layout.size), arg)?;

        // No more arguments
        assert!(args.next().is_none(), "exchange_malloc lang item has more arguments than expected");
        Ok(())
    }

    fn find_foreign_static(
        tcx: TyCtxtAt<'a, 'tcx, 'tcx>,
        def_id: DefId,
    ) -> EvalResult<'tcx, Cow<'tcx, Allocation<Borrow, Self::AllocExtra>>> {
        let attrs = tcx.get_attrs(def_id);
        let link_name = match attr::first_attr_value_str_by_name(&attrs, "link_name") {
            Some(name) => name.as_str(),
            None => tcx.item_name(def_id).as_str(),
        };

        let alloc = match &link_name[..] {
            "__cxa_thread_atexit_impl" => {
                // This should be all-zero, pointer-sized
                let data = vec![0; tcx.data_layout.pointer_size.bytes() as usize];
                Allocation::from_bytes(&data[..], tcx.data_layout.pointer_align)
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

    fn static_with_default_tag(
        alloc: &'_ Allocation
    ) -> Cow<'_, Allocation<Borrow, Self::AllocExtra>> {
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
            extra: Self::AllocExtra::default(),
        };
        Cow::Owned(alloc)
    }

    #[inline(always)]
    fn memory_accessed(
        alloc: &Allocation<Borrow, Self::AllocExtra>,
        ptr: Pointer<Borrow>,
        size: Size,
        access: MemoryAccess,
    ) -> EvalResult<'tcx> {
        alloc.extra.memory_accessed(ptr, size, access)
    }

    #[inline(always)]
    fn memory_deallocated(
        alloc: &mut Allocation<Self::PointerTag, Self::AllocExtra>,
        ptr: Pointer<Borrow>,
    ) -> EvalResult<'tcx> {
        alloc.extra.memory_deallocated(ptr)
    }

    #[inline(always)]
    fn tag_reference(
        ecx: &mut EvalContext<'a, 'mir, 'tcx, Self>,
        ptr: Pointer<Borrow>,
        pointee_ty: Ty<'tcx>,
        pointee_size: Size,
        mutability: Option<hir::Mutability>,
    ) -> EvalResult<'tcx, Borrow> {
        if !ecx.machine.validate {
            // No tracking
            Ok(Borrow::default())
        } else {
            ecx.tag_reference(ptr, pointee_ty, pointee_size, mutability)
        }
    }

    #[inline(always)]
    fn tag_dereference(
        ecx: &EvalContext<'a, 'mir, 'tcx, Self>,
        ptr: Pointer<Borrow>,
        pointee_ty: Ty<'tcx>,
        pointee_size: Size,
        mutability: Option<hir::Mutability>,
    ) -> EvalResult<'tcx, Borrow> {
        if !ecx.machine.validate {
            // No tracking
            Ok(Borrow::default())
        } else {
            ecx.tag_dereference(ptr, pointee_ty, pointee_size, mutability)
        }
    }
}
