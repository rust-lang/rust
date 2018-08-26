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

use rustc::ty::{self, TyCtxt, query::TyCtxtAt};
use rustc::ty::layout::{TyLayout, LayoutOf, Size};
use rustc::hir::def_id::DefId;
use rustc::mir;

use rustc_data_structures::fx::FxHasher;

use syntax::ast::Mutability;
use syntax::attr;

use std::marker::PhantomData;
use std::collections::{HashMap, BTreeMap};
use std::hash::{Hash, Hasher};

pub use rustc::mir::interpret::*;
pub use rustc_mir::interpret::*;

mod fn_call;
mod operator;
mod intrinsic;
mod helpers;
mod memory;
mod tls;
mod locks;
mod range_map;

use fn_call::EvalContextExt as MissingFnsEvalContextExt;
use operator::EvalContextExt as OperatorEvalContextExt;
use intrinsic::EvalContextExt as IntrinsicEvalContextExt;
use tls::EvalContextExt as TlsEvalContextExt;
use memory::MemoryKind as MiriMemoryKind;
use locks::LockInfo;
use range_map::RangeMap;
use helpers::{ScalarExt, FalibleScalarExt};

pub fn create_ecx<'a, 'mir: 'a, 'tcx: 'mir>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    main_id: DefId,
    start_wrapper: Option<DefId>,
) -> EvalResult<'tcx, EvalContext<'a, 'mir, 'tcx, Evaluator<'tcx>>> {
    let mut ecx = EvalContext::new(
        tcx.at(syntax::source_map::DUMMY_SP),
        ty::ParamEnv::reveal_all(),
        Default::default(),
        MemoryData::new()
    );

    let main_instance = ty::Instance::mono(ecx.tcx.tcx, main_id);
    let main_mir = ecx.load_mir(main_instance.def)?;

    if !main_mir.return_ty().is_nil() || main_mir.arg_count != 0 {
        return err!(Unimplemented(
            "miri does not support main functions without `fn()` type signatures"
                .to_owned(),
        ));
    }
    let ptr_size = ecx.memory.pointer_size();

    if let Some(start_id) = start_wrapper {
        let main_ret_ty = ecx.tcx.fn_sig(main_id).output();
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
        let size = ecx.tcx.data_layout.pointer_size;
        let align = ecx.tcx.data_layout.pointer_align;
        let ret_ptr = ecx.memory_mut().allocate(size, align, MiriMemoryKind::MutStatic.into())?;

        // Push our stack frame
        ecx.push_stack_frame(
            start_instance,
            start_mir.span,
            start_mir,
            Place::from_ptr(ret_ptr, align),
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
        let foo_place = ecx.allocate(foo_layout, MemoryKind::Stack)?; // will be interned in just a second
        ecx.write_scalar(Scalar::Ptr(foo), foo_place.into())?;
        ecx.memory.intern_static(foo_place.to_ptr()?.alloc_id, Mutability::Immutable)?;
        ecx.write_scalar(foo_place.ptr, dest)?;

        assert!(args.next().is_none(), "start lang item has more arguments than expected");
    } else {
        ecx.push_stack_frame(
            main_instance,
            main_mir.span,
            main_mir,
            Place::from_scalar_ptr(Scalar::from_int(1, ptr_size).into(), ty::layout::Align::from_bytes(1, 1).unwrap()),
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
    start_wrapper: Option<DefId>,
) {
    let mut ecx = create_ecx(tcx, main_id, start_wrapper).expect("Couldn't create ecx");

    let res: EvalResult = (|| {
        ecx.run()?;
        ecx.run_tls_dtors()
    })();

    match res {
        Ok(()) => {
            let leaks = ecx.memory().leak_report();
            if leaks != 0 {
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

#[derive(Clone, Default, PartialEq, Eq)]
pub struct Evaluator<'tcx> {
    /// Environment variables set by `setenv`
    /// Miri does not expose env vars from the host to the emulated program
    pub(crate) env_vars: HashMap<Vec<u8>, Pointer>,

    /// Use the lifetime
    _dummy : PhantomData<&'tcx ()>,
}

impl<'tcx> Hash for Evaluator<'tcx> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let Evaluator {
            env_vars,
            _dummy: _,
        } = self;

        env_vars.iter()
            .map(|(env, ptr)| {
                let mut h = FxHasher::default();
                env.hash(&mut h);
                ptr.hash(&mut h);
                h.finish()
            })
            .fold(0u64, |acc, hash| acc.wrapping_add(hash))
            .hash(state);
    }
}

pub type TlsKey = u128;

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct TlsEntry<'tcx> {
    data: Scalar, // Will eventually become a map from thread IDs to `Scalar`s, if we ever support more than one thread.
    dtor: Option<ty::Instance<'tcx>>,
}

#[derive(Clone, PartialEq, Eq)]
pub struct MemoryData<'tcx> {
    /// The Key to use for the next thread-local allocation.
    next_thread_local: TlsKey,

    /// pthreads-style thread-local storage.
    thread_local: BTreeMap<TlsKey, TlsEntry<'tcx>>,

    /// Memory regions that are locked by some function
    ///
    /// Only mutable (static mut, heap, stack) allocations have an entry in this map.
    /// The entry is created when allocating the memory and deleted after deallocation.
    locks: HashMap<AllocId, RangeMap<LockInfo<'tcx>>>,
}

impl<'tcx> MemoryData<'tcx> {
    fn new() -> Self {
        MemoryData {
            next_thread_local: 1, // start with 1 as we must not use 0 on Windows
            thread_local: BTreeMap::new(),
            locks: HashMap::new(),
        }
    }
}

impl<'tcx> Hash for MemoryData<'tcx> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let MemoryData {
            next_thread_local: _,
            thread_local,
            locks: _,
        } = self;

        thread_local.hash(state);
    }
}

impl<'mir, 'tcx> Machine<'mir, 'tcx> for Evaluator<'tcx> {
    type MemoryData = MemoryData<'tcx>;
    type MemoryKinds = memory::MemoryKind;

    const MUT_STATIC_KIND: Option<memory::MemoryKind> = Some(memory::MemoryKind::MutStatic);

    /// Returns Ok() when the function was handled, fail otherwise
    fn find_fn<'a>(
        ecx: &mut EvalContext<'a, 'mir, 'tcx, Self>,
        instance: ty::Instance<'tcx>,
        args: &[OpTy<'tcx>],
        dest: Option<PlaceTy<'tcx>>,
        ret: Option<mir::BasicBlock>,
    ) -> EvalResult<'tcx, Option<&'mir mir::Mir<'tcx>>> {
        ecx.find_fn(instance, args, dest, ret)
    }

    fn call_intrinsic<'a>(
        ecx: &mut rustc_mir::interpret::EvalContext<'a, 'mir, 'tcx, Self>,
        instance: ty::Instance<'tcx>,
        args: &[OpTy<'tcx>],
        dest: PlaceTy<'tcx>,
    ) -> EvalResult<'tcx> {
        ecx.call_intrinsic(instance, args, dest)
    }

    fn try_ptr_op<'a>(
        ecx: &rustc_mir::interpret::EvalContext<'a, 'mir, 'tcx, Self>,
        bin_op: mir::BinOp,
        left: Scalar,
        left_layout: TyLayout<'tcx>,
        right: Scalar,
        right_layout: TyLayout<'tcx>,
    ) -> EvalResult<'tcx, Option<(Scalar, bool)>> {
        ecx.ptr_op(bin_op, left, left_layout, right, right_layout)
    }

    fn box_alloc<'a>(
        ecx: &mut EvalContext<'a, 'mir, 'tcx, Self>,
        dest: PlaceTy<'tcx>,
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
            *dest,
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

    fn find_foreign_static<'a>(
        tcx: TyCtxtAt<'a, 'tcx, 'tcx>,
        def_id: DefId,
    ) -> EvalResult<'tcx, &'tcx Allocation> {
        let attrs = tcx.get_attrs(def_id);
        let link_name = match attr::first_attr_value_str_by_name(&attrs, "link_name") {
            Some(name) => name.as_str(),
            None => tcx.item_name(def_id).as_str(),
        };

        let alloc = match &link_name[..] {
            "__cxa_thread_atexit_impl" => {
                // This should be all-zero, pointer-sized
                let data = vec![0; tcx.data_layout.pointer_size.bytes() as usize];
                let alloc = Allocation::from_bytes(&data[..], tcx.data_layout.pointer_align);
                tcx.intern_const_alloc(alloc)
            }
            _ => return err!(Unimplemented(
                    format!("can't access foreign static: {}", link_name),
                )),
        };
        Ok(alloc)
    }

    fn validation_op<'a>(
        _ecx: &mut EvalContext<'a, 'mir, 'tcx, Self>,
        _op: ::rustc::mir::ValidationOp,
        _operand: &::rustc::mir::ValidationOperand<'tcx, ::rustc::mir::Place<'tcx>>,
    ) -> EvalResult<'tcx> {
        // FIXME: prevent this from ICEing
        //ecx.validation_op(op, operand)
        Ok(())
    }
}
