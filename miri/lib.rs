#![feature(
    i128_type,
    rustc_private,
)]

// From rustc.
#[macro_use]
extern crate log;
#[macro_use]
extern crate rustc;
extern crate syntax;

use rustc::ty::{self, TyCtxt};
use rustc::ty::layout::Layout;
use rustc::hir::def_id::DefId;
use rustc::mir;

use syntax::ast::Mutability;
use syntax::codemap::Span;

use std::collections::{HashMap, BTreeMap};

#[macro_use]
extern crate rustc_miri;
pub use rustc_miri::interpret::*;

mod fn_call;
mod operator;
mod intrinsic;
mod helpers;
mod memory;
mod tls;

use fn_call::EvalContextExt as MissingFnsEvalContextExt;
use operator::EvalContextExt as OperatorEvalContextExt;
use intrinsic::EvalContextExt as IntrinsicEvalContextExt;
use tls::EvalContextExt as TlsEvalContextExt;

pub fn eval_main<'a, 'tcx: 'a>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    main_id: DefId,
    start_wrapper: Option<DefId>,
    limits: ResourceLimits,
) {
    fn run_main<'a, 'tcx: 'a>(
        ecx: &mut rustc_miri::interpret::EvalContext<'a, 'tcx, Evaluator>,
        main_id: DefId,
        start_wrapper: Option<DefId>,
    ) -> EvalResult<'tcx> {
        let main_instance = ty::Instance::mono(ecx.tcx, main_id);
        let main_mir = ecx.load_mir(main_instance.def)?;
        let mut cleanup_ptr = None; // Pointer to be deallocated when we are done

        if !main_mir.return_ty.is_nil() || main_mir.arg_count != 0 {
            return err!(Unimplemented(
                "miri does not support main functions without `fn()` type signatures"
                    .to_owned(),
            ));
        }

        if let Some(start_id) = start_wrapper {
            let start_instance = ty::Instance::mono(ecx.tcx, start_id);
            let start_mir = ecx.load_mir(start_instance.def)?;

            if start_mir.arg_count != 3 {
                return err!(AbiViolation(format!(
                    "'start' lang item should have three arguments, but has {}",
                    start_mir.arg_count
                )));
            }

            // Return value
            let size = ecx.tcx.data_layout.pointer_size.bytes();
            let align = ecx.tcx.data_layout.pointer_align.abi();
            let ret_ptr = ecx.memory_mut().allocate(size, align, MemoryKind::Stack)?;
            cleanup_ptr = Some(ret_ptr);

            // Push our stack frame
            ecx.push_stack_frame(
                start_instance,
                start_mir.span,
                start_mir,
                Lvalue::from_ptr(ret_ptr),
                StackPopCleanup::None,
            )?;

            let mut args = ecx.frame().mir.args_iter();

            // First argument: pointer to main()
            let main_ptr = ecx.memory_mut().create_fn_alloc(main_instance);
            let dest = ecx.eval_lvalue(&mir::Lvalue::Local(args.next().unwrap()))?;
            let main_ty = main_instance.def.def_ty(ecx.tcx);
            let main_ptr_ty = ecx.tcx.mk_fn_ptr(main_ty.fn_sig(ecx.tcx));
            ecx.write_value(
                ValTy {
                    value: Value::ByVal(PrimVal::Ptr(main_ptr)),
                    ty: main_ptr_ty,
                },
                dest,
            )?;

            // Second argument (argc): 1
            let dest = ecx.eval_lvalue(&mir::Lvalue::Local(args.next().unwrap()))?;
            let ty = ecx.tcx.types.isize;
            ecx.write_primval(dest, PrimVal::Bytes(1), ty)?;

            // FIXME: extract main source file path
            // Third argument (argv): &[b"foo"]
            let dest = ecx.eval_lvalue(&mir::Lvalue::Local(args.next().unwrap()))?;
            let ty = ecx.tcx.mk_imm_ptr(ecx.tcx.mk_imm_ptr(ecx.tcx.types.u8));
            let foo = ecx.memory.allocate_cached(b"foo\0")?;
            let ptr_size = ecx.memory.pointer_size();
            let foo_ptr = ecx.memory.allocate(ptr_size * 1, ptr_size, MemoryKind::UninitializedStatic)?;
            ecx.memory.write_primval(foo_ptr.into(), PrimVal::Ptr(foo.into()), ptr_size, false)?;
            ecx.memory.mark_static_initalized(foo_ptr.alloc_id, Mutability::Immutable)?;
            ecx.write_ptr(dest, foo_ptr.into(), ty)?;

            assert!(args.next().is_none(), "start lang item has more arguments than expected");
        } else {
            ecx.push_stack_frame(
                main_instance,
                main_mir.span,
                main_mir,
                Lvalue::undef(),
                StackPopCleanup::None,
            )?;

            // No arguments
            let mut args = ecx.frame().mir.args_iter();
            assert!(args.next().is_none(), "main function must not have arguments");
        }

        while ecx.step()? {}
        ecx.run_tls_dtors()?;
        if let Some(cleanup_ptr) = cleanup_ptr {
            ecx.memory_mut().deallocate(
                cleanup_ptr,
                None,
                MemoryKind::Stack,
            )?;
        }
        Ok(())
    }

    let mut ecx = EvalContext::new(tcx, limits, Default::default(), Default::default());
    match run_main(&mut ecx, main_id, start_wrapper) {
        Ok(()) => {
            let leaks = ecx.memory().leak_report();
            if leaks != 0 {
                tcx.sess.err("the evaluated program leaked memory");
            }
        }
        Err(mut e) => {
            ecx.report(&mut e);
        }
    }
}

pub struct Evaluator;
#[derive(Default)]
pub struct EvaluatorData {
    /// Environment variables set by `setenv`
    /// Miri does not expose env vars from the host to the emulated program
    pub(crate) env_vars: HashMap<Vec<u8>, MemoryPointer>,
}

pub type TlsKey = usize;

#[derive(Copy, Clone, Debug)]
pub struct TlsEntry<'tcx> {
    data: Pointer, // Will eventually become a map from thread IDs to `Pointer`s, if we ever support more than one thread.
    dtor: Option<ty::Instance<'tcx>>,
}

#[derive(Default)]
pub struct MemoryData<'tcx> {
    /// The Key to use for the next thread-local allocation.
    next_thread_local: TlsKey,

    /// pthreads-style thread-local storage.
    thread_local: BTreeMap<TlsKey, TlsEntry<'tcx>>,
}

impl<'tcx> Machine<'tcx> for Evaluator {
    type Data = EvaluatorData;
    type MemoryData = MemoryData<'tcx>;
    type MemoryKinds = memory::MemoryKind;

    /// Returns Ok() when the function was handled, fail otherwise
    fn eval_fn_call<'a>(
        ecx: &mut EvalContext<'a, 'tcx, Self>,
        instance: ty::Instance<'tcx>,
        destination: Option<(Lvalue, mir::BasicBlock)>,
        args: &[ValTy<'tcx>],
        span: Span,
        sig: ty::FnSig<'tcx>,
    ) -> EvalResult<'tcx, bool> {
        ecx.eval_fn_call(instance, destination, args, span, sig)
    }

    fn call_intrinsic<'a>(
        ecx: &mut rustc_miri::interpret::EvalContext<'a, 'tcx, Self>,
        instance: ty::Instance<'tcx>,
        args: &[ValTy<'tcx>],
        dest: Lvalue,
        dest_ty: ty::Ty<'tcx>,
        dest_layout: &'tcx Layout,
        target: mir::BasicBlock,
    ) -> EvalResult<'tcx> {
        ecx.call_intrinsic(instance, args, dest, dest_ty, dest_layout, target)
    }

    fn try_ptr_op<'a>(
        ecx: &rustc_miri::interpret::EvalContext<'a, 'tcx, Self>,
        bin_op: mir::BinOp,
        left: PrimVal,
        left_ty: ty::Ty<'tcx>,
        right: PrimVal,
        right_ty: ty::Ty<'tcx>,
    ) -> EvalResult<'tcx, Option<(PrimVal, bool)>> {
        ecx.ptr_op(bin_op, left, left_ty, right, right_ty)
    }

    fn mark_static_initialized(m: memory::MemoryKind) -> EvalResult<'tcx> {
        use memory::MemoryKind::*;
        match m {
            // FIXME: This could be allowed, but not for env vars set during miri execution
            Env => err!(Unimplemented("statics can't refer to env vars".to_owned())),
            _ => Ok(()),
        }
    }

    fn box_alloc<'a>(
        ecx: &mut EvalContext<'a, 'tcx, Self>,
        ty: ty::Ty<'tcx>,
        dest: Lvalue,
    ) -> EvalResult<'tcx> {
        let size = ecx.type_size(ty)?.expect("box only works with sized types");
        let align = ecx.type_align(ty)?;

        // Call the `exchange_malloc` lang item
        let malloc = ecx.tcx.lang_items().exchange_malloc_fn().unwrap();
        let malloc = ty::Instance::mono(ecx.tcx, malloc);
        let malloc_mir = ecx.load_mir(malloc.def)?;
        ecx.push_stack_frame(
            malloc,
            malloc_mir.span,
            malloc_mir,
            dest,
            // Don't do anything when we are done.  The statement() function will increment
            // the old stack frame's stmt counter to the next statement, which means that when
            // exchange_malloc returns, we go on evaluating exactly where we want to be.
            StackPopCleanup::None,
        )?;

        let mut args = ecx.frame().mir.args_iter();
        let usize = ecx.tcx.types.usize;

        // First argument: size
        let dest = ecx.eval_lvalue(&mir::Lvalue::Local(args.next().unwrap()))?;
        ecx.write_value(
            ValTy {
                value: Value::ByVal(PrimVal::Bytes(size as u128)),
                ty: usize,
            },
            dest,
        )?;

        // Second argument: align
        let dest = ecx.eval_lvalue(&mir::Lvalue::Local(args.next().unwrap()))?;
        ecx.write_value(
            ValTy {
                value: Value::ByVal(PrimVal::Bytes(align as u128)),
                ty: usize,
            },
            dest,
        )?;

        // No more arguments
        assert!(args.next().is_none(), "exchange_malloc lang item has more arguments than expected");
        Ok(())
    }

    fn global_item_with_linkage<'a>(
        ecx: &mut EvalContext<'a, 'tcx, Self>,
        instance: ty::Instance<'tcx>,
        mutability: Mutability,
    ) -> EvalResult<'tcx> {
        // FIXME: check that it's `#[linkage = "extern_weak"]`
        trace!("Initializing an extern global with NULL");
        let ptr_size = ecx.memory.pointer_size();
        let ptr = ecx.memory.allocate(
            ptr_size,
            ptr_size,
            MemoryKind::UninitializedStatic,
        )?;
        ecx.memory.write_ptr_sized_unsigned(ptr, PrimVal::Bytes(0))?;
        ecx.memory.mark_static_initalized(ptr.alloc_id, mutability)?;
        ecx.globals.insert(
            GlobalId {
                instance,
                promoted: None,
            },
            PtrAndAlign {
                ptr: ptr.into(),
                aligned: true,
            },
        );
        Ok(())
    }
}
