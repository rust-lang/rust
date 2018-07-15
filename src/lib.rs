#![feature(
    rustc_private,
    catch_expr,
    inclusive_range_fields,
    inclusive_range_methods,
)]

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
extern crate regex;
#[macro_use]
extern crate lazy_static;

use rustc::ty::{self, TyCtxt};
use rustc::ty::layout::{TyLayout, LayoutOf, Size};
use rustc::ty::subst::Subst;
use rustc::hir::def_id::DefId;
use rustc::mir;

use rustc_data_structures::fx::FxHasher;

use syntax::ast::Mutability;
use syntax::codemap::Span;

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
mod validation;

use fn_call::EvalContextExt as MissingFnsEvalContextExt;
use operator::EvalContextExt as OperatorEvalContextExt;
use intrinsic::EvalContextExt as IntrinsicEvalContextExt;
use tls::EvalContextExt as TlsEvalContextExt;
use locks::LockInfo;
use locks::MemoryExt as LockMemoryExt;
use validation::EvalContextExt as ValidationEvalContextExt;
use range_map::RangeMap;
use validation::{ValidationQuery, AbsPlace};

pub trait ScalarExt {
    fn null() -> Self;
    fn from_i32(i: i32) -> Self;
    fn from_u128(i: u128) -> Self;
    fn from_i128(i: i128) -> Self;
    fn from_usize(i: u64, ptr_size: Size) -> Self;
    fn from_isize(i: i64, ptr_size: Size) -> Self;
    fn from_f32(f: f32) -> Self;
    fn from_f64(f: f64) -> Self;
    fn to_usize<'a, 'mir, 'tcx>(self, ecx: &rustc_mir::interpret::EvalContext<'a, 'mir, 'tcx, Evaluator<'tcx>>) -> EvalResult<'static, u64>;
    fn is_null(self) -> EvalResult<'static, bool>;
    /// HACK: this function just extracts all bits if `defined != 0`
    /// Mainly used for args of C-functions and we should totally correctly fetch the size
    /// of their arguments
    fn to_bytes(self) -> EvalResult<'static, u128>;
}

impl ScalarExt for Scalar {
    fn null() -> Self {
        Scalar::Bits { bits: 0, defined: 128 }
    }

    fn from_i32(i: i32) -> Self {
        Scalar::Bits { bits: i as u32 as u128, defined: 32 }
    }

    fn from_u128(i: u128) -> Self {
        Scalar::Bits { bits: i, defined: 128 }
    }

    fn from_i128(i: i128) -> Self {
        Scalar::Bits { bits: i as u128, defined: 128 }
    }

    fn from_usize(i: u64, ptr_size: Size) -> Self {
        Scalar::Bits { bits: i as u128, defined: ptr_size.bits() as u8 }
    }

    fn from_isize(i: i64, ptr_size: Size) -> Self {
        Scalar::Bits { bits: i as i128 as u128, defined: ptr_size.bits() as u8 }
    }

    fn from_f32(f: f32) -> Self {
        Scalar::Bits { bits: f.to_bits() as u128, defined: 32 }
    }

    fn from_f64(f: f64) -> Self {
        Scalar::Bits { bits: f.to_bits() as u128, defined: 64 }
    }

    fn to_usize<'a, 'mir, 'tcx>(self, ecx: &rustc_mir::interpret::EvalContext<'a, 'mir, 'tcx, Evaluator<'tcx>>) -> EvalResult<'static, u64> {
        let b = self.to_bits(ecx.memory.pointer_size())?;
        assert_eq!(b as u64 as u128, b);
        Ok(b as u64)
    }

    fn is_null(self) -> EvalResult<'static, bool> {
        match self {
            Scalar::Bits { bits, defined } => {
                if defined > 0 {
                    Ok(bits == 0)
                } else {
                    err!(ReadUndefBytes)
                }
            }
            Scalar::Ptr(_) => Ok(false)
        }
    }

    fn to_bytes(self) -> EvalResult<'static, u128> {
        match self {
            Scalar::Bits { defined: 0, .. } => err!(ReadUndefBytes),
            Scalar::Bits { bits, .. } => Ok(bits),
            Scalar::Ptr(_) => err!(ReadPointerAsBytes),
        }
    }
}

pub fn create_ecx<'a, 'mir: 'a, 'tcx: 'mir>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    main_id: DefId,
    start_wrapper: Option<DefId>,
) -> EvalResult<'tcx, (EvalContext<'a, 'mir, 'tcx, Evaluator<'tcx>>, Option<Pointer>)> {
    let mut ecx = EvalContext::new(tcx.at(syntax::codemap::DUMMY_SP), ty::ParamEnv::reveal_all(), Default::default(), Default::default());

    let main_instance = ty::Instance::mono(ecx.tcx.tcx, main_id);
    let main_mir = ecx.load_mir(main_instance.def)?;
    let mut cleanup_ptr = None; // Scalar to be deallocated when we are done

    if !main_mir.return_ty().is_nil() || main_mir.arg_count != 0 {
        return err!(Unimplemented(
            "miri does not support main functions without `fn()` type signatures"
                .to_owned(),
        ));
    }

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

        // Return value
        let size = ecx.tcx.data_layout.pointer_size;
        let align = ecx.tcx.data_layout.pointer_align;
        let ret_ptr = ecx.memory_mut().allocate(size, align, MemoryKind::Stack)?;
        cleanup_ptr = Some(ret_ptr);

        // Push our stack frame
        ecx.push_stack_frame(
            start_instance,
            start_mir.span,
            start_mir,
            Place::from_ptr(ret_ptr, align),
            StackPopCleanup::None,
        )?;

        let mut args = ecx.frame().mir.args_iter();

        // First argument: pointer to main()
        let main_ptr = ecx.memory_mut().create_fn_alloc(main_instance);
        let dest = ecx.eval_place(&mir::Place::Local(args.next().unwrap()))?;
        let main_ty = main_instance.ty(ecx.tcx.tcx);
        let main_ptr_ty = ecx.tcx.mk_fn_ptr(main_ty.fn_sig(ecx.tcx.tcx));
        ecx.write_value(
            ValTy {
                value: Value::Scalar(Scalar::Ptr(main_ptr)),
                ty: main_ptr_ty,
            },
            dest,
        )?;

        // Second argument (argc): 1
        let dest = ecx.eval_place(&mir::Place::Local(args.next().unwrap()))?;
        let ty = ecx.tcx.types.isize;
        ecx.write_scalar(dest, Scalar::from_u128(1), ty)?;

        // FIXME: extract main source file path
        // Third argument (argv): &[b"foo"]
        let dest = ecx.eval_place(&mir::Place::Local(args.next().unwrap()))?;
        let ty = ecx.tcx.mk_imm_ptr(ecx.tcx.mk_imm_ptr(ecx.tcx.types.u8));
        let foo = ecx.memory.allocate_bytes(b"foo\0");
        let ptr_size = ecx.memory.pointer_size();
        let ptr_align = ecx.tcx.data_layout.pointer_align;
        let foo_ptr = ecx.memory.allocate(ptr_size, ptr_align, MemoryKind::Stack)?;
        ecx.memory.write_scalar(foo_ptr.into(), ptr_align, Scalar::Ptr(foo), ptr_size, false)?;
        ecx.memory.mark_static_initialized(foo_ptr.alloc_id, Mutability::Immutable)?;
        ecx.write_ptr(dest, foo_ptr.into(), ty)?;

        assert!(args.next().is_none(), "start lang item has more arguments than expected");
    } else {
        ecx.push_stack_frame(
            main_instance,
            main_mir.span,
            main_mir,
            Place::from_scalar_ptr(Scalar::from_u128(1), ty::layout::Align::from_bytes(1, 1).unwrap()),
            StackPopCleanup::None,
        )?;

        // No arguments
        let mut args = ecx.frame().mir.args_iter();
        assert!(args.next().is_none(), "main function must not have arguments");
    }

    Ok((ecx, cleanup_ptr))
}

pub fn eval_main<'a, 'tcx: 'a>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    main_id: DefId,
    start_wrapper: Option<DefId>,
) {
    let (mut ecx, cleanup_ptr) = create_ecx(tcx, main_id, start_wrapper).expect("Couldn't create ecx");

    let res: EvalResult = do catch {
        while ecx.step()? {}
        ecx.run_tls_dtors()?;
        if let Some(cleanup_ptr) = cleanup_ptr {
            ecx.memory_mut().deallocate(
                cleanup_ptr,
                None,
                MemoryKind::Stack,
            )?;
        }
    };

    match res {
        Ok(()) => {
            let leaks = ecx.memory().leak_report();
            if leaks != 0 {
                // TODO: Prevent leaks which aren't supposed to be there
                //tcx.sess.err("the evaluated program leaked memory");
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

            for (i, frame) in ecx.stack().iter().enumerate() {
                trace!("-------------------");
                trace!("Frame {}", i);
                trace!("    return: {:#?}", frame.return_place);
                for (i, local) in frame.locals.iter().enumerate() {
                    if let Some(local) = local {
                        trace!("    local {}: {:?}", i, local);
                    }
                }
            }
        }
    }
}

#[derive(Clone, Default, PartialEq, Eq)]
pub struct Evaluator<'tcx> {
    /// Environment variables set by `setenv`
    /// Miri does not expose env vars from the host to the emulated program
    pub(crate) env_vars: HashMap<Vec<u8>, Pointer>,

    /// Places that were suspended by the validation subsystem, and will be recovered later
    pub(crate) suspended: HashMap<DynamicLifetime, Vec<ValidationQuery<'tcx>>>,
}

impl<'tcx> Hash for Evaluator<'tcx> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let Evaluator {
            env_vars,
            suspended: _,
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

#[derive(Clone, Default, PartialEq, Eq)]
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

    statics: HashMap<GlobalId<'tcx>, AllocId>,
}

impl<'tcx> Hash for MemoryData<'tcx> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let MemoryData {
            next_thread_local: _,
            thread_local,
            locks: _,
            statics: _,
        } = self;

        thread_local.hash(state);
    }
}

impl<'mir, 'tcx: 'mir> Machine<'mir, 'tcx> for Evaluator<'tcx> {
    type MemoryData = MemoryData<'tcx>;
    type MemoryKinds = memory::MemoryKind;

    /// Returns Ok() when the function was handled, fail otherwise
    fn eval_fn_call<'a>(
        ecx: &mut EvalContext<'a, 'mir, 'tcx, Self>,
        instance: ty::Instance<'tcx>,
        destination: Option<(Place, mir::BasicBlock)>,
        args: &[ValTy<'tcx>],
        span: Span,
        sig: ty::FnSig<'tcx>,
    ) -> EvalResult<'tcx, bool> {
        ecx.eval_fn_call(instance, destination, args, span, sig)
    }

    fn call_intrinsic<'a>(
        ecx: &mut rustc_mir::interpret::EvalContext<'a, 'mir, 'tcx, Self>,
        instance: ty::Instance<'tcx>,
        args: &[ValTy<'tcx>],
        dest: Place,
        dest_layout: TyLayout<'tcx>,
        target: mir::BasicBlock,
    ) -> EvalResult<'tcx> {
        ecx.call_intrinsic(instance, args, dest, dest_layout, target)
    }

    fn try_ptr_op<'a>(
        ecx: &rustc_mir::interpret::EvalContext<'a, 'mir, 'tcx, Self>,
        bin_op: mir::BinOp,
        left: Scalar,
        left_ty: ty::Ty<'tcx>,
        right: Scalar,
        right_ty: ty::Ty<'tcx>,
    ) -> EvalResult<'tcx, Option<(Scalar, bool)>> {
        ecx.ptr_op(bin_op, left, left_ty, right, right_ty)
    }

    fn mark_static_initialized<'a>(
        mem: &mut Memory<'a, 'mir, 'tcx, Self>,
        id: AllocId,
        _mutability: Mutability,
    ) -> EvalResult<'tcx, bool> {
        use memory::MemoryKind::*;
        match mem.get_alloc_kind(id) {
            // FIXME: This could be allowed, but not for env vars set during miri execution
            Some(MemoryKind::Machine(Env)) => err!(Unimplemented("statics can't refer to env vars".to_owned())),
            _ => Ok(false), // TODO: What does the bool mean?
        }
    }

    fn init_static<'a>(
        ecx: &mut EvalContext<'a, 'mir, 'tcx, Self>,
        cid: GlobalId<'tcx>,
    ) -> EvalResult<'tcx, AllocId> {
        // Step 1: If the static has already been evaluated return the cached version
        if let Some(alloc_id) = ecx.memory.data.statics.get(&cid) {
            return Ok(*alloc_id);
        }

        let tcx = ecx.tcx.tcx;

        // Step 2: Load mir
        let mut mir = ecx.load_mir(cid.instance.def)?;
        if let Some(index) = cid.promoted {
            mir = &mir.promoted[index];
        }
        assert!(mir.arg_count == 0);

        // Step 3: Allocate storage
        let layout = ecx.layout_of(mir.return_ty().subst(tcx, cid.instance.substs))?;
        assert!(!layout.is_unsized());
        let ptr = ecx.memory.allocate(
            layout.size,
            layout.align,
            MemoryKind::Stack,
        )?;

        // Step 4: Cache allocation id for recursive statics
        assert!(ecx.memory.data.statics.insert(cid, ptr.alloc_id).is_none());

        // Step 5: Push stackframe to evaluate static
        let cleanup = StackPopCleanup::None;
        ecx.push_stack_frame(
            cid.instance,
            mir.span,
            mir,
            Place::from_ptr(ptr, layout.align),
            cleanup,
        )?;

        // Step 6: Step until static has been initialized
        let call_stackframe = ecx.stack().len();
        while ecx.step()? && ecx.stack().len() >= call_stackframe {
            if ecx.stack().len() == call_stackframe {
                let frame = ecx.frame_mut();
                let bb = &frame.mir.basic_blocks()[frame.block];
                if bb.statements.len() == frame.stmt && !bb.is_cleanup {
                    if let ::rustc::mir::TerminatorKind::Return = bb.terminator().kind {
                        for (local, _local_decl) in mir.local_decls.iter_enumerated().skip(1) {
                            // Don't deallocate locals, because the return value might reference them
                            frame.storage_dead(local);
                        }
                    }
                }
            }
        }

        // TODO: Freeze immutable statics without copying them to the global static cache

        // Step 7: Return the alloc
        Ok(ptr.alloc_id)
    }

    fn box_alloc<'a>(
        ecx: &mut EvalContext<'a, 'mir, 'tcx, Self>,
        ty: ty::Ty<'tcx>,
        dest: Place,
    ) -> EvalResult<'tcx> {
        let layout = ecx.layout_of(ty)?;

        // Call the `exchange_malloc` lang item
        let malloc = ecx.tcx.lang_items().exchange_malloc_fn().unwrap();
        let malloc = ty::Instance::mono(ecx.tcx.tcx, malloc);
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
        let dest = ecx.eval_place(&mir::Place::Local(args.next().unwrap()))?;
        ecx.write_value(
            ValTy {
                value: Value::Scalar(Scalar::from_u128(match layout.size.bytes() {
                    0 => 1 as u128,
                    size => size as u128,
                })),
                ty: usize,
            },
            dest,
        )?;

        // Second argument: align
        let dest = ecx.eval_place(&mir::Place::Local(args.next().unwrap()))?;
        ecx.write_value(
            ValTy {
                value: Value::Scalar(Scalar::from_u128(layout.align.abi().into())),
                ty: usize,
            },
            dest,
        )?;

        // No more arguments
        assert!(args.next().is_none(), "exchange_malloc lang item has more arguments than expected");
        Ok(())
    }

    fn global_item_with_linkage<'a>(
        _ecx: &mut EvalContext<'a, 'mir, 'tcx, Self>,
        _instance: ty::Instance<'tcx>,
        _mutability: Mutability,
    ) -> EvalResult<'tcx> {
        panic!("remove this function from rustc");
    }

    fn check_locks<'a>(
        mem: &Memory<'a, 'mir, 'tcx, Self>,
        ptr: Pointer,
        size: Size,
        access: AccessKind,
    ) -> EvalResult<'tcx> {
        mem.check_locks(ptr, size.bytes(), access)
    }

    fn add_lock<'a>(
        mem: &mut Memory<'a, 'mir, 'tcx, Self>,
        id: AllocId,
    ) {
        mem.data.locks.insert(id, RangeMap::new());
    }

    fn free_lock<'a>(
        mem: &mut Memory<'a, 'mir, 'tcx, Self>,
        id: AllocId,
        len: u64,
    ) -> EvalResult<'tcx> {
        mem.data.locks
            .remove(&id)
            .expect("allocation has no corresponding locks")
            .check(
                Some(mem.cur_frame),
                0,
                len,
                AccessKind::Read,
            )
            .map_err(|lock| {
                EvalErrorKind::DeallocatedLockedMemory {
                    //ptr, FIXME
                    ptr: Pointer {
                        alloc_id: AllocId(0),
                        offset: Size::from_bytes(0),
                    },
                    lock: lock.active,
                }.into()
            })
    }

    fn end_region<'a>(
        ecx: &mut EvalContext<'a, 'mir, 'tcx, Self>,
        reg: Option<::rustc::middle::region::Scope>,
    ) -> EvalResult<'tcx> {
        ecx.end_region(reg)
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
