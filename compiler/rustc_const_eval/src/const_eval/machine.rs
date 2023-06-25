use rustc_hir::def::DefKind;
use rustc_hir::{LangItem, CRATE_HIR_ID};
use rustc_middle::mir;
use rustc_middle::mir::interpret::PointerArithmetic;
use rustc_middle::ty::layout::{FnAbiOf, TyAndLayout};
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_session::lint::builtin::INVALID_ALIGNMENT;
use std::borrow::Borrow;
use std::hash::Hash;
use std::ops::ControlFlow;

use rustc_data_structures::fx::FxIndexMap;
use rustc_data_structures::fx::IndexEntry;
use std::fmt;

use rustc_ast::Mutability;
use rustc_hir::def_id::DefId;
use rustc_middle::mir::AssertMessage;
use rustc_span::symbol::{sym, Symbol};
use rustc_target::abi::{Align, Size};
use rustc_target::spec::abi::Abi as CallAbi;

use crate::errors::{LongRunning, LongRunningWarn};
use crate::interpret::{
    self, compile_time_machine, AllocId, ConstAllocation, FnVal, Frame, ImmTy, InterpCx,
    InterpResult, OpTy, PlaceTy, Pointer, Scalar,
};
use crate::{errors, fluent_generated as fluent};

use super::error::*;

/// When hitting this many interpreted terminators we emit a deny by default lint
/// that notfies the user that their constant takes a long time to evaluate. If that's
/// what they intended, they can just allow the lint.
const LINT_TERMINATOR_LIMIT: usize = 2_000_000;
/// The limit used by `-Z tiny-const-eval-limit`. This smaller limit is useful for internal
/// tests not needing to run 30s or more to show some behaviour.
const TINY_LINT_TERMINATOR_LIMIT: usize = 20;
/// After this many interpreted terminators, we start emitting progress indicators at every
/// power of two of interpreted terminators.
const PROGRESS_INDICATOR_START: usize = 4_000_000;

/// Extra machine state for CTFE, and the Machine instance
pub struct CompileTimeInterpreter<'mir, 'tcx> {
    /// The number of terminators that have been evaluated.
    ///
    /// This is used to produce lints informing the user that the compiler is not stuck.
    /// Set to `usize::MAX` to never report anything.
    pub(super) num_evaluated_steps: usize,

    /// The virtual call stack.
    pub(super) stack: Vec<Frame<'mir, 'tcx, AllocId, ()>>,

    /// We need to make sure consts never point to anything mutable, even recursively. That is
    /// relied on for pattern matching on consts with references.
    /// To achieve this, two pieces have to work together:
    /// * Interning makes everything outside of statics immutable.
    /// * Pointers to allocations inside of statics can never leak outside, to a non-static global.
    /// This boolean here controls the second part.
    pub(super) can_access_statics: CanAccessStatics,

    /// Whether to check alignment during evaluation.
    pub(super) check_alignment: CheckAlignment,
}

#[derive(Copy, Clone)]
pub enum CheckAlignment {
    /// Ignore alignment when following relocations.
    /// This is mainly used in interning.
    No,
    /// Hard error when dereferencing a misaligned pointer.
    Error,
    /// Emit a future incompat lint when dereferencing a misaligned pointer.
    FutureIncompat,
}

impl CheckAlignment {
    pub fn should_check(&self) -> bool {
        match self {
            CheckAlignment::No => false,
            CheckAlignment::Error | CheckAlignment::FutureIncompat => true,
        }
    }
}

#[derive(Copy, Clone, PartialEq)]
pub(crate) enum CanAccessStatics {
    No,
    Yes,
}

impl From<bool> for CanAccessStatics {
    fn from(value: bool) -> Self {
        if value { Self::Yes } else { Self::No }
    }
}

impl<'mir, 'tcx> CompileTimeInterpreter<'mir, 'tcx> {
    pub(crate) fn new(
        can_access_statics: CanAccessStatics,
        check_alignment: CheckAlignment,
    ) -> Self {
        CompileTimeInterpreter {
            num_evaluated_steps: 0,
            stack: Vec::new(),
            can_access_statics,
            check_alignment,
        }
    }
}

impl<K: Hash + Eq, V> interpret::AllocMap<K, V> for FxIndexMap<K, V> {
    #[inline(always)]
    fn contains_key<Q: ?Sized + Hash + Eq>(&mut self, k: &Q) -> bool
    where
        K: Borrow<Q>,
    {
        FxIndexMap::contains_key(self, k)
    }

    #[inline(always)]
    fn insert(&mut self, k: K, v: V) -> Option<V> {
        FxIndexMap::insert(self, k, v)
    }

    #[inline(always)]
    fn remove<Q: ?Sized + Hash + Eq>(&mut self, k: &Q) -> Option<V>
    where
        K: Borrow<Q>,
    {
        FxIndexMap::remove(self, k)
    }

    #[inline(always)]
    fn filter_map_collect<T>(&self, mut f: impl FnMut(&K, &V) -> Option<T>) -> Vec<T> {
        self.iter().filter_map(move |(k, v)| f(k, &*v)).collect()
    }

    #[inline(always)]
    fn get_or<E>(&self, k: K, vacant: impl FnOnce() -> Result<V, E>) -> Result<&V, E> {
        match self.get(&k) {
            Some(v) => Ok(v),
            None => {
                vacant()?;
                bug!("The CTFE machine shouldn't ever need to extend the alloc_map when reading")
            }
        }
    }

    #[inline(always)]
    fn get_mut_or<E>(&mut self, k: K, vacant: impl FnOnce() -> Result<V, E>) -> Result<&mut V, E> {
        match self.entry(k) {
            IndexEntry::Occupied(e) => Ok(e.into_mut()),
            IndexEntry::Vacant(e) => {
                let v = vacant()?;
                Ok(e.insert(v))
            }
        }
    }
}

pub(crate) type CompileTimeEvalContext<'mir, 'tcx> =
    InterpCx<'mir, 'tcx, CompileTimeInterpreter<'mir, 'tcx>>;

#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub enum MemoryKind {
    Heap,
}

impl fmt::Display for MemoryKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MemoryKind::Heap => write!(f, "heap allocation"),
        }
    }
}

impl interpret::MayLeak for MemoryKind {
    #[inline(always)]
    fn may_leak(self) -> bool {
        match self {
            MemoryKind::Heap => false,
        }
    }
}

impl interpret::MayLeak for ! {
    #[inline(always)]
    fn may_leak(self) -> bool {
        // `self` is uninhabited
        self
    }
}

impl<'mir, 'tcx: 'mir> CompileTimeEvalContext<'mir, 'tcx> {
    /// "Intercept" a function call, because we have something special to do for it.
    /// All `#[rustc_do_not_const_check]` functions should be hooked here.
    /// If this returns `Some` function, which may be `instance` or a different function with
    /// compatible arguments, then evaluation should continue with that function.
    /// If this returns `None`, the function call has been handled and the function has returned.
    fn hook_special_const_fn(
        &mut self,
        instance: ty::Instance<'tcx>,
        args: &[OpTy<'tcx>],
        dest: &PlaceTy<'tcx>,
        ret: Option<mir::BasicBlock>,
    ) -> InterpResult<'tcx, Option<ty::Instance<'tcx>>> {
        let def_id = instance.def_id();

        if Some(def_id) == self.tcx.lang_items().panic_display()
            || Some(def_id) == self.tcx.lang_items().begin_panic_fn()
        {
            // &str or &&str
            assert!(args.len() == 1);

            let mut msg_place = self.deref_operand(&args[0])?;
            while msg_place.layout.ty.is_ref() {
                msg_place = self.deref_operand(&msg_place.into())?;
            }

            let msg = Symbol::intern(self.read_str(&msg_place)?);
            let span = self.find_closest_untracked_caller_location();
            let (file, line, col) = self.location_triple_for_span(span);
            return Err(ConstEvalErrKind::Panic { msg, file, line, col }.into());
        } else if Some(def_id) == self.tcx.lang_items().panic_fmt() {
            // For panic_fmt, call const_panic_fmt instead.
            let const_def_id = self.tcx.require_lang_item(LangItem::ConstPanicFmt, None);
            let new_instance = ty::Instance::resolve(
                *self.tcx,
                ty::ParamEnv::reveal_all(),
                const_def_id,
                instance.substs,
            )
            .unwrap()
            .unwrap();

            return Ok(Some(new_instance));
        } else if Some(def_id) == self.tcx.lang_items().align_offset_fn() {
            // For align_offset, we replace the function call if the pointer has no address.
            match self.align_offset(instance, args, dest, ret)? {
                ControlFlow::Continue(()) => return Ok(Some(instance)),
                ControlFlow::Break(()) => return Ok(None),
            }
        }
        Ok(Some(instance))
    }

    /// `align_offset(ptr, target_align)` needs special handling in const eval, because the pointer
    /// may not have an address.
    ///
    /// If `ptr` does have a known address, then we return `Continue(())` and the function call should
    /// proceed as normal.
    ///
    /// If `ptr` doesn't have an address, but its underlying allocation's alignment is at most
    /// `target_align`, then we call the function again with an dummy address relative to the
    /// allocation.
    ///
    /// If `ptr` doesn't have an address and `target_align` is stricter than the underlying
    /// allocation's alignment, then we return `usize::MAX` immediately.
    fn align_offset(
        &mut self,
        instance: ty::Instance<'tcx>,
        args: &[OpTy<'tcx>],
        dest: &PlaceTy<'tcx>,
        ret: Option<mir::BasicBlock>,
    ) -> InterpResult<'tcx, ControlFlow<()>> {
        assert_eq!(args.len(), 2);

        let ptr = self.read_pointer(&args[0])?;
        let target_align = self.read_scalar(&args[1])?.to_target_usize(self)?;

        if !target_align.is_power_of_two() {
            throw_ub_custom!(
                fluent::const_eval_align_offset_invalid_align,
                target_align = target_align,
            );
        }

        match self.ptr_try_get_alloc_id(ptr) {
            Ok((alloc_id, offset, _extra)) => {
                let (_size, alloc_align, _kind) = self.get_alloc_info(alloc_id);

                if target_align <= alloc_align.bytes() {
                    // Extract the address relative to the allocation base that is definitely
                    // sufficiently aligned and call `align_offset` again.
                    let addr = ImmTy::from_uint(offset.bytes(), args[0].layout).into();
                    let align = ImmTy::from_uint(target_align, args[1].layout).into();
                    let fn_abi = self.fn_abi_of_instance(instance, ty::List::empty())?;

                    // We replace the entire function call with a "tail call".
                    // Note that this happens before the frame of the original function
                    // is pushed on the stack.
                    self.eval_fn_call(
                        FnVal::Instance(instance),
                        (CallAbi::Rust, fn_abi),
                        &[addr, align],
                        /* with_caller_location = */ false,
                        dest,
                        ret,
                        mir::UnwindAction::Unreachable,
                    )?;
                    Ok(ControlFlow::Break(()))
                } else {
                    // Not alignable in const, return `usize::MAX`.
                    let usize_max = Scalar::from_target_usize(self.target_usize_max(), self);
                    self.write_scalar(usize_max, dest)?;
                    self.return_to_block(ret)?;
                    Ok(ControlFlow::Break(()))
                }
            }
            Err(_addr) => {
                // The pointer has an address, continue with function call.
                Ok(ControlFlow::Continue(()))
            }
        }
    }

    /// See documentation on the `ptr_guaranteed_cmp` intrinsic.
    fn guaranteed_cmp(&mut self, a: Scalar, b: Scalar) -> InterpResult<'tcx, u8> {
        Ok(match (a, b) {
            // Comparisons between integers are always known.
            (Scalar::Int { .. }, Scalar::Int { .. }) => {
                if a == b {
                    1
                } else {
                    0
                }
            }
            // Comparisons of abstract pointers with null pointers are known if the pointer
            // is in bounds, because if they are in bounds, the pointer can't be null.
            // Inequality with integers other than null can never be known for sure.
            (Scalar::Int(int), ptr @ Scalar::Ptr(..))
            | (ptr @ Scalar::Ptr(..), Scalar::Int(int))
                if int.is_null() && !self.scalar_may_be_null(ptr)? =>
            {
                0
            }
            // Equality with integers can never be known for sure.
            (Scalar::Int { .. }, Scalar::Ptr(..)) | (Scalar::Ptr(..), Scalar::Int { .. }) => 2,
            // FIXME: return a `1` for when both sides are the same pointer, *except* that
            // some things (like functions and vtables) do not have stable addresses
            // so we need to be careful around them (see e.g. #73722).
            // FIXME: return `0` for at least some comparisons where we can reliably
            // determine the result of runtime inequality tests at compile-time.
            // Examples include comparison of addresses in different static items.
            (Scalar::Ptr(..), Scalar::Ptr(..)) => 2,
        })
    }
}

impl<'mir, 'tcx> interpret::Machine<'mir, 'tcx> for CompileTimeInterpreter<'mir, 'tcx> {
    compile_time_machine!(<'mir, 'tcx>);

    type MemoryKind = MemoryKind;

    const PANIC_ON_ALLOC_FAIL: bool = false; // will be raised as a proper error

    #[inline(always)]
    fn enforce_alignment(ecx: &InterpCx<'mir, 'tcx, Self>) -> CheckAlignment {
        ecx.machine.check_alignment
    }

    #[inline(always)]
    fn enforce_validity(ecx: &InterpCx<'mir, 'tcx, Self>, layout: TyAndLayout<'tcx>) -> bool {
        ecx.tcx.sess.opts.unstable_opts.extra_const_ub_checks || layout.abi.is_uninhabited()
    }

    fn alignment_check_failed(
        ecx: &InterpCx<'mir, 'tcx, Self>,
        has: Align,
        required: Align,
        check: CheckAlignment,
    ) -> InterpResult<'tcx, ()> {
        let err = err_ub!(AlignmentCheckFailed { has, required }).into();
        match check {
            CheckAlignment::Error => Err(err),
            CheckAlignment::No => span_bug!(
                ecx.cur_span(),
                "`alignment_check_failed` called when no alignment check requested"
            ),
            CheckAlignment::FutureIncompat => {
                let (_, backtrace) = err.into_parts();
                backtrace.print_backtrace();
                let (span, frames) = super::get_span_and_frames(&ecx);

                ecx.tcx.emit_spanned_lint(
                    INVALID_ALIGNMENT,
                    ecx.stack().iter().find_map(|frame| frame.lint_root()).unwrap_or(CRATE_HIR_ID),
                    span,
                    errors::AlignmentCheckFailed {
                        has: has.bytes(),
                        required: required.bytes(),
                        frames,
                    },
                );
                Ok(())
            }
        }
    }

    fn load_mir(
        ecx: &InterpCx<'mir, 'tcx, Self>,
        instance: ty::InstanceDef<'tcx>,
    ) -> InterpResult<'tcx, &'tcx mir::Body<'tcx>> {
        match instance {
            ty::InstanceDef::Item(def) => {
                if ecx.tcx.is_ctfe_mir_available(def) {
                    Ok(ecx.tcx.mir_for_ctfe(def))
                } else if ecx.tcx.def_kind(def) == DefKind::AssocConst {
                    let guar = ecx.tcx.sess.delay_span_bug(
                        rustc_span::DUMMY_SP,
                        "This is likely a const item that is missing from its impl",
                    );
                    throw_inval!(AlreadyReported(guar.into()));
                } else {
                    // `find_mir_or_eval_fn` checks that this is a const fn before even calling us,
                    // so this should be unreachable.
                    let path = ecx.tcx.def_path_str(def);
                    bug!("trying to call extern function `{path}` at compile-time");
                }
            }
            _ => Ok(ecx.tcx.instance_mir(instance)),
        }
    }

    fn find_mir_or_eval_fn(
        ecx: &mut InterpCx<'mir, 'tcx, Self>,
        instance: ty::Instance<'tcx>,
        _abi: CallAbi,
        args: &[OpTy<'tcx>],
        dest: &PlaceTy<'tcx>,
        ret: Option<mir::BasicBlock>,
        _unwind: mir::UnwindAction, // unwinding is not supported in consts
    ) -> InterpResult<'tcx, Option<(&'mir mir::Body<'tcx>, ty::Instance<'tcx>)>> {
        debug!("find_mir_or_eval_fn: {:?}", instance);

        // Only check non-glue functions
        if let ty::InstanceDef::Item(def) = instance.def {
            // Execution might have wandered off into other crates, so we cannot do a stability-
            // sensitive check here. But we can at least rule out functions that are not const
            // at all.
            if !ecx.tcx.is_const_fn_raw(def) {
                // allow calling functions inside a trait marked with #[const_trait].
                if !ecx.tcx.is_const_default_method(def) {
                    // We certainly do *not* want to actually call the fn
                    // though, so be sure we return here.
                    throw_unsup_format!("calling non-const function `{}`", instance)
                }
            }

            let Some(new_instance) = ecx.hook_special_const_fn(instance, args, dest, ret)? else {
                return Ok(None);
            };

            if new_instance != instance {
                // We call another const fn instead.
                // However, we return the *original* instance to make backtraces work out
                // (and we hope this does not confuse the FnAbi checks too much).
                return Ok(Self::find_mir_or_eval_fn(
                    ecx,
                    new_instance,
                    _abi,
                    args,
                    dest,
                    ret,
                    _unwind,
                )?
                .map(|(body, _instance)| (body, instance)));
            }
        }

        // This is a const fn. Call it.
        Ok(Some((ecx.load_mir(instance.def, None)?, instance)))
    }

    fn call_intrinsic(
        ecx: &mut InterpCx<'mir, 'tcx, Self>,
        instance: ty::Instance<'tcx>,
        args: &[OpTy<'tcx>],
        dest: &PlaceTy<'tcx, Self::Provenance>,
        target: Option<mir::BasicBlock>,
        _unwind: mir::UnwindAction,
    ) -> InterpResult<'tcx> {
        // Shared intrinsics.
        if ecx.emulate_intrinsic(instance, args, dest, target)? {
            return Ok(());
        }
        let intrinsic_name = ecx.tcx.item_name(instance.def_id());

        // CTFE-specific intrinsics.
        let Some(ret) = target else {
            throw_unsup_format!("intrinsic `{intrinsic_name}` is not supported at compile-time");
        };
        match intrinsic_name {
            sym::ptr_guaranteed_cmp => {
                let a = ecx.read_scalar(&args[0])?;
                let b = ecx.read_scalar(&args[1])?;
                let cmp = ecx.guaranteed_cmp(a, b)?;
                ecx.write_scalar(Scalar::from_u8(cmp), dest)?;
            }
            sym::const_allocate => {
                let size = ecx.read_scalar(&args[0])?.to_target_usize(ecx)?;
                let align = ecx.read_scalar(&args[1])?.to_target_usize(ecx)?;

                let align = match Align::from_bytes(align) {
                    Ok(a) => a,
                    Err(err) => throw_ub_custom!(
                        fluent::const_eval_invalid_align_details,
                        name = "const_allocate",
                        err_kind = err.diag_ident(),
                        align = err.align()
                    ),
                };

                let ptr = ecx.allocate_ptr(
                    Size::from_bytes(size as u64),
                    align,
                    interpret::MemoryKind::Machine(MemoryKind::Heap),
                )?;
                ecx.write_pointer(ptr, dest)?;
            }
            sym::const_deallocate => {
                let ptr = ecx.read_pointer(&args[0])?;
                let size = ecx.read_scalar(&args[1])?.to_target_usize(ecx)?;
                let align = ecx.read_scalar(&args[2])?.to_target_usize(ecx)?;

                let size = Size::from_bytes(size);
                let align = match Align::from_bytes(align) {
                    Ok(a) => a,
                    Err(err) => throw_ub_custom!(
                        fluent::const_eval_invalid_align_details,
                        name = "const_deallocate",
                        err_kind = err.diag_ident(),
                        align = err.align()
                    ),
                };

                // If an allocation is created in an another const,
                // we don't deallocate it.
                let (alloc_id, _, _) = ecx.ptr_get_alloc_id(ptr)?;
                let is_allocated_in_another_const = matches!(
                    ecx.tcx.try_get_global_alloc(alloc_id),
                    Some(interpret::GlobalAlloc::Memory(_))
                );

                if !is_allocated_in_another_const {
                    ecx.deallocate_ptr(
                        ptr,
                        Some((size, align)),
                        interpret::MemoryKind::Machine(MemoryKind::Heap),
                    )?;
                }
            }
            _ => {
                throw_unsup_format!(
                    "intrinsic `{intrinsic_name}` is not supported at compile-time"
                );
            }
        }

        ecx.go_to_block(ret);
        Ok(())
    }

    fn assert_panic(
        ecx: &mut InterpCx<'mir, 'tcx, Self>,
        msg: &AssertMessage<'tcx>,
        _unwind: mir::UnwindAction,
    ) -> InterpResult<'tcx> {
        use rustc_middle::mir::AssertKind::*;
        // Convert `AssertKind<Operand>` to `AssertKind<Scalar>`.
        let eval_to_int =
            |op| ecx.read_immediate(&ecx.eval_operand(op, None)?).map(|x| x.to_const_int());
        let err = match msg {
            BoundsCheck { len, index } => {
                let len = eval_to_int(len)?;
                let index = eval_to_int(index)?;
                BoundsCheck { len, index }
            }
            Overflow(op, l, r) => Overflow(*op, eval_to_int(l)?, eval_to_int(r)?),
            OverflowNeg(op) => OverflowNeg(eval_to_int(op)?),
            DivisionByZero(op) => DivisionByZero(eval_to_int(op)?),
            RemainderByZero(op) => RemainderByZero(eval_to_int(op)?),
            ResumedAfterReturn(generator_kind) => ResumedAfterReturn(*generator_kind),
            ResumedAfterPanic(generator_kind) => ResumedAfterPanic(*generator_kind),
            MisalignedPointerDereference { ref required, ref found } => {
                MisalignedPointerDereference {
                    required: eval_to_int(required)?,
                    found: eval_to_int(found)?,
                }
            }
        };
        Err(ConstEvalErrKind::AssertFailure(err).into())
    }

    fn abort(_ecx: &mut InterpCx<'mir, 'tcx, Self>, msg: String) -> InterpResult<'tcx, !> {
        Err(ConstEvalErrKind::Abort(msg).into())
    }

    fn binary_ptr_op(
        _ecx: &InterpCx<'mir, 'tcx, Self>,
        _bin_op: mir::BinOp,
        _left: &ImmTy<'tcx>,
        _right: &ImmTy<'tcx>,
    ) -> InterpResult<'tcx, (Scalar, bool, Ty<'tcx>)> {
        throw_unsup_format!("pointer arithmetic or comparison is not supported at compile-time");
    }

    fn increment_const_eval_counter(ecx: &mut InterpCx<'mir, 'tcx, Self>) -> InterpResult<'tcx> {
        // The step limit has already been hit in a previous call to `increment_const_eval_counter`.

        if let Some(new_steps) = ecx.machine.num_evaluated_steps.checked_add(1) {
            let (limit, start) = if ecx.tcx.sess.opts.unstable_opts.tiny_const_eval_limit {
                (TINY_LINT_TERMINATOR_LIMIT, TINY_LINT_TERMINATOR_LIMIT)
            } else {
                (LINT_TERMINATOR_LIMIT, PROGRESS_INDICATOR_START)
            };

            ecx.machine.num_evaluated_steps = new_steps;
            // By default, we have a *deny* lint kicking in after some time
            // to ensure `loop {}` doesn't just go forever.
            // In case that lint got reduced, in particular for `--cap-lint` situations, we also
            // have a hard warning shown every now and then for really long executions.
            if new_steps == limit {
                // By default, we stop after a million steps, but the user can disable this lint
                // to be able to run until the heat death of the universe or power loss, whichever
                // comes first.
                let hir_id = ecx.best_lint_scope();
                let is_error = ecx
                    .tcx
                    .lint_level_at_node(
                        rustc_session::lint::builtin::LONG_RUNNING_CONST_EVAL,
                        hir_id,
                    )
                    .0
                    .is_error();
                let span = ecx.cur_span();
                ecx.tcx.emit_spanned_lint(
                    rustc_session::lint::builtin::LONG_RUNNING_CONST_EVAL,
                    hir_id,
                    span,
                    LongRunning { item_span: ecx.tcx.span },
                );
                // If this was a hard error, don't bother continuing evaluation.
                if is_error {
                    let guard = ecx
                        .tcx
                        .sess
                        .delay_span_bug(span, "The deny lint should have already errored");
                    throw_inval!(AlreadyReported(guard.into()));
                }
            } else if new_steps > start && new_steps.is_power_of_two() {
                // Only report after a certain number of terminators have been evaluated and the
                // current number of evaluated terminators is a power of 2. The latter gives us a cheap
                // way to implement exponential backoff.
                let span = ecx.cur_span();
                ecx.tcx.sess.emit_warning(LongRunningWarn { span, item_span: ecx.tcx.span });
            }
        }

        Ok(())
    }

    #[inline(always)]
    fn expose_ptr(
        _ecx: &mut InterpCx<'mir, 'tcx, Self>,
        _ptr: Pointer<AllocId>,
    ) -> InterpResult<'tcx> {
        // This is only reachable with -Zunleash-the-miri-inside-of-you.
        throw_unsup_format!("exposing pointers is not possible at compile-time")
    }

    #[inline(always)]
    fn init_frame_extra(
        ecx: &mut InterpCx<'mir, 'tcx, Self>,
        frame: Frame<'mir, 'tcx>,
    ) -> InterpResult<'tcx, Frame<'mir, 'tcx>> {
        // Enforce stack size limit. Add 1 because this is run before the new frame is pushed.
        if !ecx.recursion_limit.value_within_limit(ecx.stack().len() + 1) {
            throw_exhaust!(StackFrameLimitReached)
        } else {
            Ok(frame)
        }
    }

    #[inline(always)]
    fn stack<'a>(
        ecx: &'a InterpCx<'mir, 'tcx, Self>,
    ) -> &'a [Frame<'mir, 'tcx, Self::Provenance, Self::FrameExtra>] {
        &ecx.machine.stack
    }

    #[inline(always)]
    fn stack_mut<'a>(
        ecx: &'a mut InterpCx<'mir, 'tcx, Self>,
    ) -> &'a mut Vec<Frame<'mir, 'tcx, Self::Provenance, Self::FrameExtra>> {
        &mut ecx.machine.stack
    }

    fn before_access_global(
        _tcx: TyCtxt<'tcx>,
        machine: &Self,
        alloc_id: AllocId,
        alloc: ConstAllocation<'tcx>,
        static_def_id: Option<DefId>,
        is_write: bool,
    ) -> InterpResult<'tcx> {
        let alloc = alloc.inner();
        if is_write {
            // Write access. These are never allowed, but we give a targeted error message.
            match alloc.mutability {
                Mutability::Not => Err(err_ub!(WriteToReadOnly(alloc_id)).into()),
                Mutability::Mut => Err(ConstEvalErrKind::ModifiedGlobal.into()),
            }
        } else {
            // Read access. These are usually allowed, with some exceptions.
            if machine.can_access_statics == CanAccessStatics::Yes {
                // Machine configuration allows us read from anything (e.g., `static` initializer).
                Ok(())
            } else if static_def_id.is_some() {
                // Machine configuration does not allow us to read statics
                // (e.g., `const` initializer).
                // See const_eval::machine::MemoryExtra::can_access_statics for why
                // this check is so important: if we could read statics, we could read pointers
                // to mutable allocations *inside* statics. These allocations are not themselves
                // statics, so pointers to them can get around the check in `validity.rs`.
                Err(ConstEvalErrKind::ConstAccessesStatic.into())
            } else {
                // Immutable global, this read is fine.
                // But make sure we never accept a read from something mutable, that would be
                // unsound. The reason is that as the content of this allocation may be different
                // now and at run-time, so if we permit reading now we might return the wrong value.
                assert_eq!(alloc.mutability, Mutability::Not);
                Ok(())
            }
        }
    }
}

// Please do not add any code below the above `Machine` trait impl. I (oli-obk) plan more cleanups
// so we can end up having a file with just that impl, but for now, let's keep the impl discoverable
// at the bottom of this file.
