use std::borrow::{Borrow, Cow};
use std::fmt;
use std::hash::Hash;

use rustc_abi::{Align, Size};
use rustc_ast::Mutability;
use rustc_data_structures::fx::{FxHashMap, FxIndexMap, IndexEntry};
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir::{self as hir, CRATE_HIR_ID, LangItem};
use rustc_middle::mir::AssertMessage;
use rustc_middle::mir::interpret::ReportedErrorInfo;
use rustc_middle::query::TyCtxtAt;
use rustc_middle::ty::layout::{HasTypingEnv, TyAndLayout};
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_middle::{bug, mir};
use rustc_span::{Span, Symbol, sym};
use rustc_target::callconv::FnAbi;
use tracing::debug;

use super::error::*;
use crate::errors::{LongRunning, LongRunningWarn};
use crate::fluent_generated as fluent;
use crate::interpret::{
    self, AllocId, AllocInit, AllocRange, ConstAllocation, CtfeProvenance, FnArg, Frame,
    GlobalAlloc, ImmTy, InterpCx, InterpResult, OpTy, PlaceTy, Pointer, RangeSet, Scalar,
    compile_time_machine, interp_ok, throw_exhaust, throw_inval, throw_ub, throw_ub_custom,
    throw_unsup, throw_unsup_format,
};

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

/// Extra machine state for CTFE, and the Machine instance.
//
// Should be public because out-of-tree rustc consumers need this
// if they want to interact with constant values.
pub struct CompileTimeMachine<'tcx> {
    /// The number of terminators that have been evaluated.
    ///
    /// This is used to produce lints informing the user that the compiler is not stuck.
    /// Set to `usize::MAX` to never report anything.
    pub(super) num_evaluated_steps: usize,

    /// The virtual call stack.
    pub(super) stack: Vec<Frame<'tcx>>,

    /// Pattern matching on consts with references would be unsound if those references
    /// could point to anything mutable. Therefore, when evaluating consts and when constructing valtrees,
    /// we ensure that only immutable global memory can be accessed.
    pub(super) can_access_mut_global: CanAccessMutGlobal,

    /// Whether to check alignment during evaluation.
    pub(super) check_alignment: CheckAlignment,

    /// If `Some`, we are evaluating the initializer of the static with the given `LocalDefId`,
    /// storing the result in the given `AllocId`.
    /// Used to prevent reads from a static's base allocation, as that may allow for self-initialization loops.
    pub(crate) static_root_ids: Option<(AllocId, LocalDefId)>,

    /// A cache of "data range" computations for unions (i.e., the offsets of non-padding bytes).
    union_data_ranges: FxHashMap<Ty<'tcx>, RangeSet>,
}

#[derive(Copy, Clone)]
pub enum CheckAlignment {
    /// Ignore all alignment requirements.
    /// This is mainly used in interning.
    No,
    /// Hard error when dereferencing a misaligned pointer.
    Error,
}

#[derive(Copy, Clone, PartialEq)]
pub(crate) enum CanAccessMutGlobal {
    No,
    Yes,
}

impl From<bool> for CanAccessMutGlobal {
    fn from(value: bool) -> Self {
        if value { Self::Yes } else { Self::No }
    }
}

impl<'tcx> CompileTimeMachine<'tcx> {
    pub(crate) fn new(
        can_access_mut_global: CanAccessMutGlobal,
        check_alignment: CheckAlignment,
    ) -> Self {
        CompileTimeMachine {
            num_evaluated_steps: 0,
            stack: Vec::new(),
            can_access_mut_global,
            check_alignment,
            static_root_ids: None,
            union_data_ranges: FxHashMap::default(),
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
    fn contains_key_ref<Q: ?Sized + Hash + Eq>(&self, k: &Q) -> bool
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
        // FIXME(#120456) - is `swap_remove` correct?
        FxIndexMap::swap_remove(self, k)
    }

    #[inline(always)]
    fn filter_map_collect<T>(&self, mut f: impl FnMut(&K, &V) -> Option<T>) -> Vec<T> {
        self.iter().filter_map(move |(k, v)| f(k, v)).collect()
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

pub type CompileTimeInterpCx<'tcx> = InterpCx<'tcx, CompileTimeMachine<'tcx>>;

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

impl<'tcx> CompileTimeInterpCx<'tcx> {
    fn location_triple_for_span(&self, span: Span) -> (Symbol, u32, u32) {
        let topmost = span.ctxt().outer_expn().expansion_cause().unwrap_or(span);
        let caller = self.tcx.sess.source_map().lookup_char_pos(topmost.lo());

        use rustc_session::RemapFileNameExt;
        use rustc_session::config::RemapPathScopeComponents;
        (
            Symbol::intern(
                &caller
                    .file
                    .name
                    .for_scope(self.tcx.sess, RemapPathScopeComponents::DIAGNOSTICS)
                    .to_string_lossy(),
            ),
            u32::try_from(caller.line).unwrap(),
            u32::try_from(caller.col_display).unwrap().checked_add(1).unwrap(),
        )
    }

    /// "Intercept" a function call, because we have something special to do for it.
    /// All `#[rustc_do_not_const_check]` functions MUST be hooked here.
    /// If this returns `Some` function, which may be `instance` or a different function with
    /// compatible arguments, then evaluation should continue with that function.
    /// If this returns `None`, the function call has been handled and the function has returned.
    fn hook_special_const_fn(
        &mut self,
        instance: ty::Instance<'tcx>,
        args: &[FnArg<'tcx>],
        _dest: &PlaceTy<'tcx>,
        _ret: Option<mir::BasicBlock>,
    ) -> InterpResult<'tcx, Option<ty::Instance<'tcx>>> {
        let def_id = instance.def_id();

        if self.tcx.has_attr(def_id, sym::rustc_const_panic_str)
            || self.tcx.is_lang_item(def_id, LangItem::BeginPanic)
        {
            let args = self.copy_fn_args(args);
            // &str or &&str
            assert!(args.len() == 1);

            let mut msg_place = self.deref_pointer(&args[0])?;
            while msg_place.layout.ty.is_ref() {
                msg_place = self.deref_pointer(&msg_place)?;
            }

            let msg = Symbol::intern(self.read_str(&msg_place)?);
            let span = self.find_closest_untracked_caller_location();
            let (file, line, col) = self.location_triple_for_span(span);
            return Err(ConstEvalErrKind::Panic { msg, file, line, col }).into();
        } else if self.tcx.is_lang_item(def_id, LangItem::PanicFmt) {
            // For panic_fmt, call const_panic_fmt instead.
            let const_def_id = self.tcx.require_lang_item(LangItem::ConstPanicFmt, self.tcx.span);
            let new_instance = ty::Instance::expect_resolve(
                *self.tcx,
                self.typing_env(),
                const_def_id,
                instance.args,
                self.cur_span(),
            );

            return interp_ok(Some(new_instance));
        }
        interp_ok(Some(instance))
    }

    /// See documentation on the `ptr_guaranteed_cmp` intrinsic.
    /// Returns `2` if the result is unknown.
    /// Returns `1` if the pointers are guaranteed equal.
    /// Returns `0` if the pointers are guaranteed inequal.
    ///
    /// Note that this intrinsic is exposed on stable for comparison with null. In other words, any
    /// change to this function that affects comparison with null is insta-stable!
    fn guaranteed_cmp(&mut self, a: Scalar, b: Scalar) -> InterpResult<'tcx, u8> {
        interp_ok(match (a, b) {
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

impl<'tcx> CompileTimeMachine<'tcx> {
    #[inline(always)]
    /// Find the first stack frame that is within the current crate, if any.
    /// Otherwise, return the crate's HirId
    pub fn best_lint_scope(&self, tcx: TyCtxt<'tcx>) -> hir::HirId {
        self.stack.iter().find_map(|frame| frame.lint_root(tcx)).unwrap_or(CRATE_HIR_ID)
    }
}

impl<'tcx> interpret::Machine<'tcx> for CompileTimeMachine<'tcx> {
    compile_time_machine!(<'tcx>);

    type MemoryKind = MemoryKind;

    const PANIC_ON_ALLOC_FAIL: bool = false; // will be raised as a proper error

    #[inline(always)]
    fn enforce_alignment(ecx: &InterpCx<'tcx, Self>) -> bool {
        matches!(ecx.machine.check_alignment, CheckAlignment::Error)
    }

    #[inline(always)]
    fn enforce_validity(ecx: &InterpCx<'tcx, Self>, layout: TyAndLayout<'tcx>) -> bool {
        ecx.tcx.sess.opts.unstable_opts.extra_const_ub_checks || layout.is_uninhabited()
    }

    fn load_mir(
        ecx: &InterpCx<'tcx, Self>,
        instance: ty::InstanceKind<'tcx>,
    ) -> InterpResult<'tcx, &'tcx mir::Body<'tcx>> {
        match instance {
            ty::InstanceKind::Item(def) => interp_ok(ecx.tcx.mir_for_ctfe(def)),
            _ => interp_ok(ecx.tcx.instance_mir(instance)),
        }
    }

    fn find_mir_or_eval_fn(
        ecx: &mut InterpCx<'tcx, Self>,
        orig_instance: ty::Instance<'tcx>,
        _abi: &FnAbi<'tcx, Ty<'tcx>>,
        args: &[FnArg<'tcx>],
        dest: &PlaceTy<'tcx>,
        ret: Option<mir::BasicBlock>,
        _unwind: mir::UnwindAction, // unwinding is not supported in consts
    ) -> InterpResult<'tcx, Option<(&'tcx mir::Body<'tcx>, ty::Instance<'tcx>)>> {
        debug!("find_mir_or_eval_fn: {:?}", orig_instance);

        // Replace some functions.
        let Some(instance) = ecx.hook_special_const_fn(orig_instance, args, dest, ret)? else {
            // Call has already been handled.
            return interp_ok(None);
        };

        // Only check non-glue functions
        if let ty::InstanceKind::Item(def) = instance.def {
            // Execution might have wandered off into other crates, so we cannot do a stability-
            // sensitive check here. But we can at least rule out functions that are not const at
            // all. That said, we have to allow calling functions inside a trait marked with
            // #[const_trait]. These *are* const-checked!
            if !ecx.tcx.is_const_fn(def) || ecx.tcx.has_attr(def, sym::rustc_do_not_const_check) {
                // We certainly do *not* want to actually call the fn
                // though, so be sure we return here.
                throw_unsup_format!("calling non-const function `{}`", instance)
            }
        }

        // This is a const fn. Call it.
        // In case of replacement, we return the *original* instance to make backtraces work out
        // (and we hope this does not confuse the FnAbi checks too much).
        interp_ok(Some((ecx.load_mir(instance.def, None)?, orig_instance)))
    }

    fn panic_nounwind(ecx: &mut InterpCx<'tcx, Self>, msg: &str) -> InterpResult<'tcx> {
        let msg = Symbol::intern(msg);
        let span = ecx.find_closest_untracked_caller_location();
        let (file, line, col) = ecx.location_triple_for_span(span);
        Err(ConstEvalErrKind::Panic { msg, file, line, col }).into()
    }

    fn call_intrinsic(
        ecx: &mut InterpCx<'tcx, Self>,
        instance: ty::Instance<'tcx>,
        args: &[OpTy<'tcx>],
        dest: &PlaceTy<'tcx, Self::Provenance>,
        target: Option<mir::BasicBlock>,
        _unwind: mir::UnwindAction,
    ) -> InterpResult<'tcx, Option<ty::Instance<'tcx>>> {
        // Shared intrinsics.
        if ecx.eval_intrinsic(instance, args, dest, target)? {
            return interp_ok(None);
        }
        let intrinsic_name = ecx.tcx.item_name(instance.def_id());

        // CTFE-specific intrinsics.
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
                    Size::from_bytes(size),
                    align,
                    interpret::MemoryKind::Machine(MemoryKind::Heap),
                    AllocInit::Uninit,
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
                let (alloc_id, _, _) = ecx.ptr_get_alloc_id(ptr, 0)?;
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
            // The intrinsic represents whether the value is known to the optimizer (LLVM).
            // We're not doing any optimizations here, so there is no optimizer that could know the value.
            // (We know the value here in the machine of course, but this is the runtime of that code,
            // not the optimization stage.)
            sym::is_val_statically_known => ecx.write_scalar(Scalar::from_bool(false), dest)?,
            _ => {
                // We haven't handled the intrinsic, let's see if we can use a fallback body.
                if ecx.tcx.intrinsic(instance.def_id()).unwrap().must_be_overridden {
                    throw_unsup_format!(
                        "intrinsic `{intrinsic_name}` is not supported at compile-time"
                    );
                }
                return interp_ok(Some(ty::Instance {
                    def: ty::InstanceKind::Item(instance.def_id()),
                    args: instance.args,
                }));
            }
        }

        // Intrinsic is done, jump to next block.
        ecx.return_to_block(target)?;
        interp_ok(None)
    }

    fn assert_panic(
        ecx: &mut InterpCx<'tcx, Self>,
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
            ResumedAfterReturn(coroutine_kind) => ResumedAfterReturn(*coroutine_kind),
            ResumedAfterPanic(coroutine_kind) => ResumedAfterPanic(*coroutine_kind),
            ResumedAfterDrop(coroutine_kind) => ResumedAfterDrop(*coroutine_kind),
            MisalignedPointerDereference { required, found } => MisalignedPointerDereference {
                required: eval_to_int(required)?,
                found: eval_to_int(found)?,
            },
            NullPointerDereference => NullPointerDereference,
        };
        Err(ConstEvalErrKind::AssertFailure(err)).into()
    }

    fn binary_ptr_op(
        _ecx: &InterpCx<'tcx, Self>,
        _bin_op: mir::BinOp,
        _left: &ImmTy<'tcx>,
        _right: &ImmTy<'tcx>,
    ) -> InterpResult<'tcx, ImmTy<'tcx>> {
        throw_unsup_format!("pointer arithmetic or comparison is not supported at compile-time");
    }

    fn increment_const_eval_counter(ecx: &mut InterpCx<'tcx, Self>) -> InterpResult<'tcx> {
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
                let hir_id = ecx.machine.best_lint_scope(*ecx.tcx);
                let is_error = ecx
                    .tcx
                    .lint_level_at_node(
                        rustc_session::lint::builtin::LONG_RUNNING_CONST_EVAL,
                        hir_id,
                    )
                    .level
                    .is_error();
                let span = ecx.cur_span();
                ecx.tcx.emit_node_span_lint(
                    rustc_session::lint::builtin::LONG_RUNNING_CONST_EVAL,
                    hir_id,
                    span,
                    LongRunning { item_span: ecx.tcx.span },
                );
                // If this was a hard error, don't bother continuing evaluation.
                if is_error {
                    let guard = ecx
                        .tcx
                        .dcx()
                        .span_delayed_bug(span, "The deny lint should have already errored");
                    throw_inval!(AlreadyReported(ReportedErrorInfo::allowed_in_infallible(guard)));
                }
            } else if new_steps > start && new_steps.is_power_of_two() {
                // Only report after a certain number of terminators have been evaluated and the
                // current number of evaluated terminators is a power of 2. The latter gives us a cheap
                // way to implement exponential backoff.
                let span = ecx.cur_span();
                // We store a unique number in `force_duplicate` to evade `-Z deduplicate-diagnostics`.
                // `new_steps` is guaranteed to be unique because `ecx.machine.num_evaluated_steps` is
                // always increasing.
                ecx.tcx.dcx().emit_warn(LongRunningWarn {
                    span,
                    item_span: ecx.tcx.span,
                    force_duplicate: new_steps,
                });
            }
        }

        interp_ok(())
    }

    #[inline(always)]
    fn expose_provenance(
        _ecx: &InterpCx<'tcx, Self>,
        _provenance: Self::Provenance,
    ) -> InterpResult<'tcx> {
        // This is only reachable with -Zunleash-the-miri-inside-of-you.
        throw_unsup_format!("exposing pointers is not possible at compile-time")
    }

    #[inline(always)]
    fn init_frame(
        ecx: &mut InterpCx<'tcx, Self>,
        frame: Frame<'tcx>,
    ) -> InterpResult<'tcx, Frame<'tcx>> {
        // Enforce stack size limit. Add 1 because this is run before the new frame is pushed.
        if !ecx.recursion_limit.value_within_limit(ecx.stack().len() + 1) {
            throw_exhaust!(StackFrameLimitReached)
        } else {
            interp_ok(frame)
        }
    }

    #[inline(always)]
    fn stack<'a>(
        ecx: &'a InterpCx<'tcx, Self>,
    ) -> &'a [Frame<'tcx, Self::Provenance, Self::FrameExtra>] {
        &ecx.machine.stack
    }

    #[inline(always)]
    fn stack_mut<'a>(
        ecx: &'a mut InterpCx<'tcx, Self>,
    ) -> &'a mut Vec<Frame<'tcx, Self::Provenance, Self::FrameExtra>> {
        &mut ecx.machine.stack
    }

    fn before_access_global(
        _tcx: TyCtxtAt<'tcx>,
        machine: &Self,
        alloc_id: AllocId,
        alloc: ConstAllocation<'tcx>,
        _static_def_id: Option<DefId>,
        is_write: bool,
    ) -> InterpResult<'tcx> {
        let alloc = alloc.inner();
        if is_write {
            // Write access. These are never allowed, but we give a targeted error message.
            match alloc.mutability {
                Mutability::Not => throw_ub!(WriteToReadOnly(alloc_id)),
                Mutability::Mut => Err(ConstEvalErrKind::ModifiedGlobal).into(),
            }
        } else {
            // Read access. These are usually allowed, with some exceptions.
            if machine.can_access_mut_global == CanAccessMutGlobal::Yes {
                // Machine configuration allows us read from anything (e.g., `static` initializer).
                interp_ok(())
            } else if alloc.mutability == Mutability::Mut {
                // Machine configuration does not allow us to read statics (e.g., `const`
                // initializer).
                Err(ConstEvalErrKind::ConstAccessesMutGlobal).into()
            } else {
                // Immutable global, this read is fine.
                assert_eq!(alloc.mutability, Mutability::Not);
                interp_ok(())
            }
        }
    }

    fn retag_ptr_value(
        ecx: &mut InterpCx<'tcx, Self>,
        _kind: mir::RetagKind,
        val: &ImmTy<'tcx, CtfeProvenance>,
    ) -> InterpResult<'tcx, ImmTy<'tcx, CtfeProvenance>> {
        // If it's a frozen shared reference that's not already immutable, potentially make it immutable.
        // (Do nothing on `None` provenance, that cannot store immutability anyway.)
        if let ty::Ref(_, ty, mutbl) = val.layout.ty.kind()
            && *mutbl == Mutability::Not
            && val
                .to_scalar_and_meta()
                .0
                .to_pointer(ecx)?
                .provenance
                .is_some_and(|p| !p.immutable())
        {
            // That next check is expensive, that's why we have all the guards above.
            let is_immutable = ty.is_freeze(*ecx.tcx, ecx.typing_env());
            let place = ecx.ref_to_mplace(val)?;
            let new_place = if is_immutable {
                place.map_provenance(CtfeProvenance::as_immutable)
            } else {
                // Even if it is not immutable, remember that it is a shared reference.
                // This allows it to become part of the final value of the constant.
                // (See <https://github.com/rust-lang/rust/pull/128543> for why we allow this
                // even when there is interior mutability.)
                place.map_provenance(CtfeProvenance::as_shared_ref)
            };
            interp_ok(ImmTy::from_immediate(new_place.to_ref(ecx), val.layout))
        } else {
            interp_ok(val.clone())
        }
    }

    fn before_memory_write(
        _tcx: TyCtxtAt<'tcx>,
        _machine: &mut Self,
        _alloc_extra: &mut Self::AllocExtra,
        _ptr: Pointer<Option<Self::Provenance>>,
        (_alloc_id, immutable): (AllocId, bool),
        range: AllocRange,
    ) -> InterpResult<'tcx> {
        if range.size == Size::ZERO {
            // Nothing to check.
            return interp_ok(());
        }
        // Reject writes through immutable pointers.
        if immutable {
            return Err(ConstEvalErrKind::WriteThroughImmutablePointer).into();
        }
        // Everything else is fine.
        interp_ok(())
    }

    fn before_alloc_read(ecx: &InterpCx<'tcx, Self>, alloc_id: AllocId) -> InterpResult<'tcx> {
        // Check if this is the currently evaluated static.
        if Some(alloc_id) == ecx.machine.static_root_ids.map(|(id, _)| id) {
            return Err(ConstEvalErrKind::RecursiveStatic).into();
        }
        // If this is another static, make sure we fire off the query to detect cycles.
        // But only do that when checks for static recursion are enabled.
        if ecx.machine.static_root_ids.is_some() {
            if let Some(GlobalAlloc::Static(def_id)) = ecx.tcx.try_get_global_alloc(alloc_id) {
                if ecx.tcx.is_foreign_item(def_id) {
                    throw_unsup!(ExternStatic(def_id));
                }
                ecx.ctfe_query(|tcx| tcx.eval_static_initializer(def_id))?;
            }
        }
        interp_ok(())
    }

    fn cached_union_data_range<'e>(
        ecx: &'e mut InterpCx<'tcx, Self>,
        ty: Ty<'tcx>,
        compute_range: impl FnOnce() -> RangeSet,
    ) -> Cow<'e, RangeSet> {
        if ecx.tcx.sess.opts.unstable_opts.extra_const_ub_checks {
            Cow::Borrowed(ecx.machine.union_data_ranges.entry(ty).or_insert_with(compute_range))
        } else {
            // Don't bother caching, we're only doing one validation at the end anyway.
            Cow::Owned(compute_range())
        }
    }

    fn get_default_alloc_params(&self) -> <Self::Bytes as mir::interpret::AllocBytes>::AllocParams {
    }
}

// Please do not add any code below the above `Machine` trait impl. I (oli-obk) plan more cleanups
// so we can end up having a file with just that impl, but for now, let's keep the impl discoverable
// at the bottom of this file.
