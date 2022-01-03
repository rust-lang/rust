use rustc_middle::mir;
use rustc_middle::ty::{self, Ty};
use std::borrow::Borrow;
use std::collections::hash_map::Entry;
use std::hash::Hash;

use rustc_data_structures::fx::FxHashMap;
use std::fmt;

use rustc_ast::Mutability;
use rustc_hir::def_id::DefId;
use rustc_middle::mir::AssertMessage;
use rustc_session::Limit;
use rustc_span::symbol::{sym, Symbol};
use rustc_target::abi::{Align, Size};
use rustc_target::spec::abi::Abi;

use crate::interpret::{
    self, compile_time_machine, AllocId, Allocation, Frame, ImmTy, InterpCx, InterpResult, OpTy,
    PlaceTy, Scalar, StackPopUnwind,
};

use super::error::*;

impl<'mir, 'tcx> InterpCx<'mir, 'tcx, CompileTimeInterpreter<'mir, 'tcx>> {
    /// "Intercept" a function call to a panic-related function
    /// because we have something special to do for it.
    /// If this returns successfully (`Ok`), the function should just be evaluated normally.
    fn hook_special_const_fn(
        &mut self,
        instance: ty::Instance<'tcx>,
        args: &[OpTy<'tcx>],
    ) -> InterpResult<'tcx, Option<ty::Instance<'tcx>>> {
        // All `#[rustc_do_not_const_check]` functions should be hooked here.
        let def_id = instance.def_id();

        if Some(def_id) == self.tcx.lang_items().const_eval_select() {
            // redirect to const_eval_select_ct
            if let Some(const_eval_select) = self.tcx.lang_items().const_eval_select_ct() {
                return Ok(Some(
                    ty::Instance::resolve(
                        *self.tcx,
                        ty::ParamEnv::reveal_all(),
                        const_eval_select,
                        instance.substs,
                    )
                    .unwrap()
                    .unwrap(),
                ));
            }
        } else if Some(def_id) == self.tcx.lang_items().panic_display()
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
            if let Some(const_panic_fmt) = self.tcx.lang_items().const_panic_fmt() {
                return Ok(Some(
                    ty::Instance::resolve(
                        *self.tcx,
                        ty::ParamEnv::reveal_all(),
                        const_panic_fmt,
                        self.tcx.intern_substs(&[]),
                    )
                    .unwrap()
                    .unwrap(),
                ));
            }
        }
        Ok(None)
    }
}

/// Extra machine state for CTFE, and the Machine instance
pub struct CompileTimeInterpreter<'mir, 'tcx> {
    /// For now, the number of terminators that can be evaluated before we throw a resource
    /// exhaustion error.
    ///
    /// Setting this to `0` disables the limit and allows the interpreter to run forever.
    pub steps_remaining: usize,

    /// The virtual call stack.
    pub(crate) stack: Vec<Frame<'mir, 'tcx, AllocId, ()>>,
}

#[derive(Copy, Clone, Debug)]
pub struct MemoryExtra {
    /// We need to make sure consts never point to anything mutable, even recursively. That is
    /// relied on for pattern matching on consts with references.
    /// To achieve this, two pieces have to work together:
    /// * Interning makes everything outside of statics immutable.
    /// * Pointers to allocations inside of statics can never leak outside, to a non-static global.
    /// This boolean here controls the second part.
    pub(super) can_access_statics: bool,
}

impl<'mir, 'tcx> CompileTimeInterpreter<'mir, 'tcx> {
    pub(super) fn new(const_eval_limit: Limit) -> Self {
        CompileTimeInterpreter { steps_remaining: const_eval_limit.0, stack: Vec::new() }
    }
}

impl<K: Hash + Eq, V> interpret::AllocMap<K, V> for FxHashMap<K, V> {
    #[inline(always)]
    fn contains_key<Q: ?Sized + Hash + Eq>(&mut self, k: &Q) -> bool
    where
        K: Borrow<Q>,
    {
        FxHashMap::contains_key(self, k)
    }

    #[inline(always)]
    fn insert(&mut self, k: K, v: V) -> Option<V> {
        FxHashMap::insert(self, k, v)
    }

    #[inline(always)]
    fn remove<Q: ?Sized + Hash + Eq>(&mut self, k: &Q) -> Option<V>
    where
        K: Borrow<Q>,
    {
        FxHashMap::remove(self, k)
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
            Entry::Occupied(e) => Ok(e.into_mut()),
            Entry::Vacant(e) => {
                let v = vacant()?;
                Ok(e.insert(v))
            }
        }
    }
}

crate type CompileTimeEvalContext<'mir, 'tcx> =
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
    fn guaranteed_eq(&mut self, a: Scalar, b: Scalar) -> bool {
        match (a, b) {
            // Comparisons between integers are always known.
            (Scalar::Int { .. }, Scalar::Int { .. }) => a == b,
            // Equality with integers can never be known for sure.
            (Scalar::Int { .. }, Scalar::Ptr(..)) | (Scalar::Ptr(..), Scalar::Int { .. }) => false,
            // FIXME: return `true` for when both sides are the same pointer, *except* that
            // some things (like functions and vtables) do not have stable addresses
            // so we need to be careful around them (see e.g. #73722).
            (Scalar::Ptr(..), Scalar::Ptr(..)) => false,
        }
    }

    fn guaranteed_ne(&mut self, a: Scalar, b: Scalar) -> bool {
        match (a, b) {
            // Comparisons between integers are always known.
            (Scalar::Int(_), Scalar::Int(_)) => a != b,
            // Comparisons of abstract pointers with null pointers are known if the pointer
            // is in bounds, because if they are in bounds, the pointer can't be null.
            // Inequality with integers other than null can never be known for sure.
            (Scalar::Int(int), Scalar::Ptr(ptr, _)) | (Scalar::Ptr(ptr, _), Scalar::Int(int)) => {
                int.is_null() && !self.memory.ptr_may_be_null(ptr.into())
            }
            // FIXME: return `true` for at least some comparisons where we can reliably
            // determine the result of runtime inequality tests at compile-time.
            // Examples include comparison of addresses in different static items.
            (Scalar::Ptr(..), Scalar::Ptr(..)) => false,
        }
    }
}

impl<'mir, 'tcx> interpret::Machine<'mir, 'tcx> for CompileTimeInterpreter<'mir, 'tcx> {
    compile_time_machine!(<'mir, 'tcx>);

    type MemoryKind = MemoryKind;

    type MemoryExtra = MemoryExtra;

    const PANIC_ON_ALLOC_FAIL: bool = false; // will be raised as a proper error

    fn load_mir(
        ecx: &InterpCx<'mir, 'tcx, Self>,
        instance: ty::InstanceDef<'tcx>,
    ) -> InterpResult<'tcx, &'tcx mir::Body<'tcx>> {
        match instance {
            ty::InstanceDef::Item(def) => {
                if ecx.tcx.is_ctfe_mir_available(def.did) {
                    Ok(ecx.tcx.mir_for_ctfe_opt_const_arg(def))
                } else {
                    let path = ecx.tcx.def_path_str(def.did);
                    Err(ConstEvalErrKind::NeedsRfc(format!("calling extern function `{}`", path))
                        .into())
                }
            }
            _ => Ok(ecx.tcx.instance_mir(instance)),
        }
    }

    fn find_mir_or_eval_fn(
        ecx: &mut InterpCx<'mir, 'tcx, Self>,
        instance: ty::Instance<'tcx>,
        _abi: Abi,
        args: &[OpTy<'tcx>],
        _ret: Option<(&PlaceTy<'tcx>, mir::BasicBlock)>,
        _unwind: StackPopUnwind, // unwinding is not supported in consts
    ) -> InterpResult<'tcx, Option<(&'mir mir::Body<'tcx>, ty::Instance<'tcx>)>> {
        debug!("find_mir_or_eval_fn: {:?}", instance);

        // Only check non-glue functions
        if let ty::InstanceDef::Item(def) = instance.def {
            // Execution might have wandered off into other crates, so we cannot do a stability-
            // sensitive check here.  But we can at least rule out functions that are not const
            // at all.
            if !ecx.tcx.is_const_fn_raw(def.did) {
                // allow calling functions marked with #[default_method_body_is_const].
                if !ecx.tcx.has_attr(def.did, sym::default_method_body_is_const) {
                    // We certainly do *not* want to actually call the fn
                    // though, so be sure we return here.
                    throw_unsup_format!("calling non-const function `{}`", instance)
                }
            }

            if let Some(new_instance) = ecx.hook_special_const_fn(instance, args)? {
                // We call another const fn instead.
                // However, we return the *original* instance to make backtraces work out
                // (and we hope this does not confuse the FnAbi checks too much).
                return Ok(Self::find_mir_or_eval_fn(
                    ecx,
                    new_instance,
                    _abi,
                    args,
                    _ret,
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
        ret: Option<(&PlaceTy<'tcx>, mir::BasicBlock)>,
        _unwind: StackPopUnwind,
    ) -> InterpResult<'tcx> {
        // Shared intrinsics.
        if ecx.emulate_intrinsic(instance, args, ret)? {
            return Ok(());
        }
        let intrinsic_name = ecx.tcx.item_name(instance.def_id());

        // CTFE-specific intrinsics.
        let (dest, ret) = match ret {
            None => {
                return Err(ConstEvalErrKind::NeedsRfc(format!(
                    "calling intrinsic `{}`",
                    intrinsic_name
                ))
                .into());
            }
            Some(p) => p,
        };
        match intrinsic_name {
            sym::ptr_guaranteed_eq | sym::ptr_guaranteed_ne => {
                let a = ecx.read_immediate(&args[0])?.to_scalar()?;
                let b = ecx.read_immediate(&args[1])?.to_scalar()?;
                let cmp = if intrinsic_name == sym::ptr_guaranteed_eq {
                    ecx.guaranteed_eq(a, b)
                } else {
                    ecx.guaranteed_ne(a, b)
                };
                ecx.write_scalar(Scalar::from_bool(cmp), dest)?;
            }
            sym::const_allocate => {
                let size = ecx.read_scalar(&args[0])?.to_machine_usize(ecx)?;
                let align = ecx.read_scalar(&args[1])?.to_machine_usize(ecx)?;

                let align = match Align::from_bytes(align) {
                    Ok(a) => a,
                    Err(err) => throw_ub_format!("align has to be a power of 2, {}", err),
                };

                let ptr = ecx.memory.allocate(
                    Size::from_bytes(size as u64),
                    align,
                    interpret::MemoryKind::Machine(MemoryKind::Heap),
                )?;
                ecx.write_pointer(ptr, dest)?;
            }
            _ => {
                return Err(ConstEvalErrKind::NeedsRfc(format!(
                    "calling intrinsic `{}`",
                    intrinsic_name
                ))
                .into());
            }
        }

        ecx.go_to_block(ret);
        Ok(())
    }

    fn assert_panic(
        ecx: &mut InterpCx<'mir, 'tcx, Self>,
        msg: &AssertMessage<'tcx>,
        _unwind: Option<mir::BasicBlock>,
    ) -> InterpResult<'tcx> {
        use rustc_middle::mir::AssertKind::*;
        // Convert `AssertKind<Operand>` to `AssertKind<Scalar>`.
        let eval_to_int =
            |op| ecx.read_immediate(&ecx.eval_operand(op, None)?).map(|x| x.to_const_int());
        let err = match msg {
            BoundsCheck { ref len, ref index } => {
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
        Err(ConstEvalErrKind::NeedsRfc("pointer arithmetic or comparison".to_string()).into())
    }

    fn before_terminator(ecx: &mut InterpCx<'mir, 'tcx, Self>) -> InterpResult<'tcx> {
        // The step limit has already been hit in a previous call to `before_terminator`.
        if ecx.machine.steps_remaining == 0 {
            return Ok(());
        }

        ecx.machine.steps_remaining -= 1;
        if ecx.machine.steps_remaining == 0 {
            throw_exhaust!(StepLimitReached)
        }

        Ok(())
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
    ) -> &'a [Frame<'mir, 'tcx, Self::PointerTag, Self::FrameExtra>] {
        &ecx.machine.stack
    }

    #[inline(always)]
    fn stack_mut<'a>(
        ecx: &'a mut InterpCx<'mir, 'tcx, Self>,
    ) -> &'a mut Vec<Frame<'mir, 'tcx, Self::PointerTag, Self::FrameExtra>> {
        &mut ecx.machine.stack
    }

    fn before_access_global(
        memory_extra: &MemoryExtra,
        alloc_id: AllocId,
        allocation: &Allocation,
        static_def_id: Option<DefId>,
        is_write: bool,
    ) -> InterpResult<'tcx> {
        if is_write {
            // Write access. These are never allowed, but we give a targeted error message.
            if allocation.mutability == Mutability::Not {
                Err(err_ub!(WriteToReadOnly(alloc_id)).into())
            } else {
                Err(ConstEvalErrKind::ModifiedGlobal.into())
            }
        } else {
            // Read access. These are usually allowed, with some exceptions.
            if memory_extra.can_access_statics {
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
                assert_eq!(allocation.mutability, Mutability::Not);
                Ok(())
            }
        }
    }
}

// Please do not add any code below the above `Machine` trait impl. I (oli-obk) plan more cleanups
// so we can end up having a file with just that impl, but for now, let's keep the impl discoverable
// at the bottom of this file.
