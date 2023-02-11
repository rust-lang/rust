//! This module contains everything needed to instantiate an interpreter.
//! This separation exists to ensure that no fancy miri features like
//! interpreting common C functions leak into CTFE.

use std::borrow::{Borrow, Cow};
use std::fmt::Debug;
use std::hash::Hash;

use rustc_ast::{InlineAsmOptions, InlineAsmTemplatePiece};
use rustc_middle::mir;
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_span::def_id::DefId;
use rustc_target::abi::{Align, Size};
use rustc_target::spec::abi::Abi as CallAbi;

use crate::const_eval::CheckAlignment;

use super::{
    AllocId, AllocRange, Allocation, ConstAllocation, Frame, ImmTy, InterpCx, InterpResult,
    MemoryKind, OpTy, Operand, PlaceTy, Pointer, Provenance, Scalar, StackPopUnwind,
};

/// Data returned by Machine::stack_pop,
/// to provide further control over the popping of the stack frame
#[derive(Eq, PartialEq, Debug, Copy, Clone)]
pub enum StackPopJump {
    /// Indicates that no special handling should be
    /// done - we'll either return normally or unwind
    /// based on the terminator for the function
    /// we're leaving.
    Normal,

    /// Indicates that we should *not* jump to the return/unwind address, as the callback already
    /// took care of everything.
    NoJump,
}

/// Whether this kind of memory is allowed to leak
pub trait MayLeak: Copy {
    fn may_leak(self) -> bool;
}

/// The functionality needed by memory to manage its allocations
pub trait AllocMap<K: Hash + Eq, V> {
    /// Tests if the map contains the given key.
    /// Deliberately takes `&mut` because that is sufficient, and some implementations
    /// can be more efficient then (using `RefCell::get_mut`).
    fn contains_key<Q: ?Sized + Hash + Eq>(&mut self, k: &Q) -> bool
    where
        K: Borrow<Q>;

    /// Inserts a new entry into the map.
    fn insert(&mut self, k: K, v: V) -> Option<V>;

    /// Removes an entry from the map.
    fn remove<Q: ?Sized + Hash + Eq>(&mut self, k: &Q) -> Option<V>
    where
        K: Borrow<Q>;

    /// Returns data based on the keys and values in the map.
    fn filter_map_collect<T>(&self, f: impl FnMut(&K, &V) -> Option<T>) -> Vec<T>;

    /// Returns a reference to entry `k`. If no such entry exists, call
    /// `vacant` and either forward its error, or add its result to the map
    /// and return a reference to *that*.
    fn get_or<E>(&self, k: K, vacant: impl FnOnce() -> Result<V, E>) -> Result<&V, E>;

    /// Returns a mutable reference to entry `k`. If no such entry exists, call
    /// `vacant` and either forward its error, or add its result to the map
    /// and return a reference to *that*.
    fn get_mut_or<E>(&mut self, k: K, vacant: impl FnOnce() -> Result<V, E>) -> Result<&mut V, E>;

    /// Read-only lookup.
    fn get(&self, k: K) -> Option<&V> {
        self.get_or(k, || Err(())).ok()
    }

    /// Mutable lookup.
    fn get_mut(&mut self, k: K) -> Option<&mut V> {
        self.get_mut_or(k, || Err(())).ok()
    }
}

/// Methods of this trait signifies a point where CTFE evaluation would fail
/// and some use case dependent behaviour can instead be applied.
pub trait Machine<'mir, 'tcx>: Sized {
    /// Additional memory kinds a machine wishes to distinguish from the builtin ones
    type MemoryKind: Debug + std::fmt::Display + MayLeak + Eq + 'static;

    /// Pointers are "tagged" with provenance information; typically the `AllocId` they belong to.
    type Provenance: Provenance + Eq + Hash + 'static;

    /// When getting the AllocId of a pointer, some extra data is also obtained from the provenance
    /// that is passed to memory access hooks so they can do things with it.
    type ProvenanceExtra: Copy + 'static;

    /// Machines can define extra (non-instance) things that represent values of function pointers.
    /// For example, Miri uses this to return a function pointer from `dlsym`
    /// that can later be called to execute the right thing.
    type ExtraFnVal: Debug + Copy;

    /// Extra data stored in every call frame.
    type FrameExtra;

    /// Extra data stored in every allocation.
    type AllocExtra: Debug + Clone + 'static;

    /// Memory's allocation map
    type MemoryMap: AllocMap<
            AllocId,
            (MemoryKind<Self::MemoryKind>, Allocation<Self::Provenance, Self::AllocExtra>),
        > + Default
        + Clone;

    /// The memory kind to use for copied global memory (held in `tcx`) --
    /// or None if such memory should not be mutated and thus any such attempt will cause
    /// a `ModifiedStatic` error to be raised.
    /// Statics are copied under two circumstances: When they are mutated, and when
    /// `adjust_allocation` (see below) returns an owned allocation
    /// that is added to the memory so that the work is not done twice.
    const GLOBAL_KIND: Option<Self::MemoryKind>;

    /// Should the machine panic on allocation failures?
    const PANIC_ON_ALLOC_FAIL: bool;

    /// Whether memory accesses should be alignment-checked.
    fn enforce_alignment(ecx: &InterpCx<'mir, 'tcx, Self>) -> CheckAlignment;

    /// Whether, when checking alignment, we should look at the actual address and thus support
    /// custom alignment logic based on whatever the integer address happens to be.
    ///
    /// If this returns true, Provenance::OFFSET_IS_ADDR must be true.
    fn use_addr_for_alignment_check(ecx: &InterpCx<'mir, 'tcx, Self>) -> bool;

    fn alignment_check_failed(
        ecx: &InterpCx<'mir, 'tcx, Self>,
        has: Align,
        required: Align,
        check: CheckAlignment,
    ) -> InterpResult<'tcx, ()>;

    /// Whether to enforce the validity invariant
    fn enforce_validity(ecx: &InterpCx<'mir, 'tcx, Self>) -> bool;

    /// Whether function calls should be [ABI](CallAbi)-checked.
    fn enforce_abi(_ecx: &InterpCx<'mir, 'tcx, Self>) -> bool {
        true
    }

    /// Whether CheckedBinOp MIR statements should actually check for overflow.
    fn checked_binop_checks_overflow(_ecx: &InterpCx<'mir, 'tcx, Self>) -> bool;

    /// Entry point for obtaining the MIR of anything that should get evaluated.
    /// So not just functions and shims, but also const/static initializers, anonymous
    /// constants, ...
    fn load_mir(
        ecx: &InterpCx<'mir, 'tcx, Self>,
        instance: ty::InstanceDef<'tcx>,
    ) -> InterpResult<'tcx, &'tcx mir::Body<'tcx>> {
        Ok(ecx.tcx.instance_mir(instance))
    }

    /// Entry point to all function calls.
    ///
    /// Returns either the mir to use for the call, or `None` if execution should
    /// just proceed (which usually means this hook did all the work that the
    /// called function should usually have done). In the latter case, it is
    /// this hook's responsibility to advance the instruction pointer!
    /// (This is to support functions like `__rust_maybe_catch_panic` that neither find a MIR
    /// nor just jump to `ret`, but instead push their own stack frame.)
    /// Passing `dest`and `ret` in the same `Option` proved very annoying when only one of them
    /// was used.
    fn find_mir_or_eval_fn(
        ecx: &mut InterpCx<'mir, 'tcx, Self>,
        instance: ty::Instance<'tcx>,
        abi: CallAbi,
        args: &[OpTy<'tcx, Self::Provenance>],
        destination: &PlaceTy<'tcx, Self::Provenance>,
        target: Option<mir::BasicBlock>,
        unwind: StackPopUnwind,
    ) -> InterpResult<'tcx, Option<(&'mir mir::Body<'tcx>, ty::Instance<'tcx>)>>;

    /// Execute `fn_val`. It is the hook's responsibility to advance the instruction
    /// pointer as appropriate.
    fn call_extra_fn(
        ecx: &mut InterpCx<'mir, 'tcx, Self>,
        fn_val: Self::ExtraFnVal,
        abi: CallAbi,
        args: &[OpTy<'tcx, Self::Provenance>],
        destination: &PlaceTy<'tcx, Self::Provenance>,
        target: Option<mir::BasicBlock>,
        unwind: StackPopUnwind,
    ) -> InterpResult<'tcx>;

    /// Directly process an intrinsic without pushing a stack frame. It is the hook's
    /// responsibility to advance the instruction pointer as appropriate.
    fn call_intrinsic(
        ecx: &mut InterpCx<'mir, 'tcx, Self>,
        instance: ty::Instance<'tcx>,
        args: &[OpTy<'tcx, Self::Provenance>],
        destination: &PlaceTy<'tcx, Self::Provenance>,
        target: Option<mir::BasicBlock>,
        unwind: StackPopUnwind,
    ) -> InterpResult<'tcx>;

    /// Called to evaluate `Assert` MIR terminators that trigger a panic.
    fn assert_panic(
        ecx: &mut InterpCx<'mir, 'tcx, Self>,
        msg: &mir::AssertMessage<'tcx>,
        unwind: Option<mir::BasicBlock>,
    ) -> InterpResult<'tcx>;

    /// Called to evaluate `Abort` MIR terminator.
    fn abort(_ecx: &mut InterpCx<'mir, 'tcx, Self>, _msg: String) -> InterpResult<'tcx, !> {
        throw_unsup_format!("aborting execution is not supported")
    }

    /// Called for all binary operations where the LHS has pointer type.
    ///
    /// Returns a (value, overflowed) pair if the operation succeeded
    fn binary_ptr_op(
        ecx: &InterpCx<'mir, 'tcx, Self>,
        bin_op: mir::BinOp,
        left: &ImmTy<'tcx, Self::Provenance>,
        right: &ImmTy<'tcx, Self::Provenance>,
    ) -> InterpResult<'tcx, (Scalar<Self::Provenance>, bool, Ty<'tcx>)>;

    /// Called to write the specified `local` from the `frame`.
    /// Since writing a ZST is not actually accessing memory or locals, this is never invoked
    /// for ZST reads.
    ///
    /// Due to borrow checker trouble, we indicate the `frame` as an index rather than an `&mut
    /// Frame`.
    #[inline]
    fn access_local_mut<'a>(
        ecx: &'a mut InterpCx<'mir, 'tcx, Self>,
        frame: usize,
        local: mir::Local,
    ) -> InterpResult<'tcx, &'a mut Operand<Self::Provenance>>
    where
        'tcx: 'mir,
    {
        ecx.stack_mut()[frame].locals[local].access_mut()
    }

    /// Called before a basic block terminator is executed.
    #[inline]
    fn before_terminator(_ecx: &mut InterpCx<'mir, 'tcx, Self>) -> InterpResult<'tcx> {
        Ok(())
    }

    /// Called when the interpreter encounters a `StatementKind::ConstEvalCounter` instruction.
    /// You can use this to detect long or endlessly running programs.
    #[inline]
    fn increment_const_eval_counter(_ecx: &mut InterpCx<'mir, 'tcx, Self>) -> InterpResult<'tcx> {
        Ok(())
    }

    /// Called before a global allocation is accessed.
    /// `def_id` is `Some` if this is the "lazy" allocation of a static.
    #[inline]
    fn before_access_global(
        _tcx: TyCtxt<'tcx>,
        _machine: &Self,
        _alloc_id: AllocId,
        _allocation: ConstAllocation<'tcx>,
        _static_def_id: Option<DefId>,
        _is_write: bool,
    ) -> InterpResult<'tcx> {
        Ok(())
    }

    /// Return the `AllocId` for the given thread-local static in the current thread.
    fn thread_local_static_base_pointer(
        _ecx: &mut InterpCx<'mir, 'tcx, Self>,
        def_id: DefId,
    ) -> InterpResult<'tcx, Pointer<Self::Provenance>> {
        throw_unsup!(ThreadLocalStatic(def_id))
    }

    /// Return the root pointer for the given `extern static`.
    fn extern_static_base_pointer(
        ecx: &InterpCx<'mir, 'tcx, Self>,
        def_id: DefId,
    ) -> InterpResult<'tcx, Pointer<Self::Provenance>>;

    /// Return a "base" pointer for the given allocation: the one that is used for direct
    /// accesses to this static/const/fn allocation, or the one returned from the heap allocator.
    ///
    /// Not called on `extern` or thread-local statics (those use the methods above).
    fn adjust_alloc_base_pointer(
        ecx: &InterpCx<'mir, 'tcx, Self>,
        ptr: Pointer,
    ) -> InterpResult<'tcx, Pointer<Self::Provenance>>;

    /// "Int-to-pointer cast"
    fn ptr_from_addr_cast(
        ecx: &InterpCx<'mir, 'tcx, Self>,
        addr: u64,
    ) -> InterpResult<'tcx, Pointer<Option<Self::Provenance>>>;

    /// Marks a pointer as exposed, allowing it's provenance
    /// to be recovered. "Pointer-to-int cast"
    fn expose_ptr(
        ecx: &mut InterpCx<'mir, 'tcx, Self>,
        ptr: Pointer<Self::Provenance>,
    ) -> InterpResult<'tcx>;

    /// Convert a pointer with provenance into an allocation-offset pair
    /// and extra provenance info.
    ///
    /// The returned `AllocId` must be the same as `ptr.provenance.get_alloc_id()`.
    ///
    /// When this fails, that means the pointer does not point to a live allocation.
    fn ptr_get_alloc(
        ecx: &InterpCx<'mir, 'tcx, Self>,
        ptr: Pointer<Self::Provenance>,
    ) -> Option<(AllocId, Size, Self::ProvenanceExtra)>;

    /// Called to adjust allocations to the Provenance and AllocExtra of this machine.
    ///
    /// The way we construct allocations is to always first construct it without extra and then add
    /// the extra. This keeps uniform code paths for handling both allocations created by CTFE for
    /// globals, and allocations created by Miri during evaluation.
    ///
    /// `kind` is the kind of the allocation being adjusted; it can be `None` when
    /// it's a global and `GLOBAL_KIND` is `None`.
    ///
    /// This should avoid copying if no work has to be done! If this returns an owned
    /// allocation (because a copy had to be done to adjust things), machine memory will
    /// cache the result. (This relies on `AllocMap::get_or` being able to add the
    /// owned allocation to the map even when the map is shared.)
    ///
    /// This must only fail if `alloc` contains provenance.
    fn adjust_allocation<'b>(
        ecx: &InterpCx<'mir, 'tcx, Self>,
        id: AllocId,
        alloc: Cow<'b, Allocation>,
        kind: Option<MemoryKind<Self::MemoryKind>>,
    ) -> InterpResult<'tcx, Cow<'b, Allocation<Self::Provenance, Self::AllocExtra>>>;

    fn eval_inline_asm(
        _ecx: &mut InterpCx<'mir, 'tcx, Self>,
        _template: &'tcx [InlineAsmTemplatePiece],
        _operands: &[mir::InlineAsmOperand<'tcx>],
        _options: InlineAsmOptions,
    ) -> InterpResult<'tcx> {
        throw_unsup_format!("inline assembly is not supported")
    }

    /// Hook for performing extra checks on a memory read access.
    ///
    /// Takes read-only access to the allocation so we can keep all the memory read
    /// operations take `&self`. Use a `RefCell` in `AllocExtra` if you
    /// need to mutate.
    #[inline(always)]
    fn before_memory_read(
        _tcx: TyCtxt<'tcx>,
        _machine: &Self,
        _alloc_extra: &Self::AllocExtra,
        _prov: (AllocId, Self::ProvenanceExtra),
        _range: AllocRange,
    ) -> InterpResult<'tcx> {
        Ok(())
    }

    /// Hook for performing extra checks on a memory write access.
    #[inline(always)]
    fn before_memory_write(
        _tcx: TyCtxt<'tcx>,
        _machine: &mut Self,
        _alloc_extra: &mut Self::AllocExtra,
        _prov: (AllocId, Self::ProvenanceExtra),
        _range: AllocRange,
    ) -> InterpResult<'tcx> {
        Ok(())
    }

    /// Hook for performing extra operations on a memory deallocation.
    #[inline(always)]
    fn before_memory_deallocation(
        _tcx: TyCtxt<'tcx>,
        _machine: &mut Self,
        _alloc_extra: &mut Self::AllocExtra,
        _prov: (AllocId, Self::ProvenanceExtra),
        _range: AllocRange,
    ) -> InterpResult<'tcx> {
        Ok(())
    }

    /// Executes a retagging operation for a single pointer.
    /// Returns the possibly adjusted pointer.
    #[inline]
    fn retag_ptr_value(
        _ecx: &mut InterpCx<'mir, 'tcx, Self>,
        _kind: mir::RetagKind,
        val: &ImmTy<'tcx, Self::Provenance>,
    ) -> InterpResult<'tcx, ImmTy<'tcx, Self::Provenance>> {
        Ok(val.clone())
    }

    /// Executes a retagging operation on a compound value.
    /// Replaces all pointers stored in the given place.
    #[inline]
    fn retag_place_contents(
        _ecx: &mut InterpCx<'mir, 'tcx, Self>,
        _kind: mir::RetagKind,
        _place: &PlaceTy<'tcx, Self::Provenance>,
    ) -> InterpResult<'tcx> {
        Ok(())
    }

    /// Called immediately before a new stack frame gets pushed.
    fn init_frame_extra(
        ecx: &mut InterpCx<'mir, 'tcx, Self>,
        frame: Frame<'mir, 'tcx, Self::Provenance>,
    ) -> InterpResult<'tcx, Frame<'mir, 'tcx, Self::Provenance, Self::FrameExtra>>;

    /// Borrow the current thread's stack.
    fn stack<'a>(
        ecx: &'a InterpCx<'mir, 'tcx, Self>,
    ) -> &'a [Frame<'mir, 'tcx, Self::Provenance, Self::FrameExtra>];

    /// Mutably borrow the current thread's stack.
    fn stack_mut<'a>(
        ecx: &'a mut InterpCx<'mir, 'tcx, Self>,
    ) -> &'a mut Vec<Frame<'mir, 'tcx, Self::Provenance, Self::FrameExtra>>;

    /// Called immediately after a stack frame got pushed and its locals got initialized.
    fn after_stack_push(_ecx: &mut InterpCx<'mir, 'tcx, Self>) -> InterpResult<'tcx> {
        Ok(())
    }

    /// Called immediately after a stack frame got popped, but before jumping back to the caller.
    /// The `locals` have already been destroyed!
    fn after_stack_pop(
        _ecx: &mut InterpCx<'mir, 'tcx, Self>,
        _frame: Frame<'mir, 'tcx, Self::Provenance, Self::FrameExtra>,
        unwinding: bool,
    ) -> InterpResult<'tcx, StackPopJump> {
        // By default, we do not support unwinding from panics
        assert!(!unwinding);
        Ok(StackPopJump::Normal)
    }
}

/// A lot of the flexibility above is just needed for `Miri`, but all "compile-time" machines
/// (CTFE and ConstProp) use the same instance. Here, we share that code.
pub macro compile_time_machine(<$mir: lifetime, $tcx: lifetime>) {
    type Provenance = AllocId;
    type ProvenanceExtra = ();

    type ExtraFnVal = !;

    type MemoryMap =
        rustc_data_structures::fx::FxIndexMap<AllocId, (MemoryKind<Self::MemoryKind>, Allocation)>;
    const GLOBAL_KIND: Option<Self::MemoryKind> = None; // no copying of globals from `tcx` to machine memory

    type AllocExtra = ();
    type FrameExtra = ();

    #[inline(always)]
    fn use_addr_for_alignment_check(_ecx: &InterpCx<$mir, $tcx, Self>) -> bool {
        // We do not support `use_addr`.
        false
    }

    #[inline(always)]
    fn checked_binop_checks_overflow(_ecx: &InterpCx<$mir, $tcx, Self>) -> bool {
        true
    }

    #[inline(always)]
    fn call_extra_fn(
        _ecx: &mut InterpCx<$mir, $tcx, Self>,
        fn_val: !,
        _abi: CallAbi,
        _args: &[OpTy<$tcx>],
        _destination: &PlaceTy<$tcx, Self::Provenance>,
        _target: Option<mir::BasicBlock>,
        _unwind: StackPopUnwind,
    ) -> InterpResult<$tcx> {
        match fn_val {}
    }

    #[inline(always)]
    fn adjust_allocation<'b>(
        _ecx: &InterpCx<$mir, $tcx, Self>,
        _id: AllocId,
        alloc: Cow<'b, Allocation>,
        _kind: Option<MemoryKind<Self::MemoryKind>>,
    ) -> InterpResult<$tcx, Cow<'b, Allocation<Self::Provenance>>> {
        Ok(alloc)
    }

    fn extern_static_base_pointer(
        ecx: &InterpCx<$mir, $tcx, Self>,
        def_id: DefId,
    ) -> InterpResult<$tcx, Pointer> {
        // Use the `AllocId` associated with the `DefId`. Any actual *access* will fail.
        Ok(Pointer::new(ecx.tcx.create_static_alloc(def_id), Size::ZERO))
    }

    #[inline(always)]
    fn adjust_alloc_base_pointer(
        _ecx: &InterpCx<$mir, $tcx, Self>,
        ptr: Pointer<AllocId>,
    ) -> InterpResult<$tcx, Pointer<AllocId>> {
        Ok(ptr)
    }

    #[inline(always)]
    fn ptr_from_addr_cast(
        _ecx: &InterpCx<$mir, $tcx, Self>,
        addr: u64,
    ) -> InterpResult<$tcx, Pointer<Option<AllocId>>> {
        // Allow these casts, but make the pointer not dereferenceable.
        // (I.e., they behave like transmutation.)
        // This is correct because no pointers can ever be exposed in compile-time evaluation.
        Ok(Pointer::from_addr(addr))
    }

    #[inline(always)]
    fn ptr_get_alloc(
        _ecx: &InterpCx<$mir, $tcx, Self>,
        ptr: Pointer<AllocId>,
    ) -> Option<(AllocId, Size, Self::ProvenanceExtra)> {
        // We know `offset` is relative to the allocation, so we can use `into_parts`.
        let (alloc_id, offset) = ptr.into_parts();
        Some((alloc_id, offset, ()))
    }
}
