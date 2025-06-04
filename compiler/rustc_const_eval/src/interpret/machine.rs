//! This module contains everything needed to instantiate an interpreter.
//! This separation exists to ensure that no fancy miri features like
//! interpreting common C functions leak into CTFE.

use std::borrow::{Borrow, Cow};
use std::fmt::Debug;
use std::hash::Hash;

use rustc_abi::{Align, Size};
use rustc_apfloat::{Float, FloatConvert};
use rustc_middle::query::TyCtxtAt;
use rustc_middle::ty::Ty;
use rustc_middle::ty::layout::TyAndLayout;
use rustc_middle::{mir, ty};
use rustc_span::Span;
use rustc_span::def_id::DefId;
use rustc_target::callconv::FnAbi;

use super::{
    AllocBytes, AllocId, AllocKind, AllocRange, Allocation, CTFE_ALLOC_SALT, ConstAllocation,
    CtfeProvenance, FnArg, Frame, ImmTy, InterpCx, InterpResult, MPlaceTy, MemoryKind,
    Misalignment, OpTy, PlaceTy, Pointer, Provenance, RangeSet, interp_ok, throw_unsup,
};

/// Data returned by [`Machine::after_stack_pop`], and consumed by
/// [`InterpCx::return_from_current_stack_frame`] to determine what actions should be done when
/// returning from a stack frame.
#[derive(Eq, PartialEq, Debug, Copy, Clone)]
pub enum ReturnAction {
    /// Indicates that no special handling should be
    /// done - we'll either return normally or unwind
    /// based on the terminator for the function
    /// we're leaving.
    Normal,

    /// Indicates that we should *not* jump to the return/unwind address, as the callback already
    /// took care of everything.
    NoJump,

    /// Returned by [`InterpCx::pop_stack_frame_raw`] when no cleanup should be done.
    NoCleanup,
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

    /// Callers should prefer [`AllocMap::contains_key`] when it is possible to call because it may
    /// be more efficient. This function exists for callers that only have a shared reference
    /// (which might make it slightly less efficient than `contains_key`, e.g. if
    /// the data is stored inside a `RefCell`).
    fn contains_key_ref<Q: ?Sized + Hash + Eq>(&self, k: &Q) -> bool
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
pub trait Machine<'tcx>: Sized {
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
    type AllocExtra: Debug + Clone + 'tcx;

    /// Type for the bytes of the allocation.
    type Bytes: AllocBytes + 'static;

    /// Memory's allocation map
    type MemoryMap: AllocMap<
            AllocId,
            (
                MemoryKind<Self::MemoryKind>,
                Allocation<Self::Provenance, Self::AllocExtra, Self::Bytes>,
            ),
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

    /// Determines whether `eval_mir_constant` can never fail because all required consts have
    /// already been checked before.
    const ALL_CONSTS_ARE_PRECHECKED: bool = true;

    /// Determines whether rustc_const_eval functions that make use of the [Machine] should make
    /// tracing calls (to the `tracing` library). By default this is `false`, meaning the tracing
    /// calls will supposedly be optimized out. This flag is set to `true` inside Miri, to allow
    /// tracing the interpretation steps, among other things.
    const TRACING_ENABLED: bool = false;

    /// Whether memory accesses should be alignment-checked.
    fn enforce_alignment(ecx: &InterpCx<'tcx, Self>) -> bool;

    /// Gives the machine a chance to detect more misalignment than the built-in checks would catch.
    #[inline(always)]
    fn alignment_check(
        _ecx: &InterpCx<'tcx, Self>,
        _alloc_id: AllocId,
        _alloc_align: Align,
        _alloc_kind: AllocKind,
        _offset: Size,
        _align: Align,
    ) -> Option<Misalignment> {
        None
    }

    /// Whether to enforce the validity invariant for a specific layout.
    fn enforce_validity(ecx: &InterpCx<'tcx, Self>, layout: TyAndLayout<'tcx>) -> bool;
    /// Whether to enforce the validity invariant *recursively*.
    fn enforce_validity_recursively(
        _ecx: &InterpCx<'tcx, Self>,
        _layout: TyAndLayout<'tcx>,
    ) -> bool {
        false
    }

    /// Whether Assert(OverflowNeg) and Assert(Overflow) MIR terminators should actually
    /// check for overflow.
    fn ignore_optional_overflow_checks(_ecx: &InterpCx<'tcx, Self>) -> bool;

    /// Entry point for obtaining the MIR of anything that should get evaluated.
    /// So not just functions and shims, but also const/static initializers, anonymous
    /// constants, ...
    fn load_mir(
        ecx: &InterpCx<'tcx, Self>,
        instance: ty::InstanceKind<'tcx>,
    ) -> InterpResult<'tcx, &'tcx mir::Body<'tcx>> {
        interp_ok(ecx.tcx.instance_mir(instance))
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
        ecx: &mut InterpCx<'tcx, Self>,
        instance: ty::Instance<'tcx>,
        abi: &FnAbi<'tcx, Ty<'tcx>>,
        args: &[FnArg<'tcx, Self::Provenance>],
        destination: &PlaceTy<'tcx, Self::Provenance>,
        target: Option<mir::BasicBlock>,
        unwind: mir::UnwindAction,
    ) -> InterpResult<'tcx, Option<(&'tcx mir::Body<'tcx>, ty::Instance<'tcx>)>>;

    /// Execute `fn_val`. It is the hook's responsibility to advance the instruction
    /// pointer as appropriate.
    fn call_extra_fn(
        ecx: &mut InterpCx<'tcx, Self>,
        fn_val: Self::ExtraFnVal,
        abi: &FnAbi<'tcx, Ty<'tcx>>,
        args: &[FnArg<'tcx, Self::Provenance>],
        destination: &PlaceTy<'tcx, Self::Provenance>,
        target: Option<mir::BasicBlock>,
        unwind: mir::UnwindAction,
    ) -> InterpResult<'tcx>;

    /// Directly process an intrinsic without pushing a stack frame. It is the hook's
    /// responsibility to advance the instruction pointer as appropriate.
    ///
    /// Returns `None` if the intrinsic was fully handled.
    /// Otherwise, returns an `Instance` of the function that implements the intrinsic.
    fn call_intrinsic(
        ecx: &mut InterpCx<'tcx, Self>,
        instance: ty::Instance<'tcx>,
        args: &[OpTy<'tcx, Self::Provenance>],
        destination: &PlaceTy<'tcx, Self::Provenance>,
        target: Option<mir::BasicBlock>,
        unwind: mir::UnwindAction,
    ) -> InterpResult<'tcx, Option<ty::Instance<'tcx>>>;

    /// Check whether the given function may be executed on the current machine, in terms of the
    /// target features is requires.
    fn check_fn_target_features(
        _ecx: &InterpCx<'tcx, Self>,
        _instance: ty::Instance<'tcx>,
    ) -> InterpResult<'tcx>;

    /// Called to evaluate `Assert` MIR terminators that trigger a panic.
    fn assert_panic(
        ecx: &mut InterpCx<'tcx, Self>,
        msg: &mir::AssertMessage<'tcx>,
        unwind: mir::UnwindAction,
    ) -> InterpResult<'tcx>;

    /// Called to trigger a non-unwinding panic.
    fn panic_nounwind(_ecx: &mut InterpCx<'tcx, Self>, msg: &str) -> InterpResult<'tcx>;

    /// Called when unwinding reached a state where execution should be terminated.
    fn unwind_terminate(
        ecx: &mut InterpCx<'tcx, Self>,
        reason: mir::UnwindTerminateReason,
    ) -> InterpResult<'tcx>;

    /// Called for all binary operations where the LHS has pointer type.
    ///
    /// Returns a (value, overflowed) pair if the operation succeeded
    fn binary_ptr_op(
        ecx: &InterpCx<'tcx, Self>,
        bin_op: mir::BinOp,
        left: &ImmTy<'tcx, Self::Provenance>,
        right: &ImmTy<'tcx, Self::Provenance>,
    ) -> InterpResult<'tcx, ImmTy<'tcx, Self::Provenance>>;

    /// Generate the NaN returned by a float operation, given the list of inputs.
    /// (This is all inputs, not just NaN inputs!)
    fn generate_nan<F1: Float + FloatConvert<F2>, F2: Float>(
        _ecx: &InterpCx<'tcx, Self>,
        _inputs: &[F1],
    ) -> F2 {
        // By default we always return the preferred NaN.
        F2::NAN
    }

    /// Apply non-determinism to float operations that do not return a precise result.
    fn apply_float_nondet(
        _ecx: &mut InterpCx<'tcx, Self>,
        val: ImmTy<'tcx, Self::Provenance>,
    ) -> InterpResult<'tcx, ImmTy<'tcx, Self::Provenance>> {
        interp_ok(val)
    }

    /// Determines the result of `min`/`max` on floats when the arguments are equal.
    fn equal_float_min_max<F: Float>(_ecx: &InterpCx<'tcx, Self>, a: F, _b: F) -> F {
        // By default, we pick the left argument.
        a
    }

    /// Called before a basic block terminator is executed.
    #[inline]
    fn before_terminator(_ecx: &mut InterpCx<'tcx, Self>) -> InterpResult<'tcx> {
        interp_ok(())
    }

    /// Determines the result of a `NullaryOp::UbChecks` invocation.
    fn ub_checks(_ecx: &InterpCx<'tcx, Self>) -> InterpResult<'tcx, bool>;

    /// Determines the result of a `NullaryOp::ContractChecks` invocation.
    fn contract_checks(_ecx: &InterpCx<'tcx, Self>) -> InterpResult<'tcx, bool>;

    /// Called when the interpreter encounters a `StatementKind::ConstEvalCounter` instruction.
    /// You can use this to detect long or endlessly running programs.
    #[inline]
    fn increment_const_eval_counter(_ecx: &mut InterpCx<'tcx, Self>) -> InterpResult<'tcx> {
        interp_ok(())
    }

    /// Called before a global allocation is accessed.
    /// `def_id` is `Some` if this is the "lazy" allocation of a static.
    #[inline]
    fn before_access_global(
        _tcx: TyCtxtAt<'tcx>,
        _machine: &Self,
        _alloc_id: AllocId,
        _allocation: ConstAllocation<'tcx>,
        _static_def_id: Option<DefId>,
        _is_write: bool,
    ) -> InterpResult<'tcx> {
        interp_ok(())
    }

    /// Return the `AllocId` for the given thread-local static in the current thread.
    fn thread_local_static_pointer(
        _ecx: &mut InterpCx<'tcx, Self>,
        def_id: DefId,
    ) -> InterpResult<'tcx, Pointer<Self::Provenance>> {
        throw_unsup!(ThreadLocalStatic(def_id))
    }

    /// Return the `AllocId` for the given `extern static`.
    fn extern_static_pointer(
        ecx: &InterpCx<'tcx, Self>,
        def_id: DefId,
    ) -> InterpResult<'tcx, Pointer<Self::Provenance>>;

    /// "Int-to-pointer cast"
    fn ptr_from_addr_cast(
        ecx: &InterpCx<'tcx, Self>,
        addr: u64,
    ) -> InterpResult<'tcx, Pointer<Option<Self::Provenance>>>;

    /// Marks a pointer as exposed, allowing its provenance
    /// to be recovered. "Pointer-to-int cast"
    fn expose_provenance(
        ecx: &InterpCx<'tcx, Self>,
        provenance: Self::Provenance,
    ) -> InterpResult<'tcx>;

    /// Convert a pointer with provenance into an allocation-offset pair and extra provenance info.
    /// `size` says how many bytes of memory are expected at that pointer. The *sign* of `size` can
    /// be used to disambiguate situations where a wildcard pointer sits right in between two
    /// allocations.
    ///
    /// If `ptr.provenance.get_alloc_id()` is `Some(p)`, the returned `AllocId` must be `p`.
    /// The resulting `AllocId` will just be used for that one step and the forgotten again
    /// (i.e., we'll never turn the data returned here back into a `Pointer` that might be
    /// stored in machine state).
    ///
    /// When this fails, that means the pointer does not point to a live allocation.
    fn ptr_get_alloc(
        ecx: &InterpCx<'tcx, Self>,
        ptr: Pointer<Self::Provenance>,
        size: i64,
    ) -> Option<(AllocId, Size, Self::ProvenanceExtra)>;

    /// Return a "root" pointer for the given allocation: the one that is used for direct
    /// accesses to this static/const/fn allocation, or the one returned from the heap allocator.
    ///
    /// Not called on `extern` or thread-local statics (those use the methods above).
    ///
    /// `kind` is the kind of the allocation the pointer points to; it can be `None` when
    /// it's a global and `GLOBAL_KIND` is `None`.
    fn adjust_alloc_root_pointer(
        ecx: &InterpCx<'tcx, Self>,
        ptr: Pointer,
        kind: Option<MemoryKind<Self::MemoryKind>>,
    ) -> InterpResult<'tcx, Pointer<Self::Provenance>>;

    /// Called to adjust global allocations to the Provenance and AllocExtra of this machine.
    ///
    /// If `alloc` contains pointers, then they are all pointing to globals.
    ///
    /// This should avoid copying if no work has to be done! If this returns an owned
    /// allocation (because a copy had to be done to adjust things), machine memory will
    /// cache the result. (This relies on `AllocMap::get_or` being able to add the
    /// owned allocation to the map even when the map is shared.)
    fn adjust_global_allocation<'b>(
        ecx: &InterpCx<'tcx, Self>,
        id: AllocId,
        alloc: &'b Allocation,
    ) -> InterpResult<'tcx, Cow<'b, Allocation<Self::Provenance, Self::AllocExtra, Self::Bytes>>>;

    /// Initialize the extra state of an allocation local to this machine.
    ///
    /// This is guaranteed to be called exactly once on all allocations local to this machine.
    /// It will not be called automatically for global allocations; `adjust_global_allocation`
    /// has to do that itself if that is desired.
    fn init_local_allocation(
        ecx: &InterpCx<'tcx, Self>,
        id: AllocId,
        kind: MemoryKind<Self::MemoryKind>,
        size: Size,
        align: Align,
    ) -> InterpResult<'tcx, Self::AllocExtra>;

    /// Hook for performing extra checks on a memory read access.
    /// `ptr` will always be a pointer with the provenance in `prov` pointing to the beginning of
    /// `range`.
    ///
    /// This will *not* be called during validation!
    ///
    /// Takes read-only access to the allocation so we can keep all the memory read
    /// operations take `&self`. Use a `RefCell` in `AllocExtra` if you
    /// need to mutate.
    ///
    /// This is not invoked for ZST accesses, as no read actually happens.
    #[inline(always)]
    fn before_memory_read(
        _tcx: TyCtxtAt<'tcx>,
        _machine: &Self,
        _alloc_extra: &Self::AllocExtra,
        _ptr: Pointer<Option<Self::Provenance>>,
        _prov: (AllocId, Self::ProvenanceExtra),
        _range: AllocRange,
    ) -> InterpResult<'tcx> {
        interp_ok(())
    }

    /// Hook for performing extra checks on any memory read access,
    /// that involves an allocation, even ZST reads.
    ///
    /// This will *not* be called during validation!
    ///
    /// Used to prevent statics from self-initializing by reading from their own memory
    /// as it is being initialized.
    fn before_alloc_read(_ecx: &InterpCx<'tcx, Self>, _alloc_id: AllocId) -> InterpResult<'tcx> {
        interp_ok(())
    }

    /// Hook for performing extra checks on a memory write access.
    /// This is not invoked for ZST accesses, as no write actually happens.
    /// `ptr` will always be a pointer with the provenance in `prov` pointing to the beginning of
    /// `range`.
    #[inline(always)]
    fn before_memory_write(
        _tcx: TyCtxtAt<'tcx>,
        _machine: &mut Self,
        _alloc_extra: &mut Self::AllocExtra,
        _ptr: Pointer<Option<Self::Provenance>>,
        _prov: (AllocId, Self::ProvenanceExtra),
        _range: AllocRange,
    ) -> InterpResult<'tcx> {
        interp_ok(())
    }

    /// Hook for performing extra operations on a memory deallocation.
    /// `ptr` will always be a pointer with the provenance in `prov` pointing to the beginning of
    /// the allocation.
    #[inline(always)]
    fn before_memory_deallocation(
        _tcx: TyCtxtAt<'tcx>,
        _machine: &mut Self,
        _alloc_extra: &mut Self::AllocExtra,
        _ptr: Pointer<Option<Self::Provenance>>,
        _prov: (AllocId, Self::ProvenanceExtra),
        _size: Size,
        _align: Align,
        _kind: MemoryKind<Self::MemoryKind>,
    ) -> InterpResult<'tcx> {
        interp_ok(())
    }

    /// Executes a retagging operation for a single pointer.
    /// Returns the possibly adjusted pointer.
    #[inline]
    fn retag_ptr_value(
        _ecx: &mut InterpCx<'tcx, Self>,
        _kind: mir::RetagKind,
        val: &ImmTy<'tcx, Self::Provenance>,
    ) -> InterpResult<'tcx, ImmTy<'tcx, Self::Provenance>> {
        interp_ok(val.clone())
    }

    /// Executes a retagging operation on a compound value.
    /// Replaces all pointers stored in the given place.
    #[inline]
    fn retag_place_contents(
        _ecx: &mut InterpCx<'tcx, Self>,
        _kind: mir::RetagKind,
        _place: &PlaceTy<'tcx, Self::Provenance>,
    ) -> InterpResult<'tcx> {
        interp_ok(())
    }

    /// Called on places used for in-place function argument and return value handling.
    ///
    /// These places need to be protected to make sure the program cannot tell whether the
    /// argument/return value was actually copied or passed in-place..
    fn protect_in_place_function_argument(
        ecx: &mut InterpCx<'tcx, Self>,
        mplace: &MPlaceTy<'tcx, Self::Provenance>,
    ) -> InterpResult<'tcx> {
        // Without an aliasing model, all we can do is put `Uninit` into the place.
        // Conveniently this also ensures that the place actually points to suitable memory.
        ecx.write_uninit(mplace)
    }

    /// Called immediately before a new stack frame gets pushed.
    fn init_frame(
        ecx: &mut InterpCx<'tcx, Self>,
        frame: Frame<'tcx, Self::Provenance>,
    ) -> InterpResult<'tcx, Frame<'tcx, Self::Provenance, Self::FrameExtra>>;

    /// Borrow the current thread's stack.
    fn stack<'a>(
        ecx: &'a InterpCx<'tcx, Self>,
    ) -> &'a [Frame<'tcx, Self::Provenance, Self::FrameExtra>];

    /// Mutably borrow the current thread's stack.
    fn stack_mut<'a>(
        ecx: &'a mut InterpCx<'tcx, Self>,
    ) -> &'a mut Vec<Frame<'tcx, Self::Provenance, Self::FrameExtra>>;

    /// Called immediately after a stack frame got pushed and its locals got initialized.
    fn after_stack_push(_ecx: &mut InterpCx<'tcx, Self>) -> InterpResult<'tcx> {
        interp_ok(())
    }

    /// Called just before the frame is removed from the stack (followed by return value copy and
    /// local cleanup).
    fn before_stack_pop(_ecx: &mut InterpCx<'tcx, Self>) -> InterpResult<'tcx> {
        interp_ok(())
    }

    /// Called immediately after a stack frame got popped, but before jumping back to the caller.
    /// The `locals` have already been destroyed!
    #[inline(always)]
    fn after_stack_pop(
        _ecx: &mut InterpCx<'tcx, Self>,
        _frame: Frame<'tcx, Self::Provenance, Self::FrameExtra>,
        unwinding: bool,
    ) -> InterpResult<'tcx, ReturnAction> {
        // By default, we do not support unwinding from panics
        assert!(!unwinding);
        interp_ok(ReturnAction::Normal)
    }

    /// Called immediately after an "immediate" local variable is read in a given frame
    /// (i.e., this is called for reads that do not end up accessing addressable memory).
    #[inline(always)]
    fn after_local_read(
        _ecx: &InterpCx<'tcx, Self>,
        _frame: &Frame<'tcx, Self::Provenance, Self::FrameExtra>,
        _local: mir::Local,
    ) -> InterpResult<'tcx> {
        interp_ok(())
    }

    /// Called immediately after an "immediate" local variable is assigned a new value
    /// (i.e., this is called for writes that do not end up in memory).
    /// `storage_live` indicates whether this is the initial write upon `StorageLive`.
    #[inline(always)]
    fn after_local_write(
        _ecx: &mut InterpCx<'tcx, Self>,
        _local: mir::Local,
        _storage_live: bool,
    ) -> InterpResult<'tcx> {
        interp_ok(())
    }

    /// Called immediately after actual memory was allocated for a local
    /// but before the local's stack frame is updated to point to that memory.
    #[inline(always)]
    fn after_local_moved_to_memory(
        _ecx: &mut InterpCx<'tcx, Self>,
        _local: mir::Local,
        _mplace: &MPlaceTy<'tcx, Self::Provenance>,
    ) -> InterpResult<'tcx> {
        interp_ok(())
    }

    /// Evaluate the given constant. The `eval` function will do all the required evaluation,
    /// but this hook has the chance to do some pre/postprocessing.
    #[inline(always)]
    fn eval_mir_constant<F>(
        ecx: &InterpCx<'tcx, Self>,
        val: mir::Const<'tcx>,
        span: Span,
        layout: Option<TyAndLayout<'tcx>>,
        eval: F,
    ) -> InterpResult<'tcx, OpTy<'tcx, Self::Provenance>>
    where
        F: Fn(
            &InterpCx<'tcx, Self>,
            mir::Const<'tcx>,
            Span,
            Option<TyAndLayout<'tcx>>,
        ) -> InterpResult<'tcx, OpTy<'tcx, Self::Provenance>>,
    {
        eval(ecx, val, span, layout)
    }

    /// Returns the salt to be used for a deduplicated global alloation.
    /// If the allocation is for a function, the instance is provided as well
    /// (this lets Miri ensure unique addresses for some functions).
    fn get_global_alloc_salt(
        ecx: &InterpCx<'tcx, Self>,
        instance: Option<ty::Instance<'tcx>>,
    ) -> usize;

    fn cached_union_data_range<'e>(
        _ecx: &'e mut InterpCx<'tcx, Self>,
        _ty: Ty<'tcx>,
        compute_range: impl FnOnce() -> RangeSet,
    ) -> Cow<'e, RangeSet> {
        // Default to no caching.
        Cow::Owned(compute_range())
    }

    /// Compute the value passed to the constructors of the `AllocBytes` type for
    /// abstract machine allocations.
    fn get_default_alloc_params(&self) -> <Self::Bytes as AllocBytes>::AllocParams;
}

/// A lot of the flexibility above is just needed for `Miri`, but all "compile-time" machines
/// (CTFE and ConstProp) use the same instance. Here, we share that code.
pub macro compile_time_machine(<$tcx: lifetime>) {
    type Provenance = CtfeProvenance;
    type ProvenanceExtra = bool; // the "immutable" flag

    type ExtraFnVal = !;

    type MemoryMap =
        rustc_data_structures::fx::FxIndexMap<AllocId, (MemoryKind<Self::MemoryKind>, Allocation)>;
    const GLOBAL_KIND: Option<Self::MemoryKind> = None; // no copying of globals from `tcx` to machine memory

    type AllocExtra = ();
    type FrameExtra = ();
    type Bytes = Box<[u8]>;

    #[inline(always)]
    fn ignore_optional_overflow_checks(_ecx: &InterpCx<$tcx, Self>) -> bool {
        false
    }

    #[inline(always)]
    fn unwind_terminate(
        _ecx: &mut InterpCx<$tcx, Self>,
        _reason: mir::UnwindTerminateReason,
    ) -> InterpResult<$tcx> {
        unreachable!("unwinding cannot happen during compile-time evaluation")
    }

    #[inline(always)]
    fn check_fn_target_features(
        _ecx: &InterpCx<$tcx, Self>,
        _instance: ty::Instance<$tcx>,
    ) -> InterpResult<$tcx> {
        // For now we don't do any checking here. We can't use `tcx.sess` because that can differ
        // between crates, and we need to ensure that const-eval always behaves the same.
        interp_ok(())
    }

    #[inline(always)]
    fn call_extra_fn(
        _ecx: &mut InterpCx<$tcx, Self>,
        fn_val: !,
        _abi: &FnAbi<$tcx, Ty<$tcx>>,
        _args: &[FnArg<$tcx>],
        _destination: &PlaceTy<$tcx, Self::Provenance>,
        _target: Option<mir::BasicBlock>,
        _unwind: mir::UnwindAction,
    ) -> InterpResult<$tcx> {
        match fn_val {}
    }

    #[inline(always)]
    fn ub_checks(_ecx: &InterpCx<$tcx, Self>) -> InterpResult<$tcx, bool> {
        // We can't look at `tcx.sess` here as that can differ across crates, which can lead to
        // unsound differences in evaluating the same constant at different instantiation sites.
        interp_ok(true)
    }

    #[inline(always)]
    fn contract_checks(_ecx: &InterpCx<$tcx, Self>) -> InterpResult<$tcx, bool> {
        // We can't look at `tcx.sess` here as that can differ across crates, which can lead to
        // unsound differences in evaluating the same constant at different instantiation sites.
        interp_ok(true)
    }

    #[inline(always)]
    fn adjust_global_allocation<'b>(
        _ecx: &InterpCx<$tcx, Self>,
        _id: AllocId,
        alloc: &'b Allocation,
    ) -> InterpResult<$tcx, Cow<'b, Allocation<Self::Provenance>>> {
        // Overwrite default implementation: no need to adjust anything.
        interp_ok(Cow::Borrowed(alloc))
    }

    fn init_local_allocation(
        _ecx: &InterpCx<$tcx, Self>,
        _id: AllocId,
        _kind: MemoryKind<Self::MemoryKind>,
        _size: Size,
        _align: Align,
    ) -> InterpResult<$tcx, Self::AllocExtra> {
        interp_ok(())
    }

    fn extern_static_pointer(
        ecx: &InterpCx<$tcx, Self>,
        def_id: DefId,
    ) -> InterpResult<$tcx, Pointer> {
        // Use the `AllocId` associated with the `DefId`. Any actual *access* will fail.
        interp_ok(Pointer::new(ecx.tcx.reserve_and_set_static_alloc(def_id).into(), Size::ZERO))
    }

    #[inline(always)]
    fn adjust_alloc_root_pointer(
        _ecx: &InterpCx<$tcx, Self>,
        ptr: Pointer<CtfeProvenance>,
        _kind: Option<MemoryKind<Self::MemoryKind>>,
    ) -> InterpResult<$tcx, Pointer<CtfeProvenance>> {
        interp_ok(ptr)
    }

    #[inline(always)]
    fn ptr_from_addr_cast(
        _ecx: &InterpCx<$tcx, Self>,
        addr: u64,
    ) -> InterpResult<$tcx, Pointer<Option<CtfeProvenance>>> {
        // Allow these casts, but make the pointer not dereferenceable.
        // (I.e., they behave like transmutation.)
        // This is correct because no pointers can ever be exposed in compile-time evaluation.
        interp_ok(Pointer::from_addr_invalid(addr))
    }

    #[inline(always)]
    fn ptr_get_alloc(
        _ecx: &InterpCx<$tcx, Self>,
        ptr: Pointer<CtfeProvenance>,
        _size: i64,
    ) -> Option<(AllocId, Size, Self::ProvenanceExtra)> {
        // We know `offset` is relative to the allocation, so we can use `into_parts`.
        let (prov, offset) = ptr.into_parts();
        Some((prov.alloc_id(), offset, prov.immutable()))
    }

    #[inline(always)]
    fn get_global_alloc_salt(
        _ecx: &InterpCx<$tcx, Self>,
        _instance: Option<ty::Instance<$tcx>>,
    ) -> usize {
        CTFE_ALLOC_SALT
    }
}
