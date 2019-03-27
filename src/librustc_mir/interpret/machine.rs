//! This module contains everything needed to instantiate an interpreter.
//! This separation exists to ensure that no fancy miri features like
//! interpreting common C functions leak into CTFE.

use std::borrow::{Borrow, Cow};
use std::hash::Hash;

use rustc::hir::{self, def_id::DefId};
use rustc::mir;
use rustc::ty::{self, query::TyCtxtAt};

use super::{
    Allocation, AllocId, EvalResult, Scalar, AllocationExtra,
    InterpretCx, PlaceTy, MPlaceTy, OpTy, ImmTy, Pointer, MemoryKind,
};

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
        where K: Borrow<Q>;

    /// Inserts a new entry into the map.
    fn insert(&mut self, k: K, v: V) -> Option<V>;

    /// Removes an entry from the map.
    fn remove<Q: ?Sized + Hash + Eq>(&mut self, k: &Q) -> Option<V>
        where K: Borrow<Q>;

    /// Returns data based the keys and values in the map.
    fn filter_map_collect<T>(&self, f: impl FnMut(&K, &V) -> Option<T>) -> Vec<T>;

    /// Returns a reference to entry `k`. If no such entry exists, call
    /// `vacant` and either forward its error, or add its result to the map
    /// and return a reference to *that*.
    fn get_or<E>(
        &self,
        k: K,
        vacant: impl FnOnce() -> Result<V, E>
    ) -> Result<&V, E>;

    /// Returns a mutable reference to entry `k`. If no such entry exists, call
    /// `vacant` and either forward its error, or add its result to the map
    /// and return a reference to *that*.
    fn get_mut_or<E>(
        &mut self,
        k: K,
        vacant: impl FnOnce() -> Result<V, E>
    ) -> Result<&mut V, E>;
}

/// Methods of this trait signifies a point where CTFE evaluation would fail
/// and some use case dependent behaviour can instead be applied.
pub trait Machine<'a, 'mir, 'tcx>: Sized {
    /// Additional memory kinds a machine wishes to distinguish from the builtin ones
    type MemoryKinds: ::std::fmt::Debug + MayLeak + Eq + 'static;

    /// Tag tracked alongside every pointer. This is used to implement "Stacked Borrows"
    /// <https://www.ralfj.de/blog/2018/08/07/stacked-borrows.html>.
    /// The `default()` is used for pointers to consts, statics, vtables and functions.
    type PointerTag: ::std::fmt::Debug + Default + Copy + Eq + Hash + 'static;

    /// Extra data stored in every call frame.
    type FrameExtra;

    /// Extra data stored in memory. A reference to this is available when `AllocExtra`
    /// gets initialized, so you can e.g., have an `Rc` here if there is global state you
    /// need access to in the `AllocExtra` hooks.
    type MemoryExtra: Default;

    /// Extra data stored in every allocation.
    type AllocExtra: AllocationExtra<Self::PointerTag, Self::MemoryExtra> + 'static;

    /// Memory's allocation map
    type MemoryMap:
        AllocMap<
            AllocId,
            (MemoryKind<Self::MemoryKinds>, Allocation<Self::PointerTag, Self::AllocExtra>)
        > +
        Default +
        Clone;

    /// The memory kind to use for copied statics -- or None if statics should not be mutated
    /// and thus any such attempt will cause a `ModifiedStatic` error to be raised.
    /// Statics are copied under two circumstances: When they are mutated, and when
    /// `static_with_default_tag` or `find_foreign_static` (see below) returns an owned allocation
    /// that is added to the memory so that the work is not done twice.
    const STATIC_KIND: Option<Self::MemoryKinds>;

    /// Whether to enforce the validity invariant
    fn enforce_validity(ecx: &InterpretCx<'a, 'mir, 'tcx, Self>) -> bool;

    /// Called before a basic block terminator is executed.
    /// You can use this to detect endlessly running programs.
    fn before_terminator(ecx: &mut InterpretCx<'a, 'mir, 'tcx, Self>) -> EvalResult<'tcx>;

    /// Entry point to all function calls.
    ///
    /// Returns either the mir to use for the call, or `None` if execution should
    /// just proceed (which usually means this hook did all the work that the
    /// called function should usually have done). In the latter case, it is
    /// this hook's responsibility to call `goto_block(ret)` to advance the instruction pointer!
    /// (This is to support functions like `__rust_maybe_catch_panic` that neither find a MIR
    /// nor just jump to `ret`, but instead push their own stack frame.)
    /// Passing `dest`and `ret` in the same `Option` proved very annoying when only one of them
    /// was used.
    fn find_fn(
        ecx: &mut InterpretCx<'a, 'mir, 'tcx, Self>,
        instance: ty::Instance<'tcx>,
        args: &[OpTy<'tcx, Self::PointerTag>],
        dest: Option<PlaceTy<'tcx, Self::PointerTag>>,
        ret: Option<mir::BasicBlock>,
    ) -> EvalResult<'tcx, Option<&'mir mir::Mir<'tcx>>>;

    /// Directly process an intrinsic without pushing a stack frame.
    /// If this returns successfully, the engine will take care of jumping to the next block.
    fn call_intrinsic(
        ecx: &mut InterpretCx<'a, 'mir, 'tcx, Self>,
        instance: ty::Instance<'tcx>,
        args: &[OpTy<'tcx, Self::PointerTag>],
        dest: PlaceTy<'tcx, Self::PointerTag>,
    ) -> EvalResult<'tcx>;

    /// Called for read access to a foreign static item.
    ///
    /// This will only be called once per static and machine; the result is cached in
    /// the machine memory. (This relies on `AllocMap::get_or` being able to add the
    /// owned allocation to the map even when the map is shared.)
    fn find_foreign_static(
        def_id: DefId,
        tcx: TyCtxtAt<'a, 'tcx, 'tcx>,
        memory_extra: &Self::MemoryExtra,
    ) -> EvalResult<'tcx, Cow<'tcx, Allocation<Self::PointerTag, Self::AllocExtra>>>;

    /// Called to turn an allocation obtained from the `tcx` into one that has
    /// the right type for this machine.
    ///
    /// This should avoid copying if no work has to be done! If this returns an owned
    /// allocation (because a copy had to be done to add tags or metadata), machine memory will
    /// cache the result. (This relies on `AllocMap::get_or` being able to add the
    /// owned allocation to the map even when the map is shared.)
    fn adjust_static_allocation<'b>(
        alloc: &'b Allocation,
        memory_extra: &Self::MemoryExtra,
    ) -> Cow<'b, Allocation<Self::PointerTag, Self::AllocExtra>>;

    /// Called for all binary operations on integer(-like) types when one operand is a pointer
    /// value, and for the `Offset` operation that is inherently about pointers.
    ///
    /// Returns a (value, overflowed) pair if the operation succeeded
    fn ptr_op(
        ecx: &InterpretCx<'a, 'mir, 'tcx, Self>,
        bin_op: mir::BinOp,
        left: ImmTy<'tcx, Self::PointerTag>,
        right: ImmTy<'tcx, Self::PointerTag>,
    ) -> EvalResult<'tcx, (Scalar<Self::PointerTag>, bool)>;

    /// Heap allocations via the `box` keyword.
    fn box_alloc(
        ecx: &mut InterpretCx<'a, 'mir, 'tcx, Self>,
        dest: PlaceTy<'tcx, Self::PointerTag>,
    ) -> EvalResult<'tcx>;

    /// Adds the tag for a newly allocated pointer.
    fn tag_new_allocation(
        ecx: &mut InterpretCx<'a, 'mir, 'tcx, Self>,
        ptr: Pointer,
        kind: MemoryKind<Self::MemoryKinds>,
    ) -> Pointer<Self::PointerTag>;

    /// Executed when evaluating the `*` operator: Following a reference.
    /// This has the chance to adjust the tag. It should not change anything else!
    /// `mutability` can be `None` in case a raw ptr is being dereferenced.
    #[inline]
    fn tag_dereference(
        _ecx: &InterpretCx<'a, 'mir, 'tcx, Self>,
        place: MPlaceTy<'tcx, Self::PointerTag>,
        _mutability: Option<hir::Mutability>,
    ) -> EvalResult<'tcx, Scalar<Self::PointerTag>> {
        Ok(place.ptr)
    }

    /// Executes a retagging operation
    #[inline]
    fn retag(
        _ecx: &mut InterpretCx<'a, 'mir, 'tcx, Self>,
        _kind: mir::RetagKind,
        _place: PlaceTy<'tcx, Self::PointerTag>,
    ) -> EvalResult<'tcx> {
        Ok(())
    }

    /// Called immediately before a new stack frame got pushed
    fn stack_push(
        ecx: &mut InterpretCx<'a, 'mir, 'tcx, Self>,
    ) -> EvalResult<'tcx, Self::FrameExtra>;

    /// Called immediately after a stack frame gets popped
    fn stack_pop(
        ecx: &mut InterpretCx<'a, 'mir, 'tcx, Self>,
        extra: Self::FrameExtra,
    ) -> EvalResult<'tcx>;
}
