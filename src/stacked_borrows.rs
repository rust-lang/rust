use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;

use rustc::ty::{self, layout::Size};
use rustc::hir::{Mutability, MutMutable, MutImmutable};
use rustc::mir::RetagKind;

use crate::{
    EvalResult, InterpError, MiriEvalContext, HelpersEvalContextExt, Evaluator, MutValueVisitor,
    MemoryKind, MiriMemoryKind, RangeMap, AllocId, Allocation, AllocationExtra,
    Pointer, Immediate, ImmTy, PlaceTy, MPlaceTy,
};

pub type Timestamp = u64;
pub type CallId = u64;

/// Information about which kind of borrow was used to create the reference this is tagged with.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum Borrow {
    /// A unique (mutable) reference.
    Uniq(Timestamp),
    /// An aliasing reference. This is also used by raw pointers, which do not track details
    /// of how or when they were created, hence the timestamp is optional.
    /// `Shr(Some(_))` does *not* mean that the destination of this reference is frozen;
    /// that depends on the type! Only those parts outside of an `UnsafeCell` are actually
    /// frozen.
    Alias(Option<Timestamp>),
}

impl Borrow {
    #[inline(always)]
    pub fn is_aliasing(self) -> bool {
        match self {
            Borrow::Alias(_) => true,
            _ => false,
        }
    }

    #[inline(always)]
    pub fn is_unique(self) -> bool {
        match self {
            Borrow::Uniq(_) => true,
            _ => false,
        }
    }
}

impl Default for Borrow {
    fn default() -> Self {
        Borrow::Alias(None)
    }
}

/// An item in the per-location borrow stack.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum BorStackItem {
    /// Indicates the unique reference that may mutate.
    Uniq(Timestamp),
    /// Indicates that the location has been mutably shared. Used for raw pointers as
    /// well as for unfrozen shared references.
    Raw,
    /// A barrier, tracking the function it belongs to by its index on the call stack.
    FnBarrier(CallId)
}

/// Extra per-location state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stack {
    /// Used as the stack; never empty.
    borrows: Vec<BorStackItem>,
    /// A virtual frozen "item" on top of the stack.
    frozen_since: Option<Timestamp>,
}

impl Stack {
    #[inline(always)]
    pub fn is_frozen(&self) -> bool {
        self.frozen_since.is_some()
    }
}

/// Indicates which kind of reference is being used.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum RefKind {
    /// `&mut`.
    Unique,
    /// `&` without interior mutability.
    Frozen,
    /// `*` (raw pointer) or `&` to `UnsafeCell`.
    Raw,
}

/// Indicates which kind of access is being performed.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum AccessKind {
    Read,
    Write,
    Dealloc,
}

/// Extra global state in the memory, available to the memory access hooks.
#[derive(Debug)]
pub struct BarrierTracking {
    next_id: CallId,
    active_calls: HashSet<CallId>,
}
pub type MemoryState = Rc<RefCell<BarrierTracking>>;

impl Default for BarrierTracking {
    fn default() -> Self {
        BarrierTracking {
            next_id: 0,
            active_calls: HashSet::default(),
        }
    }
}

impl BarrierTracking {
    pub fn new_call(&mut self) -> CallId {
        let id = self.next_id;
        trace!("new_call: Assigning ID {}", id);
        self.active_calls.insert(id);
        self.next_id += 1;
        id
    }

    pub fn end_call(&mut self, id: CallId) {
        assert!(self.active_calls.remove(&id));
    }

    fn is_active(&self, id: CallId) -> bool {
        self.active_calls.contains(&id)
    }
}

/// Extra global machine state.
#[derive(Clone, Debug)]
pub struct State {
    clock: Timestamp
}

impl Default for State {
    fn default() -> Self {
        State { clock: 0 }
    }
}

impl State {
    fn increment_clock(&mut self) -> Timestamp {
        let val = self.clock;
        self.clock = val + 1;
        val
    }
}

/// Extra per-allocation state.
#[derive(Clone, Debug)]
pub struct Stacks {
    // Even reading memory can have effects on the stack, so we need a `RefCell` here.
    stacks: RefCell<RangeMap<Stack>>,
    barrier_tracking: MemoryState,
}

/// Core per-location operations: deref, access, create.
/// We need to make at least the following things true:
///
/// U1: After creating a `Uniq`, it is at the top (and unfrozen).
/// U2: If the top is `Uniq` (and unfrozen), accesses must be through that `Uniq` or pop it.
/// U3: If an access (deref sufficient?) happens with a `Uniq`, it requires the `Uniq` to be in the stack.
///
/// F1: After creating a `&`, the parts outside `UnsafeCell` are frozen.
/// F2: If a write access happens, it unfreezes.
/// F3: If an access (well, a deref) happens with an `&` outside `UnsafeCell`,
///     it requires the location to still be frozen.
impl<'tcx> Stack {
    /// Deref `bor`: check if the location is frozen and the tag in the stack.
    /// This dos *not* constitute an access! "Deref" refers to the `*` operator
    /// in Rust, and includs cases like `&*x` or `(*x).foo` where no or only part
    /// of the memory actually gets accessed. Also we cannot know if we are
    /// going to read or write.
    /// Returns the index of the item we matched, `None` if it was the frozen one.
    /// `kind` indicates which kind of reference is being dereferenced.
    fn deref(
        &self,
        bor: Borrow,
        kind: RefKind,
    ) -> Result<Option<usize>, String> {
        // Exclude unique ref with frozen tag.
        if let (RefKind::Unique, Borrow::Alias(Some(_))) = (kind, bor) {
            return Err(format!("encountered mutable reference with frozen tag ({:?})", bor));
        }
        // Checks related to freezing.
        match bor {
            Borrow::Alias(Some(bor_t)) if kind == RefKind::Frozen => {
                // We need the location to be frozen. This ensures F3.
                let frozen = self.frozen_since.map_or(false, |itm_t| itm_t <= bor_t);
                return if frozen { Ok(None) } else {
                    Err(format!("location is not frozen long enough"))
                }
            }
            Borrow::Alias(_) if self.frozen_since.is_some() => {
                // Shared deref to frozen location; looking good.
                return Ok(None)
            }
            // Not sufficient; go on looking.
            _ => {}
        }
        // If we got here, we have to look for our item in the stack.
        for (idx, &itm) in self.borrows.iter().enumerate().rev() {
            match (itm, bor) {
                (BorStackItem::Uniq(itm_t), Borrow::Uniq(bor_t)) if itm_t == bor_t => {
                    // Found matching unique item. This satisfies U3.
                    return Ok(Some(idx))
                }
                (BorStackItem::Raw, Borrow::Alias(_)) => {
                    // Found matching aliasing/raw item.
                    return Ok(Some(idx))
                }
                // Go on looking. We ignore barriers! When an `&mut` and an `&` alias,
                // dereferencing the `&` is still possible (to reborrow), but doing
                // an access is not.
                _ => {}
            }
        }
        // If we got here, we did not find our item. We have to error to satisfy U3.
        Err(format!("Borrow being dereferenced ({:?}) does not exist on the borrow stack", bor))
    }

    /// Performs an actual memory access using `bor`. We do not know any types here
    /// or whether things should be frozen, but we *do* know if this is reading
    /// or writing.
    fn access(
        &mut self,
        bor: Borrow,
        kind: AccessKind,
        barrier_tracking: &BarrierTracking,
    ) -> EvalResult<'tcx> {
        // Check if we can match the frozen "item".
        // Not possible on writes!
        if self.is_frozen() {
            if kind == AccessKind::Read {
                // When we are frozen, we just accept all reads. No harm in this.
                // The deref already checked that `Uniq` items are in the stack, and that
                // the location is frozen if it should be.
                return Ok(());
            }
            trace!("access: unfreezing");
        }
        // Unfreeze on writes. This ensures F2.
        self.frozen_since = None;
        // Pop the stack until we have something matching.
        while let Some(&itm) = self.borrows.last() {
            match (itm, bor) {
                (BorStackItem::FnBarrier(call), _) if barrier_tracking.is_active(call) => {
                    return err!(MachineError(format!(
                        "stopping looking for borrow being accessed ({:?}) because of barrier ({})",
                        bor, call
                    )))
                }
                (BorStackItem::Uniq(itm_t), Borrow::Uniq(bor_t)) if itm_t == bor_t => {
                    // Found matching unique item. Continue after the match.
                }
                (BorStackItem::Raw, _) if kind == AccessKind::Read => {
                    // When reading, everything can use a raw item!
                    // We do not want to do this when writing: Writing to an `&mut`
                    // should reaffirm its exclusivity (i.e., make sure it is
                    // on top of the stack). Continue after the match.
                }
                (BorStackItem::Raw, Borrow::Alias(_)) => {
                    // Found matching raw item. Continue after the match.
                }
                _ => {
                    // Pop this, go on. This ensures U2.
                    let itm = self.borrows.pop().unwrap();
                    trace!("access: Popping {:?}", itm);
                    continue
                }
            }
            // If we got here, we found a matching item. Congratulations!
            // However, we are not done yet: If this access is deallocating, we must make sure
            // there are no active barriers remaining on the stack.
            if kind == AccessKind::Dealloc {
                for &itm in self.borrows.iter().rev() {
                    match itm {
                        BorStackItem::FnBarrier(call) if barrier_tracking.is_active(call) => {
                            return err!(MachineError(format!(
                                "deallocating with active barrier ({})", call
                            )))
                        }
                        _ => {},
                    }
                }
            }
            // Now we are done.
            return Ok(())
        }
        // If we got here, we did not find our item.
        err!(MachineError(format!(
            "borrow being accessed ({:?}) does not exist on the borrow stack",
            bor
        )))
    }

    /// Initiate `bor`; mostly this means pushing.
    /// This operation cannot fail; it is up to the caller to ensure that the precondition
    /// is met: We cannot push `Uniq` onto frozen stacks.
    /// `kind` indicates which kind of reference is being created.
    fn create(&mut self, bor: Borrow, kind: RefKind) {
        // When creating a frozen reference, freeze. This ensures F1.
        // We also do *not* push anything else to the stack, making sure that no nother kind
        // of access (like writing through raw pointers) is permitted.
        if kind == RefKind::Frozen {
            let bor_t = match bor {
                Borrow::Alias(Some(t)) => t,
                _ => bug!("Creating illegal borrow {:?} for frozen ref", bor),
            };
            // It is possible that we already are frozen (e.g., if we just pushed a barrier,
            // the redundancy check would not have kicked in).
            match self.frozen_since {
                Some(loc_t) => assert!(
                    loc_t <= bor_t,
                    "trying to freeze location for longer than it was already frozen"
                ),
                None => {
                    trace!("create: Freezing");
                    self.frozen_since = Some(bor_t);
                }
            }
            return;
        }
        assert!(
            self.frozen_since.is_none(),
            "trying to create non-frozen reference to frozen location"
        );

        // Push new item to the stack.
        let itm = match bor {
            Borrow::Uniq(t) => BorStackItem::Uniq(t),
            Borrow::Alias(_) => BorStackItem::Raw,
        };
        if *self.borrows.last().unwrap() == itm {
            // This is just an optimization, no functional change: Avoid stacking
            // multiple `Shr` on top of each other.
            assert!(bor.is_aliasing());
            trace!("create: sharing a shared location is a NOP");
        } else {
            // This ensures U1.
            trace!("create: pushing {:?}", itm);
            self.borrows.push(itm);
        }
    }

    /// Adds a barrier.
    fn barrier(&mut self, call: CallId) {
        let itm = BorStackItem::FnBarrier(call);
        if *self.borrows.last().unwrap() == itm {
            // This is just an optimization, no functional change: Avoid stacking
            // multiple identical barriers on top of each other.
            // This can happen when a function receives several shared references
            // that overlap.
            trace!("barrier: avoiding redundant extra barrier");
        } else {
            trace!("barrier: pushing barrier for call {}", call);
            self.borrows.push(itm);
        }
    }
}

/// Higher-level per-location operations: deref, access, reborrow.
impl<'tcx> Stacks {
    /// Checks that this stack is fine with being dereferenced.
    fn deref(
        &self,
        ptr: Pointer<Borrow>,
        size: Size,
        kind: RefKind,
    ) -> EvalResult<'tcx> {
        trace!("deref for tag {:?} as {:?}: {:?}, size {}",
            ptr.tag, kind, ptr, size.bytes());
        let stacks = self.stacks.borrow();
        for stack in stacks.iter(ptr.offset, size) {
            stack.deref(ptr.tag, kind).map_err(InterpError::MachineError)?;
        }
        Ok(())
    }

    /// `ptr` got used, reflect that in the stack.
    fn access(
        &self,
        ptr: Pointer<Borrow>,
        size: Size,
        kind: AccessKind,
    ) -> EvalResult<'tcx> {
        trace!("{:?} access of tag {:?}: {:?}, size {}", kind, ptr.tag, ptr, size.bytes());
        // Even reads can have a side-effect, by invalidating other references.
        // This is fundamentally necessary since `&mut` asserts that there
        // are no accesses through other references, not even reads.
        let barrier_tracking = self.barrier_tracking.borrow();
        let mut stacks = self.stacks.borrow_mut();
        for stack in stacks.iter_mut(ptr.offset, size) {
            stack.access(ptr.tag, kind, &*barrier_tracking)?;
        }
        Ok(())
    }

    /// Reborrow the given pointer to the new tag for the given kind of reference.
    /// This works on `&self` because we might encounter references to constant memory.
    fn reborrow(
        &self,
        ptr: Pointer<Borrow>,
        size: Size,
        mut barrier: Option<CallId>,
        new_bor: Borrow,
        new_kind: RefKind,
    ) -> EvalResult<'tcx> {
        assert_eq!(new_bor.is_unique(), new_kind == RefKind::Unique);
        trace!(
            "reborrow for tag {:?} to {:?} as {:?}: {:?}, size {}",
            ptr.tag, new_bor, new_kind, ptr, size.bytes(),
        );
        if new_kind == RefKind::Raw {
            // No barrier for raw, including `&UnsafeCell`. They can rightfully alias with `&mut`.
            // FIXME: This means that the `dereferencable` attribute on non-frozen shared references
            // is incorrect! They are dereferencable when the function is called, but might become
            // non-dereferencable during the course of execution.
            // Also see [1], [2].
            //
            // [1]: <https://internals.rust-lang.org/t/
            //       is-it-possible-to-be-memory-safe-with-deallocated-self/8457/8>,
            // [2]: <https://lists.llvm.org/pipermail/llvm-dev/2018-July/124555.html>
            barrier = None;
        }
        let barrier_tracking = self.barrier_tracking.borrow();
        let mut stacks = self.stacks.borrow_mut();
        for stack in stacks.iter_mut(ptr.offset, size) {
            // Access source `ptr`, create new ref.
            let ptr_idx = stack.deref(ptr.tag, new_kind).map_err(InterpError::MachineError)?;
            // If we can deref the new tag already, and if that tag lives higher on
            // the stack than the one we come from, just use that.
            // That is, we check if `new_bor` *already* is "derived from" `ptr.tag`.
            // This also checks frozenness, if required.
            let bor_redundant = barrier.is_none() &&
                match (ptr_idx, stack.deref(new_bor, new_kind)) {
                    // If the new borrow works with the frozen item, or else if it lives
                    // above the old one in the stack, our job here is done.
                    (_, Ok(None)) => true,
                    (Some(ptr_idx), Ok(Some(new_idx))) if new_idx >= ptr_idx => true,
                    // Otherwise, we need to create a new borrow.
                    _ => false,
                };
            if bor_redundant {
                assert!(new_bor.is_aliasing(), "a unique reborrow can never be redundant");
                trace!("reborrow is redundant");
                continue;
            }
            // We need to do some actual work.
            let access_kind = if new_kind == RefKind::Unique {
                AccessKind::Write
            } else {
                AccessKind::Read
            };
            stack.access(ptr.tag, access_kind, &*barrier_tracking)?;
            if let Some(call) = barrier {
                stack.barrier(call);
            }
            stack.create(new_bor, new_kind);
        }
        Ok(())
    }
}

/// Hooks and glue.
impl AllocationExtra<Borrow, MemoryState> for Stacks {
    #[inline(always)]
    fn memory_allocated<'tcx>(size: Size, extra: &MemoryState) -> Self {
        let stack = Stack {
            borrows: vec![BorStackItem::Raw],
            frozen_since: None,
        };
        Stacks {
            stacks: RefCell::new(RangeMap::new(size, stack)),
            barrier_tracking: Rc::clone(extra),
        }
    }

    #[inline(always)]
    fn memory_read<'tcx>(
        alloc: &Allocation<Borrow, Stacks>,
        ptr: Pointer<Borrow>,
        size: Size,
    ) -> EvalResult<'tcx> {
        alloc.extra.access(ptr, size, AccessKind::Read)
    }

    #[inline(always)]
    fn memory_written<'tcx>(
        alloc: &mut Allocation<Borrow, Stacks>,
        ptr: Pointer<Borrow>,
        size: Size,
    ) -> EvalResult<'tcx> {
        alloc.extra.access(ptr, size, AccessKind::Write)
    }

    #[inline(always)]
    fn memory_deallocated<'tcx>(
        alloc: &mut Allocation<Borrow, Stacks>,
        ptr: Pointer<Borrow>,
        size: Size,
    ) -> EvalResult<'tcx> {
        alloc.extra.access(ptr, size, AccessKind::Dealloc)
    }
}

impl<'tcx> Stacks {
    /// Pushes the first item to the stacks.
    pub(crate) fn first_item(
        &mut self,
        itm: BorStackItem,
        size: Size
    ) {
        for stack in self.stacks.get_mut().iter_mut(Size::ZERO, size) {
            assert!(stack.borrows.len() == 1);
            assert_eq!(stack.borrows.pop().unwrap(), BorStackItem::Raw);
            stack.borrows.push(itm);
        }
    }
}

impl<'a, 'mir, 'tcx> EvalContextPrivExt<'a, 'mir, 'tcx> for crate::MiriEvalContext<'a, 'mir, 'tcx> {}
trait EvalContextPrivExt<'a, 'mir, 'tcx: 'a+'mir>: crate::MiriEvalContextExt<'a, 'mir, 'tcx> {
    fn reborrow(
        &mut self,
        place: MPlaceTy<'tcx, Borrow>,
        size: Size,
        fn_barrier: bool,
        new_bor: Borrow
    ) -> EvalResult<'tcx> {
        let this = self.eval_context_mut();
        let ptr = place.ptr.to_ptr()?;
        let barrier = if fn_barrier { Some(this.frame().extra) } else { None };
        trace!("reborrow: creating new reference for {:?} (pointee {}): {:?}",
            ptr, place.layout.ty, new_bor);

        // Get the allocation. It might not be mutable, so we cannot use `get_mut`.
        let alloc = this.memory().get(ptr.alloc_id)?;
        alloc.check_bounds(this, ptr, size)?;
        // Update the stacks.
        if let Borrow::Alias(Some(_)) = new_bor {
            // Reference that cares about freezing. We need a frozen-sensitive reborrow.
            this.visit_freeze_sensitive(place, size, |cur_ptr, size, frozen| {
                let kind = if frozen { RefKind::Frozen } else { RefKind::Raw };
                alloc.extra.reborrow(cur_ptr, size, barrier, new_bor, kind)
            })?;
        } else {
            // Just treat this as one big chunk.
            let kind = if new_bor.is_unique() { RefKind::Unique } else { RefKind::Raw };
            alloc.extra.reborrow(ptr, size, barrier, new_bor, kind)?;
        }
        Ok(())
    }

    /// Retags an indidual pointer, returning the retagged version.
    /// `mutbl` can be `None` to make this a raw pointer.
    fn retag_reference(
        &mut self,
        val: ImmTy<'tcx, Borrow>,
        mutbl: Option<Mutability>,
        fn_barrier: bool,
        two_phase: bool,
    ) -> EvalResult<'tcx, Immediate<Borrow>> {
        let this = self.eval_context_mut();
        // We want a place for where the ptr *points to*, so we get one.
        let place = this.ref_to_mplace(val)?;
        let size = this.size_and_align_of_mplace(place)?
            .map(|(size, _)| size)
            .unwrap_or_else(|| place.layout.size);
        if size == Size::ZERO {
            // Nothing to do for ZSTs.
            return Ok(*val);
        }

        // Compute new borrow.
        let time = this.machine.stacked_borrows.increment_clock();
        let new_bor = match mutbl {
            Some(MutMutable) => Borrow::Uniq(time),
            Some(MutImmutable) => Borrow::Alias(Some(time)),
            None => Borrow::default(),
        };

        // Reborrow.
        this.reborrow(place, size, fn_barrier, new_bor)?;
        let new_place = place.with_tag(new_bor);
        // Handle two-phase borrows.
        if two_phase {
            assert!(mutbl == Some(MutMutable), "two-phase shared borrows make no sense");
            // We immediately share it, to allow read accesses
            let two_phase_time = this.machine.stacked_borrows.increment_clock();
            let two_phase_bor = Borrow::Alias(Some(two_phase_time));
            this.reborrow(new_place, size, false /* fn_barrier */, two_phase_bor)?;
        }

        // Return new pointer.
        Ok(new_place.to_ref())
    }
}

impl<'a, 'mir, 'tcx> EvalContextExt<'a, 'mir, 'tcx> for crate::MiriEvalContext<'a, 'mir, 'tcx> {}
pub trait EvalContextExt<'a, 'mir, 'tcx: 'a+'mir>: crate::MiriEvalContextExt<'a, 'mir, 'tcx> {
    fn tag_new_allocation(
        &mut self,
        id: AllocId,
        kind: MemoryKind<MiriMemoryKind>,
    ) -> Borrow {
        let this = self.eval_context_mut();
        let time = match kind {
            MemoryKind::Stack => {
                // New unique borrow. This `Uniq` is not accessible by the program,
                // so it will only ever be used when using the local directly (i.e.,
                // not through a pointer). That is, whenever we directly use a local, this will pop
                // everything else off the stack, invalidating all previous pointers,
                // and in particular, *all* raw pointers. This subsumes the explicit
                // `reset` which the blog post [1] says to perform when accessing a local.
                //
                // [1]: <https://www.ralfj.de/blog/2018/08/07/stacked-borrows.html>
                this.machine.stacked_borrows.increment_clock()
            }
            _ => {
                // Nothing to do for everything else.
                return Borrow::default()
            }
        };
        // Make this the active borrow for this allocation.
        let alloc = this
            .memory_mut()
            .get_mut(id)
            .expect("this is a new allocation; it must still exist");
        let size = Size::from_bytes(alloc.bytes.len() as u64);
        alloc.extra.first_item(BorStackItem::Uniq(time), size);
        Borrow::Uniq(time)
    }

    /// Called for value-to-place conversion. `mutability` is `None` for raw pointers.
    ///
    /// Note that this does *not* mean that all this memory will actually get accessed/referenced!
    /// We could be in the middle of `&(*var).1`.
    fn ptr_dereference(
        &self,
        place: MPlaceTy<'tcx, Borrow>,
        size: Size,
        mutability: Option<Mutability>,
    ) -> EvalResult<'tcx> {
        let this = self.eval_context_ref();
        trace!(
            "ptr_dereference: Accessing {} reference for {:?} (pointee {})",
            if let Some(mutability) = mutability {
                format!("{:?}", mutability)
            } else {
                format!("raw")
            },
            place.ptr, place.layout.ty
        );
        let ptr = place.ptr.to_ptr()?;
        if mutability.is_none() {
            // No further checks on raw derefs -- only the access itself will be checked.
            return Ok(());
        }

        // Get the allocation
        let alloc = this.memory().get(ptr.alloc_id)?;
        alloc.check_bounds(this, ptr, size)?;
        // If we got here, we do some checking, *but* we leave the tag unchanged.
        if let Borrow::Alias(Some(_)) = ptr.tag {
            assert_eq!(mutability, Some(MutImmutable));
            // We need a frozen-sensitive check.
            this.visit_freeze_sensitive(place, size, |cur_ptr, size, frozen| {
                let kind = if frozen { RefKind::Frozen } else { RefKind::Raw };
                alloc.extra.deref(cur_ptr, size, kind)
            })?;
        } else {
            // Just treat this as one big chunk.
            let kind = if mutability == Some(MutMutable) { RefKind::Unique } else { RefKind::Raw };
            alloc.extra.deref(ptr, size, kind)?;
        }

        // All is good.
        Ok(())
    }

    fn retag(
        &mut self,
        kind: RetagKind,
        place: PlaceTy<'tcx, Borrow>
    ) -> EvalResult<'tcx> {
        let this = self.eval_context_mut();
        // Determine mutability and whether to add a barrier.
        // Cannot use `builtin_deref` because that reports *immutable* for `Box`,
        // making it useless.
        fn qualify(ty: ty::Ty<'_>, kind: RetagKind) -> Option<(Option<Mutability>, bool)> {
            match ty.sty {
                // References are simple.
                ty::Ref(_, _, mutbl) => Some((Some(mutbl), kind == RetagKind::FnEntry)),
                // Raw pointers need to be enabled.
                ty::RawPtr(..) if kind == RetagKind::Raw => Some((None, false)),
                // Boxes do not get a barrier: barriers reflect that references outlive the call
                // they were passed in to; that's just not the case for boxes.
                ty::Adt(..) if ty.is_box() => Some((Some(MutMutable), false)),
                _ => None,
            }
        }

        // We need a visitor to visit all references. However, that requires
        // a `MemPlace`, so we have a fast path for reference types that
        // avoids allocating.
        if let Some((mutbl, barrier)) = qualify(place.layout.ty, kind) {
            // Fast path.
            let val = this.read_immediate(this.place_to_op(place)?)?;
            let val = this.retag_reference(val, mutbl, barrier, kind == RetagKind::TwoPhase)?;
            this.write_immediate(val, place)?;
            return Ok(());
        }
        let place = this.force_allocation(place)?;

        let mut visitor = RetagVisitor { ecx: this, kind };
        visitor.visit_value(place)?;

        // The actual visitor.
        struct RetagVisitor<'ecx, 'a, 'mir, 'tcx> {
            ecx: &'ecx mut MiriEvalContext<'a, 'mir, 'tcx>,
            kind: RetagKind,
        }
        impl<'ecx, 'a, 'mir, 'tcx>
            MutValueVisitor<'a, 'mir, 'tcx, Evaluator<'tcx>>
        for
            RetagVisitor<'ecx, 'a, 'mir, 'tcx>
        {
            type V = MPlaceTy<'tcx, Borrow>;

            #[inline(always)]
            fn ecx(&mut self) -> &mut MiriEvalContext<'a, 'mir, 'tcx> {
                &mut self.ecx
            }

            // Primitives of reference type, that is the one thing we are interested in.
            fn visit_primitive(&mut self, place: MPlaceTy<'tcx, Borrow>) -> EvalResult<'tcx>
            {
                // Cannot use `builtin_deref` because that reports *immutable* for `Box`,
                // making it useless.
                if let Some((mutbl, barrier)) = qualify(place.layout.ty, self.kind) {
                    let val = self.ecx.read_immediate(place.into())?;
                    let val = self.ecx.retag_reference(
                        val,
                        mutbl,
                        barrier,
                        self.kind == RetagKind::TwoPhase
                    )?;
                    self.ecx.write_immediate(val, place.into())?;
                }
                Ok(())
            }
        }

        Ok(())
    }
}
