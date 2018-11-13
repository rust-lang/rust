use std::cell::RefCell;

use rustc::ty::{self, layout::Size};
use rustc::hir::{Mutability, MutMutable, MutImmutable};

use crate::{
    EvalResult, EvalErrorKind, MiriEvalContext, HelpersEvalContextExt,
    MemoryKind, MiriMemoryKind, RangeMap, AllocId, Allocation, AllocationExtra,
    Pointer, MemPlace, Scalar, Immediate, ImmTy, PlaceTy, MPlaceTy,
};

pub type Timestamp = u64;

/// Information about which kind of borrow was used to create the reference this is tagged
/// with.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum Borrow {
    /// A unique (mutable) reference.
    Uniq(Timestamp),
    /// A shared reference.  This is also used by raw pointers, which do not track details
    /// of how or when they were created, hence the timestamp is optional.
    /// Shr(Some(_)) does NOT mean that the destination of this reference is frozen;
    /// that depends on the type!  Only those parts outside of an `UnsafeCell` are actually
    /// frozen.
    Shr(Option<Timestamp>),
}

impl Borrow {
    #[inline(always)]
    pub fn is_shared(self) -> bool {
        match self {
            Borrow::Shr(_) => true,
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
        Borrow::Shr(None)
    }
}

/// An item in the per-location borrow stack
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum BorStackItem {
    /// Indicates the unique reference that may mutate.
    Uniq(Timestamp),
    /// Indicates that the location has been shared.  Used for raw pointers, but
    /// also for shared references.  The latter *additionally* get frozen
    /// when there is no `UnsafeCell`.
    Shr,
    /// A barrier, tracking the function it belongs to by its index on the call stack
    #[allow(dead_code)] // for future use
    FnBarrier(usize)
}

impl BorStackItem {
    #[inline(always)]
    pub fn is_fn_barrier(self) -> bool {
        match self {
            BorStackItem::FnBarrier(_) => true,
            _ => false,
        }
    }
}

/// Extra per-location state
#[derive(Clone, Debug)]
pub struct Stack {
    borrows: Vec<BorStackItem>, // used as a stack; never empty
    frozen_since: Option<Timestamp>, // virtual frozen "item" on top of the stack
}

impl Default for Stack {
    fn default() -> Self {
        Stack {
            borrows: vec![BorStackItem::Shr],
            frozen_since: None,
        }
    }
}

impl Stack {
    #[inline(always)]
    pub fn is_frozen(&self) -> bool {
        self.frozen_since.is_some()
    }
}

/// What kind of reference is being used?
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum RefKind {
    /// &mut
    Unique,
    /// & without interior mutability
    Frozen,
    /// * (raw pointer) or & to `UnsafeCell`
    Raw,
}

/// Extra global machine state
#[derive(Clone, Debug)]
pub struct State {
    clock: Timestamp
}

impl State {
    pub fn new() -> State {
        State { clock: 0 }
    }

    fn increment_clock(&mut self) -> Timestamp {
        let val = self.clock;
        self.clock = val + 1;
        val
    }
}

/// Extra per-allocation state
#[derive(Clone, Debug, Default)]
pub struct Stacks {
    // Even reading memory can have effects on the stack, so we need a `RefCell` here.
    stacks: RefCell<RangeMap<Stack>>,
}

/// Core per-location operations: deref, access, create.
/// We need to make at least the following things true:
///
/// U1: After creating a Uniq, it is at the top (+unfrozen).
/// U2: If the top is Uniq (+unfrozen), accesses must be through that Uniq or pop it.
/// U3: If an access (deref sufficient?) happens with a Uniq, it requires the Uniq to be in the stack.
///
/// F1: After creating a &, the parts outside `UnsafeCell` are frozen.
/// F2: If a write access happens, it unfreezes.
/// F3: If an access (well, a deref) happens with an & outside `UnsafeCell`, it requires the location to still be frozen.
impl<'tcx> Stack {
    /// Deref `bor`: Check if the location is frozen and the tag in the stack.
    /// This dos *not* constitute an access!  "Deref" refers to the `*` operator
    /// in Rust, and includs cases like `&*x` or `(*x).foo` where no or only part
    /// of the memory actually gets accessed.  Also we cannot know if we are
    /// going to read or write.
    /// Returns the index of the item we matched, `None` if it was the frozen one.
    /// `kind` indicates which kind of reference is being dereferenced.
    fn deref(&self, bor: Borrow, kind: RefKind) -> Result<Option<usize>, String> {
        // Checks related to freezing
        match bor {
            Borrow::Shr(Some(bor_t)) if kind == RefKind::Frozen => {
                // We need the location to be frozen. This ensures F3.
                let frozen = self.frozen_since.map_or(false, |itm_t| itm_t <= bor_t);
                return if frozen { Ok(None) } else {
                    Err(format!("Location is not frozen long enough"))
                }
            }
            Borrow::Shr(_) if self.frozen_since.is_some() => {
                return Ok(None) // Shared deref to frozen location, looking good
            }
            _ => {} // Not sufficient, go on looking.
        }
        // If we got here, we have to look for our item in the stack.
        for (idx, &itm) in self.borrows.iter().enumerate().rev() {
            match (itm, bor) {
                (BorStackItem::FnBarrier(_), _) => break,
                (BorStackItem::Uniq(itm_t), Borrow::Uniq(bor_t)) if itm_t == bor_t => {
                    // Found matching unique item.  This satisfies U3.
                    return Ok(Some(idx))
                }
                (BorStackItem::Shr, Borrow::Shr(_)) => {
                    // Found matching shared/raw item.
                    return Ok(Some(idx))
                }
                // Go on looking.
                _ => {}
            }
        }
        // If we got here, we did not find our item.  We have to error to satisfy U3.
        Err(format!(
            "Borrow being dereferenced ({:?}) does not exist on the stack, or is guarded by a barrier",
            bor
        ))
    }

    /// Perform an actual memory access using `bor`.  We do not know any types here
    /// or whether things should be frozen, but we *do* know if this is reading
    /// or writing.
    fn access(&mut self, bor: Borrow, is_write: bool) -> EvalResult<'tcx> {
        // Check if we can match the frozen "item".
        // Not possible on writes!
        if self.is_frozen() {
            if !is_write {
                // When we are frozen, we just accept all reads.  No harm in this.
                // The deref already checked that `Uniq` items are in the stack, and that
                // the location is frozen if it should be.
                return Ok(());
            }
            trace!("access: Unfreezing");
        }
        // Unfreeze on writes.  This ensures F2.
        self.frozen_since = None;
        // Pop the stack until we have something matching.
        while let Some(&itm) = self.borrows.last() {
            match (itm, bor) {
                (BorStackItem::FnBarrier(_), _) => break,
                (BorStackItem::Uniq(itm_t), Borrow::Uniq(bor_t)) if itm_t == bor_t => {
                    // Found matching unique item.
                    return Ok(())
                }
                (BorStackItem::Shr, _) if !is_write => {
                    // When reading, everything can use a shared item!
                    // We do not want to do this when writing: Writing to an `&mut`
                    // should reaffirm its exclusivity (i.e., make sure it is
                    // on top of the stack).
                    return Ok(())
                }
                (BorStackItem::Shr, Borrow::Shr(_)) => {
                    // Found matching shared item.
                    return Ok(())
                }
                _ => {
                    // Pop this.  This ensures U2.
                    let itm = self.borrows.pop().unwrap();
                    trace!("access: Popping {:?}", itm);
                }
            }
        }
        // If we got here, we did not find our item.
        err!(MachineError(format!(
            "Borrow being accessed ({:?}) does not exist on the stack, or is guarded by a barrier",
            bor
        )))
    }

    /// Initiate `bor`; mostly this means pushing.
    /// This operation cannot fail; it is up to the caller to ensure that the precondition
    /// is met: We cannot push `Uniq` onto frozen stacks.
    /// `kind` indicates which kind of reference is being created.
    fn create(&mut self, bor: Borrow, kind: RefKind) {
        // First, push the item.  We do this even if we will later freeze, because we
        // will allow mutation of shared data at the expense of unfreezing.
        if self.frozen_since.is_some() {
            // A frozen location, this should be impossible!
            bug!("We should never try pushing to a frozen stack");
        }
        // First, push.
        let itm = match bor {
            Borrow::Uniq(t) => BorStackItem::Uniq(t),
            Borrow::Shr(_) => BorStackItem::Shr,
        };
        if *self.borrows.last().unwrap() == itm {
            assert!(bor.is_shared());
            trace!("create: Sharing a shared location is a NOP");
        } else {
            // This ensures U1.
            trace!("create: Pushing {:?}", itm);
            self.borrows.push(itm);
        }
        // Then, maybe freeze.  This is part 2 of ensuring F1.
        if kind == RefKind::Frozen {
            let bor_t = match bor {
                Borrow::Shr(Some(t)) => t,
                _ => bug!("Creating illegal borrow {:?} for frozen ref", bor),
            };
            trace!("create: Freezing");
            self.frozen_since = Some(bor_t);
        }
    }
}

/// Higher-level per-location operations: deref, access, reborrow.
impl<'tcx> Stacks {
    /// Check that this stack is fine with being dereferenced
    fn deref(
        &self,
        ptr: Pointer<Borrow>,
        size: Size,
        kind: RefKind,
    ) -> EvalResult<'tcx> {
        trace!("deref for tag {:?} as {:?}: {:?}, size {}",
            ptr.tag, kind, ptr, size.bytes());
        let mut stacks = self.stacks.borrow_mut();
        // We need `iter_mut` because `iter` would skip gaps!
        for stack in stacks.iter_mut(ptr.offset, size) {
            stack.deref(ptr.tag, kind).map_err(EvalErrorKind::MachineError)?;
        }
        Ok(())
    }

    /// `ptr` got used, reflect that in the stack.
    fn access(
        &self,
        ptr: Pointer<Borrow>,
        size: Size,
        is_write: bool,
    ) -> EvalResult<'tcx> {
        trace!("{} access of tag {:?}: {:?}, size {}",
            if is_write { "read" } else { "write" },
            ptr.tag, ptr, size.bytes());
        let mut stacks = self.stacks.borrow_mut();
        for stack in stacks.iter_mut(ptr.offset, size) {
            stack.access(ptr.tag, is_write)?;
        }
        Ok(())
    }

    /// Reborrow the given pointer to the new tag for the given kind of reference.
    fn reborrow(
        &self,
        ptr: Pointer<Borrow>,
        size: Size,
        new_bor: Borrow,
        new_kind: RefKind,
    ) -> EvalResult<'tcx> {
        trace!("reborrow for tag {:?} to {:?} as {:?}: {:?}, size {}",
            ptr.tag, new_bor, new_kind, ptr, size.bytes());
        let mut stacks = self.stacks.borrow_mut();
        for stack in stacks.iter_mut(ptr.offset, size) {
            // Access source `ptr`, create new ref.
            let ptr_idx = stack.deref(ptr.tag, new_kind).map_err(EvalErrorKind::MachineError)?;
            // If we can deref the new tag already, and if that tag lives higher on
            // the stack than the one we come from, just use that.
            // IOW, we check if `new_bor` *already* is "derived from" `ptr.tag`.
            // This also checks frozenness, if required.
            let bor_redundant = match (ptr_idx, stack.deref(new_bor, new_kind)) {
                // If the new borrow works with the frozen item, or else if it lives
                // above the old one in the stack, our job here is done.
                (_, Ok(None)) => true,
                (Some(ptr_idx), Ok(Some(new_idx))) if new_idx >= ptr_idx => true,
                // Otherwise we need to create a new borrow.
                _ => false,
            };
            if bor_redundant {
                assert!(new_bor.is_shared(), "A unique reborrow can never be redundant");
                trace!("reborrow is redundant");
                continue;
            }
            // We need to do some actual work.
            stack.access(ptr.tag, new_kind == RefKind::Unique)?;
            stack.create(new_bor, new_kind);
        }
        Ok(())
    }
}

/// Hooks and glue
impl AllocationExtra<Borrow> for Stacks {
    #[inline(always)]
    fn memory_read<'tcx>(
        alloc: &Allocation<Borrow, Stacks>,
        ptr: Pointer<Borrow>,
        size: Size,
    ) -> EvalResult<'tcx> {
        alloc.extra.access(ptr, size, /*is_write*/false)
    }

    #[inline(always)]
    fn memory_written<'tcx>(
        alloc: &mut Allocation<Borrow, Stacks>,
        ptr: Pointer<Borrow>,
        size: Size,
    ) -> EvalResult<'tcx> {
        alloc.extra.access(ptr, size, /*is_write*/true)
    }

    #[inline(always)]
    fn memory_deallocated<'tcx>(
        alloc: &mut Allocation<Borrow, Stacks>,
        ptr: Pointer<Borrow>,
        size: Size,
    ) -> EvalResult<'tcx> {
        // This is like mutating
        alloc.extra.access(ptr, size, /*is_write*/true)
        // FIXME: Error out of there are any barriers?
    }
}

impl<'tcx> Stacks {
    /// Pushes the first item to the stacks.
    pub fn first_item(
        &mut self,
        itm: BorStackItem,
        size: Size
    ) {
        assert!(!itm.is_fn_barrier());
        for stack in self.stacks.get_mut().iter_mut(Size::ZERO, size) {
            assert!(stack.borrows.len() == 1);
            assert_eq!(stack.borrows.pop().unwrap(), BorStackItem::Shr);
            stack.borrows.push(itm);
        }
    }
}



pub trait EvalContextExt<'tcx> {
    fn tag_dereference(
        &self,
        place: MPlaceTy<'tcx, Borrow>,
        size: Size,
        mutability: Option<Mutability>,
    ) -> EvalResult<'tcx, Borrow>;

    fn tag_new_allocation(
        &mut self,
        id: AllocId,
        kind: MemoryKind<MiriMemoryKind>,
    ) -> Borrow;

    /// Retag an indidual pointer, returning the retagged version.
    fn reborrow(
        &mut self,
        ptr: ImmTy<'tcx, Borrow>,
        mutbl: Mutability,
    ) -> EvalResult<'tcx, Immediate<Borrow>>;

    fn retag(
        &mut self,
        fn_entry: bool,
        place: PlaceTy<'tcx, Borrow>
    ) -> EvalResult<'tcx>;

    fn escape_to_raw(
        &mut self,
        place: MPlaceTy<'tcx, Borrow>,
        size: Size,
    ) -> EvalResult<'tcx>;
}

impl<'a, 'mir, 'tcx> EvalContextExt<'tcx> for MiriEvalContext<'a, 'mir, 'tcx> {
    fn tag_new_allocation(
        &mut self,
        id: AllocId,
        kind: MemoryKind<MiriMemoryKind>,
    ) -> Borrow {
        let time = match kind {
            MemoryKind::Stack => {
                // New unique borrow. This `Uniq` is not accessible by the program,
                // so it will only ever be used when using the local directly (i.e.,
                // not through a pointer).  IOW, whenever we directly use a local this will pop
                // everything else off the stack, invalidating all previous pointers
                // and, in particular, *all* raw pointers.  This subsumes the explicit
                // `reset` which the blog post [1] says to perform when accessing a local.
                //
                // [1] https://www.ralfj.de/blog/2018/08/07/stacked-borrows.html
                self.machine.stacked_borrows.increment_clock()
            }
            _ => {
                // Nothing to do for everything else
                return Borrow::default()
            }
        };
        // Make this the active borrow for this allocation
        let alloc = self.memory_mut().get_mut(id).expect("This is a new allocation, it must still exist");
        let size = Size::from_bytes(alloc.bytes.len() as u64);
        alloc.extra.first_item(BorStackItem::Uniq(time), size);
        Borrow::Uniq(time)
    }

    /// Called for value-to-place conversion.  `mutability` is `None` for raw pointers.
    ///
    /// Note that this does NOT mean that all this memory will actually get accessed/referenced!
    /// We could be in the middle of `&(*var).1`.
    fn tag_dereference(
        &self,
        place: MPlaceTy<'tcx, Borrow>,
        size: Size,
        mutability: Option<Mutability>,
    ) -> EvalResult<'tcx, Borrow> {
        trace!("tag_dereference: Accessing {} reference for {:?} (pointee {})",
            if let Some(mutability) = mutability { format!("{:?}", mutability) } else { format!("raw") },
            place.ptr, place.layout.ty);
        let ptr = place.ptr.to_ptr()?;
        // In principle we should not have to do anything here.  However, with transmutes involved,
        // it can happen that the tag of `ptr` does not actually match `mutability`, and we
        // should adjust for that.
        // Notably, the compiler can introduce such transmutes by optimizing away `&[mut]*`.
        // That can transmute a raw ptr to a (shared/mut) ref, and a mut ref to a shared one.
        match (mutability, ptr.tag) {
            (None, _) => {
                // Don't use the tag, this is a raw access!  They should happen tagless.
                // This is needed for `*mut` to make any sense: Writes *do* enforce the
                // `Uniq` tag to be up top, but we must make sure raw writes do not do that.
                // This does mean, however, that `&*foo` is *not* a NOP *if* `foo` is a raw ptr.
                // Also don't do any further validation, this is raw after all.
                return Ok(Borrow::default());
            }
            (Some(MutMutable), Borrow::Uniq(_)) |
            (Some(MutImmutable), Borrow::Shr(_)) => {
                // Expected combinations.  Nothing to do.
            }
            (Some(MutMutable), Borrow::Shr(None)) => {
                // Raw transmuted to mut ref.  This is something real unsafe code does.
                // We cannot reborrow here because we do not want to mutate state on a deref.
            }
            (Some(MutImmutable), Borrow::Uniq(_)) => {
                // A mut got transmuted to shr.  Can happen even from compiler transformations:
                // `&*x` gets optimized to `x` even when `x` is a `&mut`.
            }
            (Some(MutMutable), Borrow::Shr(Some(_))) => {
                // This is just invalid: A shr got transmuted to a mut.
                // If we ever allow this, we have to consider what we do when a turn a
                // `Raw`-tagged `&mut` into a raw pointer pointing to a frozen location.
                // We probably do not want to allow that, but we have to allow
                // turning a `Raw`-tagged `&` into a raw ptr to a frozen location.
                return err!(MachineError(format!("Encountered mutable reference with frozen tag {:?}", ptr.tag)))
            }
        }

        // Get the allocation
        self.memory().check_bounds(ptr, size, false)?;
        let alloc = self.memory().get(ptr.alloc_id).expect("We checked that the ptr is fine!");
        // If we got here, we do some checking, *but* we leave the tag unchanged.
        if let Borrow::Shr(Some(_)) = ptr.tag {
            assert_eq!(mutability, Some(MutImmutable));
            // We need a frozen-sensitive check
            self.visit_freeze_sensitive(place, size, |cur_ptr, size, frozen| {
                let kind = if frozen { RefKind::Frozen } else { RefKind::Raw };
                alloc.extra.deref(cur_ptr, size, kind)
            })?;
        } else {
            // Just treat this as one big chunk
            let kind = if mutability == Some(MutMutable) { RefKind::Unique } else { RefKind::Raw };
            alloc.extra.deref(ptr, size, kind)?;
        }

        // All is good, and do not change the tag
        Ok(ptr.tag)
    }

    /// The given place may henceforth be accessed through raw pointers.
    fn escape_to_raw(
        &mut self,
        place: MPlaceTy<'tcx, Borrow>,
        size: Size,
    ) -> EvalResult<'tcx> {
        trace!("escape_to_raw: {:?} is now accessible by raw pointers", *place);
        // Get the allocation
        let ptr = place.ptr.to_ptr()?;
        self.memory().check_bounds(ptr, size, false)?; // `ptr_dereference` wouldn't do any checks if this is a raw ptr
        let alloc = self.memory().get(ptr.alloc_id).expect("We checked that the ptr is fine!");
        // Re-borrow to raw.  This is a NOP for shared borrows, but we do not know the borrow
        // type here and that's also okay.  Freezing does not matter here.
        alloc.extra.reborrow(ptr, size, Borrow::default(), RefKind::Raw)
    }

    fn reborrow(
        &mut self,
        val: ImmTy<'tcx, Borrow>,
        mutbl: Mutability,
    ) -> EvalResult<'tcx, Immediate<Borrow>> {
        // We want a place for where the ptr *points to*, so we get one.
        let place = self.ref_to_mplace(val)?;
        let size = self.size_and_align_of_mplace(place)?
            .map(|(size, _)| size)
            .unwrap_or_else(|| place.layout.size);
        if size == Size::ZERO {
            // Nothing to do for ZSTs.
            return Ok(*val);
        }

        // Prepare to re-borrow this place.
        let ptr = place.ptr.to_ptr()?;
        let time = self.machine.stacked_borrows.increment_clock();
        let new_bor = match mutbl {
            MutMutable => Borrow::Uniq(time),
            MutImmutable => Borrow::Shr(Some(time)),
        };
        trace!("reborrow: Creating new {:?} reference for {:?} (pointee {}): {:?}",
            mutbl, ptr, place.layout.ty, new_bor);

        // Get the allocation.  It might not be mutable, so we cannot use `get_mut`.
        self.memory().check_bounds(ptr, size, false)?;
        let alloc = self.memory().get(ptr.alloc_id).expect("We checked that the ptr is fine!");
        // Update the stacks.
        if mutbl == MutImmutable {
            // Shared reference. We need a frozen-sensitive reborrow.
            self.visit_freeze_sensitive(place, size, |cur_ptr, size, frozen| {
                let kind = if frozen { RefKind::Frozen } else { RefKind::Raw };
                alloc.extra.reborrow(cur_ptr, size, new_bor, kind)
            })?;
        } else {
            // Mutable reference. Just treat this as one big chunk.
            alloc.extra.reborrow(ptr, size, new_bor, RefKind::Unique)?;
        }

        // Return new ptr
        let new_ptr = Pointer::new_with_tag(ptr.alloc_id, ptr.offset, new_bor);
        let new_place = MemPlace { ptr: Scalar::Ptr(new_ptr), ..*place };
        Ok(new_place.to_ref())
    }

    fn retag(
        &mut self,
        _fn_entry: bool,
        place: PlaceTy<'tcx, Borrow>
    ) -> EvalResult<'tcx> {
        // For now, we only retag if the toplevel type is a reference.
        // TODO: Recurse into structs and enums, sharing code with validation.
        // TODO: Honor `fn_entry`.
        let mutbl = match place.layout.ty.sty {
            ty::Ref(_, _, mutbl) => mutbl, // go ahead
            _ => return Ok(()), // do nothing, for now
        };
        // Retag the pointer and write it back.
        let val = self.read_immediate(self.place_to_op(place)?)?;
        let val = self.reborrow(val, mutbl)?;
        self.write_immediate(val, place)?;
        Ok(())
    }
}
