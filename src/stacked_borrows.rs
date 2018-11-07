use std::cell::RefCell;

use rustc::ty::{self, layout::Size};
use rustc::hir;

use crate::{
    EvalResult, MiriEvalContext, HelpersEvalContextExt,
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
    pub fn is_shr(self) -> bool {
        match self {
            Borrow::Shr(_) => true,
            _ => false,
        }
    }

    #[inline(always)]
    pub fn is_uniq(self) -> bool {
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

/// What kind of usage of the pointer are we talking about?
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum UsageKind {
    /// Write, or create &mut
    Write,
    /// Read, or create &
    Read,
    /// Create * (raw ptr)
    Raw,
}

impl From<Option<hir::Mutability>> for UsageKind {
    fn from(mutbl: Option<hir::Mutability>) -> Self {
        match mutbl {
            None => UsageKind::Raw,
            Some(hir::MutMutable) => UsageKind::Write,
            Some(hir::MutImmutable) => UsageKind::Read,
        }
    }
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
}

/// Extra per-allocation state
#[derive(Clone, Debug, Default)]
pub struct Stacks {
    // Even reading memory can have effects on the stack, so we need a `RefCell` here.
    stacks: RefCell<RangeMap<Stack>>,
}

/// Core operations
impl<'tcx> Stack {
    /// Check if `bor` could be activated by unfreezing and popping.
    /// `is_write` indicates whether this is being used to write (or, equivalently, to
    /// borrow as &mut).
    /// Returns `Err` if the answer is "no"; otherwise the return value indicates what to
    /// do: With `Some(n)` you need to unfreeze, and then additionally pop `n` items.
    fn reactivatable(&self, bor: Borrow, is_write: bool) -> Result<Option<usize>, String> {
        // Check if we can match the frozen "item".  Not possible on writes!
        if !is_write {
            // For now, we do NOT check the timestamp.  That might be surprising, but
            // we cannot even notice when a location should be frozen but is not!
            // Those checks are both done in `tag_dereference`, where we have type information.
            // Either way, it is crucial that the frozen "item" matches raw pointers:
            // Reading through a raw should not unfreeze.
            match (self.frozen_since, bor) {
                (Some(_), Borrow::Shr(_)) => {
                    return Ok(None)
                }
                _ => {},
            }
        }
        // See if we can find this borrow.
        for (idx, &itm) in self.borrows.iter().rev().enumerate() {
            // Check borrow and stack item for compatibility.
            match (itm, bor) {
                (BorStackItem::FnBarrier(_), _) => {
                    return Err(format!("Trying to reactivate a borrow ({:?}) that lives \
                                        behind a barrier", bor))
                }
                (BorStackItem::Uniq(itm_t), Borrow::Uniq(bor_t)) if itm_t == bor_t => {
                    // Found matching unique item.
                    if !is_write {
                        // As a special case, if we are reading and since we *did* find the `Uniq`,
                        // we try to pop less: We are happy with making a `Shr` or `Frz` active;
                        // that one will not mind concurrent reads.
                        match self.reactivatable(Borrow::default(), is_write) {
                            // If we got something better that `idx`, use that
                            Ok(None) => return Ok(None),
                            Ok(Some(shr_idx)) if shr_idx <= idx => return Ok(Some(shr_idx)),
                            // Otherwise just go on.
                            _ => {},
                        }
                    }
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
        // Nothing to be found.
        Err(format!("Borrow-to-reactivate {:?} does not exist on the stack", bor))
    }

    /// Reactive `bor` for this stack.  `is_write` indicates whether this is being
    /// used to write (or, equivalently, to borrow as &mut).
    fn reactivate(&mut self, bor: Borrow, is_write: bool) -> EvalResult<'tcx> {
        let mut pop = match self.reactivatable(bor, is_write) {
            Ok(None) => return Ok(()),
            Ok(Some(pop)) => pop,
            Err(err) => return err!(MachineError(err)),
        };
        // Pop what `reactivatable` told us to pop. Always unfreeze.
        if self.is_frozen() {
            trace!("reactivate: Unfreezing");
        }
        self.frozen_since = None;
        while pop > 0 {
            let itm = self.borrows.pop().unwrap();
            trace!("reactivate: Popping {:?}", itm);
            pop -= 1;
        }
        Ok(())
    }

    /// Initiate `bor`; mostly this means pushing.
    /// This operation cannot fail; it is up to the caller to ensure that the precondition
    /// is met: We cannot push `Uniq` onto frozen stacks.
    /// Crucially, this makes pushing a `Shr` onto a frozen location a NOP.  We do not want
    /// such a location to get mutably shared this way!
    fn initiate(&mut self, bor: Borrow) {
        if let Some(_) = self.frozen_since {
            // A frozen location, we won't change anything here!
            match bor {
                Borrow::Uniq(_) => bug!("Trying to create unique ref to frozen location"),
                Borrow::Shr(_) => trace!("initiate: New shared ref to frozen location is a NOP"),
            }
        } else {
            // Just push.
            let itm = match bor {
                Borrow::Uniq(t) => BorStackItem::Uniq(t),
                Borrow::Shr(_) if *self.borrows.last().unwrap() == BorStackItem::Shr => {
                    // Optimization: Don't push a Shr onto a Shr.
                    trace!("initiate: New shared ref to already shared location is a NOP");
                    return
                },
                Borrow::Shr(_) => BorStackItem::Shr,
            };
            trace!("initiate: Pushing {:?}", itm);
            self.borrows.push(itm)
        }
    }

    /// Check if this location is "frozen enough".
    fn check_frozen(&self, bor_t: Timestamp) -> EvalResult<'tcx> {
        let frozen = self.frozen_since.map_or(false, |itm_t| itm_t <= bor_t);
        if !frozen {
            err!(MachineError(format!("Location is not frozen long enough")))
        } else {
            Ok(())
        }
    }

    /// Freeze this location, since `bor_t`.
    fn freeze(&mut self, bor_t: Timestamp) {
        if let Some(itm_t) = self.frozen_since {
            assert!(itm_t <= bor_t, "Trying to freeze shorter than it was frozen?");
        } else {
            trace!("Freezing");
            self.frozen_since = Some(bor_t);
        }
    }
}

impl State {
    fn increment_clock(&mut self) -> Timestamp {
        let val = self.clock;
        self.clock = val + 1;
        val
    }
}

/// Higher-level operations
impl<'tcx> Stacks {
    /// `ptr` got used, reflect that in the stack.
    fn reactivate(
        &self,
        ptr: Pointer<Borrow>,
        size: Size,
        usage: UsageKind,
    ) -> EvalResult<'tcx> {
        trace!("use_borrow of tag {:?} as {:?}: {:?}, size {}",
            ptr.tag, usage, ptr, size.bytes());
        let mut stacks = self.stacks.borrow_mut();
        for stack in stacks.iter_mut(ptr.offset, size) {
            stack.reactivate(ptr.tag, usage == UsageKind::Write)?;
        }
        Ok(())
    }

    /// Create a new borrow, the ptr must already have the new tag.
    /// Also freezes the location if `freeze` is set and the tag is a timestamped `Shr`.
    fn initiate(
        &self,
        ptr: Pointer<Borrow>,
        size: Size,
        freeze: bool,
    ) {
        trace!("reborrow for tag {:?}: {:?}, size {}",
            ptr.tag, ptr, size.bytes());
        let mut stacks = self.stacks.borrow_mut();
        for stack in stacks.iter_mut(ptr.offset, size) {
            stack.initiate(ptr.tag);
            if freeze {
                if let Borrow::Shr(Some(bor_t)) = ptr.tag {
                    stack.freeze(bor_t);
                }
            }
        }
    }

    /// Check that this stack is fine with being dereferenced
    fn check_deref(
        &self,
        ptr: Pointer<Borrow>,
        size: Size,
        frozen: bool,
    ) -> EvalResult<'tcx> {
        let mut stacks = self.stacks.borrow_mut();
        // We need `iter_mut` because `iter` would skip gaps!
        for stack in stacks.iter_mut(ptr.offset, size) {
            // Conservatively assume we will just read
            if let Err(err) = stack.reactivatable(ptr.tag, /*is_write*/false) {
                return err!(MachineError(format!(
                    "Encountered reference with non-reactivatable tag: {}",
                    err
                )))
            }
            // Sometimes we also need to be frozen.
            if frozen {
                // Even shared refs can have uniq tags (after transmute).  That's not an error
                // but they do not get any freezing benefits.
                if let Borrow::Shr(Some(bor_t)) = ptr.tag {
                    stack.check_frozen(bor_t)?;
                }
            }
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
        // Reads behave exactly like the first half of a reborrow-to-shr
        alloc.extra.reactivate(ptr, size, UsageKind::Read)
    }

    #[inline(always)]
    fn memory_written<'tcx>(
        alloc: &mut Allocation<Borrow, Stacks>,
        ptr: Pointer<Borrow>,
        size: Size,
    ) -> EvalResult<'tcx> {
        // Writes behave exactly like the first half of a reborrow-to-mut
        alloc.extra.reactivate(ptr, size, UsageKind::Read)
    }

    #[inline(always)]
    fn memory_deallocated<'tcx>(
        alloc: &mut Allocation<Borrow, Stacks>,
        ptr: Pointer<Borrow>,
        size: Size,
    ) -> EvalResult<'tcx> {
        // This is like mutating
        alloc.extra.reactivate(ptr, size, UsageKind::Write)
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
        usage: UsageKind,
    ) -> EvalResult<'tcx, Borrow>;

    fn tag_new_allocation(
        &mut self,
        id: AllocId,
        kind: MemoryKind<MiriMemoryKind>,
    ) -> Borrow;

    /// Retag an indidual pointer, returning the retagged version.
    fn retag_ptr(
        &mut self,
        ptr: ImmTy<'tcx, Borrow>,
        mutbl: hir::Mutability,
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

    /// Called for value-to-place conversion.
    ///
    /// Note that this does NOT mean that all this memory will actually get accessed/referenced!
    /// We could be in the middle of `&(*var).1`.
    fn tag_dereference(
        &self,
        place: MPlaceTy<'tcx, Borrow>,
        size: Size,
        usage: UsageKind,
    ) -> EvalResult<'tcx, Borrow> {
        trace!("tag_dereference: Accessing reference ({:?}) for {:?} (pointee {})",
            usage, place.ptr, place.layout.ty);
        let ptr = place.ptr.to_ptr()?;
        // In principle we should not have to do anything here.  However, with transmutes involved,
        // it can happen that the tag of `ptr` does not actually match `usage`, and we
        // should adjust for that.
        // Notably, the compiler can introduce such transmutes by optimizing away `&[mut]*`.
        // That can transmute a raw ptr to a (shared/mut) ref, and a mut ref to a shared one.
        match (usage, ptr.tag) {
            (UsageKind::Raw, _) => {
                // Don't use the tag, this is a raw access!  They should happen tagless.
                // This does mean, however, that `&*foo` is *not* a NOP *if* `foo` is a raw ptr.
                // Also don't do any further validation, this is raw after all.
                return Ok(Borrow::default());
            }
            (UsageKind::Write, Borrow::Uniq(_)) |
            (UsageKind::Read, Borrow::Shr(_)) => {
                // Expected combinations.  Nothing to do.
            }
            (UsageKind::Write, Borrow::Shr(None)) => {
                // Raw transmuted to mut ref.  Keep this as raw access.
                // We cannot reborrow here; there might be a raw in `&(*var).1` where
                // `var` is an `&mut`.  The other field of the struct might be already frozen,
                // also using `var`, and that would be okay.
            }
            (UsageKind::Read, Borrow::Uniq(_)) => {
                // A mut got transmuted to shr.  Can happen even from compiler transformations:
                // `&*x` gets optimized to `x` even when `x` is a `&mut`.
            }
            (UsageKind::Write, Borrow::Shr(Some(_))) => {
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
            // We need a frozen-sensitive check
            self.visit_freeze_sensitive(place, size, |cur_ptr, size, frozen| {
                alloc.extra.check_deref(cur_ptr, size, frozen)
            })?;
        } else {
            // Just treat this as one big chunk
            alloc.extra.check_deref(ptr, size, /*frozen*/false)?;
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
        trace!("self: {:?} is now accessible by raw pointers", *place);
        // Get the allocation
        let mut ptr = place.ptr.to_ptr()?;
        self.memory().check_bounds(ptr, size, false)?; // `ptr_dereference` wouldn't do any checks if this is a raw ptr
        let alloc = self.memory().get(ptr.alloc_id).expect("We checked that the ptr is fine!");
        // Re-borrow to raw.  This is a NOP for shared borrows, but we do not know the borrow
        // type here and that's also okay.  Freezing does not matter here.
        alloc.extra.reactivate(ptr, size, UsageKind::Raw)?;
        ptr.tag = Borrow::default();
        alloc.extra.initiate(ptr, size, /*freeze*/false);
        Ok(())
    }

    fn retag_ptr(
        &mut self,
        val: ImmTy<'tcx, Borrow>,
        mutbl: hir::Mutability,
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
            hir::MutMutable => Borrow::Uniq(time),
            hir::MutImmutable => Borrow::Shr(Some(time)),
        };
        let new_ptr = Pointer::new_with_tag(ptr.alloc_id, ptr.offset, new_bor);
        trace!("retag: Creating new reference ({:?}) for {:?} (pointee {}): {:?}",
            mutbl, ptr, place.layout.ty, new_bor);

        // Get the allocation
        self.memory().check_bounds(ptr, size, false)?; // `ptr_dereference` wouldn't do any checks if this is a raw ptr
        let alloc = self.memory().get(ptr.alloc_id).expect("We checked that the ptr is fine!");
        // Update the stacks.  First use old borrow, then initiate new one.
        alloc.extra.reactivate(ptr, size, Some(mutbl).into())?;
        if mutbl == hir::MutImmutable {
            // We need a frozen-sensitive initiate
            self.visit_freeze_sensitive(place, size, |mut cur_ptr, size, frozen| {
                cur_ptr.tag = new_bor;
                Ok(alloc.extra.initiate(cur_ptr, size, frozen))
            })?;
        } else {
            // Just treat this as one big chunk
            alloc.extra.initiate(new_ptr, size, /*frozen*/false);
        }

        // Return new ptr
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
        let val = self.retag_ptr(val, mutbl)?;
        self.write_immediate(val, place)?;
        Ok(())
    }
}
