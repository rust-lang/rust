use std::cell::RefCell;

use rustc::ty::{self, layout::Size};
use rustc::hir;

use crate::{
    EvalResult, MiriEvalContext, HelpersEvalContextExt,
    MemoryKind, MiriMemoryKind, RangeMap, AllocId,
    Pointer, PlaceTy, MPlaceTy,
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
    /// `usage` indicates whether this is being used to read/write (or, equivalently, to
    /// borrow as &/&mut), or to borrow as raw.
    /// Returns `Err` if the answer is "no"; otherwise the return value indicates what to
    /// do: With `Some(n)` you need to unfreeze, and then additionally pop `n` items.
    fn reactivatable(&self, bor: Borrow, usage: UsageKind) -> Result<Option<usize>, String> {
        // Check if we can match the frozen "item".  Not possible on writes!
        if usage != UsageKind::Write {
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
                    if usage == UsageKind::Read {
                        // As a special case, if we are reading and since we *did* find the `Uniq`,
                        // we try to pop less: We are happy with making a `Shr` or `Frz` active;
                        // that one will not mind concurrent reads.
                        match self.reactivatable(Borrow::default(), usage) {
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

    /// Reactive `bor` for this stack.  `usage` indicates whether this is being
    /// used to read/write (or, equivalently, to borrow as &/&mut), or to borrow as raw.
    fn reactivate(&mut self, bor: Borrow, usage: UsageKind) -> EvalResult<'tcx> {
        let mut pop = match self.reactivatable(bor, usage) {
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
    /// is met: We cannot push onto frozen stacks.
    fn initiate(&mut self, bor: Borrow) {
        if let Some(_) = self.frozen_since {
            // "Pushing" a Shr or Frz on top is redundant.
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
    /// The single most important operation: Make sure that using `ptr` as `usage` is okay,
    /// and if `new_bor` is present then make that the new current borrow.
    fn use_and_maybe_re_borrow(
        &self,
        ptr: Pointer<Borrow>,
        size: Size,
        usage: UsageKind,
        new_bor: Option<Borrow>,
    ) -> EvalResult<'tcx> {
        trace!("use_and_maybe_re_borrow of tag {:?} as {:?}, new {:?}: {:?}, size {}",
            ptr.tag, usage, new_bor, ptr, size.bytes());
        let mut stacks = self.stacks.borrow_mut();
        for stack in stacks.iter_mut(ptr.offset, size) {
            stack.reactivate(ptr.tag, usage)?;
            if let Some(new_bor) = new_bor {
                stack.initiate(new_bor);
            }
        }
        Ok(())
    }

    /// Freeze the given memory range.
    fn freeze(
        &self,
        ptr: Pointer<Borrow>,
        size: Size,
        bor_t: Timestamp
    ) -> EvalResult<'tcx> {
        let mut stacks = self.stacks.borrow_mut();
        for stack in stacks.iter_mut(ptr.offset, size) {
            stack.freeze(bor_t);
        }
        Ok(())
    }

    /// Check that this stack is fine with being dereferenced
    fn check_deref(
        &self,
        ptr: Pointer<Borrow>,
        size: Size,
    ) -> EvalResult<'tcx> {
        let mut stacks = self.stacks.borrow_mut();
        // We need `iter_mut` because `iter` would skip gaps!
        for stack in stacks.iter_mut(ptr.offset, size) {
            // Conservatively assume we will just read
            if let Err(err) = stack.reactivatable(ptr.tag, UsageKind::Read) {
                return err!(MachineError(format!(
                    "Encountered reference with non-reactivatable tag: {}",
                    err
                )))
            }
        }
        Ok(())
    }

    /// Check that this stack is appropriately frozen
    fn check_frozen(
        &self,
        ptr: Pointer<Borrow>,
        size: Size,
        bor_t: Timestamp
    ) -> EvalResult<'tcx> {
        let mut stacks = self.stacks.borrow_mut();
        for stack in stacks.iter_mut(ptr.offset, size) {
            stack.check_frozen(bor_t)?;
        }
        Ok(())
    }
}

/// Hooks and glue
impl<'tcx> Stacks {
    #[inline(always)]
    pub fn memory_read(
        &self,
        ptr: Pointer<Borrow>,
        size: Size,
    ) -> EvalResult<'tcx> {
        // Reads behave exactly like the first half of a reborrow-to-shr
        self.use_and_maybe_re_borrow(ptr, size, UsageKind::Read, None)
    }

    #[inline(always)]
    pub fn memory_written(
        &mut self,
        ptr: Pointer<Borrow>,
        size: Size,
    ) -> EvalResult<'tcx> {
        // Writes behave exactly like the first half of a reborrow-to-mut
        self.use_and_maybe_re_borrow(ptr, size, UsageKind::Write, None)
    }

    pub fn memory_deallocated(
        &mut self,
        ptr: Pointer<Borrow>,
        size: Size,
    ) -> EvalResult<'tcx> {
        // This is like mutating
        self.use_and_maybe_re_borrow(ptr, size, UsageKind::Write, None)
        // FIXME: Error out of there are any barriers?
    }

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
    fn tag_reference(
        &mut self,
        place: MPlaceTy<'tcx, Borrow>,
        size: Size,
        usage: UsageKind,
    ) -> EvalResult<'tcx, Borrow>;

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

    fn retag(
        &mut self,
        fn_entry: bool,
        place: PlaceTy<'tcx, Borrow>
    ) -> EvalResult<'tcx>;
}

impl<'a, 'mir, 'tcx> EvalContextExt<'tcx> for MiriEvalContext<'a, 'mir, 'tcx> {
    /// Called for place-to-value conversion.
    fn tag_reference(
        &mut self,
        place: MPlaceTy<'tcx, Borrow>,
        size: Size,
        usage: UsageKind,
    ) -> EvalResult<'tcx, Borrow> {
        let ptr = place.ptr.to_ptr()?;
        let time = self.machine.stacked_borrows.increment_clock();
        let new_bor = match usage {
            UsageKind::Write => Borrow::Uniq(time),
            UsageKind::Read => Borrow::Shr(Some(time)),
            UsageKind::Raw => Borrow::Shr(None),
        };
        trace!("tag_reference: Creating new reference ({:?}) for {:?} (pointee {}): {:?}",
            usage, ptr, place.layout.ty, new_bor);

        // Update the stacks.  First create the new ref as usual, then maybe freeze stuff.
        self.memory().check_bounds(ptr, size, false)?;
        let alloc = self.memory().get(ptr.alloc_id).expect("We checked that the ptr is fine!");
        alloc.extra.use_and_maybe_re_borrow(ptr, size, usage, Some(new_bor))?;
        // Maybe freeze stuff
        if let Borrow::Shr(Some(bor_t)) = new_bor {
            self.visit_frozen(place, size, |frz_ptr, size| {
                debug_assert_eq!(frz_ptr.alloc_id, ptr.alloc_id);
                // Be frozen!
                alloc.extra.freeze(frz_ptr, size, bor_t)
            })?;
        }

        Ok(new_bor)
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
        let ptr = place.ptr.to_ptr()?;
        trace!("tag_dereference: Accessing reference ({:?}) for {:?} (pointee {})",
            usage, ptr, place.layout.ty);
        // In principle we should not have to do anything here.  However, with transmutes involved,
        // it can happen that the tag of `ptr` does not actually match `usage`, and we
        // should adjust for that.
        // Notably, the compiler can introduce such transmutes by optimizing away `&[mut]*`.
        // That can transmute a raw ptr to a (shared/mut) ref, and a mut ref to a shared one.
        match (usage, ptr.tag) {
            (UsageKind::Raw, _) => {
                // Don't use the tag, this is a raw access!  Even if there is a tag,
                // that means transmute happened and we ignore the tag.
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

        // If we got here, we do some checking, *but* we leave the tag unchanged.
        self.memory().check_bounds(ptr, size, false)?;
        let alloc = self.memory().get(ptr.alloc_id).expect("We checked that the ptr is fine!");
        alloc.extra.check_deref(ptr, size)?;
        // Maybe check frozen stuff
        if let Borrow::Shr(Some(bor_t)) = ptr.tag {
            self.visit_frozen(place, size, |frz_ptr, size| {
                debug_assert_eq!(frz_ptr.alloc_id, ptr.alloc_id);
                // Are you frozen?
                alloc.extra.check_frozen(frz_ptr, size, bor_t)
            })?;
        }

        // All is good, and do not change the tag
        Ok(ptr.tag)
    }

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

    fn retag(
        &mut self,
        _fn_entry: bool,
        place: PlaceTy<'tcx, Borrow>
    ) -> EvalResult<'tcx> {
        // For now, we only retag if the toplevel type is a reference.
        // TODO: Recurse into structs and enums, sharing code with validation.
        let mutbl = match place.layout.ty.sty {
            ty::Ref(_, _, mutbl) => mutbl, // go ahead
            _ => return Ok(()), // don't do a thing
        };
        // We want to reborrow the reference stored there. This will call the hooks
        // above.  First deref, which will call `tag_dereference`.
        // (This is somewhat redundant because validation already did the same thing,
        // but what can you do.)
        let val = self.read_immediate(self.place_to_op(place)?)?;
        let dest = self.ref_to_mplace(val)?;
        // Now put a new ref into the old place, which will call `tag_reference`.
        // FIXME: Honor `fn_entry`!
        let val = self.create_ref(dest, Some(mutbl))?;
        self.write_immediate(val, place)?;
        Ok(())
    }
}
