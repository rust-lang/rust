use std::cell::{Cell, RefCell};

use rustc::ty::{Ty, layout::Size};
use rustc::hir;

use super::{
    MemoryAccess, MemoryKind, MiriMemoryKind, RangeMap, EvalResult, AllocId,
    Pointer,
};

pub type Timestamp = u64;

/// Information about a potentially mutable borrow
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum Mut {
    /// A unique, mutable reference
    Uniq(Timestamp),
    /// Any raw pointer, or a shared borrow with interior mutability
    Raw,
}

impl Mut {
    #[inline(always)]
    pub fn is_raw(self) -> bool {
        match self {
            Mut::Raw => true,
            _ => false,
        }
    }

    #[inline(always)]
    pub fn is_uniq(self) -> bool {
        match self {
            Mut::Uniq(_) => true,
            _ => false,
        }
    }
}

/// Information about any kind of borrow
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum Borrow {
    /// A mutable borrow, a raw pointer, or a shared borrow with interior mutability
    Mut(Mut),
    /// A shared borrow without interior mutability
    Frz(Timestamp)
}

impl Borrow {
    #[inline(always)]
    pub fn is_uniq(self) -> bool {
        match self {
            Borrow::Mut(m) => m.is_uniq(),
            _ => false,
        }
    }

    #[inline(always)]
    pub fn is_frz(self) -> bool {
        match self {
            Borrow::Frz(_) => true,
            _ => false,
        }
    }
}

/// An item in the borrow stack
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum BorStackItem {
    /// Defines which references are permitted to mutate *if* the location is not frozen
    Mut(Mut),
    /// A barrier, tracking the function it belongs to by its index on the call stack
    #[allow(dead_code)] // for future use
    FnBarrier(usize)
}

impl Default for Borrow {
    fn default() -> Self {
        Borrow::Mut(Mut::Raw)
    }
}

/// What kind of reference are we talking about?
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum RefKind {
    Mut,
    Shr,
    Raw,
}

impl From<Option<hir::Mutability>> for RefKind {
    fn from(mutbl: Option<hir::Mutability>) -> Self {
        match mutbl {
            None => RefKind::Raw,
            Some(hir::MutMutable) => RefKind::Mut,
            Some(hir::MutImmutable) => RefKind::Shr,
        }
    }
}

/// Extra global machine state
#[derive(Clone, Debug)]
pub struct State {
    clock: Cell<Timestamp>
}

impl State {
    pub fn new() -> State {
        State { clock: Cell::new(0) }
    }
}

/// Extra per-location state
#[derive(Clone, Debug)]
struct Stack {
    borrows: Vec<BorStackItem>, // used as a stack
    frozen_since: Option<Timestamp>,
}

impl Default for Stack {
    fn default() -> Self {
        Stack {
            borrows: vec![BorStackItem::Mut(Mut::Raw)],
            frozen_since: None,
        }
    }
}

/// Extra per-allocation state
#[derive(Clone, Debug, Default)]
pub struct Stacks {
    stacks: RefCell<RangeMap<Stack>>,
}

/// Core operations
impl<'tcx> Stack {
    /// Check if `bor` is currently active.  We accept a `Raw` on a frozen location
    /// because this could be a shared (re)borrow.  If you want to mutate, this
    /// is not the right function to call!
    fn check(&self, bor: Borrow) -> bool {
        match bor {
            Borrow::Frz(acc_t) =>
                // Must be frozen at least as long as the `acc_t` says.
                self.frozen_since.map_or(false, |loc_t| loc_t <= acc_t),
            Borrow::Mut(acc_m) =>
                // Raw pointers are fine with frozen locations. This is important because &Cell is raw!
                if self.frozen_since.is_some() {
                    acc_m.is_raw()
                } else {
                    self.borrows.last().map_or(false, |&loc_itm| loc_itm == BorStackItem::Mut(acc_m))
                }
        }
    }

    /// Check if `bor` could be activated by unfreezing and popping.
    /// `force_mut` indicates whether being frozen is potentially acceptable.
    /// Returns `Err` if the answer is "no"; otherwise the data says
    /// what needs to happen to activate this: `None` = nothing,
    /// `Some(n)` = unfreeze and make item `n` the top item of the stack.
    fn reactivatable(&self, bor: Borrow, force_mut: bool) -> Result<Option<usize>, String> {
        // Unless mutation is bound to happen, do NOT change anything if `bor` is already active.
        // In particular, if it is a `Mut(Raw)` and we are frozen, this should be a NOP.
        if !force_mut && self.check(bor) {
            return Ok(None);
        }

        let acc_m = match bor {
            Borrow::Frz(since) =>
                return Err(if force_mut {
                    format!("Using a shared borrow for mutation")
                } else {
                    format!(
                        "Location should be frozen since {} but {}",
                        since,
                        match self.frozen_since {
                            None => format!("it is not frozen at all"),
                            Some(since) => format!("it is only frozen since {}", since),
                        }
                    )
                }),
            Borrow::Mut(acc_m) => acc_m
        };
        // This is where we would unfreeze.
        for (idx, &itm) in self.borrows.iter().enumerate().rev() {
            match itm {
                BorStackItem::FnBarrier(_) =>
                    return Err(format!("Trying to reactivate a mutable borrow ({:?}) that lives behind a barrier", acc_m)),
                BorStackItem::Mut(loc_m) => {
                    if loc_m == acc_m { return Ok(Some(idx)); }
                }
            }
        }
        // Nothing to be found.
        Err(format!("Mutable borrow-to-reactivate ({:?}) does not exist on the stack", acc_m))
    }

    /// Reactive `bor` for this stack.  If `force_mut` is set, we want to aggressively
    /// unfreeze this location (because we are about to mutate, so a frozen `Raw` is not okay).
    fn reactivate(&mut self, bor: Borrow, force_mut: bool) -> EvalResult<'tcx> {
        let action = match self.reactivatable(bor, force_mut) {
            Ok(action) => action,
            Err(err) => return err!(MachineError(err)),
        };

        match action {
            None => {}, // nothing to do
            Some(top) => {
                self.frozen_since = None;
                self.borrows.truncate(top+1);
            }
        }

        Ok(())
    }

    /// Initiate `bor`; mostly this means freezing or pushing.
    fn initiate(&mut self, bor: Borrow) -> EvalResult<'tcx> {
        match bor {
            Borrow::Frz(t) => {
                match self.frozen_since {
                    None => {
                        trace!("initiate: Freezing");
                        self.frozen_since = Some(t);
                    }
                    Some(since) => {
                        trace!("initiate: Already frozen");
                        assert!(since <= t);
                    }
                }
            }
            Borrow::Mut(m) => {
                match self.frozen_since {
                    None => {
                        trace!("initiate: Pushing {:?}", bor);
                        self.borrows.push(BorStackItem::Mut(m))
                    }
                    Some(_) if m.is_raw() =>
                        // We only ever initiate right after activating the ref we come from.
                        // If the source ref is fine being frozen, then a raw ref we create
                        // from it is fine with this as well.
                        trace!("initiate: Initiating a raw on a frozen location, not doing a thing"),
                    Some(_) =>
                        return err!(MachineError(format!("Trying to mutate frozen location")))
                }
            }
        }
        Ok(())
    }
}

impl State {
    fn increment_clock(&self) -> Timestamp {
        let val = self.clock.get();
        self.clock.set(val+1);
        val
    }
}

/// Higher-level operations
impl<'tcx> Stacks {
    pub fn memory_accessed(
        &self,
        ptr: Pointer<Borrow>,
        size: Size,
        access: MemoryAccess,
    ) -> EvalResult<'tcx> {
        trace!("memory_accessed({:?}) with tag {:?}: {:?}, size {}", access, ptr.tag, ptr, size.bytes());
        let mut stacks = self.stacks.borrow_mut();
        for stack in stacks.iter_mut(ptr.offset, size) {
            // FIXME: Compare this with what the blog post says.
            stack.reactivate(ptr.tag, /*force_mut*/access == MemoryAccess::Write)?;
        }
        Ok(())
    }

    pub fn memory_deallocated(
        &mut self,
        ptr: Pointer<Borrow>,
    ) -> EvalResult<'tcx> {
        trace!("memory_deallocated with tag {:?}: {:?}", ptr.tag, ptr);
        let stacks = self.stacks.get_mut();
        for stack in stacks.iter_mut_all() {
            // This is like mutating.
            stack.reactivate(ptr.tag, /*force_mut*/true)?;
        }
        Ok(())
    }

    fn reborrow(
        &self,
        ptr: Pointer<Borrow>,
        size: Size,
        new_bor: Borrow,
        permit_redundant: bool,
    ) -> EvalResult<'tcx> {
        let mut stacks = self.stacks.borrow_mut();
        for stack in stacks.iter_mut(ptr.offset, size) {
            if permit_redundant && stack.check(new_bor) {
                // The new borrow is already active!  This can happen when creating multiple
                // shared references from the same mutable reference.  Do nothing.
                trace!("reborrow: New borrow {:?} is already active, not doing a thing", new_bor);
            } else {
                // If we are creating a uniq ref, we certainly want to unfreeze.
                // Even if we are doing so from a raw.
                // Notice that if this is a local, whenever we access it directly the
                // tag here will be the bottommost `Uniq` for that local.  That `Uniq`
                // never is accessible by the program, so it will not be used by any
                // other access.  IOW, whenever we directly use a local this will pop
                // everything else off the stack, invalidating all previous pointers
                // and, in particular, *all* raw pointers.  This subsumes the explicit
                // `reset` which the blog post [1] says to perform when accessing a local.
                //
                // [1] https://www.ralfj.de/blog/2018/08/07/stacked-borrows.html
                stack.reactivate(ptr.tag, /*force_mut*/new_bor.is_uniq())?;
                stack.initiate(new_bor)?;
            }
        }

        Ok(())
    }

    /// Pushes the first borrow to the stacks, must be a mutable one.
    pub fn first_borrow(
        &mut self,
        r#mut: Mut,
        size: Size
    ) {
        for stack in self.stacks.get_mut().iter_mut(Size::ZERO, size) {
            assert!(stack.borrows.len() == 1 && stack.frozen_since.is_none());
            assert_eq!(stack.borrows.pop().unwrap(), BorStackItem::Mut(Mut::Raw));
            stack.borrows.push(BorStackItem::Mut(r#mut));
        }
    }
}

pub trait EvalContextExt<'tcx> {
    fn tag_for_pointee(
        &self,
        pointee_ty: Ty<'tcx>,
        ref_kind: RefKind,
    ) -> Borrow;

    fn tag_reference(
        &self,
        ptr: Pointer<Borrow>,
        pointee_ty: Ty<'tcx>,
        size: Size,
        ref_kind: RefKind,
    ) -> EvalResult<'tcx, Borrow>;


    fn tag_dereference(
        &self,
        ptr: Pointer<Borrow>,
        pointee_ty: Ty<'tcx>,
        size: Size,
        ref_kind: RefKind,
    ) -> EvalResult<'tcx, Borrow>;

    fn tag_new_allocation(
        &mut self,
        id: AllocId,
        kind: MemoryKind<MiriMemoryKind>,
    ) -> Borrow;
}

impl<'a, 'mir, 'tcx> EvalContextExt<'tcx> for super::MiriEvalContext<'a, 'mir, 'tcx> {
    fn tag_for_pointee(
        &self,
        pointee_ty: Ty<'tcx>,
        ref_kind: RefKind,
    ) -> Borrow {
        let time = self.machine.stacked_borrows.increment_clock();
        match ref_kind {
            RefKind::Mut => Borrow::Mut(Mut::Uniq(time)),
            RefKind::Shr =>
                // FIXME This does not do enough checking when only part of the data has
                // interior mutability. When the type is `(i32, Cell<i32>)`, we want the
                // first field to be frozen but not the second.
                if self.type_is_freeze(pointee_ty) {
                    Borrow::Frz(time)
                } else {
                    // Shared reference with interior mutability.
                    Borrow::Mut(Mut::Raw)
                },
            RefKind::Raw => Borrow::Mut(Mut::Raw),
        }
    }

    /// Called for place-to-value conversion.
    fn tag_reference(
        &self,
        ptr: Pointer<Borrow>,
        pointee_ty: Ty<'tcx>,
        size: Size,
        ref_kind: RefKind,
    ) -> EvalResult<'tcx, Borrow> {
        let new_bor = self.tag_for_pointee(pointee_ty, ref_kind);
        trace!("tag_reference: Creating new reference ({:?}) for {:?} (pointee {}, size {}): {:?}",
            ref_kind, ptr, pointee_ty, size.bytes(), new_bor);

        // Make sure this reference is not dangling or so
        self.memory().check_bounds(ptr, size, false)?;

        // Update the stacks.  We cannot use `get_mut` becuse this might be immutable
        // memory.
        let alloc = self.memory().get(ptr.alloc_id).expect("We checked that the ptr is fine!");
        let permit_redundant = ref_kind == RefKind::Shr; // redundant shared refs are okay
        alloc.extra.reborrow(ptr, size, new_bor, permit_redundant)?;

        Ok(new_bor)
    }

    /// Called for value-to-place conversion.
    ///
    /// Note that this does NOT mean that all this memory will actually get accessed/referenced!
    /// We could be in the middle of `&(*var).1`.
    fn tag_dereference(
        &self,
        ptr: Pointer<Borrow>,
        pointee_ty: Ty<'tcx>,
        size: Size,
        ref_kind: RefKind,
    ) -> EvalResult<'tcx, Borrow> {
        // In principle we should not have to do anything here.  However, with transmutes involved,
        // it can happen that the tag of `ptr` does not actually match `ref_kind`, and we
        // should adjust for that.
        // Notably, the compiler can introduce such transmutes by optimizing away `&[mut]*`.
        // That can transmute a raw ptr to a (shared/mut) ref, and a mut ref to a shared one.
        match (ref_kind, ptr.tag) {
            (RefKind::Raw, _) => {
                // Don't use the tag, this is a raw access!  Even if there is a tag,
                // that means transmute happened and we ignore the tag.
                // Also don't do any further validation, this is raw after all.
                return Ok(Borrow::Mut(Mut::Raw));
            }
            (RefKind::Mut, Borrow::Mut(Mut::Uniq(_))) |
            (RefKind::Shr, Borrow::Frz(_)) |
            (RefKind::Shr, Borrow::Mut(Mut::Raw)) => {
                // Expected combinations.  Nothing to do.
                // FIXME: We probably shouldn't accept this if we got a raw shr without
                // interior mutability.
            }
            (RefKind::Mut, Borrow::Mut(Mut::Raw)) => {
                // Raw transmuted to mut ref.  Keep this as raw access.
                // We cannot reborrow here; there might be a raw in `&(*var).1` where
                // `var` is an `&mut`.  The other field of the struct might be already frozen,
                // also using `var`, and that would be okay.
            }
            (RefKind::Shr, Borrow::Mut(Mut::Uniq(_))) => {
                // A mut got transmuted to shr.  High time we freeze this location!
                // Make this a delayed reborrow.  Redundant reborows to shr are okay,
                // so we do not have to be worried about doing too much.
                // FIXME: Reconsider if we really want to mutate things while doing just a deref,
                // which, in particular, validation does.
                trace!("tag_dereference: Lazy freezing of {:?}", ptr);
                return self.tag_reference(ptr, pointee_ty, size, ref_kind);
            }
            (RefKind::Mut, Borrow::Frz(_)) => {
                // This is just invalid.
                // If we ever allow this, we have to consider what we do when a turn a
                // `Raw`-tagged `&mut` into a raw pointer pointing to a frozen location.
                // We probably do not want to allow that, but we have to allow
                // turning a `Raw`-tagged `&` into a raw ptr to a frozen location.
                return err!(MachineError(format!("Encountered mutable reference with frozen tag {:?}", ptr.tag)))
            }
        }
        // Even if we don't touch the tag, this operation is only okay if we *could*
        // activate it.  Also it must not be dangling.
        self.memory().check_bounds(ptr, size, false)?;
        let alloc = self.memory().get(ptr.alloc_id).expect("We checked that the ptr is fine!");
        let mut stacks = alloc.extra.stacks.borrow_mut();
        // We need `iter_mut` because `iter` would skip gaps!
        for stack in stacks.iter_mut(ptr.offset, size) {
            // We accept &mut to a frozen location here, that is just normal.  There might
            // be shared reborrows that we are about to invalidate with this access.
            // We cannot invalidate them aggressively here because the deref might also be
            // to just create more shared refs.
            if let Err(err) = stack.reactivatable(ptr.tag, /*force_mut*/false) {
                return err!(MachineError(format!("Encountered {:?} reference with non-reactivatable tag: {}", ref_kind, err)))
            }
        }
        // All is good.
        Ok(ptr.tag)
    }

    fn tag_new_allocation(
        &mut self,
        id: AllocId,
        kind: MemoryKind<MiriMemoryKind>,
    ) -> Borrow {
        let r#mut = match kind {
            MemoryKind::Stack => {
                // New unique borrow
                let time = self.machine.stacked_borrows.increment_clock();
                Mut::Uniq(time)
            }
            _ => {
                // Raw for everything else
                Mut::Raw
            }
        };
        // Make this the active borrow for this allocation
        let alloc = self.memory_mut().get_mut(id).expect("This is a new allocation, it must still exist");
        let size = Size::from_bytes(alloc.bytes.len() as u64);
        alloc.extra.first_borrow(r#mut, size);
        Borrow::Mut(r#mut)
    }
}
