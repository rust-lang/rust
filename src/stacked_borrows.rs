use std::cell::RefCell;

use rustc::ty::{Ty, layout::Size};
use rustc::mir;

use super::{
    MemoryAccess, RangeMap, EvalResult,
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
    fn is_raw(self) -> bool {
        match self {
            Mut::Raw => true,
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
    fn is_uniq(self) -> bool {
        match self {
            Borrow::Mut(Mut::Uniq(_)) => true,
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

/// Extra per-location state
#[derive(Clone, Debug)]
struct Stack {
    borrows: Vec<BorStackItem>, // used as a stack
    frozen_since: Option<Timestamp>,
}

impl Default for Stack {
    fn default() -> Self {
        Stack {
            borrows: Vec::new(),
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

    /// Reactive `bor` for this stack.  If `force_mut` is set, we want to aggressively
    /// unfreeze this location (because we are about to push a `Uniq`).
    fn reactivate(&mut self, bor: Borrow, force_mut: bool) -> EvalResult<'tcx> {
        // Unless mutation is bound to happen, do NOT change anything if `bor` is already active.
        // In particular, if it is a `Mut(Raw)` and we are frozen, this should be a NOP.
        if !force_mut && self.check(bor) {
            return Ok(());
        }

        let acc_m = match bor {
            Borrow::Frz(_) =>
                if force_mut {
                    return err!(MachineError(format!("Using a shared borrow for mutation")))
                } else {
                    return err!(MachineError(format!("Location should be frozen but it is not")))
                }
            Borrow::Mut(acc_m) => acc_m,
        };
        // We definitely have to unfreeze this, even if we use the topmost item.
        self.frozen_since = None;
        // Pop until we see the one we are looking for.
        while let Some(&itm) = self.borrows.last() {
            match itm {
                BorStackItem::FnBarrier(_) => {
                    return err!(MachineError(format!("Trying to reactivate a borrow that lives behind a barrier")));
                }
                BorStackItem::Mut(loc_m) => {
                    if loc_m == acc_m { return Ok(()); }
                    trace!("reactivate: Popping {:?}", itm);
                    self.borrows.pop();
                }
            }
        }
        // Nothing to be found.  Simulate a "virtual raw" element at the bottom of the stack.
        if acc_m.is_raw() {
            Ok(())
        } else {
            err!(MachineError(format!("Borrow-to-reactivate does not exist on the stack")))
        }
    }

    fn initiate(&mut self, bor: Borrow) -> EvalResult<'tcx> {
        match bor {
            Borrow::Frz(t) => {
                trace!("initiate: Freezing");
                match self.frozen_since {
                    None => self.frozen_since = Some(t),
                    Some(since) => assert!(since <= t),
                }
            }
            Borrow::Mut(m) => {
                trace!("initiate: Pushing {:?}", bor);
                match self.frozen_since {
                    None => self.borrows.push(BorStackItem::Mut(m)),
                    Some(_) =>
                        // FIXME: Do we want an exception for raw borrows?
                        return err!(MachineError(format!("Trying to mutate frozen location")))
                }
            }
        }
        Ok(())
    }
}

impl State {
    fn increment_clock(&mut self) -> Timestamp {
        self.clock += 1;
        self.clock
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
    ) -> EvalResult<'tcx> {
        let mut stacks = self.stacks.borrow_mut();
        for stack in stacks.iter_mut(ptr.offset, size) {
            if stack.check(new_bor) {
                // The new borrow is already active!  This can happen when creating multiple
                // shared references from the same mutable reference.  Do nothing.
            } else {
                // FIXME: The blog post says we should `reset` if this is a local.
                stack.reactivate(ptr.tag, /*force_mut*/new_bor.is_uniq())?;
                stack.initiate(new_bor)?;
            }
        }

        Ok(())
    }
}

/// Machine hooks
pub trait EvalContextExt<'tcx> {
    fn tag_reference(
        &mut self,
        ptr: Pointer<Borrow>,
        pointee_ty: Ty<'tcx>,
        size: Size,
        borrow_kind: Option<mir::BorrowKind>,
    ) -> EvalResult<'tcx, Borrow>;

    fn tag_dereference(
        &self,
        ptr: Pointer<Borrow>,
        ptr_ty: Ty<'tcx>,
    ) -> EvalResult<'tcx, Borrow>;

    fn ref_to_raw_cast(
        &mut self,
        ptr: Pointer<Borrow>,
        ptr_ty: Ty<'tcx>,
        size: Size,
    ) -> EvalResult<'tcx>;
}

impl<'a, 'mir, 'tcx> EvalContextExt<'tcx> for super::MiriEvalContext<'a, 'mir, 'tcx> {
    fn tag_reference(
        &mut self,
        ptr: Pointer<Borrow>,
        pointee_ty: Ty<'tcx>,
        size: Size,
        borrow_kind: Option<mir::BorrowKind>,
    ) -> EvalResult<'tcx, Borrow> {
        let time = self.machine.stacked_borrows.increment_clock();
        // FIXME This does not do enough checking when only part of the data has
        // interior mutability.
        let new_bor = match borrow_kind {
            Some(mir::BorrowKind::Mut { .. }) => Borrow::Mut(Mut::Uniq(time)),
            Some(_) =>
                if self.type_is_freeze(pointee_ty) {
                    Borrow::Frz(time)
                } else {
                    Borrow::Mut(Mut::Raw)
                },
            None => Borrow::Mut(Mut::Raw),
        };
        trace!("tag_reference: Creating new reference ({:?}) for {:?} (pointee {}, size {}): {:?}",
            borrow_kind, ptr, pointee_ty, size.bytes(), new_bor);

        // Make sure this reference is not dangling or so
        self.memory.check_bounds(ptr, size, false)?;

        // Update the stacks.  We cannot use `get_mut` becuse this might be immutable
        // memory.
        let alloc = self.memory.get(ptr.alloc_id).expect("We checked that the ptr is fine!");
        alloc.extra.reborrow(ptr, size, new_bor)?;

        Ok(new_bor)
    }

    fn tag_dereference(
        &self,
        ptr: Pointer<Borrow>,
        ptr_ty: Ty<'tcx>,
    ) -> EvalResult<'tcx, Borrow> {
        // If this is a raw ptr, forget about the tag.
        Ok(if ptr_ty.is_unsafe_ptr() {
            trace!("tag_dereference: Erasing tag for {:?} ({})", ptr, ptr_ty);
            Borrow::Mut(Mut::Raw)
        } else {
            // FIXME: Do we want to adjust the tag if it does not match the type?
            ptr.tag
        })
    }

    fn ref_to_raw_cast(
        &mut self,
        ptr: Pointer<Borrow>,
        _ptr_ty: Ty<'tcx>,
        size: Size,
    ) -> EvalResult<'tcx> {
        trace!("ref_to_raw_cast: Escaping {:?}", ptr);

        // Make sure this reference is not dangling or so
        self.memory.check_bounds(ptr, size, false)?;

        // Update the stacks.  We cannot use `get_mut` becuse this might be immutable
        // memory.
        let alloc = self.memory.get(ptr.alloc_id).expect("We checked that the ptr is fine!");
        alloc.extra.reborrow(ptr, size, Borrow::Mut(Mut::Raw))?;

        Ok(())
    }
}
