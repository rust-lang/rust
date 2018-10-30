use std::cell::RefCell;

use rustc::ty::{self, Ty, layout::Size};
use rustc::hir;

use super::{
    MemoryKind, MiriMemoryKind, RangeMap, EvalResult, AllocId,
    Pointer, PlaceTy,
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

impl Default for Borrow {
    fn default() -> Self {
        Borrow::Mut(Mut::Raw)
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

impl BorStackItem {
    #[inline(always)]
    pub fn is_fn_barrier(self) -> bool {
        match self {
            BorStackItem::FnBarrier(_) => true,
            _ => false,
        }
    }
}

/// What kind of usage of the pointer are we talking about?
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum UsageKind {
    /// Write, or create &mut
    Write,
    /// Read, or create &
    Read,
    /// Create *
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

/// Extra per-location state
#[derive(Clone, Debug)]
pub struct Stack {
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

impl Stack {
    #[inline(always)]
    fn is_frozen(&self) -> bool {
        self.frozen_since.is_some()
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
    /// Returns `Err` if the answer is "no"; otherwise the data says
    /// what needs to happen to activate this: `None` = nothing,
    /// `Some(n)` = unfreeze and make item `n` the top item of the stack.
    fn reactivatable(&self, bor: Borrow, usage: UsageKind) -> Result<Option<usize>, String> {
        let mut_borrow = match bor {
            Borrow::Frz(since) =>
                // The only way to reactivate a `Frz` is if this is already frozen.
                return match self.frozen_since {
                    _ if usage == UsageKind::Write =>
                        Err(format!("Using a shared borrow for mutation")),
                    None =>
                        Err(format!("Location should be frozen but it is not")),
                    Some(loc) if loc <= since =>
                        Ok(None),
                    Some(loc) =>
                        Err(format!("Location should be frozen since {} but it is only frozen \
                                     since {}", since, loc)),
                },
            Borrow::Mut(Mut::Raw) if self.is_frozen() && usage != UsageKind::Write =>
                // Non-mutating access with a raw from a frozen location is a special case: The
                // shared refs do not mind raw reads, and the raw itself does not assume any
                // exclusivity. So we do not even require there to be a raw on the stack,
                // the raw is instead "matched" by the fact that this location is frozen.
                // This does not break the assumption that an `&mut` we own is
                // exclusive for reads, because there we have the invariant that
                // the location is *not* frozen.
                return Ok(None),
            Borrow::Mut(mut_borrow) => mut_borrow
        };
        // See if we can get there via popping.
        for (idx, &itm) in self.borrows.iter().enumerate().rev() {
            match itm {
                BorStackItem::FnBarrier(_) =>
                    return Err(format!("Trying to reactivate a mutable borrow ({:?}) that lives \
                                        behind a barrier", mut_borrow)),
                BorStackItem::Mut(loc) => {
                    if loc == mut_borrow {
                        // We found it!  This is good to know.
                        // Yet, maybe we do not really want to pop?
                        if usage == UsageKind::Read && self.is_frozen() {
                            // Whoever had exclusive access to this location allowed it
                            // to become frozen.  That can only happen if they reborrowed
                            // to a shared ref, at which point they gave up on exclusive access.
                            // Hence we allow more reads, entirely ignoring everything above
                            // on the stack (but still making sure it is on the stack).
                            // This does not break the assumption that an `&mut` we own is
                            // exclusive for reads, because there we have the invariant that
                            // the location is *not* frozen.
                            return Ok(None);
                        } else {
                            return Ok(Some(idx));
                        }
                    }
                }
            }
        }
        // Nothing to be found.
        Err(format!("Mutable borrow-to-reactivate ({:?}) does not exist on the stack", mut_borrow))
    }

    /// Reactive `bor` for this stack.  `usage` indicates whether this is being
    /// used to read/write (or, equivalently, to borrow as &/&mut), or to borrow as raw.
    fn reactivate(&mut self, bor: Borrow, usage: UsageKind) -> EvalResult<'tcx> {
        let action = match self.reactivatable(bor, usage) {
            Ok(action) => action,
            Err(err) => return err!(MachineError(err)),
        };
        // Execute what `reactivatable` told us to do.
        match action {
            None => {}, // nothing to do
            Some(top) => {
                if self.frozen_since.is_some() {
                    trace!("reactivate: Unfreezing");
                }
                self.frozen_since = None;
                for itm in self.borrows.drain(top+1..).rev() {
                    trace!("reactivate: Popping {:?}", itm);
                }
            }
        }

        Ok(())
    }

    /// Initiate `bor`; mostly this means freezing or pushing.
    /// This operation cannot fail; it is up to the caller to ensure that the precondition
    /// is met: We cannot push onto frozen stacks.
    fn initiate(&mut self, bor: Borrow) {
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
                        bug!("Trying to mutate frozen location")
                }
            }
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
    /// The single most operation: Make sure that using `ptr` as `usage` is okay,
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

    /// Pushes the first borrow to the stacks, must be a mutable one.
    pub fn first_borrow(
        &mut self,
        mut_borrow: Mut,
        size: Size
    ) {
        for stack in self.stacks.get_mut().iter_mut(Size::ZERO, size) {
            assert!(stack.borrows.len() == 1 && stack.frozen_since.is_none());
            assert_eq!(stack.borrows.pop().unwrap(), BorStackItem::Mut(Mut::Raw));
            stack.borrows.push(BorStackItem::Mut(mut_borrow));
        }
    }
}

pub trait EvalContextExt<'tcx> {
    fn tag_reference(
        &mut self,
        ptr: Pointer<Borrow>,
        pointee_ty: Ty<'tcx>,
        size: Size,
        usage: UsageKind,
    ) -> EvalResult<'tcx, Borrow>;


    fn tag_dereference(
        &self,
        ptr: Pointer<Borrow>,
        pointee_ty: Ty<'tcx>,
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

impl<'a, 'mir, 'tcx> EvalContextExt<'tcx> for super::MiriEvalContext<'a, 'mir, 'tcx> {
    /// Called for place-to-value conversion.
    fn tag_reference(
        &mut self,
        ptr: Pointer<Borrow>,
        pointee_ty: Ty<'tcx>,
        size: Size,
        usage: UsageKind,
    ) -> EvalResult<'tcx, Borrow> {
        let time = self.machine.stacked_borrows.increment_clock();
        let new_bor = match usage {
            UsageKind::Write => Borrow::Mut(Mut::Uniq(time)),
            UsageKind::Read =>
                // FIXME This does not do enough checking when only part of the data has
                // interior mutability. When the type is `(i32, Cell<i32>)`, we want the
                // first field to be frozen but not the second.
                if self.type_is_freeze(pointee_ty) {
                    Borrow::Frz(time)
                } else {
                    // Shared reference with interior mutability.
                    Borrow::Mut(Mut::Raw)
                },
            UsageKind::Raw => Borrow::Mut(Mut::Raw),
        };
        trace!("tag_reference: Creating new reference ({:?}) for {:?} (pointee {}, size {}): {:?}",
            usage, ptr, pointee_ty, size.bytes(), new_bor);

        // Make sure this reference is not dangling or so
        self.memory().check_bounds(ptr, size, false)?;

        // Update the stacks.  We cannot use `get_mut` becuse this might be immutable
        // memory.
        let alloc = self.memory().get(ptr.alloc_id).expect("We checked that the ptr is fine!");
        alloc.extra.use_and_maybe_re_borrow(ptr, size, usage, Some(new_bor))?;

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
        usage: UsageKind,
    ) -> EvalResult<'tcx, Borrow> {
        trace!("tag_reference: Accessing reference ({:?}) for {:?} (pointee {}, size {})",
            usage, ptr, pointee_ty, size.bytes());
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
                return Ok(Borrow::Mut(Mut::Raw));
            }
            (UsageKind::Write, Borrow::Mut(Mut::Uniq(_))) |
            (UsageKind::Read, Borrow::Frz(_)) |
            (UsageKind::Read, Borrow::Mut(Mut::Raw)) => {
                // Expected combinations.  Nothing to do.
                // FIXME: We probably shouldn't accept this if we got a raw shr without
                // interior mutability.
            }
            (UsageKind::Write, Borrow::Mut(Mut::Raw)) => {
                // Raw transmuted to mut ref.  Keep this as raw access.
                // We cannot reborrow here; there might be a raw in `&(*var).1` where
                // `var` is an `&mut`.  The other field of the struct might be already frozen,
                // also using `var`, and that would be okay.
            }
            (UsageKind::Read, Borrow::Mut(Mut::Uniq(_))) => {
                // A mut got transmuted to shr.  The mut borrow must be reactivatable.
            }
            (UsageKind::Write, Borrow::Frz(_)) => {
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
            // Conservatively assume that we will only read.
            if let Err(err) = stack.reactivatable(ptr.tag, UsageKind::Read) {
                return err!(MachineError(format!(
                    "Encountered {} reference with non-reactivatable tag: {}",
                    if usage == UsageKind::Write { "mutable" } else { "shared" },
                    err
                )))
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
        let mut_borrow = match kind {
            MemoryKind::Stack => {
                // New unique borrow. This `Uniq` is not accessible by the program,
                // so it will only ever be used when using the local directly (i.e.,
                // not through a pointer).  IOW, whenever we directly use a local this will pop
                // everything else off the stack, invalidating all previous pointers
                // and, in particular, *all* raw pointers.  This subsumes the explicit
                // `reset` which the blog post [1] says to perform when accessing a local.
                //
                // [1] https://www.ralfj.de/blog/2018/08/07/stacked-borrows.html
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
        alloc.extra.first_borrow(mut_borrow, size);
        Borrow::Mut(mut_borrow)
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
        let val = self.read_value(self.place_to_op(place)?)?;
        let dest = self.ref_to_mplace(val)?;
        // Now put a new ref into the old place, which will call `tag_reference`.
        // FIXME: Honor `fn_entry`!
        let val = self.create_ref(dest, Some(mutbl))?;
        self.write_value(val, place)?;
        Ok(())
    }
}
