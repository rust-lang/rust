use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;
use std::fmt;
use std::num::NonZeroU64;

use rustc::ty::{self, layout::Size};
use rustc::hir::{MutMutable, MutImmutable};
use rustc::mir::RetagKind;

use crate::{
    EvalResult, InterpError, MiriEvalContext, HelpersEvalContextExt, Evaluator, MutValueVisitor,
    MemoryKind, MiriMemoryKind, RangeMap, Allocation, AllocationExtra,
    Pointer, Immediate, ImmTy, PlaceTy, MPlaceTy,
};

pub type PtrId = NonZeroU64;
pub type CallId = NonZeroU64;

/// Tracking pointer provenance
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum Tag {
    Tagged(PtrId),
    Untagged,
}

impl fmt::Display for Tag {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Tag::Tagged(id) => write!(f, "{}", id),
            Tag::Untagged => write!(f, "<untagged>"),
        }
    }
}

/// Indicates which permission is granted (by this item to some pointers)
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum Permission {
    /// Grants unique mutable access.
    Unique,
    /// Grants shared mutable access.
    SharedReadWrite,
    /// Greants shared read-only access.
    SharedReadOnly,
}

/// An item in the per-location borrow stack.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct Item {
    /// The permission this item grants.
    perm: Permission,
    /// The pointers the permission is granted to.
    tag: Tag,
    /// An optional protector, ensuring the item cannot get popped until `CallId` is over.
    protector: Option<CallId>,
}

impl fmt::Display for Item {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[{:?} for {}", self.perm, self.tag)?;
        if let Some(call) = self.protector {
            write!(f, " (call {})", call)?;
        }
        write!(f, "]")?;
        Ok(())
    }
}

/// Extra per-location state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stack {
    /// Used *mostly* as a stack; never empty.
    /// We sometimes push into the middle but never remove from the middle.
    /// The same tag may occur multiple times, e.g. from a two-phase borrow.
    /// Invariants:
    /// * Above a `SharedReadOnly` there can only be more `SharedReadOnly`.
    borrows: Vec<Item>,
}


/// Extra per-allocation state.
#[derive(Clone, Debug)]
pub struct Stacks {
    // Even reading memory can have effects on the stack, so we need a `RefCell` here.
    stacks: RefCell<RangeMap<Stack>>,
    // Pointer to global state
    global: MemoryState,
}

/// Extra global state, available to the memory access hooks.
#[derive(Debug)]
pub struct GlobalState {
    next_ptr_id: PtrId,
    next_call_id: CallId,
    active_calls: HashSet<CallId>,
}
pub type MemoryState = Rc<RefCell<GlobalState>>;

/// Indicates which kind of access is being performed.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum AccessKind {
    Read,
    Write,
}

impl fmt::Display for AccessKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            AccessKind::Read => write!(f, "read"),
            AccessKind::Write => write!(f, "write"),
        }
    }
}

/// Indicates which kind of reference is being created.
/// Used by high-level `reborrow` to compute which permissions to grant to the
/// new pointer.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum RefKind {
    /// `&mut` and `Box`.
    Unique,
    /// `&` with or without interior mutability.
    Shared,
    /// `*mut`/`*const` (raw pointers).
    Raw { mutable: bool },
}

impl fmt::Display for RefKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            RefKind::Unique => write!(f, "unique"),
            RefKind::Shared => write!(f, "shared"),
            RefKind::Raw { mutable: true } => write!(f, "raw (mutable)"),
            RefKind::Raw { mutable: false } => write!(f, "raw (constant)"),
        }
    }
}

/// Utilities for initialization and ID generation
impl Default for GlobalState {
    fn default() -> Self {
        GlobalState {
            next_ptr_id: NonZeroU64::new(1).unwrap(),
            next_call_id: NonZeroU64::new(1).unwrap(),
            active_calls: HashSet::default(),
        }
    }
}

impl GlobalState {
    pub fn new_ptr(&mut self) -> PtrId {
        let id = self.next_ptr_id;
        self.next_ptr_id = NonZeroU64::new(id.get() + 1).unwrap();
        id
    }

    pub fn new_call(&mut self) -> CallId {
        let id = self.next_call_id;
        trace!("new_call: Assigning ID {}", id);
        self.active_calls.insert(id);
        self.next_call_id = NonZeroU64::new(id.get() + 1).unwrap();
        id
    }

    pub fn end_call(&mut self, id: CallId) {
        assert!(self.active_calls.remove(&id));
    }

    fn is_active(&self, id: CallId) -> bool {
        self.active_calls.contains(&id)
    }
}

// # Stacked Borrows Core Begin

/// We need to make at least the following things true:
///
/// U1: After creating a `Uniq`, it is at the top.
/// U2: If the top is `Uniq`, accesses must be through that `Uniq` or remove it it.
/// U3: If an access happens with a `Uniq`, it requires the `Uniq` to be in the stack.
///
/// F1: After creating a `&`, the parts outside `UnsafeCell` have our `SharedReadOnly` on top.
/// F2: If a write access happens, it pops the `SharedReadOnly`.  This has three pieces:
///     F2a: If a write happens granted by an item below our `SharedReadOnly`, the `SharedReadOnly`
///          gets popped.
///     F2b: No `SharedReadWrite` or `Unique` will ever be added on top of our `SharedReadOnly`.
/// F3: If an access happens with an `&` outside `UnsafeCell`,
///     it requires the `SharedReadOnly` to still be in the stack.

impl Default for Tag {
    #[inline(always)]
    fn default() -> Tag {
        Tag::Untagged
    }
}

/// Core relations on `Permission` define which accesses are allowed:
/// On every access, we try to find a *granting* item, and then we remove all
/// *incompatible* items above it.
impl Permission {
    /// This defines for a given permission, whether it permits the given kind of access.
    fn grants(self, access: AccessKind) -> bool {
        match (self, access) {
            // Unique and SharedReadWrite allow any kind of access.
            (Permission::Unique, _) |
            (Permission::SharedReadWrite, _) =>
                true,
            // SharedReadOnly only permits read access.
            (Permission::SharedReadOnly, AccessKind::Read) =>
                true,
            (Permission::SharedReadOnly, AccessKind::Write) =>
                false,
        }
    }

    /// This defines for a given permission, which other permissions it can tolerate "above" itself
    /// for which kinds of accesses.
    /// If true, then `other` is allowed to remain on top of `self` when `access` happens.
    fn compatible_with(self, access: AccessKind, other: Permission) -> bool {
        use self::Permission::*;

        match (self, access, other) {
            // Some cases are impossible.
            (SharedReadOnly, _, SharedReadWrite) |
            (SharedReadOnly, _, Unique) =>
                bug!("There can never be a SharedReadWrite or a Unique on top of a SharedReadOnly"),
            // When `other` is `SharedReadOnly`, that is NEVER compatible with
            // write accesses.
            // This makes sure read-only pointers become invalid on write accesses (ensures F2a).
            (_, AccessKind::Write, SharedReadOnly) =>
                false,
            // When `other` is `Unique`, that is compatible with nothing.
            // This makes sure unique pointers become invalid on incompatible accesses (ensures U2).
            (_, _, Unique) =>
                false,
            // When we are unique and this is a write/dealloc, we tolerate nothing.
            // This makes sure we re-assert uniqueness ("being on top") on write accesses.
            // (This is particularily important such that when a new mutable ref gets created, it gets
            // pushed onto the right item -- this behaves like a write and we assert uniqueness of the
            // pointer from which this comes, *if* it was a unique pointer.)
            (Unique, AccessKind::Write, _) =>
                false,
            // `SharedReadWrite` items can tolerate any other akin items for any kind of access.
            (SharedReadWrite, _, SharedReadWrite) =>
                true,
            // Any item can tolerate read accesses for shared items.
            // This includes unique items!  Reads from unique pointers do not invalidate
            // other pointers.
            (_, AccessKind::Read, SharedReadWrite) |
            (_, AccessKind::Read, SharedReadOnly) =>
                true,
            // That's it.
        }
    }
}

/// Core per-location operations: access, dealloc, reborrow.
impl<'tcx> Stack {
    /// Find the item granting the given kind of access to the given tag, and where
    /// *the first incompatible item above it* is on the stack.
    fn find_granting(&self, access: AccessKind, tag: Tag) -> Option<(Permission, usize)> {
        let (perm, idx) = self.borrows.iter()
            .enumerate() // we also need to know *where* in the stack
            .rev() // search top-to-bottom
            // Return permission of first item that grants access.
            // We require a permission with the right tag, ensuring U3 and F3.
            .find_map(|(idx, item)|
                if item.perm.grants(access) && tag == item.tag {
                    Some((item.perm, idx))
                } else {
                    None
                }
            )?;

        let mut first_incompatible_idx = idx+1;
        while let Some(item) = self.borrows.get(first_incompatible_idx) {
            if perm.compatible_with(access, item.perm) {
                // Keep this, check next.
                first_incompatible_idx += 1;
            } else {
                // Found it!
                break;
            }
        }
        return Some((perm, first_incompatible_idx));
    }

    /// Test if a memory `access` using pointer tagged `tag` is granted.
    /// If yes, return the index of the item that granted it.
    fn access(
        &mut self,
        access: AccessKind,
        tag: Tag,
        global: &GlobalState,
    ) -> EvalResult<'tcx> {
        // Two main steps: Find granting item, remove all incompatible items above.

        // Step 1: Find granting item.
        let (granting_perm, first_incompatible_idx) = self.find_granting(access, tag)
            .ok_or_else(|| InterpError::MachineError(format!(
                "no item granting {} access to tag {} found in borrow stack",
                access, tag,
            )))?;

        // Step 2: Remove everything incompatible above them.  Make sure we do not remove protected
        // items.
        // For writes, this is a simple stack. For reads, however, it is not:
        // in `let raw = &mut *x as *mut _; let _val = *x;`, the second statement would pop the `Unique`
        // from the reborrow of the first statement, and subsequently also pop the `SharedReadWrite` for `raw`.
        // This pattern occurs a lot in the standard library: create a raw pointer, then also create a shared
        // reference and use that.
        {
            // Implemented with indices because there does not seem to be a nice iterator and range-based
            // API for this.
            let mut cur = first_incompatible_idx;
            while let Some(item) = self.borrows.get(cur) {
                // If this is a read, we double-check if we really want to kill this.
                if access == AccessKind::Read && granting_perm.compatible_with(access, item.perm) {
                    // Keep this, check next.
                    cur += 1;
                } else {
                    // Aha! This is a bad one, remove it, and make sure it is not protected.
                    let item = self.borrows.remove(cur);
                    if let Some(call) = item.protector {
                        if global.is_active(call) {
                            return err!(MachineError(format!(
                                "not granting {} access to tag {} because incompatible item {} is protected",
                                access, tag, item
                            )));
                        }
                    }
                    trace!("access: removing item {}", item);
                }
            }
        }

        // Done.
        Ok(())
    }

    /// Deallocate a location: Like a write access, but also there must be no
    /// active protectors at all.
    fn dealloc(
        &mut self,
        tag: Tag,
        global: &GlobalState,
    ) -> EvalResult<'tcx> {
        // Step 1: Find granting item.
        self.find_granting(AccessKind::Write, tag)
            .ok_or_else(|| InterpError::MachineError(format!(
                "no item granting write access for deallocation to tag {} found in borrow stack",
                tag,
            )))?;

        // We must make sure there are no protected items remaining on the stack.
        // Also clear the stack, no more accesses are possible.
        for item in self.borrows.drain(..) {
            if let Some(call) = item.protector {
                if global.is_active(call) {
                    return err!(MachineError(format!(
                        "deallocating with active protector ({})", call
                    )))
                }
            }
        }

        Ok(())
    }

    /// `reborrow` helper function: test that the stack invariants are still maintained.
    fn test_invariants(&self) {
        let mut saw_shared_read_only = false;
        for item in self.borrows.iter() {
            match item.perm {
                Permission::SharedReadOnly => {
                    saw_shared_read_only = true;
                }
                // Otherwise, if we saw one before, that's a bug.
                perm if saw_shared_read_only => {
                    bug!("Found {:?} on top of a SharedReadOnly!", perm);
                }
                _ => {}
            }
        }
    }

    /// Derived a new pointer from one with the given tag.
    /// `weak` controls whether this operation is weak or strong: weak granting does not act as
    /// an access, and they add the new item directly on top of the one it is derived
    /// from instead of all the way at the top of the stack.
    fn grant(
        &mut self,
        derived_from: Tag,
        weak: bool,
        new: Item,
        global: &GlobalState,
    ) -> EvalResult<'tcx> {
        // Figure out which access `perm` corresponds to.
        let access = if new.perm.grants(AccessKind::Write) {
            AccessKind::Write
        } else {
            AccessKind::Read
        };
        // Now we figure out which item grants our parent (`derived_from`) this kind of access.
        // We use that to determine where to put the new item.
        let (_, first_incompatible_idx) = self.find_granting(access, derived_from)
            .ok_or_else(|| InterpError::MachineError(format!(
                "no item to reborrow for {:?} from tag {} found in borrow stack", new.perm, derived_from,
            )))?;

        // Compute where to put the new item.
        // Either way, we ensure that we insert the new item in a way that between
        // `derived_from` and the new one, there are only items *compatible with* `derived_from`.
        let new_idx = if weak {
            // A weak SharedReadOnly reborrow might be added below other items, violating the
            // invariant that only SharedReadOnly can sit on top of SharedReadOnly.
            assert!(new.perm != Permission::SharedReadOnly, "Weak SharedReadOnly reborrows don't work");
            // A very liberal reborrow because the new pointer does not expect any kind of aliasing guarantee.
            // Just insert new permission as child of old permission, and maintain everything else.
            // This inserts "as far down as possible", which is good because it makes this pointer as
            // long-lived as possible *and* we want all the items that are incompatible with this
            // to actually get removed from the stack.  If we pushed a `SharedReadWrite` on top of
            // a `SharedReadOnly`, we'd violate the invariant that `SaredReadOnly` are at the top
            // and we'd allow write access without invalidating frozen shared references!
            // This ensures F2b for `SharedReadWrite` by adding the new item below any
            // potentially existing `SharedReadOnly`.
            first_incompatible_idx
        } else {
            // A "safe" reborrow for a pointer that actually expects some aliasing guarantees.
            // Here, creating a reference actually counts as an access, and pops incompatible
            // stuff off the stack.
            // This ensures F2b for `Unique`, by removing offending `SharedReadOnly`.
            self.access(access, derived_from, global)?;
            if access == AccessKind::Write {
                // For write accesses, the position is the same as what it would have been weakly!
                assert_eq!(first_incompatible_idx, self.borrows.len());
            }

            // We insert "as far up as possible": We know only compatible items are remaining
            // on top of `derived_from`, and we want the new item at the top so that we
            // get the strongest possible guarantees.
            // This ensures U1 and F1.
            self.borrows.len()
        };

        // Put the new item there. As an optimization, deduplicate if it is equal to one of its new neighbors.
        if self.borrows[new_idx-1] == new || self.borrows.get(new_idx) == Some(&new) {
            // Optimization applies, done.
            trace!("reborrow: avoiding adding redundant item {}", new);
        } else {
            trace!("reborrow: adding item {}", new);
            self.borrows.insert(new_idx, new);
        }

        // Make sure that after all this, the stack's invariant is still maintained.
        if cfg!(debug_assertions) {
            self.test_invariants();
        }

        Ok(())
    }
}
// # Stacked Borrows Core End

/// Map per-stack operations to higher-level per-location-range operations.
impl<'tcx> Stacks {
    /// Creates new stack with initial tag.
    pub(crate) fn new(
        size: Size,
        tag: Tag,
        extra: MemoryState,
    ) -> Self {
        let item = Item { perm: Permission::Unique, tag, protector: None };
        let stack = Stack {
            borrows: vec![item],
        };
        Stacks {
            stacks: RefCell::new(RangeMap::new(size, stack)),
            global: extra,
        }
    }

    /// Call `f` on every stack in the range.
    fn for_each(
        &self,
        ptr: Pointer<Tag>,
        size: Size,
        f: impl Fn(&mut Stack, &GlobalState) -> EvalResult<'tcx>,
    ) -> EvalResult<'tcx> {
        let global = self.global.borrow();
        let mut stacks = self.stacks.borrow_mut();
        for stack in stacks.iter_mut(ptr.offset, size) {
            f(stack, &*global)?;
        }
        Ok(())
    }
}

/// Glue code to connect with Miri Machine Hooks
impl Stacks {
    pub fn new_allocation(
        size: Size,
        extra: &MemoryState,
        kind: MemoryKind<MiriMemoryKind>,
    ) -> (Self, Tag) {
        let tag = match kind {
            MemoryKind::Stack => {
                // New unique borrow. This `Uniq` is not accessible by the program,
                // so it will only ever be used when using the local directly (i.e.,
                // not through a pointer). That is, whenever we directly use a local, this will pop
                // everything else off the stack, invalidating all previous pointers,
                // and in particular, *all* raw pointers. This subsumes the explicit
                // `reset` which the blog post [1] says to perform when accessing a local.
                //
                // [1]: <https://www.ralfj.de/blog/2018/08/07/stacked-borrows.html>
                Tag::Tagged(extra.borrow_mut().new_ptr())
            }
            _ => {
                Tag::Untagged
            }
        };
        let stack = Stacks::new(size, tag, Rc::clone(extra));
        (stack, tag)
    }
}

impl AllocationExtra<Tag> for Stacks {
    #[inline(always)]
    fn memory_read<'tcx>(
        alloc: &Allocation<Tag, Stacks>,
        ptr: Pointer<Tag>,
        size: Size,
    ) -> EvalResult<'tcx> {
        trace!("read access with tag {}: {:?}, size {}", ptr.tag, ptr, size.bytes());
        alloc.extra.for_each(ptr, size, |stack, global| {
            stack.access(AccessKind::Read, ptr.tag, global)?;
            Ok(())
        })
    }

    #[inline(always)]
    fn memory_written<'tcx>(
        alloc: &mut Allocation<Tag, Stacks>,
        ptr: Pointer<Tag>,
        size: Size,
    ) -> EvalResult<'tcx> {
        trace!("write access with tag {}: {:?}, size {}", ptr.tag, ptr, size.bytes());
        alloc.extra.for_each(ptr, size, |stack, global| {
            stack.access(AccessKind::Write, ptr.tag, global)?;
            Ok(())
        })
    }

    #[inline(always)]
    fn memory_deallocated<'tcx>(
        alloc: &mut Allocation<Tag, Stacks>,
        ptr: Pointer<Tag>,
        size: Size,
    ) -> EvalResult<'tcx> {
        trace!("deallocation with tag {}: {:?}, size {}", ptr.tag, ptr, size.bytes());
        alloc.extra.for_each(ptr, size, |stack, global| {
            stack.dealloc(ptr.tag, global)
        })
    }
}

/// Retagging/reborrowing.  There is some policy in here, such as which permissions
/// to grant for which references, when to add protectors, and how to realize two-phase
/// borrows in terms of the primitives above.
impl<'a, 'mir, 'tcx> EvalContextPrivExt<'a, 'mir, 'tcx> for crate::MiriEvalContext<'a, 'mir, 'tcx> {}
trait EvalContextPrivExt<'a, 'mir, 'tcx: 'a+'mir>: crate::MiriEvalContextExt<'a, 'mir, 'tcx> {
    fn reborrow(
        &mut self,
        place: MPlaceTy<'tcx, Tag>,
        size: Size,
        kind: RefKind,
        new_tag: Tag,
        force_weak: bool,
        protect: bool,
    ) -> EvalResult<'tcx> {
        let this = self.eval_context_mut();
        let protector = if protect { Some(this.frame().extra) } else { None };
        let ptr = place.ptr.to_ptr()?;
        trace!("reborrow: {:?} reference {} derived from {} (pointee {}): {:?}, size {}",
            kind, new_tag, ptr.tag, place.layout.ty, ptr, size.bytes());

        // Get the allocation. It might not be mutable, so we cannot use `get_mut`.
        let alloc = this.memory().get(ptr.alloc_id)?;
        alloc.check_bounds(this, ptr, size)?;
        // Update the stacks.
        // Make sure that raw pointers and mutable shared references are reborrowed "weak":
        // There could be existing unique pointers reborrowed from them that should remain valid!
        let perm = match kind {
            RefKind::Unique => Permission::Unique,
            RefKind::Raw { mutable: true } => Permission::SharedReadWrite,
            RefKind::Shared | RefKind::Raw { mutable: false } => {
                // Shared references and *const are a whole different kind of game, the
                // permission is not uniform across the entire range!
                // We need a frozen-sensitive reborrow.
                return this.visit_freeze_sensitive(place, size, |cur_ptr, size, frozen| {
                    // We are only ever `SharedReadOnly` inside the frozen bits.
                    let perm = if frozen { Permission::SharedReadOnly } else { Permission::SharedReadWrite };
                    let weak = perm == Permission::SharedReadWrite;
                    let item = Item { perm, tag: new_tag, protector };
                    alloc.extra.for_each(cur_ptr, size, |stack, global| {
                        stack.grant(cur_ptr.tag, force_weak || weak, item, global)
                    })
                });
            }
        };
        debug_assert_ne!(perm, Permission::SharedReadOnly, "SharedReadOnly must be used frozen-sensitive");
        let weak = perm == Permission::SharedReadWrite;
        let item = Item { perm, tag: new_tag, protector };
        alloc.extra.for_each(ptr, size, |stack, global| {
            stack.grant(ptr.tag, force_weak || weak, item, global)
        })
    }

    /// Retags an indidual pointer, returning the retagged version.
    /// `mutbl` can be `None` to make this a raw pointer.
    fn retag_reference(
        &mut self,
        val: ImmTy<'tcx, Tag>,
        kind: RefKind,
        protect: bool,
        two_phase: bool,
    ) -> EvalResult<'tcx, Immediate<Tag>> {
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
        let new_tag = match kind {
            RefKind::Raw { .. } => Tag::Untagged,
            _ => Tag::Tagged(this.memory().extra.borrow_mut().new_ptr()),
        };

        // Reborrow.
        // TODO: With `two_phase == true`, this performs a weak reborrow for a `Unique`. That
        // can lead to some possibly surprising effects, if the parent permission is
        // `SharedReadWrite` then we now have a `Unique` in the middle of them, which "splits"
        // them in terms of what remains valid when the `Unique` gets used.  Is that really
        // what we want?
        this.reborrow(place, size, kind, new_tag, /*force_weak:*/ two_phase, protect)?;
        let new_place = place.replace_tag(new_tag);
        // Handle two-phase borrows.
        if two_phase {
            assert!(kind == RefKind::Unique, "two-phase shared borrows make no sense");
            // Grant read access *to the parent pointer* with the old tag *derived from the new tag* (`new_place`). 
            // This means the old pointer has multiple items in the stack now, which otherwise cannot happen
            // for unique references -- but in this case it precisely expresses the semantics we want.
            let old_tag = place.ptr.to_ptr().unwrap().tag;
            this.reborrow(new_place, size, RefKind::Shared, old_tag, /*force_weak:*/ false, /*protect:*/ false)?;
        }

        // Return new pointer.
        Ok(new_place.to_ref())
    }
}

impl<'a, 'mir, 'tcx> EvalContextExt<'a, 'mir, 'tcx> for crate::MiriEvalContext<'a, 'mir, 'tcx> {}
pub trait EvalContextExt<'a, 'mir, 'tcx: 'a+'mir>: crate::MiriEvalContextExt<'a, 'mir, 'tcx> {
    fn retag(
        &mut self,
        kind: RetagKind,
        place: PlaceTy<'tcx, Tag>
    ) -> EvalResult<'tcx> {
        let this = self.eval_context_mut();
        // Determine mutability and whether to add a protector.
        // Cannot use `builtin_deref` because that reports *immutable* for `Box`,
        // making it useless.
        fn qualify(ty: ty::Ty<'_>, kind: RetagKind) -> Option<(RefKind, bool)> {
            match ty.sty {
                // References are simple.
                ty::Ref(_, _, MutMutable) =>
                    Some((RefKind::Unique, kind == RetagKind::FnEntry)),
                ty::Ref(_, _, MutImmutable) =>
                    Some((RefKind::Shared, kind == RetagKind::FnEntry)),
                // Raw pointers need to be enabled.
                ty::RawPtr(tym) if kind == RetagKind::Raw =>
                    Some((RefKind::Raw { mutable: tym.mutbl == MutMutable }, false)),
                // Boxes do not get a protector: protectors reflect that references outlive the call
                // they were passed in to; that's just not the case for boxes.
                ty::Adt(..) if ty.is_box() => Some((RefKind::Unique, false)),
                _ => None,
            }
        }

        // We need a visitor to visit all references. However, that requires
        // a `MemPlace`, so we have a fast path for reference types that
        // avoids allocating.
        if let Some((mutbl, protector)) = qualify(place.layout.ty, kind) {
            // Fast path.
            let val = this.read_immediate(this.place_to_op(place)?)?;
            let val = this.retag_reference(val, mutbl, protector, kind == RetagKind::TwoPhase)?;
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
            type V = MPlaceTy<'tcx, Tag>;

            #[inline(always)]
            fn ecx(&mut self) -> &mut MiriEvalContext<'a, 'mir, 'tcx> {
                &mut self.ecx
            }

            // Primitives of reference type, that is the one thing we are interested in.
            fn visit_primitive(&mut self, place: MPlaceTy<'tcx, Tag>) -> EvalResult<'tcx>
            {
                // Cannot use `builtin_deref` because that reports *immutable* for `Box`,
                // making it useless.
                if let Some((mutbl, protector)) = qualify(place.layout.ty, self.kind) {
                    let val = self.ecx.read_immediate(place.into())?;
                    let val = self.ecx.retag_reference(
                        val,
                        mutbl,
                        protector,
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
