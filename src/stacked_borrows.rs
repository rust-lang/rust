use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;
use std::fmt;
use std::num::NonZeroU64;

use rustc::ty::{self, layout::Size};
use rustc::hir::{Mutability, MutMutable, MutImmutable};
use rustc::mir::RetagKind;

use crate::{
    EvalResult, InterpError, MiriEvalContext, HelpersEvalContextExt, Evaluator, MutValueVisitor,
    MemoryKind, MiriMemoryKind, RangeMap, Allocation, AllocationExtra,
    Pointer, Immediate, ImmTy, PlaceTy, MPlaceTy,
};

pub type PtrId = NonZeroU64;
pub type CallId = u64;

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
pub enum Item {
    /// Grants the given permission for pointers with this tag.
    Permission(Permission, Tag),
    /// A barrier, tracking the function it belongs to by its index on the call stack.
    FnBarrier(CallId),
}

impl fmt::Display for Item {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Item::Permission(perm, tag) => write!(f, "[{:?} for {}]", perm, tag),
            Item::FnBarrier(call) => write!(f, "[barrier {}]", call),
        }
    }
}

/// Extra per-location state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stack {
    /// Used *mostly* as a stack; never empty.
    /// We sometimes push into the middle but never remove from the middle.
    /// The same tag may occur multiple times, e.g. from a two-phase borrow.
    /// Invariants:
    /// * Above a `SharedReadOnly` there can only be barriers and more `SharedReadOnly`.
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
    Write { dealloc: bool },
}

// "Fake" constructors
impl AccessKind {
    fn write() -> AccessKind {
        AccessKind::Write { dealloc: false }
    }

    fn dealloc() -> AccessKind {
        AccessKind::Write { dealloc: true }
    }
}

impl fmt::Display for AccessKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            AccessKind::Read => write!(f, "read"),
            AccessKind::Write { dealloc: false } => write!(f, "write"),
            AccessKind::Write { dealloc: true } => write!(f, "deallocation"),
        }
    }
}

/// Indicates which kind of reference is being created.
/// Used by `reborrow` to compute which permissions to grant to the
/// new pointer.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum RefKind {
    /// `&mut`.
    Mutable,
    /// `&` with or without interior mutability.
    Shared { frozen: bool },
    /// `*` (raw pointer).
    Raw,
}

impl fmt::Display for RefKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            RefKind::Mutable => write!(f, "mutable"),
            RefKind::Shared { frozen: true } => write!(f, "shared (frozen)"),
            RefKind::Shared { frozen: false } => write!(f, "shared (mutable)"),
            RefKind::Raw => write!(f, "raw"),
        }
    }
}

/// Utilities for initialization and ID generation
impl Default for GlobalState {
    fn default() -> Self {
        GlobalState {
            next_ptr_id: NonZeroU64::new(1).unwrap(),
            next_call_id: 0,
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
        self.next_call_id = id+1;
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
/// U1: After creating a `Uniq`, it is at the top (and unfrozen).
/// U2: If the top is `Uniq` (and unfrozen), accesses must be through that `Uniq` or pop it.
/// U3: If an access happens with a `Uniq`, it requires the `Uniq` to be in the stack.
///
/// F1: After creating a `&`, the parts outside `UnsafeCell` are frozen.
/// F2: If a write access happens, it unfreezes.
/// F3: If an access happens with an `&` outside `UnsafeCell`,
///     it requires the location to still be frozen.

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
            (Permission::SharedReadOnly, AccessKind::Write { .. }) =>
                false,
        }
    }

    /// This defines for a given permission, which other items it can tolerate "above" itself
    /// for which kinds of accesses.
    /// If true, then `other` is allowed to remain on top of `self` when `access` happens.
    fn compatible_with(self, access: AccessKind, other: Item) -> bool {
        use self::Permission::*;

        let other = match other {
            Item::Permission(perm, _) => perm,
            Item::FnBarrier(_) => return false, // Remove all barriers -- if they are active, cause UB.
        };

        match (self, access, other) {
            // Some cases are impossible.
            (SharedReadOnly, _, SharedReadWrite) |
            (SharedReadOnly, _, Unique) =>
                bug!("There can never be a SharedReadWrite or a Unique on top of a SharedReadOnly"),
            // When `other` is `SharedReadOnly`, that is NEVER compatible with
            // write accesses.
            // This makes sure read-only pointers become invalid on write accesses.
            (_, AccessKind::Write { .. }, SharedReadOnly) =>
                false,
            // When `other` is `Unique`, that is compatible with nothing.
            // This makes sure unique pointers become invalid on incompatible accesses (ensures U2).
            (_, _, Unique) =>
                false,
            // When we are unique and this is a write/dealloc, we tolerate nothing.
            // This makes sure we re-assert uniqueness on write accesses.
            // (This is particularily important such that when a new mutable ref gets created, it gets
            // pushed into the right item -- this behaves like a write and we assert uniqueness of the
            // pointer from which this comes, *if* it was a unique pointer.)
            (Unique, AccessKind::Write { .. }, _) =>
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

impl<'tcx> RefKind {
    /// Defines which kind of access the "parent" must grant to create this reference.
    fn access(self) -> AccessKind {
        match self {
            RefKind::Mutable | RefKind::Shared { frozen: false } => AccessKind::write(),
            RefKind::Raw | RefKind::Shared { frozen: true } => AccessKind::Read,
            // FIXME: Just requiring read-only access for raw means that a raw ptr might not be writeable
            // even when we think it should be!  Think about this some more.
        }
    }

    /// This defines the new permission used when a pointer gets created: For raw pointers, whether these are read-only
    /// or read-write depends on the permission from which they derive.
    fn new_perm(self, derived_from: Permission) -> EvalResult<'tcx, Permission> {
        Ok(match (self, derived_from) {
            // Do not derive writable safe pointer from read-only pointer!
            (RefKind::Mutable, Permission::SharedReadOnly) =>
                return err!(MachineError(format!(
                    "deriving mutable reference from read-only pointer"
                ))),
            (RefKind::Shared { frozen: false }, Permission::SharedReadOnly) =>
                return err!(MachineError(format!(
                    "deriving shared reference with interior mutability from read-only pointer"
                ))),
            // Safe pointer cases.
            (RefKind::Mutable, _) => Permission::Unique,
            (RefKind::Shared { frozen: true }, _) => Permission::SharedReadOnly,
            (RefKind::Shared { frozen: false }, _) => Permission::SharedReadWrite,
            // Raw pointer cases.
            (RefKind::Raw, Permission::SharedReadOnly) => Permission::SharedReadOnly,
            (RefKind::Raw, _) => Permission::SharedReadWrite,
        })
    }
}

/// Core per-location operations: access, create.
impl<'tcx> Stack {
    /// Find the item granting the given kind of access to the given tag, and where that item is in the stack.
    fn find_granting(&self, access: AccessKind, tag: Tag) -> Option<(usize, Permission)> {
        self.borrows.iter()
            .enumerate() // we also need to know *where* in the stack
            .rev() // search top-to-bottom
            // Return permission of first item that grants access.
            .filter_map(|(idx, item)| match item {
                &Item::Permission(perm, item_tag) if perm.grants(access) && tag == item_tag =>
                    Some((idx, perm)),
                _ => None,
            })
            .next()
    }

    /// Test if a memory `access` using pointer tagged `tag` is granted.
    /// If yes, return the index of the item that granted it.
    fn access(
        &mut self,
        access: AccessKind,
        tag: Tag,
        global: &GlobalState,
    ) -> EvalResult<'tcx, usize> {
        // Two main steps: Find granting item, remove all incompatible items above.
        // Afterwards we just do some post-processing for deallocation accesses.

        // Step 1: Find granting item.
        let (granting_idx, granting_perm) = self.find_granting(access, tag)
            .ok_or_else(|| InterpError::MachineError(format!(
                    "no item granting {} access to tag {} found in borrow stack",
                    access, tag,
            )))?;
        
        // Step 2: Remove everything incompatible above them.
        // Implemented with indices because there does not seem to be a nice iterator and range-based
        // API for this.
        {
            let mut cur = granting_idx + 1;
            while let Some(item) = self.borrows.get(cur) {
                if granting_perm.compatible_with(access, *item) {
                    // Keep this, check next.
                    cur += 1;
                } else {
                    // Aha! This is a bad one, remove it, and if it is an *active* barrier
                    // we have a problem.
                    match self.borrows.remove(cur) {
                        Item::FnBarrier(call) if global.is_active(call) => {
                            return err!(MachineError(format!(
                                "not granting access because of barrier ({})", call
                            )));
                        }
                        _ => {}
                    }
                }
            }
        }

        // Post-processing.
        // If we got here, we found a matching item. Congratulations!
        // However, we are not done yet: If this access is deallocating, we must make sure
        // there are no active barriers remaining on the stack.
        if access == AccessKind::dealloc() {
            for &itm in self.borrows.iter().rev() {
                match itm {
                    Item::FnBarrier(call) if global.is_active(call) => {
                        return err!(MachineError(format!(
                            "deallocating with active barrier ({})", call
                        )))
                    }
                    _ => {},
                }
            }
        }

        // Done.
        return Ok(granting_idx);
    }

    /// `reborrow` helper function.
    /// Grant `permisson` to new pointer tagged `tag`, added at `position` in the stack.
    fn grant(&mut self, perm: Permission, tag: Tag, position: usize) {
        // Simply add it to the "stack" -- this might add in the middle.
        // As an optimization, do nothing if the new item is identical to one of its neighbors.
        let item = Item::Permission(perm, tag);
        if self.borrows[position-1] == item || self.borrows.get(position) == Some(&item) {
            // Optimization applies, done.
            trace!("reborrow: avoiding redundant item {}", item);
            return;
        }
        trace!("reborrow: pushing item {}", item);
        self.borrows.insert(position, item);
    }

    /// `reborrow` helper function.
    /// Adds a barrier.
    fn barrier(&mut self, call: CallId) {
        let itm = Item::FnBarrier(call);
        if *self.borrows.last().unwrap() == itm {
            // This is just an optimization, no functional change: Avoid stacking
            // multiple identical barriers on top of each other.
            // This can happen when a function receives several shared references
            // that overlap.
            trace!("reborrow: avoiding redundant extra barrier");
        } else {
            trace!("reborrow: pushing barrier for call {}", call);
            self.borrows.push(itm);
        }
    }

    /// `reborrow` helper function: test that the stack invariants are still maintained.
    fn test_invariants(&self) {
        let mut saw_shared_read_only = false;
        for item in self.borrows.iter() {
            match item {
                Item::Permission(Permission::SharedReadOnly, _) => {
                    saw_shared_read_only = true;
                }
                Item::Permission(perm, _) if saw_shared_read_only => {
                    panic!("Found {:?} on top of a SharedReadOnly!", perm);
                }
                _ => {}
            }
        }
    }

    /// Derived a new pointer from one with the given tag .
    fn reborrow(
        &mut self,
        derived_from: Tag,
        barrier: Option<CallId>,
        new_kind: RefKind,
        new_tag: Tag,
        global: &GlobalState,
    ) -> EvalResult<'tcx> {
        // Find the permission "from which we derive".  To this end we first have to decide
        // if we derive from a permission that grants writes or just reads.
        let access = new_kind.access();
        let (derived_from_idx, derived_from_perm) = self.find_granting(access, derived_from)
            .ok_or_else(|| InterpError::MachineError(format!(
                    "no item to reborrow as {} from tag {} found in borrow stack", new_kind, derived_from,
            )))?;
        // With this we can compute the permission for the new pointer.
        let new_perm = new_kind.new_perm(derived_from_perm)?;

        // We behave very differently for the "unsafe" case of a shared-read-write pointer
        // ("unsafe" because this also applies to shared references with interior mutability).
        // This is because such pointers may be reborrowed to unique pointers that actually
        // remain valid when their "parents" get further reborrows!
        if new_perm == Permission::SharedReadWrite {
            // A very liberal reborrow because the new pointer does not expect any kind of aliasing guarantee.
            // Just insert new permission as child of old permission, and maintain everything else.
            // This inserts "as far down as possible", which is good because it makes this pointer as
            // long-lived as possible *and* we want all the items that are incompatible with this
            // to actually get removed from the stack.  If we pushed a `SharedReadWrite` on top of
            // a `SharedReadOnly`, we'd violate the invariant that `SaredReadOnly` are at the top
            // and we'd allow write access without invalidating frozen shared references!
            self.grant(new_perm, new_tag, derived_from_idx+1);

            // No barrier. They can rightfully alias with `&mut`.
            // FIXME: This means that the `dereferencable` attribute on non-frozen shared references
            // is incorrect! They are dereferencable when the function is called, but might become
            // non-dereferencable during the course of execution.
            // Also see [1], [2].
            //
            // [1]: <https://internals.rust-lang.org/t/
            //       is-it-possible-to-be-memory-safe-with-deallocated-self/8457/8>,
            // [2]: <https://lists.llvm.org/pipermail/llvm-dev/2018-July/124555.html>
        } else {
            // A "safe" reborrow for a pointer that actually expects some aliasing guarantees.
            // Here, creating a reference actually counts as an access, and pops incompatible
            // stuff off the stack.
            let check_idx = self.access(access, derived_from, global)?;
            assert_eq!(check_idx, derived_from_idx, "somehow we saw different items??");

            // Now is a good time to add the barrier.
            if let Some(call) = barrier {
                self.barrier(call);
            }

            // We insert "as far up as possible": We know only compatible items are remaining
            // on top of `derived_from`, and we want the new item at the top so that we
            // get the strongest possible guarantees.
            self.grant(new_perm, new_tag, self.borrows.len());
        }

        // Make sure that after all this, the stack's invariant is still maintained.
        if cfg!(debug_assertions) {
            self.test_invariants();
        }

        Ok(())
    }
}

/// Higher-level per-location operations: deref, access, reborrow.
impl<'tcx> Stacks {
    /// Creates new stack with initial tag.
    pub(crate) fn new(
        size: Size,
        tag: Tag,
        extra: MemoryState,
    ) -> Self {
        let item = Item::Permission(Permission::Unique, tag);
        let stack = Stack {
            borrows: vec![item],
        };
        Stacks {
            stacks: RefCell::new(RangeMap::new(size, stack)),
            global: extra,
        }
    }

    /// `ptr` got used, reflect that in the stack.
    fn access(
        &self,
        ptr: Pointer<Tag>,
        size: Size,
        kind: AccessKind,
    ) -> EvalResult<'tcx> {
        trace!("{} access of tag {}: {:?}, size {}", kind, ptr.tag, ptr, size.bytes());
        // Even reads can have a side-effect, by invalidating other references.
        // This is fundamentally necessary since `&mut` asserts that there
        // are no accesses through other references, not even reads.
        let global = self.global.borrow();
        let mut stacks = self.stacks.borrow_mut();
        for stack in stacks.iter_mut(ptr.offset, size) {
            stack.access(kind, ptr.tag, &*global)?;
        }
        Ok(())
    }

    /// Reborrow the given pointer to the new tag for the given kind of reference.
    /// This works on `&self` because we might encounter references to constant memory.
    fn reborrow(
        &self,
        ptr: Pointer<Tag>,
        size: Size,
        barrier: Option<CallId>,
        new_kind: RefKind,
        new_tag: Tag,
    ) -> EvalResult<'tcx> {
        trace!(
            "{} reborrow for tag {} to {}: {:?}, size {}",
            new_kind, ptr.tag, new_tag, ptr, size.bytes(),
        );
        let global = self.global.borrow();
        let mut stacks = self.stacks.borrow_mut();
        for stack in stacks.iter_mut(ptr.offset, size) {
            stack.reborrow(ptr.tag, barrier, new_kind, new_tag, &*global)?;
        }
        Ok(())
    }
}

// # Stacked Borrows Core End

// Glue code to connect with Miri Machine Hooks

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
        alloc.extra.access(ptr, size, AccessKind::Read)
    }

    #[inline(always)]
    fn memory_written<'tcx>(
        alloc: &mut Allocation<Tag, Stacks>,
        ptr: Pointer<Tag>,
        size: Size,
    ) -> EvalResult<'tcx> {
        alloc.extra.access(ptr, size, AccessKind::write())
    }

    #[inline(always)]
    fn memory_deallocated<'tcx>(
        alloc: &mut Allocation<Tag, Stacks>,
        ptr: Pointer<Tag>,
        size: Size,
    ) -> EvalResult<'tcx> {
        alloc.extra.access(ptr, size, AccessKind::dealloc())
    }
}

impl<'a, 'mir, 'tcx> EvalContextPrivExt<'a, 'mir, 'tcx> for crate::MiriEvalContext<'a, 'mir, 'tcx> {}
trait EvalContextPrivExt<'a, 'mir, 'tcx: 'a+'mir>: crate::MiriEvalContextExt<'a, 'mir, 'tcx> {
    fn reborrow(
        &mut self,
        place: MPlaceTy<'tcx, Tag>,
        size: Size,
        mutbl: Option<Mutability>,
        new_tag: Tag,
        fn_barrier: bool,
    ) -> EvalResult<'tcx> {
        let this = self.eval_context_mut();
        let barrier = if fn_barrier { Some(this.frame().extra) } else { None };
        let ptr = place.ptr.to_ptr()?;
        trace!("reborrow: creating new reference for {:?} (pointee {}): {:?}",
            ptr, place.layout.ty, new_tag);

        // Get the allocation. It might not be mutable, so we cannot use `get_mut`.
        let alloc = this.memory().get(ptr.alloc_id)?;
        alloc.check_bounds(this, ptr, size)?;
        // Update the stacks.
        if mutbl == Some(MutImmutable) {
            // Reference that cares about freezing. We need a frozen-sensitive reborrow.
            this.visit_freeze_sensitive(place, size, |cur_ptr, size, frozen| {
                let new_kind = RefKind::Shared { frozen };
                alloc.extra.reborrow(cur_ptr, size, barrier, new_kind, new_tag)
            })?;
        } else {
            // Just treat this as one big chunk.
            let new_kind = if mutbl == Some(MutMutable) { RefKind::Mutable } else { RefKind::Raw };
            alloc.extra.reborrow(ptr, size, barrier, new_kind, new_tag)?;
        }
        Ok(())
    }

    /// Retags an indidual pointer, returning the retagged version.
    /// `mutbl` can be `None` to make this a raw pointer.
    fn retag_reference(
        &mut self,
        val: ImmTy<'tcx, Tag>,
        mutbl: Option<Mutability>,
        fn_barrier: bool,
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
        let new_tag = match mutbl {
            Some(_) => Tag::Tagged(this.memory().extra.borrow_mut().new_ptr()),
            None => Tag::Untagged,
        };

        // Reborrow.
        this.reborrow(place, size, mutbl, new_tag, fn_barrier)?;
        let new_place = place.replace_tag(new_tag);
        // Handle two-phase borrows.
        if two_phase {
            assert!(mutbl == Some(MutMutable), "two-phase shared borrows make no sense");
            // Grant read access *to the parent pointer* with the old tag.  This means the same pointer
            // has multiple items in the stack now!
            // FIXME: Think about this some more, in particular about the interaction with cast-to-raw.
            // Maybe find a better way to express 2-phase, now that we have a "more expressive language"
            // in the stack.
            let old_tag = place.ptr.to_ptr().unwrap().tag;
            this.reborrow(new_place, size, Some(MutImmutable), old_tag, /* fn_barrier: */ false)?;
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
