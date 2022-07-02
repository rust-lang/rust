//! Implements "Stacked Borrows".  See <https://github.com/rust-lang/unsafe-code-guidelines/blob/master/wip/stacked-borrows.md>
//! for further information.

use log::trace;
use std::cell::RefCell;
use std::cmp;
use std::fmt;
use std::num::NonZeroU64;

use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_hir::Mutability;
use rustc_middle::mir::RetagKind;
use rustc_middle::ty::{
    self,
    layout::{HasParamEnv, LayoutOf},
};
use rustc_span::DUMMY_SP;
use rustc_target::abi::Size;
use std::collections::HashSet;

use crate::*;

pub mod diagnostics;
use diagnostics::{AllocHistory, TagHistory};

pub mod stack;
use stack::Stack;

pub type CallId = NonZeroU64;

// Even reading memory can have effects on the stack, so we need a `RefCell` here.
pub type AllocExtra = RefCell<Stacks>;

/// Tracking pointer provenance
#[derive(Copy, Clone, Hash, PartialEq, Eq)]
pub struct SbTag(NonZeroU64);

impl SbTag {
    pub fn new(i: u64) -> Option<Self> {
        NonZeroU64::new(i).map(SbTag)
    }

    // The default to be used when SB is disabled
    pub fn default() -> Self {
        Self::new(1).unwrap()
    }
}

impl fmt::Debug for SbTag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<{}>", self.0)
    }
}

/// The "extra" information an SB pointer has over a regular AllocId.
/// Newtype for `Option<SbTag>`.
#[derive(Copy, Clone)]
pub enum SbTagExtra {
    Concrete(SbTag),
    Wildcard,
}

impl fmt::Debug for SbTagExtra {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SbTagExtra::Concrete(pid) => write!(f, "{pid:?}"),
            SbTagExtra::Wildcard => write!(f, "<wildcard>"),
        }
    }
}

impl SbTagExtra {
    fn and_then<T>(self, f: impl FnOnce(SbTag) -> Option<T>) -> Option<T> {
        match self {
            SbTagExtra::Concrete(pid) => f(pid),
            SbTagExtra::Wildcard => None,
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
    /// Grants shared read-only access.
    SharedReadOnly,
    /// Grants no access, but separates two groups of SharedReadWrite so they are not
    /// all considered mutually compatible.
    Disabled,
}

/// An item in the per-location borrow stack.
#[derive(Copy, Clone, Hash, PartialEq, Eq)]
pub struct Item {
    /// The permission this item grants.
    perm: Permission,
    /// The pointers the permission is granted to.
    tag: SbTag,
    /// An optional protector, ensuring the item cannot get popped until `CallId` is over.
    protector: Option<CallId>,
}

impl fmt::Debug for Item {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{:?} for {:?}", self.perm, self.tag)?;
        if let Some(call) = self.protector {
            write!(f, " (call {})", call)?;
        }
        write!(f, "]")?;
        Ok(())
    }
}

/// Extra per-allocation state.
#[derive(Clone, Debug)]
pub struct Stacks {
    // Even reading memory can have effects on the stack, so we need a `RefCell` here.
    stacks: RangeMap<Stack>,
    /// Stores past operations on this allocation
    history: AllocHistory,
    /// The set of tags that have been exposed inside this allocation.
    exposed_tags: FxHashSet<SbTag>,
}

/// Extra global state, available to the memory access hooks.
#[derive(Debug)]
pub struct GlobalStateInner {
    /// Next unused pointer ID (tag).
    next_ptr_tag: SbTag,
    /// Table storing the "base" tag for each allocation.
    /// The base tag is the one used for the initial pointer.
    /// We need this in a separate table to handle cyclic statics.
    base_ptr_tags: FxHashMap<AllocId, SbTag>,
    /// Next unused call ID (for protectors).
    next_call_id: CallId,
    /// Those call IDs corresponding to functions that are still running.
    active_calls: FxHashSet<CallId>,
    /// The pointer ids to trace
    tracked_pointer_tags: HashSet<SbTag>,
    /// The call ids to trace
    tracked_call_ids: HashSet<CallId>,
    /// Whether to recurse into datatypes when searching for pointers to retag.
    retag_fields: bool,
}

/// We need interior mutable access to the global state.
pub type GlobalState = RefCell<GlobalStateInner>;

/// Indicates which kind of access is being performed.
#[derive(Copy, Clone, Hash, PartialEq, Eq, Debug)]
pub enum AccessKind {
    Read,
    Write,
}

impl fmt::Display for AccessKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AccessKind::Read => write!(f, "read access"),
            AccessKind::Write => write!(f, "write access"),
        }
    }
}

/// Indicates which kind of reference is being created.
/// Used by high-level `reborrow` to compute which permissions to grant to the
/// new pointer.
#[derive(Copy, Clone, Hash, PartialEq, Eq)]
pub enum RefKind {
    /// `&mut` and `Box`.
    Unique { two_phase: bool },
    /// `&` with or without interior mutability.
    Shared,
    /// `*mut`/`*const` (raw pointers).
    Raw { mutable: bool },
}

impl fmt::Display for RefKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RefKind::Unique { two_phase: false } => write!(f, "unique"),
            RefKind::Unique { two_phase: true } => write!(f, "unique (two-phase)"),
            RefKind::Shared => write!(f, "shared"),
            RefKind::Raw { mutable: true } => write!(f, "raw (mutable)"),
            RefKind::Raw { mutable: false } => write!(f, "raw (constant)"),
        }
    }
}

/// Utilities for initialization and ID generation
impl GlobalStateInner {
    pub fn new(
        tracked_pointer_tags: HashSet<SbTag>,
        tracked_call_ids: HashSet<CallId>,
        retag_fields: bool,
    ) -> Self {
        GlobalStateInner {
            next_ptr_tag: SbTag(NonZeroU64::new(1).unwrap()),
            base_ptr_tags: FxHashMap::default(),
            next_call_id: NonZeroU64::new(1).unwrap(),
            active_calls: FxHashSet::default(),
            tracked_pointer_tags,
            tracked_call_ids,
            retag_fields,
        }
    }

    fn new_ptr(&mut self) -> SbTag {
        let id = self.next_ptr_tag;
        if self.tracked_pointer_tags.contains(&id) {
            register_diagnostic(NonHaltingDiagnostic::CreatedPointerTag(id.0));
        }
        self.next_ptr_tag = SbTag(NonZeroU64::new(id.0.get() + 1).unwrap());
        id
    }

    pub fn new_call(&mut self) -> CallId {
        let id = self.next_call_id;
        trace!("new_call: Assigning ID {}", id);
        if self.tracked_call_ids.contains(&id) {
            register_diagnostic(NonHaltingDiagnostic::CreatedCallId(id));
        }
        assert!(self.active_calls.insert(id));
        self.next_call_id = NonZeroU64::new(id.get() + 1).unwrap();
        id
    }

    pub fn end_call(&mut self, id: CallId) {
        assert!(self.active_calls.remove(&id));
    }

    fn is_active(&self, id: CallId) -> bool {
        self.active_calls.contains(&id)
    }

    pub fn base_ptr_tag(&mut self, id: AllocId) -> SbTag {
        self.base_ptr_tags.get(&id).copied().unwrap_or_else(|| {
            let tag = self.new_ptr();
            trace!("New allocation {:?} has base tag {:?}", id, tag);
            self.base_ptr_tags.try_insert(id, tag).unwrap();
            tag
        })
    }
}

/// Error reporting
pub fn err_sb_ub<'tcx>(
    msg: String,
    help: Option<String>,
    history: Option<TagHistory>,
) -> InterpError<'tcx> {
    err_machine_stop!(TerminationInfo::StackedBorrowsUb { msg, help, history })
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

/// Core relation on `Permission` to define which accesses are allowed
impl Permission {
    /// This defines for a given permission, whether it permits the given kind of access.
    fn grants(self, access: AccessKind) -> bool {
        // Disabled grants nothing. Otherwise, all items grant read access, and except for SharedReadOnly they grant write access.
        self != Permission::Disabled
            && (access == AccessKind::Read || self != Permission::SharedReadOnly)
    }
}

/// Core per-location operations: access, dealloc, reborrow.
impl<'tcx> Stack {
    /// Find the first write-incompatible item above the given one --
    /// i.e, find the height to which the stack will be truncated when writing to `granting`.
    fn find_first_write_incompatible(&self, granting: usize) -> usize {
        let perm = self.get(granting).unwrap().perm;
        match perm {
            Permission::SharedReadOnly => bug!("Cannot use SharedReadOnly for writing"),
            Permission::Disabled => bug!("Cannot use Disabled for anything"),
            Permission::Unique => {
                // On a write, everything above us is incompatible.
                granting + 1
            }
            Permission::SharedReadWrite => {
                // The SharedReadWrite *just* above us are compatible, to skip those.
                let mut idx = granting + 1;
                while let Some(item) = self.get(idx) {
                    if item.perm == Permission::SharedReadWrite {
                        // Go on.
                        idx += 1;
                    } else {
                        // Found first incompatible!
                        break;
                    }
                }
                idx
            }
        }
    }

    /// Check if the given item is protected.
    ///
    /// The `provoking_access` argument is only used to produce diagnostics.
    /// It is `Some` when we are granting the contained access for said tag, and it is
    /// `None` during a deallocation.
    /// Within `provoking_access, the `AllocRange` refers the entire operation, and
    /// the `Size` refers to the specific location in the `AllocRange` that we are
    /// currently checking.
    fn item_popped(
        item: &Item,
        provoking_access: Option<(SbTagExtra, AllocRange, Size, AccessKind)>, // just for debug printing and error messages
        global: &GlobalStateInner,
        alloc_history: &mut AllocHistory,
    ) -> InterpResult<'tcx> {
        if global.tracked_pointer_tags.contains(&item.tag) {
            register_diagnostic(NonHaltingDiagnostic::PoppedPointerTag(
                *item,
                provoking_access.map(|(tag, _alloc_range, _size, access)| (tag, access)),
            ));
        }

        if let Some(call) = item.protector {
            if global.is_active(call) {
                if let Some((tag, _alloc_range, _offset, _access)) = provoking_access {
                    Err(err_sb_ub(
                        format!(
                            "not granting access to tag {:?} because incompatible item is protected: {:?}",
                            tag, item
                        ),
                        None,
                        tag.and_then(|tag| alloc_history.get_logs_relevant_to(tag, Some(item.tag))),
                    ))?
                } else {
                    Err(err_sb_ub(
                        format!("deallocating while item is protected: {:?}", item),
                        None,
                        None,
                    ))?
                }
            }
        }
        Ok(())
    }

    /// Test if a memory `access` using pointer tagged `tag` is granted.
    /// If yes, return the index of the item that granted it.
    /// `range` refers the entire operation, and `offset` refers to the specific offset into the
    /// allocation that we are currently checking.
    fn access(
        &mut self,
        access: AccessKind,
        tag: SbTagExtra,
        (alloc_id, alloc_range, offset): (AllocId, AllocRange, Size), // just for debug printing and error messages
        global: &mut GlobalStateInner,
        current_span: &mut CurrentSpan<'_, '_, 'tcx>,
        alloc_history: &mut AllocHistory,
        exposed_tags: &FxHashSet<SbTag>,
    ) -> InterpResult<'tcx> {
        // Two main steps: Find granting item, remove incompatible items above.

        // Step 1: Find granting item.
        let granting_idx = self.find_granting(access, tag, exposed_tags).map_err(|_| {
            alloc_history.access_error(access, tag, alloc_id, alloc_range, offset, self)
        })?;

        // Step 2: Remove incompatible items above them.  Make sure we do not remove protected
        // items.  Behavior differs for reads and writes.
        // In case of wildcards/unknown matches, we remove everything that is *definitely* gone.
        if access == AccessKind::Write {
            // Remove everything above the write-compatible items, like a proper stack. This makes sure read-only and unique
            // pointers become invalid on write accesses (ensures F2a, and ensures U2 for write accesses).
            let first_incompatible_idx = if let Some(granting_idx) = granting_idx {
                // The granting_idx *might* be approximate, but any lower idx would remove more
                // things. Even if this is a Unique and the lower idx is an SRW (which removes
                // less), there is an SRW group boundary here so strictly more would get removed.
                self.find_first_write_incompatible(granting_idx)
            } else {
                // We are writing to something in the unknown part.
                // There is a SRW group boundary between the unknown and the known, so everything is incompatible.
                0
            };
            self.pop_items_after(first_incompatible_idx, |item| {
                Stack::item_popped(
                    &item,
                    Some((tag, alloc_range, offset, access)),
                    global,
                    alloc_history,
                )?;
                alloc_history.log_invalidation(item.tag, alloc_range, current_span);
                Ok(())
            })?;
        } else {
            // On a read, *disable* all `Unique` above the granting item.  This ensures U2 for read accesses.
            // The reason this is not following the stack discipline (by removing the first Unique and
            // everything on top of it) is that in `let raw = &mut *x as *mut _; let _val = *x;`, the second statement
            // would pop the `Unique` from the reborrow of the first statement, and subsequently also pop the
            // `SharedReadWrite` for `raw`.
            // This pattern occurs a lot in the standard library: create a raw pointer, then also create a shared
            // reference and use that.
            // We *disable* instead of removing `Unique` to avoid "connecting" two neighbouring blocks of SRWs.
            let first_incompatible_idx = if let Some(granting_idx) = granting_idx {
                // The granting_idx *might* be approximate, but any lower idx would disable more things.
                granting_idx + 1
            } else {
                // We are reading from something in the unknown part. That means *all* `Unique` we know about are dead now.
                0
            };
            self.disable_uniques_starting_at(first_incompatible_idx, |item| {
                Stack::item_popped(
                    &item,
                    Some((tag, alloc_range, offset, access)),
                    global,
                    alloc_history,
                )?;
                alloc_history.log_invalidation(item.tag, alloc_range, current_span);
                Ok(())
            })?;
        }

        // If this was an approximate action, we now collapse everything into an unknown.
        if granting_idx.is_none() || matches!(tag, SbTagExtra::Wildcard) {
            // Compute the upper bound of the items that remain.
            // (This is why we did all the work above: to reduce the items we have to consider here.)
            let mut max = NonZeroU64::new(1).unwrap();
            for i in 0..self.len() {
                let item = self.get(i).unwrap();
                // Skip disabled items, they cannot be matched anyway.
                if !matches!(item.perm, Permission::Disabled) {
                    // We are looking for a strict upper bound, so add 1 to this tag.
                    max = cmp::max(item.tag.0.checked_add(1).unwrap(), max);
                }
            }
            if let Some(unk) = self.unknown_bottom() {
                max = cmp::max(unk.0, max);
            }
            // Use `max` as new strict upper bound for everything.
            trace!(
                "access: forgetting stack to upper bound {max} due to wildcard or unknown access"
            );
            self.set_unknown_bottom(SbTag(max));
        }

        // Done.
        Ok(())
    }

    /// Deallocate a location: Like a write access, but also there must be no
    /// active protectors at all because we will remove all items.
    fn dealloc(
        &mut self,
        tag: SbTagExtra,
        (alloc_id, _alloc_range, _offset): (AllocId, AllocRange, Size), // just for debug printing and error messages
        global: &GlobalStateInner,
        alloc_history: &mut AllocHistory,
        exposed_tags: &FxHashSet<SbTag>,
    ) -> InterpResult<'tcx> {
        // Step 1: Make sure there is a granting item.
        self.find_granting(AccessKind::Write, tag, exposed_tags).map_err(|_| {
            err_sb_ub(format!(
                "no item granting write access for deallocation to tag {:?} at {:?} found in borrow stack",
                tag, alloc_id,
                ),
                None,
                tag.and_then(|tag| alloc_history.get_logs_relevant_to(tag, None)),
            )
        })?;

        // Step 2: Consider all items removed. This checks for protectors.
        for idx in (0..self.len()).rev() {
            let item = self.get(idx).unwrap();
            Stack::item_popped(&item, None, global, alloc_history)?;
        }
        Ok(())
    }

    /// Derive a new pointer from one with the given tag.
    /// `weak` controls whether this operation is weak or strong: weak granting does not act as
    /// an access, and they add the new item directly on top of the one it is derived
    /// from instead of all the way at the top of the stack.
    /// `range` refers the entire operation, and `offset` refers to the specific location in
    /// `range` that we are currently checking.
    fn grant(
        &mut self,
        derived_from: SbTagExtra,
        new: Item,
        (alloc_id, alloc_range, offset): (AllocId, AllocRange, Size), // just for debug printing and error messages
        global: &mut GlobalStateInner,
        current_span: &mut CurrentSpan<'_, '_, 'tcx>,
        alloc_history: &mut AllocHistory,
        exposed_tags: &FxHashSet<SbTag>,
    ) -> InterpResult<'tcx> {
        // Figure out which access `perm` corresponds to.
        let access =
            if new.perm.grants(AccessKind::Write) { AccessKind::Write } else { AccessKind::Read };

        // Now we figure out which item grants our parent (`derived_from`) this kind of access.
        // We use that to determine where to put the new item.
        let granting_idx =
            self.find_granting(access, derived_from, exposed_tags).map_err(|_| {
                alloc_history.grant_error(derived_from, new, alloc_id, alloc_range, offset, self)
            })?;

        // Compute where to put the new item.
        // Either way, we ensure that we insert the new item in a way such that between
        // `derived_from` and the new one, there are only items *compatible with* `derived_from`.
        let new_idx = if new.perm == Permission::SharedReadWrite {
            assert!(
                access == AccessKind::Write,
                "this case only makes sense for stack-like accesses"
            );

            let (Some(granting_idx), SbTagExtra::Concrete(_)) = (granting_idx, derived_from) else {
                // The parent is a wildcard pointer or matched the unknown bottom.
                // This is approximate. Nobody knows what happened, so forget everything.
                // The new thing is SRW anyway, so we cannot push it "on top of the unkown part"
                // (for all we know, it might join an SRW group inside the unknown).
                trace!("reborrow: forgetting stack entirely due to SharedReadWrite reborrow from wildcard or unknown");
                self.set_unknown_bottom(global.next_ptr_tag);
                return Ok(());
            };

            // SharedReadWrite can coexist with "existing loans", meaning they don't act like a write
            // access.  Instead of popping the stack, we insert the item at the place the stack would
            // be popped to (i.e., we insert it above all the write-compatible items).
            // This ensures F2b by adding the new item below any potentially existing `SharedReadOnly`.
            self.find_first_write_incompatible(granting_idx)
        } else {
            // A "safe" reborrow for a pointer that actually expects some aliasing guarantees.
            // Here, creating a reference actually counts as an access.
            // This ensures F2b for `Unique`, by removing offending `SharedReadOnly`.
            self.access(
                access,
                derived_from,
                (alloc_id, alloc_range, offset),
                global,
                current_span,
                alloc_history,
                exposed_tags,
            )?;

            // We insert "as far up as possible": We know only compatible items are remaining
            // on top of `derived_from`, and we want the new item at the top so that we
            // get the strongest possible guarantees.
            // This ensures U1 and F1.
            self.len()
        };

        // Put the new item there. As an optimization, deduplicate if it is equal to one of its new neighbors.
        // `new_idx` might be 0 if we just cleared the entire stack.
        if self.get(new_idx) == Some(new) || (new_idx > 0 && self.get(new_idx - 1).unwrap() == new)
        {
            // Optimization applies, done.
            trace!("reborrow: avoiding adding redundant item {:?}", new);
        } else {
            trace!("reborrow: adding item {:?}", new);
            self.insert(new_idx, new);
        }
        Ok(())
    }
}
// # Stacked Borrows Core End

/// Map per-stack operations to higher-level per-location-range operations.
impl<'tcx> Stacks {
    /// Creates new stack with initial tag.
    fn new(size: Size, perm: Permission, tag: SbTag) -> Self {
        let item = Item { perm, tag, protector: None };
        let stack = Stack::new(item);

        Stacks {
            stacks: RangeMap::new(size, stack),
            history: AllocHistory::new(),
            exposed_tags: FxHashSet::default(),
        }
    }

    /// Call `f` on every stack in the range.
    fn for_each(
        &mut self,
        range: AllocRange,
        mut f: impl FnMut(
            Size,
            &mut Stack,
            &mut AllocHistory,
            &mut FxHashSet<SbTag>,
        ) -> InterpResult<'tcx>,
    ) -> InterpResult<'tcx> {
        for (offset, stack) in self.stacks.iter_mut(range.start, range.size) {
            f(offset, stack, &mut self.history, &mut self.exposed_tags)?;
        }
        Ok(())
    }
}

/// Glue code to connect with Miri Machine Hooks
impl Stacks {
    pub fn new_allocation(
        id: AllocId,
        size: Size,
        state: &GlobalState,
        kind: MemoryKind<MiriMemoryKind>,
        mut current_span: CurrentSpan<'_, '_, '_>,
    ) -> Self {
        let mut extra = state.borrow_mut();
        let (base_tag, perm) = match kind {
            // New unique borrow. This tag is not accessible by the program,
            // so it will only ever be used when using the local directly (i.e.,
            // not through a pointer). That is, whenever we directly write to a local, this will pop
            // everything else off the stack, invalidating all previous pointers,
            // and in particular, *all* raw pointers.
            MemoryKind::Stack => (extra.base_ptr_tag(id), Permission::Unique),
            // Everything else is shared by default.
            _ => (extra.base_ptr_tag(id), Permission::SharedReadWrite),
        };
        let mut stacks = Stacks::new(size, perm, base_tag);
        stacks.history.log_creation(
            None,
            base_tag,
            alloc_range(Size::ZERO, size),
            &mut current_span,
        );
        stacks
    }

    #[inline(always)]
    pub fn memory_read<'tcx>(
        &mut self,
        alloc_id: AllocId,
        tag: SbTagExtra,
        range: AllocRange,
        state: &GlobalState,
        mut current_span: CurrentSpan<'_, '_, 'tcx>,
    ) -> InterpResult<'tcx> {
        trace!(
            "read access with tag {:?}: {:?}, size {}",
            tag,
            Pointer::new(alloc_id, range.start),
            range.size.bytes()
        );
        let mut state = state.borrow_mut();
        self.for_each(range, |offset, stack, history, exposed_tags| {
            stack.access(
                AccessKind::Read,
                tag,
                (alloc_id, range, offset),
                &mut state,
                &mut current_span,
                history,
                exposed_tags,
            )
        })
    }

    #[inline(always)]
    pub fn memory_written<'tcx>(
        &mut self,
        alloc_id: AllocId,
        tag: SbTagExtra,
        range: AllocRange,
        state: &GlobalState,
        mut current_span: CurrentSpan<'_, '_, 'tcx>,
    ) -> InterpResult<'tcx> {
        trace!(
            "write access with tag {:?}: {:?}, size {}",
            tag,
            Pointer::new(alloc_id, range.start),
            range.size.bytes()
        );
        let mut state = state.borrow_mut();
        self.for_each(range, |offset, stack, history, exposed_tags| {
            stack.access(
                AccessKind::Write,
                tag,
                (alloc_id, range, offset),
                &mut state,
                &mut current_span,
                history,
                exposed_tags,
            )
        })
    }

    #[inline(always)]
    pub fn memory_deallocated<'tcx>(
        &mut self,
        alloc_id: AllocId,
        tag: SbTagExtra,
        range: AllocRange,
        state: &GlobalState,
    ) -> InterpResult<'tcx> {
        trace!("deallocation with tag {:?}: {:?}, size {}", tag, alloc_id, range.size.bytes());
        let state = state.borrow();
        self.for_each(range, |offset, stack, history, exposed_tags| {
            stack.dealloc(tag, (alloc_id, range, offset), &state, history, exposed_tags)
        })?;
        Ok(())
    }
}

/// Retagging/reborrowing.  There is some policy in here, such as which permissions
/// to grant for which references, and when to add protectors.
impl<'mir, 'tcx: 'mir> EvalContextPrivExt<'mir, 'tcx> for crate::MiriEvalContext<'mir, 'tcx> {}
trait EvalContextPrivExt<'mir, 'tcx: 'mir>: crate::MiriEvalContextExt<'mir, 'tcx> {
    /// Returns the `AllocId` the reborrow was done in, if some actual borrow stack manipulation
    /// happened.
    fn reborrow(
        &mut self,
        place: &MPlaceTy<'tcx, Tag>,
        size: Size,
        kind: RefKind,
        new_tag: SbTag,
        protect: bool,
    ) -> InterpResult<'tcx, Option<AllocId>> {
        let this = self.eval_context_mut();
        let current_span = &mut this.machine.current_span();

        let log_creation = |this: &MiriEvalContext<'mir, 'tcx>,
                            current_span: &mut CurrentSpan<'_, 'mir, 'tcx>,
                            alloc_id,
                            base_offset,
                            orig_tag|
         -> InterpResult<'tcx> {
            let SbTagExtra::Concrete(orig_tag) = orig_tag else {
                // FIXME: should we log this?
                return Ok(())
            };
            let extra = this.get_alloc_extra(alloc_id)?;
            let mut stacked_borrows = extra
                .stacked_borrows
                .as_ref()
                .expect("we should have Stacked Borrows data")
                .borrow_mut();
            stacked_borrows.history.log_creation(
                Some(orig_tag),
                new_tag,
                alloc_range(base_offset, size),
                current_span,
            );
            if protect {
                stacked_borrows.history.log_protector(orig_tag, new_tag, current_span);
            }
            Ok(())
        };

        if size == Size::ZERO {
            trace!(
                "reborrow of size 0: {} reference {:?} derived from {:?} (pointee {})",
                kind,
                new_tag,
                place.ptr,
                place.layout.ty,
            );
            // Don't update any stacks for a zero-sized access; borrow stacks are per-byte and this
            // touches no bytes so there is no stack to put this tag in.
            // However, if the pointer for this operation points at a real allocation we still
            // record where it was created so that we can issue a helpful diagnostic if there is an
            // attempt to use it for a non-zero-sized access.
            // Dangling slices are a common case here; it's valid to get their length but with raw
            // pointer tagging for example all calls to get_unchecked on them are invalid.
            if let Ok((alloc_id, base_offset, orig_tag)) = this.ptr_try_get_alloc_id(place.ptr) {
                log_creation(this, current_span, alloc_id, base_offset, orig_tag)?;
                return Ok(Some(alloc_id));
            }
            // This pointer doesn't come with an AllocId. :shrug:
            return Ok(None);
        }
        let (alloc_id, base_offset, orig_tag) = this.ptr_get_alloc_id(place.ptr)?;
        log_creation(this, current_span, alloc_id, base_offset, orig_tag)?;

        // Ensure we bail out if the pointer goes out-of-bounds (see miri#1050).
        let (alloc_size, _) = this.get_live_alloc_size_and_align(alloc_id)?;
        if base_offset + size > alloc_size {
            throw_ub!(PointerOutOfBounds {
                alloc_id,
                alloc_size,
                ptr_offset: this.machine_usize_to_isize(base_offset.bytes()),
                ptr_size: size,
                msg: CheckInAllocMsg::InboundsTest
            });
        }

        let protector = if protect { Some(this.frame().extra.call_id) } else { None };
        trace!(
            "reborrow: {} reference {:?} derived from {:?} (pointee {}): {:?}, size {}",
            kind,
            new_tag,
            orig_tag,
            place.layout.ty,
            Pointer::new(alloc_id, base_offset),
            size.bytes()
        );

        // Update the stacks.
        // Make sure that raw pointers and mutable shared references are reborrowed "weak":
        // There could be existing unique pointers reborrowed from them that should remain valid!
        let perm = match kind {
            RefKind::Unique { two_phase: false }
                if place.layout.ty.is_unpin(this.tcx.at(DUMMY_SP), this.param_env()) =>
            {
                // Only if the type is unpin do we actually enforce uniqueness
                Permission::Unique
            }
            RefKind::Unique { .. } => {
                // Two-phase references and !Unpin references are treated as SharedReadWrite
                Permission::SharedReadWrite
            }
            RefKind::Raw { mutable: true } => Permission::SharedReadWrite,
            RefKind::Shared | RefKind::Raw { mutable: false } => {
                // Shared references and *const are a whole different kind of game, the
                // permission is not uniform across the entire range!
                // We need a frozen-sensitive reborrow.
                // We have to use shared references to alloc/memory_extra here since
                // `visit_freeze_sensitive` needs to access the global state.
                let extra = this.get_alloc_extra(alloc_id)?;
                let mut stacked_borrows = extra
                    .stacked_borrows
                    .as_ref()
                    .expect("we should have Stacked Borrows data")
                    .borrow_mut();
                this.visit_freeze_sensitive(place, size, |mut range, frozen| {
                    // Adjust range.
                    range.start += base_offset;
                    // We are only ever `SharedReadOnly` inside the frozen bits.
                    let perm = if frozen {
                        Permission::SharedReadOnly
                    } else {
                        Permission::SharedReadWrite
                    };
                    let protector = if frozen {
                        protector
                    } else {
                        // We do not protect inside UnsafeCell.
                        // This fixes https://github.com/rust-lang/rust/issues/55005.
                        None
                    };
                    let item = Item { perm, tag: new_tag, protector };
                    let mut global = this.machine.stacked_borrows.as_ref().unwrap().borrow_mut();
                    stacked_borrows.for_each(range, |offset, stack, history, exposed_tags| {
                        stack.grant(
                            orig_tag,
                            item,
                            (alloc_id, range, offset),
                            &mut global,
                            current_span,
                            history,
                            exposed_tags,
                        )
                    })
                })?;
                return Ok(Some(alloc_id));
            }
        };
        // Here we can avoid `borrow()` calls because we have mutable references.
        // Note that this asserts that the allocation is mutable -- but since we are creating a
        // mutable pointer, that seems reasonable.
        let (alloc_extra, machine) = this.get_alloc_extra_mut(alloc_id)?;
        let mut stacked_borrows = alloc_extra
            .stacked_borrows
            .as_mut()
            .expect("we should have Stacked Borrows data")
            .borrow_mut();
        let item = Item { perm, tag: new_tag, protector };
        let range = alloc_range(base_offset, size);
        let mut global = machine.stacked_borrows.as_ref().unwrap().borrow_mut();
        let current_span = &mut machine.current_span(); // `get_alloc_extra_mut` invalidated our old `current_span`
        stacked_borrows.for_each(range, |offset, stack, history, exposed_tags| {
            stack.grant(
                orig_tag,
                item,
                (alloc_id, range, offset),
                &mut global,
                current_span,
                history,
                exposed_tags,
            )
        })?;

        Ok(Some(alloc_id))
    }

    /// Retags an indidual pointer, returning the retagged version.
    /// `mutbl` can be `None` to make this a raw pointer.
    fn retag_reference(
        &mut self,
        val: &ImmTy<'tcx, Tag>,
        kind: RefKind,
        protect: bool,
    ) -> InterpResult<'tcx, ImmTy<'tcx, Tag>> {
        let this = self.eval_context_mut();
        // We want a place for where the ptr *points to*, so we get one.
        let place = this.ref_to_mplace(val)?;
        let size = this.size_and_align_of_mplace(&place)?.map(|(size, _)| size);
        // FIXME: If we cannot determine the size (because the unsized tail is an `extern type`),
        // bail out -- we cannot reasonably figure out which memory range to reborrow.
        // See https://github.com/rust-lang/unsafe-code-guidelines/issues/276.
        let size = match size {
            Some(size) => size,
            None => return Ok(*val),
        };

        // Compute new borrow.
        let new_tag = this.machine.stacked_borrows.as_mut().unwrap().get_mut().new_ptr();

        // Reborrow.
        let alloc_id = this.reborrow(&place, size, kind, new_tag, protect)?;

        // Adjust pointer.
        let new_place = place.map_provenance(|p| {
            p.map(|prov| {
                match alloc_id {
                    Some(alloc_id) => {
                        // If `reborrow` could figure out the AllocId of this ptr, hard-code it into the new one.
                        // Even if we started out with a wildcard, this newly retagged pointer is tied to that allocation.
                        Tag::Concrete { alloc_id, sb: new_tag }
                    }
                    None => {
                        // Looks like this has to stay a wildcard pointer.
                        assert!(matches!(prov, Tag::Wildcard));
                        Tag::Wildcard
                    }
                }
            })
        });

        // Return new pointer.
        Ok(ImmTy::from_immediate(new_place.to_ref(this), val.layout))
    }
}

impl<'mir, 'tcx: 'mir> EvalContextExt<'mir, 'tcx> for crate::MiriEvalContext<'mir, 'tcx> {}
pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriEvalContextExt<'mir, 'tcx> {
    fn retag(&mut self, kind: RetagKind, place: &PlaceTy<'tcx, Tag>) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        // Determine mutability and whether to add a protector.
        // Cannot use `builtin_deref` because that reports *immutable* for `Box`,
        // making it useless.
        fn qualify(ty: ty::Ty<'_>, kind: RetagKind) -> Option<(RefKind, bool)> {
            match ty.kind() {
                // References are simple.
                ty::Ref(_, _, Mutability::Mut) =>
                    Some((
                        RefKind::Unique { two_phase: kind == RetagKind::TwoPhase },
                        kind == RetagKind::FnEntry,
                    )),
                ty::Ref(_, _, Mutability::Not) =>
                    Some((RefKind::Shared, kind == RetagKind::FnEntry)),
                // Raw pointers need to be enabled.
                ty::RawPtr(tym) if kind == RetagKind::Raw =>
                    Some((RefKind::Raw { mutable: tym.mutbl == Mutability::Mut }, false)),
                // Boxes do not get a protector: protectors reflect that references outlive the call
                // they were passed in to; that's just not the case for boxes.
                ty::Adt(..) if ty.is_box() => Some((RefKind::Unique { two_phase: false }, false)),
                _ => None,
            }
        }

        // We need a visitor to visit all references. However, that requires
        // a `MPlaceTy` (or `OpTy), so we have a fast path for reference types that
        // avoids allocating.

        if let Some((mutbl, protector)) = qualify(place.layout.ty, kind) {
            // Fast path.
            let val = this.read_immediate(&this.place_to_op(place)?)?;
            let val = this.retag_reference(&val, mutbl, protector)?;
            this.write_immediate(*val, place)?;
            return Ok(());
        }

        // If we don't want to recurse, we are already done.
        if !this.machine.stacked_borrows.as_mut().unwrap().get_mut().retag_fields {
            return Ok(());
        }

        // Skip some types that have no further structure we might care about.
        if matches!(
            place.layout.ty.kind(),
            ty::RawPtr(..)
                | ty::Ref(..)
                | ty::Int(..)
                | ty::Uint(..)
                | ty::Float(..)
                | ty::Bool
                | ty::Char
        ) {
            return Ok(());
        }
        // Now go visit this thing.
        let place = this.force_allocation(place)?;

        let mut visitor = RetagVisitor { ecx: this, kind };
        return visitor.visit_value(&place);

        // The actual visitor.
        struct RetagVisitor<'ecx, 'mir, 'tcx> {
            ecx: &'ecx mut MiriEvalContext<'mir, 'tcx>,
            kind: RetagKind,
        }
        impl<'ecx, 'mir, 'tcx> MutValueVisitor<'mir, 'tcx, Evaluator<'mir, 'tcx>>
            for RetagVisitor<'ecx, 'mir, 'tcx>
        {
            type V = MPlaceTy<'tcx, Tag>;

            #[inline(always)]
            fn ecx(&mut self) -> &mut MiriEvalContext<'mir, 'tcx> {
                self.ecx
            }

            fn visit_value(&mut self, place: &MPlaceTy<'tcx, Tag>) -> InterpResult<'tcx> {
                if let Some((mutbl, protector)) = qualify(place.layout.ty, self.kind) {
                    let val = self.ecx.read_immediate(&place.into())?;
                    let val = self.ecx.retag_reference(&val, mutbl, protector)?;
                    self.ecx.write_immediate(*val, &place.into())?;
                } else {
                    // Maybe we need to go deeper.
                    self.walk_value(place)?;
                }
                Ok(())
            }
        }
    }

    /// After a stack frame got pushed, retag the return place so that we are sure
    /// it does not alias with anything.
    ///
    /// This is a HACK because there is nothing in MIR that would make the retag
    /// explicit. Also see <https://github.com/rust-lang/rust/issues/71117>.
    fn retag_return_place(&mut self) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        let return_place = this.frame_mut().return_place;
        if return_place.layout.is_zst() {
            // There may not be any memory here, nothing to do.
            return Ok(());
        }
        // We need this to be in-memory to use tagged pointers.
        let return_place = this.force_allocation(&return_place)?;

        // We have to turn the place into a pointer to use the existing code.
        // (The pointer type does not matter, so we use a raw pointer.)
        let ptr_layout = this.layout_of(this.tcx.mk_mut_ptr(return_place.layout.ty))?;
        let val = ImmTy::from_immediate(return_place.to_ref(this), ptr_layout);
        // Reborrow it.
        let val = this.retag_reference(
            &val,
            RefKind::Unique { two_phase: false },
            /*protector*/ true,
        )?;
        // And use reborrowed pointer for return place.
        let return_place = this.ref_to_mplace(&val)?;
        this.frame_mut().return_place = return_place.into();

        Ok(())
    }

    /// Mark the given tag as exposed. It was found on a pointer with the given AllocId.
    fn expose_tag(&mut self, alloc_id: AllocId, tag: SbTag) {
        let this = self.eval_context_mut();

        // Function pointers and dead objects don't have an alloc_extra so we ignore them.
        // This is okay because accessing them is UB anyway, no need for any Stacked Borrows checks.
        // NOT using `get_alloc_extra_mut` since this might be a read-only allocation!
        let (_size, _align, kind) = this.get_alloc_info(alloc_id);
        match kind {
            AllocKind::LiveData => {
                // This should have alloc_extra data.
                let alloc_extra = this.get_alloc_extra(alloc_id).unwrap();
                trace!("Stacked Borrows tag {tag:?} exposed in {alloc_id}");
                alloc_extra.stacked_borrows.as_ref().unwrap().borrow_mut().exposed_tags.insert(tag);
            }
            AllocKind::Function | AllocKind::Dead => {
                // No stacked borrows on these allocations.
            }
        }
    }
}
