//! Implements "Stacked Borrows".  See <https://github.com/rust-lang/unsafe-code-guidelines/blob/master/wip/stacked-borrows.md>
//! for further information.

use log::trace;
use std::cell::RefCell;
use std::cmp;
use std::fmt;
use std::fmt::Write;
use std::num::NonZeroU64;

use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_hir::Mutability;
use rustc_middle::mir::RetagKind;
use rustc_middle::ty::{
    self,
    layout::{HasParamEnv, LayoutOf},
};
use rustc_target::abi::Abi;
use rustc_target::abi::Size;
use smallvec::SmallVec;

use crate::*;

pub mod diagnostics;
use diagnostics::{AllocHistory, DiagnosticCx, DiagnosticCxBuilder, RetagCause, TagHistory};

mod item;
pub use item::{Item, Permission};
mod stack;
pub use stack::Stack;

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
    #[allow(clippy::should_implement_trait)]
    pub fn default() -> Self {
        Self::new(1).unwrap()
    }
}

impl fmt::Debug for SbTag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<{}>", self.0)
    }
}

#[derive(Debug)]
pub struct FrameExtra {
    /// The ID of the call this frame corresponds to.
    call_id: CallId,

    /// If this frame is protecting any tags, they are listed here. We use this list to do
    /// incremental updates of the global list of protected tags stored in the
    /// `stacked_borrows::GlobalState` upon function return, and if we attempt to pop a protected
    /// tag, to identify which call is responsible for protecting the tag.
    /// See `Stack::item_invalidated` for more explanation.
    ///
    /// This will contain one tag per reference passed to the function, so
    /// a size of 2 is enough for the vast majority of functions.
    protected_tags: SmallVec<[SbTag; 2]>,
}

impl VisitTags for FrameExtra {
    fn visit_tags(&self, _visit: &mut dyn FnMut(SbTag)) {
        // `protected_tags` are fine to GC.
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
    /// Whether this memory has been modified since the last time the tag GC ran
    modified_since_last_gc: bool,
}

/// The flavor of the protector.
#[derive(Copy, Clone, Debug)]
enum ProtectorKind {
    /// Protected against aliasing violations from other pointers.
    ///
    /// Items protected like this cause UB when they are invalidated, *but* the pointer itself may
    /// still be used to issue a deallocation.
    ///
    /// This is required for LLVM IR pointers that are `noalias` but *not* `dereferenceable`.
    WeakProtector,

    /// Protected against any kind of invalidation.
    ///
    /// Items protected like this cause UB when they are invalidated or the memory is deallocated.
    /// This is strictly stronger protection than `WeakProtector`.
    ///
    /// This is required for LLVM IR pointers that are `dereferenceable` (and also allows `noalias`).
    StrongProtector,
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
    /// All currently protected tags, and the status of their protection.
    /// An item is protected if its tag is in this set, *and* it has the "protected" bit set.
    /// We add tags to this when they are created with a protector in `reborrow`, and
    /// we remove tags from this when the call which is protecting them returns, in
    /// `GlobalStateInner::end_call`. See `Stack::item_invalidated` for more details.
    protected_tags: FxHashMap<SbTag, ProtectorKind>,
    /// The pointer ids to trace
    tracked_pointer_tags: FxHashSet<SbTag>,
    /// The call ids to trace
    tracked_call_ids: FxHashSet<CallId>,
    /// Whether to recurse into datatypes when searching for pointers to retag.
    retag_fields: RetagFields,
}

#[derive(Copy, Clone, Debug)]
pub enum RetagFields {
    /// Don't retag any fields.
    No,
    /// Retag all fields.
    Yes,
    /// Only retag fields of types with Scalar and ScalarPair layout,
    /// to match the LLVM `noalias` we generate.
    OnlyScalar,
}

impl VisitTags for GlobalStateInner {
    fn visit_tags(&self, _visit: &mut dyn FnMut(SbTag)) {
        // The only candidate is base_ptr_tags, and that does not need visiting since we don't ever
        // GC the bottommost tag.
    }
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
            RefKind::Unique { two_phase: false } => write!(f, "unique reference"),
            RefKind::Unique { two_phase: true } => write!(f, "unique reference (two-phase)"),
            RefKind::Shared => write!(f, "shared reference"),
            RefKind::Raw { mutable: true } => write!(f, "raw (mutable) pointer"),
            RefKind::Raw { mutable: false } => write!(f, "raw (constant) pointer"),
        }
    }
}

/// Utilities for initialization and ID generation
impl GlobalStateInner {
    pub fn new(
        tracked_pointer_tags: FxHashSet<SbTag>,
        tracked_call_ids: FxHashSet<CallId>,
        retag_fields: RetagFields,
    ) -> Self {
        GlobalStateInner {
            next_ptr_tag: SbTag(NonZeroU64::new(1).unwrap()),
            base_ptr_tags: FxHashMap::default(),
            next_call_id: NonZeroU64::new(1).unwrap(),
            protected_tags: FxHashMap::default(),
            tracked_pointer_tags,
            tracked_call_ids,
            retag_fields,
        }
    }

    /// Generates a new pointer tag. Remember to also check track_pointer_tags and log its creation!
    fn new_ptr(&mut self) -> SbTag {
        let id = self.next_ptr_tag;
        self.next_ptr_tag = SbTag(NonZeroU64::new(id.0.get() + 1).unwrap());
        id
    }

    pub fn new_frame(&mut self, machine: &MiriMachine<'_, '_>) -> FrameExtra {
        let call_id = self.next_call_id;
        trace!("new_frame: Assigning call ID {}", call_id);
        if self.tracked_call_ids.contains(&call_id) {
            machine.emit_diagnostic(NonHaltingDiagnostic::CreatedCallId(call_id));
        }
        self.next_call_id = NonZeroU64::new(call_id.get() + 1).unwrap();
        FrameExtra { call_id, protected_tags: SmallVec::new() }
    }

    pub fn end_call(&mut self, frame: &machine::FrameData<'_>) {
        for tag in &frame
            .stacked_borrows
            .as_ref()
            .expect("we should have Stacked Borrows data")
            .protected_tags
        {
            self.protected_tags.remove(tag);
        }
    }

    pub fn base_ptr_tag(&mut self, id: AllocId, machine: &MiriMachine<'_, '_>) -> SbTag {
        self.base_ptr_tags.get(&id).copied().unwrap_or_else(|| {
            let tag = self.new_ptr();
            if self.tracked_pointer_tags.contains(&tag) {
                machine.emit_diagnostic(NonHaltingDiagnostic::CreatedPointerTag(tag.0, None, None));
            }
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
/// U2: If the top is `Uniq`, accesses must be through that `Uniq` or remove it.
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

/// Determines whether an item was invalidated by a conflicting access, or by deallocation.
#[derive(Copy, Clone, Debug)]
enum ItemInvalidationCause {
    Conflict,
    Dealloc,
}

/// Core per-location operations: access, dealloc, reborrow.
impl<'tcx> Stack {
    /// Find the first write-incompatible item above the given one --
    /// i.e, find the height to which the stack will be truncated when writing to `granting`.
    fn find_first_write_incompatible(&self, granting: usize) -> usize {
        let perm = self.get(granting).unwrap().perm();
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
                    if item.perm() == Permission::SharedReadWrite {
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
    fn item_invalidated(
        item: &Item,
        global: &GlobalStateInner,
        dcx: &mut DiagnosticCx<'_, '_, '_, 'tcx>,
        cause: ItemInvalidationCause,
    ) -> InterpResult<'tcx> {
        if !global.tracked_pointer_tags.is_empty() {
            dcx.check_tracked_tag_popped(item, global);
        }

        if !item.protected() {
            return Ok(());
        }

        // We store tags twice, once in global.protected_tags and once in each call frame.
        // We do this because consulting a single global set in this function is faster
        // than attempting to search all call frames in the program for the `FrameExtra`
        // (if any) which is protecting the popped tag.
        //
        // This duplication trades off making `end_call` slower to make this function faster. This
        // trade-off is profitable in practice for a combination of two reasons.
        // 1. A single protected tag can (and does in some programs) protect thousands of `Item`s.
        //    Therefore, adding overhead in function call/return is profitable even if it only
        //    saves a little work in this function.
        // 2. Most frames protect only one or two tags. So this duplicative global turns a search
        //    which ends up about linear in the number of protected tags in the program into a
        //    constant time check (and a slow linear, because the tags in the frames aren't contiguous).
        if let Some(&protector_kind) = global.protected_tags.get(&item.tag()) {
            // The only way this is okay is if the protector is weak and we are deallocating with
            // the right pointer.
            let allowed = matches!(cause, ItemInvalidationCause::Dealloc)
                && matches!(protector_kind, ProtectorKind::WeakProtector);
            if !allowed {
                return Err(dcx.protector_error(item, protector_kind).into());
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
        tag: ProvenanceExtra,
        global: &GlobalStateInner,
        dcx: &mut DiagnosticCx<'_, '_, '_, 'tcx>,
        exposed_tags: &FxHashSet<SbTag>,
    ) -> InterpResult<'tcx> {
        // Two main steps: Find granting item, remove incompatible items above.

        // Step 1: Find granting item.
        let granting_idx =
            self.find_granting(access, tag, exposed_tags).map_err(|()| dcx.access_error(self))?;

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
                Stack::item_invalidated(&item, global, dcx, ItemInvalidationCause::Conflict)?;
                dcx.log_invalidation(item.tag());
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
                Stack::item_invalidated(&item, global, dcx, ItemInvalidationCause::Conflict)?;
                dcx.log_invalidation(item.tag());
                Ok(())
            })?;
        }

        // If this was an approximate action, we now collapse everything into an unknown.
        if granting_idx.is_none() || matches!(tag, ProvenanceExtra::Wildcard) {
            // Compute the upper bound of the items that remain.
            // (This is why we did all the work above: to reduce the items we have to consider here.)
            let mut max = NonZeroU64::new(1).unwrap();
            for i in 0..self.len() {
                let item = self.get(i).unwrap();
                // Skip disabled items, they cannot be matched anyway.
                if !matches!(item.perm(), Permission::Disabled) {
                    // We are looking for a strict upper bound, so add 1 to this tag.
                    max = cmp::max(item.tag().0.checked_add(1).unwrap(), max);
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
        tag: ProvenanceExtra,
        global: &GlobalStateInner,
        dcx: &mut DiagnosticCx<'_, '_, '_, 'tcx>,
        exposed_tags: &FxHashSet<SbTag>,
    ) -> InterpResult<'tcx> {
        // Step 1: Make a write access.
        // As part of this we do regular protector checking, i.e. even weakly protected items cause UB when popped.
        self.access(AccessKind::Write, tag, global, dcx, exposed_tags)?;

        // Step 2: Pretend we remove the remaining items, checking if any are strongly protected.
        for idx in (0..self.len()).rev() {
            let item = self.get(idx).unwrap();
            Stack::item_invalidated(&item, global, dcx, ItemInvalidationCause::Dealloc)?;
        }

        Ok(())
    }

    /// Derive a new pointer from one with the given tag.
    ///
    /// `access` indicates which kind of memory access this retag itself should correspond to.
    fn grant(
        &mut self,
        derived_from: ProvenanceExtra,
        new: Item,
        access: Option<AccessKind>,
        global: &GlobalStateInner,
        dcx: &mut DiagnosticCx<'_, '_, '_, 'tcx>,
        exposed_tags: &FxHashSet<SbTag>,
    ) -> InterpResult<'tcx> {
        dcx.start_grant(new.perm());

        // Compute where to put the new item.
        // Either way, we ensure that we insert the new item in a way such that between
        // `derived_from` and the new one, there are only items *compatible with* `derived_from`.
        let new_idx = if let Some(access) = access {
            // Simple case: We are just a regular memory access, and then push our thing on top,
            // like a regular stack.
            // This ensures F2b for `Unique`, by removing offending `SharedReadOnly`.
            self.access(access, derived_from, global, dcx, exposed_tags)?;

            // We insert "as far up as possible": We know only compatible items are remaining
            // on top of `derived_from`, and we want the new item at the top so that we
            // get the strongest possible guarantees.
            // This ensures U1 and F1.
            self.len()
        } else {
            // The tricky case: creating a new SRW permission without actually being an access.
            assert!(new.perm() == Permission::SharedReadWrite);

            // First we figure out which item grants our parent (`derived_from`) this kind of access.
            // We use that to determine where to put the new item.
            let granting_idx = self
                .find_granting(AccessKind::Write, derived_from, exposed_tags)
                .map_err(|()| dcx.grant_error(self))?;

            let (Some(granting_idx), ProvenanceExtra::Concrete(_)) = (granting_idx, derived_from) else {
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
        };

        // Put the new item there.
        trace!("reborrow: adding item {:?}", new);
        self.insert(new_idx, new);
        Ok(())
    }
}
// # Stacked Borrows Core End

/// Integration with the SbTag garbage collector
impl Stacks {
    pub fn remove_unreachable_tags(&mut self, live_tags: &FxHashSet<SbTag>) {
        if self.modified_since_last_gc {
            for stack in self.stacks.iter_mut_all() {
                if stack.len() > 64 {
                    stack.retain(live_tags);
                }
            }
            self.modified_since_last_gc = false;
        }
    }
}

impl VisitTags for Stacks {
    fn visit_tags(&self, visit: &mut dyn FnMut(SbTag)) {
        for tag in self.exposed_tags.iter().copied() {
            visit(tag);
        }
    }
}

/// Map per-stack operations to higher-level per-location-range operations.
impl<'tcx> Stacks {
    /// Creates a new stack with an initial tag. For diagnostic purposes, we also need to know
    /// the [`AllocId`] of the allocation this is associated with.
    fn new(
        size: Size,
        perm: Permission,
        tag: SbTag,
        id: AllocId,
        machine: &MiriMachine<'_, '_>,
    ) -> Self {
        let item = Item::new(tag, perm, false);
        let stack = Stack::new(item);

        Stacks {
            stacks: RangeMap::new(size, stack),
            history: AllocHistory::new(id, item, machine),
            exposed_tags: FxHashSet::default(),
            modified_since_last_gc: false,
        }
    }

    /// Call `f` on every stack in the range.
    fn for_each(
        &mut self,
        range: AllocRange,
        mut dcx_builder: DiagnosticCxBuilder<'_, '_, 'tcx>,
        mut f: impl FnMut(
            &mut Stack,
            &mut DiagnosticCx<'_, '_, '_, 'tcx>,
            &mut FxHashSet<SbTag>,
        ) -> InterpResult<'tcx>,
    ) -> InterpResult<'tcx> {
        self.modified_since_last_gc = true;
        for (offset, stack) in self.stacks.iter_mut(range.start, range.size) {
            let mut dcx = dcx_builder.build(&mut self.history, offset);
            f(stack, &mut dcx, &mut self.exposed_tags)?;
            dcx_builder = dcx.unbuild();
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
        machine: &MiriMachine<'_, '_>,
    ) -> Self {
        let mut extra = state.borrow_mut();
        let (base_tag, perm) = match kind {
            // New unique borrow. This tag is not accessible by the program,
            // so it will only ever be used when using the local directly (i.e.,
            // not through a pointer). That is, whenever we directly write to a local, this will pop
            // everything else off the stack, invalidating all previous pointers,
            // and in particular, *all* raw pointers.
            MemoryKind::Stack => (extra.base_ptr_tag(id, machine), Permission::Unique),
            // Everything else is shared by default.
            _ => (extra.base_ptr_tag(id, machine), Permission::SharedReadWrite),
        };
        Stacks::new(size, perm, base_tag, id, machine)
    }

    #[inline(always)]
    pub fn before_memory_read<'tcx, 'mir, 'ecx>(
        &mut self,
        alloc_id: AllocId,
        tag: ProvenanceExtra,
        range: AllocRange,
        machine: &'ecx MiriMachine<'mir, 'tcx>,
    ) -> InterpResult<'tcx>
    where
        'tcx: 'ecx,
    {
        trace!(
            "read access with tag {:?}: {:?}, size {}",
            tag,
            Pointer::new(alloc_id, range.start),
            range.size.bytes()
        );
        let dcx = DiagnosticCxBuilder::read(machine, tag, range);
        let state = machine.stacked_borrows.as_ref().unwrap().borrow();
        self.for_each(range, dcx, |stack, dcx, exposed_tags| {
            stack.access(AccessKind::Read, tag, &state, dcx, exposed_tags)
        })
    }

    #[inline(always)]
    pub fn before_memory_write<'tcx, 'mir, 'ecx>(
        &mut self,
        alloc_id: AllocId,
        tag: ProvenanceExtra,
        range: AllocRange,
        machine: &'ecx mut MiriMachine<'mir, 'tcx>,
    ) -> InterpResult<'tcx> {
        trace!(
            "write access with tag {:?}: {:?}, size {}",
            tag,
            Pointer::new(alloc_id, range.start),
            range.size.bytes()
        );
        let dcx = DiagnosticCxBuilder::write(machine, tag, range);
        let state = machine.stacked_borrows.as_ref().unwrap().borrow();
        self.for_each(range, dcx, |stack, dcx, exposed_tags| {
            stack.access(AccessKind::Write, tag, &state, dcx, exposed_tags)
        })
    }

    #[inline(always)]
    pub fn before_memory_deallocation<'tcx, 'mir, 'ecx>(
        &mut self,
        alloc_id: AllocId,
        tag: ProvenanceExtra,
        range: AllocRange,
        machine: &'ecx mut MiriMachine<'mir, 'tcx>,
    ) -> InterpResult<'tcx> {
        trace!("deallocation with tag {:?}: {:?}, size {}", tag, alloc_id, range.size.bytes());
        let dcx = DiagnosticCxBuilder::dealloc(machine, tag);
        let state = machine.stacked_borrows.as_ref().unwrap().borrow();
        self.for_each(range, dcx, |stack, dcx, exposed_tags| {
            stack.dealloc(tag, &state, dcx, exposed_tags)
        })?;
        Ok(())
    }
}

/// Retagging/reborrowing.  There is some policy in here, such as which permissions
/// to grant for which references, and when to add protectors.
impl<'mir: 'ecx, 'tcx: 'mir, 'ecx> EvalContextPrivExt<'mir, 'tcx, 'ecx>
    for crate::MiriInterpCx<'mir, 'tcx>
{
}
trait EvalContextPrivExt<'mir: 'ecx, 'tcx: 'mir, 'ecx>: crate::MiriInterpCxExt<'mir, 'tcx> {
    /// Returns the `AllocId` the reborrow was done in, if some actual borrow stack manipulation
    /// happened.
    fn reborrow(
        &mut self,
        place: &MPlaceTy<'tcx, Provenance>,
        size: Size,
        kind: RefKind,
        retag_cause: RetagCause, // What caused this retag, for diagnostics only
        new_tag: SbTag,
        protect: Option<ProtectorKind>,
    ) -> InterpResult<'tcx, Option<AllocId>> {
        let this = self.eval_context_mut();

        // It is crucial that this gets called on all code paths, to ensure we track tag creation.
        let log_creation = |this: &MiriInterpCx<'mir, 'tcx>,
                            loc: Option<(AllocId, Size, ProvenanceExtra)>| // alloc_id, base_offset, orig_tag
         -> InterpResult<'tcx> {
            let global = this.machine.stacked_borrows.as_ref().unwrap().borrow();
            let ty = place.layout.ty;
            if global.tracked_pointer_tags.contains(&new_tag) {
                let mut kind_str = format!("{kind}");
                match kind {
                    RefKind::Unique { two_phase: false }
                        if !ty.is_unpin(*this.tcx, this.param_env()) =>
                    {
                        write!(kind_str, " (!Unpin pointee type {ty})").unwrap()
                    },
                    RefKind::Shared
                        if !ty.is_freeze(*this.tcx, this.param_env()) =>
                    {
                        write!(kind_str, " (!Freeze pointee type {ty})").unwrap()
                    },
                    _ => write!(kind_str, " (pointee type {ty})").unwrap(),
                };
                this.emit_diagnostic(NonHaltingDiagnostic::CreatedPointerTag(
                    new_tag.0,
                    Some(kind_str),
                    loc.map(|(alloc_id, base_offset, orig_tag)| (alloc_id, alloc_range(base_offset, size), orig_tag)),
                ));
            }
            drop(global); // don't hold that reference any longer than we have to

            let Some((alloc_id, base_offset, orig_tag)) = loc else {
                return Ok(())
            };

            let (_size, _align, alloc_kind) = this.get_alloc_info(alloc_id);
            match alloc_kind {
                AllocKind::LiveData => {
                    // This should have alloc_extra data, but `get_alloc_extra` can still fail
                    // if converting this alloc_id from a global to a local one
                    // uncovers a non-supported `extern static`.
                    let extra = this.get_alloc_extra(alloc_id)?;
                    let mut stacked_borrows = extra
                        .stacked_borrows
                        .as_ref()
                        .expect("we should have Stacked Borrows data")
                        .borrow_mut();
                    // Note that we create a *second* `DiagnosticCxBuilder` below for the actual retag.
                    // FIXME: can this be done cleaner?
                    let dcx = DiagnosticCxBuilder::retag(
                        &this.machine,
                        retag_cause,
                        new_tag,
                        orig_tag,
                        alloc_range(base_offset, size),
                    );
                    let mut dcx = dcx.build(&mut stacked_borrows.history, base_offset);
                    dcx.log_creation();
                    if protect.is_some() {
                        dcx.log_protector();
                    }
                }
                AllocKind::Function | AllocKind::VTable | AllocKind::Dead => {
                    // No stacked borrows on these allocations.
                }
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
                log_creation(this, Some((alloc_id, base_offset, orig_tag)))?;
                return Ok(Some(alloc_id));
            }
            // This pointer doesn't come with an AllocId. :shrug:
            log_creation(this, None)?;
            return Ok(None);
        }

        let (alloc_id, base_offset, orig_tag) = this.ptr_get_alloc_id(place.ptr)?;
        log_creation(this, Some((alloc_id, base_offset, orig_tag)))?;

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

        trace!(
            "reborrow: {} reference {:?} derived from {:?} (pointee {}): {:?}, size {}",
            kind,
            new_tag,
            orig_tag,
            place.layout.ty,
            Pointer::new(alloc_id, base_offset),
            size.bytes()
        );

        if let Some(protect) = protect {
            // See comment in `Stack::item_invalidated` for why we store the tag twice.
            this.frame_mut().extra.stacked_borrows.as_mut().unwrap().protected_tags.push(new_tag);
            this.machine
                .stacked_borrows
                .as_mut()
                .unwrap()
                .get_mut()
                .protected_tags
                .insert(new_tag, protect);
        }

        // Update the stacks.
        // Make sure that raw pointers and mutable shared references are reborrowed "weak":
        // There could be existing unique pointers reborrowed from them that should remain valid!
        let (perm, access) = match kind {
            RefKind::Unique { two_phase } => {
                // Permission is Unique only if the type is `Unpin` and this is not twophase
                let perm = if !two_phase && place.layout.ty.is_unpin(*this.tcx, this.param_env()) {
                    Permission::Unique
                } else {
                    Permission::SharedReadWrite
                };
                // We do an access for all full borrows, even if `!Unpin`.
                let access = if !two_phase { Some(AccessKind::Write) } else { None };
                (perm, access)
            }
            RefKind::Raw { mutable: true } => {
                // Creating a raw ptr does not count as an access
                (Permission::SharedReadWrite, None)
            }
            RefKind::Shared | RefKind::Raw { mutable: false } => {
                // Shared references and *const are a whole different kind of game, the
                // permission is not uniform across the entire range!
                // We need a frozen-sensitive reborrow.
                // We have to use shared references to alloc/memory_extra here since
                // `visit_freeze_sensitive` needs to access the global state.
                let alloc_extra = this.get_alloc_extra(alloc_id)?;
                let mut stacked_borrows = alloc_extra
                    .stacked_borrows
                    .as_ref()
                    .expect("we should have Stacked Borrows data")
                    .borrow_mut();
                this.visit_freeze_sensitive(place, size, |mut range, frozen| {
                    // Adjust range.
                    range.start += base_offset;
                    // We are only ever `SharedReadOnly` inside the frozen bits.
                    let (perm, access) = if frozen {
                        (Permission::SharedReadOnly, Some(AccessKind::Read))
                    } else {
                        // Inside UnsafeCell, this does *not* count as an access, as there
                        // might actually be mutable references further up the stack that
                        // we have to keep alive.
                        (Permission::SharedReadWrite, None)
                    };
                    let protected = if frozen {
                        protect.is_some()
                    } else {
                        // We do not protect inside UnsafeCell.
                        // This fixes https://github.com/rust-lang/rust/issues/55005.
                        false
                    };
                    let item = Item::new(new_tag, perm, protected);
                    let global = this.machine.stacked_borrows.as_ref().unwrap().borrow();
                    let dcx = DiagnosticCxBuilder::retag(
                        &this.machine,
                        retag_cause,
                        new_tag,
                        orig_tag,
                        alloc_range(base_offset, size),
                    );
                    stacked_borrows.for_each(range, dcx, |stack, dcx, exposed_tags| {
                        stack.grant(orig_tag, item, access, &global, dcx, exposed_tags)
                    })?;
                    drop(global);
                    if let Some(access) = access {
                        assert_eq!(access, AccessKind::Read);
                        // Make sure the data race model also knows about this.
                        if let Some(data_race) = alloc_extra.data_race.as_ref() {
                            data_race.read(alloc_id, range, &this.machine)?;
                        }
                    }
                    Ok(())
                })?;
                return Ok(Some(alloc_id));
            }
        };

        // Here we can avoid `borrow()` calls because we have mutable references.
        // Note that this asserts that the allocation is mutable -- but since we are creating a
        // mutable pointer, that seems reasonable.
        let (alloc_extra, machine) = this.get_alloc_extra_mut(alloc_id)?;
        let stacked_borrows = alloc_extra
            .stacked_borrows
            .as_mut()
            .expect("we should have Stacked Borrows data")
            .get_mut();
        let item = Item::new(new_tag, perm, protect.is_some());
        let range = alloc_range(base_offset, size);
        let global = machine.stacked_borrows.as_ref().unwrap().borrow();
        let dcx = DiagnosticCxBuilder::retag(
            machine,
            retag_cause,
            new_tag,
            orig_tag,
            alloc_range(base_offset, size),
        );
        stacked_borrows.for_each(range, dcx, |stack, dcx, exposed_tags| {
            stack.grant(orig_tag, item, access, &global, dcx, exposed_tags)
        })?;
        drop(global);
        if let Some(access) = access {
            assert_eq!(access, AccessKind::Write);
            // Make sure the data race model also knows about this.
            if let Some(data_race) = alloc_extra.data_race.as_mut() {
                data_race.write(alloc_id, range, machine)?;
            }
        }

        Ok(Some(alloc_id))
    }

    /// Retags an indidual pointer, returning the retagged version.
    /// `mutbl` can be `None` to make this a raw pointer.
    fn retag_reference(
        &mut self,
        val: &ImmTy<'tcx, Provenance>,
        kind: RefKind,
        retag_cause: RetagCause, // What caused this retag, for diagnostics only
        protect: Option<ProtectorKind>,
    ) -> InterpResult<'tcx, ImmTy<'tcx, Provenance>> {
        let this = self.eval_context_mut();
        // We want a place for where the ptr *points to*, so we get one.
        let place = this.ref_to_mplace(val)?;
        let size = this.size_and_align_of_mplace(&place)?.map(|(size, _)| size);
        // FIXME: If we cannot determine the size (because the unsized tail is an `extern type`),
        // bail out -- we cannot reasonably figure out which memory range to reborrow.
        // See https://github.com/rust-lang/unsafe-code-guidelines/issues/276.
        let size = match size {
            Some(size) => size,
            None => return Ok(val.clone()),
        };

        // Compute new borrow.
        let new_tag = this.machine.stacked_borrows.as_mut().unwrap().get_mut().new_ptr();

        // Reborrow.
        let alloc_id = this.reborrow(&place, size, kind, retag_cause, new_tag, protect)?;

        // Adjust pointer.
        let new_place = place.map_provenance(|p| {
            p.map(|prov| {
                match alloc_id {
                    Some(alloc_id) => {
                        // If `reborrow` could figure out the AllocId of this ptr, hard-code it into the new one.
                        // Even if we started out with a wildcard, this newly retagged pointer is tied to that allocation.
                        Provenance::Concrete { alloc_id, sb: new_tag }
                    }
                    None => {
                        // Looks like this has to stay a wildcard pointer.
                        assert!(matches!(prov, Provenance::Wildcard));
                        Provenance::Wildcard
                    }
                }
            })
        });

        // Return new pointer.
        Ok(ImmTy::from_immediate(new_place.to_ref(this), val.layout))
    }
}

impl<'mir, 'tcx: 'mir> EvalContextExt<'mir, 'tcx> for crate::MiriInterpCx<'mir, 'tcx> {}
pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriInterpCxExt<'mir, 'tcx> {
    fn retag(&mut self, kind: RetagKind, place: &PlaceTy<'tcx, Provenance>) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        let retag_fields = this.machine.stacked_borrows.as_mut().unwrap().get_mut().retag_fields;
        let retag_cause = match kind {
            RetagKind::TwoPhase { .. } => RetagCause::TwoPhase,
            RetagKind::FnEntry => RetagCause::FnEntry,
            RetagKind::Raw | RetagKind::Default => RetagCause::Normal,
        };
        let mut visitor = RetagVisitor { ecx: this, kind, retag_cause, retag_fields };
        return visitor.visit_value(place);

        // The actual visitor.
        struct RetagVisitor<'ecx, 'mir, 'tcx> {
            ecx: &'ecx mut MiriInterpCx<'mir, 'tcx>,
            kind: RetagKind,
            retag_cause: RetagCause,
            retag_fields: RetagFields,
        }
        impl<'ecx, 'mir, 'tcx> RetagVisitor<'ecx, 'mir, 'tcx> {
            #[inline(always)] // yes this helps in our benchmarks
            fn retag_place(
                &mut self,
                place: &PlaceTy<'tcx, Provenance>,
                ref_kind: RefKind,
                retag_cause: RetagCause,
                protector: Option<ProtectorKind>,
            ) -> InterpResult<'tcx> {
                let val = self.ecx.read_immediate(&self.ecx.place_to_op(place)?)?;
                let val = self.ecx.retag_reference(&val, ref_kind, retag_cause, protector)?;
                self.ecx.write_immediate(*val, place)?;
                Ok(())
            }
        }
        impl<'ecx, 'mir, 'tcx> MutValueVisitor<'mir, 'tcx, MiriMachine<'mir, 'tcx>>
            for RetagVisitor<'ecx, 'mir, 'tcx>
        {
            type V = PlaceTy<'tcx, Provenance>;

            #[inline(always)]
            fn ecx(&mut self) -> &mut MiriInterpCx<'mir, 'tcx> {
                self.ecx
            }

            fn visit_box(&mut self, place: &PlaceTy<'tcx, Provenance>) -> InterpResult<'tcx> {
                // Boxes get a weak protectors, since they may be deallocated.
                self.retag_place(
                    place,
                    RefKind::Unique { two_phase: false },
                    self.retag_cause,
                    /*protector*/
                    (self.kind == RetagKind::FnEntry).then_some(ProtectorKind::WeakProtector),
                )
            }

            fn visit_value(&mut self, place: &PlaceTy<'tcx, Provenance>) -> InterpResult<'tcx> {
                // If this place is smaller than a pointer, we know that it can't contain any
                // pointers we need to retag, so we can stop recursion early.
                // This optimization is crucial for ZSTs, because they can contain way more fields
                // than we can ever visit.
                if place.layout.is_sized() && place.layout.size < self.ecx.pointer_size() {
                    return Ok(());
                }

                // Check the type of this value to see what to do with it (retag, or recurse).
                match place.layout.ty.kind() {
                    ty::Ref(_, _, mutbl) => {
                        let ref_kind = match mutbl {
                            Mutability::Mut =>
                                RefKind::Unique { two_phase: self.kind == RetagKind::TwoPhase },
                            Mutability::Not => RefKind::Shared,
                        };
                        self.retag_place(
                            place,
                            ref_kind,
                            self.retag_cause,
                            /*protector*/
                            (self.kind == RetagKind::FnEntry)
                                .then_some(ProtectorKind::StrongProtector),
                        )?;
                    }
                    ty::RawPtr(tym) => {
                        // We definitely do *not* want to recurse into raw pointers -- wide raw
                        // pointers have fields, and for dyn Trait pointees those can have reference
                        // type!
                        if self.kind == RetagKind::Raw {
                            // Raw pointers need to be enabled.
                            self.retag_place(
                                place,
                                RefKind::Raw { mutable: tym.mutbl == Mutability::Mut },
                                self.retag_cause,
                                /*protector*/ None,
                            )?;
                        }
                    }
                    _ if place.layout.ty.ty_adt_def().is_some_and(|adt| adt.is_box()) => {
                        // Recurse for boxes, they require some tricky handling and will end up in `visit_box` above.
                        // (Yes this means we technically also recursively retag the allocator itself
                        // even if field retagging is not enabled. *shrug*)
                        self.walk_value(place)?;
                    }
                    _ => {
                        // Not a reference/pointer/box. Only recurse if configured appropriately.
                        let recurse = match self.retag_fields {
                            RetagFields::No => false,
                            RetagFields::Yes => true,
                            RetagFields::OnlyScalar => {
                                // Matching `ArgAbi::new` at the time of writing, only fields of
                                // `Scalar` and `ScalarPair` ABI are considered.
                                matches!(place.layout.abi, Abi::Scalar(..) | Abi::ScalarPair(..))
                            }
                        };
                        if recurse {
                            self.walk_value(place)?;
                        }
                    }
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
        let return_place = &this.frame().return_place;
        if return_place.layout.is_zst() {
            // There may not be any memory here, nothing to do.
            return Ok(());
        }
        // We need this to be in-memory to use tagged pointers.
        let return_place = this.force_allocation(&return_place.clone())?;

        // We have to turn the place into a pointer to use the existing code.
        // (The pointer type does not matter, so we use a raw pointer.)
        let ptr_layout = this.layout_of(this.tcx.mk_mut_ptr(return_place.layout.ty))?;
        let val = ImmTy::from_immediate(return_place.to_ref(this), ptr_layout);
        // Reborrow it. With protection! That is part of the point.
        let val = this.retag_reference(
            &val,
            RefKind::Unique { two_phase: false },
            RetagCause::FnReturn,
            /*protector*/ Some(ProtectorKind::StrongProtector),
        )?;
        // And use reborrowed pointer for return place.
        let return_place = this.ref_to_mplace(&val)?;
        this.frame_mut().return_place = return_place.into();

        Ok(())
    }

    /// Mark the given tag as exposed. It was found on a pointer with the given AllocId.
    fn expose_tag(&mut self, alloc_id: AllocId, tag: SbTag) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        // Function pointers and dead objects don't have an alloc_extra so we ignore them.
        // This is okay because accessing them is UB anyway, no need for any Stacked Borrows checks.
        // NOT using `get_alloc_extra_mut` since this might be a read-only allocation!
        let (_size, _align, kind) = this.get_alloc_info(alloc_id);
        match kind {
            AllocKind::LiveData => {
                // This should have alloc_extra data, but `get_alloc_extra` can still fail
                // if converting this alloc_id from a global to a local one
                // uncovers a non-supported `extern static`.
                let alloc_extra = this.get_alloc_extra(alloc_id)?;
                trace!("Stacked Borrows tag {tag:?} exposed in {alloc_id:?}");
                alloc_extra.stacked_borrows.as_ref().unwrap().borrow_mut().exposed_tags.insert(tag);
            }
            AllocKind::Function | AllocKind::VTable | AllocKind::Dead => {
                // No stacked borrows on these allocations.
            }
        }
        Ok(())
    }

    fn print_stacks(&mut self, alloc_id: AllocId) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        let alloc_extra = this.get_alloc_extra(alloc_id)?;
        let stacks = alloc_extra.stacked_borrows.as_ref().unwrap().borrow();
        for (range, stack) in stacks.stacks.iter_all() {
            print!("{range:?}: [");
            if let Some(bottom) = stack.unknown_bottom() {
                print!(" unknown-bottom(..{bottom:?})");
            }
            for i in 0..stack.len() {
                let item = stack.get(i).unwrap();
                print!(" {:?}{:?}", item.perm(), item.tag());
            }
            println!(" ]");
        }
        Ok(())
    }
}
