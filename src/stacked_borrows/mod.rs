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
    Ty,
};
use rustc_span::DUMMY_SP;
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
    /// See `Stack::item_popped` for more explanation.
    ///
    /// This will contain one tag per reference passed to the function, so
    /// a size of 2 is enough for the vast majority of functions.
    protected_tags: SmallVec<[SbTag; 2]>,
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
    /// All currently protected tags.
    /// An item is protected if its tag is in this set, *and* it has the "protected" bit set.
    /// We add tags to this when they are created with a protector in `reborrow`, and
    /// we remove tags from this when the call which is protecting them returns, in
    /// `GlobalStateInner::end_call`. See `Stack::item_popped` for more details.
    protected_tags: FxHashSet<SbTag>,
    /// The pointer ids to trace
    tracked_pointer_tags: FxHashSet<SbTag>,
    /// The call ids to trace
    tracked_call_ids: FxHashSet<CallId>,
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
        retag_fields: bool,
    ) -> Self {
        GlobalStateInner {
            next_ptr_tag: SbTag(NonZeroU64::new(1).unwrap()),
            base_ptr_tags: FxHashMap::default(),
            next_call_id: NonZeroU64::new(1).unwrap(),
            protected_tags: FxHashSet::default(),
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
                machine.emit_diagnostic(NonHaltingDiagnostic::CreatedPointerTag(tag.0, None));
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
    fn item_popped(
        item: &Item,
        global: &GlobalStateInner,
        dcx: &mut DiagnosticCx<'_, '_, '_, '_, 'tcx>,
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
        if global.protected_tags.contains(&item.tag()) {
            return Err(dcx.protector_error(item).into());
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
        global: &mut GlobalStateInner,
        dcx: &mut DiagnosticCx<'_, '_, '_, '_, 'tcx>,
        exposed_tags: &FxHashSet<SbTag>,
    ) -> InterpResult<'tcx> {
        // Two main steps: Find granting item, remove incompatible items above.

        // Step 1: Find granting item.
        let granting_idx =
            self.find_granting(access, tag, exposed_tags).map_err(|_| dcx.access_error(self))?;

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
                Stack::item_popped(&item, global, dcx)?;
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
                Stack::item_popped(&item, global, dcx)?;
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
        dcx: &mut DiagnosticCx<'_, '_, '_, '_, 'tcx>,
        exposed_tags: &FxHashSet<SbTag>,
    ) -> InterpResult<'tcx> {
        // Step 1: Make sure there is a granting item.
        self.find_granting(AccessKind::Write, tag, exposed_tags)
            .map_err(|_| dcx.dealloc_error())?;

        // Step 2: Consider all items removed. This checks for protectors.
        for idx in (0..self.len()).rev() {
            let item = self.get(idx).unwrap();
            Stack::item_popped(&item, global, dcx)?;
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
        derived_from: ProvenanceExtra,
        new: Item,
        global: &mut GlobalStateInner,
        dcx: &mut DiagnosticCx<'_, '_, '_, '_, 'tcx>,
        exposed_tags: &FxHashSet<SbTag>,
    ) -> InterpResult<'tcx> {
        dcx.start_grant(new.perm());

        // Figure out which access `perm` corresponds to.
        let access =
            if new.perm().grants(AccessKind::Write) { AccessKind::Write } else { AccessKind::Read };

        // Now we figure out which item grants our parent (`derived_from`) this kind of access.
        // We use that to determine where to put the new item.
        let granting_idx = self
            .find_granting(access, derived_from, exposed_tags)
            .map_err(|_| dcx.grant_error(new.perm(), self))?;

        // Compute where to put the new item.
        // Either way, we ensure that we insert the new item in a way such that between
        // `derived_from` and the new one, there are only items *compatible with* `derived_from`.
        let new_idx = if new.perm() == Permission::SharedReadWrite {
            assert!(
                access == AccessKind::Write,
                "this case only makes sense for stack-like accesses"
            );

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
        } else {
            // A "safe" reborrow for a pointer that actually expects some aliasing guarantees.
            // Here, creating a reference actually counts as an access.
            // This ensures F2b for `Unique`, by removing offending `SharedReadOnly`.
            self.access(access, derived_from, global, dcx, exposed_tags)?;

            // We insert "as far up as possible": We know only compatible items are remaining
            // on top of `derived_from`, and we want the new item at the top so that we
            // get the strongest possible guarantees.
            // This ensures U1 and F1.
            self.len()
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

/// Map per-stack operations to higher-level per-location-range operations.
impl<'tcx> Stacks {
    /// Creates a new stack with an initial tag. For diagnostic purposes, we also need to know
    /// the [`AllocId`] of the allocation this is associated with.
    fn new(
        size: Size,
        perm: Permission,
        tag: SbTag,
        id: AllocId,
        current_span: &mut CurrentSpan<'_, '_, '_>,
    ) -> Self {
        let item = Item::new(tag, perm, false);
        let stack = Stack::new(item);

        Stacks {
            stacks: RangeMap::new(size, stack),
            history: AllocHistory::new(id, item, current_span),
            exposed_tags: FxHashSet::default(),
            modified_since_last_gc: false,
        }
    }

    /// Call `f` on every stack in the range.
    fn for_each(
        &mut self,
        range: AllocRange,
        mut dcx_builder: DiagnosticCxBuilder<'_, '_, '_, 'tcx>,
        mut f: impl FnMut(
            &mut Stack,
            &mut DiagnosticCx<'_, '_, '_, '_, 'tcx>,
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
        mut current_span: CurrentSpan<'_, '_, '_>,
    ) -> Self {
        let mut extra = state.borrow_mut();
        let (base_tag, perm) = match kind {
            // New unique borrow. This tag is not accessible by the program,
            // so it will only ever be used when using the local directly (i.e.,
            // not through a pointer). That is, whenever we directly write to a local, this will pop
            // everything else off the stack, invalidating all previous pointers,
            // and in particular, *all* raw pointers.
            MemoryKind::Stack =>
                (extra.base_ptr_tag(id, current_span.machine()), Permission::Unique),
            // Everything else is shared by default.
            _ => (extra.base_ptr_tag(id, current_span.machine()), Permission::SharedReadWrite),
        };
        Stacks::new(size, perm, base_tag, id, &mut current_span)
    }

    #[inline(always)]
    pub fn before_memory_read<'tcx, 'mir, 'ecx>(
        &mut self,
        alloc_id: AllocId,
        tag: ProvenanceExtra,
        range: AllocRange,
        state: &GlobalState,
        mut current_span: CurrentSpan<'ecx, 'mir, 'tcx>,
        threads: &'ecx ThreadManager<'mir, 'tcx>,
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
        let dcx = DiagnosticCxBuilder::read(&mut current_span, threads, tag, range);
        let mut state = state.borrow_mut();
        self.for_each(range, dcx, |stack, dcx, exposed_tags| {
            stack.access(AccessKind::Read, tag, &mut state, dcx, exposed_tags)
        })
    }

    #[inline(always)]
    pub fn before_memory_write<'tcx, 'mir, 'ecx>(
        &mut self,
        alloc_id: AllocId,
        tag: ProvenanceExtra,
        range: AllocRange,
        state: &GlobalState,
        mut current_span: CurrentSpan<'ecx, 'mir, 'tcx>,
        threads: &'ecx ThreadManager<'mir, 'tcx>,
    ) -> InterpResult<'tcx> {
        trace!(
            "write access with tag {:?}: {:?}, size {}",
            tag,
            Pointer::new(alloc_id, range.start),
            range.size.bytes()
        );
        let dcx = DiagnosticCxBuilder::write(&mut current_span, threads, tag, range);
        let mut state = state.borrow_mut();
        self.for_each(range, dcx, |stack, dcx, exposed_tags| {
            stack.access(AccessKind::Write, tag, &mut state, dcx, exposed_tags)
        })
    }

    #[inline(always)]
    pub fn before_memory_deallocation<'tcx, 'mir, 'ecx>(
        &mut self,
        alloc_id: AllocId,
        tag: ProvenanceExtra,
        range: AllocRange,
        state: &GlobalState,
        mut current_span: CurrentSpan<'ecx, 'mir, 'tcx>,
        threads: &'ecx ThreadManager<'mir, 'tcx>,
    ) -> InterpResult<'tcx> {
        trace!("deallocation with tag {:?}: {:?}, size {}", tag, alloc_id, range.size.bytes());
        let dcx = DiagnosticCxBuilder::dealloc(&mut current_span, threads, tag);
        let state = state.borrow();
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
        protect: bool,
    ) -> InterpResult<'tcx, Option<AllocId>> {
        let this = self.eval_context_mut();

        // It is crucial that this gets called on all code paths, to ensure we track tag creation.
        let log_creation = |this: &MiriInterpCx<'mir, 'tcx>,
                            loc: Option<(AllocId, Size, ProvenanceExtra)>| // alloc_id, base_offset, orig_tag
         -> InterpResult<'tcx> {
            let global = this.machine.stacked_borrows.as_ref().unwrap().borrow();
            if global.tracked_pointer_tags.contains(&new_tag) {
                this.emit_diagnostic(NonHaltingDiagnostic::CreatedPointerTag(
                    new_tag.0,
                    loc.map(|(alloc_id, base_offset, _)| (alloc_id, alloc_range(base_offset, size))),
                ));
            }
            drop(global); // don't hold that reference any longer than we have to

            let Some((alloc_id, base_offset, orig_tag)) = loc else {
                return Ok(())
            };

            let (_size, _align, alloc_kind) = this.get_alloc_info(alloc_id);
            match alloc_kind {
                AllocKind::LiveData => {
                    let current_span = &mut this.machine.current_span();
                    // This should have alloc_extra data, but `get_alloc_extra` can still fail
                    // if converting this alloc_id from a global to a local one
                    // uncovers a non-supported `extern static`.
                    let extra = this.get_alloc_extra(alloc_id)?;
                    let mut stacked_borrows = extra
                        .stacked_borrows
                        .as_ref()
                        .expect("we should have Stacked Borrows data")
                        .borrow_mut();
                    let threads = &this.machine.threads;
                    // Note that we create a *second* `DiagnosticCxBuilder` below for the actual retag.
                    // FIXME: can this be done cleaner?
                    let dcx = DiagnosticCxBuilder::retag(
                        current_span,
                        threads,
                        retag_cause,
                        new_tag,
                        orig_tag,
                        alloc_range(base_offset, size),
                    );
                    let mut dcx = dcx.build(&mut stacked_borrows.history, base_offset);
                    dcx.log_creation();
                    if protect {
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

        if protect {
            // See comment in `Stack::item_popped` for why we store the tag twice.
            this.frame_mut().extra.stacked_borrows.as_mut().unwrap().protected_tags.push(new_tag);
            this.machine.stacked_borrows.as_mut().unwrap().get_mut().protected_tags.insert(new_tag);
        }

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
                // FIXME: can't share this with the current_span inside log_creation
                let mut current_span = this.machine.current_span();
                this.visit_freeze_sensitive(place, size, |mut range, frozen| {
                    // Adjust range.
                    range.start += base_offset;
                    // We are only ever `SharedReadOnly` inside the frozen bits.
                    let perm = if frozen {
                        Permission::SharedReadOnly
                    } else {
                        Permission::SharedReadWrite
                    };
                    let protected = if frozen {
                        protect
                    } else {
                        // We do not protect inside UnsafeCell.
                        // This fixes https://github.com/rust-lang/rust/issues/55005.
                        false
                    };
                    let item = Item::new(new_tag, perm, protected);
                    let mut global = this.machine.stacked_borrows.as_ref().unwrap().borrow_mut();
                    let dcx = DiagnosticCxBuilder::retag(
                        &mut current_span, // FIXME avoid this `clone`
                        &this.machine.threads,
                        retag_cause,
                        new_tag,
                        orig_tag,
                        alloc_range(base_offset, size),
                    );
                    stacked_borrows.for_each(range, dcx, |stack, dcx, exposed_tags| {
                        stack.grant(orig_tag, item, &mut global, dcx, exposed_tags)
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
        let item = Item::new(new_tag, perm, protect);
        let range = alloc_range(base_offset, size);
        let mut global = machine.stacked_borrows.as_ref().unwrap().borrow_mut();
        // FIXME: can't share this with the current_span inside log_creation
        let current_span = &mut machine.current_span();
        let dcx = DiagnosticCxBuilder::retag(
            current_span,
            &machine.threads,
            retag_cause,
            new_tag,
            orig_tag,
            alloc_range(base_offset, size),
        );
        stacked_borrows.for_each(range, dcx, |stack, dcx, exposed_tags| {
            stack.grant(orig_tag, item, &mut global, dcx, exposed_tags)
        })?;

        Ok(Some(alloc_id))
    }

    /// Retags an indidual pointer, returning the retagged version.
    /// `mutbl` can be `None` to make this a raw pointer.
    fn retag_reference(
        &mut self,
        val: &ImmTy<'tcx, Provenance>,
        kind: RefKind,
        retag_cause: RetagCause, // What caused this retag, for diagnostics only
        protect: bool,
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

        // Determine mutability and whether to add a protector.
        // Cannot use `builtin_deref` because that reports *immutable* for `Box`,
        // making it useless.
        fn qualify(ty: Ty<'_>, kind: RetagKind) -> Option<(RefKind, bool)> {
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
                // Boxes are handled separately due to that allocator situation,
                // see the visitor below.
                _ => None,
            }
        }

        // The actual visitor.
        struct RetagVisitor<'ecx, 'mir, 'tcx> {
            ecx: &'ecx mut MiriInterpCx<'mir, 'tcx>,
            kind: RetagKind,
            retag_cause: RetagCause,
            retag_fields: bool,
        }
        impl<'ecx, 'mir, 'tcx> RetagVisitor<'ecx, 'mir, 'tcx> {
            #[inline(always)] // yes this helps in our benchmarks
            fn retag_place(
                &mut self,
                place: &PlaceTy<'tcx, Provenance>,
                ref_kind: RefKind,
                retag_cause: RetagCause,
                protector: bool,
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
                // Boxes do not get a protector: protectors reflect that references outlive the call
                // they were passed in to; that's just not the case for boxes.
                self.retag_place(
                    place,
                    RefKind::Unique { two_phase: false },
                    self.retag_cause,
                    /*protector*/ false,
                )
            }

            fn visit_value(&mut self, place: &PlaceTy<'tcx, Provenance>) -> InterpResult<'tcx> {
                // If this place is smaller than a pointer, we know that it can't contain any
                // pointers we need to retag, so we can stop recursion early.
                // This optimization is crucial for ZSTs, because they can contain way more fields
                // than we can ever visit.
                if !place.layout.is_unsized() && place.layout.size < self.ecx.pointer_size() {
                    return Ok(());
                }

                if let Some((ref_kind, protector)) = qualify(place.layout.ty, self.kind) {
                    self.retag_place(place, ref_kind, self.retag_cause, protector)?;
                } else if matches!(place.layout.ty.kind(), ty::RawPtr(..)) {
                    // Wide raw pointers *do* have fields and their types are strange.
                    // vtables have a type like `&[*const (); 3]` or so!
                    // Do *not* recurse into them.
                    // (No need to worry about wide references, those always "qualify". And Boxes
                    // are handles specially by the visitor anyway.)
                } else if self.retag_fields
                    || place.layout.ty.ty_adt_def().is_some_and(|adt| adt.is_box())
                {
                    // Recurse deeper. Need to always recurse for `Box` to even hit `visit_box`.
                    // (Yes this means we technically also recursively retag the allocator itself
                    // even if field retagging is not enabled. *shrug*)
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
        // Reborrow it.
        let val = this.retag_reference(
            &val,
            RefKind::Unique { two_phase: false },
            RetagCause::FnReturn,
            /*protector*/ true,
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
}
