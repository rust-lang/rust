use std::cell::RefCell;
use std::fmt;
use std::num::NonZeroU64;

use log::trace;
use smallvec::SmallVec;

use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_middle::mir::RetagKind;
use rustc_target::abi::Size;

use crate::*;
pub mod stacked_borrows;
pub mod tree_borrows;

pub type CallId = NonZeroU64;

/// Tracking pointer provenance
#[derive(Copy, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct BorTag(NonZeroU64);

impl BorTag {
    pub fn new(i: u64) -> Option<Self> {
        NonZeroU64::new(i).map(BorTag)
    }

    pub fn get(&self) -> u64 {
        self.0.get()
    }

    pub fn inner(&self) -> NonZeroU64 {
        self.0
    }

    pub fn succ(self) -> Option<Self> {
        self.0.checked_add(1).map(Self)
    }

    /// The minimum representable tag
    pub fn one() -> Self {
        Self::new(1).unwrap()
    }
}

impl std::default::Default for BorTag {
    /// The default to be used when borrow tracking is disabled
    fn default() -> Self {
        Self::one()
    }
}

impl fmt::Debug for BorTag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<{}>", self.0)
    }
}

/// Per-call-stack-frame data for borrow tracking
#[derive(Debug)]
pub struct FrameState {
    /// The ID of the call this frame corresponds to.
    pub call_id: CallId,

    /// If this frame is protecting any tags, they are listed here. We use this list to do
    /// incremental updates of the global list of protected tags stored in the
    /// `stacked_borrows::GlobalState` upon function return, and if we attempt to pop a protected
    /// tag, to identify which call is responsible for protecting the tag.
    /// See `Stack::item_popped` for more explanation.
    ///
    /// This will contain one tag per reference passed to the function, so
    /// a size of 2 is enough for the vast majority of functions.
    pub protected_tags: SmallVec<[BorTag; 2]>,
}

impl VisitTags for FrameState {
    fn visit_tags(&self, _visit: &mut dyn FnMut(BorTag)) {
        // `protected_tags` are already recorded by `GlobalStateInner`.
    }
}

/// Extra global state, available to the memory access hooks.
#[derive(Debug)]
pub struct GlobalStateInner {
    /// Borrow tracker method currently in use.
    pub borrow_tracker_method: BorrowTrackerMethod,
    /// Next unused pointer ID (tag).
    pub next_ptr_tag: BorTag,
    /// Table storing the "base" tag for each allocation.
    /// The base tag is the one used for the initial pointer.
    /// We need this in a separate table to handle cyclic statics.
    pub base_ptr_tags: FxHashMap<AllocId, BorTag>,
    /// Next unused call ID (for protectors).
    pub next_call_id: CallId,
    /// All currently protected tags.
    /// An item is protected if its tag is in this set, *and* it has the "protected" bit set.
    /// We add tags to this when they are created with a protector in `reborrow`, and
    /// we remove tags from this when the call which is protecting them returns, in
    /// `GlobalStateInner::end_call`. See `Stack::item_popped` for more details.
    pub protected_tags: FxHashMap<BorTag, ProtectorKind>,
    /// The pointer ids to trace
    pub tracked_pointer_tags: FxHashSet<BorTag>,
    /// The call ids to trace
    pub tracked_call_ids: FxHashSet<CallId>,
    /// Whether to recurse into datatypes when searching for pointers to retag.
    pub retag_fields: RetagFields,
    /// Whether `core::ptr::Unique` gets special (`Box`-like) handling.
    pub unique_is_unique: bool,
}

impl VisitTags for GlobalStateInner {
    fn visit_tags(&self, visit: &mut dyn FnMut(BorTag)) {
        for &tag in self.protected_tags.keys() {
            visit(tag);
        }
        // The only other candidate is base_ptr_tags, and that does not need visiting since we don't ever
        // GC the bottommost/root tag.
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

/// Policy on whether to recurse into fields to retag
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

/// The flavor of the protector.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ProtectorKind {
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

/// Utilities for initialization and ID generation
impl GlobalStateInner {
    pub fn new(
        borrow_tracker_method: BorrowTrackerMethod,
        tracked_pointer_tags: FxHashSet<BorTag>,
        tracked_call_ids: FxHashSet<CallId>,
        retag_fields: RetagFields,
        unique_is_unique: bool,
    ) -> Self {
        GlobalStateInner {
            borrow_tracker_method,
            next_ptr_tag: BorTag::one(),
            base_ptr_tags: FxHashMap::default(),
            next_call_id: NonZeroU64::new(1).unwrap(),
            protected_tags: FxHashMap::default(),
            tracked_pointer_tags,
            tracked_call_ids,
            retag_fields,
            unique_is_unique,
        }
    }

    /// Generates a new pointer tag. Remember to also check track_pointer_tags and log its creation!
    pub fn new_ptr(&mut self) -> BorTag {
        let id = self.next_ptr_tag;
        self.next_ptr_tag = id.succ().unwrap();
        id
    }

    pub fn new_frame(&mut self, machine: &MiriMachine<'_, '_>) -> FrameState {
        let call_id = self.next_call_id;
        trace!("new_frame: Assigning call ID {}", call_id);
        if self.tracked_call_ids.contains(&call_id) {
            machine.emit_diagnostic(NonHaltingDiagnostic::CreatedCallId(call_id));
        }
        self.next_call_id = NonZeroU64::new(call_id.get() + 1).unwrap();
        FrameState { call_id, protected_tags: SmallVec::new() }
    }

    pub fn end_call(&mut self, frame: &machine::FrameExtra<'_>) {
        for tag in &frame
            .borrow_tracker
            .as_ref()
            .expect("we should have borrow tracking data")
            .protected_tags
        {
            self.protected_tags.remove(tag);
        }
    }

    pub fn base_ptr_tag(&mut self, id: AllocId, machine: &MiriMachine<'_, '_>) -> BorTag {
        self.base_ptr_tags.get(&id).copied().unwrap_or_else(|| {
            let tag = self.new_ptr();
            if self.tracked_pointer_tags.contains(&tag) {
                machine.emit_diagnostic(NonHaltingDiagnostic::CreatedPointerTag(
                    tag.inner(),
                    None,
                    None,
                ));
            }
            trace!("New allocation {:?} has base tag {:?}", id, tag);
            self.base_ptr_tags.try_insert(id, tag).unwrap();
            tag
        })
    }
}

/// Which borrow tracking method to use
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum BorrowTrackerMethod {
    /// Stacked Borrows, as implemented in borrow_tracker/stacked_borrows
    StackedBorrows,
    /// Tree borrows, as implemented in borrow_tracker/tree_borrows
    TreeBorrows,
}

impl BorrowTrackerMethod {
    pub fn instantiate_global_state(self, config: &MiriConfig) -> GlobalState {
        RefCell::new(GlobalStateInner::new(
            self,
            config.tracked_pointer_tags.clone(),
            config.tracked_call_ids.clone(),
            config.retag_fields,
            config.unique_is_unique,
        ))
    }
}

impl GlobalStateInner {
    pub fn new_allocation(
        &mut self,
        id: AllocId,
        alloc_size: Size,
        kind: MemoryKind<machine::MiriMemoryKind>,
        machine: &MiriMachine<'_, '_>,
    ) -> AllocState {
        match self.borrow_tracker_method {
            BorrowTrackerMethod::StackedBorrows =>
                AllocState::StackedBorrows(Box::new(RefCell::new(Stacks::new_allocation(
                    id, alloc_size, self, kind, machine,
                )))),
            BorrowTrackerMethod::TreeBorrows =>
                AllocState::TreeBorrows(Box::new(RefCell::new(Tree::new_allocation(
                    id, alloc_size, self, kind, machine,
                )))),
        }
    }
}

impl<'mir, 'tcx: 'mir> EvalContextExt<'mir, 'tcx> for crate::MiriInterpCx<'mir, 'tcx> {}
pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriInterpCxExt<'mir, 'tcx> {
    fn retag_ptr_value(
        &mut self,
        kind: RetagKind,
        val: &ImmTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx, ImmTy<'tcx, Provenance>> {
        let this = self.eval_context_mut();
        let method = this.machine.borrow_tracker.as_ref().unwrap().borrow().borrow_tracker_method;
        match method {
            BorrowTrackerMethod::StackedBorrows => this.sb_retag_ptr_value(kind, val),
            BorrowTrackerMethod::TreeBorrows => this.tb_retag_ptr_value(kind, val),
        }
    }

    fn retag_place_contents(
        &mut self,
        kind: RetagKind,
        place: &PlaceTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        let method = this.machine.borrow_tracker.as_ref().unwrap().borrow().borrow_tracker_method;
        match method {
            BorrowTrackerMethod::StackedBorrows => this.sb_retag_place_contents(kind, place),
            BorrowTrackerMethod::TreeBorrows => this.tb_retag_place_contents(kind, place),
        }
    }

    fn protect_place(&mut self, place: &MPlaceTy<'tcx, Provenance>) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        let method = this.machine.borrow_tracker.as_ref().unwrap().borrow().borrow_tracker_method;
        match method {
            BorrowTrackerMethod::StackedBorrows => this.sb_protect_place(place),
            BorrowTrackerMethod::TreeBorrows => this.tb_protect_place(place),
        }
    }

    fn expose_tag(&mut self, alloc_id: AllocId, tag: BorTag) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        let method = this.machine.borrow_tracker.as_ref().unwrap().borrow().borrow_tracker_method;
        match method {
            BorrowTrackerMethod::StackedBorrows => this.sb_expose_tag(alloc_id, tag),
            BorrowTrackerMethod::TreeBorrows => this.tb_expose_tag(alloc_id, tag),
        }
    }

    fn give_pointer_debug_name(
        &mut self,
        ptr: Pointer<Option<Provenance>>,
        nth_parent: u8,
        name: &str,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        let method = this.machine.borrow_tracker.as_ref().unwrap().borrow().borrow_tracker_method;
        match method {
            BorrowTrackerMethod::StackedBorrows => {
                this.tcx.tcx.sess.warn("Stacked Borrows does not support named pointers; `miri_pointer_name` is a no-op");
                Ok(())
            }
            BorrowTrackerMethod::TreeBorrows =>
                this.tb_give_pointer_debug_name(ptr, nth_parent, name),
        }
    }

    fn print_borrow_state(&mut self, alloc_id: AllocId, show_unnamed: bool) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        let method = this.machine.borrow_tracker.as_ref().unwrap().borrow().borrow_tracker_method;
        match method {
            BorrowTrackerMethod::StackedBorrows => this.print_stacks(alloc_id),
            BorrowTrackerMethod::TreeBorrows => this.print_tree(alloc_id, show_unnamed),
        }
    }
}

/// Extra per-allocation data for borrow tracking
#[derive(Debug, Clone)]
pub enum AllocState {
    /// Data corresponding to Stacked Borrows
    StackedBorrows(Box<RefCell<stacked_borrows::AllocState>>),
    /// Data corresponding to Tree Borrows
    TreeBorrows(Box<RefCell<tree_borrows::AllocState>>),
}

impl machine::AllocExtra<'_> {
    #[track_caller]
    pub fn borrow_tracker_sb(&self) -> &RefCell<stacked_borrows::AllocState> {
        match self.borrow_tracker {
            Some(AllocState::StackedBorrows(ref sb)) => sb,
            _ => panic!("expected Stacked Borrows borrow tracking, got something else"),
        }
    }

    #[track_caller]
    pub fn borrow_tracker_sb_mut(&mut self) -> &mut RefCell<stacked_borrows::AllocState> {
        match self.borrow_tracker {
            Some(AllocState::StackedBorrows(ref mut sb)) => sb,
            _ => panic!("expected Stacked Borrows borrow tracking, got something else"),
        }
    }

    #[track_caller]
    pub fn borrow_tracker_tb(&self) -> &RefCell<tree_borrows::AllocState> {
        match self.borrow_tracker {
            Some(AllocState::TreeBorrows(ref tb)) => tb,
            _ => panic!("expected Tree Borrows borrow tracking, got something else"),
        }
    }
}

impl AllocState {
    pub fn before_memory_read<'tcx>(
        &self,
        alloc_id: AllocId,
        prov_extra: ProvenanceExtra,
        range: AllocRange,
        machine: &MiriMachine<'_, 'tcx>,
    ) -> InterpResult<'tcx> {
        match self {
            AllocState::StackedBorrows(sb) =>
                sb.borrow_mut().before_memory_read(alloc_id, prov_extra, range, machine),
            AllocState::TreeBorrows(tb) =>
                tb.borrow_mut().before_memory_access(
                    AccessKind::Read,
                    alloc_id,
                    prov_extra,
                    range,
                    machine,
                ),
        }
    }

    pub fn before_memory_write<'tcx>(
        &mut self,
        alloc_id: AllocId,
        prov_extra: ProvenanceExtra,
        range: AllocRange,
        machine: &mut MiriMachine<'_, 'tcx>,
    ) -> InterpResult<'tcx> {
        match self {
            AllocState::StackedBorrows(sb) =>
                sb.get_mut().before_memory_write(alloc_id, prov_extra, range, machine),
            AllocState::TreeBorrows(tb) =>
                tb.get_mut().before_memory_access(
                    AccessKind::Write,
                    alloc_id,
                    prov_extra,
                    range,
                    machine,
                ),
        }
    }

    pub fn before_memory_deallocation<'tcx>(
        &mut self,
        alloc_id: AllocId,
        prov_extra: ProvenanceExtra,
        range: AllocRange,
        machine: &mut MiriMachine<'_, 'tcx>,
    ) -> InterpResult<'tcx> {
        match self {
            AllocState::StackedBorrows(sb) =>
                sb.get_mut().before_memory_deallocation(alloc_id, prov_extra, range, machine),
            AllocState::TreeBorrows(tb) =>
                tb.get_mut().before_memory_deallocation(alloc_id, prov_extra, range, machine),
        }
    }

    pub fn remove_unreachable_tags(&self, tags: &FxHashSet<BorTag>) {
        match self {
            AllocState::StackedBorrows(sb) => sb.borrow_mut().remove_unreachable_tags(tags),
            AllocState::TreeBorrows(tb) => tb.borrow_mut().remove_unreachable_tags(tags),
        }
    }
}

impl VisitTags for AllocState {
    fn visit_tags(&self, visit: &mut dyn FnMut(BorTag)) {
        match self {
            AllocState::StackedBorrows(sb) => sb.visit_tags(visit),
            AllocState::TreeBorrows(tb) => tb.visit_tags(visit),
        }
    }
}
