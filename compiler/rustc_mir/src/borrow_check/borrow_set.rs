use crate::borrow_check::nll::ToRegionVid;
use crate::borrow_check::path_utils::allow_two_phase_borrow;
use crate::borrow_check::place_ext::PlaceExt;
use crate::dataflow::move_paths::MoveData;
use rustc_data_structures::fx::{FxHashMap, FxHashSet, FxIndexMap};
use rustc_index::bit_set::BitSet;
pub use rustc_middle::mir::borrows::{
    BorrowData, BorrowIndex, BorrowSet, LocalsStateAtExit, TwoPhaseActivation,
};
use rustc_middle::mir::traversal;
use rustc_middle::mir::visit::{MutatingUseContext, NonUseContext, PlaceContext, Visitor};
use rustc_middle::mir::{self, Body, Local, Location};
use rustc_middle::ty::TyCtxt;

pub trait LocalsStateAtExitExt {
    fn build(
        locals_are_invalidated_at_exit: bool,
        body: &Body<'tcx>,
        move_data: &MoveData<'tcx>,
    ) -> Self;
}

impl LocalsStateAtExitExt for LocalsStateAtExit {
    fn build(
        locals_are_invalidated_at_exit: bool,
        body: &Body<'tcx>,
        move_data: &MoveData<'tcx>,
    ) -> Self {
        struct HasStorageDead(BitSet<Local>);

        impl<'tcx> Visitor<'tcx> for HasStorageDead {
            fn visit_local(&mut self, local: &Local, ctx: PlaceContext, _: Location) {
                if ctx == PlaceContext::NonUse(NonUseContext::StorageDead) {
                    self.0.insert(*local);
                }
            }
        }

        if locals_are_invalidated_at_exit {
            LocalsStateAtExit::AllAreInvalidated
        } else {
            let mut has_storage_dead = HasStorageDead(BitSet::new_empty(body.local_decls.len()));
            has_storage_dead.visit_body(&body);
            let mut has_storage_dead_or_moved = has_storage_dead.0;
            for move_out in &move_data.moves {
                if let Some(index) = move_data.base_local(move_out.path) {
                    has_storage_dead_or_moved.insert(index);
                }
            }
            LocalsStateAtExit::SomeAreInvalidated { has_storage_dead_or_moved }
        }
    }
}

pub trait BorrowSetExt<'tcx> {
    fn build(
        tcx: TyCtxt<'tcx>,
        body: &Body<'tcx>,
        locals_are_invalidated_at_exit: bool,
        move_data: &MoveData<'tcx>,
    ) -> Self;
}

impl<'tcx> BorrowSetExt<'tcx> for BorrowSet<'tcx> {
    fn build(
        tcx: TyCtxt<'tcx>,
        body: &Body<'tcx>,
        locals_are_invalidated_at_exit: bool,
        move_data: &MoveData<'tcx>,
    ) -> Self {
        let mut visitor = GatherBorrows {
            tcx,
            body: &body,
            location_map: Default::default(),
            activation_map: Default::default(),
            local_map: Default::default(),
            pending_activations: Default::default(),
            locals_state_at_exit: LocalsStateAtExit::build(
                locals_are_invalidated_at_exit,
                body,
                move_data,
            ),
        };

        for (block, block_data) in traversal::preorder(&body) {
            visitor.visit_basic_block_data(block, block_data);
        }

        BorrowSet {
            location_map: visitor.location_map,
            activation_map: visitor.activation_map,
            local_map: visitor.local_map,
            locals_state_at_exit: visitor.locals_state_at_exit,
        }
    }
}

struct GatherBorrows<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    body: &'a Body<'tcx>,
    location_map: FxIndexMap<Location, BorrowData<'tcx>>,
    activation_map: FxHashMap<Location, Vec<BorrowIndex>>,
    local_map: FxHashMap<mir::Local, FxHashSet<BorrowIndex>>,

    /// When we encounter a 2-phase borrow statement, it will always
    /// be assigning into a temporary TEMP:
    ///
    ///    TEMP = &foo
    ///
    /// We add TEMP into this map with `b`, where `b` is the index of
    /// the borrow. When we find a later use of this activation, we
    /// remove from the map (and add to the "tombstone" set below).
    pending_activations: FxHashMap<mir::Local, BorrowIndex>,

    locals_state_at_exit: LocalsStateAtExit,
}

impl<'a, 'tcx> Visitor<'tcx> for GatherBorrows<'a, 'tcx> {
    fn visit_assign(
        &mut self,
        assigned_place: &mir::Place<'tcx>,
        rvalue: &mir::Rvalue<'tcx>,
        location: mir::Location,
    ) {
        if let mir::Rvalue::Ref(region, kind, ref borrowed_place) = *rvalue {
            if borrowed_place.ignore_borrow(self.tcx, self.body, &self.locals_state_at_exit) {
                debug!("ignoring_borrow of {:?}", borrowed_place);
                return;
            }

            let region = region.to_region_vid();

            let borrow = BorrowData {
                kind,
                region,
                reserve_location: location,
                activation_location: TwoPhaseActivation::NotTwoPhase,
                borrowed_place: *borrowed_place,
                assigned_place: *assigned_place,
            };
            let (idx, _) = self.location_map.insert_full(location, borrow);
            let idx = BorrowIndex::from(idx);

            self.insert_as_pending_if_two_phase(location, assigned_place, kind, idx);

            self.local_map.entry(borrowed_place.local).or_default().insert(idx);
        }

        self.super_assign(assigned_place, rvalue, location)
    }

    fn visit_local(&mut self, temp: &Local, context: PlaceContext, location: Location) {
        if !context.is_use() {
            return;
        }

        // We found a use of some temporary TMP
        // check whether we (earlier) saw a 2-phase borrow like
        //
        //     TMP = &mut place
        if let Some(&borrow_index) = self.pending_activations.get(temp) {
            let borrow_data = &mut self.location_map[borrow_index.as_usize()];

            // Watch out: the use of TMP in the borrow itself
            // doesn't count as an activation. =)
            if borrow_data.reserve_location == location
                && context == PlaceContext::MutatingUse(MutatingUseContext::Store)
            {
                return;
            }

            if let TwoPhaseActivation::ActivatedAt(other_location) = borrow_data.activation_location
            {
                span_bug!(
                    self.body.source_info(location).span,
                    "found two uses for 2-phase borrow temporary {:?}: \
                     {:?} and {:?}",
                    temp,
                    location,
                    other_location,
                );
            }

            // Otherwise, this is the unique later use that we expect.
            // Double check: This borrow is indeed a two-phase borrow (that is,
            // we are 'transitioning' from `NotActivated` to `ActivatedAt`) and
            // we've not found any other activations (checked above).
            assert_eq!(
                borrow_data.activation_location,
                TwoPhaseActivation::NotActivated,
                "never found an activation for this borrow!",
            );
            self.activation_map.entry(location).or_default().push(borrow_index);

            borrow_data.activation_location = TwoPhaseActivation::ActivatedAt(location);
        }
    }

    fn visit_rvalue(&mut self, rvalue: &mir::Rvalue<'tcx>, location: mir::Location) {
        if let mir::Rvalue::Ref(region, kind, ref place) = *rvalue {
            // double-check that we already registered a BorrowData for this

            let borrow_data = &self.location_map[&location];
            assert_eq!(borrow_data.reserve_location, location);
            assert_eq!(borrow_data.kind, kind);
            assert_eq!(borrow_data.region, region.to_region_vid());
            assert_eq!(borrow_data.borrowed_place, *place);
        }

        self.super_rvalue(rvalue, location)
    }
}

impl<'a, 'tcx> GatherBorrows<'a, 'tcx> {
    /// If this is a two-phase borrow, then we will record it
    /// as "pending" until we find the activating use.
    fn insert_as_pending_if_two_phase(
        &mut self,
        start_location: Location,
        assigned_place: &mir::Place<'tcx>,
        kind: mir::BorrowKind,
        borrow_index: BorrowIndex,
    ) {
        debug!(
            "Borrows::insert_as_pending_if_two_phase({:?}, {:?}, {:?})",
            start_location, assigned_place, borrow_index,
        );

        if !allow_two_phase_borrow(kind) {
            debug!("  -> {:?}", start_location);
            return;
        }

        // When we encounter a 2-phase borrow statement, it will always
        // be assigning into a temporary TEMP:
        //
        //    TEMP = &foo
        //
        // so extract `temp`.
        let temp = if let Some(temp) = assigned_place.as_local() {
            temp
        } else {
            span_bug!(
                self.body.source_info(start_location).span,
                "expected 2-phase borrow to assign to a local, not `{:?}`",
                assigned_place,
            );
        };

        // Consider the borrow not activated to start. When we find an activation, we'll update
        // this field.
        {
            let borrow_data = &mut self.location_map[borrow_index.as_usize()];
            borrow_data.activation_location = TwoPhaseActivation::NotActivated;
        }

        // Insert `temp` into the list of pending activations. From
        // now on, we'll be on the lookout for a use of it. Note that
        // we are guaranteed that this use will come after the
        // assignment.
        let old_value = self.pending_activations.insert(temp, borrow_index);
        if let Some(old_index) = old_value {
            span_bug!(
                self.body.source_info(start_location).span,
                "found already pending activation for temp: {:?} \
                       at borrow_index: {:?} with associated data {:?}",
                temp,
                old_index,
                self.location_map[old_index.as_usize()]
            );
        }
    }
}
