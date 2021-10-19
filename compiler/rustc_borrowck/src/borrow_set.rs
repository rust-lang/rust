use crate::nll::ToRegionVid;
use crate::path_utils::allow_two_phase_borrow;
use crate::place_ext::PlaceExt;
use crate::BorrowIndex;
use rustc_data_structures::fx::{FxHashMap, FxHashSet, FxIndexMap};
use rustc_index::bit_set::BitSet;
use rustc_middle::mir::traversal;
use rustc_middle::mir::visit::{MutatingUseContext, NonUseContext, PlaceContext, Visitor};
use rustc_middle::mir::{self, Body, Local, Location};
use rustc_middle::ty::{RegionVid, TyCtxt};
use rustc_mir_dataflow::move_paths::MoveData;
use std::fmt;
use std::ops::Index;

crate struct BorrowSet<'tcx> {
    /// The fundamental map relating bitvector indexes to the borrows
    /// in the MIR. Each borrow is also uniquely identified in the MIR
    /// by the `Location` of the assignment statement in which it
    /// appears on the right hand side. Thus the location is the map
    /// key, and its position in the map corresponds to `BorrowIndex`.
    crate location_map: FxIndexMap<Location, BorrowData<'tcx>>,

    /// Locations which activate borrows.
    /// NOTE: a given location may activate more than one borrow in the future
    /// when more general two-phase borrow support is introduced, but for now we
    /// only need to store one borrow index.
    crate activation_map: FxHashMap<Location, Vec<BorrowIndex>>,

    /// Map from local to all the borrows on that local.
    crate local_map: FxHashMap<mir::Local, FxHashSet<BorrowIndex>>,

    crate locals_state_at_exit: LocalsStateAtExit,
}

impl<'tcx> Index<BorrowIndex> for BorrowSet<'tcx> {
    type Output = BorrowData<'tcx>;

    fn index(&self, index: BorrowIndex) -> &BorrowData<'tcx> {
        &self.location_map[index.as_usize()]
    }
}

/// Location where a two-phase borrow is activated, if a borrow
/// is in fact a two-phase borrow.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
crate enum TwoPhaseActivation {
    NotTwoPhase,
    NotActivated,
    ActivatedAt(Location),
}

#[derive(Debug, Clone)]
crate struct BorrowData<'tcx> {
    /// Location where the borrow reservation starts.
    /// In many cases, this will be equal to the activation location but not always.
    crate reserve_location: Location,
    /// Location where the borrow is activated.
    crate activation_location: TwoPhaseActivation,
    /// What kind of borrow this is
    crate kind: mir::BorrowKind,
    /// The region for which this borrow is live
    crate region: RegionVid,
    /// Place from which we are borrowing
    crate borrowed_place: mir::Place<'tcx>,
    /// Place to which the borrow was stored
    crate assigned_place: mir::Place<'tcx>,
}

impl<'tcx> fmt::Display for BorrowData<'tcx> {
    fn fmt(&self, w: &mut fmt::Formatter<'_>) -> fmt::Result {
        let kind = match self.kind {
            mir::BorrowKind::Shared => "",
            mir::BorrowKind::Shallow => "shallow ",
            mir::BorrowKind::Unique => "uniq ",
            mir::BorrowKind::Mut { .. } => "mut ",
        };
        write!(w, "&{:?} {}{:?}", self.region, kind, self.borrowed_place)
    }
}

crate enum LocalsStateAtExit {
    AllAreInvalidated,
    SomeAreInvalidated { has_storage_dead_or_moved: BitSet<Local> },
}

impl LocalsStateAtExit {
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

impl<'tcx> BorrowSet<'tcx> {
    pub fn build(
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

    crate fn activations_at_location(&self, location: Location) -> &[BorrowIndex] {
        self.activation_map.get(&location).map_or(&[], |activations| &activations[..])
    }

    crate fn len(&self) -> usize {
        self.location_map.len()
    }

    crate fn indices(&self) -> impl Iterator<Item = BorrowIndex> {
        BorrowIndex::from_usize(0)..BorrowIndex::from_usize(self.len())
    }

    crate fn iter_enumerated(&self) -> impl Iterator<Item = (BorrowIndex, &BorrowData<'tcx>)> {
        self.indices().zip(self.location_map.values())
    }

    crate fn get_index_of(&self, location: &Location) -> Option<BorrowIndex> {
        self.location_map.get_index_of(location).map(BorrowIndex::from)
    }

    crate fn contains(&self, location: &Location) -> bool {
        self.location_map.contains_key(location)
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
        let Some(temp) = assigned_place.as_local() else {
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
