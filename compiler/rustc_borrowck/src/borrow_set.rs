use std::fmt;
use std::ops::Index;

use rustc_data_structures::fx::{FxIndexMap, FxIndexSet};
use rustc_hir::Mutability;
use rustc_index::bit_set::DenseBitSet;
use rustc_middle::mir::visit::{MutatingUseContext, NonUseContext, PlaceContext, Visitor};
use rustc_middle::mir::{self, Body, Local, Location, traversal};
use rustc_middle::ty::{RegionVid, TyCtxt};
use rustc_middle::{bug, span_bug, ty};
use rustc_mir_dataflow::move_paths::MoveData;
use tracing::debug;

use crate::BorrowIndex;
use crate::place_ext::PlaceExt;

pub struct BorrowSet<'tcx> {
    /// The fundamental map relating bitvector indexes to the borrows
    /// in the MIR. Each borrow is also uniquely identified in the MIR
    /// by the `Location` of the assignment statement in which it
    /// appears on the right hand side. Thus the location is the map
    /// key, and its position in the map corresponds to `BorrowIndex`.
    pub(crate) location_map: FxIndexMap<Location, BorrowData<'tcx>>,

    /// Locations which activate borrows.
    /// NOTE: a given location may activate more than one borrow in the future
    /// when more general two-phase borrow support is introduced, but for now we
    /// only need to store one borrow index.
    pub(crate) activation_map: FxIndexMap<Location, Vec<BorrowIndex>>,

    /// Map from local to all the borrows on that local.
    pub(crate) local_map: FxIndexMap<mir::Local, FxIndexSet<BorrowIndex>>,

    pub(crate) locals_state_at_exit: LocalsStateAtExit,
}

// These methods are public to support borrowck consumers.
impl<'tcx> BorrowSet<'tcx> {
    pub fn location_map(&self) -> &FxIndexMap<Location, BorrowData<'tcx>> {
        &self.location_map
    }

    pub fn activation_map(&self) -> &FxIndexMap<Location, Vec<BorrowIndex>> {
        &self.activation_map
    }

    pub fn local_map(&self) -> &FxIndexMap<mir::Local, FxIndexSet<BorrowIndex>> {
        &self.local_map
    }

    pub fn locals_state_at_exit(&self) -> &LocalsStateAtExit {
        &self.locals_state_at_exit
    }
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
pub enum TwoPhaseActivation {
    NotTwoPhase,
    NotActivated,
    ActivatedAt(Location),
}

#[derive(Debug, Clone)]
pub struct BorrowData<'tcx> {
    /// Location where the borrow reservation starts.
    /// In many cases, this will be equal to the activation location but not always.
    pub(crate) reserve_location: Location,
    /// Location where the borrow is activated.
    pub(crate) activation_location: TwoPhaseActivation,
    /// What kind of borrow this is
    pub(crate) kind: mir::BorrowKind,
    /// The region for which this borrow is live
    pub(crate) region: RegionVid,
    /// Place from which we are borrowing
    pub(crate) borrowed_place: mir::Place<'tcx>,
    /// Place to which the borrow was stored
    pub(crate) assigned_place: mir::Place<'tcx>,
}

// These methods are public to support borrowck consumers.
impl<'tcx> BorrowData<'tcx> {
    pub fn reserve_location(&self) -> Location {
        self.reserve_location
    }

    pub fn activation_location(&self) -> TwoPhaseActivation {
        self.activation_location
    }

    pub fn kind(&self) -> mir::BorrowKind {
        self.kind
    }

    pub fn region(&self) -> RegionVid {
        self.region
    }

    pub fn borrowed_place(&self) -> mir::Place<'tcx> {
        self.borrowed_place
    }

    pub fn assigned_place(&self) -> mir::Place<'tcx> {
        self.assigned_place
    }
}

impl<'tcx> fmt::Display for BorrowData<'tcx> {
    fn fmt(&self, w: &mut fmt::Formatter<'_>) -> fmt::Result {
        let kind = match self.kind {
            mir::BorrowKind::Shared => "",
            mir::BorrowKind::Fake(mir::FakeBorrowKind::Deep) => "fake ",
            mir::BorrowKind::Fake(mir::FakeBorrowKind::Shallow) => "fake shallow ",
            mir::BorrowKind::Mut { kind: mir::MutBorrowKind::ClosureCapture } => "uniq ",
            // FIXME: differentiate `TwoPhaseBorrow`
            mir::BorrowKind::Mut {
                kind: mir::MutBorrowKind::Default | mir::MutBorrowKind::TwoPhaseBorrow,
            } => "mut ",
        };
        write!(w, "&{:?} {}{:?}", self.region, kind, self.borrowed_place)
    }
}

pub enum LocalsStateAtExit {
    AllAreInvalidated,
    SomeAreInvalidated { has_storage_dead_or_moved: DenseBitSet<Local> },
}

impl LocalsStateAtExit {
    fn build<'tcx>(
        locals_are_invalidated_at_exit: bool,
        body: &Body<'tcx>,
        move_data: &MoveData<'tcx>,
    ) -> Self {
        struct HasStorageDead(DenseBitSet<Local>);

        impl<'tcx> Visitor<'tcx> for HasStorageDead {
            fn visit_local(&mut self, local: Local, ctx: PlaceContext, _: Location) {
                if ctx == PlaceContext::NonUse(NonUseContext::StorageDead) {
                    self.0.insert(local);
                }
            }
        }

        if locals_are_invalidated_at_exit {
            LocalsStateAtExit::AllAreInvalidated
        } else {
            let mut has_storage_dead =
                HasStorageDead(DenseBitSet::new_empty(body.local_decls.len()));
            has_storage_dead.visit_body(body);
            let mut has_storage_dead_or_moved = has_storage_dead.0;
            for move_out in &move_data.moves {
                has_storage_dead_or_moved.insert(move_data.base_local(move_out.path));
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
            body,
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

        for (block, block_data) in traversal::preorder(body) {
            visitor.visit_basic_block_data(block, block_data);
        }

        BorrowSet {
            location_map: visitor.location_map,
            activation_map: visitor.activation_map,
            local_map: visitor.local_map,
            locals_state_at_exit: visitor.locals_state_at_exit,
        }
    }

    pub(crate) fn activations_at_location(&self, location: Location) -> &[BorrowIndex] {
        self.activation_map.get(&location).map_or(&[], |activations| &activations[..])
    }

    pub(crate) fn len(&self) -> usize {
        self.location_map.len()
    }

    pub(crate) fn indices(&self) -> impl Iterator<Item = BorrowIndex> {
        BorrowIndex::ZERO..BorrowIndex::from_usize(self.len())
    }

    pub(crate) fn iter_enumerated(&self) -> impl Iterator<Item = (BorrowIndex, &BorrowData<'tcx>)> {
        self.indices().zip(self.location_map.values())
    }

    pub(crate) fn get_index_of(&self, location: &Location) -> Option<BorrowIndex> {
        self.location_map.get_index_of(location).map(BorrowIndex::from)
    }
}

struct GatherBorrows<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    body: &'a Body<'tcx>,
    location_map: FxIndexMap<Location, BorrowData<'tcx>>,
    activation_map: FxIndexMap<Location, Vec<BorrowIndex>>,
    local_map: FxIndexMap<mir::Local, FxIndexSet<BorrowIndex>>,

    /// When we encounter a 2-phase borrow statement, it will always
    /// be assigning into a temporary TEMP:
    ///
    ///    TEMP = &foo
    ///
    /// We add TEMP into this map with `b`, where `b` is the index of
    /// the borrow. When we find a later use of this activation, we
    /// remove from the map (and add to the "tombstone" set below).
    pending_activations: FxIndexMap<mir::Local, BorrowIndex>,

    locals_state_at_exit: LocalsStateAtExit,
}

impl<'a, 'tcx> Visitor<'tcx> for GatherBorrows<'a, 'tcx> {
    fn visit_assign(
        &mut self,
        assigned_place: &mir::Place<'tcx>,
        rvalue: &mir::Rvalue<'tcx>,
        location: mir::Location,
    ) {
        if let &mir::Rvalue::Ref(region, kind, borrowed_place) = rvalue {
            if borrowed_place.ignore_borrow(self.tcx, self.body, &self.locals_state_at_exit) {
                debug!("ignoring_borrow of {:?}", borrowed_place);
                return;
            }

            let region = region.as_var();
            let borrow = |activation_location| BorrowData {
                kind,
                region,
                reserve_location: location,
                activation_location,
                borrowed_place,
                assigned_place: *assigned_place,
            };

            let idx = if !kind.is_two_phase_borrow() {
                debug!("  -> {:?}", location);
                let (idx, _) = self
                    .location_map
                    .insert_full(location, borrow(TwoPhaseActivation::NotTwoPhase));
                BorrowIndex::from(idx)
            } else {
                // When we encounter a 2-phase borrow statement, it will always
                // be assigning into a temporary TEMP:
                //
                //    TEMP = &foo
                //
                // so extract `temp`.
                let Some(temp) = assigned_place.as_local() else {
                    span_bug!(
                        self.body.source_info(location).span,
                        "expected 2-phase borrow to assign to a local, not `{:?}`",
                        assigned_place,
                    );
                };

                // Consider the borrow not activated to start. When we find an activation, we'll update
                // this field.
                let (idx, _) = self
                    .location_map
                    .insert_full(location, borrow(TwoPhaseActivation::NotActivated));
                let idx = BorrowIndex::from(idx);

                // Insert `temp` into the list of pending activations. From
                // now on, we'll be on the lookout for a use of it. Note that
                // we are guaranteed that this use will come after the
                // assignment.
                let prev = self.pending_activations.insert(temp, idx);
                assert_eq!(prev, None, "temporary associated with multiple two phase borrows");

                idx
            };

            self.local_map.entry(borrowed_place.local).or_default().insert(idx);
        } else if let &mir::Rvalue::Reborrow(mutability, borrowed_place) = rvalue {
            let borrowed_place_ty = borrowed_place.ty(self.body, self.tcx).ty;
            let &ty::Adt(reborrowed_adt, _reborrowed_args) = borrowed_place_ty.kind() else {
                unreachable!()
            };
            let &ty::Adt(target_adt, assigned_args) =
                assigned_place.ty(self.body, self.tcx).ty.kind()
            else {
                unreachable!()
            };
            let borrow = if mutability == Mutability::Mut {
                // Reborrow
                if target_adt.did() != reborrowed_adt.did() {
                    bug!(
                        "hir-typeck passed but Reborrow involves mismatching types at {location:?}"
                    )
                }
                let Some(ty::GenericArgKind::Lifetime(region)) =
                    assigned_args.get(0).map(|r| r.kind())
                else {
                    bug!(
                        "hir-typeck passed but {} does not have a lifetime argument",
                        if mutability == Mutability::Mut { "Reborrow" } else { "CoerceShared" }
                    );
                };
                let region = region.as_var();
                let kind = mir::BorrowKind::Mut { kind: mir::MutBorrowKind::Default };
                BorrowData {
                    kind,
                    region,
                    reserve_location: location,
                    activation_location: TwoPhaseActivation::NotTwoPhase,
                    borrowed_place,
                    assigned_place: *assigned_place,
                }
            } else {
                // CoerceShared
                if target_adt.did() == reborrowed_adt.did() {
                    bug!(
                        "hir-typeck passed but CoerceShared involves matching types at {location:?}"
                    )
                }
                let Some(ty::GenericArgKind::Lifetime(region)) =
                    assigned_args.get(0).map(|r| r.kind())
                else {
                    bug!(
                        "hir-typeck passed but {} does not have a lifetime argument",
                        if mutability == Mutability::Mut { "Reborrow" } else { "CoerceShared" }
                    );
                };
                let region = region.as_var();
                let kind = mir::BorrowKind::Shared;
                BorrowData {
                    kind,
                    region,
                    reserve_location: location,
                    activation_location: TwoPhaseActivation::NotTwoPhase,
                    borrowed_place,
                    assigned_place: *assigned_place,
                }
            };
            let kind = borrow.kind;
            let (idx, _) = self.location_map.insert_full(location, borrow);
            let idx = BorrowIndex::from(idx);

            self.insert_as_pending_if_two_phase(location, assigned_place, kind, idx);

            self.local_map.entry(borrowed_place.local).or_default().insert(idx);
        }

        self.super_assign(assigned_place, rvalue, location)
    }

    fn visit_local(&mut self, temp: Local, context: PlaceContext, location: Location) {
        if !context.is_use() {
            return;
        }

        // We found a use of some temporary TMP
        // check whether we (earlier) saw a 2-phase borrow like
        //
        //     TMP = &mut place
        if let Some(&borrow_index) = self.pending_activations.get(&temp) {
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
        if let &mir::Rvalue::Ref(region, kind, place) = rvalue {
            // double-check that we already registered a BorrowData for this

            let borrow_data = &self.location_map[&location];
            assert_eq!(borrow_data.reserve_location, location);
            assert_eq!(borrow_data.kind, kind);
            assert_eq!(borrow_data.region, region.as_var());
            assert_eq!(borrow_data.borrowed_place, place);
        }

        self.super_rvalue(rvalue, location)
    }
}
