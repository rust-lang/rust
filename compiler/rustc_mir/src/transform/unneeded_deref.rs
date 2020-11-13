use rustc_data_structures::fx::FxHashMap;
use rustc_index::bit_set::BitSet;
use rustc_middle::{
    mir::visit::Visitor,
    mir::{
        visit::{MutVisitor, PlaceContext},
        Body, BorrowKind, Local, Location, Place, PlaceRef, ProjectionElem, Rvalue, Statement,
        StatementKind,
    },
    ty::TyCtxt,
};

use super::MirPass;
use crate::dataflow::Analysis;
use crate::dataflow::{
    impls::{AvailableLocals, LocalWithLocationIndex},
    lattice::Dual,
    Results, ResultsVisitor,
};

/// Pass to find cases where a dereference of a reference can be avoided by
/// using the referenced value directly.
///
/// Consider the following example:
/// ```
/// _1 = 4;
/// _2 = &_1;
/// _3 = *_2
/// ```
///
/// This is optimized into:
/// ```
/// _1 = 4;
/// _2 = &_1;
/// _3 = _1
/// ```
///
/// Later passes can then potentially remove `_2` as it is now unused.
pub struct UnneededDeref;

impl<'tcx> MirPass<'tcx> for UnneededDeref {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        let optimizations = UnneededDerefVisitor::go(body, tcx);
        if !optimizations.is_empty() {
            MutVisitor::visit_body(&mut ApplyOpts { optimizations, tcx }, body);
        }
    }
}

struct UnneededDerefVisitor<'a, 'tcx> {
    refs: FxHashMap<Local, (Place<'tcx>, (LocalWithLocationIndex, LocalWithLocationIndex))>,
    optimizations: &'a mut FxHashMap<(Location, Place<'tcx>), Place<'tcx>>,
    results: &'a Results<'tcx, AvailableLocals>,
    state: *const Dual<BitSet<LocalWithLocationIndex>>,
}

impl<'a, 'tcx> Visitor<'tcx> for UnneededDerefVisitor<'a, 'tcx> {
    fn visit_place(&mut self, place: &Place<'tcx>, _context: PlaceContext, location: Location) {
        let _: Option<_> = try {
            debug!("Visiting place {:?}", place);
            // SAFETY: We only use self.state here which is always called from statement_before_primary_effect,
            // which guarantees that self.state is still alive.
            let state = unsafe { &*self.state };

            match place.as_ref() {
                PlaceRef { projection: [ProjectionElem::Deref], .. } => {
                    let place_derefed = place;
                    debug!("Refs {:?}", self.refs);
                    // See if we have recorded an earlier assignment where we took the reference of the place we are now dereferencing
                    let (place_taken_ref_of, (lhs_idx, place_taken_ref_of_location)) =
                        self.refs.get(&place_derefed.local)?;

                    // We found a reference. Let's check if it is still valid
                    let place_taken_ref_of_available =
                        state.0.contains(*place_taken_ref_of_location);

                    debug!(
                        "{:?} has availability {:?}",
                        place_taken_ref_of.local, place_taken_ref_of_available
                    );
                    if place_taken_ref_of_available {
                        // And then check if the place we are dereferencing is still valid
                        let place_available = state.0.contains(*lhs_idx);
                        debug!("{:?} has availability {:?}", place.local, place_available);
                        if place_available {
                            self.optimizations
                                .insert((location, place.clone()), place_taken_ref_of.clone());
                        }
                    }
                }

                _ => {},
            }
        };
        // We explicitly do not call super_place as we don't need to explore the graph deeper
    }
}

impl<'a, 'tcx> UnneededDerefVisitor<'a, 'tcx> {
    fn go(
        body: &'a Body<'tcx>,
        tcx: TyCtxt<'tcx>,
    ) -> FxHashMap<(Location, Place<'tcx>), Place<'tcx>> {
        let analysis = AvailableLocals::new(body);
        let results = analysis.into_engine(tcx, body).iterate_to_fixpoint();
        let refs = FxHashMap::default();
        let mut optimizations = FxHashMap::default();

        let mut _self = UnneededDerefVisitor {
            refs,
            optimizations: &mut optimizations,
            results: &results,
            state: std::ptr::null(),
        };

        results.visit_reachable_with(body, &mut _self);

        optimizations
    }
}

impl<'a, 'tcx> ResultsVisitor<'a, 'tcx> for UnneededDerefVisitor<'a, 'tcx> {
    type FlowState = Dual<BitSet<LocalWithLocationIndex>>;
    fn visit_statement_before_primary_effect(
        &mut self,
        state: &Self::FlowState,
        stmt: &'mir Statement<'tcx>,
        location: Location,
    ) {
        self.state = state;
        let analysis = &self.results.analysis;
        debug!("state: {:?} before statement {:?}", analysis.debug_state(state), stmt);
        let _: Option<_> = try {
            match &stmt.kind {
                StatementKind::Assign(box (
                    lhs,
                    Rvalue::Ref(_, BorrowKind::Shared, place_taken_ref_of),
                )) => {
                    let analysis = &self.results.analysis;
                    // Only insert if the place that is referenced is available
                    if let Some(place_taken_ref_of_idx) =
                        analysis.is_available(place_taken_ref_of.local, state)
                    {
                        let lhs_idx = analysis.get_local_with_location_index(lhs.local, location);
                        self.refs.insert(
                            lhs.local,
                            (place_taken_ref_of.clone(), (lhs_idx, place_taken_ref_of_idx)),
                        );
                    }
                }
                StatementKind::Assign(box (_, rvalue)) => match rvalue {
                    rvalue => self.visit_rvalue(rvalue, location),
                },
                _ => {}
            }
        };
    }
}

struct ApplyOpts<'tcx> {
    optimizations: FxHashMap<(Location, Place<'tcx>), Place<'tcx>>,
    tcx: TyCtxt<'tcx>,
}

impl<'tcx> MutVisitor<'tcx> for ApplyOpts<'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn visit_place(&mut self, place: &mut Place<'tcx>, _context: PlaceContext, location: Location) {
        if let Some(found_place) = self.optimizations.remove(&(location, *place)) {
            debug!("unneeded_deref: replacing {:?} with {:?}", place, found_place);
            *place = found_place;
        }
    }
}
