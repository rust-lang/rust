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
    refs: FxHashMap<Local, Place<'tcx>>,
    optimizations: &'a mut FxHashMap<(Location, Place<'tcx>), Place<'tcx>>,
    results: &'a Results<'tcx, AvailableLocals>,
    state: *const Dual<BitSet<LocalWithLocationIndex>>,
}

impl<'a, 'tcx> Visitor<'tcx> for UnneededDerefVisitor<'a, 'tcx> {
    fn visit_place(&mut self, place: &Place<'tcx>, context: PlaceContext, location: Location) {
        let analysis = &self.results.analysis;
        let _: Option<_> = try {
            debug!("Visiting place {:?}", place);
            // SAFETY: We only use self.state here which is always called from statement_before_primary_effect, 
            // which guarantees that self.state is still alive.
            let state = unsafe{self.state.as_ref().unwrap()};

            match place.as_ref() {
                PlaceRef { projection: [ProjectionElem::Deref], .. } => {
                    debug!("Refs {:?}", self.refs);
                    let place_taken_ref_of = self.refs.get(&place.local)?;
                    let place_taken_ref_of_available =
                        analysis.is_available(place_taken_ref_of.local, state);

                    debug!(
                        "{:?} has availability {:?}",
                        place_taken_ref_of.local, place_taken_ref_of_available
                    );
                    if place_taken_ref_of_available {
                        let place_available = analysis.is_available(place.local, state);
                        debug!("{:?} has availability {:?}", place.local, place_available);
                        if place_available {
                            self.optimizations
                                .insert((location, place.clone()), place_taken_ref_of.clone());
                        }
                    }
                }

                _ => None?,
            }
        };
        self.super_place(place, context, location);
    }
}

impl<'a, 'tcx> UnneededDerefVisitor<'a, 'tcx> {
    fn go(
        body: &'a Body<'tcx>,
        tcx: TyCtxt<'tcx>,
    ) -> FxHashMap<(Location, Place<'tcx>), Place<'tcx>> {
        let mut ref_finder = RefFinder::new();
        ref_finder.visit_body(body);
        let refs = ref_finder.refs;
        let mut optimizations = FxHashMap::default();

        // There is point in looking for derefs, if we haven't seen any refs
        if refs.is_empty() {
            return optimizations;
        }

        let analysis = AvailableLocals::new(body);
        let results = analysis.into_engine(tcx, body).iterate_to_fixpoint();

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
                StatementKind::Assign(box (_, rvalue)) => match rvalue {
                    rvalue => self.visit_rvalue(rvalue, location),
                },
                _ => {}
            }
        };
    }
}

struct RefFinder<'tcx> {
    refs: FxHashMap<Local, Place<'tcx>>,
}

impl<'tcx> RefFinder<'tcx> {
    fn new() -> Self {
        Self { refs: FxHashMap::<Local, Place<'tcx>>::default() }
    }
}

impl<'tcx> Visitor<'tcx> for RefFinder<'tcx> {
    fn visit_assign(&mut self, lhs: &Place<'tcx>, rvalue: &Rvalue<'tcx>, _: Location) {
        match rvalue {
            Rvalue::Ref(_, BorrowKind::Shared, place_taken_ref_of) => {
                self.refs.insert(lhs.local, place_taken_ref_of.clone());
            }
            _ => {}
        }
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
