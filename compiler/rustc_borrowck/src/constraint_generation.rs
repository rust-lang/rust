#![deny(rustc::untranslatable_diagnostic)]
#![deny(rustc::diagnostic_outside_of_impl)]
use rustc_infer::infer::InferCtxt;
use rustc_middle::mir::visit::TyContext;
use rustc_middle::mir::visit::Visitor;
use rustc_middle::mir::{
    Body, Local, Location, Place, PlaceRef, ProjectionElem, Rvalue, SourceInfo, Statement,
    StatementKind, Terminator, TerminatorKind, UserTypeProjection,
};
use rustc_middle::ty::visit::TypeVisitable;
use rustc_middle::ty::GenericArgsRef;
use rustc_middle::ty::{self, RegionVid, Ty, TyCtxt};

use crate::{
    borrow_set::BorrowSet, facts::AllFacts, location::LocationTable, places_conflict,
    region_infer::values::LivenessValues,
};

pub(super) fn generate_constraints<'tcx>(
    infcx: &InferCtxt<'tcx>,
    liveness_constraints: &mut LivenessValues<RegionVid>,
    all_facts: &mut Option<AllFacts>,
    location_table: &LocationTable,
    body: &Body<'tcx>,
    borrow_set: &BorrowSet<'tcx>,
) {
    let mut cg = ConstraintGeneration {
        borrow_set,
        infcx,
        liveness_constraints,
        location_table,
        all_facts,
        body,
    };

    for (bb, data) in body.basic_blocks.iter_enumerated() {
        cg.visit_basic_block_data(bb, data);
    }
}

/// 'cg = the duration of the constraint generation process itself.
struct ConstraintGeneration<'cg, 'tcx> {
    infcx: &'cg InferCtxt<'tcx>,
    all_facts: &'cg mut Option<AllFacts>,
    location_table: &'cg LocationTable,
    liveness_constraints: &'cg mut LivenessValues<RegionVid>,
    borrow_set: &'cg BorrowSet<'tcx>,
    body: &'cg Body<'tcx>,
}

impl<'cg, 'tcx> Visitor<'tcx> for ConstraintGeneration<'cg, 'tcx> {
    /// We sometimes have `args` within an rvalue, or within a
    /// call. Make them live at the location where they appear.
    fn visit_args(&mut self, args: &GenericArgsRef<'tcx>, location: Location) {
        self.add_regular_live_constraint(*args, location);
        self.super_args(args);
    }

    /// We sometimes have `region` within an rvalue, or within a
    /// call. Make them live at the location where they appear.
    fn visit_region(&mut self, region: ty::Region<'tcx>, location: Location) {
        self.add_regular_live_constraint(region, location);
        self.super_region(region);
    }

    /// We sometimes have `ty` within an rvalue, or within a
    /// call. Make them live at the location where they appear.
    fn visit_ty(&mut self, ty: Ty<'tcx>, ty_context: TyContext) {
        match ty_context {
            TyContext::ReturnTy(SourceInfo { span, .. })
            | TyContext::YieldTy(SourceInfo { span, .. })
            | TyContext::UserTy(span)
            | TyContext::LocalDecl { source_info: SourceInfo { span, .. }, .. } => {
                span_bug!(span, "should not be visiting outside of the CFG: {:?}", ty_context);
            }
            TyContext::Location(location) => {
                self.add_regular_live_constraint(ty, location);
            }
        }

        self.super_ty(ty);
    }

    fn visit_statement(&mut self, statement: &Statement<'tcx>, location: Location) {
        if let Some(all_facts) = self.all_facts {
            let _prof_timer = self.infcx.tcx.prof.generic_activity("polonius_fact_generation");
            all_facts.cfg_edge.push((
                self.location_table.start_index(location),
                self.location_table.mid_index(location),
            ));

            all_facts.cfg_edge.push((
                self.location_table.mid_index(location),
                self.location_table.start_index(location.successor_within_block()),
            ));

            // If there are borrows on this now dead local, we need to record them as `killed`.
            if let StatementKind::StorageDead(local) = statement.kind {
                record_killed_borrows_for_local(
                    all_facts,
                    self.borrow_set,
                    self.location_table,
                    local,
                    location,
                );
            }
        }

        self.super_statement(statement, location);
    }

    fn visit_assign(&mut self, place: &Place<'tcx>, rvalue: &Rvalue<'tcx>, location: Location) {
        // When we see `X = ...`, then kill borrows of
        // `(*X).foo` and so forth.
        self.record_killed_borrows_for_place(*place, location);

        self.super_assign(place, rvalue, location);
    }

    fn visit_terminator(&mut self, terminator: &Terminator<'tcx>, location: Location) {
        if let Some(all_facts) = self.all_facts {
            let _prof_timer = self.infcx.tcx.prof.generic_activity("polonius_fact_generation");
            all_facts.cfg_edge.push((
                self.location_table.start_index(location),
                self.location_table.mid_index(location),
            ));

            let successor_blocks = terminator.successors();
            all_facts.cfg_edge.reserve(successor_blocks.size_hint().0);
            for successor_block in successor_blocks {
                all_facts.cfg_edge.push((
                    self.location_table.mid_index(location),
                    self.location_table.start_index(successor_block.start_location()),
                ));
            }
        }

        // A `Call` terminator's return value can be a local which has borrows,
        // so we need to record those as `killed` as well.
        if let TerminatorKind::Call { destination, .. } = terminator.kind {
            self.record_killed_borrows_for_place(destination, location);
        }

        self.super_terminator(terminator, location);
    }

    fn visit_ascribe_user_ty(
        &mut self,
        _place: &Place<'tcx>,
        _variance: ty::Variance,
        _user_ty: &UserTypeProjection,
        _location: Location,
    ) {
    }
}

impl<'cx, 'tcx> ConstraintGeneration<'cx, 'tcx> {
    /// Some variable with type `live_ty` is "regular live" at
    /// `location` -- i.e., it may be used later. This means that all
    /// regions appearing in the type `live_ty` must be live at
    /// `location`.
    fn add_regular_live_constraint<T>(&mut self, live_ty: T, location: Location)
    where
        T: TypeVisitable<TyCtxt<'tcx>>,
    {
        debug!("add_regular_live_constraint(live_ty={:?}, location={:?})", live_ty, location);

        self.infcx.tcx.for_each_free_region(&live_ty, |live_region| {
            let vid = live_region.as_var();
            self.liveness_constraints.add_element(vid, location);
        });
    }

    /// When recording facts for Polonius, records the borrows on the specified place
    /// as `killed`. For example, when assigning to a local, or on a call's return destination.
    fn record_killed_borrows_for_place(&mut self, place: Place<'tcx>, location: Location) {
        if let Some(all_facts) = self.all_facts {
            let _prof_timer = self.infcx.tcx.prof.generic_activity("polonius_fact_generation");

            // Depending on the `Place` we're killing:
            // - if it's a local, or a single deref of a local,
            //   we kill all the borrows on the local.
            // - if it's a deeper projection, we have to filter which
            //   of the borrows are killed: the ones whose `borrowed_place`
            //   conflicts with the `place`.
            match place.as_ref() {
                PlaceRef { local, projection: &[] }
                | PlaceRef { local, projection: &[ProjectionElem::Deref] } => {
                    debug!(
                        "Recording `killed` facts for borrows of local={:?} at location={:?}",
                        local, location
                    );

                    record_killed_borrows_for_local(
                        all_facts,
                        self.borrow_set,
                        self.location_table,
                        local,
                        location,
                    );
                }

                PlaceRef { local, projection: &[.., _] } => {
                    // Kill conflicting borrows of the innermost local.
                    debug!(
                        "Recording `killed` facts for borrows of \
                            innermost projected local={:?} at location={:?}",
                        local, location
                    );

                    if let Some(borrow_indices) = self.borrow_set.local_map.get(&local) {
                        for &borrow_index in borrow_indices {
                            let places_conflict = places_conflict::places_conflict(
                                self.infcx.tcx,
                                self.body,
                                self.borrow_set[borrow_index].borrowed_place,
                                place,
                                places_conflict::PlaceConflictBias::NoOverlap,
                            );

                            if places_conflict {
                                let location_index = self.location_table.mid_index(location);
                                all_facts.loan_killed_at.push((borrow_index, location_index));
                            }
                        }
                    }
                }
            }
        }
    }
}

/// When recording facts for Polonius, records the borrows on the specified local as `killed`.
fn record_killed_borrows_for_local(
    all_facts: &mut AllFacts,
    borrow_set: &BorrowSet<'_>,
    location_table: &LocationTable,
    local: Local,
    location: Location,
) {
    if let Some(borrow_indices) = borrow_set.local_map.get(&local) {
        all_facts.loan_killed_at.reserve(borrow_indices.len());
        for &borrow_index in borrow_indices {
            let location_index = location_table.mid_index(location);
            all_facts.loan_killed_at.push((borrow_index, location_index));
        }
    }
}
