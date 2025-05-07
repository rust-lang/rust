use rustc_middle::mir::visit::Visitor;
use rustc_middle::mir::{
    Body, Local, Location, Place, PlaceRef, ProjectionElem, Rvalue, Statement, StatementKind,
    Terminator, TerminatorKind,
};
use rustc_middle::ty::TyCtxt;
use tracing::debug;

use super::{PoloniusFacts, PoloniusLocationTable};
use crate::borrow_set::BorrowSet;
use crate::places_conflict;

/// Emit `loan_killed_at` and `cfg_edge` facts at the same time.
pub(super) fn emit_loan_kills<'tcx>(
    tcx: TyCtxt<'tcx>,
    facts: &mut PoloniusFacts,
    body: &Body<'tcx>,
    location_table: &PoloniusLocationTable,
    borrow_set: &BorrowSet<'tcx>,
) {
    let mut visitor = LoanKillsGenerator { borrow_set, tcx, location_table, facts, body };
    for (bb, data) in body.basic_blocks.iter_enumerated() {
        visitor.visit_basic_block_data(bb, data);
    }
}

struct LoanKillsGenerator<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    facts: &'a mut PoloniusFacts,
    location_table: &'a PoloniusLocationTable,
    borrow_set: &'a BorrowSet<'tcx>,
    body: &'a Body<'tcx>,
}

impl<'a, 'tcx> Visitor<'tcx> for LoanKillsGenerator<'a, 'tcx> {
    fn visit_statement(&mut self, statement: &Statement<'tcx>, location: Location) {
        // Also record CFG facts here.
        self.facts.cfg_edge.push((
            self.location_table.start_index(location),
            self.location_table.mid_index(location),
        ));

        self.facts.cfg_edge.push((
            self.location_table.mid_index(location),
            self.location_table.start_index(location.successor_within_block()),
        ));

        // If there are borrows on this now dead local, we need to record them as `killed`.
        if let StatementKind::StorageDead(local) = statement.kind {
            self.record_killed_borrows_for_local(local, location);
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
        // Also record CFG facts here.
        self.facts.cfg_edge.push((
            self.location_table.start_index(location),
            self.location_table.mid_index(location),
        ));

        let successor_blocks = terminator.successors();
        self.facts.cfg_edge.reserve(successor_blocks.size_hint().0);
        for successor_block in successor_blocks {
            self.facts.cfg_edge.push((
                self.location_table.mid_index(location),
                self.location_table.start_index(successor_block.start_location()),
            ));
        }

        // A `Call` terminator's return value can be a local which has borrows,
        // so we need to record those as `killed` as well.
        if let TerminatorKind::Call { destination, .. } = terminator.kind {
            self.record_killed_borrows_for_place(destination, location);
        }

        self.super_terminator(terminator, location);
    }
}

impl<'tcx> LoanKillsGenerator<'_, 'tcx> {
    /// Records the borrows on the specified place as `killed`. For example, when assigning to a
    /// local, or on a call's return destination.
    fn record_killed_borrows_for_place(&mut self, place: Place<'tcx>, location: Location) {
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

                self.record_killed_borrows_for_local(local, location);
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
                            self.tcx,
                            self.body,
                            self.borrow_set[borrow_index].borrowed_place,
                            place,
                            places_conflict::PlaceConflictBias::NoOverlap,
                        );

                        if places_conflict {
                            let location_index = self.location_table.mid_index(location);
                            self.facts.loan_killed_at.push((borrow_index, location_index));
                        }
                    }
                }
            }
        }
    }

    /// Records the borrows on the specified local as `killed`.
    fn record_killed_borrows_for_local(&mut self, local: Local, location: Location) {
        if let Some(borrow_indices) = self.borrow_set.local_map.get(&local) {
            let location_index = self.location_table.mid_index(location);
            self.facts.loan_killed_at.reserve(borrow_indices.len());
            for &borrow_index in borrow_indices {
                self.facts.loan_killed_at.push((borrow_index, location_index));
            }
        }
    }
}
