//! Functions dedicated to fact generation for the `-Zpolonius=legacy` datalog implementation.
//!
//! Will be removed in the future, once the in-tree `-Zpolonius=next` implementation reaches feature
//! parity.

use rustc_middle::mir::{Body, LocalKind, Location, START_BLOCK};
use rustc_middle::ty::TyCtxt;
use rustc_mir_dataflow::move_paths::{InitKind, InitLocation, MoveData};
use tracing::debug;

use crate::borrow_set::BorrowSet;
use crate::facts::{AllFacts, PoloniusRegionVid};
use crate::location::LocationTable;
use crate::type_check::free_region_relations::UniversalRegionRelations;

mod loan_invalidations;
mod loan_kills;

/// When requested, emit most of the facts needed by polonius:
/// - moves and assignments
/// - universal regions and their relations
/// - CFG points and edges
/// - loan kills
/// - loan invalidations
///
/// The rest of the facts are emitted during typeck and liveness.
pub(crate) fn emit_facts<'tcx>(
    all_facts: &mut Option<AllFacts>,
    tcx: TyCtxt<'tcx>,
    location_table: &LocationTable,
    body: &Body<'tcx>,
    borrow_set: &BorrowSet<'tcx>,
    move_data: &MoveData<'_>,
    universal_region_relations: &UniversalRegionRelations<'_>,
) {
    let Some(all_facts) = all_facts else {
        // We don't do anything if there are no facts to fill.
        return;
    };
    let _prof_timer = tcx.prof.generic_activity("polonius_fact_generation");
    emit_move_facts(all_facts, move_data, location_table, body);
    emit_universal_region_facts(all_facts, borrow_set, universal_region_relations);
    emit_cfg_and_loan_kills_facts(all_facts, tcx, location_table, body, borrow_set);
    emit_loan_invalidations_facts(all_facts, tcx, location_table, body, borrow_set);
}

/// Emit facts needed for move/init analysis: moves and assignments.
fn emit_move_facts(
    all_facts: &mut AllFacts,
    move_data: &MoveData<'_>,
    location_table: &LocationTable,
    body: &Body<'_>,
) {
    all_facts
        .path_is_var
        .extend(move_data.rev_lookup.iter_locals_enumerated().map(|(l, r)| (r, l)));

    for (child, move_path) in move_data.move_paths.iter_enumerated() {
        if let Some(parent) = move_path.parent {
            all_facts.child_path.push((child, parent));
        }
    }

    let fn_entry_start =
        location_table.start_index(Location { block: START_BLOCK, statement_index: 0 });

    // initialized_at
    for init in move_data.inits.iter() {
        match init.location {
            InitLocation::Statement(location) => {
                let block_data = &body[location.block];
                let is_terminator = location.statement_index == block_data.statements.len();

                if is_terminator && init.kind == InitKind::NonPanicPathOnly {
                    // We are at the terminator of an init that has a panic path,
                    // and where the init should not happen on panic

                    for successor in block_data.terminator().successors() {
                        if body[successor].is_cleanup {
                            continue;
                        }

                        // The initialization happened in (or rather, when arriving at)
                        // the successors, but not in the unwind block.
                        let first_statement = Location { block: successor, statement_index: 0 };
                        all_facts
                            .path_assigned_at_base
                            .push((init.path, location_table.start_index(first_statement)));
                    }
                } else {
                    // In all other cases, the initialization just happens at the
                    // midpoint, like any other effect.
                    all_facts
                        .path_assigned_at_base
                        .push((init.path, location_table.mid_index(location)));
                }
            }
            // Arguments are initialized on function entry
            InitLocation::Argument(local) => {
                assert!(body.local_kind(local) == LocalKind::Arg);
                all_facts.path_assigned_at_base.push((init.path, fn_entry_start));
            }
        }
    }

    for (local, path) in move_data.rev_lookup.iter_locals_enumerated() {
        if body.local_kind(local) != LocalKind::Arg {
            // Non-arguments start out deinitialised; we simulate this with an
            // initial move:
            all_facts.path_moved_at_base.push((path, fn_entry_start));
        }
    }

    // moved_out_at
    // deinitialisation is assumed to always happen!
    all_facts
        .path_moved_at_base
        .extend(move_data.moves.iter().map(|mo| (mo.path, location_table.mid_index(mo.source))));
}

/// Emit universal regions facts, and their relations.
fn emit_universal_region_facts(
    all_facts: &mut AllFacts,
    borrow_set: &BorrowSet<'_>,
    universal_region_relations: &UniversalRegionRelations<'_>,
) {
    // 1: universal regions are modeled in Polonius as a pair:
    // - the universal region vid itself.
    // - a "placeholder loan" associated to this universal region. Since they don't exist in
    //   the `borrow_set`, their `BorrowIndex` are synthesized as the universal region index
    //   added to the existing number of loans, as if they succeeded them in the set.
    //
    let universal_regions = &universal_region_relations.universal_regions;
    all_facts
        .universal_region
        .extend(universal_regions.universal_regions_iter().map(PoloniusRegionVid::from));
    let borrow_count = borrow_set.len();
    debug!(
        "emit_universal_region_facts: polonius placeholders, num_universals={}, borrow_count={}",
        universal_regions.len(),
        borrow_count
    );

    for universal_region in universal_regions.universal_regions_iter() {
        let universal_region_idx = universal_region.index();
        let placeholder_loan_idx = borrow_count + universal_region_idx;
        all_facts.placeholder.push((universal_region.into(), placeholder_loan_idx.into()));
    }

    // 2: the universal region relations `outlives` constraints are emitted as
    //  `known_placeholder_subset` facts.
    for (fr1, fr2) in universal_region_relations.known_outlives() {
        if fr1 != fr2 {
            debug!(
                "emit_universal_region_facts: emitting polonius `known_placeholder_subset` \
                     fr1={:?}, fr2={:?}",
                fr1, fr2
            );
            all_facts.known_placeholder_subset.push((fr1.into(), fr2.into()));
        }
    }
}

/// Emit facts about loan invalidations.
fn emit_loan_invalidations_facts<'tcx>(
    all_facts: &mut AllFacts,
    tcx: TyCtxt<'tcx>,
    location_table: &LocationTable,
    body: &Body<'tcx>,
    borrow_set: &BorrowSet<'tcx>,
) {
    loan_invalidations::emit_loan_invalidations(tcx, all_facts, location_table, body, borrow_set);
}

/// Emit facts about CFG points and edges, as well as locations where loans are killed.
fn emit_cfg_and_loan_kills_facts<'tcx>(
    all_facts: &mut AllFacts,
    tcx: TyCtxt<'tcx>,
    location_table: &LocationTable,
    body: &Body<'tcx>,
    borrow_set: &BorrowSet<'tcx>,
) {
    loan_kills::emit_loan_kills(tcx, all_facts, location_table, body, borrow_set);
}
