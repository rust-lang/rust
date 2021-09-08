use rustc_data_structures::fx::FxHashSet;
use rustc_middle::mir::{Body, Local};
use rustc_middle::ty::{RegionVid, TyCtxt};
use std::rc::Rc;

use rustc_mir_dataflow::impls::MaybeInitializedPlaces;
use rustc_mir_dataflow::move_paths::MoveData;
use rustc_mir_dataflow::ResultsCursor;

use crate::{
    constraints::OutlivesConstraintSet,
    facts::{AllFacts, AllFactsExt},
    location::LocationTable,
    nll::ToRegionVid,
    region_infer::values::RegionValueElements,
    universal_regions::UniversalRegions,
};

use super::TypeChecker;

mod local_use_map;
mod polonius;
mod trace;

/// Combines liveness analysis with initialization analysis to
/// determine which variables are live at which points, both due to
/// ordinary uses and drops. Returns a set of (ty, location) pairs
/// that indicate which types must be live at which point in the CFG.
/// This vector is consumed by `constraint_generation`.
///
/// N.B., this computation requires normalization; therefore, it must be
/// performed before
pub(super) fn generate<'mir, 'tcx>(
    typeck: &mut TypeChecker<'_, 'tcx>,
    body: &Body<'tcx>,
    elements: &Rc<RegionValueElements>,
    flow_inits: &mut ResultsCursor<'mir, 'tcx, MaybeInitializedPlaces<'mir, 'tcx>>,
    move_data: &MoveData<'tcx>,
    location_table: &LocationTable,
) {
    debug!("liveness::generate");

    let free_regions = regions_that_outlive_free_regions(
        typeck.infcx.num_region_vars(),
        &typeck.borrowck_context.universal_regions,
        &typeck.borrowck_context.constraints.outlives_constraints,
    );
    let live_locals = compute_live_locals(typeck.tcx(), &free_regions, &body);
    let facts_enabled = AllFacts::enabled(typeck.tcx());

    let polonius_drop_used = if facts_enabled {
        let mut drop_used = Vec::new();
        polonius::populate_access_facts(typeck, body, location_table, move_data, &mut drop_used);
        Some(drop_used)
    } else {
        None
    };

    if !live_locals.is_empty() || facts_enabled {
        trace::trace(
            typeck,
            body,
            elements,
            flow_inits,
            move_data,
            live_locals,
            polonius_drop_used,
        );
    }
}

// The purpose of `compute_live_locals` is to define the subset of `Local`
// variables for which we need to do a liveness computation. We only need
// to compute whether a variable `X` is live if that variable contains
// some region `R` in its type where `R` is not known to outlive a free
// region (i.e., where `R` may be valid for just a subset of the fn body).
fn compute_live_locals(
    tcx: TyCtxt<'tcx>,
    free_regions: &FxHashSet<RegionVid>,
    body: &Body<'tcx>,
) -> Vec<Local> {
    let live_locals: Vec<Local> = body
        .local_decls
        .iter_enumerated()
        .filter_map(|(local, local_decl)| {
            if tcx.all_free_regions_meet(&local_decl.ty, |r| {
                free_regions.contains(&r.to_region_vid())
            }) {
                None
            } else {
                Some(local)
            }
        })
        .collect();

    debug!("{} total variables", body.local_decls.len());
    debug!("{} variables need liveness", live_locals.len());
    debug!("{} regions outlive free regions", free_regions.len());

    live_locals
}

/// Computes all regions that are (currently) known to outlive free
/// regions. For these regions, we do not need to compute
/// liveness, since the outlives constraints will ensure that they
/// are live over the whole fn body anyhow.
fn regions_that_outlive_free_regions(
    num_region_vars: usize,
    universal_regions: &UniversalRegions<'tcx>,
    constraint_set: &OutlivesConstraintSet<'tcx>,
) -> FxHashSet<RegionVid> {
    // Build a graph of the outlives constraints thus far. This is
    // a reverse graph, so for each constraint `R1: R2` we have an
    // edge `R2 -> R1`. Therefore, if we find all regions
    // reachable from each free region, we will have all the
    // regions that are forced to outlive some free region.
    let rev_constraint_graph = constraint_set.reverse_graph(num_region_vars);
    let fr_static = universal_regions.fr_static;
    let rev_region_graph = rev_constraint_graph.region_graph(constraint_set, fr_static);

    // Stack for the depth-first search. Start out with all the free regions.
    let mut stack: Vec<_> = universal_regions.universal_regions().collect();

    // Set of all free regions, plus anything that outlives them. Initially
    // just contains the free regions.
    let mut outlives_free_region: FxHashSet<_> = stack.iter().cloned().collect();

    // Do the DFS -- for each thing in the stack, find all things
    // that outlive it and add them to the set. If they are not,
    // push them onto the stack for later.
    while let Some(sub_region) = stack.pop() {
        stack.extend(
            rev_region_graph
                .outgoing_regions(sub_region)
                .filter(|&r| outlives_free_region.insert(r)),
        );
    }

    // Return the final set of things we visited.
    outlives_free_region
}
