use crate::borrow_check::location::LocationTable;
use crate::borrow_check::nll::region_infer::values::RegionValueElements;
use crate::borrow_check::nll::constraints::ConstraintSet;
use crate::borrow_check::nll::NllLivenessMap;
use crate::borrow_check::nll::universal_regions::UniversalRegions;
use crate::dataflow::move_paths::MoveData;
use crate::dataflow::MaybeInitializedPlaces;
use crate::dataflow::FlowAtLocation;
use rustc::mir::Mir;
use rustc::ty::RegionVid;
use rustc_data_structures::fx::FxHashSet;
use std::rc::Rc;

use super::TypeChecker;

crate mod liveness_map;
mod local_use_map;
mod trace;

/// Combines liveness analysis with initialization analysis to
/// determine which variables are live at which points, both due to
/// ordinary uses and drops. Returns a set of (ty, location) pairs
/// that indicate which types must be live at which point in the CFG.
/// This vector is consumed by `constraint_generation`.
///
/// N.B., this computation requires normalization; therefore, it must be
/// performed before
pub(super) fn generate<'gcx, 'tcx>(
    typeck: &mut TypeChecker<'_, 'gcx, 'tcx>,
    mir: &Mir<'tcx>,
    elements: &Rc<RegionValueElements>,
    flow_inits: &mut FlowAtLocation<'tcx, MaybeInitializedPlaces<'_, 'gcx, 'tcx>>,
    move_data: &MoveData<'tcx>,
    location_table: &LocationTable,
) {
    debug!("liveness::generate");
    let free_regions = {
        let borrowck_context = typeck.borrowck_context.as_ref().unwrap();
        regions_that_outlive_free_regions(
            typeck.infcx.num_region_vars(),
            &borrowck_context.universal_regions,
            &borrowck_context.constraints.outlives_constraints,
        )
    };
    let liveness_map = NllLivenessMap::compute(typeck.tcx(), &free_regions, mir);
    trace::trace(typeck, mir, elements, flow_inits, move_data, &liveness_map, location_table);
}

/// Computes all regions that are (currently) known to outlive free
/// regions. For these regions, we do not need to compute
/// liveness, since the outlives constraints will ensure that they
/// are live over the whole fn body anyhow.
fn regions_that_outlive_free_regions(
    num_region_vars: usize,
    universal_regions: &UniversalRegions<'tcx>,
    constraint_set: &ConstraintSet,
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
