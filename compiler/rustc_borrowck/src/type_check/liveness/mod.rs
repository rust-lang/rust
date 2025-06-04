use itertools::{Either, Itertools};
use rustc_data_structures::fx::FxHashSet;
use rustc_middle::mir::visit::{TyContext, Visitor};
use rustc_middle::mir::{Body, Local, Location, SourceInfo};
use rustc_middle::span_bug;
use rustc_middle::ty::relate::Relate;
use rustc_middle::ty::{GenericArgsRef, Region, RegionVid, Ty, TyCtxt, TypeVisitable};
use rustc_mir_dataflow::move_paths::MoveData;
use rustc_mir_dataflow::points::DenseLocationMap;
use tracing::debug;

use super::TypeChecker;
use crate::constraints::OutlivesConstraintSet;
use crate::polonius::PoloniusLivenessContext;
use crate::region_infer::values::LivenessValues;
use crate::universal_regions::UniversalRegions;

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
pub(super) fn generate<'tcx>(
    typeck: &mut TypeChecker<'_, 'tcx>,
    location_map: &DenseLocationMap,
    move_data: &MoveData<'tcx>,
) {
    debug!("liveness::generate");

    let mut free_regions = regions_that_outlive_free_regions(
        typeck.infcx.num_region_vars(),
        &typeck.universal_regions,
        &typeck.constraints.outlives_constraints,
    );

    // NLLs can avoid computing some liveness data here because its constraints are
    // location-insensitive, but that doesn't work in polonius: locals whose type contains a region
    // that outlives a free region are not necessarily live everywhere in a flow-sensitive setting,
    // unlike NLLs.
    // We do record these regions in the polonius context, since they're used to differentiate
    // relevant and boring locals, which is a key distinction used later in diagnostics.
    if typeck.tcx().sess.opts.unstable_opts.polonius.is_next_enabled() {
        let (_, boring_locals) =
            compute_relevant_live_locals(typeck.tcx(), &free_regions, typeck.body);
        typeck.polonius_liveness.as_mut().unwrap().boring_nll_locals =
            boring_locals.into_iter().collect();
        free_regions = typeck.universal_regions.universal_regions_iter().collect();
    }
    let (relevant_live_locals, boring_locals) =
        compute_relevant_live_locals(typeck.tcx(), &free_regions, typeck.body);

    trace::trace(typeck, location_map, move_data, relevant_live_locals, boring_locals);

    // Mark regions that should be live where they appear within rvalues or within a call: like
    // args, regions, and types.
    record_regular_live_regions(
        typeck.tcx(),
        &mut typeck.constraints.liveness_constraints,
        &typeck.universal_regions,
        &mut typeck.polonius_liveness,
        typeck.body,
    );
}

// The purpose of `compute_relevant_live_locals` is to define the subset of `Local`
// variables for which we need to do a liveness computation. We only need
// to compute whether a variable `X` is live if that variable contains
// some region `R` in its type where `R` is not known to outlive a free
// region (i.e., where `R` may be valid for just a subset of the fn body).
fn compute_relevant_live_locals<'tcx>(
    tcx: TyCtxt<'tcx>,
    free_regions: &FxHashSet<RegionVid>,
    body: &Body<'tcx>,
) -> (Vec<Local>, Vec<Local>) {
    let (boring_locals, relevant_live_locals): (Vec<_>, Vec<_>) =
        body.local_decls.iter_enumerated().partition_map(|(local, local_decl)| {
            if tcx.all_free_regions_meet(&local_decl.ty, |r| free_regions.contains(&r.as_var())) {
                Either::Left(local)
            } else {
                Either::Right(local)
            }
        });

    debug!("{} total variables", body.local_decls.len());
    debug!("{} variables need liveness", relevant_live_locals.len());
    debug!("{} regions outlive free regions", free_regions.len());

    (relevant_live_locals, boring_locals)
}

/// Computes all regions that are (currently) known to outlive free
/// regions. For these regions, we do not need to compute
/// liveness, since the outlives constraints will ensure that they
/// are live over the whole fn body anyhow.
fn regions_that_outlive_free_regions<'tcx>(
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
    let mut stack: Vec<_> = universal_regions.universal_regions_iter().collect();

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

/// Some variables are "regular live" at `location` -- i.e., they may be used later. This means that
/// all regions appearing in their type must be live at `location`.
fn record_regular_live_regions<'tcx>(
    tcx: TyCtxt<'tcx>,
    liveness_constraints: &mut LivenessValues,
    universal_regions: &UniversalRegions<'tcx>,
    polonius_liveness: &mut Option<PoloniusLivenessContext>,
    body: &Body<'tcx>,
) {
    let mut visitor =
        LiveVariablesVisitor { tcx, liveness_constraints, universal_regions, polonius_liveness };
    for (bb, data) in body.basic_blocks.iter_enumerated() {
        visitor.visit_basic_block_data(bb, data);
    }
}

/// Visitor looking for regions that should be live within rvalues or calls.
struct LiveVariablesVisitor<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    liveness_constraints: &'a mut LivenessValues,
    universal_regions: &'a UniversalRegions<'tcx>,
    polonius_liveness: &'a mut Option<PoloniusLivenessContext>,
}

impl<'a, 'tcx> Visitor<'tcx> for LiveVariablesVisitor<'a, 'tcx> {
    /// We sometimes have `args` within an rvalue, or within a
    /// call. Make them live at the location where they appear.
    fn visit_args(&mut self, args: &GenericArgsRef<'tcx>, location: Location) {
        self.record_regions_live_at(*args, location);
        self.super_args(args);
    }

    /// We sometimes have `region`s within an rvalue, or within a
    /// call. Make them live at the location where they appear.
    fn visit_region(&mut self, region: Region<'tcx>, location: Location) {
        self.record_regions_live_at(region, location);
        self.super_region(region);
    }

    /// We sometimes have `ty`s within an rvalue, or within a
    /// call. Make them live at the location where they appear.
    fn visit_ty(&mut self, ty: Ty<'tcx>, ty_context: TyContext) {
        match ty_context {
            TyContext::ReturnTy(SourceInfo { span, .. })
            | TyContext::YieldTy(SourceInfo { span, .. })
            | TyContext::ResumeTy(SourceInfo { span, .. })
            | TyContext::UserTy(span)
            | TyContext::LocalDecl { source_info: SourceInfo { span, .. }, .. } => {
                span_bug!(span, "should not be visiting outside of the CFG: {:?}", ty_context);
            }
            TyContext::Location(location) => {
                self.record_regions_live_at(ty, location);
            }
        }

        self.super_ty(ty);
    }
}

impl<'a, 'tcx> LiveVariablesVisitor<'a, 'tcx> {
    /// Some variable is "regular live" at `location` -- i.e., it may be used later. This means that
    /// all regions appearing in the type of `value` must be live at `location`.
    fn record_regions_live_at<T>(&mut self, value: T, location: Location)
    where
        T: TypeVisitable<TyCtxt<'tcx>> + Relate<TyCtxt<'tcx>>,
    {
        debug!("record_regions_live_at(value={:?}, location={:?})", value, location);
        self.tcx.for_each_free_region(&value, |live_region| {
            let live_region_vid = live_region.as_var();
            self.liveness_constraints.add_location(live_region_vid, location);
        });

        // When using `-Zpolonius=next`, we record the variance of each live region.
        if let Some(polonius_liveness) = self.polonius_liveness {
            polonius_liveness.record_live_region_variance(self.tcx, self.universal_regions, value);
        }
    }
}
