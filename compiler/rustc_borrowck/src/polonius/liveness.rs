use rustc_index::IndexVec;
use rustc_index::interval::{IntervalSet, SparseIntervalMatrix};
use rustc_middle::mir::Local;
use rustc_middle::ty::{GenericArg, RegionVid, Ty, TyCtxt};
use rustc_mir_dataflow::points::PointIndex;

use crate::polonius::{ConstraintDirection, LiveRegionVariances};
use crate::region_infer::values::LivenessValues;
use crate::type_check::liveness::LivenessComputation;
use crate::universal_regions::UniversalRegions;

/// The source of liveness information for a given region.
pub(super) trait LivenessSource {
    fn liveness_for_region(&mut self, region: RegionVid) -> RegionLiveness<'_>;
}

/// For a given region, the relevant liveness and variance information.
pub(super) struct RegionLiveness<'a> {
    pub(super) direction: ConstraintDirection,
    live_points: Option<&'a IntervalSet<PointIndex>>,
}

impl<'a> RegionLiveness<'a> {
    #[inline]
    pub(super) fn new<'tcx>(
        region: RegionVid,
        live_region_variances: &LiveRegionVariances,
        universal_regions: &UniversalRegions<'tcx>,
        live_points: &'a SparseIntervalMatrix<RegionVid, PointIndex>,
    ) -> Self {
        // Universal regions propagate loans along the CFG, i.e. forwards only.
        let is_universal_region = universal_regions.is_universal_region(region);

        // Note: there currently are cases related to promoted and const generics, where we don't yet
        // have variance information (possibly about temporary regions created when typeck sanitizes the
        // promoteds). Until that is done, we conservatively fallback to maximizing reachability by
        // adding a bidirectional edge here. This will not limit traversal whatsoever, and thus
        // propagate liveness when needed.
        //
        // FIXME: add the missing variance information and remove this fallback bidirectional edge.
        let direction = if is_universal_region {
            ConstraintDirection::Forward
        } else {
            live_region_variances
                .get(region)
                .copied()
                .flatten()
                .unwrap_or(ConstraintDirection::Bidirectional)
        };
        let live_points = live_points.row(region);
        Self { direction, live_points }
    }

    pub(super) fn is_live_at(&self, point: PointIndex) -> bool {
        self.live_points.map_or(false, |points| points.contains(point))
    }
}

/// The data needed to compute region liveness on-demand while traversing the localized outlives
/// constraint graph to compute loan liveness.
#[derive(Default)]
pub(crate) struct DeferredLocals<'tcx> {
    /// For each region, the local whose liveness is deferred.
    ///
    /// Importantly, because of MIR renumbering, this will always be a 1:1 relationship.
    by_region: IndexVec<RegionVid, Option<Local>>,

    /// For each deferred local, gets the regions contained within that local at use and drop.
    drop_args_by_local: IndexVec<Local, Option<Vec<GenericArg<'tcx>>>>,
}

impl<'tcx> DeferredLocals<'tcx> {
    pub(crate) fn defer_local(
        &mut self,
        tcx: TyCtxt<'tcx>,
        universal_regions: &UniversalRegions<'tcx>,
        local: Local,
        local_ty: Ty<'tcx>,
        dropck_kinds: &[GenericArg<'tcx>],
    ) {
        // We already have drop data for this local, because we need to register
        // region constraints eagerly. So, we'll store this so we don't need to
        // recompute.
        self.drop_args_by_local.insert(local, dropck_kinds.to_vec());

        // Then, we want to map all the regions contained within this local to
        // the local itself. Later, when asked for liveness of a given region,
        // we can trace liveness for the local containing it.
        let by_region = &mut self.by_region;
        tcx.for_each_free_region(&local_ty, |region| {
            // See note in `VarianceExtractor::record_variance`.
            if region.is_bound() || region.is_erased() {
                return;
            }
            let vid = universal_regions.to_region_vid(region);
            // Because of MIR renumbering, we should always have a 1:1 mapping
            // between a region and a local.
            let previous = by_region.insert(vid, local);
            debug_assert!(
                previous.is_none(),
                "{vid:?} is in the type of both {previous:?} and {local:?}, but \
                MIR renumbering should ensure that this is impossible.",
            );
        });
    }

    /// For a given region, compute the liveness for the local containing it, if it is deferred.
    #[inline]
    pub(crate) fn compute_deferred_local(
        &mut self,
        region: RegionVid,
        universal_regions: &UniversalRegions<'tcx>,
        liveness: &mut LivenessValues,
        live_region_variances: &mut LiveRegionVariances,
        comp: &mut LivenessComputation<'_, 'tcx>,
    ) {
        let Some(local) = self.by_region.remove(region) else {
            return;
        };
        let Some(drop_args) = self.drop_args_by_local.remove(local) else {
            return;
        };

        comp.compute(local, universal_regions, Some(live_region_variances), liveness, || {
            &drop_args
        });
    }
}
