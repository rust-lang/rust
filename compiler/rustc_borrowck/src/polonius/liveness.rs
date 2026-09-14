use rustc_index::IndexVec;
use rustc_middle::mir::Local;
use rustc_middle::ty::{GenericArg, RegionVid, Ty};
use rustc_mir_dataflow::points::PointIndex;

use crate::BorrowckInferCtxt;
use crate::polonius::{ConstraintDirection, LiveRegionVariances};
use crate::region_infer::values::LivenessValues;
use crate::type_check::liveness::LivenessComputation;
use crate::universal_regions::UniversalRegions;

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
        infcx: &BorrowckInferCtxt<'tcx>,
        universal_regions: &UniversalRegions<'tcx>,
        local: Local,
        local_ty: Ty<'tcx>,
        dropck_kinds: &[GenericArg<'tcx>],
    ) {
        let tcx = infcx.tcx;

        // We already have drop data for this local, because we need to register
        // region constraints eagerly. So, we'll store this so we don't need to
        // recompute.
        self.drop_args_by_local.insert(local, dropck_kinds.to_vec());

        // Then, we want to map all the regions contained within this local to
        // the local itself. Later, when asked for liveness of a given region,
        // we can trace liveness for the local containing it.
        let by_region = &mut self.by_region;
        tcx.for_each_free_region(&local_ty, |region| {
            // See note in [`VarianceExtractor::record_variance`].
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

    /// For a given region, compute the liveness for the local containing it, if if is deferred.
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

/// For a given region, the relevant liveness and variance information.
pub(super) struct RegionLiveness<'a> {
    region: RegionVid,
    pub(super) direction: ConstraintDirection,
    liveness: &'a LivenessValues,
}

impl<'a> RegionLiveness<'a> {
    pub(super) fn new<'tcx>(
        region: RegionVid,
        live_region_variances: &LiveRegionVariances,
        universal_regions: &UniversalRegions<'tcx>,
        liveness: &'a LivenessValues,
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
        Self { region, direction, liveness }
    }

    pub(super) fn is_live_at(&self, point: PointIndex) -> bool {
        self.liveness.points().contains(self.region, point)
    }
}
