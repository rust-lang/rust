use rustc_data_structures::fx::FxHashMap;
use rustc_middle::mir::Local;
use rustc_middle::ty::{GenericArg, RegionVid, Ty};

use crate::BorrowckInferCtxt;
use crate::universal_regions::UniversalRegions;

#[derive(Default)]
pub(crate) struct DeferredLocals<'tcx> {
    /// For each region, the local whose liveness is deferred.
    ///
    /// Importantly, because of MIR renumbering, this will always be a 1:1 relationship.
    by_region: FxHashMap<RegionVid, Local>,

    /// For each deferred local, gets the regions contained within that local at use and drop.
    drop_args_by_local: FxHashMap<Local, Vec<GenericArg<'tcx>>>,
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

    /// For a given region, return the local whose liveness is deferred, and
    /// the regions within that local at use and drop.
    pub(crate) fn use_deferred_local(
        &mut self,
        region: RegionVid,
    ) -> Option<(Local, Vec<GenericArg<'tcx>>)> {
        let local = *self.by_region.get(&region)?;
        let drop_args = self.drop_args_by_local.remove(&local)?;
        Some((local, drop_args))
    }
}
