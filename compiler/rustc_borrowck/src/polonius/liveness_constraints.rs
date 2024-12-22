use std::collections::BTreeMap;

use rustc_middle::ty::relate::{self, Relate, RelateResult, TypeRelation};
use rustc_middle::ty::{self, RegionVid, Ty, TyCtxt, TypeVisitable};

use super::{ConstraintDirection, PoloniusContext};
use crate::universal_regions::UniversalRegions;

impl PoloniusContext {
    /// Record the variance of each region contained within the given value.
    pub(crate) fn record_live_region_variance<'tcx>(
        &mut self,
        tcx: TyCtxt<'tcx>,
        universal_regions: &UniversalRegions<'tcx>,
        value: impl TypeVisitable<TyCtxt<'tcx>> + Relate<TyCtxt<'tcx>>,
    ) {
        let mut extractor = VarianceExtractor {
            tcx,
            ambient_variance: ty::Variance::Covariant,
            directions: &mut self.live_region_variances,
            universal_regions,
        };
        extractor.relate(value, value).expect("Can't have a type error relating to itself");
    }
}

/// Extracts variances for regions contained within types. Follows the same structure as
/// `rustc_infer`'s `Generalizer`: we try to relate a type with itself to track and extract the
/// variances of regions.
struct VarianceExtractor<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    ambient_variance: ty::Variance,
    directions: &'a mut BTreeMap<RegionVid, ConstraintDirection>,
    universal_regions: &'a UniversalRegions<'tcx>,
}

impl<'tcx> VarianceExtractor<'_, 'tcx> {
    fn record_variance(&mut self, region: ty::Region<'tcx>, variance: ty::Variance) {
        // We're only interested in the variance of vars and free regions.

        if region.is_bound() || region.is_erased() {
            // ignore these
            return;
        }

        let direction = match variance {
            ty::Variance::Covariant => ConstraintDirection::Forward,
            ty::Variance::Contravariant => ConstraintDirection::Backward,
            ty::Variance::Invariant => ConstraintDirection::Bidirectional,
            ty::Variance::Bivariant => {
                // We don't add edges for bivariant cases.
                return;
            }
        };

        let region = self.universal_regions.to_region_vid(region);
        self.directions
            .entry(region)
            .and_modify(|entry| {
                // If there's already a recorded direction for this region, we combine the two:
                // - combining the same direction is idempotent
                // - combining different directions is trivially bidirectional
                if entry != &direction {
                    *entry = ConstraintDirection::Bidirectional;
                }
            })
            .or_insert(direction);
    }
}

impl<'tcx> TypeRelation<TyCtxt<'tcx>> for VarianceExtractor<'_, 'tcx> {
    fn cx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn relate_with_variance<T: Relate<TyCtxt<'tcx>>>(
        &mut self,
        variance: ty::Variance,
        _info: ty::VarianceDiagInfo<TyCtxt<'tcx>>,
        a: T,
        b: T,
    ) -> RelateResult<'tcx, T> {
        let old_ambient_variance = self.ambient_variance;
        self.ambient_variance = self.ambient_variance.xform(variance);
        let r = self.relate(a, b)?;
        self.ambient_variance = old_ambient_variance;
        Ok(r)
    }

    fn tys(&mut self, a: Ty<'tcx>, b: Ty<'tcx>) -> RelateResult<'tcx, Ty<'tcx>> {
        assert_eq!(a, b); // we are misusing TypeRelation here; both LHS and RHS ought to be ==
        relate::structurally_relate_tys(self, a, b)
    }

    fn regions(
        &mut self,
        a: ty::Region<'tcx>,
        b: ty::Region<'tcx>,
    ) -> RelateResult<'tcx, ty::Region<'tcx>> {
        assert_eq!(a, b); // we are misusing TypeRelation here; both LHS and RHS ought to be ==
        self.record_variance(a, self.ambient_variance);
        Ok(a)
    }

    fn consts(
        &mut self,
        a: ty::Const<'tcx>,
        b: ty::Const<'tcx>,
    ) -> RelateResult<'tcx, ty::Const<'tcx>> {
        assert_eq!(a, b); // we are misusing TypeRelation here; both LHS and RHS ought to be ==
        relate::structurally_relate_consts(self, a, b)
    }

    fn binders<T>(
        &mut self,
        a: ty::Binder<'tcx, T>,
        _: ty::Binder<'tcx, T>,
    ) -> RelateResult<'tcx, ty::Binder<'tcx, T>>
    where
        T: Relate<TyCtxt<'tcx>>,
    {
        self.relate(a.skip_binder(), a.skip_binder())?;
        Ok(a)
    }
}
