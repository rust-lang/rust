use std::collections::BTreeMap;

use rustc_middle::mir::visit::{TyContext, Visitor};
use rustc_middle::mir::{Body, Location, SourceInfo};
use rustc_middle::ty::relate::{self, Relate, RelateResult, TypeRelation};
use rustc_middle::ty::{GenericArgsRef, Region, RegionVid, Ty, TyCtxt, TypeVisitable};
use rustc_middle::{span_bug, ty};

use super::ConstraintDirection;
use crate::RegionInferenceContext;
use crate::universal_regions::UniversalRegions;

/// Some variables are "regular live" at `location` -- i.e., they may be used later. This means that
/// all regions appearing in their type must be live at `location`.
pub(super) fn compute_live_region_variances<'tcx>(
    tcx: TyCtxt<'tcx>,
    regioncx: &RegionInferenceContext<'tcx>,
    body: &Body<'tcx>,
) -> BTreeMap<RegionVid, ConstraintDirection> {
    let mut directions = BTreeMap::new();

    let variance_extractor = VarianceExtractor {
        tcx,
        ambient_variance: ty::Variance::Covariant,
        directions: &mut directions,
        universal_regions: regioncx.universal_regions(),
    };

    let mut visitor = LiveVariablesVisitor { tcx, regioncx, variance_extractor };

    for (bb, data) in body.basic_blocks.iter_enumerated() {
        visitor.visit_basic_block_data(bb, data);
    }

    directions
}

struct LiveVariablesVisitor<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    regioncx: &'a RegionInferenceContext<'tcx>,
    variance_extractor: VarianceExtractor<'a, 'tcx>,
}

impl<'a, 'tcx> Visitor<'tcx> for LiveVariablesVisitor<'a, 'tcx> {
    /// We sometimes have `args` within an rvalue, or within a
    /// call. Make them live at the location where they appear.
    fn visit_args(&mut self, args: &GenericArgsRef<'tcx>, _: Location) {
        self.record_regions_live_at(*args);
        self.super_args(args);
    }

    /// We sometimes have `region`s within an rvalue, or within a
    /// call. Make them live at the location where they appear.
    fn visit_region(&mut self, region: Region<'tcx>, _: Location) {
        self.record_regions_live_at(region);
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
            TyContext::Location(_) => {
                self.record_regions_live_at(ty);
            }
        }

        self.super_ty(ty);
    }
}

impl<'a, 'tcx> LiveVariablesVisitor<'a, 'tcx> {
    /// Some variable is "regular live" at `location` -- i.e., it may be used later. This means that
    /// all regions appearing in the type of `value` must be live at `location`.
    fn record_regions_live_at<T>(&mut self, value: T)
    where
        T: TypeVisitable<TyCtxt<'tcx>> + Relate<TyCtxt<'tcx>>,
    {
        self.variance_extractor
            .relate(value, value)
            .expect("Can't have a type error relating to itself");
    }
}

/// Extracts variances for regions contained within types. Follows the same structure as
/// `rustc_infer`'s `Generalizer`: we try to relate a type with itself to track and extract the
/// variances of regions.
pub(super) struct VarianceExtractor<'a, 'tcx> {
    pub tcx: TyCtxt<'tcx>,
    pub ambient_variance: ty::Variance,
    pub directions: &'a mut BTreeMap<RegionVid, ConstraintDirection>,
    pub universal_regions: &'a UniversalRegions<'tcx>,
}

impl<'tcx> VarianceExtractor<'_, 'tcx> {
    fn record_variance(&mut self, region: ty::Region<'tcx>, variance: ty::Variance) {
        // We're only interested in the variance of vars and free regions.
        //
        // Note: even if we currently bail for two cases of unexpected region kinds here, missing
        // variance data is not a soundness problem: the regions with missing variance will still be
        // present in the constraint graph as they are live, and liveness edges construction has a
        // fallback for this case.
        //
        // FIXME: that being said, we need to investigate these cases better to not ignore regions
        // in general.
        if region.is_bound() {
            // We ignore these because they cannot be turned into the vids we need.
            return;
        }

        if region.is_erased() {
            // These cannot be turned into a vid either, and we also ignore them: the fact that they
            // show up here looks like either an issue upstream or a combination with unexpectedly
            // continuing compilation too far when we're in a tainted by errors situation.
            //
            // FIXME: investigate the `generic_const_exprs` test that triggers this issue,
            // `ui/const-generics/generic_const_exprs/issue-97047-ice-2.rs`
            return;
        }

        let direction = match variance {
            ty::Covariant => ConstraintDirection::Forward,
            ty::Contravariant => ConstraintDirection::Backward,
            ty::Invariant => ConstraintDirection::Bidirectional,
            ty::Bivariant => {
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
