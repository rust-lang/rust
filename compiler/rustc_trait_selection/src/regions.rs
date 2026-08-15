use rustc_data_structures::fx::FxIndexSet;
use rustc_hir::def_id::LocalDefId;
use rustc_infer::infer::outlives::env::OutlivesEnvironment;
use rustc_infer::infer::{
    InferCtxt, RegionResolutionError, SubregionOrigin, TyCtxtInferExt, TypeOutlivesConstraint,
};
use rustc_macros::extension;
use rustc_middle::ty::{self, Ty, TyCtxt, TypeVisitableExt, TypingMode, elaborate};
use rustc_span::DUMMY_SP;

use crate::traits::outlives_bounds::InferCtxtExt;

#[extension(pub trait OutlivesEnvironmentBuildExt<'tcx>)]
impl<'tcx> OutlivesEnvironment<'tcx> {
    fn new(
        infcx: &InferCtxt<'tcx>,
        body_def_id: LocalDefId,
        param_env: ty::ParamEnv<'tcx>,
        assumed_wf_tys: impl IntoIterator<Item = Ty<'tcx>>,
    ) -> Self {
        Self::new_with_implied_bounds_compat(infcx, body_def_id, param_env, assumed_wf_tys, false)
    }

    fn new_with_implied_bounds_compat(
        infcx: &InferCtxt<'tcx>,
        body_def_id: LocalDefId,
        param_env: ty::ParamEnv<'tcx>,
        assumed_wf_tys: impl IntoIterator<Item = Ty<'tcx>>,
        disable_implied_bounds_hack: bool,
    ) -> Self {
        let mut bounds = vec![];

        for bound in param_env.caller_bounds() {
            if let Some(type_outlives) = bound.as_type_outlives_clause() {
                debug_assert!(!infcx.next_trait_solver() || !type_outlives.has_non_rigid_aliases());
                bounds.push(type_outlives);
            }
        }

        // FIXME(-Znext-trait-solver): Normalize these.
        let higher_ranked_assumptions = infcx.take_registered_region_assumptions();
        let higher_ranked_assumptions =
            elaborate::elaborate_outlives_assumptions(infcx.tcx, higher_ranked_assumptions);

        // FIXME: This needs to be modified so that we normalize the known type
        // outlives obligations then elaborate them into their region/type components.
        // Otherwise, `<W<'a> as Mirror>::Assoc: 'b` will not imply `'a: 'b` even
        // if we can normalize `'a`.
        OutlivesEnvironment::from_normalized_bounds(
            param_env,
            bounds,
            infcx.implied_bounds_tys(
                body_def_id,
                param_env,
                assumed_wf_tys,
                disable_implied_bounds_hack,
            ),
            higher_ranked_assumptions,
        )
    }
}

#[extension(pub trait InferCtxtRegionExt<'tcx>)]
impl<'tcx> InferCtxt<'tcx> {
    /// Resolve regions lexically.
    ///
    /// This function assumes that all infer variables are already constrained.
    ///
    /// FIXME(#155345): this can probably be moved back to `rustc_infer` now that normalization is
    /// no longer required. These two extension traits won't be needed then.
    fn resolve_regions(
        &self,
        body_def_id: LocalDefId,
        param_env: ty::ParamEnv<'tcx>,
        assumed_wf_tys: impl IntoIterator<Item = Ty<'tcx>>,
    ) -> Vec<RegionResolutionError<'tcx>> {
        self.resolve_regions_with_outlives_env(
            &OutlivesEnvironment::new(self, body_def_id, param_env, assumed_wf_tys),
            self.tcx.def_span(body_def_id),
        )
    }
}

/// Given a known `param_env` and a set of well formed types, can we prove that
/// `ty` outlives `region`.
pub fn ty_known_to_outlive<'tcx>(
    tcx: TyCtxt<'tcx>,
    id: LocalDefId,
    param_env: ty::ParamEnv<'tcx>,
    wf_tys: &FxIndexSet<Ty<'tcx>>,
    ty: Ty<'tcx>,
    region: ty::Region<'tcx>,
) -> bool {
    test_region_obligations(tcx, id, param_env, wf_tys, |infcx| {
        infcx.register_type_outlives_constraint_inner(TypeOutlivesConstraint {
            sub_region: region,
            sup_type: ty,
            origin: SubregionOrigin::RelateParamBound(DUMMY_SP, ty, None),
        });
    })
}

/// Given a known `param_env` and a set of well formed types, can we prove that
/// `region_a` outlives `region_b`
pub fn region_known_to_outlive<'tcx>(
    tcx: TyCtxt<'tcx>,
    id: LocalDefId,
    param_env: ty::ParamEnv<'tcx>,
    wf_tys: &FxIndexSet<Ty<'tcx>>,
    region_a: ty::Region<'tcx>,
    region_b: ty::Region<'tcx>,
) -> bool {
    test_region_obligations(tcx, id, param_env, wf_tys, |infcx| {
        infcx.sub_regions(
            SubregionOrigin::RelateRegionParamBound(DUMMY_SP, None),
            region_b,
            region_a,
            ty::VisibleForLeakCheck::Unreachable,
        );
    })
}

/// Given a known `param_env` and a set of well formed types, set up an
/// `InferCtxt`, call the passed function (to e.g. set up region constraints
/// to be tested), then resolve region and return errors
pub fn test_region_obligations<'tcx>(
    tcx: TyCtxt<'tcx>,
    id: LocalDefId,
    param_env: ty::ParamEnv<'tcx>,
    wf_tys: &FxIndexSet<Ty<'tcx>>,
    add_constraints: impl FnOnce(&InferCtxt<'tcx>),
) -> bool {
    // Unfortunately, we have to use a new `InferCtxt` each call, because
    // region constraints get added and solved there and we need to test each
    // call individually.
    let infcx = tcx.infer_ctxt().build(TypingMode::non_body_analysis());

    add_constraints(&infcx);

    let errors = infcx.resolve_regions(id, param_env, wf_tys.iter().copied());
    tracing::debug!(?errors, "errors");

    // If we were able to prove that the type outlives the region without
    // an error, it must be because of the implied or explicit bounds...
    errors.is_empty()
}
