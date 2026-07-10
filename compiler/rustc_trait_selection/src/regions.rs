use rustc_data_structures::fx::FxIndexSet;
use rustc_hir::def_id::LocalDefId;
use rustc_infer::infer::outlives::env::OutlivesEnvironment;
use rustc_infer::infer::{
    InferCtxt, RegionResolutionError, SubregionOrigin, TyCtxtInferExt, TypeOutlivesConstraint,
};
use rustc_macros::extension;
use rustc_middle::traits::ObligationCause;
use rustc_middle::traits::query::NoSolution;
use rustc_middle::ty::{self, Ty, TyCtxt, TypingMode, Unnormalized, elaborate};
use rustc_span::{DUMMY_SP, Span};

use crate::traits::ScrubbedTraitError;
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
            if let Some(mut type_outlives) = bound.as_type_outlives_clause() {
                if infcx.next_trait_solver() {
                    match crate::solve::deeply_normalize::<_, ScrubbedTraitError<'tcx>>(
                        infcx.at(&ObligationCause::dummy(), param_env),
                        Unnormalized::new_wip(type_outlives),
                    ) {
                        Ok(new) => type_outlives = new,
                        Err(_) => {
                            infcx.dcx().delayed_bug(format!("could not normalize `{bound}`"));
                        }
                    }
                }
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
    /// Resolve regions, using the deep normalizer to normalize any type-outlives
    /// obligations in the process. This is in `rustc_trait_selection` because
    /// we need to normalize.
    ///
    /// Prefer this method over `resolve_regions_with_normalize`, unless you are
    /// doing something specific for normalization.
    ///
    /// This function assumes that all infer variables are already constrained.
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

    /// Don't call this directly unless you know what you're doing.
    fn resolve_regions_with_outlives_env(
        &self,
        outlives_env: &OutlivesEnvironment<'tcx>,
        span: Span,
    ) -> Vec<RegionResolutionError<'tcx>> {
        self.resolve_regions_with_normalize(
            &outlives_env,
            |ty, origin| {
                let ty = self.resolve_vars_if_possible(ty);

                if self.next_trait_solver() {
                    crate::solve::deeply_normalize(
                        self.at(
                            &ObligationCause::dummy_with_span(origin.span()),
                            outlives_env.param_env,
                        ),
                        Unnormalized::new_wip(ty),
                    )
                    .map_err(|_: Vec<ScrubbedTraitError<'tcx>>| NoSolution)
                } else {
                    Ok(ty)
                }
            },
            span,
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
