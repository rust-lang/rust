//! Provider for the `implied_outlives_bounds` query.
//! Do not call this query directory. See
//! [`rustc_trait_selection::traits::query::type_op::implied_outlives_bounds`].

use rustc_infer::infer::TyCtxtInferExt;
use rustc_infer::infer::canonical::{self, Canonical};
use rustc_infer::traits::query::OutlivesBound;
use rustc_infer::traits::query::type_op::ImpliedOutlivesBounds;
use rustc_middle::query::Providers;
use rustc_middle::ty::{ParamEnvAnd, TyCtxt};
use rustc_span::DUMMY_SP;
use rustc_trait_selection::infer::InferCtxtBuilderExt;
use rustc_trait_selection::traits::query::type_op::implied_outlives_bounds::compute_implied_outlives_bounds_inner;
use rustc_trait_selection::traits::query::{CanonicalImpliedOutlivesBoundsGoal, NoSolution};

pub(crate) fn provide(p: &mut Providers) {
    *p = Providers { implied_outlives_bounds, ..*p };
}

fn implied_outlives_bounds<'tcx>(
    tcx: TyCtxt<'tcx>,
    (goal, disable_implied_bounds_hack): (CanonicalImpliedOutlivesBoundsGoal<'tcx>, bool),
) -> Result<
    &'tcx Canonical<'tcx, canonical::QueryResponse<'tcx, Vec<OutlivesBound<'tcx>>>>,
    NoSolution,
> {
    tcx.infer_ctxt().enter_canonical_trait_query(&goal, |ocx, key| {
        let ParamEnvAnd { param_env, value: ImpliedOutlivesBounds { ty } } = key;
        compute_implied_outlives_bounds_inner(
            ocx,
            param_env,
            ty,
            DUMMY_SP,
            disable_implied_bounds_hack,
        )
    })
}
