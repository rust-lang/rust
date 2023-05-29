//! Provider for the `implied_outlives_bounds` query.
//! Do not call this query directory. See
//! [`rustc_trait_selection::traits::query::type_op::implied_outlives_bounds`].

use rustc_infer::infer::canonical::{self, Canonical};
use rustc_infer::infer::TyCtxtInferExt;
use rustc_infer::traits::query::OutlivesBound;
use rustc_middle::query::Providers;
use rustc_middle::ty::TyCtxt;
use rustc_trait_selection::infer::InferCtxtBuilderExt;
use rustc_trait_selection::traits::query::type_op::implied_outlives_bounds::compute_implied_outlives_bounds_inner;
use rustc_trait_selection::traits::query::{CanonicalTyGoal, NoSolution};

pub(crate) fn provide(p: &mut Providers) {
    *p = Providers { implied_outlives_bounds, ..*p };
}

fn implied_outlives_bounds<'tcx>(
    tcx: TyCtxt<'tcx>,
    goal: CanonicalTyGoal<'tcx>,
) -> Result<
    &'tcx Canonical<'tcx, canonical::QueryResponse<'tcx, Vec<OutlivesBound<'tcx>>>>,
    NoSolution,
> {
    tcx.infer_ctxt().enter_canonical_trait_query(&goal, |ocx, key| {
        let (param_env, ty) = key.into_parts();
        compute_implied_outlives_bounds_inner(ocx, param_env, ty)
    })
}
