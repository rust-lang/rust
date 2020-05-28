//! Various code related to computing outlives relations.

pub mod env;
pub mod obligations;
pub mod verify;

use rustc_middle::traits::query::OutlivesBound;
use rustc_middle::ty;

pub fn explicit_outlives_bounds<'tcx>(
    param_env: ty::ParamEnv<'tcx>,
) -> impl Iterator<Item = OutlivesBound<'tcx>> + 'tcx {
    debug!("explicit_outlives_bounds()");
    param_env.caller_bounds.into_iter().filter_map(move |predicate| match predicate.kind() {
        ty::PredicateKind::Projection(..)
        | ty::PredicateKind::Trait(..)
        | ty::PredicateKind::Subtype(..)
        | ty::PredicateKind::WellFormed(..)
        | ty::PredicateKind::ObjectSafe(..)
        | ty::PredicateKind::ClosureKind(..)
        | ty::PredicateKind::TypeOutlives(..)
        | ty::PredicateKind::ConstEvaluatable(..)
        | ty::PredicateKind::ConstEquate(..) => None,
        ty::PredicateKind::RegionOutlives(ref data) => data
            .no_bound_vars()
            .map(|ty::OutlivesPredicate(r_a, r_b)| OutlivesBound::RegionSubRegion(r_b, r_a)),
    })
}
