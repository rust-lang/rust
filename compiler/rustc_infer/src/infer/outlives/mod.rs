//! Various code related to computing outlives relations.

pub mod components;
pub mod env;
pub mod obligations;
pub mod test_type_match;
pub mod verify;

use rustc_middle::traits::query::OutlivesBound;
use rustc_middle::ty;

#[instrument(level = "debug", skip(param_env), ret)]
pub fn explicit_outlives_bounds<'tcx>(
    param_env: ty::ParamEnv<'tcx>,
) -> impl Iterator<Item = OutlivesBound<'tcx>> + 'tcx {
    param_env
        .caller_bounds()
        .into_iter()
        .map(ty::Predicate::kind)
        .filter_map(ty::Binder::no_bound_vars)
        .filter_map(move |kind| match kind {
            ty::PredicateKind::Clause(ty::Clause::Projection(..))
            | ty::PredicateKind::Clause(ty::Clause::Trait(..))
            | ty::PredicateKind::Clause(ty::Clause::ConstArgHasType(..))
            | ty::PredicateKind::AliasEq(..)
            | ty::PredicateKind::Coerce(..)
            | ty::PredicateKind::Subtype(..)
            | ty::PredicateKind::WellFormed(..)
            | ty::PredicateKind::ObjectSafe(..)
            | ty::PredicateKind::ClosureKind(..)
            | ty::PredicateKind::Clause(ty::Clause::TypeOutlives(..))
            | ty::PredicateKind::ConstEvaluatable(..)
            | ty::PredicateKind::ConstEquate(..)
            | ty::PredicateKind::Ambiguous
            | ty::PredicateKind::TypeWellFormedFromEnv(..) => None,
            ty::PredicateKind::Clause(ty::Clause::RegionOutlives(ty::OutlivesPredicate(
                r_a,
                r_b,
            ))) => Some(OutlivesBound::RegionSubRegion(r_b, r_a)),
        })
}
