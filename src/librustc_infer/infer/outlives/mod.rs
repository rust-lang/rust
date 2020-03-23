//! Various code related to computing outlives relations.

pub mod env;
pub mod obligations;
pub mod verify;

use rustc::traits::query::OutlivesBound;
use rustc::ty;

pub fn explicit_outlives_bounds<'tcx>(
    param_env: ty::ParamEnv<'tcx>,
) -> impl Iterator<Item = OutlivesBound<'tcx>> + 'tcx {
    debug!("explicit_outlives_bounds()");
    param_env.caller_bounds.into_iter().filter_map(move |predicate| match predicate {
        ty::Predicate::Projection(..)
        | ty::Predicate::Trait(..)
        | ty::Predicate::Subtype(..)
        | ty::Predicate::WellFormed(..)
        | ty::Predicate::ObjectSafe(..)
        | ty::Predicate::ClosureKind(..)
        | ty::Predicate::TypeOutlives(..)
        | ty::Predicate::ConstEvaluatable(..) => None,
        ty::Predicate::RegionOutlives(ref data) => data
            .no_bound_vars()
            .map(|ty::OutlivesPredicate(r_a, r_b)| OutlivesBound::RegionSubRegion(r_b, r_a)),
    })
}
