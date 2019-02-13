//! Experimental types for the trait query interface. The methods
//! defined in this module are all based on **canonicalization**,
//! which makes a canonical query by replacing unbound inference
//! variables and regions, so that results can be reused more broadly.
//! The providers for the queries defined here can be found in
//! `librustc_traits`.

use crate::infer::canonical::Canonical;
use crate::ty::error::TypeError;
use crate::ty::{self, Ty};

pub mod dropck_outlives;
pub mod evaluate_obligation;
pub mod method_autoderef;
pub mod normalize;
pub mod normalize_erasing_regions;
pub mod outlives_bounds;
pub mod type_op;

pub type CanonicalProjectionGoal<'tcx> =
    Canonical<'tcx, ty::ParamEnvAnd<'tcx, ty::ProjectionTy<'tcx>>>;

pub type CanonicalTyGoal<'tcx> = Canonical<'tcx, ty::ParamEnvAnd<'tcx, Ty<'tcx>>>;

pub type CanonicalPredicateGoal<'tcx> =
    Canonical<'tcx, ty::ParamEnvAnd<'tcx, ty::Predicate<'tcx>>>;

pub type CanonicalTypeOpAscribeUserTypeGoal<'tcx> =
    Canonical<'tcx, ty::ParamEnvAnd<'tcx, type_op::ascribe_user_type::AscribeUserType<'tcx>>>;

pub type CanonicalTypeOpEqGoal<'tcx> =
    Canonical<'tcx, ty::ParamEnvAnd<'tcx, type_op::eq::Eq<'tcx>>>;

pub type CanonicalTypeOpSubtypeGoal<'tcx> =
    Canonical<'tcx, ty::ParamEnvAnd<'tcx, type_op::subtype::Subtype<'tcx>>>;

pub type CanonicalTypeOpProvePredicateGoal<'tcx> =
    Canonical<'tcx, ty::ParamEnvAnd<'tcx, type_op::prove_predicate::ProvePredicate<'tcx>>>;

pub type CanonicalTypeOpNormalizeGoal<'tcx, T> =
    Canonical<'tcx, ty::ParamEnvAnd<'tcx, type_op::normalize::Normalize<T>>>;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct NoSolution;

pub type Fallible<T> = Result<T, NoSolution>;

impl<'tcx> From<TypeError<'tcx>> for NoSolution {
    fn from(_: TypeError<'tcx>) -> NoSolution {
        NoSolution
    }
}

impl_stable_hash_for!(struct NoSolution { });
