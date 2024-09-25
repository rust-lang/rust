use std::fmt;

use rustc_middle::traits::ObligationCause;
use rustc_middle::traits::query::NoSolution;
pub use rustc_middle::traits::query::type_op::Normalize;
use rustc_middle::ty::fold::TypeFoldable;
use rustc_middle::ty::{self, Lift, ParamEnvAnd, Ty, TyCtxt, TypeVisitableExt};

use crate::infer::canonical::{Canonical, CanonicalQueryResponse};
use crate::traits::ObligationCtxt;

impl<'tcx, T> super::QueryTypeOp<'tcx> for Normalize<T>
where
    T: Normalizable<'tcx> + 'tcx,
{
    type QueryResponse = T;

    fn try_fast_path(_tcx: TyCtxt<'tcx>, key: &ParamEnvAnd<'tcx, Self>) -> Option<T> {
        if !key.value.value.has_aliases() { Some(key.value.value) } else { None }
    }

    fn perform_query(
        tcx: TyCtxt<'tcx>,
        canonicalized: Canonical<'tcx, ParamEnvAnd<'tcx, Self>>,
    ) -> Result<CanonicalQueryResponse<'tcx, Self::QueryResponse>, NoSolution> {
        T::type_op_method(tcx, canonicalized)
    }

    fn perform_locally_with_next_solver(
        ocx: &ObligationCtxt<'_, 'tcx>,
        key: ParamEnvAnd<'tcx, Self>,
    ) -> Result<Self::QueryResponse, NoSolution> {
        // FIXME(-Znext-solver): shouldn't be using old normalizer
        Ok(ocx.normalize(&ObligationCause::dummy(), key.param_env, key.value.value))
    }
}

pub trait Normalizable<'tcx>:
    fmt::Debug + TypeFoldable<TyCtxt<'tcx>> + Lift<TyCtxt<'tcx>> + Copy
{
    fn type_op_method(
        tcx: TyCtxt<'tcx>,
        canonicalized: Canonical<'tcx, ParamEnvAnd<'tcx, Normalize<Self>>>,
    ) -> Result<CanonicalQueryResponse<'tcx, Self>, NoSolution>;
}

impl<'tcx> Normalizable<'tcx> for Ty<'tcx> {
    fn type_op_method(
        tcx: TyCtxt<'tcx>,
        canonicalized: Canonical<'tcx, ParamEnvAnd<'tcx, Normalize<Self>>>,
    ) -> Result<CanonicalQueryResponse<'tcx, Self>, NoSolution> {
        tcx.type_op_normalize_ty(canonicalized)
    }
}

impl<'tcx> Normalizable<'tcx> for ty::Clause<'tcx> {
    fn type_op_method(
        tcx: TyCtxt<'tcx>,
        canonicalized: Canonical<'tcx, ParamEnvAnd<'tcx, Normalize<Self>>>,
    ) -> Result<CanonicalQueryResponse<'tcx, Self>, NoSolution> {
        tcx.type_op_normalize_clause(canonicalized)
    }
}

impl<'tcx> Normalizable<'tcx> for ty::PolyFnSig<'tcx> {
    fn type_op_method(
        tcx: TyCtxt<'tcx>,
        canonicalized: Canonical<'tcx, ParamEnvAnd<'tcx, Normalize<Self>>>,
    ) -> Result<CanonicalQueryResponse<'tcx, Self>, NoSolution> {
        tcx.type_op_normalize_poly_fn_sig(canonicalized)
    }
}

impl<'tcx> Normalizable<'tcx> for ty::FnSig<'tcx> {
    fn type_op_method(
        tcx: TyCtxt<'tcx>,
        canonicalized: Canonical<'tcx, ParamEnvAnd<'tcx, Normalize<Self>>>,
    ) -> Result<CanonicalQueryResponse<'tcx, Self>, NoSolution> {
        tcx.type_op_normalize_fn_sig(canonicalized)
    }
}
