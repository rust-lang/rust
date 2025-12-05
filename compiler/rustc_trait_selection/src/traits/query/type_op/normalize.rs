use std::fmt;

use rustc_middle::traits::ObligationCause;
use rustc_middle::traits::query::NoSolution;
pub use rustc_middle::traits::query::type_op::{DeeplyNormalize, Normalize};
use rustc_middle::ty::{self, Lift, ParamEnvAnd, Ty, TyCtxt, TypeFoldable, TypeVisitableExt};
use rustc_span::Span;

use crate::infer::canonical::{CanonicalQueryInput, CanonicalQueryResponse};
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
        canonicalized: CanonicalQueryInput<'tcx, ParamEnvAnd<'tcx, Self>>,
    ) -> Result<CanonicalQueryResponse<'tcx, Self::QueryResponse>, NoSolution> {
        T::type_op_method(tcx, canonicalized)
    }

    fn perform_locally_with_next_solver(
        _ocx: &ObligationCtxt<'_, 'tcx>,
        key: ParamEnvAnd<'tcx, Self>,
        _span: Span,
    ) -> Result<Self::QueryResponse, NoSolution> {
        Ok(key.value.value)
    }
}

impl<'tcx, T> super::QueryTypeOp<'tcx> for DeeplyNormalize<T>
where
    T: Normalizable<'tcx> + 'tcx,
{
    type QueryResponse = T;

    fn try_fast_path(_tcx: TyCtxt<'tcx>, key: &ParamEnvAnd<'tcx, Self>) -> Option<T> {
        if !key.value.value.has_aliases() { Some(key.value.value) } else { None }
    }

    fn perform_query(
        tcx: TyCtxt<'tcx>,
        canonicalized: CanonicalQueryInput<'tcx, ParamEnvAnd<'tcx, Self>>,
    ) -> Result<CanonicalQueryResponse<'tcx, Self::QueryResponse>, NoSolution> {
        T::type_op_method(
            tcx,
            CanonicalQueryInput {
                typing_mode: canonicalized.typing_mode,
                canonical: canonicalized.canonical.unchecked_map(
                    |ty::ParamEnvAnd { param_env, value }| ty::ParamEnvAnd {
                        param_env,
                        value: Normalize { value: value.value },
                    },
                ),
            },
        )
    }

    fn perform_locally_with_next_solver(
        ocx: &ObligationCtxt<'_, 'tcx>,
        key: ParamEnvAnd<'tcx, Self>,
        span: Span,
    ) -> Result<Self::QueryResponse, NoSolution> {
        ocx.deeply_normalize(
            &ObligationCause::dummy_with_span(span),
            key.param_env,
            key.value.value,
        )
        .map_err(|_| NoSolution)
    }
}

pub trait Normalizable<'tcx>:
    fmt::Debug + TypeFoldable<TyCtxt<'tcx>> + Lift<TyCtxt<'tcx>> + Copy
{
    fn type_op_method(
        tcx: TyCtxt<'tcx>,
        canonicalized: CanonicalQueryInput<'tcx, ParamEnvAnd<'tcx, Normalize<Self>>>,
    ) -> Result<CanonicalQueryResponse<'tcx, Self>, NoSolution>;
}

impl<'tcx> Normalizable<'tcx> for Ty<'tcx> {
    fn type_op_method(
        tcx: TyCtxt<'tcx>,
        canonicalized: CanonicalQueryInput<'tcx, ParamEnvAnd<'tcx, Normalize<Self>>>,
    ) -> Result<CanonicalQueryResponse<'tcx, Self>, NoSolution> {
        tcx.type_op_normalize_ty(canonicalized)
    }
}

impl<'tcx> Normalizable<'tcx> for ty::Clause<'tcx> {
    fn type_op_method(
        tcx: TyCtxt<'tcx>,
        canonicalized: CanonicalQueryInput<'tcx, ParamEnvAnd<'tcx, Normalize<Self>>>,
    ) -> Result<CanonicalQueryResponse<'tcx, Self>, NoSolution> {
        tcx.type_op_normalize_clause(canonicalized)
    }
}

impl<'tcx> Normalizable<'tcx> for ty::PolyFnSig<'tcx> {
    fn type_op_method(
        tcx: TyCtxt<'tcx>,
        canonicalized: CanonicalQueryInput<'tcx, ParamEnvAnd<'tcx, Normalize<Self>>>,
    ) -> Result<CanonicalQueryResponse<'tcx, Self>, NoSolution> {
        tcx.type_op_normalize_poly_fn_sig(canonicalized)
    }
}

impl<'tcx> Normalizable<'tcx> for ty::FnSig<'tcx> {
    fn type_op_method(
        tcx: TyCtxt<'tcx>,
        canonicalized: CanonicalQueryInput<'tcx, ParamEnvAnd<'tcx, Normalize<Self>>>,
    ) -> Result<CanonicalQueryResponse<'tcx, Self>, NoSolution> {
        tcx.type_op_normalize_fn_sig(canonicalized)
    }
}

/// This impl is not needed, since we never normalize type outlives predicates
/// in the old solver, but is required by trait bounds to be happy.
impl<'tcx> Normalizable<'tcx> for ty::PolyTypeOutlivesPredicate<'tcx> {
    fn type_op_method(
        _tcx: TyCtxt<'tcx>,
        _canonicalized: CanonicalQueryInput<'tcx, ParamEnvAnd<'tcx, Normalize<Self>>>,
    ) -> Result<CanonicalQueryResponse<'tcx, Self>, NoSolution> {
        unreachable!("we never normalize PolyTypeOutlivesPredicate")
    }
}
