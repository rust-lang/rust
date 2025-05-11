use rustc_middle::traits::query::{DropckOutlivesResult, NoSolution};
use rustc_middle::ty::{ParamEnvAnd, TyCtxt};
use rustc_span::Span;

use crate::infer::canonical::{CanonicalQueryInput, CanonicalQueryResponse};
use crate::traits::ObligationCtxt;
use crate::traits::query::dropck_outlives::{
    compute_dropck_outlives_inner, trivial_dropck_outlives,
};
use crate::traits::query::type_op::DropckOutlives;

impl<'tcx> super::QueryTypeOp<'tcx> for DropckOutlives<'tcx> {
    type QueryResponse = DropckOutlivesResult<'tcx>;

    fn try_fast_path(
        tcx: TyCtxt<'tcx>,
        key: &ParamEnvAnd<'tcx, Self>,
    ) -> Option<Self::QueryResponse> {
        trivial_dropck_outlives(tcx, key.value.dropped_ty).then(DropckOutlivesResult::default)
    }

    fn perform_query(
        tcx: TyCtxt<'tcx>,
        canonicalized: CanonicalQueryInput<'tcx, ParamEnvAnd<'tcx, Self>>,
    ) -> Result<CanonicalQueryResponse<'tcx, Self::QueryResponse>, NoSolution> {
        tcx.dropck_outlives(canonicalized)
    }

    fn perform_locally_with_next_solver(
        ocx: &ObligationCtxt<'_, 'tcx>,
        key: ParamEnvAnd<'tcx, Self>,
        span: Span,
    ) -> Result<Self::QueryResponse, NoSolution> {
        compute_dropck_outlives_inner(ocx, key.param_env.and(key.value), span)
    }
}
