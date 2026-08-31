use rustc_middle::traits::query::{DropckOutlivesResult, NoSolution};
use rustc_middle::ty::{ParamEnvAnd, TyCtxt};

use crate::infer::canonical::{CanonicalQueryInput, CanonicalQueryResponse};
use crate::traits::query::dropck_outlives::trivial_dropck_outlives;
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
}
