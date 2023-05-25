use crate::infer::canonical::{Canonical, CanonicalQueryResponse};
use rustc_middle::traits::query::NoSolution;
use rustc_middle::ty::{ParamEnvAnd, TyCtxt};

pub use rustc_middle::traits::query::type_op::AscribeUserType;

impl<'tcx> super::QueryTypeOp<'tcx> for AscribeUserType<'tcx> {
    type QueryResponse = ();

    fn try_fast_path(
        _tcx: TyCtxt<'tcx>,
        _key: &ParamEnvAnd<'tcx, Self>,
    ) -> Option<Self::QueryResponse> {
        None
    }

    fn perform_query(
        tcx: TyCtxt<'tcx>,
        canonicalized: Canonical<'tcx, ParamEnvAnd<'tcx, Self>>,
    ) -> Result<CanonicalQueryResponse<'tcx, ()>, NoSolution> {
        tcx.type_op_ascribe_user_type(canonicalized)
    }
}
