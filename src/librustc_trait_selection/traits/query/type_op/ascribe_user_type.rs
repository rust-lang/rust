use crate::infer::canonical::{Canonicalized, CanonicalizedQueryResponse};
use crate::traits::query::Fallible;
use rustc::ty::{ParamEnvAnd, TyCtxt};

pub use rustc::traits::query::type_op::AscribeUserType;

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
        canonicalized: Canonicalized<'tcx, ParamEnvAnd<'tcx, Self>>,
    ) -> Fallible<CanonicalizedQueryResponse<'tcx, ()>> {
        tcx.type_op_ascribe_user_type(canonicalized)
    }
}
