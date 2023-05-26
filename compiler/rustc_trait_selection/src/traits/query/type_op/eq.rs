use crate::infer::canonical::{Canonical, CanonicalQueryResponse};
use rustc_middle::traits::query::NoSolution;
use rustc_middle::ty::{ParamEnvAnd, TyCtxt};

pub use rustc_middle::traits::query::type_op::Eq;

impl<'tcx> super::QueryTypeOp<'tcx> for Eq<'tcx> {
    type QueryResponse = ();

    fn try_fast_path(
        _tcx: TyCtxt<'tcx>,
        key: &ParamEnvAnd<'tcx, Eq<'tcx>>,
    ) -> Option<Self::QueryResponse> {
        if key.value.a == key.value.b { Some(()) } else { None }
    }

    fn perform_query(
        tcx: TyCtxt<'tcx>,
        canonicalized: Canonical<'tcx, ParamEnvAnd<'tcx, Self>>,
    ) -> Result<CanonicalQueryResponse<'tcx, ()>, NoSolution> {
        tcx.type_op_eq(canonicalized)
    }
}
