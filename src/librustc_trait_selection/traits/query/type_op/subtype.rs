use crate::infer::canonical::{Canonicalized, CanonicalizedQueryResponse};
use crate::traits::query::Fallible;
use rustc::ty::{ParamEnvAnd, TyCtxt};

pub use rustc::traits::query::type_op::Subtype;

impl<'tcx> super::QueryTypeOp<'tcx> for Subtype<'tcx> {
    type QueryResponse = ();

    fn try_fast_path(_tcx: TyCtxt<'tcx>, key: &ParamEnvAnd<'tcx, Self>) -> Option<()> {
        if key.value.sub == key.value.sup { Some(()) } else { None }
    }

    fn perform_query(
        tcx: TyCtxt<'tcx>,
        canonicalized: Canonicalized<'tcx, ParamEnvAnd<'tcx, Self>>,
    ) -> Fallible<CanonicalizedQueryResponse<'tcx, ()>> {
        tcx.type_op_subtype(canonicalized)
    }
}
