use crate::infer::canonical::{Canonicalized, CanonicalizedQueryResponse};
use crate::traits::query::Fallible;
use crate::ty::{ParamEnvAnd, Ty, TyCtxt};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq, HashStable, TypeFoldable, Lift)]
pub struct Eq<'tcx> {
    pub a: Ty<'tcx>,
    pub b: Ty<'tcx>,
}

impl<'tcx> Eq<'tcx> {
    pub fn new(a: Ty<'tcx>, b: Ty<'tcx>) -> Self {
        Self { a, b }
    }
}

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
        canonicalized: Canonicalized<'tcx, ParamEnvAnd<'tcx, Self>>,
    ) -> Fallible<CanonicalizedQueryResponse<'tcx, ()>> {
        tcx.type_op_eq(canonicalized)
    }
}
