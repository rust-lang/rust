use crate::infer::canonical::{Canonical, Canonicalized, CanonicalizedQueryResponse, QueryResponse};
use crate::traits::query::Fallible;
use crate::ty::{ParamEnvAnd, Ty, TyCtxt};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
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
        if key.value.a == key.value.b {
            Some(())
        } else {
            None
        }
    }

    fn perform_query(
        tcx: TyCtxt<'tcx>,
        canonicalized: Canonicalized<'tcx, ParamEnvAnd<'tcx, Self>>,
    ) -> Fallible<CanonicalizedQueryResponse<'tcx, ()>> {
        tcx.type_op_eq(canonicalized)
    }

    fn shrink_to_tcx_lifetime(
        v: &'a CanonicalizedQueryResponse<'tcx, ()>,
    ) -> &'a Canonical<'tcx, QueryResponse<'tcx, ()>> {
        v
    }
}

BraceStructTypeFoldableImpl! {
    impl<'tcx> TypeFoldable<'tcx> for Eq<'tcx> {
        a,
        b,
    }
}

BraceStructLiftImpl! {
    impl<'a, 'tcx> Lift<'tcx> for Eq<'a> {
        type Lifted = Eq<'tcx>;
        a,
        b,
    }
}

impl_stable_hash_for! {
    struct Eq<'tcx> { a, b }
}
