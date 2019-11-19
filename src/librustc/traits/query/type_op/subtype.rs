use crate::infer::canonical::{Canonicalized, CanonicalizedQueryResponse};
use crate::traits::query::Fallible;
use crate::ty::{ParamEnvAnd, Ty, TyCtxt};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq, TypeFoldable, Lift)]
pub struct Subtype<'tcx> {
    pub sub: Ty<'tcx>,
    pub sup: Ty<'tcx>,
}

impl<'tcx> Subtype<'tcx> {
    pub fn new(sub: Ty<'tcx>, sup: Ty<'tcx>) -> Self {
        Self {
            sub,
            sup,
        }
    }
}

impl<'tcx> super::QueryTypeOp<'tcx> for Subtype<'tcx> {
    type QueryResponse = ();

    fn try_fast_path(_tcx: TyCtxt<'tcx>, key: &ParamEnvAnd<'tcx, Self>) -> Option<()> {
        if key.value.sub == key.value.sup {
            Some(())
        } else {
            None
        }
    }

    fn perform_query(
        tcx: TyCtxt<'tcx>,
        canonicalized: Canonicalized<'tcx, ParamEnvAnd<'tcx, Self>>,
    ) -> Fallible<CanonicalizedQueryResponse<'tcx, ()>> {
        tcx.type_op_subtype(canonicalized)
    }
}

impl_stable_hash_for! {
    struct Subtype<'tcx> { sub, sup }
}
