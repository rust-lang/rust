use crate::infer::canonical::{Canonicalized, CanonicalizedQueryResponse};
use crate::traits::query::Fallible;
use rustc_infer::traits::query::OutlivesBound;
use rustc_middle::ty::{self, ParamEnvAnd, Ty, TyCtxt};

#[derive(Copy, Clone, Debug, HashStable, TypeFoldable, Lift)]
pub struct ImpliedOutlivesBounds<'tcx> {
    pub ty: Ty<'tcx>,
}

impl<'tcx> super::QueryTypeOp<'tcx> for ImpliedOutlivesBounds<'tcx> {
    type QueryResponse = Vec<OutlivesBound<'tcx>>;

    fn try_fast_path(
        _tcx: TyCtxt<'tcx>,
        key: &ParamEnvAnd<'tcx, Self>,
    ) -> Option<Self::QueryResponse> {
        // Don't go into the query for things that can't possibly have lifetimes.
        match key.value.ty.kind() {
            ty::Tuple(elems) if elems.is_empty() => Some(vec![]),
            ty::Never | ty::Str | ty::Bool | ty::Char | ty::Int(_) | ty::Uint(_) | ty::Float(_) => {
                Some(vec![])
            }
            _ => None,
        }
    }

    fn perform_query(
        tcx: TyCtxt<'tcx>,
        canonicalized: Canonicalized<'tcx, ParamEnvAnd<'tcx, Self>>,
    ) -> Fallible<CanonicalizedQueryResponse<'tcx, Self::QueryResponse>> {
        // FIXME this `unchecked_map` is only necessary because the
        // query is defined as taking a `ParamEnvAnd<Ty>`; it should
        // take an `ImpliedOutlivesBounds` instead
        let canonicalized = canonicalized.unchecked_map(|ParamEnvAnd { param_env, value }| {
            let ImpliedOutlivesBounds { ty } = value;
            param_env.and(ty)
        });

        tcx.implied_outlives_bounds(canonicalized)
    }
}
