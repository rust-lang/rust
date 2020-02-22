use crate::infer::canonical::{Canonicalized, CanonicalizedQueryResponse};
use crate::traits::query::outlives_bounds::OutlivesBound;
use crate::traits::query::Fallible;
use rustc::ty::{ParamEnvAnd, Ty, TyCtxt};

#[derive(Clone, Debug, HashStable, TypeFoldable, Lift)]
pub struct ImpliedOutlivesBounds<'tcx> {
    pub ty: Ty<'tcx>,
}

impl<'tcx> ImpliedOutlivesBounds<'tcx> {
    pub fn new(ty: Ty<'tcx>) -> Self {
        ImpliedOutlivesBounds { ty }
    }
}

impl<'tcx> super::QueryTypeOp<'tcx> for ImpliedOutlivesBounds<'tcx> {
    type QueryResponse = Vec<OutlivesBound<'tcx>>;

    fn try_fast_path(
        _tcx: TyCtxt<'tcx>,
        _key: &ParamEnvAnd<'tcx, Self>,
    ) -> Option<Self::QueryResponse> {
        None
    }

    fn perform_query(
        tcx: TyCtxt<'tcx>,
        canonicalized: Canonicalized<'tcx, ParamEnvAnd<'tcx, Self>>,
    ) -> Fallible<CanonicalizedQueryResponse<'tcx, Self::QueryResponse>> {
        // FIXME this `unchecked_map` is only necessary because the
        // query is defined as taking a `ParamEnvAnd<Ty>`; it should
        // take a `ImpliedOutlivesBounds` instead
        let canonicalized = canonicalized.unchecked_map(|ParamEnvAnd { param_env, value }| {
            let ImpliedOutlivesBounds { ty } = value;
            param_env.and(ty)
        });

        tcx.implied_outlives_bounds(canonicalized)
    }
}
