use crate::infer::canonical::{Canonicalized, CanonicalizedQueryResponse};
use crate::traits::query::dropck_outlives::{trivial_dropck_outlives, DropckOutlivesResult};
use crate::traits::query::Fallible;
use rustc_middle::ty::{ParamEnvAnd, Ty, TyCtxt};

#[derive(Copy, Clone, Debug, HashStable, TypeFoldable, Lift)]
pub struct DropckOutlives<'tcx> {
    dropped_ty: Ty<'tcx>,
}

impl<'tcx> DropckOutlives<'tcx> {
    pub fn new(dropped_ty: Ty<'tcx>) -> Self {
        DropckOutlives { dropped_ty }
    }
}

impl<'tcx> super::QueryTypeOp<'tcx> for DropckOutlives<'tcx> {
    type QueryResponse = DropckOutlivesResult<'tcx>;

    fn try_fast_path(
        tcx: TyCtxt<'tcx>,
        key: &ParamEnvAnd<'tcx, Self>,
    ) -> Option<Self::QueryResponse> {
        if trivial_dropck_outlives(tcx, key.value.dropped_ty) {
            Some(DropckOutlivesResult::default())
        } else {
            None
        }
    }

    fn perform_query(
        tcx: TyCtxt<'tcx>,
        canonicalized: Canonicalized<'tcx, ParamEnvAnd<'tcx, Self>>,
    ) -> Fallible<CanonicalizedQueryResponse<'tcx, Self::QueryResponse>> {
        // Subtle: note that we are not invoking
        // `infcx.at(...).dropck_outlives(...)` here, but rather the
        // underlying `dropck_outlives` query. This same underlying
        // query is also used by the
        // `infcx.at(...).dropck_outlives(...)` fn. Avoiding the
        // wrapper means we don't need an infcx in this code, which is
        // good because the interface doesn't give us one (so that we
        // know we are not registering any subregion relations or
        // other things).

        // FIXME convert to the type expected by the `dropck_outlives`
        // query. This should eventually be fixed by changing the
        // *underlying query*.
        let canonicalized = canonicalized.unchecked_map(|ParamEnvAnd { param_env, value }| {
            let DropckOutlives { dropped_ty } = value;
            param_env.and(dropped_ty)
        });

        tcx.dropck_outlives(canonicalized)
    }
}
