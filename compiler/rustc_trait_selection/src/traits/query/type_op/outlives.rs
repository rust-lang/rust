use rustc_macros::{HashStable, TypeFoldable, TypeVisitable};
use rustc_middle::traits::query::{DropckOutlivesResult, NoSolution};
use rustc_middle::ty::{ParamEnvAnd, Ty, TyCtxt};

use crate::infer::canonical::{Canonical, CanonicalQueryResponse};
use crate::traits::ObligationCtxt;
use crate::traits::query::dropck_outlives::{
    compute_dropck_outlives_inner, trivial_dropck_outlives,
};

#[derive(Copy, Clone, Debug, HashStable, TypeFoldable, TypeVisitable)]
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
        trivial_dropck_outlives(tcx, key.value.dropped_ty).then(DropckOutlivesResult::default)
    }

    fn perform_query(
        tcx: TyCtxt<'tcx>,
        canonicalized: Canonical<'tcx, ParamEnvAnd<'tcx, Self>>,
    ) -> Result<CanonicalQueryResponse<'tcx, Self::QueryResponse>, NoSolution> {
        // FIXME convert to the type expected by the `dropck_outlives`
        // query. This should eventually be fixed by changing the
        // *underlying query*.
        let canonicalized = canonicalized.unchecked_map(|ParamEnvAnd { param_env, value }| {
            let DropckOutlives { dropped_ty } = value;
            param_env.and(dropped_ty)
        });

        tcx.dropck_outlives(canonicalized)
    }

    fn perform_locally_with_next_solver(
        ocx: &ObligationCtxt<'_, 'tcx>,
        key: ParamEnvAnd<'tcx, Self>,
    ) -> Result<Self::QueryResponse, NoSolution> {
        compute_dropck_outlives_inner(ocx, key.param_env.and(key.value.dropped_ty))
    }
}
