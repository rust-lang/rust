// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use infer::canonical::{Canonical, Canonicalized, CanonicalizedQueryResult, QueryResult};
use traits::query::dropck_outlives::trivial_dropck_outlives;
use traits::query::dropck_outlives::DropckOutlivesResult;
use traits::query::Fallible;
use ty::{ParamEnv, ParamEnvAnd, Ty, TyCtxt};

#[derive(Debug)]
pub struct DropckOutlives<'tcx> {
    param_env: ParamEnv<'tcx>,
    dropped_ty: Ty<'tcx>,
}

impl<'tcx> DropckOutlives<'tcx> {
    pub fn new(param_env: ParamEnv<'tcx>, dropped_ty: Ty<'tcx>) -> Self {
        DropckOutlives {
            param_env,
            dropped_ty,
        }
    }
}

impl super::QueryTypeOp<'gcx, 'tcx> for DropckOutlives<'tcx>
where
    'gcx: 'tcx,
{
    type QueryKey = ParamEnvAnd<'tcx, Ty<'tcx>>;
    type QueryResult = DropckOutlivesResult<'tcx>;

    fn prequery(self, tcx: TyCtxt<'_, 'gcx, 'tcx>) -> Result<Self::QueryResult, Self::QueryKey> {
        if trivial_dropck_outlives(tcx, self.dropped_ty) {
            Ok(DropckOutlivesResult::default())
        } else {
            Err(self.param_env.and(self.dropped_ty))
        }
    }

    fn param_env(key: &Self::QueryKey) -> ParamEnv<'tcx> {
        key.param_env
    }

    fn perform_query(
        tcx: TyCtxt<'_, 'gcx, 'tcx>,
        canonicalized: Canonicalized<'gcx, Self::QueryKey>,
    ) -> Fallible<CanonicalizedQueryResult<'gcx, Self::QueryResult>> {
        // Subtle: note that we are not invoking
        // `infcx.at(...).dropck_outlives(...)` here, but rather the
        // underlying `dropck_outlives` query. This same underlying
        // query is also used by the
        // `infcx.at(...).dropck_outlives(...)` fn. Avoiding the
        // wrapper means we don't need an infcx in this code, which is
        // good because the interface doesn't give us one (so that we
        // know we are not registering any subregion relations or
        // other things).
        tcx.dropck_outlives(canonicalized)
    }

    fn cast_to_tcx_lifetime(
        lifted_query_result: &'a CanonicalizedQueryResult<'gcx, Self::QueryResult>,
    ) -> &'a Canonical<'tcx, QueryResult<'tcx, Self::QueryResult>> {
        lifted_query_result
    }
}
