// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use infer::canonical::{Canonical, Canonicalized, CanonicalizedQueryResponse, QueryResponse};
use traits::query::outlives_bounds::OutlivesBound;
use traits::query::Fallible;
use ty::{ParamEnvAnd, Ty, TyCtxt};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct ImpliedOutlivesBounds<'tcx> {
    pub ty: Ty<'tcx>,
}

impl<'tcx> ImpliedOutlivesBounds<'tcx> {
    pub fn new(ty: Ty<'tcx>) -> Self {
        ImpliedOutlivesBounds { ty }
    }
}

impl<'gcx: 'tcx, 'tcx> super::QueryTypeOp<'gcx, 'tcx> for ImpliedOutlivesBounds<'tcx> {
    type QueryResponse = Vec<OutlivesBound<'tcx>>;

    fn try_fast_path(
        _tcx: TyCtxt<'_, 'gcx, 'tcx>,
        _key: &ParamEnvAnd<'tcx, Self>,
    ) -> Option<Self::QueryResponse> {
        None
    }

    fn perform_query(
        tcx: TyCtxt<'_, 'gcx, 'tcx>,
        canonicalized: Canonicalized<'gcx, ParamEnvAnd<'tcx, Self>>,
    ) -> Fallible<CanonicalizedQueryResponse<'gcx, Self::QueryResponse>> {
        // FIXME this `unchecked_map` is only necessary because the
        // query is defined as taking a `ParamEnvAnd<Ty>`; it should
        // take a `ImpliedOutlivesBounds` instead
        let canonicalized = canonicalized.unchecked_map(|ParamEnvAnd { param_env, value }| {
            let ImpliedOutlivesBounds { ty } = value;
            param_env.and(ty)
        });

        tcx.implied_outlives_bounds(canonicalized)
    }

    fn shrink_to_tcx_lifetime(
        v: &'a CanonicalizedQueryResponse<'gcx, Self::QueryResponse>,
    ) -> &'a Canonical<'tcx, QueryResponse<'tcx, Self::QueryResponse>> {
        v
    }
}

BraceStructTypeFoldableImpl! {
    impl<'tcx> TypeFoldable<'tcx> for ImpliedOutlivesBounds<'tcx> {
        ty,
    }
}

BraceStructLiftImpl! {
    impl<'a, 'tcx> Lift<'tcx> for ImpliedOutlivesBounds<'a> {
        type Lifted = ImpliedOutlivesBounds<'tcx>;
        ty,
    }
}

impl_stable_hash_for! {
    struct ImpliedOutlivesBounds<'tcx> { ty }
}
