// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use infer::canonical::{Canonical, CanonicalizedQueryResult, QueryResult};
use traits::query::Fallible;
use ty::{ParamEnv, Ty, TyCtxt};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct Eq<'tcx> {
    pub param_env: ParamEnv<'tcx>,
    pub a: Ty<'tcx>,
    pub b: Ty<'tcx>,
}

impl<'tcx> Eq<'tcx> {
    pub fn new(param_env: ParamEnv<'tcx>, a: Ty<'tcx>, b: Ty<'tcx>) -> Self {
        Self { param_env, a, b }
    }
}

impl<'gcx: 'tcx, 'tcx> super::QueryTypeOp<'gcx, 'tcx> for Eq<'tcx> {
    type QueryKey = Self;
    type QueryResult = ();

    fn prequery(self, _tcx: TyCtxt<'_, 'gcx, 'tcx>) -> Result<Self::QueryResult, Self> {
        if self.a == self.b {
            Ok(())
        } else {
            Err(self)
        }
    }

    fn param_env(key: &Self::QueryKey) -> ParamEnv<'tcx> {
        key.param_env
    }

    fn perform_query(
        tcx: TyCtxt<'_, 'gcx, 'tcx>,
        canonicalized: Canonical<'gcx, Eq<'gcx>>,
    ) -> Fallible<CanonicalizedQueryResult<'gcx, ()>> {
        tcx.type_op_eq(canonicalized)
    }

    fn cast_to_tcx_lifetime(
        v: &'a CanonicalizedQueryResult<'gcx, ()>,
    ) -> &'a Canonical<'tcx, QueryResult<'tcx, ()>> {
        v
    }
}

BraceStructTypeFoldableImpl! {
    impl<'tcx> TypeFoldable<'tcx> for Eq<'tcx> {
        param_env,
        a,
        b,
    }
}

BraceStructLiftImpl! {
    impl<'a, 'tcx> Lift<'tcx> for Eq<'a> {
        type Lifted = Eq<'tcx>;
        param_env,
        a,
        b,
    }
}

impl_stable_hash_for! {
    struct Eq<'tcx> { param_env, a, b }
}
