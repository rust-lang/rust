// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use infer::canonical::{Canonical, CanonicalizedQueryResult};
use ty::{ParamEnv, Predicate, TyCtxt};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct ProvePredicate<'tcx> {
    pub param_env: ParamEnv<'tcx>,
    pub predicate: Predicate<'tcx>,
}

impl<'tcx> ProvePredicate<'tcx> {
    pub fn new(
        param_env: ParamEnv<'tcx>,
        predicate: Predicate<'tcx>,
    ) -> Self {
        ProvePredicate { param_env, predicate }
    }
}

impl<'gcx: 'tcx, 'tcx> super::QueryTypeOp<'gcx, 'tcx> for ProvePredicate<'tcx> {
    type QueryResult = ();

    fn trivial_noop(self, _tcx: TyCtxt<'_, 'gcx, 'tcx>) -> Result<Self::QueryResult, Self> {
        Err(self)
    }

    fn param_env(&self) -> ParamEnv<'tcx> {
        self.param_env
    }

    fn perform_query(
        tcx: TyCtxt<'_, 'gcx, 'tcx>,
        canonicalized: Canonical<'gcx, ProvePredicate<'gcx>>,
    ) -> CanonicalizedQueryResult<'gcx, ()> {
        tcx.type_op_prove_predicate(canonicalized).unwrap()
    }
}

BraceStructTypeFoldableImpl! {
    impl<'tcx> TypeFoldable<'tcx> for ProvePredicate<'tcx> {
        param_env,
        predicate,
    }
}

BraceStructLiftImpl! {
    impl<'a, 'tcx> Lift<'tcx> for ProvePredicate<'a> {
        type Lifted = ProvePredicate<'tcx>;
        param_env,
        predicate,
    }
}

impl_stable_hash_for! {
    struct ProvePredicate<'tcx> { param_env, predicate }
}
