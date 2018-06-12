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
pub struct Subtype<'tcx> {
    pub param_env: ParamEnv<'tcx>,
    pub sub: Ty<'tcx>,
    pub sup: Ty<'tcx>,
}

impl<'tcx> Subtype<'tcx> {
    pub fn new(param_env: ParamEnv<'tcx>, sub: Ty<'tcx>, sup: Ty<'tcx>) -> Self {
        Self {
            param_env,
            sub,
            sup,
        }
    }
}

impl<'gcx: 'tcx, 'tcx> super::QueryTypeOp<'gcx, 'tcx> for Subtype<'tcx> {
    type QueryResult = ();

    fn trivial_noop(self, _tcx: TyCtxt<'_, 'gcx, 'tcx>) -> Result<(), Self> {
        if self.sub == self.sup {
            Ok(())
        } else {
            Err(self)
        }
    }

    fn param_env(&self) -> ParamEnv<'tcx> {
        self.param_env
    }

    fn perform_query(
        tcx: TyCtxt<'_, 'gcx, 'tcx>,
        canonicalized: Canonical<'gcx, Subtype<'gcx>>,
    ) -> Fallible<CanonicalizedQueryResult<'gcx, ()>> {
        tcx.type_op_subtype(canonicalized)
    }

    fn upcast_result(
        v: &'a CanonicalizedQueryResult<'gcx, ()>,
    ) -> &'a Canonical<'tcx, QueryResult<'tcx, ()>> {
        v
    }
}

BraceStructTypeFoldableImpl! {
    impl<'tcx> TypeFoldable<'tcx> for Subtype<'tcx> {
        param_env,
        sub,
        sup,
    }
}

BraceStructLiftImpl! {
    impl<'a, 'tcx> Lift<'tcx> for Subtype<'a> {
        type Lifted = Subtype<'tcx>;
        param_env,
        sub,
        sup,
    }
}

impl_stable_hash_for! {
    struct Subtype<'tcx> { param_env, sub, sup }
}
