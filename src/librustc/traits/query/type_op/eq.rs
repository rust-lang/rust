// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use infer::canonical::{CanonicalizedQueryResult, Canonical};
use traits::query::NoSolution;
use traits::{FulfillmentContext, ObligationCause};
use ty::{self, ParamEnv, Ty, TyCtxt};
use syntax::codemap::DUMMY_SP;

#[derive(Copy, Clone, Debug)]
pub struct Eq<'tcx> {
    param_env: ParamEnv<'tcx>,
    a: Ty<'tcx>,
    b: Ty<'tcx>,
}

impl<'tcx> Eq<'tcx> {
    pub fn new(param_env: ParamEnv<'tcx>, a: Ty<'tcx>, b: Ty<'tcx>) -> Self {
        Self { param_env, a, b }
    }
}

impl<'gcx: 'tcx, 'tcx> super::QueryTypeOp<'gcx, 'tcx> for Eq<'tcx> {
    type QueryResult = ();

    fn trivial_noop(self, _tcx: TyCtxt<'_, 'gcx, 'tcx>) -> Result<Self::QueryResult, Self> {
        if self.a == self.b {
            Ok(())
        } else {
            Err(self)
        }
    }

    fn param_env(&self) -> ty::ParamEnv<'tcx> {
        self.param_env
    }

    fn perform_query(
        tcx: TyCtxt<'_, 'gcx, 'tcx>,
        canonicalized: Canonical<'gcx, Eq<'gcx>>,
    ) -> CanonicalizedQueryResult<'gcx, ()> {
        let tcx = tcx.global_tcx();
        tcx.infer_ctxt()
            .enter(|ref infcx| {
                let (Eq { param_env, a, b }, canonical_inference_vars) =
                    infcx.instantiate_canonical_with_fresh_inference_vars(DUMMY_SP, &canonicalized);
                let fulfill_cx = &mut FulfillmentContext::new();
                let obligations = match infcx.at(&ObligationCause::dummy(), param_env).eq(a, b) {
                    Ok(v) => v.into_obligations(),
                    Err(_) => return Err(NoSolution),
                };
                fulfill_cx.register_predicate_obligations(infcx, obligations);
                infcx.make_canonicalized_query_result(canonical_inference_vars, (), fulfill_cx)
            })
            .unwrap()
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
