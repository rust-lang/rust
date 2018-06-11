// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use infer::{InferCtxt, InferOk, InferResult};
use traits::{Obligation, ObligationCause};
use ty::{ParamEnv, Predicate, TyCtxt};

#[derive(Debug)]
pub struct ProvePredicate<'tcx> {
    param_env: ParamEnv<'tcx>,
    predicate: Predicate<'tcx>,
}

impl<'tcx> ProvePredicate<'tcx> {
    pub fn new(
        param_env: ParamEnv<'tcx>,
        predicate: Predicate<'tcx>,
    ) -> Self {
        ProvePredicate { param_env, predicate }
    }
}

impl<'gcx, 'tcx> super::TypeOp<'gcx, 'tcx> for ProvePredicate<'tcx> {
    type Output = ();

    fn trivial_noop(self, _tcx: TyCtxt<'_, 'gcx, 'tcx>) -> Result<Self::Output, Self> {
        Err(self)
    }

    fn perform(self, _infcx: &InferCtxt<'_, 'gcx, 'tcx>) -> InferResult<'tcx, Self::Output> {
        let obligation = Obligation::new(ObligationCause::dummy(), self.param_env, self.predicate);
        Ok(InferOk {
            value: (),
            obligations: vec![obligation],
        })
    }
}
