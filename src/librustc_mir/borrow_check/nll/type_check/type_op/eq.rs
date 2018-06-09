// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::infer::{InferCtxt, InferResult};
use rustc::traits::ObligationCause;
use rustc::ty::{ParamEnv, Ty, TyCtxt};

#[derive(Debug)]
crate struct Eq<'tcx> {
    param_env: ParamEnv<'tcx>,
    a: Ty<'tcx>,
    b: Ty<'tcx>,
}

impl<'tcx> Eq<'tcx> {
    crate fn new(param_env: ParamEnv<'tcx>, a: Ty<'tcx>, b: Ty<'tcx>) -> Self {
        Self { param_env, a, b }
    }
}

impl<'gcx, 'tcx> super::TypeOp<'gcx, 'tcx> for Eq<'tcx> {
    type Output = ();

    fn trivial_noop(self, _tcx: TyCtxt<'_, 'gcx, 'tcx>) -> Result<Self::Output, Self> {
        if self.a == self.b {
            Ok(())
        } else {
            Err(self)
        }
    }

    fn perform(self, infcx: &InferCtxt<'_, 'gcx, 'tcx>) -> InferResult<'tcx, Self::Output> {
        infcx
            .at(&ObligationCause::dummy(), self.param_env)
            .eq(self.a, self.b)
    }
}
