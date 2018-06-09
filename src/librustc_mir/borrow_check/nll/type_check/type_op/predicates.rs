// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::infer::{InferCtxt, InferOk, InferResult};
use rustc::traits::{Obligation, ObligationCause, PredicateObligation};
use rustc::ty::{ParamEnv, Predicate, TyCtxt};

#[derive(Debug)]
crate struct ProvePredicates<'tcx> {
    obligations: Vec<PredicateObligation<'tcx>>,
}

impl<'tcx> ProvePredicates<'tcx> {
    crate fn new(
        param_env: ParamEnv<'tcx>,
        predicates: impl IntoIterator<Item = Predicate<'tcx>>,
    ) -> Self {
        ProvePredicates {
            obligations: predicates
                .into_iter()
                .map(|p| Obligation::new(ObligationCause::dummy(), param_env, p))
                .collect(),
        }
    }
}

impl<'gcx, 'tcx> super::TypeOp<'gcx, 'tcx> for ProvePredicates<'tcx> {
    type Output = ();

    fn trivial_noop(self, _tcx: TyCtxt<'_, 'gcx, 'tcx>) -> Result<Self::Output, Self> {
        if self.obligations.is_empty() {
            Ok(())
        } else {
            Err(self)
        }
    }

    fn perform(self, _infcx: &InferCtxt<'_, 'gcx, 'tcx>) -> InferResult<'tcx, Self::Output> {
        Ok(InferOk {
            value: (),
            obligations: self.obligations,
        })
    }
}
