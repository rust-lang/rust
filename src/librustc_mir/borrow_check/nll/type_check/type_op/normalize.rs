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
use rustc::traits::query::NoSolution;
use rustc::traits::{Normalized, ObligationCause};
use rustc::ty::fold::TypeFoldable;
use rustc::ty::{ParamEnv, TyCtxt};
use std::fmt;

#[derive(Debug)]
crate struct Normalize<'tcx, T> {
    param_env: ParamEnv<'tcx>,
    value: T,
}

impl<'tcx, T> Normalize<'tcx, T>
where
    T: fmt::Debug + TypeFoldable<'tcx>,
{
    crate fn new(param_env: ParamEnv<'tcx>, value: T) -> Self {
        Self { param_env, value }
    }
}

impl<'gcx, 'tcx, T> super::TypeOp<'gcx, 'tcx> for Normalize<'tcx, T>
where
    T: fmt::Debug + TypeFoldable<'tcx>,
{
    type Output = T;

    fn trivial_noop(self, _tcx: TyCtxt<'_, 'gcx, 'tcx>) -> Result<Self::Output, Self> {
        if !self.value.has_projections() {
            Ok(self.value)
        } else {
            Err(self)
        }
    }

    fn perform(self, infcx: &InferCtxt<'_, 'gcx, 'tcx>) -> InferResult<'tcx, Self::Output> {
        let Normalized { value, obligations } = infcx
            .at(&ObligationCause::dummy(), self.param_env)
            .normalize(&self.value)
            .unwrap_or_else(|NoSolution| {
                bug!("normalization of `{:?}` failed", self.value,);
            });
        Ok(InferOk { value, obligations })
    }
}
