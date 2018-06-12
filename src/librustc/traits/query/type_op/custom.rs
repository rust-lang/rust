// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use infer::{InferCtxt, InferOk};
use traits::query::Fallible;
use ty::TyCtxt;
use std::fmt;

pub struct CustomTypeOp<F, G> {
    closure: F,
    description: G,
}

impl<F, G> CustomTypeOp<F, G> {
    pub fn new<'gcx, 'tcx, R>(closure: F, description: G) -> Self
    where
        F: FnOnce(&InferCtxt<'_, 'gcx, 'tcx>) -> Fallible<InferOk<'tcx, R>>,
        G: Fn() -> String,
    {
        CustomTypeOp {
            closure,
            description,
        }
    }
}

impl<'gcx, 'tcx, F, R, G> super::TypeOp<'gcx, 'tcx> for CustomTypeOp<F, G>
where
    F: for<'a, 'cx> FnOnce(&'a InferCtxt<'cx, 'gcx, 'tcx>) -> Fallible<InferOk<'tcx, R>>,
    G: Fn() -> String,
{
    type Output = R;

    fn trivial_noop(self, _tcx: TyCtxt<'_, 'gcx, 'tcx>) -> Result<Self::Output, Self> {
        Err(self)
    }

    fn perform(self, infcx: &InferCtxt<'_, 'gcx, 'tcx>) -> Fallible<InferOk<'tcx, R>> {
        Ok((self.closure)(infcx)?)
    }
}

impl<F, G> fmt::Debug for CustomTypeOp<F, G>
where
    G: Fn() -> String,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", (self.description)())
    }
}
