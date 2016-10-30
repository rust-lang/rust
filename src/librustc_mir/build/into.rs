// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! In general, there are a number of things for which it's convenient
//! to just call `builder.into` and have it emit its result into a
//! given location. This is basically for expressions or things that can be
//! wrapped up as expressions (e.g. blocks). To make this ergonomic, we use this
//! latter `EvalInto` trait.

use build::{BlockAnd, Builder};
use hair::*;
use rustc::mir::*;

pub trait EvalInto<'tcx> {
    fn eval_into<'a, 'gcx>(self,
                           builder: &mut Builder<'a, 'gcx, 'tcx>,
                           destination: &Lvalue<'tcx>,
                           block: BasicBlock)
                           -> BlockAnd<()>;
}

impl<'a, 'gcx, 'tcx> Builder<'a, 'gcx, 'tcx> {
    pub fn into<E>(&mut self,
                   destination: &Lvalue<'tcx>,
                   block: BasicBlock,
                   expr: E)
                   -> BlockAnd<()>
        where E: EvalInto<'tcx>
    {
        expr.eval_into(self, destination, block)
    }
}

impl<'tcx> EvalInto<'tcx> for ExprRef<'tcx> {
    fn eval_into<'a, 'gcx>(self,
                           builder: &mut Builder<'a, 'gcx, 'tcx>,
                           destination: &Lvalue<'tcx>,
                           block: BasicBlock)
                           -> BlockAnd<()> {
        let expr = builder.hir.mirror(self);
        builder.into_expr(destination, block, expr)
    }
}

impl<'tcx> EvalInto<'tcx> for Expr<'tcx> {
    fn eval_into<'a, 'gcx>(self,
                           builder: &mut Builder<'a, 'gcx, 'tcx>,
                           destination: &Lvalue<'tcx>,
                           block: BasicBlock)
                           -> BlockAnd<()> {
        builder.into_expr(destination, block, self)
    }
}
