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
use repr::*;

pub trait EvalInto<H:Hair> {
    fn eval_into(self, builder: &mut Builder<H>, destination: &Lvalue<H>,
                 block: BasicBlock) -> BlockAnd<()>;
}

impl<H:Hair> Builder<H> {
    pub fn into<E>(&mut self,
                   destination: &Lvalue<H>,
                   block: BasicBlock,
                   expr: E)
                   -> BlockAnd<()>
        where E: EvalInto<H>
    {
        expr.eval_into(self, destination, block)
    }
}

impl<H:Hair> EvalInto<H> for ExprRef<H> {
    fn eval_into(self,
                 builder: &mut Builder<H>,
                 destination: &Lvalue<H>,
                 block: BasicBlock)
                 -> BlockAnd<()> {
        let expr = builder.hir.mirror(self);
        builder.into_expr(destination, block, expr)
    }
}

impl<H:Hair> EvalInto<H> for Expr<H> {
    fn eval_into(self,
                 builder: &mut Builder<H>,
                 destination: &Lvalue<H>,
                 block: BasicBlock)
                 -> BlockAnd<()> {
        builder.into_expr(destination, block, self)
    }
}

impl<H:Hair> EvalInto<H> for Option<ExprRef<H>> {
    fn eval_into(self,
                 builder: &mut Builder<H>,
                 destination: &Lvalue<H>,
                 block: BasicBlock)
                 -> BlockAnd<()> {
        match self {
            Some(expr) => builder.into(destination, block, expr),
            None => block.unit()
        }
    }
}
