// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! See docs in build/expr/mod.rs

use build::Builder;
use hair::*;
use rustc::mir::*;

impl<'a, 'gcx, 'tcx> Builder<'a, 'gcx, 'tcx> {
    /// Compile `expr`, yielding a compile-time constant. Assumes that
    /// `expr` is a valid compile-time constant!
    pub fn as_constant<M>(&mut self, expr: M) -> Constant<'tcx>
        where M: Mirror<'tcx, Output=Expr<'tcx>>
    {
        let expr = self.hir.mirror(expr);
        self.expr_as_constant(expr)
    }

    fn expr_as_constant(&mut self, expr: Expr<'tcx>) -> Constant<'tcx> {
        let this = self;
        let Expr { ty, temp_lifetime: _, span, kind } = expr;
        match kind {
            ExprKind::Scope { extent: _, value } =>
                this.as_constant(value),
            ExprKind::Literal { literal } =>
                Constant { span: span, ty: ty, literal: literal },
            _ =>
                span_bug!(
                    span,
                    "expression is not a valid constant {:?}",
                    kind),
        }
    }
}
