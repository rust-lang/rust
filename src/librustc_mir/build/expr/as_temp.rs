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

use build::{BlockAnd, BlockAndExtension, Builder};
use build::expr::category::Category;
use hair::*;
use rustc::mir::*;

impl<'a, 'gcx, 'tcx> Builder<'a, 'gcx, 'tcx> {
    /// Compile `expr` into a fresh temporary. This is used when building
    /// up rvalues so as to freeze the value that will be consumed.
    pub fn as_temp<M>(&mut self, block: BasicBlock, expr: M) -> BlockAnd<Lvalue<'tcx>>
        where M: Mirror<'tcx, Output = Expr<'tcx>>
    {
        let expr = self.hir.mirror(expr);
        self.expr_as_temp(block, expr)
    }

    fn expr_as_temp(&mut self, mut block: BasicBlock, expr: Expr<'tcx>) -> BlockAnd<Lvalue<'tcx>> {
        debug!("expr_as_temp(block={:?}, expr={:?})", block, expr);
        let this = self;

        if let ExprKind::Scope { extent, value } = expr.kind {
            return this.in_scope(extent, block, |this| this.as_temp(block, value));
        }

        let expr_ty = expr.ty.clone();
        let temp = this.temp(expr_ty.clone());
        let temp_lifetime = expr.temp_lifetime;
        let expr_span = expr.span;
        let source_info = this.source_info(expr_span);

        if temp_lifetime.is_some() {
            this.cfg.push(block, Statement {
                source_info: source_info,
                kind: StatementKind::StorageLive(temp.clone())
            });
        }

        // Careful here not to cause an infinite cycle. If we always
        // called `into`, then for lvalues like `x.f`, it would
        // eventually fallback to us, and we'd loop. There's a reason
        // for this: `as_temp` is the point where we bridge the "by
        // reference" semantics of `as_lvalue` with the "by value"
        // semantics of `into`, `as_operand`, `as_rvalue`, and (of
        // course) `as_temp`.
        match Category::of(&expr.kind).unwrap() {
            Category::Lvalue => {
                let lvalue = unpack!(block = this.as_lvalue(block, expr));
                let rvalue = Rvalue::Use(Operand::Consume(lvalue));
                this.cfg.push_assign(block, source_info, &temp, rvalue);
            }
            _ => {
                unpack!(block = this.into(&temp, block, expr));
            }
        }

        // In constants, temp_lifetime is None. We should not need to drop
        // anything because no values with a destructor can be created in
        // a constant at this time, even if the type may need dropping.
        if let Some(temp_lifetime) = temp_lifetime {
            this.schedule_drop(expr_span, temp_lifetime, &temp, expr_ty);
        }

        block.and(temp)
    }
}
