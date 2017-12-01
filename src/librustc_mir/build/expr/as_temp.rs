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
use rustc::middle::region;
use rustc::mir::*;

impl<'a, 'gcx, 'tcx> Builder<'a, 'gcx, 'tcx> {
    /// Compile `expr` into a fresh temporary. This is used when building
    /// up rvalues so as to freeze the value that will be consumed.
    pub fn as_temp<M>(&mut self,
                      block: BasicBlock,
                      temp_lifetime: Option<region::Scope>,
                      expr: M)
                      -> BlockAnd<Local>
        where M: Mirror<'tcx, Output = Expr<'tcx>>
    {
        let expr = self.hir.mirror(expr);
        self.expr_as_temp(block, temp_lifetime, expr)
    }

    fn expr_as_temp(&mut self,
                    mut block: BasicBlock,
                    temp_lifetime: Option<region::Scope>,
                    expr: Expr<'tcx>)
                    -> BlockAnd<Local> {
        debug!("expr_as_temp(block={:?}, temp_lifetime={:?}, expr={:?})",
               block, temp_lifetime, expr);
        let this = self;

        let expr_span = expr.span;
        let source_info = this.source_info(expr_span);
        if let ExprKind::Scope { region_scope, lint_level, value } = expr.kind {
            return this.in_scope((region_scope, source_info), lint_level, block, |this| {
                this.as_temp(block, temp_lifetime, value)
            });
        }

        let expr_ty = expr.ty;
        let temp = this.local_decls.push(LocalDecl::new_temp(expr_ty, expr_span));

        if !expr_ty.is_never() {
            this.cfg.push(block, Statement {
                source_info,
                kind: StatementKind::StorageLive(temp)
            });
        }

        // Careful here not to cause an infinite cycle. If we always
        // called `into`, then for places like `x.f`, it would
        // eventually fallback to us, and we'd loop. There's a reason
        // for this: `as_temp` is the point where we bridge the "by
        // reference" semantics of `as_place` with the "by value"
        // semantics of `into`, `as_operand`, `as_rvalue`, and (of
        // course) `as_temp`.
        match Category::of(&expr.kind).unwrap() {
            Category::Place => {
                let place = unpack!(block = this.as_place(block, expr));
                let rvalue = Rvalue::Use(this.consume_by_copy_or_move(place));
                this.cfg.push_assign(block, source_info, &Place::Local(temp), rvalue);
            }
            _ => {
                unpack!(block = this.into(&Place::Local(temp), block, expr));
            }
        }

        // In constants, temp_lifetime is None. We should not need to drop
        // anything because no values with a destructor can be created in
        // a constant at this time, even if the type may need dropping.
        if let Some(temp_lifetime) = temp_lifetime {
            this.schedule_drop(expr_span, temp_lifetime, &Place::Local(temp), expr_ty);
        }

        block.and(temp)
    }
}
