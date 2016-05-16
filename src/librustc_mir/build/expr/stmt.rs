// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use build::{BlockAnd, BlockAndExtension, Builder};
use build::scope::LoopScope;
use hair::*;
use rustc::middle::region::CodeExtent;
use rustc::mir::repr::*;
use syntax::codemap::Span;

impl<'a, 'gcx, 'tcx> Builder<'a, 'gcx, 'tcx> {

    pub fn stmt_expr(&mut self, mut block: BasicBlock, expr: Expr<'tcx>) -> BlockAnd<()> {
        let this = self;
        let expr_span = expr.span;
        let scope_id = this.innermost_scope_id();
        // Handle a number of expressions that don't need a destination at all. This
        // avoids needing a mountain of temporary `()` variables.
        match expr.kind {
            ExprKind::Scope { extent, value } => {
                let value = this.hir.mirror(value);
                this.in_scope(extent, block, |this, _| this.stmt_expr(block, value))
            }
            ExprKind::Assign { lhs, rhs } => {
                let lhs = this.hir.mirror(lhs);
                let rhs = this.hir.mirror(rhs);
                let scope_id = this.innermost_scope_id();
                let lhs_span = lhs.span;

                // Note: we evaluate assignments right-to-left. This
                // is better for borrowck interaction with overloaded
                // operators like x[j] = x[i].

                // Generate better code for things that don't need to be
                // dropped.
                if this.hir.needs_drop(lhs.ty) {
                    let rhs = unpack!(block = this.as_operand(block, rhs));
                    let lhs = unpack!(block = this.as_lvalue(block, lhs));
                    unpack!(block = this.build_drop_and_replace(
                        block, lhs_span, lhs, rhs
                    ));
                    block.unit()
                } else {
                    let rhs = unpack!(block = this.as_rvalue(block, rhs));
                    let lhs = unpack!(block = this.as_lvalue(block, lhs));
                    this.cfg.push_assign(block, scope_id, expr_span, &lhs, rhs);
                    block.unit()
                }
            }
            ExprKind::AssignOp { op, lhs, rhs } => {
                // FIXME(#28160) there is an interesting semantics
                // question raised here -- should we "freeze" the
                // value of the lhs here?  I'm inclined to think not,
                // since it seems closer to the semantics of the
                // overloaded version, which takes `&mut self`.  This
                // only affects weird things like `x += {x += 1; x}`
                // -- is that equal to `x + (x + 1)` or `2*(x+1)`?

                // As above, RTL.
                let rhs = unpack!(block = this.as_operand(block, rhs));
                let lhs = unpack!(block = this.as_lvalue(block, lhs));

                // we don't have to drop prior contents or anything
                // because AssignOp is only legal for Copy types
                // (overloaded ops should be desugared into a call).
                this.cfg.push_assign(block, scope_id, expr_span, &lhs,
                                     Rvalue::BinaryOp(op,
                                                      Operand::Consume(lhs.clone()),
                                                      rhs));

                block.unit()
            }
            ExprKind::Continue { label } => {
                this.break_or_continue(expr_span, label, block,
                                       |loop_scope| loop_scope.continue_block)
            }
            ExprKind::Break { label } => {
                this.break_or_continue(expr_span, label, block, |loop_scope| {
                    loop_scope.might_break = true;
                    loop_scope.break_block
                })
            }
            ExprKind::Return { value } => {
                block = match value {
                    Some(value) => unpack!(this.into(&Lvalue::ReturnPointer, block, value)),
                    None => {
                        this.cfg.push_assign_unit(block, scope_id,
                                                  expr_span, &Lvalue::ReturnPointer);
                        block
                    }
                };
                let extent = this.extent_of_return_scope();
                let return_block = this.return_block();
                this.exit_scope(expr_span, extent, block, return_block);
                this.cfg.start_new_block().unit()
            }
            _ => {
                let expr_span = expr.span;
                let expr_ty = expr.ty;
                let temp = this.temp(expr.ty.clone());
                unpack!(block = this.into(&temp, block, expr));
                unpack!(block = this.build_drop(block, expr_span, temp, expr_ty));
                block.unit()
            }
        }
    }

    fn break_or_continue<F>(&mut self,
                            span: Span,
                            label: Option<CodeExtent>,
                            block: BasicBlock,
                            exit_selector: F)
                            -> BlockAnd<()>
        where F: FnOnce(&mut LoopScope) -> BasicBlock
    {
        let (exit_block, extent) = {
            let loop_scope = self.find_loop_scope(span, label);
            (exit_selector(loop_scope), loop_scope.extent)
        };
        self.exit_scope(span, extent, block, exit_block);
        self.cfg.start_new_block().unit()
    }

}
