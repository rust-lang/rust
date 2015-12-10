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
use build::expr::category::{Category, RvalueFunc};
use build::scope::LoopScope;
use hair::*;
use rustc::middle::region::CodeExtent;
use rustc::mir::repr::*;
use syntax::codemap::Span;

impl<'a,'tcx> Builder<'a,'tcx> {
    /// Compile `expr`, storing the result into `destination`, which
    /// is assumed to be uninitialized.
    pub fn into_expr(&mut self,
                     destination: &Lvalue<'tcx>,
                     mut block: BasicBlock,
                     expr: Expr<'tcx>)
                     -> BlockAnd<()>
    {
        debug!("into_expr(destination={:?}, block={:?}, expr={:?})",
               destination, block, expr);

        // since we frequently have to reference `self` from within a
        // closure, where `self` would be shadowed, it's easier to
        // just use the name `this` uniformly
        let this = self;
        let expr_span = expr.span;

        match expr.kind {
            ExprKind::Scope { extent, value } => {
                this.in_scope(extent, block, |this| this.into(destination, block, value))
            }
            ExprKind::Block { body: ast_block } => {
                this.ast_block(destination, block, ast_block)
            }
            ExprKind::Match { discriminant, arms } => {
                this.match_expr(destination, expr_span, block, discriminant, arms)
            }
            ExprKind::If { condition: cond_expr, then: then_expr, otherwise: else_expr } => {
                let operand = unpack!(block = this.as_operand(block, cond_expr));

                let mut then_block = this.cfg.start_new_block();
                let mut else_block = this.cfg.start_new_block();
                this.cfg.terminate(block, Terminator::If {
                    cond: operand,
                    targets: (then_block, else_block)
                });

                unpack!(then_block = this.into(destination, then_block, then_expr));
                unpack!(else_block = this.into(destination, else_block, else_expr));

                let join_block = this.cfg.start_new_block();
                this.cfg.terminate(then_block, Terminator::Goto { target: join_block });
                this.cfg.terminate(else_block, Terminator::Goto { target: join_block });

                join_block.unit()
            }
            ExprKind::LogicalOp { op, lhs, rhs } => {
                // And:
                //
                // [block: If(lhs)] -true-> [else_block: If(rhs)] -true-> [true_block]
                //        |                          | (false)
                //        +----------false-----------+------------------> [false_block]
                //
                // Or:
                //
                // [block: If(lhs)] -false-> [else_block: If(rhs)] -true-> [true_block]
                //        |                          | (false)
                //        +----------true------------+-------------------> [false_block]

                let (true_block, false_block, mut else_block, join_block) =
                    (this.cfg.start_new_block(), this.cfg.start_new_block(),
                     this.cfg.start_new_block(), this.cfg.start_new_block());

                let lhs = unpack!(block = this.as_operand(block, lhs));
                let blocks = match op {
                    LogicalOp::And => (else_block, false_block),
                    LogicalOp::Or => (true_block, else_block),
                };
                this.cfg.terminate(block, Terminator::If { cond: lhs, targets: blocks });

                let rhs = unpack!(else_block = this.as_operand(else_block, rhs));
                this.cfg.terminate(else_block, Terminator::If {
                    cond: rhs,
                    targets: (true_block, false_block)
                });

                this.cfg.push_assign_constant(
                    true_block, expr_span, destination,
                    Constant {
                        span: expr_span,
                        ty: this.hir.bool_ty(),
                        literal: this.hir.true_literal(),
                    });

                this.cfg.push_assign_constant(
                    false_block, expr_span, destination,
                    Constant {
                        span: expr_span,
                        ty: this.hir.bool_ty(),
                        literal: this.hir.false_literal(),
                    });

                this.cfg.terminate(true_block, Terminator::Goto { target: join_block });
                this.cfg.terminate(false_block, Terminator::Goto { target: join_block });

                join_block.unit()
            }
            ExprKind::Loop { condition: opt_cond_expr, body } => {
                // [block] --> [loop_block] ~~> [loop_block_end] -1-> [exit_block]
                //                  ^                  |
                //                  |                  0
                //                  |                  |
                //                  |                  v
                //           [body_block_end] <~~~ [body_block]
                //
                // If `opt_cond_expr` is `None`, then the graph is somewhat simplified:
                //
                // [block] --> [loop_block / body_block ] ~~> [body_block_end]    [exit_block]
                //                         ^                          |
                //                         |                          |
                //                         +--------------------------+
                //

                let loop_block = this.cfg.start_new_block();
                let exit_block = this.cfg.start_new_block();

                // start the loop
                this.cfg.terminate(block, Terminator::Goto { target: loop_block });

                this.in_loop_scope(loop_block, exit_block, |this| {
                    // conduct the test, if necessary
                    let body_block;
                    let opt_cond_expr = opt_cond_expr; // FIXME rustc bug
                    if let Some(cond_expr) = opt_cond_expr {
                        let loop_block_end;
                        let cond = unpack!(loop_block_end = this.as_operand(loop_block, cond_expr));
                        body_block = this.cfg.start_new_block();
                        this.cfg.terminate(loop_block_end,
                                           Terminator::If {
                                               cond: cond,
                                               targets: (body_block, exit_block)
                                           });
                    } else {
                        body_block = loop_block;
                    }

                    // execute the body, branching back to the test
                    let unit_temp = this.unit_temp.clone();
                    let body_block_end = unpack!(this.into(&unit_temp, body_block, body));
                    this.cfg.terminate(body_block_end, Terminator::Goto { target: loop_block });

                    // final point is exit_block
                    exit_block.unit()
                })
            }
            ExprKind::Assign { lhs, rhs } => {
                // Note: we evaluate assignments right-to-left. This
                // is better for borrowck interaction with overloaded
                // operators like x[j] = x[i].
                let rhs = unpack!(block = this.as_operand(block, rhs));
                let lhs = unpack!(block = this.as_lvalue(block, lhs));
                this.cfg.push_drop(block, expr_span, DropKind::Deep, &lhs);
                this.cfg.push_assign(block, expr_span, &lhs, Rvalue::Use(rhs));
                block.unit()
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
                this.cfg.push_assign(block, expr_span, &lhs,
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
                this.break_or_continue(expr_span, label, block, |loop_scope| loop_scope.break_block)
            }
            ExprKind::Return { value } => {
                unpack!(block = this.into(&Lvalue::ReturnPointer, block, value));
                let extent = this.extent_of_outermost_scope();
                this.exit_scope(expr_span, extent, block, END_BLOCK);
                this.cfg.start_new_block().unit()
            }
            ExprKind::Call { fun, args } => {
                let fun = unpack!(block = this.as_operand(block, fun));
                let args: Vec<_> =
                    args.into_iter()
                        .map(|arg| unpack!(block = this.as_operand(block, arg)))
                        .collect();
                let success = this.cfg.start_new_block();
                let panic = this.diverge_cleanup();
                this.cfg.terminate(block,
                                   Terminator::Call {
                                       data: CallData {
                                           destination: destination.clone(),
                                           func: fun,
                                           args: args,
                                       },
                                       targets: (success, panic),
                                   });
                success.unit()
            }

            // these are the cases that are more naturally handled by some other mode
            ExprKind::Unary { .. } |
            ExprKind::Binary { .. } |
            ExprKind::Box { .. } |
            ExprKind::Cast { .. } |
            ExprKind::ReifyFnPointer { .. } |
            ExprKind::UnsafeFnPointer { .. } |
            ExprKind::Unsize { .. } |
            ExprKind::Repeat { .. } |
            ExprKind::Borrow { .. } |
            ExprKind::VarRef { .. } |
            ExprKind::SelfRef |
            ExprKind::StaticRef { .. } |
            ExprKind::Vec { .. } |
            ExprKind::Tuple { .. } |
            ExprKind::Adt { .. } |
            ExprKind::Closure { .. } |
            ExprKind::Index { .. } |
            ExprKind::Deref { .. } |
            ExprKind::Literal { .. } |
            ExprKind::InlineAsm { .. } |
            ExprKind::Field { .. } => {
                debug_assert!(match Category::of(&expr.kind).unwrap() {
                    Category::Rvalue(RvalueFunc::Into) => false,
                    _ => true,
                });

                let rvalue = unpack!(block = this.as_rvalue(block, expr));
                this.cfg.push_assign(block, expr_span, destination, rvalue);
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
        where F: FnOnce(&LoopScope) -> BasicBlock
    {
        let loop_scope = self.find_loop_scope(span, label);
        let exit_block = exit_selector(&loop_scope);
        self.exit_scope(span, loop_scope.extent, block, exit_block);
        self.cfg.start_new_block().unit()
    }
}
