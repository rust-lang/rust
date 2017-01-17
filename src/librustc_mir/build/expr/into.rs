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
use hair::*;
use rustc::ty;
use rustc::mir::*;

impl<'a, 'gcx, 'tcx> Builder<'a, 'gcx, 'tcx> {
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
        let source_info = this.source_info(expr_span);

        match expr.kind {
            ExprKind::Scope { extent, value } => {
                this.in_scope(extent, block, |this| this.into(destination, block, value))
            }
            ExprKind::Block { body: ast_block } => {
                this.ast_block(destination, expr.ty.is_nil(), block, ast_block)
            }
            ExprKind::Match { discriminant, arms } => {
                this.match_expr(destination, expr_span, block, discriminant, arms)
            }
            ExprKind::NeverToAny { source } => {
                let source = this.hir.mirror(source);
                let is_call = match source.kind {
                    ExprKind::Call { .. } => true,
                    _ => false,
                };

                unpack!(block = this.as_rvalue(block, source));

                // This is an optimization. If the expression was a call then we already have an
                // unreachable block. Don't bother to terminate it and create a new one.
                if is_call {
                    block.unit()
                } else {
                    this.cfg.terminate(block, source_info, TerminatorKind::Unreachable);
                    let end_block = this.cfg.start_new_block();
                    end_block.unit()
                }
            }
            ExprKind::If { condition: cond_expr, then: then_expr, otherwise: else_expr } => {
                let operand = unpack!(block = this.as_operand(block, cond_expr));

                let mut then_block = this.cfg.start_new_block();
                let mut else_block = this.cfg.start_new_block();
                this.cfg.terminate(block, source_info, TerminatorKind::If {
                    cond: operand,
                    targets: (then_block, else_block)
                });

                unpack!(then_block = this.into(destination, then_block, then_expr));
                else_block = if let Some(else_expr) = else_expr {
                    unpack!(this.into(destination, else_block, else_expr))
                } else {
                    // Body of the `if` expression without an `else` clause must return `()`, thus
                    // we implicitly generate a `else {}` if it is not specified.
                    this.cfg.push_assign_unit(else_block, source_info, destination);
                    else_block
                };

                let join_block = this.cfg.start_new_block();
                this.cfg.terminate(then_block, source_info,
                                   TerminatorKind::Goto { target: join_block });
                this.cfg.terminate(else_block, source_info,
                                   TerminatorKind::Goto { target: join_block });

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
                this.cfg.terminate(block, source_info,
                                   TerminatorKind::If { cond: lhs, targets: blocks });

                let rhs = unpack!(else_block = this.as_operand(else_block, rhs));
                this.cfg.terminate(else_block, source_info, TerminatorKind::If {
                    cond: rhs,
                    targets: (true_block, false_block)
                });

                this.cfg.push_assign_constant(
                    true_block, source_info, destination,
                    Constant {
                        span: expr_span,
                        ty: this.hir.bool_ty(),
                        literal: this.hir.true_literal(),
                    });

                this.cfg.push_assign_constant(
                    false_block, source_info, destination,
                    Constant {
                        span: expr_span,
                        ty: this.hir.bool_ty(),
                        literal: this.hir.false_literal(),
                    });

                this.cfg.terminate(true_block, source_info,
                                   TerminatorKind::Goto { target: join_block });
                this.cfg.terminate(false_block, source_info,
                                   TerminatorKind::Goto { target: join_block });

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
                this.cfg.terminate(block, source_info,
                                   TerminatorKind::Goto { target: loop_block });

                this.in_loop_scope(
                    loop_block, exit_block, destination.clone(),
                    move |this| {
                        // conduct the test, if necessary
                        let body_block;
                        if let Some(cond_expr) = opt_cond_expr {
                            let loop_block_end;
                            let cond = unpack!(
                                loop_block_end = this.as_operand(loop_block, cond_expr));
                            body_block = this.cfg.start_new_block();
                            this.cfg.terminate(loop_block_end, source_info,
                                               TerminatorKind::If {
                                                   cond: cond,
                                                   targets: (body_block, exit_block)
                                               });

                            // if the test is false, there's no `break` to assign `destination`, so
                            // we have to do it; this overwrites any `break`-assigned value but it's
                            // always `()` anyway
                            this.cfg.push_assign_unit(exit_block, source_info, destination);
                        } else {
                            body_block = loop_block;
                        }

                        // The “return” value of the loop body must always be an unit. We therefore
                        // introduce a unit temporary as the destination for the loop body.
                        let tmp = this.get_unit_temp();
                        // Execute the body, branching back to the test.
                        let body_block_end = unpack!(this.into(&tmp, body_block, body));
                        this.cfg.terminate(body_block_end, source_info,
                                           TerminatorKind::Goto { target: loop_block });
                    }
                );
                exit_block.unit()
            }
            ExprKind::Call { ty, fun, args } => {
                let diverges = match ty.sty {
                    ty::TyFnDef(_, _, ref f) | ty::TyFnPtr(ref f) => {
                        // FIXME(canndrew): This is_never should probably be an is_uninhabited
                        f.sig.skip_binder().output().is_never()
                    }
                    _ => false
                };
                let fun = unpack!(block = this.as_operand(block, fun));
                let args: Vec<_> =
                    args.into_iter()
                        .map(|arg| unpack!(block = this.as_operand(block, arg)))
                        .collect();

                let success = this.cfg.start_new_block();
                let cleanup = this.diverge_cleanup();
                this.cfg.terminate(block, source_info, TerminatorKind::Call {
                    func: fun,
                    args: args,
                    cleanup: cleanup,
                    destination: if diverges {
                        None
                    } else {
                        Some ((destination.clone(), success))
                    }
                });
                success.unit()
            }

            // These cases don't actually need a destination
            ExprKind::Assign { .. } |
            ExprKind::AssignOp { .. } |
            ExprKind::Continue { .. } |
            ExprKind::Break { .. } |
            ExprKind::Return {.. } => {
                this.stmt_expr(block, expr)
            }

            // these are the cases that are more naturally handled by some other mode
            ExprKind::Unary { .. } |
            ExprKind::Binary { .. } |
            ExprKind::Box { .. } |
            ExprKind::Cast { .. } |
            ExprKind::Use { .. } |
            ExprKind::ReifyFnPointer { .. } |
            ExprKind::UnsafeFnPointer { .. } |
            ExprKind::Unsize { .. } |
            ExprKind::Repeat { .. } |
            ExprKind::Borrow { .. } |
            ExprKind::VarRef { .. } |
            ExprKind::SelfRef |
            ExprKind::StaticRef { .. } |
            ExprKind::Array { .. } |
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
                this.cfg.push_assign(block, source_info, destination, rvalue);
                block.unit()
            }
        }
    }
}
