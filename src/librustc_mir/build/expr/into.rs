//! See docs in build/expr/mod.rs

use crate::build::expr::category::{Category, RvalueFunc};
use crate::build::{BlockAnd, BlockAndExtension, BlockFrame, Builder};
use crate::hair::*;
use rustc::mir::*;
use rustc::ty;

use rustc_target::spec::abi::Abi;

impl<'a, 'tcx> Builder<'a, 'tcx> {
    /// Compile `expr`, storing the result into `destination`, which
    /// is assumed to be uninitialized.
    pub fn into_expr(
        &mut self,
        destination: &Place<'tcx>,
        mut block: BasicBlock,
        expr: Expr<'tcx>,
    ) -> BlockAnd<()> {
        debug!(
            "into_expr(destination={:?}, block={:?}, expr={:?})",
            destination, block, expr
        );

        // since we frequently have to reference `self` from within a
        // closure, where `self` would be shadowed, it's easier to
        // just use the name `this` uniformly
        let this = self;
        let expr_span = expr.span;
        let source_info = this.source_info(expr_span);

        let expr_is_block_or_scope = match expr.kind {
            ExprKind::Block { .. } => true,
            ExprKind::Scope { .. } => true,
            _ => false,
        };

        if !expr_is_block_or_scope {
            this.block_context.push(BlockFrame::SubExpr);
        }

        let block_and = match expr.kind {
            ExprKind::Scope {
                region_scope,
                lint_level,
                value,
            } => {
                let region_scope = (region_scope, source_info);
                this.in_scope(region_scope, lint_level, |this| {
                    this.into(destination, block, value)
                })
            }
            ExprKind::Block { body: ast_block } => {
                this.ast_block(destination, block, ast_block, source_info)
            }
            ExprKind::Match { scrutinee, arms } => {
                this.match_expr(destination, expr_span, block, scrutinee, arms)
            }
            ExprKind::NeverToAny { source } => {
                let source = this.hir.mirror(source);
                let is_call = match source.kind {
                    ExprKind::Call { .. } => true,
                    _ => false,
                };

                unpack!(block = this.as_local_rvalue(block, source));

                // This is an optimization. If the expression was a call then we already have an
                // unreachable block. Don't bother to terminate it and create a new one.
                if is_call {
                    block.unit()
                } else {
                    this.cfg
                        .terminate(block, source_info, TerminatorKind::Unreachable);
                    let end_block = this.cfg.start_new_block();
                    end_block.unit()
                }
            }
            ExprKind::LogicalOp { op, lhs, rhs } => {
                // And:
                //
                // [block: If(lhs)] -true-> [else_block: dest = (rhs)]
                //        | (false)
                //  [shortcurcuit_block: dest = false]
                //
                // Or:
                //
                // [block: If(lhs)] -false-> [else_block: dest = (rhs)]
                //        | (true)
                //  [shortcurcuit_block: dest = true]

                let (shortcircuit_block, mut else_block, join_block) = (
                    this.cfg.start_new_block(),
                    this.cfg.start_new_block(),
                    this.cfg.start_new_block(),
                );

                let lhs = unpack!(block = this.as_local_operand(block, lhs));
                let blocks = match op {
                    LogicalOp::And => (else_block, shortcircuit_block),
                    LogicalOp::Or => (shortcircuit_block, else_block),
                };
                let term = TerminatorKind::if_(this.hir.tcx(), lhs, blocks.0, blocks.1);
                this.cfg.terminate(block, source_info, term);

                this.cfg.push_assign_constant(
                    shortcircuit_block,
                    source_info,
                    destination,
                    Constant {
                        span: expr_span,
                        ty: this.hir.bool_ty(),
                        user_ty: None,
                        literal: match op {
                            LogicalOp::And => this.hir.false_literal(),
                            LogicalOp::Or => this.hir.true_literal(),
                        },
                    },
                );
                this.cfg.terminate(
                    shortcircuit_block,
                    source_info,
                    TerminatorKind::Goto { target: join_block },
                );

                let rhs = unpack!(else_block = this.as_local_operand(else_block, rhs));
                this.cfg.push_assign(
                    else_block,
                    source_info,
                    destination,
                    Rvalue::Use(rhs),
                );
                this.cfg.terminate(
                    else_block,
                    source_info,
                    TerminatorKind::Goto { target: join_block },
                );

                join_block.unit()
            }
            ExprKind::Loop {
                condition: opt_cond_expr,
                body,
            } => {
                // [block] --> [loop_block] -/eval. cond./-> [loop_block_end] -1-> [exit_block]
                //                  ^                               |
                //                  |                               0
                //                  |                               |
                //                  |                               v
                //           [body_block_end] <-/eval. body/-- [body_block]
                //
                // If `opt_cond_expr` is `None`, then the graph is somewhat simplified:
                //
                // [block]
                //    |
                //   [loop_block] -> [body_block] -/eval. body/-> [body_block_end]
                //    |        ^                                         |
                // false link  |                                         |
                //    |        +-----------------------------------------+
                //    +-> [diverge_cleanup]
                // The false link is required to make sure borrowck considers unwinds through the
                // body, even when the exact code in the body cannot unwind

                let loop_block = this.cfg.start_new_block();
                let exit_block = this.cfg.start_new_block();

                // start the loop
                this.cfg.terminate(
                    block,
                    source_info,
                    TerminatorKind::Goto { target: loop_block },
                );

                this.in_breakable_scope(
                    Some(loop_block),
                    exit_block,
                    destination.clone(),
                    move |this| {
                        // conduct the test, if necessary
                        let body_block;
                        if let Some(cond_expr) = opt_cond_expr {
                            let cond_expr = this.hir.mirror(cond_expr);
                            let (true_block, false_block)
                                = this.test_bool(loop_block, cond_expr, source_info);
                            body_block = true_block;

                            // if the test is false, there's no `break` to assign `destination`, so
                            // we have to do it
                            this.cfg.push_assign_unit(false_block, source_info, destination);
                            this.cfg.terminate(
                                false_block,
                                source_info,
                                TerminatorKind::Goto { target: exit_block },
                            );
                        } else {
                            body_block = this.cfg.start_new_block();
                            let diverge_cleanup = this.diverge_cleanup();
                            this.cfg.terminate(
                                loop_block,
                                source_info,
                                TerminatorKind::FalseUnwind {
                                    real_target: body_block,
                                    unwind: Some(diverge_cleanup),
                                },
                            )
                        }

                        // The “return” value of the loop body must always be an unit. We therefore
                        // introduce a unit temporary as the destination for the loop body.
                        let tmp = this.get_unit_temp();
                        // Execute the body, branching back to the test.
                        let body_block_end = unpack!(this.into(&tmp, body_block, body));
                        this.cfg.terminate(
                            body_block_end,
                            source_info,
                            TerminatorKind::Goto { target: loop_block },
                        );
                    },
                );
                exit_block.unit()
            }
            ExprKind::Call { ty, fun, args, from_hir_call } => {
                let intrinsic = match ty.sty {
                    ty::FnDef(def_id, _) => {
                        let f = ty.fn_sig(this.hir.tcx());
                        if f.abi() == Abi::RustIntrinsic || f.abi() == Abi::PlatformIntrinsic {
                            Some(this.hir.tcx().item_name(def_id).as_str())
                        } else {
                            None
                        }
                    }
                    _ => None,
                };
                let intrinsic = intrinsic.as_ref().map(|s| &s[..]);
                let fun = unpack!(block = this.as_local_operand(block, fun));
                if intrinsic == Some("move_val_init") {
                    // `move_val_init` has "magic" semantics - the second argument is
                    // always evaluated "directly" into the first one.

                    let mut args = args.into_iter();
                    let ptr = args.next().expect("0 arguments to `move_val_init`");
                    let val = args.next().expect("1 argument to `move_val_init`");
                    assert!(args.next().is_none(), ">2 arguments to `move_val_init`");

                    let ptr = this.hir.mirror(ptr);
                    let ptr_ty = ptr.ty;
                    // Create an *internal* temp for the pointer, so that unsafety
                    // checking won't complain about the raw pointer assignment.
                    let ptr_temp = this.local_decls.push(LocalDecl {
                        mutability: Mutability::Mut,
                        ty: ptr_ty,
                        user_ty: UserTypeProjections::none(),
                        name: None,
                        source_info,
                        visibility_scope: source_info.scope,
                        internal: true,
                        is_user_variable: None,
                        is_block_tail: None,
                    });
                    let ptr_temp = Place::from(ptr_temp);
                    let block = unpack!(this.into(&ptr_temp, block, ptr));
                    this.into(&ptr_temp.deref(), block, val)
                } else {
                    let args: Vec<_> = args
                        .into_iter()
                        .map(|arg| unpack!(block = this.as_local_operand(block, arg)))
                        .collect();

                    let success = this.cfg.start_new_block();
                    let cleanup = this.diverge_cleanup();
                    this.cfg.terminate(
                        block,
                        source_info,
                        TerminatorKind::Call {
                            func: fun,
                            args,
                            cleanup: Some(cleanup),
                            // FIXME(varkor): replace this with an uninhabitedness-based check.
                            // This requires getting access to the current module to call
                            // `tcx.is_ty_uninhabited_from`, which is currently tricky to do.
                            destination: if expr.ty.is_never() {
                                None
                            } else {
                                Some((destination.clone(), success))
                            },
                            from_hir_call,
                        },
                    );
                    success.unit()
                }
            }
            ExprKind::Use { source } => {
                this.into(destination, block, source)
            }

            // These cases don't actually need a destination
            ExprKind::Assign { .. }
            | ExprKind::AssignOp { .. }
            | ExprKind::Continue { .. }
            | ExprKind::Break { .. }
            | ExprKind::InlineAsm { .. }
            | ExprKind::Return { .. } => {
                unpack!(block = this.stmt_expr(block, expr, None));
                this.cfg.push_assign_unit(block, source_info, destination);
                block.unit()
            }

            // Avoid creating a temporary
            ExprKind::VarRef { .. } |
            ExprKind::SelfRef |
            ExprKind::StaticRef { .. } |
            ExprKind::PlaceTypeAscription { .. } |
            ExprKind::ValueTypeAscription { .. } => {
                debug_assert!(Category::of(&expr.kind) == Some(Category::Place));

                let place = unpack!(block = this.as_place(block, expr));
                let rvalue = Rvalue::Use(this.consume_by_copy_or_move(place));
                this.cfg
                    .push_assign(block, source_info, destination, rvalue);
                block.unit()
            }
            ExprKind::Index { .. } | ExprKind::Deref { .. } | ExprKind::Field { .. } => {
                debug_assert!(Category::of(&expr.kind) == Some(Category::Place));

                // Create a "fake" temporary variable so that we check that the
                // value is Sized. Usually, this is caught in type checking, but
                // in the case of box expr there is no such check.
                if let Place::Projection(..) = destination {
                    this.local_decls
                        .push(LocalDecl::new_temp(expr.ty, expr.span));
                }

                debug_assert!(Category::of(&expr.kind) == Some(Category::Place));

                let place = unpack!(block = this.as_place(block, expr));
                let rvalue = Rvalue::Use(this.consume_by_copy_or_move(place));
                this.cfg
                    .push_assign(block, source_info, destination, rvalue);
                block.unit()
            }

            // these are the cases that are more naturally handled by some other mode
            ExprKind::Unary { .. }
            | ExprKind::Binary { .. }
            | ExprKind::Box { .. }
            | ExprKind::Cast { .. }
            | ExprKind::Pointer { .. }
            | ExprKind::Repeat { .. }
            | ExprKind::Borrow { .. }
            | ExprKind::Array { .. }
            | ExprKind::Tuple { .. }
            | ExprKind::Adt { .. }
            | ExprKind::Closure { .. }
            | ExprKind::Literal { .. }
            | ExprKind::Yield { .. } => {
                debug_assert!(match Category::of(&expr.kind).unwrap() {
                    // should be handled above
                    Category::Rvalue(RvalueFunc::Into) => false,

                    // must be handled above or else we get an
                    // infinite loop in the builder; see
                    // e.g., `ExprKind::VarRef` above
                    Category::Place => false,

                    _ => true,
                });

                let rvalue = unpack!(block = this.as_local_rvalue(block, expr));
                this.cfg.push_assign(block, source_info, destination, rvalue);
                block.unit()
            }
        };

        if !expr_is_block_or_scope {
            let popped = this.block_context.pop();
            assert!(popped.is_some());
        }

        block_and
    }
}
