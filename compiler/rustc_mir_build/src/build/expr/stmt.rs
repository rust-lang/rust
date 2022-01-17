use crate::build::scope::BreakableTarget;
use crate::build::{BlockAnd, BlockAndExtension, BlockFrame, Builder};
use rustc_middle::middle::region;
use rustc_middle::mir::*;
use rustc_middle::thir::*;

impl<'a, 'tcx> Builder<'a, 'tcx> {
    /// Builds a block of MIR statements to evaluate the THIR `expr`.
    /// If the original expression was an AST statement,
    /// (e.g., `some().code(&here());`) then `opt_stmt_span` is the
    /// span of that statement (including its semicolon, if any).
    /// The scope is used if a statement temporary must be dropped.
    crate fn stmt_expr(
        &mut self,
        mut block: BasicBlock,
        expr: &Expr<'tcx>,
        statement_scope: Option<region::Scope>,
    ) -> BlockAnd<()> {
        let this = self;
        let expr_span = expr.span;
        let source_info = this.source_info(expr.span);
        // Handle a number of expressions that don't need a destination at all. This
        // avoids needing a mountain of temporary `()` variables.
        match expr.kind {
            ExprKind::Scope { region_scope, lint_level, value } => {
                this.in_scope((region_scope, source_info), lint_level, |this| {
                    this.stmt_expr(block, &this.thir[value], statement_scope)
                })
            }
            ExprKind::Assign { lhs, rhs } => {
                let lhs = &this.thir[lhs];
                let rhs = &this.thir[rhs];
                let lhs_span = lhs.span;

                // Note: we evaluate assignments right-to-left. This
                // is better for borrowck interaction with overloaded
                // operators like x[j] = x[i].

                debug!("stmt_expr Assign block_context.push(SubExpr) : {:?}", expr);
                this.block_context.push(BlockFrame::SubExpr);

                // Generate better code for things that don't need to be
                // dropped.
                if lhs.ty.needs_drop(this.tcx, this.param_env) {
                    let rhs = unpack!(block = this.as_local_operand(block, rhs));
                    let lhs = unpack!(block = this.as_place(block, lhs));
                    unpack!(block = this.build_drop_and_replace(block, lhs_span, lhs, rhs));
                } else {
                    let rhs = unpack!(block = this.as_local_rvalue(block, rhs));
                    let lhs = unpack!(block = this.as_place(block, lhs));
                    this.cfg.push_assign(block, source_info, lhs, rhs);
                }

                this.block_context.pop();
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

                let lhs = &this.thir[lhs];
                let rhs = &this.thir[rhs];
                let lhs_ty = lhs.ty;

                debug!("stmt_expr AssignOp block_context.push(SubExpr) : {:?}", expr);
                this.block_context.push(BlockFrame::SubExpr);

                // As above, RTL.
                let rhs = unpack!(block = this.as_local_operand(block, rhs));
                let lhs = unpack!(block = this.as_place(block, lhs));

                // we don't have to drop prior contents or anything
                // because AssignOp is only legal for Copy types
                // (overloaded ops should be desugared into a call).
                let result = unpack!(
                    block =
                        this.build_binary_op(block, op, expr_span, lhs_ty, Operand::Copy(lhs), rhs)
                );
                this.cfg.push_assign(block, source_info, lhs, result);

                this.block_context.pop();
                block.unit()
            }
            ExprKind::Continue { label } => {
                this.break_scope(block, None, BreakableTarget::Continue(label), source_info)
            }
            ExprKind::Break { label, value } => this.break_scope(
                block,
                value.map(|value| &this.thir[value]),
                BreakableTarget::Break(label),
                source_info,
            ),
            ExprKind::Return { value } => this.break_scope(
                block,
                value.map(|value| &this.thir[value]),
                BreakableTarget::Return,
                source_info,
            ),
            _ => {
                assert!(
                    statement_scope.is_some(),
                    "Should not be calling `stmt_expr` on a general expression \
                     without a statement scope",
                );

                // Issue #54382: When creating temp for the value of
                // expression like:
                //
                // `{ side_effects(); { let l = stuff(); the_value } }`
                //
                // it is usually better to focus on `the_value` rather
                // than the entirety of block(s) surrounding it.
                let adjusted_span = (|| {
                    if let ExprKind::Block { body } = &expr.kind {
                        if let Some(tail_expr) = body.expr {
                            let mut expr = &this.thir[tail_expr];
                            while let ExprKind::Block {
                                body: Block { expr: Some(nested_expr), .. },
                            }
                            | ExprKind::Scope { value: nested_expr, .. } = expr.kind
                            {
                                expr = &this.thir[nested_expr];
                            }
                            this.block_context.push(BlockFrame::TailExpr {
                                tail_result_is_ignored: true,
                                span: expr.span,
                            });
                            return Some(expr.span);
                        }
                    }
                    None
                })();

                let temp =
                    unpack!(block = this.as_temp(block, statement_scope, expr, Mutability::Not));

                if let Some(span) = adjusted_span {
                    this.local_decls[temp].source_info.span = span;
                    this.block_context.pop();
                }

                block.unit()
            }
        }
    }
}
