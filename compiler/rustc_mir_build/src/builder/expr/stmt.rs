use rustc_middle::middle::region;
use rustc_middle::mir::*;
use rustc_middle::span_bug;
use rustc_middle::thir::*;
use rustc_span::source_map::Spanned;
use tracing::debug;

use crate::builder::scope::BreakableTarget;
use crate::builder::{BlockAnd, BlockAndExtension, BlockFrame, Builder};

impl<'a, 'tcx> Builder<'a, 'tcx> {
    /// Builds a block of MIR statements to evaluate the THIR `expr`.
    ///
    /// The `statement_scope` is used if a statement temporary must be dropped.
    pub(crate) fn stmt_expr(
        &mut self,
        mut block: BasicBlock,
        expr_id: ExprId,
        statement_scope: Option<region::Scope>,
    ) -> BlockAnd<()> {
        let expr = &self.thir[expr_id];
        let expr_span = expr.span;
        let source_info = self.source_info(expr.span);
        // Handle a number of expressions that don't need a destination at all. This
        // avoids needing a mountain of temporary `()` variables.
        match expr.kind {
            ExprKind::Scope { region_scope, lint_level, value } => {
                self.in_scope((region_scope, source_info), lint_level, |this| {
                    this.stmt_expr(block, value, statement_scope)
                })
            }
            ExprKind::Assign { lhs, rhs } => {
                let lhs_expr = &self.thir[lhs];

                // Note: we evaluate assignments right-to-left. This
                // is better for borrowck interaction with overloaded
                // operators like x[j] = x[i].

                debug!("stmt_expr Assign block_context.push(SubExpr) : {:?}", expr);
                self.block_context.push(BlockFrame::SubExpr);

                // Generate better code for things that don't need to be
                // dropped.
                if lhs_expr.ty.needs_drop(self.tcx, self.typing_env()) {
                    let rhs = unpack!(block = self.as_local_rvalue(block, rhs));
                    let lhs = unpack!(block = self.as_place(block, lhs));
                    block =
                        self.build_drop_and_replace(block, lhs_expr.span, lhs, rhs).into_block();
                } else {
                    let rhs = unpack!(block = self.as_local_rvalue(block, rhs));
                    let lhs = unpack!(block = self.as_place(block, lhs));
                    self.cfg.push_assign(block, source_info, lhs, rhs);
                }

                self.block_context.pop();
                block.unit()
            }
            ExprKind::AssignOp { op, lhs, rhs } => {
                // FIXME(#28160) there is an interesting semantics
                // question raised here -- should we "freeze" the
                // value of the lhs here?  I'm inclined to think not,
                // since it seems closer to the semantics of the
                // overloaded version, which takes `&mut self`. This
                // only affects weird things like `x += {x += 1; x}`
                // -- is that equal to `x + (x + 1)` or `2*(x+1)`?

                let lhs_ty = self.thir[lhs].ty;

                debug!("stmt_expr AssignOp block_context.push(SubExpr) : {:?}", expr);
                self.block_context.push(BlockFrame::SubExpr);

                // As above, RTL.
                let rhs = unpack!(block = self.as_local_operand(block, rhs));
                let lhs = unpack!(block = self.as_place(block, lhs));

                // we don't have to drop prior contents or anything
                // because AssignOp is only legal for Copy types
                // (overloaded ops should be desugared into a call).
                let result = unpack!(
                    block = self.build_binary_op(
                        block,
                        op.into(),
                        expr_span,
                        lhs_ty,
                        Operand::Copy(lhs),
                        rhs
                    )
                );
                self.cfg.push_assign(block, source_info, lhs, result);

                self.block_context.pop();
                block.unit()
            }
            ExprKind::Continue { label } => {
                self.break_scope(block, None, BreakableTarget::Continue(label), source_info)
            }
            ExprKind::Break { label, value } => {
                self.break_scope(block, value, BreakableTarget::Break(label), source_info)
            }
            ExprKind::ConstContinue { label, value } => {
                self.break_const_continuable_scope(block, value, label, source_info)
            }
            ExprKind::Return { value } => {
                self.break_scope(block, value, BreakableTarget::Return, source_info)
            }
            ExprKind::Become { value } => {
                let v = &self.thir[value];
                let ExprKind::Scope { value, lint_level, region_scope } = v.kind else {
                    span_bug!(v.span, "`thir_check_tail_calls` should have disallowed this {v:?}")
                };

                let v = &self.thir[value];
                let ExprKind::Call { ref args, fun, fn_span, .. } = v.kind else {
                    span_bug!(v.span, "`thir_check_tail_calls` should have disallowed this {v:?}")
                };

                self.in_scope((region_scope, source_info), lint_level, |this| {
                    let fun = unpack!(block = this.as_local_operand(block, fun));
                    let args: Box<[_]> = args
                        .into_iter()
                        .copied()
                        .map(|arg| Spanned {
                            node: unpack!(block = this.as_local_call_operand(block, arg)),
                            span: this.thir.exprs[arg].span,
                        })
                        .collect();

                    this.record_operands_moved(&args);

                    debug!("expr_into_dest: fn_span={:?}", fn_span);

                    unpack!(block = this.break_for_tail_call(block, &args, source_info));

                    this.cfg.terminate(
                        block,
                        source_info,
                        TerminatorKind::TailCall { func: fun, args, fn_span },
                    );

                    this.cfg.start_new_block().unit()
                })
            }
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
                let adjusted_span = if let ExprKind::Block { block } = expr.kind
                    && let Some(tail_ex) = self.thir[block].expr
                {
                    let mut expr = &self.thir[tail_ex];
                    loop {
                        match expr.kind {
                            ExprKind::Block { block }
                                if let Some(nested_expr) = self.thir[block].expr =>
                            {
                                expr = &self.thir[nested_expr];
                            }
                            ExprKind::Scope { value: nested_expr, .. } => {
                                expr = &self.thir[nested_expr];
                            }
                            _ => break,
                        }
                    }
                    self.block_context.push(BlockFrame::TailExpr {
                        info: BlockTailInfo { tail_result_is_ignored: true, span: expr.span },
                    });
                    Some(expr.span)
                } else {
                    None
                };

                let temp = unpack!(
                    block = self.as_temp(
                        block,
                        TempLifetime {
                            temp_lifetime: statement_scope,
                            backwards_incompatible: None
                        },
                        expr_id,
                        Mutability::Not
                    )
                );

                if let Some(span) = adjusted_span {
                    self.local_decls[temp].source_info.span = span;
                    self.block_context.pop();
                }

                block.unit()
            }
        }
    }
}
