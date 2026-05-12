use rustc_hir as hir;
use rustc_hir::{ExprKind, HirId};
use rustc_infer::traits::{IfChainCoerceCause, ObligationCauseCode};
use rustc_middle::bug;
use rustc_middle::ty::{Ty, TypeVisitableExt};
use rustc_span::Span;
use rustc_span::def_id::LocalDefId;

use crate::Expectation;
use crate::coercion::CoerceMany;
use crate::diverges::Diverges;
use crate::fn_ctxt::FnCtxt;

/// State shared across all branches of an `else if` chain check.
struct ElseIfChainCx<'a, 'tcx> {
    coerce: &'a mut CoerceMany<'tcx>,
    outer_if_expr_id: HirId,
    orig_expected: Expectation<'tcx>,
    tail_defines_return_position_impl_trait: Option<LocalDefId>,
}

#[derive(Copy, Clone)]
struct PrevBranch<'tcx> {
    hir_id: HirId,
    ty: Ty<'tcx>,
}

fn chain_has_terminal_else_block(expr: &hir::Expr<'_>) -> bool {
    let mut cur = expr;
    loop {
        match cur.kind {
            ExprKind::If(_, _, Some(next)) => cur = next,
            ExprKind::If(_, _, None) => return false,
            _ => return true,
        }
    }
}

fn has_empty_block_chains(expr: &hir::Expr<'_>) -> bool {
    let mut cur = expr;
    loop {
        match cur.kind {
            ExprKind::If(_, then, opt_else) => {
                let ExprKind::Block(then_block, _) = then.kind else {
                    return true;
                };
                if then_block.expr.is_none() {
                    return true;
                }
                match opt_else {
                    Some(next) => cur = next,
                    None => return false,
                }
            }
            ExprKind::Block(block, _) => return block.expr.is_none(),
            _ => return false,
        }
    }
}

impl<'a, 'tcx> FnCtxt<'a, 'tcx> {
    // A generic function for checking the 'then' and 'else' clauses in an 'if'
    // or 'if-else' expression.
    pub(crate) fn check_expr_if(
        &self,
        expr_id: HirId,
        cond_expr: &'tcx hir::Expr<'tcx>,
        then_expr: &'tcx hir::Expr<'tcx>,
        opt_else_expr: Option<&'tcx hir::Expr<'tcx>>,
        sp: Span,
        orig_expected: Expectation<'tcx>,
    ) -> Ty<'tcx> {
        let cond_ty = self.check_expr_has_type_or_error(cond_expr, self.tcx.types.bool, |_| {});

        let (expected, then_ty, cond_diverges, then_diverges) =
            self.check_current_if_branch(cond_expr, then_expr, orig_expected);

        // We've already taken the expected type's preferences
        // into account when typing the `then` branch. To figure
        // out the initial shot at a LUB, we thus only consider
        // `expected` if it represents a *hard* constraint
        // (`only_has_type`); otherwise, we just go with a
        // fresh type variable.
        let coerce_to_ty = expected.coercion_target_type(self, sp);
        let mut coerce = CoerceMany::with_capacity(coerce_to_ty, 2);

        coerce.coerce(self, &self.misc(sp), then_expr, then_ty);

        if let Some(else_expr) = opt_else_expr {
            let else_diverges = if chain_has_terminal_else_block(else_expr)
                && let ExprKind::If(..) = else_expr.kind
                // Chains containing `{}` arms go to the path below.
                // Otherwise for code like
                //     let x = if c1 { &() } else if c2 {} else {};
                // the "consider borrowing here" suggestion comes out as
                //     } else if c2 &{
                // which isn't valid syntax. Having an empty block in an if chain is an edge
                // case in real code, so the fallback is harmless.
                && !has_empty_block_chains(else_expr)
            {
                let mut cx = ElseIfChainCx {
                    coerce: &mut coerce,
                    outer_if_expr_id: expr_id,
                    orig_expected,
                    tail_defines_return_position_impl_trait: self
                        .return_position_impl_trait_from_match_expectation(orig_expected),
                };
                self.check_expr_with_check_fn(else_expr, expected, |this| {
                    this.check_else_if_branch(
                        &mut cx,
                        PrevBranch { hir_id: then_expr.hir_id, ty: then_ty },
                        else_expr,
                    )
                });
                self.diverges.get()
            } else {
                let else_ty = self.check_expr_with_expectation(else_expr, expected);
                let else_diverges = self.diverges.get();
                let if_cause = self.if_cause(
                    expr_id,
                    else_expr,
                    self.return_position_impl_trait_from_match_expectation(orig_expected),
                );
                coerce.coerce(self, &if_cause, else_expr, else_ty);
                else_diverges
            };

            // We won't diverge unless both branches do (or the condition does).
            self.diverges.set(cond_diverges | then_diverges & else_diverges);
        } else {
            self.if_fallback_coercion(sp, cond_expr, then_expr, &mut coerce);

            // If the condition is false we can't diverge.
            self.diverges.set(cond_diverges);
        }

        let result_ty = coerce.complete(self);
        if let Err(guar) = cond_ty.error_reported() {
            Ty::new_error(self.tcx, guar)
        } else {
            result_ty
        }
    }

    fn check_current_if_branch(
        &self,
        cond_expr: &'tcx hir::Expr<'tcx>,
        then_expr: &'tcx hir::Expr<'tcx>,
        orig_expected: Expectation<'tcx>,
    ) -> (Expectation<'tcx>, Ty<'tcx>, Diverges, Diverges) {
        self.warn_if_unreachable(
            cond_expr.hir_id,
            then_expr.span,
            "block in `if` or `while` expression",
        );

        let cond_diverges = self.diverges.get();
        self.diverges.set(Diverges::Maybe);

        let expected = orig_expected.try_structurally_resolve_and_adjust_for_branches(self);
        let then_ty = self.check_expr_with_expectation(then_expr, expected);
        let then_diverges = self.diverges.get();
        self.diverges.set(Diverges::Maybe);

        (expected, then_ty, cond_diverges, then_diverges)
    }

    fn check_else_if_branch(
        &self,
        cx: &mut ElseIfChainCx<'_, 'tcx>,
        prev_branch: PrevBranch<'tcx>,
        current_expr: &'tcx hir::Expr<'tcx>,
    ) -> Ty<'tcx> {
        let ExprKind::If(cond_expr, then_expr, opt_else_expr) = current_expr.kind else {
            bug!("check_else_if_branch called on non-`if` expr");
        };
        self.check_expr_has_type_or_error(cond_expr, self.tcx.types.bool, |_| {});

        let (expected, then_ty, cond_diverges, then_diverges) =
            self.check_current_if_branch(cond_expr, then_expr, cx.orig_expected);

        self.coerce_branch_against_prev(cx, prev_branch, then_expr, then_ty);

        let next_prev = PrevBranch { hir_id: then_expr.hir_id, ty: then_ty };

        let else_diverges = match opt_else_expr {
            Some(else_expr) if let ExprKind::If(..) = else_expr.kind => {
                self.check_expr_with_check_fn(else_expr, expected, |this| {
                    this.check_else_if_branch(cx, next_prev, else_expr)
                });
                self.diverges.get()
            }
            Some(else_expr) => {
                let else_ty = self.check_expr_with_expectation(else_expr, expected);
                let else_diverges = self.diverges.get();
                self.coerce_branch_against_prev(cx, next_prev, else_expr, else_ty);
                else_diverges
            }
            None => bug!("chain entered without a final else; broken `has_final_else_arm`"),
        };

        // We won't diverge unless cond does, or both then and else do.
        self.diverges.set(cond_diverges | then_diverges & else_diverges);

        cx.coerce.merged_ty()
    }

    fn coerce_branch_against_prev(
        &self,
        cx: &mut ElseIfChainCx<'_, 'tcx>,
        prev: PrevBranch<'tcx>,
        current_branch_expr: &'tcx hir::Expr<'tcx>,
        current_branch_ty: Ty<'tcx>,
    ) {
        let prev_branch_span = self.find_block_span_from_hir_id(prev.hir_id);
        let branch_span = self.find_block_span_from_hir_id(current_branch_expr.hir_id);
        let cause = self.cause(
            branch_span,
            ObligationCauseCode::IfChainCoerce(Box::new(IfChainCoerceCause {
                outer_if_expr_id: cx.outer_if_expr_id,
                source_branch_expr_id: prev.hir_id,
                source_branch_ty: prev.ty,
                source_branch_span: prev_branch_span,
                target_branch_expr_id: current_branch_expr.hir_id,
                target_branch_ty: current_branch_ty,
                target_branch_span: branch_span,
                tail_defines_return_position_impl_trait: cx.tail_defines_return_position_impl_trait,
            })),
        );
        cx.coerce.coerce(self, &cause, current_branch_expr, current_branch_ty);
    }
}
