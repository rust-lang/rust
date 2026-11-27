use rustc_hir as hir;
use rustc_hir::{ExprKind, HirId};
use rustc_middle::ty::{Ty, TypeVisitableExt};
use rustc_span::Span;
use rustc_span::def_id::LocalDefId;

use crate::Expectation;
use crate::coercion::CoerceMany;
use crate::diverges::Diverges;
use crate::fn_ctxt::FnCtxt;

/// State shared across all branches of an `if`/`else if` chain check.
struct IfChainCx<'a, 'tcx> {
    coerce: &'a mut CoerceMany<'tcx>,
    outer_if_expr_id: HirId,
    orig_expected: Expectation<'tcx>,
    tail_defines_return_position_impl_trait: Option<LocalDefId>,
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
        let expected = orig_expected.try_structurally_resolve_and_adjust_for_branches(self);

        let coerce_to_ty = expected.coercion_target_type(self, sp);
        let mut coerce = CoerceMany::with_capacity(coerce_to_ty, 2);

        let mut cx = IfChainCx {
            coerce: &mut coerce,
            outer_if_expr_id: expr_id,
            orig_expected,
            tail_defines_return_position_impl_trait: self
                .return_position_impl_trait_from_match_expectation(orig_expected),
        };
        let branch_ty =
            self.check_if_branch(&mut cx, None, cond_expr, then_expr, opt_else_expr, sp);

        let result_ty = coerce.complete(self);
        if let Err(guar) = branch_ty.error_reported() {
            Ty::new_error(self.tcx, guar)
        } else {
            result_ty
        }
    }

    /// Checks one `if` of an `if`/`else if` chain and recurses into the next `else if`.
    ///
    /// `prev_branch` is `None` for the outermost `if`.
    /// Every later branch is coerced against the preceding one (`prev_branch`),
    /// so a mismatch is reported at that branch rather than at the whole `else` chain.
    fn check_if_branch(
        &self,
        cx: &mut IfChainCx<'_, 'tcx>,
        prev_branch: Option<HirId>,
        cond_expr: &'tcx hir::Expr<'tcx>,
        then_expr: &'tcx hir::Expr<'tcx>,
        opt_else_expr: Option<&'tcx hir::Expr<'tcx>>,
        sp: Span,
    ) -> Ty<'tcx> {
        let cond_ty = self.check_expr_has_type_or_error(cond_expr, self.tcx.types.bool, |_| {});

        self.warn_if_unreachable(
            cond_expr.hir_id,
            then_expr.span,
            "block in `if` or `while` expression",
        );

        let cond_diverges = self.diverges.get();
        self.diverges.set(Diverges::Maybe);

        let expected = cx.orig_expected.try_structurally_resolve_and_adjust_for_branches(self);
        let then_ty = self.check_expr_with_expectation(then_expr, expected);
        let then_diverges = self.diverges.get();
        self.diverges.set(Diverges::Maybe);

        if let Some(prev_hir_id) = prev_branch {
            let cause = self.if_cause(
                cx.outer_if_expr_id,
                prev_hir_id,
                then_expr.hir_id,
                cx.tail_defines_return_position_impl_trait,
            );
            cx.coerce.coerce(self, &cause, then_expr, then_ty);
        } else {
            cx.coerce.coerce(self, &self.misc(sp), then_expr, then_ty);
        }

        match opt_else_expr {
            Some(else_expr)
                if let ExprKind::If(next_cond_expr, next_then_expr, next_opt_else_expr) =
                    else_expr.kind =>
            {
                self.check_expr_with_check_fn(else_expr, expected, |this| {
                    this.check_if_branch(
                        cx,
                        Some(then_expr.hir_id),
                        next_cond_expr,
                        next_then_expr,
                        next_opt_else_expr,
                        else_expr.span,
                    )
                });

                let else_diverges = self.diverges.get();
                self.diverges.set(cond_diverges | then_diverges & else_diverges);
            }
            Some(else_expr) => {
                let else_ty = self.check_expr_with_expectation(else_expr, expected);
                let else_diverges = self.diverges.get();
                let cause = self.if_cause(
                    cx.outer_if_expr_id,
                    then_expr.hir_id,
                    else_expr.hir_id,
                    cx.tail_defines_return_position_impl_trait,
                );
                cx.coerce.coerce(self, &cause, else_expr, else_ty);
                self.diverges.set(cond_diverges | then_diverges & else_diverges);
            }
            None => {
                self.if_fallback_coercion(sp, cond_expr, then_expr, cx.coerce);

                self.diverges.set(cond_diverges);
            }
        };

        if let Err(guar) = cond_ty.error_reported() {
            Ty::new_error(self.tcx, guar)
        } else {
            cx.coerce.merged_ty()
        }
    }
}
