use crate::{lints::NonNullCheckDiag, LateContext, LateLintPass, LintContext};
use rustc_ast::LitKind;
use rustc_hir::{BinOpKind, Expr, ExprKind, TyKind};
use rustc_middle::ty::Ty;
use rustc_session::{declare_lint, declare_lint_pass};
use rustc_span::sym;

declare_lint! {
    /// The `incorrect_non_null_checks` lint checks for expressions that check if a
    /// non-nullable type is null.
    ///
    /// ### Example
    ///
    /// ```rust
    /// # fn test() {}
    /// let fn_ptr: fn() = /* somehow obtained nullable function pointer */
    /// #   test;
    ///
    /// if (fn_ptr as *const ()).is_null() { /* ... */ }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// A non-nullable type is assumed to never be null, and therefore having an actual
    /// non-null pointer is ub.
    INCORRECT_NON_NULL_CHECKS,
    Warn,
    "incorrect checking of non null pointers"
}

declare_lint_pass!(IncorrectNonNullChecks => [INCORRECT_NON_NULL_CHECKS]);

/// Is the cast to a nonnull type?
/// If yes, return (ty, nullable_version) where former is the nonnull type while latter
/// is a nullable version (e.g. (fn, Option<fn>) or (&u8, *const u8)).
fn is_nonnull_cast<'a>(cx: &LateContext<'a>, expr: &Expr<'_>) -> Option<Ty<'a>> {
    let mut expr = expr.peel_blocks();
    let mut had_at_least_one_cast = false;
    while let ExprKind::Cast(cast_expr, cast_ty) = expr.kind
            && let TyKind::Ptr(_) = cast_ty.kind {
        expr = cast_expr.peel_blocks();
        had_at_least_one_cast = true;
    }
    if !had_at_least_one_cast {
        return None;
    }
    let ty = cx.typeck_results().expr_ty_adjusted(expr);
    if ty.is_fn() || ty.is_ref() {
        return Some(ty);
    }
    // Usually, references get coerced to pointers in a casting situation.
    // Therefore, we give also give a look to the original type.
    let ty_unadjusted = cx.typeck_results().expr_ty_opt(expr);
    if let Some(ty_unadjusted) = ty_unadjusted && ty_unadjusted.is_ref() {
        return Some(ty_unadjusted);
    }
    None
}

impl<'tcx> LateLintPass<'tcx> for IncorrectNonNullChecks {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        match expr.kind {
            // Catching:
            // <*<const/mut> <ty>>::is_null(test_ptr as *<const/mut> <ty>)
            ExprKind::Call(path, [arg])
                if let ExprKind::Path(ref qpath) = path.kind
                    && let Some(def_id) = cx.qpath_res(qpath, path.hir_id).opt_def_id()
                    && matches!(
                        cx.tcx.get_diagnostic_name(def_id),
                        Some(sym::ptr_const_is_null | sym::ptr_is_null)
                    )
                    && let Some(ty) = is_nonnull_cast(cx, arg) =>
            {
                let diag = NonNullCheckDiag { ty_desc: ty.prefix_string(cx.tcx) };
                cx.emit_spanned_lint(INCORRECT_NON_NULL_CHECKS, expr.span, diag)
            }

            // Catching:
            // (test_ptr as *<const/mut> <ty>).is_null()
            ExprKind::MethodCall(_, receiver, _, _)
                if let Some(def_id) = cx.typeck_results().type_dependent_def_id(expr.hir_id)
                    && matches!(
                        cx.tcx.get_diagnostic_name(def_id),
                        Some(sym::ptr_const_is_null | sym::ptr_is_null)
                    )
                    && let Some(ty) = is_nonnull_cast(cx, receiver) =>
            {
                let diag = NonNullCheckDiag { ty_desc: ty.prefix_string(cx.tcx) };
                cx.emit_spanned_lint(INCORRECT_NON_NULL_CHECKS, expr.span, diag)
            }

            ExprKind::Binary(op, left, right) if matches!(op.node, BinOpKind::Eq) => {
                let to_check: &Expr<'_>;
                let ty: Ty<'_>;
                if let Some(ty_) = is_nonnull_cast(cx, left) {
                    to_check = right;
                    ty = ty_;
                } else if let Some(ty_) = is_nonnull_cast(cx, right) {
                    to_check = left;
                    ty = ty_;
                } else {
                    return;
                }

                match to_check.kind {
                    // Catching:
                    // (test_ptr as *<const/mut> <ty>) == (0 as <ty>)
                    ExprKind::Cast(cast_expr, _)
                        if let ExprKind::Lit(spanned) = cast_expr.kind
                            && let LitKind::Int(v, _) = spanned.node && v == 0 =>
                    {
                        let diag = NonNullCheckDiag { ty_desc: ty.prefix_string(cx.tcx) };
                        cx.emit_spanned_lint(INCORRECT_NON_NULL_CHECKS, expr.span, diag)
                    },

                    // Catching:
                    // (test_ptr as *<const/mut> <ty>) == std::ptr::null()
                    ExprKind::Call(path, [])
                        if let ExprKind::Path(ref qpath) = path.kind
                            && let Some(def_id) = cx.qpath_res(qpath, path.hir_id).opt_def_id()
                            && let Some(diag_item) = cx.tcx.get_diagnostic_name(def_id)
                            && (diag_item == sym::ptr_null || diag_item == sym::ptr_null_mut) =>
                    {
                        let diag = NonNullCheckDiag { ty_desc: ty.prefix_string(cx.tcx) };
                        cx.emit_spanned_lint(INCORRECT_NON_NULL_CHECKS, expr.span, diag)
                    },

                    _ => {},
                }
            }
            _ => {}
        }
    }
}
