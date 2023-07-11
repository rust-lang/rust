use crate::{lints::FnNullCheckDiag, LateContext, LateLintPass, LintContext};
use rustc_ast::LitKind;
use rustc_hir::{BinOpKind, Expr, ExprKind, TyKind};
use rustc_session::{declare_lint, declare_lint_pass};
use rustc_span::sym;

declare_lint! {
    /// The `incorrect_fn_null_checks` lint checks for expression that checks if a
    /// function pointer is null.
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
    /// Function pointers are assumed to be non-null, checking them for null will always
    /// return false.
    INCORRECT_FN_NULL_CHECKS,
    Warn,
    "incorrect checking of null function pointer"
}

declare_lint_pass!(IncorrectFnNullChecks => [INCORRECT_FN_NULL_CHECKS]);

fn is_fn_ptr_cast(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
    let mut expr = expr.peel_blocks();
    let mut had_at_least_one_cast = false;
    while let ExprKind::Cast(cast_expr, cast_ty) = expr.kind
            && let TyKind::Ptr(_) = cast_ty.kind {
        expr = cast_expr.peel_blocks();
        had_at_least_one_cast = true;
    }
    had_at_least_one_cast && cx.typeck_results().expr_ty_adjusted(expr).is_fn()
}

impl<'tcx> LateLintPass<'tcx> for IncorrectFnNullChecks {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        match expr.kind {
            // Catching:
            // <*<const/mut> <ty>>::is_null(fn_ptr as *<const/mut> <ty>)
            ExprKind::Call(path, [arg])
                if let ExprKind::Path(ref qpath) = path.kind
                    && let Some(def_id) = cx.qpath_res(qpath, path.hir_id).opt_def_id()
                    && matches!(
                        cx.tcx.get_diagnostic_name(def_id),
                        Some(sym::ptr_const_is_null | sym::ptr_is_null)
                    )
                    && is_fn_ptr_cast(cx, arg) =>
            {
                cx.emit_spanned_lint(INCORRECT_FN_NULL_CHECKS, expr.span, FnNullCheckDiag)
            }

            // Catching:
            // (fn_ptr as *<const/mut> <ty>).is_null()
            ExprKind::MethodCall(_, receiver, _, _)
                if let Some(def_id) = cx.typeck_results().type_dependent_def_id(expr.hir_id)
                    && matches!(
                        cx.tcx.get_diagnostic_name(def_id),
                        Some(sym::ptr_const_is_null | sym::ptr_is_null)
                    )
                    && is_fn_ptr_cast(cx, receiver) =>
            {
                cx.emit_spanned_lint(INCORRECT_FN_NULL_CHECKS, expr.span, FnNullCheckDiag)
            }

            ExprKind::Binary(op, left, right) if matches!(op.node, BinOpKind::Eq) => {
                let to_check: &Expr<'_>;
                if is_fn_ptr_cast(cx, left) {
                    to_check = right;
                } else if is_fn_ptr_cast(cx, right) {
                    to_check = left;
                } else {
                    return;
                }

                match to_check.kind {
                    // Catching:
                    // (fn_ptr as *<const/mut> <ty>) == (0 as <ty>)
                    ExprKind::Cast(cast_expr, _)
                        if let ExprKind::Lit(spanned) = cast_expr.kind
                            && let LitKind::Int(v, _) = spanned.node && v == 0 =>
                    {
                        cx.emit_spanned_lint(INCORRECT_FN_NULL_CHECKS, expr.span, FnNullCheckDiag)
                    },

                    // Catching:
                    // (fn_ptr as *<const/mut> <ty>) == std::ptr::null()
                    ExprKind::Call(path, [])
                        if let ExprKind::Path(ref qpath) = path.kind
                            && let Some(def_id) = cx.qpath_res(qpath, path.hir_id).opt_def_id()
                            && let Some(diag_item) = cx.tcx.get_diagnostic_name(def_id)
                            && (diag_item == sym::ptr_null || diag_item == sym::ptr_null_mut) =>
                    {
                        cx.emit_spanned_lint(INCORRECT_FN_NULL_CHECKS, expr.span, FnNullCheckDiag)
                    },

                    _ => {},
                }
            }
            _ => {}
        }
    }
}
