use clippy_utils::consts::{constant, Constant};
use clippy_utils::diagnostics::span_lint_and_help;
use clippy_utils::{is_integer_literal, is_path_diagnostic_item};
use rustc_hir::{BinOpKind, Expr, ExprKind, TyKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::sym;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for comparing a function pointer to null.
    ///
    /// ### Why is this bad?
    /// Function pointers are assumed to not be null.
    ///
    /// ### Example
    /// ```rust,ignore
    /// let fn_ptr: fn() = /* somehow obtained nullable function pointer */
    ///
    /// if (fn_ptr as *const ()).is_null() { ... }
    /// ```
    /// Use instead:
    /// ```rust,ignore
    /// let fn_ptr: Option<fn()> = /* somehow obtained nullable function pointer */
    ///
    /// if fn_ptr.is_none() { ... }
    /// ```
    #[clippy::version = "1.68.0"]
    pub FN_NULL_CHECK,
    correctness,
    "`fn()` type assumed to be nullable"
}
declare_lint_pass!(FnNullCheck => [FN_NULL_CHECK]);

fn lint_expr(cx: &LateContext<'_>, expr: &Expr<'_>) {
    span_lint_and_help(
        cx,
        FN_NULL_CHECK,
        expr.span,
        "function pointer assumed to be nullable, even though it isn't",
        None,
        "try wrapping your function pointer type in `Option<T>` instead, and using `is_none` to check for null pointer value",
    );
}

fn is_fn_ptr_cast(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
    if let ExprKind::Cast(cast_expr, cast_ty) = expr.kind
        && let TyKind::Ptr(_) = cast_ty.kind
    {
        cx.typeck_results().expr_ty_adjusted(cast_expr).is_fn()
    } else {
        false
    }
}

impl<'tcx> LateLintPass<'tcx> for FnNullCheck {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        match expr.kind {
            // Catching:
            // (fn_ptr as *<const/mut> <ty>).is_null()
            ExprKind::MethodCall(method_name, receiver, _, _)
                if method_name.ident.as_str() == "is_null" && is_fn_ptr_cast(cx, receiver) =>
            {
                lint_expr(cx, expr);
            },

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
                    ExprKind::Cast(cast_expr, _) if is_integer_literal(cast_expr, 0) => {
                        lint_expr(cx, expr);
                    },

                    // Catching:
                    // (fn_ptr as *<const/mut> <ty>) == std::ptr::null()
                    ExprKind::Call(func, []) if is_path_diagnostic_item(cx, func, sym::ptr_null) => {
                        lint_expr(cx, expr);
                    },

                    // Catching:
                    // (fn_ptr as *<const/mut> <ty>) == <const that evaluates to null_ptr>
                    _ if matches!(
                        constant(cx, cx.typeck_results(), to_check),
                        Some((Constant::RawPtr(0), _))
                    ) =>
                    {
                        lint_expr(cx, expr);
                    },

                    _ => {},
                }
            },
            _ => {},
        }
    }
}
