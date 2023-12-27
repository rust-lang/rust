use clippy_utils::diagnostics::span_lint;
use rustc_hir::{BinOpKind, Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty;
use rustc_session::declare_lint_pass;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for comparisons with an address of a function item.
    ///
    /// ### Why is this bad?
    /// Function item address is not guaranteed to be unique and could vary
    /// between different code generation units. Furthermore different function items could have
    /// the same address after being merged together.
    ///
    /// ### Example
    /// ```no_run
    /// type F = fn();
    /// fn a() {}
    /// let f: F = a;
    /// if f == a {
    ///     // ...
    /// }
    /// ```
    #[clippy::version = "1.44.0"]
    pub FN_ADDRESS_COMPARISONS,
    correctness,
    "comparison with an address of a function item"
}

declare_lint_pass!(UnnamedAddress => [FN_ADDRESS_COMPARISONS]);

impl LateLintPass<'_> for UnnamedAddress {
    fn check_expr(&mut self, cx: &LateContext<'_>, expr: &Expr<'_>) {
        fn is_comparison(binop: BinOpKind) -> bool {
            matches!(
                binop,
                BinOpKind::Eq | BinOpKind::Lt | BinOpKind::Le | BinOpKind::Ne | BinOpKind::Ge | BinOpKind::Gt
            )
        }

        fn is_fn_def(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
            matches!(cx.typeck_results().expr_ty(expr).kind(), ty::FnDef(..))
        }

        if let ExprKind::Binary(binop, left, right) = expr.kind
            && is_comparison(binop.node)
            && cx.typeck_results().expr_ty_adjusted(left).is_fn_ptr()
            && cx.typeck_results().expr_ty_adjusted(right).is_fn_ptr()
            && (is_fn_def(cx, left) || is_fn_def(cx, right))
        {
            span_lint(
                cx,
                FN_ADDRESS_COMPARISONS,
                expr.span,
                "comparing with a non-unique address of a function item",
            );
        }
    }
}
