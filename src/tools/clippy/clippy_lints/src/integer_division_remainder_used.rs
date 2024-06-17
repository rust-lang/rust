use clippy_utils::diagnostics::span_lint;
use rustc_ast::BinOpKind;
use rustc_hir::{Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::{self};
use rustc_session::declare_lint_pass;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for the usage of division (`/`) and remainder (`%`) operations
    /// when performed on any integer types using the default `Div` and `Rem` trait implementations.
    ///
    /// ### Why restrict this?
    /// In cryptographic contexts, division can result in timing sidechannel vulnerabilities,
    /// and needs to be replaced with constant-time code instead (e.g. Barrett reduction).
    ///
    /// ### Example
    /// ```no_run
    /// let my_div = 10 / 2;
    /// ```
    /// Use instead:
    /// ```no_run
    /// let my_div = 10 >> 1;
    /// ```
    #[clippy::version = "1.79.0"]
    pub INTEGER_DIVISION_REMAINDER_USED,
    restriction,
    "use of disallowed default division and remainder operations"
}

declare_lint_pass!(IntegerDivisionRemainderUsed => [INTEGER_DIVISION_REMAINDER_USED]);

impl LateLintPass<'_> for IntegerDivisionRemainderUsed {
    fn check_expr(&mut self, cx: &LateContext<'_>, expr: &Expr<'_>) {
        if let ExprKind::Binary(op, lhs, rhs) = &expr.kind
            && let BinOpKind::Div | BinOpKind::Rem = op.node
            && let lhs_ty = cx.typeck_results().expr_ty(lhs)
            && let rhs_ty = cx.typeck_results().expr_ty(rhs)
            && let ty::Int(_) | ty::Uint(_) = lhs_ty.peel_refs().kind()
            && let ty::Int(_) | ty::Uint(_) = rhs_ty.peel_refs().kind()
        {
            span_lint(
                cx,
                INTEGER_DIVISION_REMAINDER_USED,
                expr.span.source_callsite(),
                format!("use of {} has been disallowed in this context", op.node.as_str()),
            );
        }
    }
}
