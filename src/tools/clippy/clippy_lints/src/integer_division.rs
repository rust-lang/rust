use clippy_utils::diagnostics::span_lint_and_help;
use if_chain::if_chain;
use rustc_hir as hir;
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for division of integers
    ///
    /// ### Why is this bad?
    /// When outside of some very specific algorithms,
    /// integer division is very often a mistake because it discards the
    /// remainder.
    ///
    /// ### Example
    /// ```rust
    /// // Bad
    /// let x = 3 / 2;
    /// println!("{}", x);
    ///
    /// // Good
    /// let x = 3f32 / 2f32;
    /// println!("{}", x);
    /// ```
    #[clippy::version = "1.37.0"]
    pub INTEGER_DIVISION,
    restriction,
    "integer division may cause loss of precision"
}

declare_lint_pass!(IntegerDivision => [INTEGER_DIVISION]);

impl<'tcx> LateLintPass<'tcx> for IntegerDivision {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx hir::Expr<'_>) {
        if is_integer_division(cx, expr) {
            span_lint_and_help(
                cx,
                INTEGER_DIVISION,
                expr.span,
                "integer division",
                None,
                "division of integers may cause loss of precision. consider using floats",
            );
        }
    }
}

fn is_integer_division<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx hir::Expr<'_>) -> bool {
    if_chain! {
        if let hir::ExprKind::Binary(binop, left, right) = &expr.kind;
        if binop.node == hir::BinOpKind::Div;
        then {
            let (left_ty, right_ty) = (cx.typeck_results().expr_ty(left), cx.typeck_results().expr_ty(right));
            return left_ty.is_integral() && right_ty.is_integral();
        }
    }

    false
}
