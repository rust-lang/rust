use crate::utils::span_help_and_lint;
use if_chain::if_chain;
use rustc::hir;
use rustc::lint::{LateContext, LateLintPass, LintArray, LintPass};
use rustc::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// **What it does:** Checks for division of integers
    ///
    /// **Why is this bad?** When outside of some very specific algorithms,
    /// integer division is very often a mistake because it discards the
    /// remainder.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust
    /// fn main() {
    ///     let x = 3 / 2;
    ///     println!("{}", x);
    /// }
    /// ```
    pub INTEGER_DIVISION,
    restriction,
    "integer division may cause loss of precision"
}

declare_lint_pass!(IntegerDivision => [INTEGER_DIVISION]);

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for IntegerDivision {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, expr: &'tcx hir::Expr) {
        if is_integer_division(cx, expr) {
            span_help_and_lint(
                cx,
                INTEGER_DIVISION,
                expr.span,
                "integer division",
                "division of integers may cause loss of precision. consider using floats.",
            );
        }
    }
}

fn is_integer_division<'a, 'tcx>(cx: &LateContext<'a, 'tcx>, expr: &'tcx hir::Expr) -> bool {
    if_chain! {
        if let hir::ExprKind::Binary(binop, left, right) = &expr.node;
        if let hir::BinOpKind::Div = &binop.node;
        then {
            let (left_ty, right_ty) = (cx.tables.expr_ty(left), cx.tables.expr_ty(right));
            return left_ty.is_integral() && right_ty.is_integral();
        }
    }

    false
}
