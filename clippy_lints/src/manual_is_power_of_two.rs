use clippy_utils::SpanlessEq;
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet_with_applicability;
use rustc_ast::LitKind;
use rustc_data_structures::packed::Pu128;
use rustc_errors::Applicability;
use rustc_hir::{BinOpKind, Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::Uint;
use rustc_session::declare_lint_pass;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for expressions like `x.count_ones() == 1` or `x & (x - 1) == 0`, with x and unsigned integer, which are manual
    /// reimplementations of `x.is_power_of_two()`.
    /// ### Why is this bad?
    /// Manual reimplementations of `is_power_of_two` increase code complexity for little benefit.
    /// ### Example
    /// ```no_run
    /// let a: u32 = 4;
    /// let result = a.count_ones() == 1;
    /// ```
    /// Use instead:
    /// ```no_run
    /// let a: u32 = 4;
    /// let result = a.is_power_of_two();
    /// ```
    #[clippy::version = "1.82.0"]
    pub MANUAL_IS_POWER_OF_TWO,
    complexity,
    "manually reimplementing `is_power_of_two`"
}

declare_lint_pass!(ManualIsPowerOfTwo => [MANUAL_IS_POWER_OF_TWO]);

impl LateLintPass<'_> for ManualIsPowerOfTwo {
    fn check_expr(&mut self, cx: &LateContext<'_>, expr: &Expr<'_>) {
        let mut applicability = Applicability::MachineApplicable;

        if let ExprKind::Binary(bin_op, left, right) = expr.kind
            && bin_op.node == BinOpKind::Eq
        {
            // a.count_ones() == 1
            if let ExprKind::MethodCall(method_name, reciever, _, _) = left.kind
                && method_name.ident.as_str() == "count_ones"
                && let &Uint(_) = cx.typeck_results().expr_ty(reciever).kind()
                && check_lit(right, 1)
            {
                build_sugg(cx, expr, reciever, &mut applicability);
            }

            // 1 == a.count_ones()
            if let ExprKind::MethodCall(method_name, reciever, _, _) = right.kind
                && method_name.ident.as_str() == "count_ones"
                && let &Uint(_) = cx.typeck_results().expr_ty(reciever).kind()
                && check_lit(left, 1)
            {
                build_sugg(cx, expr, reciever, &mut applicability);
            }

            // a & (a - 1) == 0
            if let ExprKind::Binary(op1, left1, right1) = left.kind
                && op1.node == BinOpKind::BitAnd
                && let ExprKind::Binary(op2, left2, right2) = right1.kind
                && op2.node == BinOpKind::Sub
                && check_eq_expr(cx, left1, left2)
                && let &Uint(_) = cx.typeck_results().expr_ty(left1).kind()
                && check_lit(right2, 1)
                && check_lit(right, 0)
            {
                build_sugg(cx, expr, left1, &mut applicability);
            }

            // (a - 1) & a == 0;
            if let ExprKind::Binary(op1, left1, right1) = left.kind
                && op1.node == BinOpKind::BitAnd
                && let ExprKind::Binary(op2, left2, right2) = left1.kind
                && op2.node == BinOpKind::Sub
                && check_eq_expr(cx, right1, left2)
                && let &Uint(_) = cx.typeck_results().expr_ty(right1).kind()
                && check_lit(right2, 1)
                && check_lit(right, 0)
            {
                build_sugg(cx, expr, right1, &mut applicability);
            }

            // 0 == a & (a - 1);
            if let ExprKind::Binary(op1, left1, right1) = right.kind
                && op1.node == BinOpKind::BitAnd
                && let ExprKind::Binary(op2, left2, right2) = right1.kind
                && op2.node == BinOpKind::Sub
                && check_eq_expr(cx, left1, left2)
                && let &Uint(_) = cx.typeck_results().expr_ty(left1).kind()
                && check_lit(right2, 1)
                && check_lit(left, 0)
            {
                build_sugg(cx, expr, left1, &mut applicability);
            }

            // 0 == (a - 1) & a
            if let ExprKind::Binary(op1, left1, right1) = right.kind
                && op1.node == BinOpKind::BitAnd
                && let ExprKind::Binary(op2, left2, right2) = left1.kind
                && op2.node == BinOpKind::Sub
                && check_eq_expr(cx, right1, left2)
                && let &Uint(_) = cx.typeck_results().expr_ty(right1).kind()
                && check_lit(right2, 1)
                && check_lit(left, 0)
            {
                build_sugg(cx, expr, right1, &mut applicability);
            }
        }
    }
}

fn build_sugg(cx: &LateContext<'_>, expr: &Expr<'_>, reciever: &Expr<'_>, applicability: &mut Applicability) {
    let snippet = snippet_with_applicability(cx, reciever.span, "..", applicability);

    span_lint_and_sugg(
        cx,
        MANUAL_IS_POWER_OF_TWO,
        expr.span,
        "manually reimplementing `is_power_of_two`",
        "consider using `.is_power_of_two()`",
        format!("{snippet}.is_power_of_two()"),
        *applicability,
    );
}

fn check_lit(expr: &Expr<'_>, expected_num: u128) -> bool {
    if let ExprKind::Lit(lit) = expr.kind
        && let LitKind::Int(Pu128(num), _) = lit.node
        && num == expected_num
    {
        return true;
    }
    false
}

fn check_eq_expr(cx: &LateContext<'_>, lhs: &Expr<'_>, rhs: &Expr<'_>) -> bool {
    SpanlessEq::new(cx).eq_expr(lhs, rhs)
}
