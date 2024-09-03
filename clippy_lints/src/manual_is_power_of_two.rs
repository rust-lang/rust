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
    /// Checks for expressions like `x.count_ones() == 1` or `x & (x - 1) == 0`, which are manual
    /// reimplementations of `x.is_power_of_two()`.
    /// ### Why is this bad?
    /// Manual reimplementations of `is_power_of_two` increase code complexity for little benefit.
    /// ### Example
    /// ```no_run
    /// let x: u32 = 1;
    /// let result = x.count_ones() == 1;
    /// ```
    /// Use instead:
    /// ```no_run
    /// let x: u32 = 1;
    /// let result = x.is_power_of_two();
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

        // x.count_ones() == 1
        if let ExprKind::Binary(op, left, right) = expr.kind
            && BinOpKind::Eq == op.node
            && let ExprKind::MethodCall(method_name, reciever, _, _) = left.kind
            && method_name.ident.as_str() == "count_ones"
            && let ExprKind::Lit(lit) = right.kind
            && let LitKind::Int(Pu128(1), _) = lit.node
            && let &Uint(_) = cx.typeck_results().expr_ty(reciever).kind()
        {
            let snippet = snippet_with_applicability(cx, reciever.span, "..", &mut applicability);
            let sugg = format!("{snippet}.is_power_of_two()");
            span_lint_and_sugg(
                cx,
                MANUAL_IS_POWER_OF_TWO,
                expr.span,
                "manually reimplementing `is_power_of_two`",
                "consider using `.is_power_of_two()`",
                sugg,
                applicability,
            );
        }

        // x & (x - 1) == 0
        if let ExprKind::Binary(op, left, right) = expr.kind
            && BinOpKind::Eq == op.node
            && let ExprKind::Binary(op1, left1, right1) = left.kind
            && BinOpKind::BitAnd == op1.node
            && let ExprKind::Binary(op2, left2, right2) = right1.kind
            && BinOpKind::Sub == op2.node
            && left1.span.eq_ctxt(left2.span)
            && let &Uint(_) = cx.typeck_results().expr_ty(left1).kind()
            && let ExprKind::Lit(lit) = right2.kind
            && let LitKind::Int(Pu128(1), _) = lit.node
            && let ExprKind::Lit(lit1) = right.kind
            && let LitKind::Int(Pu128(0), _) = lit1.node
        {
            let snippet = snippet_with_applicability(cx, left1.span, "..", &mut applicability);
            let sugg = format!("{snippet}.is_power_of_two()");
            span_lint_and_sugg(
                cx,
                MANUAL_IS_POWER_OF_TWO,
                expr.span,
                "manually reimplementing `is_power_of_two`",
                "consider using `.is_power_of_two()`",
                sugg,
                applicability,
            );
        }
    }
}
