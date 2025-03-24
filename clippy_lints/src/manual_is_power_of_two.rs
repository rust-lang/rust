use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::sugg::Sugg;
use clippy_utils::{SpanlessEq, is_integer_literal};
use rustc_errors::Applicability;
use rustc_hir::{BinOpKind, Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty;
use rustc_session::declare_lint_pass;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for expressions like `x.count_ones() == 1` or `x & (x - 1) == 0`, with x and unsigned integer, which may be manual
    /// reimplementations of `x.is_power_of_two()`.
    ///
    /// ### Why is this bad?
    /// Manual reimplementations of `is_power_of_two` increase code complexity for little benefit.
    ///
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
    #[clippy::version = "1.83.0"]
    pub MANUAL_IS_POWER_OF_TWO,
    pedantic,
    "manually reimplementing `is_power_of_two`"
}

declare_lint_pass!(ManualIsPowerOfTwo => [MANUAL_IS_POWER_OF_TWO]);

impl<'tcx> LateLintPass<'tcx> for ManualIsPowerOfTwo {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &Expr<'tcx>) {
        if let ExprKind::Binary(bin_op, lhs, rhs) = expr.kind
            && bin_op.node == BinOpKind::Eq
        {
            if let Some(a) = count_ones_receiver(cx, lhs)
                && is_integer_literal(rhs, 1)
            {
                build_sugg(cx, expr, a);
            } else if let Some(a) = count_ones_receiver(cx, rhs)
                && is_integer_literal(lhs, 1)
            {
                build_sugg(cx, expr, a);
            } else if is_integer_literal(rhs, 0)
                && let Some(a) = is_and_minus_one(cx, lhs)
            {
                build_sugg(cx, expr, a);
            } else if is_integer_literal(lhs, 0)
                && let Some(a) = is_and_minus_one(cx, rhs)
            {
                build_sugg(cx, expr, a);
            }
        }
    }
}

fn build_sugg(cx: &LateContext<'_>, expr: &Expr<'_>, receiver: &Expr<'_>) {
    let mut applicability = Applicability::MachineApplicable;
    let snippet = Sugg::hir_with_applicability(cx, receiver, "_", &mut applicability);

    span_lint_and_sugg(
        cx,
        MANUAL_IS_POWER_OF_TWO,
        expr.span,
        "manually reimplementing `is_power_of_two`",
        "consider using `.is_power_of_two()`",
        format!("{}.is_power_of_two()", snippet.maybe_paren()),
        applicability,
    );
}

/// Return the unsigned integer receiver of `.count_ones()`
fn count_ones_receiver<'tcx>(cx: &LateContext<'tcx>, expr: &Expr<'tcx>) -> Option<&'tcx Expr<'tcx>> {
    if let ExprKind::MethodCall(method_name, receiver, [], _) = expr.kind
        && method_name.ident.as_str() == "count_ones"
        && matches!(cx.typeck_results().expr_ty_adjusted(receiver).kind(), ty::Uint(_))
    {
        Some(receiver)
    } else {
        None
    }
}

/// Return `greater` if `smaller == greater - 1`
fn is_one_less<'tcx>(
    cx: &LateContext<'tcx>,
    greater: &'tcx Expr<'tcx>,
    smaller: &Expr<'tcx>,
) -> Option<&'tcx Expr<'tcx>> {
    if let ExprKind::Binary(op, lhs, rhs) = smaller.kind
        && op.node == BinOpKind::Sub
        && SpanlessEq::new(cx).eq_expr(greater, lhs)
        && is_integer_literal(rhs, 1)
        && matches!(cx.typeck_results().expr_ty_adjusted(greater).kind(), ty::Uint(_))
    {
        Some(greater)
    } else {
        None
    }
}

/// Return `v` if `expr` is `v & (v - 1)` or `(v - 1) & v`
fn is_and_minus_one<'tcx>(cx: &LateContext<'tcx>, expr: &Expr<'tcx>) -> Option<&'tcx Expr<'tcx>> {
    if let ExprKind::Binary(op, lhs, rhs) = expr.kind
        && op.node == BinOpKind::BitAnd
    {
        is_one_less(cx, lhs, rhs).or_else(|| is_one_less(cx, rhs, lhs))
    } else {
        None
    }
}
