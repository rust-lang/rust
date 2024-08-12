use clippy_utils::consts::{ConstEvalCtxt, Constant};
use clippy_utils::diagnostics::span_lint_and_help;
use rustc_hir::{BinOpKind, Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::declare_lint_pass;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `0.0 / 0.0`.
    ///
    /// ### Why is this bad?
    /// It's less readable than `f32::NAN` or `f64::NAN`.
    ///
    /// ### Example
    /// ```no_run
    /// let nan = 0.0f32 / 0.0;
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// let nan = f32::NAN;
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub ZERO_DIVIDED_BY_ZERO,
    complexity,
    "usage of `0.0 / 0.0` to obtain NaN instead of `f32::NAN` or `f64::NAN`"
}

declare_lint_pass!(ZeroDiv => [ZERO_DIVIDED_BY_ZERO]);

impl<'tcx> LateLintPass<'tcx> for ZeroDiv {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        // check for instances of 0.0/0.0
        if let ExprKind::Binary(ref op, left, right) = expr.kind
            && op.node == BinOpKind::Div
            // TODO - constant_simple does not fold many operations involving floats.
            // That's probably fine for this lint - it's pretty unlikely that someone would
            // do something like 0.0/(2.0 - 2.0), but it would be nice to warn on that case too.
            && let ecx = ConstEvalCtxt::new(cx)
            && let Some(lhs_value) = ecx.eval_simple(left)
            && let Some(rhs_value) = ecx.eval_simple(right)
            // FIXME(f16_f128): add these types when eq is available on all platforms
            && (Constant::F32(0.0) == lhs_value || Constant::F64(0.0) == lhs_value)
            && (Constant::F32(0.0) == rhs_value || Constant::F64(0.0) == rhs_value)
        {
            // since we're about to suggest a use of f32::NAN or f64::NAN,
            // match the precision of the literals that are given.
            let float_type = match (lhs_value, rhs_value) {
                (Constant::F64(_), _) | (_, Constant::F64(_)) => "f64",
                _ => "f32",
            };
            span_lint_and_help(
                cx,
                ZERO_DIVIDED_BY_ZERO,
                expr.span,
                "constant division of `0.0` with `0.0` will always result in NaN",
                None,
                format!("consider using `{float_type}::NAN` if you would like a constant representing NaN",),
            );
        }
    }
}
