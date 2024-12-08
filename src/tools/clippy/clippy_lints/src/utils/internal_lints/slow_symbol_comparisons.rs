use clippy_utils::consts::{ConstEvalCtxt, Constant};
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::ty::match_type;
use clippy_utils::{match_function_call, paths};
use rustc_errors::Applicability;
use rustc_hir::{BinOpKind, Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::declare_lint_pass;
use rustc_span::Span;

declare_clippy_lint! {
    /// ### What it does
    ///
    /// Detects symbol comparision using `Symbol::intern`.
    ///
    /// ### Why is this bad?
    ///
    /// Comparision via `Symbol::as_str()` is faster if the interned symbols are not reused.
    ///
    /// ### Example
    ///
    /// None, see suggestion.
    pub SLOW_SYMBOL_COMPARISONS,
    internal,
    "detects slow comparisions of symbol"
}

declare_lint_pass!(SlowSymbolComparisons => [SLOW_SYMBOL_COMPARISONS]);

fn check_slow_comparison<'tcx>(
    cx: &LateContext<'tcx>,
    op1: &'tcx Expr<'tcx>,
    op2: &'tcx Expr<'tcx>,
) -> Option<(Span, String)> {
    if match_type(cx, cx.typeck_results().expr_ty(op1), &paths::SYMBOL)
        && let Some([symbol_name_expr]) = match_function_call(cx, op2, &paths::SYMBOL_INTERN)
        && let Some(Constant::Str(symbol_name)) = ConstEvalCtxt::new(cx).eval_simple(symbol_name_expr)
    {
        Some((op1.span, symbol_name))
    } else {
        None
    }
}

impl<'tcx> LateLintPass<'tcx> for SlowSymbolComparisons {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &Expr<'tcx>) {
        if let ExprKind::Binary(op, left, right) = expr.kind
            && (op.node == BinOpKind::Eq || op.node == BinOpKind::Ne)
            && let Some((symbol_span, symbol_name)) =
                check_slow_comparison(cx, left, right).or_else(|| check_slow_comparison(cx, right, left))
        {
            let mut applicability = Applicability::MachineApplicable;
            span_lint_and_sugg(
                cx,
                SLOW_SYMBOL_COMPARISONS,
                expr.span,
                "comparing `Symbol` via `Symbol::intern`",
                "use `Symbol::as_str` and check the string instead",
                format!(
                    "{}.as_str() {} \"{symbol_name}\"",
                    snippet_with_applicability(cx, symbol_span, "symbol", &mut applicability),
                    op.node.as_str()
                ),
                applicability,
            );
        };
    }
}
