use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::is_expr_identity_function;
use clippy_utils::res::MaybeDef;
use clippy_utils::source::snippet_with_context;
use rustc_errors::Applicability;
use rustc_hir::Expr;
use rustc_lint::LateContext;
use rustc_span::Span;
use rustc_span::symbol::sym;

use super::MAP_OR_IDENTITY;

/// lint use of `_.map_or(err, |n| n)` for `Result`s and `Option`s.
pub(super) fn check(
    cx: &LateContext<'_>,
    expr: &Expr<'_>,
    recv: &Expr<'_>,
    call_span: Span,
    def_arg: &Expr<'_>,
    map_arg: &Expr<'_>,
) {
    // lint if the caller of `map_or()` is a `Result` or an `Option`
    // and if the mapping function is the identity function
    if let Some(symbol @ (sym::Result | sym::Option)) = cx.typeck_results.expr_ty_adjusted(recv).opt_diag_name(cx)
        && is_expr_identity_function(cx, map_arg)
    {
        let msg = format!("expression can be simplified using `{symbol}::unwrap_or()`");
        span_lint_and_then(cx, MAP_OR_IDENTITY, expr.span, msg, |diag| {
            let mut applicability = Applicability::MachineApplicable;
            let (err_snippet, _) = snippet_with_context(cx, def_arg.span, expr.span.ctxt(), "..", &mut applicability);
            let sugg = format!("unwrap_or({err_snippet})");

            diag.span_suggestion_verbose(call_span, "consider using `unwrap_or`", sugg, applicability);
        });
    }
}
