use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet_with_context;
use clippy_utils::sym;
use clippy_utils::visitors::is_const_evaluatable;
use rustc_ast::ast::LitKind;
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind};
use rustc_lint::LateContext;

use super::STR_SPLIT_AT_NEWLINE;

pub(super) fn check<'a>(cx: &LateContext<'a>, expr: &'_ Expr<'_>, split_recv: &'a Expr<'_>, split_arg: &'_ Expr<'_>) {
    // We're looking for `A.trim().split(B)`, where the adjusted type of `A` is `&str` (e.g. an
    // expression returning `String`), and `B` is a `Pattern` that hard-codes a newline (either `"\n"`
    // or `"\r\n"`). There are a lot of ways to specify a pattern, and this lint only checks the most
    // basic ones: a `'\n'`, `"\n"`, and `"\r\n"`.
    if let ExprKind::MethodCall(trim_method_name, trim_recv, [], _) = split_recv.kind
        && trim_method_name.ident.name == sym::trim
        && cx.typeck_results().expr_ty_adjusted(trim_recv).peel_refs().is_str()
        && !is_const_evaluatable(cx, trim_recv)
        && let ExprKind::Lit(split_lit) = split_arg.kind
        && (matches!(split_lit.node, LitKind::Char('\n'))
            || matches!(split_lit.node, LitKind::Str(sym::LF | sym::CRLF, _)))
    {
        let mut app = Applicability::MaybeIncorrect;
        span_lint_and_sugg(
            cx,
            STR_SPLIT_AT_NEWLINE,
            expr.span,
            "using `str.trim().split()` with hard-coded newlines",
            "use `str.lines()` instead",
            format!(
                "{}.lines()",
                snippet_with_context(cx, trim_recv.span, expr.span.ctxt(), "..", &mut app).0
            ),
            app,
        );
    }
}
