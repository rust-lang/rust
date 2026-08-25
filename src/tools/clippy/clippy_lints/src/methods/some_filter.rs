use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::source::{snippet_with_applicability, snippet_with_context};
use clippy_utils::{as_some_expr, pat_is_wild, peel_blocks, span_contains_comment};
use rustc_ast::util::parser::ExprPrecedence;
use rustc_errors::Applicability;
use rustc_hir::{Body, Expr, ExprKind};
use rustc_lint::LateContext;

use super::SOME_FILTER;

pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx Expr<'tcx>,
    recv: &'tcx Expr<'tcx>,
    arg: &'tcx Expr<'tcx>,
    msrv: Msrv,
) {
    let (condition, value) = if let Some(value) = as_some_expr(cx, recv)
        && let ExprKind::Closure(c) = arg.kind
        && let Body {
            params: [p],
            value: condition,
        } = cx.tcx.hir_body(c.body)
        && pat_is_wild(cx, &p.pat.kind, arg)
        && msrv.meets(cx, msrvs::BOOL_THEN_SOME)
    {
        (condition, value)
    } else {
        return;
    };
    span_lint_and_then(
        cx,
        SOME_FILTER,
        expr.span,
        "use of `Some(x).filter(|_| predicate)`",
        |diag| {
            let condition = if span_contains_comment(cx, condition.span) {
                condition
            } else {
                peel_blocks(condition)
            };
            let mut applicability = Applicability::MaybeIncorrect;
            let (condition_text, condition_is_macro) =
                snippet_with_context(cx, condition.span, arg.span.ctxt(), "_", &mut applicability);
            let parentheses = !condition_is_macro && cx.precedence(condition) < ExprPrecedence::Unambiguous;
            let value_text = snippet_with_applicability(cx, value.span, "_", &mut applicability);
            let sugg = format!(
                "{}{condition_text}{}.then_some({value_text})",
                if parentheses { "(" } else { "" },
                if parentheses { ")" } else { "" },
            );
            diag.span_suggestion_verbose(
                expr.span,
                "consider using `bool::then_some` instead",
                sugg,
                applicability,
            );
            diag.note(
                "this change will alter the order in which the condition and \
                 the value are evaluated",
            );
        },
    );
}
