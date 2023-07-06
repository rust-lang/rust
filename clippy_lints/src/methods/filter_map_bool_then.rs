use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::paths::BOOL_THEN;
use clippy_utils::source::snippet_opt;
use clippy_utils::{is_from_proc_macro, match_def_path, peel_blocks};
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind};
use rustc_lint::{LateContext, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_span::Span;

use super::FILTER_MAP_BOOL_THEN;

pub(super) fn check<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>, arg: &Expr<'_>, call_span: Span) {
    if !in_external_macro(cx.sess(), expr.span)
        && let ExprKind::Closure(closure) = arg.kind
        && let body = peel_blocks(cx.tcx.hir().body(closure.body).value)
        && let ExprKind::MethodCall(_, recv, [then_arg], _) = body.kind
        && let ExprKind::Closure(then_closure) = then_arg.kind
        && let then_body = peel_blocks(cx.tcx.hir().body(then_closure.body).value)
        && let Some(def_id) = cx.typeck_results().type_dependent_def_id(body.hir_id)
        && match_def_path(cx, def_id, &BOOL_THEN)
        && !is_from_proc_macro(cx, expr)
        && let Some(decl_snippet) = closure.fn_arg_span.and_then(|s| snippet_opt(cx, s))
        // NOTE: This will include the `()` (parenthesis) around it. Maybe there's some utils method
        // to remove them? `unused_parens` will already take care of this but it may be nice.
        && let Some(filter) = snippet_opt(cx, recv.span)
        && let Some(map) = snippet_opt(cx, then_body.span)
    {
        span_lint_and_sugg(
            cx,
            FILTER_MAP_BOOL_THEN,
            call_span,
            "usage of `bool::then` in `filter_map`",
            "use `filter` then `map` instead",
            format!("filter({decl_snippet} {filter}).map({decl_snippet} {map})"),
            Applicability::MachineApplicable,
        );
    }
}
