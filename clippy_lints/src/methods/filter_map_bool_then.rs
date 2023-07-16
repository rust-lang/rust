use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::paths::BOOL_THEN;
use clippy_utils::source::snippet_opt;
use clippy_utils::ty::is_copy;
use clippy_utils::{is_from_proc_macro, is_trait_method, match_def_path, peel_blocks};
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind};
use rustc_lint::{LateContext, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_span::{sym, Span};

use super::FILTER_MAP_BOOL_THEN;

pub(super) fn check<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>, arg: &Expr<'_>, call_span: Span) {
    if !in_external_macro(cx.sess(), expr.span)
        && is_trait_method(cx, expr, sym::Iterator)
        && let ExprKind::Closure(closure) = arg.kind
        && let body = cx.tcx.hir().body(closure.body)
        && let value = peel_blocks(body.value)
        // Indexing should be fine as `filter_map` always has 1 input, we unfortunately need both
        // `inputs` and `params` here as we need both the type and the span
        && let param_ty = closure.fn_decl.inputs[0]
        && let param = body.params[0]
        && is_copy(cx, cx.typeck_results().node_type(param_ty.hir_id).peel_refs())
        && let ExprKind::MethodCall(_, recv, [then_arg], _) = value.kind
        && let ExprKind::Closure(then_closure) = then_arg.kind
        && let then_body = peel_blocks(cx.tcx.hir().body(then_closure.body).value)
        && let Some(def_id) = cx.typeck_results().type_dependent_def_id(value.hir_id)
        && match_def_path(cx, def_id, &BOOL_THEN)
        && !is_from_proc_macro(cx, expr)
        && let Some(param_snippet) = snippet_opt(cx, param.span)
        && let Some(filter) = snippet_opt(cx, recv.span)
        && let Some(map) = snippet_opt(cx, then_body.span)
    {
        span_lint_and_sugg(
            cx,
            FILTER_MAP_BOOL_THEN,
            call_span,
            "usage of `bool::then` in `filter_map`",
            "use `filter` then `map` instead",
            format!("filter(|&{param_snippet}| {filter}).map(|{param_snippet}| {map})"),
            Applicability::MachineApplicable,
        );
    }
}
