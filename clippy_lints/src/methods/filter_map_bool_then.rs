use super::FILTER_MAP_BOOL_THEN;
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::SpanRangeExt;
use clippy_utils::ty::is_copy;
use clippy_utils::{is_from_proc_macro, is_trait_method, peel_blocks};
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind};
use rustc_lint::{LateContext, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_middle::ty::Binder;
use rustc_middle::ty::adjustment::Adjust;
use rustc_span::{Span, sym};

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
        // Issue #11309
        && let param_ty = cx.tcx.liberate_late_bound_regions(
            closure.def_id.to_def_id(),
            Binder::bind_with_vars(
                cx.typeck_results().node_type(param_ty.hir_id),
                cx.tcx.late_bound_vars(cx.tcx.local_def_id_to_hir_id(closure.def_id)),
            ),
        )
        && is_copy(cx, param_ty)
        && let ExprKind::MethodCall(_, recv, [then_arg], _) = value.kind
        && let ExprKind::Closure(then_closure) = then_arg.kind
        && let then_body = peel_blocks(cx.tcx.hir().body(then_closure.body).value)
        && let Some(def_id) = cx.typeck_results().type_dependent_def_id(value.hir_id)
        && cx.tcx.is_diagnostic_item(sym::bool_then, def_id)
        && !is_from_proc_macro(cx, expr)
        // Count the number of derefs needed to get to the bool because we need those in the suggestion
        && let needed_derefs = cx.typeck_results().expr_adjustments(recv)
            .iter()
            .filter(|adj| matches!(adj.kind, Adjust::Deref(_)))
            .count()
        && let Some(param_snippet) = param.span.get_source_text(cx)
        && let Some(filter) = recv.span.get_source_text(cx)
        && let Some(map) = then_body.span.get_source_text(cx)
    {
        span_lint_and_sugg(
            cx,
            FILTER_MAP_BOOL_THEN,
            call_span,
            "usage of `bool::then` in `filter_map`",
            "use `filter` then `map` instead",
            format!(
                "filter(|&{param_snippet}| {derefs}{filter}).map(|{param_snippet}| {map})",
                derefs = "*".repeat(needed_derefs)
            ),
            Applicability::MachineApplicable,
        );
    }
}
