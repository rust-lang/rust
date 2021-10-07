use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::ty::is_type_diagnostic_item;
use clippy_utils::{is_expr_identity_function, is_trait_method};
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_lint::LateContext;
use rustc_span::{source_map::Span, sym};

use super::MAP_IDENTITY;

pub(super) fn check(
    cx: &LateContext<'_>,
    expr: &hir::Expr<'_>,
    caller: &hir::Expr<'_>,
    map_arg: &hir::Expr<'_>,
    _map_span: Span,
) {
    let caller_ty = cx.typeck_results().expr_ty(caller);

    if_chain! {
        if is_trait_method(cx, expr, sym::Iterator)
            || is_type_diagnostic_item(cx, caller_ty, sym::Result)
            || is_type_diagnostic_item(cx, caller_ty, sym::Option);
        if is_expr_identity_function(cx, map_arg);
        if let Some(sugg_span) = expr.span.trim_start(caller.span);
        then {
            span_lint_and_sugg(
                cx,
                MAP_IDENTITY,
                sugg_span,
                "unnecessary map of the identity function",
                "remove the call to `map`",
                String::new(),
                Applicability::MachineApplicable,
            )
        }
    }
}
