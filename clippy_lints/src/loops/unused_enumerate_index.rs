use super::UNUSED_ENUMERATE_INDEX;
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::source::snippet;
use clippy_utils::ty::is_type_diagnostic_item;
use clippy_utils::{pat_is_wild, sugg};
use rustc_errors::Applicability;
use rustc_hir::def::DefKind;
use rustc_hir::{Expr, ExprKind, Pat, PatKind};
use rustc_lint::LateContext;
use rustc_span::sym;

/// Checks for the `UNUSED_ENUMERATE_INDEX` lint.
///
/// The lint is also partially implemented in `clippy_lints/src/methods/unused_enumerate_index.rs`.
pub(super) fn check<'tcx>(cx: &LateContext<'tcx>, pat: &Pat<'tcx>, arg: &Expr<'_>, body: &'tcx Expr<'tcx>) {
    if let PatKind::Tuple([index, elem], _) = pat.kind
        && let ExprKind::MethodCall(_method, self_arg, [], _) = arg.kind
        && let ty = cx.typeck_results().expr_ty(arg)
        && pat_is_wild(cx, &index.kind, body)
        && is_type_diagnostic_item(cx, ty, sym::Enumerate)
        && let Some((DefKind::AssocFn, call_id)) = cx.typeck_results().type_dependent_def(arg.hir_id)
        && cx.tcx.is_diagnostic_item(sym::enumerate_method, call_id)
    {
        span_lint_and_then(
            cx,
            UNUSED_ENUMERATE_INDEX,
            arg.span,
            "you seem to use `.enumerate()` and immediately discard the index",
            |diag| {
                let base_iter = sugg::Sugg::hir(cx, self_arg, "base iter");
                diag.multipart_suggestion(
                    "remove the `.enumerate()` call",
                    vec![
                        (pat.span, snippet(cx, elem.span, "..").into_owned()),
                        (arg.span, base_iter.to_string()),
                    ],
                    Applicability::MachineApplicable,
                );
            },
        );
    }
}
