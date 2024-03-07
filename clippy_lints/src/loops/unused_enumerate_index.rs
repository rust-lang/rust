use super::UNUSED_ENUMERATE_INDEX;
use clippy_utils::diagnostics::{multispan_sugg, span_lint_and_then};
use clippy_utils::source::snippet;
use clippy_utils::{match_def_path, pat_is_wild, sugg};
use rustc_hir::def::DefKind;
use rustc_hir::{Expr, ExprKind, Pat, PatKind};
use rustc_lint::LateContext;
use rustc_middle::ty;

/// Checks for the `UNUSED_ENUMERATE_INDEX` lint.
pub(super) fn check<'tcx>(cx: &LateContext<'tcx>, pat: &Pat<'tcx>, arg: &Expr<'_>, body: &'tcx Expr<'tcx>) {
    if let PatKind::Tuple([index, elem], _) = pat.kind
        && let ExprKind::MethodCall(_method, self_arg, [], _) = arg.kind
        && let ty = cx.typeck_results().expr_ty(arg)
        && pat_is_wild(cx, &index.kind, body)
        && let ty::Adt(base, _) = *ty.kind()
        && match_def_path(cx, base.did(), &clippy_utils::paths::CORE_ITER_ENUMERATE_STRUCT)
        && let Some((DefKind::AssocFn, call_id)) = cx.typeck_results().type_dependent_def(arg.hir_id)
        && match_def_path(cx, call_id, &clippy_utils::paths::CORE_ITER_ENUMERATE_METHOD)
    {
        span_lint_and_then(
            cx,
            UNUSED_ENUMERATE_INDEX,
            arg.span,
            "you seem to use `.enumerate()` and immediately discard the index",
            |diag| {
                let base_iter = sugg::Sugg::hir(cx, self_arg, "base iter");
                multispan_sugg(
                    diag,
                    "remove the `.enumerate()` call",
                    vec![
                        (pat.span, snippet(cx, elem.span, "..").into_owned()),
                        (arg.span, base_iter.to_string()),
                    ],
                );
            },
        );
    }
}
