use super::UNUSED_ENUMERATE_INDEX;
use clippy_utils::diagnostics::{multispan_sugg, span_lint_and_then};
use clippy_utils::source::snippet;
use clippy_utils::sugg;
use clippy_utils::visitors::is_local_used;
use rustc_hir::{Expr, ExprKind, Pat, PatKind};
use rustc_lint::LateContext;
use rustc_middle::ty;

/// Checks for the `UNUSED_ENUMERATE_INDEX` lint.
pub(super) fn check<'tcx>(cx: &LateContext<'tcx>, pat: &'tcx Pat<'_>, arg: &'tcx Expr<'_>, body: &'tcx Expr<'_>) {
    let pat_span = pat.span;

    let PatKind::Tuple(pat, _) = pat.kind else {
        return;
    };

    if pat.len() != 2 {
        return;
    }

    let arg_span = arg.span;

    let ExprKind::MethodCall(method, self_arg, [], _) = arg.kind else {
        return;
    };

    if method.ident.as_str() != "enumerate" {
        return;
    }

    let ty = cx.typeck_results().expr_ty(arg);

    if !pat_is_wild(cx, &pat[0].kind, body) {
        return;
    }

    let new_pat_span = pat[1].span;

    let name = match *ty.kind() {
        ty::Adt(base, _substs) => cx.tcx.def_path_str(base.did()),
        _ => return,
    };

    if name != "std::iter::Enumerate" && name != "core::iter::Enumerate" {
        return;
    }

    span_lint_and_then(
        cx,
        UNUSED_ENUMERATE_INDEX,
        arg_span,
        "you seem to use `.enumerate()` and immediately discard the index",
        |diag| {
            let base_iter = sugg::Sugg::hir(cx, self_arg, "base iter");
            multispan_sugg(
                diag,
                "remove the `.enumerate()` call",
                vec![
                    (pat_span, snippet(cx, new_pat_span, "value").into_owned()),
                    (arg_span, base_iter.to_string()),
                ],
            );
        },
    );
}

/// Returns `true` if the pattern is a `PatWild` or an ident prefixed with `_`.
fn pat_is_wild<'tcx>(cx: &LateContext<'tcx>, pat: &'tcx PatKind<'_>, body: &'tcx Expr<'_>) -> bool {
    match *pat {
        PatKind::Wild => true,
        PatKind::Binding(_, id, ident, None) if ident.as_str().starts_with('_') => !is_local_used(cx, body, id),
        _ => false,
    }
}
