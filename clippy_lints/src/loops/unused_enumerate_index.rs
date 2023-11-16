use super::UNUSED_ENUMERATE_INDEX;
use clippy_utils::diagnostics::{multispan_sugg, span_lint_and_then};
use clippy_utils::source::snippet;
use clippy_utils::{pat_is_wild, sugg};
use rustc_hir::def::DefKind;
use rustc_hir::{Expr, ExprKind, Pat, PatKind};
use rustc_lint::LateContext;
use rustc_middle::ty;

/// Checks for the `UNUSED_ENUMERATE_INDEX` lint.
pub(super) fn check<'tcx>(cx: &LateContext<'tcx>, pat: &'tcx Pat<'_>, arg: &'tcx Expr<'_>, body: &'tcx Expr<'_>) {
    let PatKind::Tuple([index, elem], _) = pat.kind else {
        return;
    };

    let ExprKind::MethodCall(_method, self_arg, [], _) = arg.kind else {
        return;
    };

    let ty = cx.typeck_results().expr_ty(arg);

    if !pat_is_wild(cx, &index.kind, body) {
        return;
    }

    let name = match *ty.kind() {
        ty::Adt(base, _substs) => cx.tcx.def_path_str(base.did()),
        _ => return,
    };

    if name != "std::iter::Enumerate" && name != "core::iter::Enumerate" {
        return;
    }

    let Some((DefKind::AssocFn, call_id)) = cx.typeck_results().type_dependent_def(arg.hir_id) else {
        return;
    };

    let call_name = cx.tcx.def_path_str(call_id);

    if call_name != "std::iter::Iterator::enumerate" && call_name != "core::iter::Iterator::enumerate" {
        return;
    }

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
