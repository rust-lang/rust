use crate::methods::utils::derefs_to_slice;
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::ty::is_type_diagnostic_item;
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_lint::LateContext;
use rustc_span::sym;

use super::ITER_CLONED_COLLECT;

pub(super) fn check<'tcx>(cx: &LateContext<'tcx>, expr: &hir::Expr<'_>, iter_args: &'tcx [hir::Expr<'_>]) {
    if_chain! {
        if is_type_diagnostic_item(cx, cx.typeck_results().expr_ty(expr), sym::vec_type);
        if let Some(slice) = derefs_to_slice(cx, &iter_args[0], cx.typeck_results().expr_ty(&iter_args[0]));
        if let Some(to_replace) = expr.span.trim_start(slice.span.source_callsite());

        then {
            span_lint_and_sugg(
                cx,
                ITER_CLONED_COLLECT,
                to_replace,
                "called `iter().cloned().collect()` on a slice to create a `Vec`. Calling `to_vec()` is both faster and \
                more readable",
                "try",
                ".to_vec()".to_string(),
                Applicability::MachineApplicable,
            );
        }
    }
}
