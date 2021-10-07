use super::utils::derefs_to_slice;
use crate::methods::iter_nth_zero;
use clippy_utils::diagnostics::span_lint_and_help;
use clippy_utils::ty::is_type_diagnostic_item;
use rustc_hir as hir;
use rustc_lint::LateContext;
use rustc_span::symbol::sym;

use super::ITER_NTH;

pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &hir::Expr<'_>,
    iter_recv: &'tcx hir::Expr<'tcx>,
    nth_recv: &hir::Expr<'_>,
    nth_arg: &hir::Expr<'_>,
    is_mut: bool,
) {
    let mut_str = if is_mut { "_mut" } else { "" };
    let caller_type = if derefs_to_slice(cx, iter_recv, cx.typeck_results().expr_ty(iter_recv)).is_some() {
        "slice"
    } else if is_type_diagnostic_item(cx, cx.typeck_results().expr_ty(iter_recv), sym::Vec) {
        "Vec"
    } else if is_type_diagnostic_item(cx, cx.typeck_results().expr_ty(iter_recv), sym::VecDeque) {
        "VecDeque"
    } else {
        iter_nth_zero::check(cx, expr, nth_recv, nth_arg);
        return; // caller is not a type that we want to lint
    };

    span_lint_and_help(
        cx,
        ITER_NTH,
        expr.span,
        &format!("called `.iter{0}().nth()` on a {1}", mut_str, caller_type),
        None,
        &format!("calling `.get{}()` is both faster and more readable", mut_str),
    );
}
