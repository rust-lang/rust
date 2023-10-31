use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::ty::is_type_lang_item;
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_lint::LateContext;

use super::BYTES_COUNT_TO_LEN;

pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx hir::Expr<'_>,
    count_recv: &'tcx hir::Expr<'_>,
    bytes_recv: &'tcx hir::Expr<'_>,
) {
    if_chain! {
        if let Some(bytes_id) = cx.typeck_results().type_dependent_def_id(count_recv.hir_id);
        if let Some(impl_id) = cx.tcx.impl_of_method(bytes_id);
        if cx.tcx.type_of(impl_id).instantiate_identity().is_str();
        let ty = cx.typeck_results().expr_ty(bytes_recv).peel_refs();
        if ty.is_str() || is_type_lang_item(cx, ty, hir::LangItem::String);
        then {
            let mut applicability = Applicability::MachineApplicable;
            span_lint_and_sugg(
                cx,
                BYTES_COUNT_TO_LEN,
                expr.span,
                "using long and hard to read `.bytes().count()`",
                "consider calling `.len()` instead",
                format!("{}.len()", snippet_with_applicability(cx, bytes_recv.span, "..", &mut applicability)),
                applicability
            );
        }
    };
}
