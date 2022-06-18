use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::match_def_path;
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::ty::is_type_diagnostic_item;
use rustc_errors::Applicability;
use rustc_hir::Expr;
use rustc_lint::LateContext;
use rustc_span::sym;

use super::NEEDLESS_OPTION_TAKE;

pub(super) fn check<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>, recv: &'tcx Expr<'_>) {
    // Checks if expression type is equal to sym::Option and if the expr is not a syntactic place
    if !recv.is_syntactic_place_expr() && is_expr_option(cx, recv) && has_expr_as_ref_path(cx, recv) {
        let mut applicability = Applicability::MachineApplicable;
        span_lint_and_sugg(
            cx,
            NEEDLESS_OPTION_TAKE,
            expr.span,
            "called `Option::take()` on a temporary value",
            "try",
            format!(
                "{}",
                snippet_with_applicability(cx, recv.span, "..", &mut applicability)
            ),
            applicability,
        );
    }
}

fn is_expr_option(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
    let expr_type = cx.typeck_results().expr_ty(expr);
    is_type_diagnostic_item(cx, expr_type, sym::Option)
}

fn has_expr_as_ref_path(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
    if let Some(ref_id) = cx.typeck_results().type_dependent_def_id(expr.hir_id) {
        return match_def_path(cx, ref_id, &["core", "option", "Option", "as_ref"]);
    }
    false
}
