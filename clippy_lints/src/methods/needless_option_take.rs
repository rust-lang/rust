use clippy_utils::diagnostics::span_lint;
use clippy_utils::ty::is_type_diagnostic_item;
use rustc_hir::Expr;
use rustc_lint::LateContext;
use rustc_span::sym;

use super::NEEDLESS_OPTION_TAKE;

pub(super) fn check<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>, _recv: &'tcx Expr<'_>) {
    // Checks if expression type is equal to sym::Option and if the expr is not a syntactic place
    if is_expr_option(cx, expr) && !expr.is_syntactic_place_expr() {
        span_lint(cx, OPTION_TAKE_ON_TEMPORARY, expr.span, "Format test");
    }
    /*    if_chain! {
        is_expr_option(cx, expr);
        then {
            span_lint(
                cx,
                NEEDLESS_OPTION_TAKE,
                expr.span,
                "Format test"
            );
        }
    };*/
}

fn is_expr_option(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
    let expr_type = cx.typeck_results().expr_ty(expr);
    is_type_diagnostic_item(cx, expr_type, sym::Option)
}
