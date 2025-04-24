use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::ty::is_type_diagnostic_item;
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind, QPath};
use rustc_lint::LateContext;
use rustc_span::sym;

use super::NEEDLESS_OPTION_TAKE;

pub(super) fn check<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>, recv: &'tcx Expr<'_>) {
    // Checks if expression type is equal to sym::Option and if the expr is not a syntactic place
    if !recv.is_syntactic_place_expr()
        && is_expr_option(cx, recv)
        && let Some(function_name) = source_of_temporary_value(recv)
    {
        span_lint_and_then(
            cx,
            NEEDLESS_OPTION_TAKE,
            expr.span,
            "called `Option::take()` on a temporary value",
            |diag| {
                diag.note(format!(
                    "`{function_name}` creates a temporary value, so calling take() has no effect"
                ));
                diag.span_suggestion(
                    expr.span.with_lo(recv.span.hi()),
                    "remove",
                    "",
                    Applicability::MachineApplicable,
                );
            },
        );
    }
}

fn is_expr_option(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
    let expr_type = cx.typeck_results().expr_ty(expr);
    is_type_diagnostic_item(cx, expr_type, sym::Option)
}

/// Returns the string of the function call that creates the temporary.
/// When this function is called, we are reasonably certain that the `ExprKind` is either
/// `Call` or `MethodCall` because we already checked that the expression is not
/// `is_syntactic_place_expr()`.
fn source_of_temporary_value<'a>(expr: &'a Expr<'_>) -> Option<&'a str> {
    match expr.peel_borrows().kind {
        ExprKind::Call(function, _) => {
            if let ExprKind::Path(QPath::Resolved(_, func_path)) = function.kind
                && !func_path.segments.is_empty()
            {
                return Some(func_path.segments[0].ident.name.as_str());
            }
            if let ExprKind::Path(QPath::TypeRelative(_, func_path_segment)) = function.kind {
                return Some(func_path_segment.ident.name.as_str());
            }
            None
        },
        ExprKind::MethodCall(path_segment, ..) => Some(path_segment.ident.name.as_str()),
        _ => None,
    }
}
