use clippy_utils::diagnostics::span_lint_and_help;
use clippy_utils::ty::is_type_diagnostic_item;
use if_chain::if_chain;
use rustc_hir::{Expr, ExprKind, QPath};
use rustc_lint::LateContext;
use rustc_span::{sym, Span};

use super::OR_THEN_UNWRAP;

pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    unwrap_expr: &Expr<'_>,
    recv: &'tcx Expr<'tcx>,
    or_arg: &'tcx Expr<'_>,
    or_span: Span,
) {
    let ty = cx.typeck_results().expr_ty(recv); // get type of x (we later check if it's Option or Result)
    let title;

    if is_type_diagnostic_item(cx, ty, sym::Option) {
        title = ".or(Some(…)).unwrap() found";
        if !is(or_arg, "Some") {
            return;
        }
    } else if is_type_diagnostic_item(cx, ty, sym::Result) {
        title = ".or(Ok(…)).unwrap() found";
        if !is(or_arg, "Ok") {
            return;
        }
    } else {
        // Someone has implemented a struct with .or(...).unwrap() chaining,
        // but it's not an Option or a Result, so bail
        return;
    }

    let unwrap_span = if let ExprKind::MethodCall(_, _, span) = unwrap_expr.kind {
        span
    } else {
        // unreachable. but fallback to ident's span ("()" are missing)
        unwrap_expr.span
    };

    span_lint_and_help(
        cx,
        OR_THEN_UNWRAP,
        or_span.to(unwrap_span),
        title,
        None,
        "use `unwrap_or()` instead",
    );
}

/// is expr a Call to name?
/// name might be "Some", "Ok", "Err", etc.
fn is<'a>(expr: &Expr<'a>, name: &str) -> bool {
    if_chain! {
        if let ExprKind::Call(some_expr, _some_args) = expr.kind;
        if let ExprKind::Path(QPath::Resolved(_, path)) = &some_expr.kind;
        if let Some(path_segment) = path.segments.first();
        if path_segment.ident.name.as_str() == name;
        then {
            true
        }
        else {
            false
        }
    }
}
