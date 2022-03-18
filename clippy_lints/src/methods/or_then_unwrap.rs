use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::ty::is_type_diagnostic_item;
use if_chain::if_chain;
use rustc_errors::Applicability;
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
    let or_arg_content: Span;

    if is_type_diagnostic_item(cx, ty, sym::Option) {
        title = ".or(Some(…)).unwrap() found";
        if let Some(content) = get_content_if_is(or_arg, "Some") {
            or_arg_content = content;
        } else {
            return;
        }
    } else if is_type_diagnostic_item(cx, ty, sym::Result) {
        title = ".or(Ok(…)).unwrap() found";
        if let Some(content) = get_content_if_is(or_arg, "Ok") {
            or_arg_content = content;
        } else {
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

    let mut applicability = Applicability::MachineApplicable;
    let suggestion = format!(
        "unwrap_or({})",
        snippet_with_applicability(cx, or_arg_content, "..", &mut applicability)
    );

    span_lint_and_sugg(
        cx,
        OR_THEN_UNWRAP,
        or_span.to(unwrap_span),
        title,
        "try this",
        suggestion,
        applicability,
    );
}

/// is expr a Call to name? if so, return what it's wrapping
/// name might be "Some", "Ok", "Err", etc.
fn get_content_if_is<'a>(expr: &Expr<'a>, name: &str) -> Option<Span> {
    if_chain! {
        if let ExprKind::Call(some_expr, [arg]) = expr.kind;
        if let ExprKind::Path(QPath::Resolved(_, path)) = &some_expr.kind;
        if let Some(path_segment) = path.segments.first();
        if path_segment.ident.name.as_str() == name;
        then {
            Some(arg.span)
        }
        else {
            None
        }
    }
}
