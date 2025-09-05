use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::ty::is_type_diagnostic_item;
use clippy_utils::{is_res_lang_ctor, path_res};
use rustc_errors::Applicability;
use rustc_hir::lang_items::LangItem;
use rustc_hir::{Expr, ExprKind};
use rustc_lint::LateContext;
use rustc_span::{Span, sym};

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
        title = "found `.or(Some(…)).unwrap()`";
        if let Some(content) = get_content_if_ctor_matches(cx, or_arg, LangItem::OptionSome) {
            or_arg_content = content;
        } else {
            return;
        }
    } else if is_type_diagnostic_item(cx, ty, sym::Result) {
        title = "found `.or(Ok(…)).unwrap()`";
        if let Some(content) = get_content_if_ctor_matches(cx, or_arg, LangItem::ResultOk) {
            or_arg_content = content;
        } else {
            return;
        }
    } else {
        // Someone has implemented a struct with .or(...).unwrap() chaining,
        // but it's not an Option or a Result, so bail
        return;
    }

    let mut applicability = Applicability::MachineApplicable;
    let suggestion = format!(
        "unwrap_or({})",
        snippet_with_applicability(cx, or_arg_content, "..", &mut applicability)
    );

    span_lint_and_sugg(
        cx,
        OR_THEN_UNWRAP,
        unwrap_expr.span.with_lo(or_span.lo()),
        title,
        "try",
        suggestion,
        applicability,
    );
}

fn get_content_if_ctor_matches(cx: &LateContext<'_>, expr: &Expr<'_>, item: LangItem) -> Option<Span> {
    if let ExprKind::Call(some_expr, [arg]) = expr.kind
        && is_res_lang_ctor(cx, path_res(cx, some_expr), item)
    {
        Some(arg.span.source_callsite())
    } else {
        None
    }
}
