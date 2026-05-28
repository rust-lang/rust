use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::res::MaybeDef;
use clippy_utils::{is_integer_literal, sym};
use rustc_errors::Applicability;
use rustc_hir::{Expr, LangItem};
use rustc_lint::LateContext;
use rustc_span::Span;

use super::MANUAL_CLEAR;

pub(super) fn check(cx: &LateContext<'_>, expr: &Expr<'_>, recv: &Expr<'_>, arg: &Expr<'_>, method_span: Span) {
    let ty = cx.typeck_results().expr_ty_adjusted(recv);
    let ty = ty.peel_refs();

    let diag_name = ty.ty_adt_def().and_then(|def| cx.tcx.get_diagnostic_name(def.did()));

    if (matches!(diag_name, Some(sym::Vec | sym::VecDeque | sym::OsString)) || ty.is_lang_item(cx, LangItem::String))
        && is_integer_literal(arg, 0)
    {
        span_lint_and_then(cx, MANUAL_CLEAR, expr.span, "truncating to zero length", |diag| {
            // Keep the receiver as-is and only rewrite the method.
            diag.span_suggestion_verbose(
                method_span.with_hi(expr.span.hi()),
                "use `clear()` instead",
                "clear()",
                Applicability::MachineApplicable,
            );
        });
    }
}
