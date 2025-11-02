use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::is_expr_identity_function;
use clippy_utils::res::{MaybeDef, MaybeTypeckRes};
use clippy_utils::source::SpanRangeExt;
use rustc_errors::Applicability;
use rustc_hir::Expr;
use rustc_lint::LateContext;
use rustc_span::{Span, sym};

use super::MAP_ALL_ANY_IDENTITY;

#[expect(clippy::too_many_arguments)]
pub(super) fn check(
    cx: &LateContext<'_>,
    expr: &Expr<'_>,
    recv: &Expr<'_>,
    map_call_span: Span,
    map_arg: &Expr<'_>,
    any_call_span: Span,
    any_arg: &Expr<'_>,
    method: &str,
) {
    if cx.ty_based_def(expr).opt_parent(cx).is_diag_item(cx, sym::Iterator)
        && cx.ty_based_def(recv).opt_parent(cx).is_diag_item(cx, sym::Iterator)
        && is_expr_identity_function(cx, any_arg)
        && let map_any_call_span = map_call_span.with_hi(any_call_span.hi())
        && let Some(map_arg) = map_arg.span.get_source_text(cx)
    {
        span_lint_and_then(
            cx,
            MAP_ALL_ANY_IDENTITY,
            map_any_call_span,
            format!("usage of `.map(...).{method}(identity)`"),
            |diag| {
                diag.span_suggestion_verbose(
                    map_any_call_span,
                    format!("use `.{method}(...)` instead"),
                    format!("{method}({map_arg})"),
                    Applicability::MachineApplicable,
                );
            },
        );
    }
}
