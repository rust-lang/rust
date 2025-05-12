use clippy_utils::diagnostics::{span_lint_and_sugg, span_lint_and_then};
use clippy_utils::source::SpanRangeExt;
use clippy_utils::ty::is_type_diagnostic_item;

use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind};
use rustc_lint::LateContext;
use rustc_middle::ty::Ty;
use rustc_span::{Span, sym};

use super::UNNECESSARY_GET_THEN_CHECK;

fn is_a_std_set_type(cx: &LateContext<'_>, ty: Ty<'_>) -> bool {
    is_type_diagnostic_item(cx, ty, sym::HashSet) || is_type_diagnostic_item(cx, ty, sym::BTreeSet)
}

fn is_a_std_map_type(cx: &LateContext<'_>, ty: Ty<'_>) -> bool {
    is_type_diagnostic_item(cx, ty, sym::HashMap) || is_type_diagnostic_item(cx, ty, sym::BTreeMap)
}

pub(super) fn check(
    cx: &LateContext<'_>,
    call_span: Span,
    get_call: &Expr<'_>,
    get_caller: &Expr<'_>,
    arg: &Expr<'_>,
    is_some: bool,
) {
    let caller_ty = cx.typeck_results().expr_ty(get_caller);

    let is_set = is_a_std_set_type(cx, caller_ty);
    let is_map = is_a_std_map_type(cx, caller_ty);

    if !is_set && !is_map {
        return;
    }
    let ExprKind::MethodCall(path, _, _, get_call_span) = get_call.kind else {
        return;
    };
    let both_calls_span = get_call_span.with_hi(call_span.hi());
    if let Some(snippet) = both_calls_span.get_source_text(cx)
        && let Some(arg_snippet) = arg.span.get_source_text(cx)
    {
        let generics_snippet = if let Some(generics) = path.args
            && let Some(generics_snippet) = generics.span_ext.get_source_text(cx)
        {
            format!("::{generics_snippet}")
        } else {
            String::new()
        };
        let suggestion = if is_set {
            format!("contains{generics_snippet}({arg_snippet})")
        } else {
            format!("contains_key{generics_snippet}({arg_snippet})")
        };
        if is_some {
            span_lint_and_sugg(
                cx,
                UNNECESSARY_GET_THEN_CHECK,
                both_calls_span,
                format!("unnecessary use of `{snippet}`"),
                "replace it with",
                suggestion,
                Applicability::MaybeIncorrect,
            );
        } else if let Some(caller_snippet) = get_caller.span.get_source_text(cx) {
            let full_span = get_caller.span.with_hi(call_span.hi());

            span_lint_and_then(
                cx,
                UNNECESSARY_GET_THEN_CHECK,
                both_calls_span,
                format!("unnecessary use of `{snippet}`"),
                |diag| {
                    diag.span_suggestion(
                        full_span,
                        "replace it with",
                        format!("{}{caller_snippet}.{suggestion}", if is_some { "" } else { "!" }),
                        Applicability::MaybeIncorrect,
                    );
                },
            );
        }
    }
}
