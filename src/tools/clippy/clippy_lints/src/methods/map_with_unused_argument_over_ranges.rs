use crate::methods::MAP_WITH_UNUSED_ARGUMENT_OVER_RANGES;
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::sugg::Sugg;
use clippy_utils::{eager_or_lazy, higher, std_or_core, usage};
use rustc_ast::LitKind;
use rustc_ast::ast::RangeLimits;
use rustc_data_structures::packed::Pu128;
use rustc_errors::Applicability;
use rustc_hir::{Body, Closure, Expr, ExprKind};
use rustc_lint::LateContext;
use rustc_span::Span;

fn extract_count_with_applicability(
    cx: &LateContext<'_>,
    range: higher::Range<'_>,
    applicability: &mut Applicability,
) -> Option<String> {
    let start = range.start?;
    let end = range.end?;
    // TODO: This doesn't handle if either the start or end are negative literals, or if the start is
    // not a literal. In the first case, we need to be careful about how we handle computing the
    // count to avoid overflows. In the second, we may need to add parenthesis to make the
    // suggestion correct.
    if let ExprKind::Lit(lit) = start.kind
        && let LitKind::Int(Pu128(lower_bound), _) = lit.node
    {
        if let ExprKind::Lit(lit) = end.kind
            && let LitKind::Int(Pu128(upper_bound), _) = lit.node
        {
            // Here we can explicitly calculate the number of iterations
            let count = if upper_bound >= lower_bound {
                match range.limits {
                    RangeLimits::HalfOpen => upper_bound - lower_bound,
                    RangeLimits::Closed => (upper_bound - lower_bound).checked_add(1)?,
                }
            } else {
                0
            };
            return Some(format!("{count}"));
        }
        let end_snippet = Sugg::hir_with_applicability(cx, end, "...", applicability)
            .maybe_paren()
            .into_string();
        if lower_bound == 0 {
            if range.limits == RangeLimits::Closed {
                return Some(format!("{end_snippet} + 1"));
            }
            return Some(end_snippet);
        }
        if range.limits == RangeLimits::Closed {
            return Some(format!("{end_snippet} - {}", lower_bound - 1));
        }
        return Some(format!("{end_snippet} - {lower_bound}"));
    }
    None
}

pub(super) fn check(
    cx: &LateContext<'_>,
    ex: &Expr<'_>,
    receiver: &Expr<'_>,
    arg: &Expr<'_>,
    msrv: Msrv,
    method_call_span: Span,
) {
    let mut applicability = Applicability::MaybeIncorrect;
    if let Some(range) = higher::Range::hir(receiver)
        && let ExprKind::Closure(Closure { body, .. }) = arg.kind
        && let body_hir = cx.tcx.hir_body(*body)
        && let Body {
            params: [param],
            value: body_expr,
        } = body_hir
        && !usage::BindingUsageFinder::are_params_used(cx, body_hir)
        && let Some(count) = extract_count_with_applicability(cx, range, &mut applicability)
        && let Some(exec_context) = std_or_core(cx)
    {
        let method_to_use_name;
        let new_span;
        let use_take;

        if eager_or_lazy::switch_to_eager_eval(cx, body_expr) {
            if msrv.meets(cx, msrvs::REPEAT_N) {
                method_to_use_name = "repeat_n";
                let body_snippet = snippet_with_applicability(cx, body_expr.span, "..", &mut applicability);
                new_span = (arg.span, format!("{body_snippet}, {count}"));
                use_take = false;
            } else {
                method_to_use_name = "repeat";
                let body_snippet = snippet_with_applicability(cx, body_expr.span, "..", &mut applicability);
                new_span = (arg.span, body_snippet.to_string());
                use_take = true;
            }
        } else if msrv.meets(cx, msrvs::REPEAT_WITH) {
            method_to_use_name = "repeat_with";
            new_span = (param.span, String::new());
            use_take = true;
        } else {
            return;
        }

        // We need to provide nonempty parts to diag.multipart_suggestion so we
        // collate all our parts here and then remove those that are empty.
        let mut parts = vec![
            (
                receiver.span.to(method_call_span),
                format!("{exec_context}::iter::{method_to_use_name}"),
            ),
            new_span,
        ];
        if use_take {
            parts.push((ex.span.shrink_to_hi(), format!(".take({count})")));
        }

        span_lint_and_then(
            cx,
            MAP_WITH_UNUSED_ARGUMENT_OVER_RANGES,
            ex.span,
            "map of a closure that does not depend on its parameter over a range",
            |diag| {
                diag.multipart_suggestion(
                    if use_take {
                        format!("remove the explicit range and use `{method_to_use_name}` and `take`")
                    } else {
                        format!("remove the explicit range and use `{method_to_use_name}`")
                    },
                    parts,
                    applicability,
                );
            },
        );
    }
}
