use clippy_utils::diagnostics::span_lint_and_sugg;
use rustc_ast::ast::{Pat, PatKind};
use rustc_errors::Applicability;
use rustc_lint::EarlyContext;
use rustc_span::Span;

use super::UNNEEDED_WILDCARD_PATTERN;

pub(super) fn check(cx: &EarlyContext<'_>, pat: &Pat) {
    if let PatKind::TupleStruct(_, _, ref patterns) | PatKind::Tuple(ref patterns) = pat.kind
        && let Some(rest_index) = patterns.iter().position(|pat| pat.is_rest())
    {
        if let Some((left_index, left_pat)) = patterns[..rest_index]
            .iter()
            .rev()
            .take_while(|pat| matches!(pat.kind, PatKind::Wild))
            .enumerate()
            .last()
        {
            span_lint(cx, left_pat.span.until(patterns[rest_index].span), left_index == 0);
        }

        if let Some((right_index, right_pat)) = patterns[rest_index + 1..]
            .iter()
            .take_while(|pat| matches!(pat.kind, PatKind::Wild))
            .enumerate()
            .last()
        {
            span_lint(
                cx,
                patterns[rest_index].span.shrink_to_hi().to(right_pat.span),
                right_index == 0,
            );
        }
    }
}

fn span_lint(cx: &EarlyContext<'_>, span: Span, only_one: bool) {
    span_lint_and_sugg(
        cx,
        UNNEEDED_WILDCARD_PATTERN,
        span,
        if only_one {
            "this pattern is unneeded as the `..` pattern can match that element"
        } else {
            "these patterns are unneeded as the `..` pattern can match those elements"
        },
        if only_one { "remove it" } else { "remove them" },
        String::new(),
        Applicability::MachineApplicable,
    );
}
