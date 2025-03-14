use clippy_utils::diagnostics::span_lint_and_sugg;
use rustc_ast::{Pat, PatKind};
use rustc_errors::Applicability;
use rustc_lint::{EarlyContext, LintContext};

use super::REDUNDANT_AT_REST_PATTERN;

pub(super) fn check(cx: &EarlyContext<'_>, pat: &Pat) {
    if !pat.span.in_external_macro(cx.sess().source_map())
        && let PatKind::Slice(slice) = &pat.kind
        && let [one] = &**slice
        && let PatKind::Ident(annotation, ident, Some(rest)) = &one.kind
        && let PatKind::Rest = rest.kind
    {
        span_lint_and_sugg(
            cx,
            REDUNDANT_AT_REST_PATTERN,
            pat.span,
            "using a rest pattern to bind an entire slice to a local",
            "this is better represented with just the binding",
            format!("{}{ident}", annotation.prefix_str()),
            Applicability::MachineApplicable,
        );
    }
}
