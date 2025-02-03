use super::ALLOW_ATTRIBUTES;
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::is_from_proc_macro;
use rustc_ast::{AttrStyle, Attribute};
use rustc_errors::Applicability;
use rustc_lint::{EarlyContext, LintContext};

// Separate each crate's features.
pub fn check<'cx>(cx: &EarlyContext<'cx>, attr: &'cx Attribute) {
    if !attr.span.in_external_macro(cx.sess().source_map())
        && let AttrStyle::Outer = attr.style
        && let Some(ident) = attr.ident()
        && !is_from_proc_macro(cx, attr)
    {
        #[expect(clippy::collapsible_span_lint_calls, reason = "rust-clippy#7797")]
        span_lint_and_then(cx, ALLOW_ATTRIBUTES, ident.span, "#[allow] attribute found", |diag| {
            diag.span_suggestion(
                ident.span,
                "replace it with",
                "expect",
                Applicability::MachineApplicable,
            );
        });
    }
}
