use super::ALLOW_ATTRIBUTES;
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::is_from_proc_macro;
use rustc_ast::{AttrStyle, Attribute};
use rustc_errors::Applicability;
use rustc_lint::{LateContext, LintContext};
use rustc_middle::lint::in_external_macro;

// Separate each crate's features.
pub fn check<'cx>(cx: &LateContext<'cx>, attr: &'cx Attribute) {
    if !in_external_macro(cx.sess(), attr.span)
        && let AttrStyle::Outer = attr.style
        && let Some(ident) = attr.ident()
        && !is_from_proc_macro(cx, attr)
    {
        span_lint_and_sugg(
            cx,
            ALLOW_ATTRIBUTES,
            ident.span,
            "#[allow] attribute found",
            "replace it with",
            "expect".into(),
            Applicability::MachineApplicable,
        );
    }
}
