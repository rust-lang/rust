use super::INLINE_ALWAYS;
use clippy_utils::diagnostics::span_lint;
use rustc_attr_data_structures::{find_attr, AttributeKind, InlineAttr};
use rustc_hir::Attribute;
use rustc_lint::LateContext;
use rustc_span::symbol::Symbol;
use rustc_span::Span;

pub(super) fn check(cx: &LateContext<'_>, span: Span, name: Symbol, attrs: &[Attribute]) {
    if span.from_expansion() {
        return;
    }

    if let Some(span) = find_attr!(attrs, AttributeKind::Inline(InlineAttr::Always, span) => *span) {
        span_lint(
            cx,
            INLINE_ALWAYS,
            span,
            format!("you have declared `#[inline(always)]` on `{name}`. This is usually a bad idea"),
        );
    }
}
