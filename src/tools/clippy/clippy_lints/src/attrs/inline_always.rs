use super::INLINE_ALWAYS;
use super::utils::is_word;
use clippy_utils::diagnostics::span_lint;
use rustc_hir::Attribute;
use rustc_lint::LateContext;
use rustc_span::symbol::Symbol;
use rustc_span::{Span, sym};

pub(super) fn check(cx: &LateContext<'_>, span: Span, name: Symbol, attrs: &[Attribute]) {
    if span.from_expansion() {
        return;
    }

    for attr in attrs {
        if let Some(values) = attr.meta_item_list() {
            if values.len() != 1 || !attr.has_name(sym::inline) {
                continue;
            }
            if is_word(&values[0], sym::always) {
                span_lint(
                    cx,
                    INLINE_ALWAYS,
                    attr.span,
                    format!("you have declared `#[inline(always)]` on `{name}`. This is usually a bad idea"),
                );
            }
        }
    }
}
