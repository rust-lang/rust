use super::MIXED_ATTRIBUTES_STYLE;
use clippy_utils::diagnostics::span_lint;
use rustc_ast::AttrStyle;
use rustc_lint::EarlyContext;

pub(super) fn check(cx: &EarlyContext<'_>, item: &rustc_ast::Item) {
    let mut has_outer = false;
    let mut has_inner = false;

    for attr in &item.attrs {
        if attr.span.from_expansion() {
            continue;
        }
        match attr.style {
            AttrStyle::Inner => has_inner = true,
            AttrStyle::Outer => has_outer = true,
        }
    }
    if !has_outer || !has_inner {
        return;
    }
    let mut attrs_iter = item.attrs.iter().filter(|attr| !attr.span.from_expansion());
    let span = attrs_iter.next().unwrap().span;
    span_lint(
        cx,
        MIXED_ATTRIBUTES_STYLE,
        span.with_hi(attrs_iter.last().unwrap().span.hi()),
        "item has both inner and outer attributes",
    );
}
