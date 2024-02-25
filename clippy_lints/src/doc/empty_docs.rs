use clippy_utils::diagnostics::span_lint_and_help;
use rustc_ast::Attribute;
use rustc_lint::LateContext;

use super::EMPTY_DOCS;

// TODO: Adjust the parameters as necessary
pub(super) fn check(cx: &LateContext<'_>, attrs: &[Attribute]) {
    let doc_attrs: Vec<_> = attrs.iter().filter(|attr| attr.doc_str().is_some()).collect();

    let span;
    if let Some(first) = doc_attrs.first()
        && let Some(last) = doc_attrs.last()
    {
        span = first.span.with_hi(last.span.hi());
        span_lint_and_help(
            cx,
            EMPTY_DOCS,
            span,
            "empty doc comment",
            None,
            "consider removing or filling it",
        );
    }
}
