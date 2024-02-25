use super::EMPTY_DOCS;
use clippy_utils::diagnostics::span_lint_and_help;
use rustc_ast::Attribute;
use rustc_lint::LateContext;
use rustc_resolve::rustdoc::{attrs_to_doc_fragments, span_of_fragments};

// TODO: Adjust the parameters as necessary
pub(super) fn check(cx: &LateContext<'_>, attrs: &[Attribute]) {
    let (fragments, _) = attrs_to_doc_fragments(attrs.iter().map(|attr| (attr, None)), true);
    if let Some(span) = span_of_fragments(&fragments) {
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
