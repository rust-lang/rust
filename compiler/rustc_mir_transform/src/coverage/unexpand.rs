use rustc_span::Span;

/// Walks through the expansion ancestors of `original_span` to find a span that
/// is contained in `body_span` and has the same [syntax context] as `body_span`.
pub(crate) fn unexpand_into_body_span(original_span: Span, body_span: Span) -> Option<Span> {
    // Because we don't need to return any extra ancestor information,
    // we can just delegate directly to `find_ancestor_inside_same_ctxt`.
    original_span.find_ancestor_inside_same_ctxt(body_span)
}
