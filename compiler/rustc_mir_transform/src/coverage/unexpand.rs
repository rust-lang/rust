use rustc_span::{ExpnKind, Span};

/// Walks through the expansion ancestors of `original_span` to find a span that
/// is contained in `body_span` and has the same [syntax context] as `body_span`.
pub(crate) fn unexpand_into_body_span(original_span: Span, body_span: Span) -> Option<Span> {
    // Because we don't need to return any extra ancestor information,
    // we can just delegate directly to `find_ancestor_inside_same_ctxt`.
    original_span.find_ancestor_inside_same_ctxt(body_span)
}

/// Walks through the expansion ancestors of `original_span` to find a span that
/// is contained in `body_span` and has the same [syntax context] as `body_span`.
///
/// If the returned span represents a bang-macro invocation (e.g. `foo!(..)`),
/// the returned symbol will be the name of that macro (e.g. `foo`).
pub(crate) fn unexpand_into_body_span_with_expn_kind(
    original_span: Span,
    body_span: Span,
) -> Option<(Span, Option<ExpnKind>)> {
    let (span, prev) = unexpand_into_body_span_with_prev(original_span, body_span)?;

    let expn_kind = prev.map(|prev| prev.ctxt().outer_expn_data().kind);

    Some((span, expn_kind))
}

/// Walks through the expansion ancestors of `original_span` to find a span that
/// is contained in `body_span` and has the same [syntax context] as `body_span`.
/// The ancestor that was traversed just before the matching span (if any) is
/// also returned.
///
/// For example, a return value of `Some((ancestor, Some(prev)))` means that:
/// - `ancestor == original_span.find_ancestor_inside_same_ctxt(body_span)`
/// - `prev.parent_callsite() == ancestor`
///
/// [syntax context]: rustc_span::SyntaxContext
fn unexpand_into_body_span_with_prev(
    original_span: Span,
    body_span: Span,
) -> Option<(Span, Option<Span>)> {
    let mut prev = None;
    let mut curr = original_span;

    while !body_span.contains(curr) || !curr.eq_ctxt(body_span) {
        prev = Some(curr);
        curr = curr.parent_callsite()?;
    }

    debug_assert_eq!(Some(curr), original_span.find_ancestor_inside_same_ctxt(body_span));
    if let Some(prev) = prev {
        debug_assert_eq!(Some(curr), prev.parent_callsite());
    }

    Some((curr, prev))
}
