use super::Fragments;
use crate::doc::TEST_ATTR_IN_DOCTEST;
use clippy_utils::diagnostics::span_lint;
use clippy_utils::tokenize_with_text;
use rustc_lexer::TokenKind;
use rustc_lint::LateContext;

pub fn check(cx: &LateContext<'_>, text: &str, offset: usize, fragments: Fragments<'_>) {
    if !text.contains("#[test]") {
        return;
    }

    let mut spans = Vec::new();
    let mut tokens = tokenize_with_text(text).filter(|&(kind, ..)| kind != TokenKind::Whitespace);
    while let Some(token) = tokens.next() {
        if let (TokenKind::Pound, _, pound_span) = token
            && let Some((TokenKind::OpenBracket, ..)) = tokens.next()
            && let Some((TokenKind::Ident, "test", _)) = tokens.next()
            && let Some((TokenKind::CloseBracket, _, close_span)) = tokens.next()
            && let Some(span) = fragments.span(cx, pound_span.start + offset..close_span.end + offset)
        {
            spans.push(span);
        }
    }

    if !spans.is_empty() {
        span_lint(
            cx,
            TEST_ATTR_IN_DOCTEST,
            spans,
            "unit tests in doctest are not executed",
        );
    }
}
