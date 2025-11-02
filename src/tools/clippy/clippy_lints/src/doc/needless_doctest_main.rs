use super::Fragments;
use crate::doc::NEEDLESS_DOCTEST_MAIN;
use clippy_utils::diagnostics::span_lint;
use clippy_utils::tokenize_with_text;
use rustc_lexer::TokenKind;
use rustc_lint::LateContext;
use rustc_span::InnerSpan;

fn returns_unit<'a>(mut tokens: impl Iterator<Item = (TokenKind, &'a str, InnerSpan)>) -> bool {
    let mut next = || tokens.next().map_or(TokenKind::Whitespace, |(kind, ..)| kind);

    match next() {
        // {
        TokenKind::OpenBrace => true,
        // - > ( ) {
        TokenKind::Minus => {
            next() == TokenKind::Gt
                && next() == TokenKind::OpenParen
                && next() == TokenKind::CloseParen
                && next() == TokenKind::OpenBrace
        },
        _ => false,
    }
}

pub fn check(cx: &LateContext<'_>, text: &str, offset: usize, fragments: Fragments<'_>) {
    if !text.contains("main") {
        return;
    }

    let mut tokens = tokenize_with_text(text).filter(|&(kind, ..)| {
        !matches!(
            kind,
            TokenKind::Whitespace | TokenKind::BlockComment { .. } | TokenKind::LineComment { .. }
        )
    });
    if let Some((TokenKind::Ident, "fn", fn_span)) = tokens.next()
        && let Some((TokenKind::Ident, "main", main_span)) = tokens.next()
        && let Some((TokenKind::OpenParen, ..)) = tokens.next()
        && let Some((TokenKind::CloseParen, ..)) = tokens.next()
        && returns_unit(&mut tokens)
    {
        let mut depth = 1;
        for (kind, ..) in &mut tokens {
            match kind {
                TokenKind::OpenBrace => depth += 1,
                TokenKind::CloseBrace => {
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                },
                _ => {},
            }
        }

        if tokens.next().is_none()
            && let Some(span) = fragments.span(cx, fn_span.start + offset..main_span.end + offset)
        {
            span_lint(cx, NEEDLESS_DOCTEST_MAIN, span, "needless `fn main` in doctest");
        }
    }
}
