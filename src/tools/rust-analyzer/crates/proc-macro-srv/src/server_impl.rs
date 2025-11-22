//! proc-macro server implementation
//!
//! Based on idea from <https://github.com/fedochet/rust-proc-macro-expander>
//! The lib-proc-macro server backend is `TokenStream`-agnostic, such that
//! we could provide any TokenStream implementation.
//! The original idea from fedochet is using proc-macro2 as backend,
//! we use tt instead for better integration with RA.

pub(crate) mod rust_analyzer_span;
pub(crate) mod token_id;

pub fn literal_from_str<Span: Copy>(
    s: &str,
    span: Span,
) -> Result<crate::bridge::Literal<Span>, ()> {
    use rustc_lexer::{LiteralKind, Token, TokenKind};
    let mut tokens = rustc_lexer::tokenize(s, rustc_lexer::FrontmatterAllowed::No);
    let minus_or_lit = tokens.next().unwrap_or(Token { kind: TokenKind::Eof, len: 0 });

    let lit = if minus_or_lit.kind == TokenKind::Minus {
        let lit = tokens.next().ok_or(())?;
        if !matches!(
            lit.kind,
            TokenKind::Literal { kind: LiteralKind::Int { .. } | LiteralKind::Float { .. }, .. }
        ) {
            return Err(());
        }
        lit
    } else {
        minus_or_lit
    };

    if tokens.next().is_some() {
        return Err(());
    }

    let TokenKind::Literal { kind, suffix_start } = lit.kind else { return Err(()) };
    Ok(crate::token_stream::literal_from_lexer(s, span, kind, suffix_start))
}
