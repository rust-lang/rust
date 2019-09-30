//! FIXME: write short doc here

use crate::{
    SyntaxKind::{self, *},
    TextUnit,
};

/// A token of Rust source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Token {
    /// The kind of token.
    pub kind: SyntaxKind,
    /// The length of the token.
    pub len: TextUnit,
}

fn match_literal_kind(kind: rustc_lexer::LiteralKind) -> SyntaxKind {
    match kind {
        rustc_lexer::LiteralKind::Int { .. } => INT_NUMBER,
        rustc_lexer::LiteralKind::Float { .. } => FLOAT_NUMBER,
        rustc_lexer::LiteralKind::Char { .. } => CHAR,
        rustc_lexer::LiteralKind::Byte { .. } => BYTE,
        rustc_lexer::LiteralKind::Str { .. } => STRING,
        rustc_lexer::LiteralKind::ByteStr { .. } => BYTE_STRING,
        rustc_lexer::LiteralKind::RawStr { .. } => RAW_STRING,
        rustc_lexer::LiteralKind::RawByteStr { .. } => RAW_BYTE_STRING,
    }
}

/// Break a string up into its component tokens
pub fn tokenize(text: &str) -> Vec<Token> {
    if text.is_empty() {
        return vec![];
    }
    let mut text = text;
    let mut acc = Vec::new();
    if let Some(len) = rustc_lexer::strip_shebang(text) {
        acc.push(Token { kind: SHEBANG, len: TextUnit::from_usize(len) });
        text = &text[len..];
    }
    while !text.is_empty() {
        let rustc_token = rustc_lexer::first_token(text);
        let kind = match rustc_token.kind {
            rustc_lexer::TokenKind::LineComment => COMMENT,
            rustc_lexer::TokenKind::BlockComment { .. } => COMMENT,
            rustc_lexer::TokenKind::Whitespace => WHITESPACE,
            rustc_lexer::TokenKind::Ident => {
                let token_text = &text[..rustc_token.len];
                if token_text == "_" {
                    UNDERSCORE
                } else {
                    SyntaxKind::from_keyword(&text[..rustc_token.len]).unwrap_or(IDENT)
                }
            }
            rustc_lexer::TokenKind::RawIdent => IDENT,
            rustc_lexer::TokenKind::Literal { kind, .. } => match_literal_kind(kind),
            rustc_lexer::TokenKind::Lifetime { .. } => LIFETIME,
            rustc_lexer::TokenKind::Semi => SEMI,
            rustc_lexer::TokenKind::Comma => COMMA,
            rustc_lexer::TokenKind::Dot => DOT,
            rustc_lexer::TokenKind::OpenParen => L_PAREN,
            rustc_lexer::TokenKind::CloseParen => R_PAREN,
            rustc_lexer::TokenKind::OpenBrace => L_CURLY,
            rustc_lexer::TokenKind::CloseBrace => R_CURLY,
            rustc_lexer::TokenKind::OpenBracket => L_BRACK,
            rustc_lexer::TokenKind::CloseBracket => R_BRACK,
            rustc_lexer::TokenKind::At => AT,
            rustc_lexer::TokenKind::Pound => POUND,
            rustc_lexer::TokenKind::Tilde => TILDE,
            rustc_lexer::TokenKind::Question => QUESTION,
            rustc_lexer::TokenKind::Colon => COLON,
            rustc_lexer::TokenKind::Dollar => DOLLAR,
            rustc_lexer::TokenKind::Eq => EQ,
            rustc_lexer::TokenKind::Not => EXCL,
            rustc_lexer::TokenKind::Lt => L_ANGLE,
            rustc_lexer::TokenKind::Gt => R_ANGLE,
            rustc_lexer::TokenKind::Minus => MINUS,
            rustc_lexer::TokenKind::And => AMP,
            rustc_lexer::TokenKind::Or => PIPE,
            rustc_lexer::TokenKind::Plus => PLUS,
            rustc_lexer::TokenKind::Star => STAR,
            rustc_lexer::TokenKind::Slash => SLASH,
            rustc_lexer::TokenKind::Caret => CARET,
            rustc_lexer::TokenKind::Percent => PERCENT,
            rustc_lexer::TokenKind::Unknown => ERROR,
        };
        let token = Token { kind, len: TextUnit::from_usize(rustc_token.len) };
        acc.push(token);
        text = &text[rustc_token.len..];
    }
    acc
}

pub fn classify_literal(text: &str) -> Option<Token> {
    let t = rustc_lexer::first_token(text);
    if t.len != text.len() {
        return None;
    }
    let kind = match t.kind {
        rustc_lexer::TokenKind::Literal { kind, .. } => match_literal_kind(kind),
        _ => return None,
    };
    Some(Token { kind, len: TextUnit::from_usize(t.len) })
}
