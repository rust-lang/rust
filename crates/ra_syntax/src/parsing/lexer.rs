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

/// Break a string up into its component tokens
pub fn tokenize(text: &str) -> Vec<Token> {
    if text.is_empty() {
        return vec![];
    }
    let mut text = text;
    let mut acc = Vec::new();
    if let Some(len) = ra_rustc_lexer::strip_shebang(text) {
        acc.push(Token { kind: SHEBANG, len: TextUnit::from_usize(len) });
        text = &text[len..];
    }
    while !text.is_empty() {
        let rustc_token = ra_rustc_lexer::first_token(text);
        macro_rules! decompose {
            ($t1:expr, $t2:expr) => {{
                acc.push(Token { kind: $t1, len: 1.into() });
                acc.push(Token { kind: $t2, len: 1.into() });
                text = &text[2..];
                continue;
            }};
            ($t1:expr, $t2:expr, $t3:expr) => {{
                acc.push(Token { kind: $t1, len: 1.into() });
                acc.push(Token { kind: $t2, len: 1.into() });
                acc.push(Token { kind: $t3, len: 1.into() });
                text = &text[3..];
                continue;
            }};
        }
        let kind = match rustc_token.kind {
            ra_rustc_lexer::TokenKind::LineComment => COMMENT,
            ra_rustc_lexer::TokenKind::BlockComment { .. } => COMMENT,
            ra_rustc_lexer::TokenKind::Whitespace => WHITESPACE,
            ra_rustc_lexer::TokenKind::Ident => {
                let token_text = &text[..rustc_token.len];
                if token_text == "_" {
                    UNDERSCORE
                } else {
                    SyntaxKind::from_keyword(&text[..rustc_token.len]).unwrap_or(IDENT)
                }
            }
            ra_rustc_lexer::TokenKind::RawIdent => IDENT,
            ra_rustc_lexer::TokenKind::Literal { kind, .. } => match kind {
                ra_rustc_lexer::LiteralKind::Int { .. } => INT_NUMBER,
                ra_rustc_lexer::LiteralKind::Float { .. } => FLOAT_NUMBER,
                ra_rustc_lexer::LiteralKind::Char { .. } => CHAR,
                ra_rustc_lexer::LiteralKind::Byte { .. } => BYTE,
                ra_rustc_lexer::LiteralKind::Str { .. } => STRING,
                ra_rustc_lexer::LiteralKind::ByteStr { .. } => BYTE_STRING,
                ra_rustc_lexer::LiteralKind::RawStr { .. } => RAW_STRING,
                ra_rustc_lexer::LiteralKind::RawByteStr { .. } => RAW_BYTE_STRING,
            },
            ra_rustc_lexer::TokenKind::Lifetime { .. } => LIFETIME,
            ra_rustc_lexer::TokenKind::Semi => SEMI,
            ra_rustc_lexer::TokenKind::Comma => COMMA,
            ra_rustc_lexer::TokenKind::DotDotDot => decompose!(DOT, DOT, DOT),
            ra_rustc_lexer::TokenKind::DotDotEq => decompose!(DOT, DOT, EQ),
            ra_rustc_lexer::TokenKind::DotDot => decompose!(DOT, DOT),
            ra_rustc_lexer::TokenKind::Dot => DOT,
            ra_rustc_lexer::TokenKind::OpenParen => L_PAREN,
            ra_rustc_lexer::TokenKind::CloseParen => R_PAREN,
            ra_rustc_lexer::TokenKind::OpenBrace => L_CURLY,
            ra_rustc_lexer::TokenKind::CloseBrace => R_CURLY,
            ra_rustc_lexer::TokenKind::OpenBracket => L_BRACK,
            ra_rustc_lexer::TokenKind::CloseBracket => R_BRACK,
            ra_rustc_lexer::TokenKind::At => AT,
            ra_rustc_lexer::TokenKind::Pound => POUND,
            ra_rustc_lexer::TokenKind::Tilde => TILDE,
            ra_rustc_lexer::TokenKind::Question => QUESTION,
            ra_rustc_lexer::TokenKind::ColonColon => decompose!(COLON, COLON),
            ra_rustc_lexer::TokenKind::Colon => COLON,
            ra_rustc_lexer::TokenKind::Dollar => DOLLAR,
            ra_rustc_lexer::TokenKind::EqEq => decompose!(EQ, EQ),
            ra_rustc_lexer::TokenKind::Eq => EQ,
            ra_rustc_lexer::TokenKind::FatArrow => decompose!(EQ, R_ANGLE),
            ra_rustc_lexer::TokenKind::Ne => decompose!(EXCL, EQ),
            ra_rustc_lexer::TokenKind::Not => EXCL,
            ra_rustc_lexer::TokenKind::Le => decompose!(L_ANGLE, EQ),
            ra_rustc_lexer::TokenKind::LArrow => decompose!(COLON, MINUS),
            ra_rustc_lexer::TokenKind::Lt => L_ANGLE,
            ra_rustc_lexer::TokenKind::ShlEq => decompose!(L_ANGLE, L_ANGLE, EQ),
            ra_rustc_lexer::TokenKind::Shl => decompose!(L_ANGLE, L_ANGLE),
            ra_rustc_lexer::TokenKind::Ge => decompose!(R_ANGLE, EQ),
            ra_rustc_lexer::TokenKind::Gt => R_ANGLE,
            ra_rustc_lexer::TokenKind::ShrEq => decompose!(R_ANGLE, R_ANGLE, EQ),
            ra_rustc_lexer::TokenKind::Shr => decompose!(R_ANGLE, R_ANGLE),
            ra_rustc_lexer::TokenKind::RArrow => decompose!(MINUS, R_ANGLE),
            ra_rustc_lexer::TokenKind::Minus => MINUS,
            ra_rustc_lexer::TokenKind::MinusEq => decompose!(MINUS, EQ),
            ra_rustc_lexer::TokenKind::And => AMP,
            ra_rustc_lexer::TokenKind::AndAnd => decompose!(AMP, AMP),
            ra_rustc_lexer::TokenKind::AndEq => decompose!(AMP, EQ),
            ra_rustc_lexer::TokenKind::Or => PIPE,
            ra_rustc_lexer::TokenKind::OrOr => decompose!(PIPE, PIPE),
            ra_rustc_lexer::TokenKind::OrEq => decompose!(PIPE, EQ),
            ra_rustc_lexer::TokenKind::PlusEq => decompose!(PLUS, EQ),
            ra_rustc_lexer::TokenKind::Plus => PLUS,
            ra_rustc_lexer::TokenKind::StarEq => decompose!(STAR, EQ),
            ra_rustc_lexer::TokenKind::Star => STAR,
            ra_rustc_lexer::TokenKind::SlashEq => decompose!(SLASH, EQ),
            ra_rustc_lexer::TokenKind::Slash => SLASH,
            ra_rustc_lexer::TokenKind::CaretEq => decompose!(CARET, EQ),
            ra_rustc_lexer::TokenKind::Caret => CARET,
            ra_rustc_lexer::TokenKind::PercentEq => decompose!(PERCENT, EQ),
            ra_rustc_lexer::TokenKind::Percent => PERCENT,
            ra_rustc_lexer::TokenKind::Unknown => ERROR,
        };
        let token = Token { kind, len: TextUnit::from_usize(rustc_token.len) };
        acc.push(token);
        text = &text[rustc_token.len..];
    }
    acc
}

pub fn classify_literal(text: &str) -> Option<Token> {
    let t = ra_rustc_lexer::first_token(text);
    if t.len != text.len() {
        return None;
    }
    let kind = match t.kind {
        ra_rustc_lexer::TokenKind::Literal { kind, .. } => match kind {
            ra_rustc_lexer::LiteralKind::Int { .. } => INT_NUMBER,
            ra_rustc_lexer::LiteralKind::Float { .. } => FLOAT_NUMBER,
            ra_rustc_lexer::LiteralKind::Char { .. } => CHAR,
            ra_rustc_lexer::LiteralKind::Byte { .. } => BYTE,
            ra_rustc_lexer::LiteralKind::Str { .. } => STRING,
            ra_rustc_lexer::LiteralKind::ByteStr { .. } => BYTE_STRING,
            ra_rustc_lexer::LiteralKind::RawStr { .. } => RAW_STRING,
            ra_rustc_lexer::LiteralKind::RawByteStr { .. } => RAW_BYTE_STRING,
        },
        _ => return None,
    };
    Some(Token { kind, len: TextUnit::from_usize(t.len) })
}
