mod classes;
mod comments;
mod numbers;
mod ptr;
mod strings;

use crate::{
    SyntaxKind::{self, *},
    TextUnit, T,
};

use self::{
    classes::*,
    comments::{scan_comment, scan_shebang},
    numbers::scan_number,
    ptr::Ptr,
    strings::{
        is_string_literal_start, scan_byte_char_or_string, scan_char, scan_raw_string, scan_string,
    },
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

/// Get the next token from a string
fn next_token(text: &str) -> Token {
    assert!(!text.is_empty());
    let mut ptr = Ptr::new(text);
    let c = ptr.bump().unwrap();
    let kind = next_token_inner(c, &mut ptr);
    let len = ptr.into_len();
    Token { kind, len }
}

fn next_token_inner(c: char, ptr: &mut Ptr) -> SyntaxKind {
    if is_whitespace(c) {
        ptr.bump_while(is_whitespace);
        return WHITESPACE;
    }

    match c {
        '#' => {
            if scan_shebang(ptr) {
                return SHEBANG;
            }
        }
        '/' => {
            if let Some(kind) = scan_comment(ptr) {
                return kind;
            }
        }
        _ => (),
    }

    let ident_start = is_ident_start(c) && !is_string_literal_start(c, ptr.current(), ptr.nth(1));
    if ident_start {
        return scan_ident(c, ptr);
    }

    if is_dec_digit(c) {
        let kind = scan_number(c, ptr);
        scan_literal_suffix(ptr);
        return kind;
    }

    // One-byte tokens.
    if let Some(kind) = SyntaxKind::from_char(c) {
        return kind;
    }

    match c {
        // Possiblily multi-byte tokens,
        // but we only produce single byte token now
        // T![...], T![..], T![..=], T![.]
        '.' => return T![.],
        // T![::] T![:]
        ':' => return T![:],
        // T![==] FATARROW T![=]
        '=' => return T![=],
        // T![!=] T![!]
        '!' => return T![!],
        // T![->] T![-]
        '-' => return T![-],

        // If the character is an ident start not followed by another single
        // quote, then this is a lifetime name:
        '\'' => {
            return if ptr.at_p(is_ident_start) && !ptr.at_str("''") {
                ptr.bump();
                while ptr.at_p(is_ident_continue) {
                    ptr.bump();
                }
                // lifetimes shouldn't end with a single quote
                // if we find one, then this is an invalid character literal
                if ptr.at('\'') {
                    ptr.bump();
                    return CHAR;
                }
                LIFETIME
            } else {
                scan_char(ptr);
                scan_literal_suffix(ptr);
                CHAR
            };
        }
        'b' => {
            let kind = scan_byte_char_or_string(ptr);
            scan_literal_suffix(ptr);
            return kind;
        }
        '"' => {
            scan_string(ptr);
            scan_literal_suffix(ptr);
            return STRING;
        }
        'r' => {
            scan_raw_string(ptr);
            scan_literal_suffix(ptr);
            return RAW_STRING;
        }
        _ => (),
    }
    ERROR
}

fn scan_ident(c: char, ptr: &mut Ptr) -> SyntaxKind {
    let is_raw = match (c, ptr.current()) {
        ('r', Some('#')) => {
            ptr.bump();
            true
        }
        ('_', None) => return T![_],
        ('_', Some(c)) if !is_ident_continue(c) => return T![_],
        _ => false,
    };
    ptr.bump_while(is_ident_continue);
    if !is_raw {
        if let Some(kind) = SyntaxKind::from_keyword(ptr.current_token_text()) {
            return kind;
        }
    }
    IDENT
}

fn scan_literal_suffix(ptr: &mut Ptr) {
    if ptr.at_p(is_ident_start) {
        ptr.bump();
    }
    ptr.bump_while(is_ident_continue);
}

pub fn classify_literal(text: &str) -> Option<Token> {
    let tkn = next_token(text);
    if !tkn.kind.is_literal() || tkn.len.to_usize() != text.len() {
        return None;
    }

    Some(tkn)
}
