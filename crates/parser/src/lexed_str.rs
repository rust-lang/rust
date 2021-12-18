//! Lexing `&str` into a sequence of Rust tokens.
//!
//! Note that strictly speaking the parser in this crate is not required to work
//! on tokens which originated from text. Macros, eg, can synthesize tokes out
//! of thin air. So, ideally, lexer should be an orthogonal crate. It is however
//! convenient to include a text-based lexer here!
//!
//! Note that these tokens, unlike the tokens we feed into the parser, do
//! include info about comments and whitespace.

use crate::{
    SyntaxKind::{self, *},
    T,
};

pub struct LexedStr<'a> {
    text: &'a str,
    kind: Vec<SyntaxKind>,
    start: Vec<u32>,
    error: Vec<LexError>,
}

struct LexError {
    msg: String,
    token: u32,
}

impl<'a> LexedStr<'a> {
    pub fn new(text: &'a str) -> LexedStr<'a> {
        let mut res = LexedStr { text, kind: Vec::new(), start: Vec::new(), error: Vec::new() };

        let mut offset = 0;
        if let Some(shebang_len) = rustc_lexer::strip_shebang(text) {
            res.push(SHEBANG, offset);
            offset = shebang_len
        };
        for token in rustc_lexer::tokenize(&text[offset..]) {
            let token_text = &text[offset..][..token.len];

            let (kind, err) = from_rustc(&token.kind, token_text);
            res.push(kind, offset);
            offset += token.len;

            if let Some(err) = err {
                let token = res.len() as u32;
                let msg = err.to_string();
                res.error.push(LexError { msg, token });
            }
        }
        res.push(EOF, offset);

        res
    }

    pub fn single_token(text: &'a str) -> Option<SyntaxKind> {
        if text.is_empty() {
            return None;
        }

        let token = rustc_lexer::first_token(text);
        if token.len != text.len() {
            return None;
        }

        let (kind, err) = from_rustc(&token.kind, text);
        if err.is_some() {
            return None;
        }

        Some(kind)
    }

    pub fn as_str(&self) -> &str {
        self.text
    }

    pub fn len(&self) -> usize {
        self.kind.len() - 1
    }

    pub fn kind(&self, i: usize) -> SyntaxKind {
        assert!(i < self.len());
        self.kind[i]
    }

    pub fn text(&self, i: usize) -> &str {
        assert!(i < self.len());
        let lo = self.start[i] as usize;
        let hi = self.start[i + 1] as usize;
        &self.text[lo..hi]
    }

    pub fn error(&self, i: usize) -> Option<&str> {
        assert!(i < self.len());
        let err = self.error.binary_search_by_key(&(i as u32), |i| i.token).ok()?;
        Some(self.error[err].msg.as_str())
    }

    pub fn to_tokens(&self) -> crate::Tokens {
        let mut res = crate::Tokens::default();
        let mut was_joint = false;
        for i in 0..self.len() {
            let kind = self.kind(i);
            if kind.is_trivia() {
                was_joint = false
            } else {
                if kind == SyntaxKind::IDENT {
                    let token_text = self.text(i);
                    let contextual_kw = SyntaxKind::from_contextual_keyword(token_text)
                        .unwrap_or(SyntaxKind::IDENT);
                    res.push_ident(contextual_kw);
                } else {
                    if was_joint {
                        res.was_joint();
                    }
                    res.push(kind);
                }
                was_joint = true;
            }
        }
        res
    }

    fn push(&mut self, kind: SyntaxKind, offset: usize) {
        self.kind.push(kind);
        self.start.push(offset as u32);
    }
}

/// Returns `SyntaxKind` and an optional tokenize error message.
fn from_rustc(
    kind: &rustc_lexer::TokenKind,
    token_text: &str,
) -> (SyntaxKind, Option<&'static str>) {
    // A note on an intended tradeoff:
    // We drop some useful information here (see patterns with double dots `..`)
    // Storing that info in `SyntaxKind` is not possible due to its layout requirements of
    // being `u16` that come from `rowan::SyntaxKind`.
    let mut err = "";

    let syntax_kind = {
        match kind {
            rustc_lexer::TokenKind::LineComment { doc_style: _ } => COMMENT,
            rustc_lexer::TokenKind::BlockComment { doc_style: _, terminated } => {
                if !terminated {
                    err = "Missing trailing `*/` symbols to terminate the block comment";
                }
                COMMENT
            }

            rustc_lexer::TokenKind::Whitespace => WHITESPACE,

            rustc_lexer::TokenKind::Ident if token_text == "_" => UNDERSCORE,
            rustc_lexer::TokenKind::Ident => SyntaxKind::from_keyword(token_text).unwrap_or(IDENT),

            rustc_lexer::TokenKind::RawIdent => IDENT,
            rustc_lexer::TokenKind::Literal { kind, .. } => return from_rustc_literal(kind),

            rustc_lexer::TokenKind::Lifetime { starts_with_number } => {
                if *starts_with_number {
                    err = "Lifetime name cannot start with a number";
                }
                LIFETIME_IDENT
            }

            rustc_lexer::TokenKind::Semi => T![;],
            rustc_lexer::TokenKind::Comma => T![,],
            rustc_lexer::TokenKind::Dot => T![.],
            rustc_lexer::TokenKind::OpenParen => T!['('],
            rustc_lexer::TokenKind::CloseParen => T![')'],
            rustc_lexer::TokenKind::OpenBrace => T!['{'],
            rustc_lexer::TokenKind::CloseBrace => T!['}'],
            rustc_lexer::TokenKind::OpenBracket => T!['['],
            rustc_lexer::TokenKind::CloseBracket => T![']'],
            rustc_lexer::TokenKind::At => T![@],
            rustc_lexer::TokenKind::Pound => T![#],
            rustc_lexer::TokenKind::Tilde => T![~],
            rustc_lexer::TokenKind::Question => T![?],
            rustc_lexer::TokenKind::Colon => T![:],
            rustc_lexer::TokenKind::Dollar => T![$],
            rustc_lexer::TokenKind::Eq => T![=],
            rustc_lexer::TokenKind::Bang => T![!],
            rustc_lexer::TokenKind::Lt => T![<],
            rustc_lexer::TokenKind::Gt => T![>],
            rustc_lexer::TokenKind::Minus => T![-],
            rustc_lexer::TokenKind::And => T![&],
            rustc_lexer::TokenKind::Or => T![|],
            rustc_lexer::TokenKind::Plus => T![+],
            rustc_lexer::TokenKind::Star => T![*],
            rustc_lexer::TokenKind::Slash => T![/],
            rustc_lexer::TokenKind::Caret => T![^],
            rustc_lexer::TokenKind::Percent => T![%],
            rustc_lexer::TokenKind::Unknown => ERROR,
        }
    };

    let err = if err.is_empty() { None } else { Some(err) };
    (syntax_kind, err)
}

fn from_rustc_literal(kind: &rustc_lexer::LiteralKind) -> (SyntaxKind, Option<&'static str>) {
    let mut err = "";

    let syntax_kind = match *kind {
        rustc_lexer::LiteralKind::Int { empty_int, base: _ } => {
            if empty_int {
                err = "Missing digits after the integer base prefix";
            }
            INT_NUMBER
        }
        rustc_lexer::LiteralKind::Float { empty_exponent, base: _ } => {
            if empty_exponent {
                err = "Missing digits after the exponent symbol";
            }
            FLOAT_NUMBER
        }
        rustc_lexer::LiteralKind::Char { terminated } => {
            if !terminated {
                err = "Missing trailing `'` symbol to terminate the character literal";
            }
            CHAR
        }
        rustc_lexer::LiteralKind::Byte { terminated } => {
            if !terminated {
                err = "Missing trailing `'` symbol to terminate the byte literal";
            }
            BYTE
        }
        rustc_lexer::LiteralKind::Str { terminated } => {
            if !terminated {
                err = "Missing trailing `\"` symbol to terminate the string literal";
            }
            STRING
        }
        rustc_lexer::LiteralKind::ByteStr { terminated } => {
            if !terminated {
                err = "Missing trailing `\"` symbol to terminate the byte string literal";
            }
            BYTE_STRING
        }
        rustc_lexer::LiteralKind::RawStr { err: raw_str_err, .. } => {
            if let Some(raw_str_err) = raw_str_err {
                err = match raw_str_err {
                    rustc_lexer::RawStrError::InvalidStarter { .. } => "Missing `\"` symbol after `#` symbols to begin the raw string literal",
                    rustc_lexer::RawStrError::NoTerminator { expected, found, .. } => if expected == found {
                        "Missing trailing `\"` to terminate the raw string literal"
                    } else {
                        "Missing trailing `\"` with `#` symbols to terminate the raw string literal"
                    },
                    rustc_lexer::RawStrError::TooManyDelimiters { .. } => "Too many `#` symbols: raw strings may be delimited by up to 65535 `#` symbols",
                };
            };
            STRING
        }
        rustc_lexer::LiteralKind::RawByteStr { err: raw_str_err, .. } => {
            if let Some(raw_str_err) = raw_str_err {
                err = match raw_str_err {
                    rustc_lexer::RawStrError::InvalidStarter { .. } => "Missing `\"` symbol after `#` symbols to begin the raw byte string literal",
                    rustc_lexer::RawStrError::NoTerminator { expected, found, .. } => if expected == found {
                        "Missing trailing `\"` to terminate the raw byte string literal"
                    } else {
                        "Missing trailing `\"` with `#` symbols to terminate the raw byte string literal"
                    },
                    rustc_lexer::RawStrError::TooManyDelimiters { .. } => "Too many `#` symbols: raw byte strings may be delimited by up to 65535 `#` symbols",
                };
            };

            BYTE_STRING
        }
    };

    let err = if err.is_empty() { None } else { Some(err) };
    (syntax_kind, err)
}
