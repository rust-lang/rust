//! Lexing `&str` into a sequence of Rust tokens.
//!
//! Note that strictly speaking the parser in this crate is not required to work
//! on tokens which originated from text. Macros, eg, can synthesize tokens out
//! of thin air. So, ideally, lexer should be an orthogonal crate. It is however
//! convenient to include a text-based lexer here!
//!
//! Note that these tokens, unlike the tokens we feed into the parser, do
//! include info about comments and whitespace.

use std::ops;

use rustc_literal_escaper::{EscapeError, Mode, unescape_byte, unescape_char, unescape_mixed, unescape_unicode};

use crate::{
    Edition,
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
    pub fn new(edition: Edition, text: &'a str) -> LexedStr<'a> {
        let _p = tracing::info_span!("LexedStr::new").entered();
        let mut conv = Converter::new(edition, text);
        if let Some(shebang_len) = rustc_lexer::strip_shebang(text) {
            conv.res.push(SHEBANG, conv.offset);
            conv.offset = shebang_len;
        };

        // Re-create the tokenizer from scratch every token because `GuardedStrPrefix` is one token in the lexer
        // but we want to split it to two in edition <2024.
        while let Some(token) = rustc_lexer::tokenize(&text[conv.offset..]).next() {
            let token_text = &text[conv.offset..][..token.len as usize];

            conv.extend_token(&token.kind, token_text);
        }

        conv.finalize_with_eof()
    }

    pub fn single_token(edition: Edition, text: &'a str) -> Option<(SyntaxKind, Option<String>)> {
        if text.is_empty() {
            return None;
        }

        let token = rustc_lexer::tokenize(text).next()?;
        if token.len as usize != text.len() {
            return None;
        }

        let mut conv = Converter::new(edition, text);
        conv.extend_token(&token.kind, text);
        match &*conv.res.kind {
            [kind] => Some((*kind, conv.res.error.pop().map(|it| it.msg))),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &str {
        self.text
    }

    pub fn len(&self) -> usize {
        self.kind.len() - 1
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn kind(&self, i: usize) -> SyntaxKind {
        assert!(i < self.len());
        self.kind[i]
    }

    pub fn text(&self, i: usize) -> &str {
        self.range_text(i..i + 1)
    }

    pub fn range_text(&self, r: ops::Range<usize>) -> &str {
        assert!(r.start < r.end && r.end <= self.len());
        let lo = self.start[r.start] as usize;
        let hi = self.start[r.end] as usize;
        &self.text[lo..hi]
    }

    // Naming is hard.
    pub fn text_range(&self, i: usize) -> ops::Range<usize> {
        assert!(i < self.len());
        let lo = self.start[i] as usize;
        let hi = self.start[i + 1] as usize;
        lo..hi
    }
    pub fn text_start(&self, i: usize) -> usize {
        assert!(i <= self.len());
        self.start[i] as usize
    }
    pub fn text_len(&self, i: usize) -> usize {
        assert!(i < self.len());
        let r = self.text_range(i);
        r.end - r.start
    }

    pub fn error(&self, i: usize) -> Option<&str> {
        assert!(i < self.len());
        let err = self.error.binary_search_by_key(&(i as u32), |i| i.token).ok()?;
        Some(self.error[err].msg.as_str())
    }

    pub fn errors(&self) -> impl Iterator<Item = (usize, &str)> + '_ {
        self.error.iter().map(|it| (it.token as usize, it.msg.as_str()))
    }

    fn push(&mut self, kind: SyntaxKind, offset: usize) {
        self.kind.push(kind);
        self.start.push(offset as u32);
    }
}

struct Converter<'a> {
    res: LexedStr<'a>,
    offset: usize,
    edition: Edition,
}

impl<'a> Converter<'a> {
    fn new(edition: Edition, text: &'a str) -> Self {
        Self {
            res: LexedStr { text, kind: Vec::new(), start: Vec::new(), error: Vec::new() },
            offset: 0,
            edition,
        }
    }

    fn finalize_with_eof(mut self) -> LexedStr<'a> {
        self.res.push(EOF, self.offset);
        self.res
    }

    fn push(&mut self, kind: SyntaxKind, len: usize, err: Option<&str>) {
        self.res.push(kind, self.offset);
        self.offset += len;

        if let Some(err) = err {
            let token = self.res.len() as u32;
            let msg = err.to_owned();
            self.res.error.push(LexError { msg, token });
        }
    }

    fn extend_token(&mut self, kind: &rustc_lexer::TokenKind, mut token_text: &str) {
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
                rustc_lexer::TokenKind::Ident => {
                    SyntaxKind::from_keyword(token_text, self.edition).unwrap_or(IDENT)
                }
                rustc_lexer::TokenKind::InvalidIdent => {
                    err = "Ident contains invalid characters";
                    IDENT
                }

                rustc_lexer::TokenKind::RawIdent => IDENT,

                rustc_lexer::TokenKind::GuardedStrPrefix if self.edition.at_least_2024() => {
                    // FIXME: rustc does something better for recovery.
                    err = "Invalid string literal (reserved syntax)";
                    ERROR
                }
                rustc_lexer::TokenKind::GuardedStrPrefix => {
                    // The token is `#"` or `##`, split it into two.
                    token_text = &token_text[1..];
                    POUND
                }

                rustc_lexer::TokenKind::Literal { kind, .. } => {
                    self.extend_literal(token_text.len(), kind);
                    return;
                }

                rustc_lexer::TokenKind::Lifetime { starts_with_number } => {
                    if *starts_with_number {
                        err = "Lifetime name cannot start with a number";
                    }
                    LIFETIME_IDENT
                }
                rustc_lexer::TokenKind::UnknownPrefixLifetime => {
                    err = "Unknown lifetime prefix";
                    LIFETIME_IDENT
                }
                rustc_lexer::TokenKind::RawLifetime => LIFETIME_IDENT,

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
                rustc_lexer::TokenKind::UnknownPrefix if token_text == "builtin" => IDENT,
                rustc_lexer::TokenKind::UnknownPrefix => {
                    err = "unknown literal prefix";
                    IDENT
                }
                rustc_lexer::TokenKind::Eof => EOF,
            }
        };

        let err = if err.is_empty() { None } else { Some(err) };
        self.push(syntax_kind, token_text.len(), err);
    }

    fn extend_literal(&mut self, len: usize, kind: &rustc_lexer::LiteralKind) {
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
                } else {
                    let text = &self.res.text[self.offset + 1..][..len - 1];
                    let i = text.rfind('\'').unwrap();
                    let text = &text[..i];
                    if let Err(e) = unescape_char(text) {
                        err = error_to_diagnostic_message(e, Mode::Char);
                    }
                }
                CHAR
            }
            rustc_lexer::LiteralKind::Byte { terminated } => {
                if !terminated {
                    err = "Missing trailing `'` symbol to terminate the byte literal";
                } else {
                    let text = &self.res.text[self.offset + 2..][..len - 2];
                    let i = text.rfind('\'').unwrap();
                    let text = &text[..i];
                    if let Err(e) = unescape_byte(text) {
                        err = error_to_diagnostic_message(e, Mode::Byte);
                    }
                }

                BYTE
            }
            rustc_lexer::LiteralKind::Str { terminated } => {
                if !terminated {
                    err = "Missing trailing `\"` symbol to terminate the string literal";
                } else {
                    let text = &self.res.text[self.offset + 1..][..len - 1];
                    let i = text.rfind('"').unwrap();
                    let text = &text[..i];
                    err = unescape_string_error_message(text, Mode::Str);
                }
                STRING
            }
            rustc_lexer::LiteralKind::ByteStr { terminated } => {
                if !terminated {
                    err = "Missing trailing `\"` symbol to terminate the byte string literal";
                } else {
                    let text = &self.res.text[self.offset + 2..][..len - 2];
                    let i = text.rfind('"').unwrap();
                    let text = &text[..i];
                    err = unescape_string_error_message(text, Mode::ByteStr);
                }
                BYTE_STRING
            }
            rustc_lexer::LiteralKind::CStr { terminated } => {
                if !terminated {
                    err = "Missing trailing `\"` symbol to terminate the string literal";
                } else {
                    let text = &self.res.text[self.offset + 2..][..len - 2];
                    let i = text.rfind('"').unwrap();
                    let text = &text[..i];
                    err = unescape_string_error_message(text, Mode::CStr);
                }
                C_STRING
            }
            rustc_lexer::LiteralKind::RawStr { n_hashes } => {
                if n_hashes.is_none() {
                    err = "Invalid raw string literal";
                }
                STRING
            }
            rustc_lexer::LiteralKind::RawByteStr { n_hashes } => {
                if n_hashes.is_none() {
                    err = "Invalid raw string literal";
                }
                BYTE_STRING
            }
            rustc_lexer::LiteralKind::RawCStr { n_hashes } => {
                if n_hashes.is_none() {
                    err = "Invalid raw string literal";
                }
                C_STRING
            }
        };

        let err = if err.is_empty() { None } else { Some(err) };
        self.push(syntax_kind, len, err);
    }
}

fn error_to_diagnostic_message(error: EscapeError, mode: Mode) -> &'static str {
    match error {
        EscapeError::ZeroChars => "empty character literal",
        EscapeError::MoreThanOneChar => "character literal may only contain one codepoint",
        EscapeError::LoneSlash => "",
        EscapeError::InvalidEscape if mode == Mode::Byte || mode == Mode::ByteStr => {
            "unknown byte escape"
        }
        EscapeError::InvalidEscape => "unknown character escape",
        EscapeError::BareCarriageReturn => "",
        EscapeError::BareCarriageReturnInRawString => "",
        EscapeError::EscapeOnlyChar if mode == Mode::Byte => "byte constant must be escaped",
        EscapeError::EscapeOnlyChar => "character constant must be escaped",
        EscapeError::TooShortHexEscape => "numeric character escape is too short",
        EscapeError::InvalidCharInHexEscape => "invalid character in numeric character escape",
        EscapeError::OutOfRangeHexEscape => "out of range hex escape",
        EscapeError::NoBraceInUnicodeEscape => "incorrect unicode escape sequence",
        EscapeError::InvalidCharInUnicodeEscape => "invalid character in unicode escape",
        EscapeError::EmptyUnicodeEscape => "empty unicode escape",
        EscapeError::UnclosedUnicodeEscape => "unterminated unicode escape",
        EscapeError::LeadingUnderscoreUnicodeEscape => "invalid start of unicode escape",
        EscapeError::OverlongUnicodeEscape => "overlong unicode escape",
        EscapeError::LoneSurrogateUnicodeEscape => "invalid unicode character escape",
        EscapeError::OutOfRangeUnicodeEscape => "invalid unicode character escape",
        EscapeError::UnicodeEscapeInByte => "unicode escape in byte string",
        EscapeError::NonAsciiCharInByte if mode == Mode::Byte => {
            "non-ASCII character in byte literal"
        }
        EscapeError::NonAsciiCharInByte if mode == Mode::ByteStr => {
            "non-ASCII character in byte string literal"
        }
        EscapeError::NonAsciiCharInByte => "non-ASCII character in raw byte string literal",
        EscapeError::NulInCStr => "null character in C string literal",
        EscapeError::UnskippedWhitespaceWarning => "",
        EscapeError::MultipleSkippedLinesWarning => "",
    }
}

fn unescape_string_error_message(text: &str, mode: Mode) -> &'static str {
    let mut error_message = "";
    match mode {
        Mode::CStr => {
            unescape_mixed(text, mode, &mut |_, res| {
                if let Err(e) = res {
                    error_message = error_to_diagnostic_message(e, mode);
                }
            });
        }
        Mode::ByteStr | Mode::Str => {
            unescape_unicode(text, mode, &mut |_, res| {
                if let Err(e) = res {
                    error_message = error_to_diagnostic_message(e, mode);
                }
            });
        }
        _ => {
            // Other Modes are not supported yet or do not apply
        }
    }
    error_message
}
