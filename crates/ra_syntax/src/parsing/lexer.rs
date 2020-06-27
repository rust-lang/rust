//! Lexer analyzes raw input string and produces lexemes (tokens).
//! It is just a bridge to `rustc_lexer`.

use rustc_lexer::{LiteralKind as LK, RawStrError};

use std::convert::TryInto;

use crate::{
    SyntaxError,
    SyntaxKind::{self, *},
    TextRange, TextSize, T,
};

/// A token of Rust source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Token {
    /// The kind of token.
    pub kind: SyntaxKind,
    /// The length of the token.
    pub len: TextSize,
}

/// Break a string up into its component tokens.
/// Beware that it checks for shebang first and its length contributes to resulting
/// tokens offsets.
pub fn tokenize(text: &str) -> (Vec<Token>, Vec<SyntaxError>) {
    // non-empty string is a precondtion of `rustc_lexer::strip_shebang()`.
    if text.is_empty() {
        return Default::default();
    }

    let mut tokens = Vec::new();
    let mut errors = Vec::new();

    let mut offset = match rustc_lexer::strip_shebang(text) {
        Some(shebang_len) => {
            tokens.push(Token { kind: SHEBANG, len: shebang_len.try_into().unwrap() });
            shebang_len
        }
        None => 0,
    };

    let text_without_shebang = &text[offset..];

    for rustc_token in rustc_lexer::tokenize(text_without_shebang) {
        let token_len: TextSize = rustc_token.len.try_into().unwrap();
        let token_range = TextRange::at(offset.try_into().unwrap(), token_len);

        let (syntax_kind, err_message) =
            rustc_token_kind_to_syntax_kind(&rustc_token.kind, &text[token_range]);

        tokens.push(Token { kind: syntax_kind, len: token_len });

        if let Some(err_message) = err_message {
            errors.push(SyntaxError::new(err_message, token_range));
        }

        offset += rustc_token.len;
    }

    (tokens, errors)
}

/// Returns `SyntaxKind` and `Option<SyntaxError>` of the first token
/// encountered at the beginning of the string.
///
/// Returns `None` if the string contains zero *or two or more* tokens.
/// The token is malformed if the returned error is not `None`.
///
/// Beware that unescape errors are not checked at tokenization time.
pub fn lex_single_syntax_kind(text: &str) -> Option<(SyntaxKind, Option<SyntaxError>)> {
    lex_first_token(text)
        .filter(|(token, _)| token.len == TextSize::of(text))
        .map(|(token, error)| (token.kind, error))
}

/// The same as `lex_single_syntax_kind()` but returns only `SyntaxKind` and
/// returns `None` if any tokenization error occured.
///
/// Beware that unescape errors are not checked at tokenization time.
pub fn lex_single_valid_syntax_kind(text: &str) -> Option<SyntaxKind> {
    lex_first_token(text)
        .filter(|(token, error)| !error.is_some() && token.len == TextSize::of(text))
        .map(|(token, _error)| token.kind)
}

/// Returns `SyntaxKind` and `Option<SyntaxError>` of the first token
/// encountered at the beginning of the string.
///
/// Returns `None` if the string contains zero tokens or if the token was parsed
/// with an error.
/// The token is malformed if the returned error is not `None`.
///
/// Beware that unescape errors are not checked at tokenization time.
fn lex_first_token(text: &str) -> Option<(Token, Option<SyntaxError>)> {
    // non-empty string is a precondtion of `rustc_lexer::first_token()`.
    if text.is_empty() {
        return None;
    }

    let rustc_token = rustc_lexer::first_token(text);
    let (syntax_kind, err_message) = rustc_token_kind_to_syntax_kind(&rustc_token.kind, text);

    let token = Token { kind: syntax_kind, len: rustc_token.len.try_into().unwrap() };
    let optional_error = err_message
        .map(|err_message| SyntaxError::new(err_message, TextRange::up_to(TextSize::of(text))));

    Some((token, optional_error))
}

/// Returns `SyntaxKind` and an optional tokenize error message.
fn rustc_token_kind_to_syntax_kind(
    rustc_token_kind: &rustc_lexer::TokenKind,
    token_text: &str,
) -> (SyntaxKind, Option<&'static str>) {
    // A note on an intended tradeoff:
    // We drop some useful infromation here (see patterns with double dots `..`)
    // Storing that info in `SyntaxKind` is not possible due to its layout requirements of
    // being `u16` that come from `rowan::SyntaxKind`.

    let syntax_kind = {
        match rustc_token_kind {
            rustc_lexer::TokenKind::LineComment => COMMENT,

            rustc_lexer::TokenKind::BlockComment { terminated: true } => COMMENT,
            rustc_lexer::TokenKind::BlockComment { terminated: false } => {
                return (
                    COMMENT,
                    Some("Missing trailing `*/` symbols to terminate the block comment"),
                );
            }

            rustc_lexer::TokenKind::Whitespace => WHITESPACE,

            rustc_lexer::TokenKind::Ident => {
                if token_text == "_" {
                    UNDERSCORE
                } else {
                    SyntaxKind::from_keyword(token_text).unwrap_or(IDENT)
                }
            }

            rustc_lexer::TokenKind::RawIdent => IDENT,
            rustc_lexer::TokenKind::Literal { kind, .. } => return match_literal_kind(&kind),

            rustc_lexer::TokenKind::Lifetime { starts_with_number: false } => LIFETIME,
            rustc_lexer::TokenKind::Lifetime { starts_with_number: true } => {
                return (LIFETIME, Some("Lifetime name cannot start with a number"))
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
            rustc_lexer::TokenKind::Not => T![!],
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

    return (syntax_kind, None);

    fn match_literal_kind(kind: &rustc_lexer::LiteralKind) -> (SyntaxKind, Option<&'static str>) {
        #[rustfmt::skip]
        let syntax_kind = match *kind {
            LK::Int { empty_int: false, .. } => INT_NUMBER,
            LK::Int { empty_int: true, .. } => {
                return (INT_NUMBER, Some("Missing digits after the integer base prefix"))
            }

            LK::Float { empty_exponent: false, .. } => FLOAT_NUMBER,
            LK::Float { empty_exponent: true, .. } => {
                return (FLOAT_NUMBER, Some("Missing digits after the exponent symbol"))
            }

            LK::Char { terminated: true } => CHAR,
            LK::Char { terminated: false } => {
                return (CHAR, Some("Missing trailing `'` symbol to terminate the character literal"))
            }

            LK::Byte { terminated: true } => BYTE,
            LK::Byte { terminated: false } => {
                return (BYTE, Some("Missing trailing `'` symbol to terminate the byte literal"))
            }

            LK::Str { terminated: true } => STRING,
            LK::Str { terminated: false } => {
                return (STRING, Some("Missing trailing `\"` symbol to terminate the string literal"))
            }


            LK::ByteStr { terminated: true } => BYTE_STRING,
            LK::ByteStr { terminated: false } => {
                return (BYTE_STRING, Some("Missing trailing `\"` symbol to terminate the byte string literal"))
            }

            LK::RawStr { err, .. } => match err {
                None => RAW_STRING,
                Some(RawStrError::InvalidStarter { .. }) => return (RAW_STRING, Some("Missing `\"` symbol after `#` symbols to begin the raw string literal")),
                Some(RawStrError::NoTerminator { expected, found, .. }) => if expected == found {
                    return (RAW_STRING, Some("Missing trailing `\"` to terminate the raw string literal"))
                } else {
                    return (RAW_STRING, Some("Missing trailing `\"` with `#` symbols to terminate the raw string literal"))

                },
                Some(RawStrError::TooManyDelimiters { .. }) => return (RAW_STRING, Some("Too many `#` symbols: raw strings may be delimited by up to 65535 `#` symbols")),
            },
            LK::RawByteStr { err, .. } => match err {
                None => RAW_BYTE_STRING,
                Some(RawStrError::InvalidStarter { .. }) => return (RAW_BYTE_STRING, Some("Missing `\"` symbol after `#` symbols to begin the raw byte string literal")),
                Some(RawStrError::NoTerminator { expected, found, .. }) => if expected == found {
                    return (RAW_BYTE_STRING, Some("Missing trailing `\"` to terminate the raw byte string literal"))
                } else {
                    return (RAW_BYTE_STRING, Some("Missing trailing `\"` with `#` symbols to terminate the raw byte string literal"))

                },
                Some(RawStrError::TooManyDelimiters { .. }) => return (RAW_BYTE_STRING, Some("Too many `#` symbols: raw byte strings may be delimited by up to 65535 `#` symbols")),
            },
        };

        (syntax_kind, None)
    }
}
