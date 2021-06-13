//! Lexer analyzes raw input string and produces lexemes (tokens).
//! It is just a bridge to `rustc_lexer`.

use std::convert::TryInto;

use rustc_lexer::RawStrError;

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
    // non-empty string is a precondition of `rustc_lexer::strip_shebang()`.
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

/// Returns `SyntaxKind` and `Option<SyntaxError>` if `text` parses as a single token.
///
/// Returns `None` if the string contains zero *or two or more* tokens.
/// The token is malformed if the returned error is not `None`.
///
/// Beware that unescape errors are not checked at tokenization time.
pub fn lex_single_syntax_kind(text: &str) -> Option<(SyntaxKind, Option<SyntaxError>)> {
    let (first_token, err) = lex_first_token(text)?;
    if first_token.len != TextSize::of(text) {
        return None;
    }
    Some((first_token.kind, err))
}

/// The same as `lex_single_syntax_kind()` but returns only `SyntaxKind` and
/// returns `None` if any tokenization error occurred.
///
/// Beware that unescape errors are not checked at tokenization time.
pub fn lex_single_valid_syntax_kind(text: &str) -> Option<SyntaxKind> {
    let (single_token, err) = lex_single_syntax_kind(text)?;
    if err.is_some() {
        return None;
    }
    Some(single_token)
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
    // non-empty string is a precondition of `rustc_lexer::first_token()`.
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
    // We drop some useful information here (see patterns with double dots `..`)
    // Storing that info in `SyntaxKind` is not possible due to its layout requirements of
    // being `u16` that come from `rowan::SyntaxKind`.

    let syntax_kind = {
        match rustc_token_kind {
            rustc_lexer::TokenKind::LineComment { doc_style: _ } => COMMENT,

            rustc_lexer::TokenKind::BlockComment { doc_style: _, terminated: true } => COMMENT,
            rustc_lexer::TokenKind::BlockComment { doc_style: _, terminated: false } => {
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
            rustc_lexer::TokenKind::Literal { kind, .. } => return match_literal_kind(kind),

            rustc_lexer::TokenKind::Lifetime { starts_with_number: false } => LIFETIME_IDENT,
            rustc_lexer::TokenKind::Lifetime { starts_with_number: true } => {
                return (LIFETIME_IDENT, Some("Lifetime name cannot start with a number"))
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

    return (syntax_kind, None);

    fn match_literal_kind(kind: &rustc_lexer::LiteralKind) -> (SyntaxKind, Option<&'static str>) {
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
                        RawStrError::InvalidStarter { .. } => "Missing `\"` symbol after `#` symbols to begin the raw string literal",
                        RawStrError::NoTerminator { expected, found, .. } => if expected == found {
                            "Missing trailing `\"` to terminate the raw string literal"
                        } else {
                            "Missing trailing `\"` with `#` symbols to terminate the raw string literal"
                        },
                        RawStrError::TooManyDelimiters { .. } => "Too many `#` symbols: raw strings may be delimited by up to 65535 `#` symbols",
                    };
                };
                STRING
            }
            rustc_lexer::LiteralKind::RawByteStr { err: raw_str_err, .. } => {
                if let Some(raw_str_err) = raw_str_err {
                    err = match raw_str_err {
                        RawStrError::InvalidStarter { .. } => "Missing `\"` symbol after `#` symbols to begin the raw byte string literal",
                        RawStrError::NoTerminator { expected, found, .. } => if expected == found {
                            "Missing trailing `\"` to terminate the raw byte string literal"
                        } else {
                            "Missing trailing `\"` with `#` symbols to terminate the raw byte string literal"
                        },
                        RawStrError::TooManyDelimiters { .. } => "Too many `#` symbols: raw byte strings may be delimited by up to 65535 `#` symbols",
                    };
                };

                BYTE_STRING
            }
        };

        let err = if err.is_empty() { None } else { Some(err) };

        (syntax_kind, err)
    }
}
