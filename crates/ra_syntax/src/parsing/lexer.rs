//! Lexer analyzes raw input string and produces lexemes (tokens).
//! It is just a bridge to `rustc_lexer`.

use crate::{
    SyntaxError,
    SyntaxKind::{self, *},
    TextRange, TextUnit,
};

/// A token of Rust source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Token {
    /// The kind of token.
    pub kind: SyntaxKind,
    /// The length of the token.
    pub len: TextUnit,
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

    let mut offset: usize = rustc_lexer::strip_shebang(text)
        .map(|shebang_len| {
            tokens.push(Token { kind: SHEBANG, len: TextUnit::from_usize(shebang_len) });
            shebang_len
        })
        .unwrap_or(0);

    let text_without_shebang = &text[offset..];

    for rustc_token in rustc_lexer::tokenize(text_without_shebang) {
        let token_len = TextUnit::from_usize(rustc_token.len);
        let token_range = TextRange::offset_len(TextUnit::from_usize(offset), token_len);

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
        .filter(|(token, _)| token.len == TextUnit::of_str(text))
        .map(|(token, error)| (token.kind, error))
}

/// The same as `lex_single_syntax_kind()` but returns only `SyntaxKind` and
/// returns `None` if any tokenization error occured.
///
/// Beware that unescape errors are not checked at tokenization time.
pub fn lex_single_valid_syntax_kind(text: &str) -> Option<SyntaxKind> {
    lex_first_token(text)
        .filter(|(token, error)| !error.is_some() && token.len == TextUnit::of_str(text))
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

    let token = Token { kind: syntax_kind, len: TextUnit::from_usize(rustc_token.len) };
    let optional_error = err_message.map(|err_message| {
        SyntaxError::new(err_message, TextRange::from_to(0.into(), TextUnit::of_str(text)))
    });

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
        use rustc_lexer::TokenKind as TK;
        match rustc_token_kind {
            TK::LineComment => COMMENT,

            TK::BlockComment { terminated: true } => COMMENT,
            TK::BlockComment { terminated: false } => {
                return (
                    COMMENT,
                    Some("Missing trailing `*/` symbols to terminate the block comment"),
                );
            }

            TK::Whitespace => WHITESPACE,

            TK::Ident => {
                if token_text == "_" {
                    UNDERSCORE
                } else {
                    SyntaxKind::from_keyword(token_text).unwrap_or(IDENT)
                }
            }

            TK::RawIdent => IDENT,
            TK::Literal { kind, .. } => return match_literal_kind(&kind),

            TK::Lifetime { starts_with_number: false } => LIFETIME,
            TK::Lifetime { starts_with_number: true } => {
                return (LIFETIME, Some("Lifetime name cannot start with a number"))
            }

            TK::Semi => SEMI,
            TK::Comma => COMMA,
            TK::Dot => DOT,
            TK::OpenParen => L_PAREN,
            TK::CloseParen => R_PAREN,
            TK::OpenBrace => L_CURLY,
            TK::CloseBrace => R_CURLY,
            TK::OpenBracket => L_BRACK,
            TK::CloseBracket => R_BRACK,
            TK::At => AT,
            TK::Pound => POUND,
            TK::Tilde => TILDE,
            TK::Question => QUESTION,
            TK::Colon => COLON,
            TK::Dollar => DOLLAR,
            TK::Eq => EQ,
            TK::Not => EXCL,
            TK::Lt => L_ANGLE,
            TK::Gt => R_ANGLE,
            TK::Minus => MINUS,
            TK::And => AMP,
            TK::Or => PIPE,
            TK::Plus => PLUS,
            TK::Star => STAR,
            TK::Slash => SLASH,
            TK::Caret => CARET,
            TK::Percent => PERCENT,
            TK::Unknown => ERROR,
        }
    };

    return (syntax_kind, None);

    fn match_literal_kind(kind: &rustc_lexer::LiteralKind) -> (SyntaxKind, Option<&'static str>) {
        use rustc_lexer::LiteralKind as LK;

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

            LK::RawStr { started: true, terminated: true, .. } => RAW_STRING,
            LK::RawStr { started: true, terminated: false, .. } => {
                return (RAW_STRING, Some("Missing trailing `\"` with `#` symbols to terminate the raw string literal"))
            }
            LK::RawStr { started: false, .. } => {
                return (RAW_STRING, Some("Missing `\"` symbol after `#` symbols to begin the raw string literal"))
            }

            LK::RawByteStr { started: true, terminated: true, .. } => RAW_BYTE_STRING,
            LK::RawByteStr { started: true, terminated: false, .. } => {
                return (RAW_BYTE_STRING, Some("Missing trailing `\"` with `#` symbols to terminate the raw byte string literal"))
            }
            LK::RawByteStr { started: false, .. } => {
                return (RAW_BYTE_STRING, Some("Missing `\"` symbol after `#` symbols to begin the raw byte string literal"))
            }
        };

        (syntax_kind, None)
    }
}
