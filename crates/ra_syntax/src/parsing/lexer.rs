//! Lexer analyzes raw input string and produces lexemes (tokens).
//! It is just a bridge to `rustc_lexer`.

use crate::{
    SyntaxError, SyntaxErrorKind,
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

        let (syntax_kind, error) =
            rustc_token_kind_to_syntax_kind(&rustc_token.kind, &text[token_range]);

        tokens.push(Token { kind: syntax_kind, len: token_len });

        if let Some(error) = error {
            errors.push(SyntaxError::new(SyntaxErrorKind::TokenizeError(error), token_range));
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
    first_token(text)
        .filter(|(token, _)| token.len.to_usize() == text.len())
        .map(|(token, error)| (token.kind, error))
}

/// The same as `lex_single_syntax_kind()` but returns only `SyntaxKind` and
/// returns `None` if any tokenization error occured.
///
/// Beware that unescape errors are not checked at tokenization time.
pub fn lex_single_valid_syntax_kind(text: &str) -> Option<SyntaxKind> {
    first_token(text)
        .filter(|(token, error)| !error.is_some() && token.len.to_usize() == text.len())
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
fn first_token(text: &str) -> Option<(Token, Option<SyntaxError>)> {
    // non-empty string is a precondtion of `rustc_lexer::first_token()`.
    if text.is_empty() {
        return None;
    }

    let rustc_token = rustc_lexer::first_token(text);
    let (syntax_kind, error) = rustc_token_kind_to_syntax_kind(&rustc_token.kind, text);

    let token = Token { kind: syntax_kind, len: TextUnit::from_usize(rustc_token.len) };
    let error = error.map(|error| {
        SyntaxError::new(
            SyntaxErrorKind::TokenizeError(error),
            TextRange::from_to(TextUnit::from(0), TextUnit::of_str(text)),
        )
    });

    Some((token, error))
}

// FIXME: simplify TokenizeError to `SyntaxError(String, TextRange)` as per @matklad advice:
// https://github.com/rust-analyzer/rust-analyzer/pull/2911/files#r371175067

/// Describes the values of `SyntaxErrorKind::TokenizeError` enum variant.
/// It describes all the types of errors that may happen during the tokenization
/// of Rust source.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TokenizeError {
    /// Base prefix was provided, but there were no digits
    /// after it, e.g. `0x`, `0b`.
    EmptyInt,
    /// Float exponent lacks digits e.g. `12.34e+`, `12.3E+`, `12e-`, `1_E-`,
    EmptyExponent,

    /// Block comment lacks trailing delimiter `*/`
    UnterminatedBlockComment,
    /// Character literal lacks trailing delimiter `'`
    UnterminatedChar,
    /// Characterish byte literal lacks trailing delimiter `'`
    UnterminatedByte,
    /// String literal lacks trailing delimiter `"`
    UnterminatedString,
    /// Byte string literal lacks trailing delimiter `"`
    UnterminatedByteString,
    /// Raw literal lacks trailing delimiter e.g. `"##`
    UnterminatedRawString,
    /// Raw byte string literal lacks trailing delimiter e.g. `"##`
    UnterminatedRawByteString,

    /// Raw string lacks a quote after the pound characters e.g. `r###`
    UnstartedRawString,
    /// Raw byte string lacks a quote after the pound characters e.g. `br###`
    UnstartedRawByteString,

    /// Lifetime starts with a number e.g. `'4ever`
    LifetimeStartsWithNumber,
}

fn rustc_token_kind_to_syntax_kind(
    rustc_token_kind: &rustc_lexer::TokenKind,
    token_text: &str,
) -> (SyntaxKind, Option<TokenizeError>) {
    // A note on an intended tradeoff:
    // We drop some useful infromation here (see patterns with double dots `..`)
    // Storing that info in `SyntaxKind` is not possible due to its layout requirements of
    // being `u16` that come from `rowan::SyntaxKind`.

    let syntax_kind = {
        use rustc_lexer::TokenKind as TK;
        use TokenizeError as TE;

        match rustc_token_kind {
            TK::LineComment => COMMENT,

            TK::BlockComment { terminated: true } => COMMENT,
            TK::BlockComment { terminated: false } => {
                return (COMMENT, Some(TE::UnterminatedBlockComment));
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
                return (LIFETIME, Some(TE::LifetimeStartsWithNumber))
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

    fn match_literal_kind(kind: &rustc_lexer::LiteralKind) -> (SyntaxKind, Option<TokenizeError>) {
        use rustc_lexer::LiteralKind as LK;
        use TokenizeError as TE;

        #[rustfmt::skip]
        let syntax_kind = match *kind {
            LK::Int { empty_int: false, .. } => INT_NUMBER,
            LK::Int { empty_int: true, .. } => {
                return (INT_NUMBER, Some(TE::EmptyInt))
            }

            LK::Float { empty_exponent: false, .. } => FLOAT_NUMBER,
            LK::Float { empty_exponent: true, .. } => {
                return (FLOAT_NUMBER, Some(TE::EmptyExponent))
            }

            LK::Char { terminated: true } => CHAR,
            LK::Char { terminated: false } => {
                return (CHAR, Some(TE::UnterminatedChar))
            }

            LK::Byte { terminated: true } => BYTE,
            LK::Byte { terminated: false } => {
                return (BYTE, Some(TE::UnterminatedByte))
            }

            LK::Str { terminated: true } => STRING,
            LK::Str { terminated: false } => {
                return (STRING, Some(TE::UnterminatedString))
            }


            LK::ByteStr { terminated: true } => BYTE_STRING,
            LK::ByteStr { terminated: false } => {
                return (BYTE_STRING, Some(TE::UnterminatedByteString))
            }

            LK::RawStr { started: true, terminated: true, .. } => RAW_STRING,
            LK::RawStr { started: true, terminated: false, .. } => {
                return (RAW_STRING, Some(TE::UnterminatedRawString))
            }
            LK::RawStr { started: false, .. } => {
                return (RAW_STRING, Some(TE::UnstartedRawString))
            }

            LK::RawByteStr { started: true, terminated: true, .. } => RAW_BYTE_STRING,
            LK::RawByteStr { started: true, terminated: false, .. } => {
                return (RAW_BYTE_STRING, Some(TE::UnterminatedRawByteString))
            }
            LK::RawByteStr { started: false, .. } => {
                return (RAW_BYTE_STRING, Some(TE::UnstartedRawByteString))
            }
        };

        (syntax_kind, None)
    }
}
