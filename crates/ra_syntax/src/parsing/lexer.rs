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

#[derive(Debug)]
/// Represents the result of parsing one token. Beware that the token may be malformed.
pub struct ParsedToken {
    /// Parsed token.
    pub token: Token,
    /// If error is present then parsed token is malformed.
    pub error: Option<SyntaxError>,
}

#[derive(Debug, Default)]
/// Represents the result of parsing source code of Rust language.
pub struct ParsedTokens {
    /// Parsed tokens in order they appear in source code.
    pub tokens: Vec<Token>,
    /// Collection of all occured tokenization errors.
    /// In general `self.errors.len() <= self.tokens.len()`
    pub errors: Vec<SyntaxError>,
}
impl ParsedTokens {
    /// Append `token` and `error` (if pressent) to the result.
    pub fn push(&mut self, ParsedToken { token, error }: ParsedToken) {
        self.tokens.push(token);
        if let Some(error) = error {
            self.errors.push(error)
        }
    }
}

/// Same as `tokenize_append()`, just a shortcut for creating `ParsedTokens`
/// and returning the result the usual way.
pub fn tokenize(text: &str) -> ParsedTokens {
    let mut parsed = ParsedTokens::default();
    tokenize_append(text, &mut parsed);
    parsed
}

/// Break a string up into its component tokens.
/// Returns `ParsedTokens` which are basically a pair `(Vec<Token>, Vec<SyntaxError>)`.
/// Beware that it checks for shebang first and its length contributes to resulting
/// tokens offsets.
pub fn tokenize_append(text: &str, parsed: &mut ParsedTokens) {
    // non-empty string is a precondtion of `rustc_lexer::strip_shebang()`.
    if text.is_empty() {
        return;
    }

    let mut offset: usize = rustc_lexer::strip_shebang(text)
        .map(|shebang_len| {
            parsed.tokens.push(Token { kind: SHEBANG, len: TextUnit::from_usize(shebang_len) });
            shebang_len
        })
        .unwrap_or(0);

    let text_without_shebang = &text[offset..];

    for rustc_token in rustc_lexer::tokenize(text_without_shebang) {
        parsed.push(rustc_token_to_parsed_token(&rustc_token, text, TextUnit::from_usize(offset)));
        offset += rustc_token.len;
    }
}

/// Returns the first encountered token at the beginning of the string.
/// If the string contains zero or *two or more tokens* returns `None`.
///
/// The main difference between `first_token()` and `single_token()` is that
/// the latter returns `None` if the string contains more than one token.
pub fn single_token(text: &str) -> Option<ParsedToken> {
    first_token(text).filter(|parsed| parsed.token.len.to_usize() == text.len())
}

/// Returns the first encountered token at the beginning of the string.
/// If the string contains zero tokens returns `None`.
///
/// The main difference between `first_token() and single_token()` is that
/// the latter returns `None` if the string contains more than one token.
pub fn first_token(text: &str) -> Option<ParsedToken> {
    // non-empty string is a precondtion of `rustc_lexer::first_token()`.
    if text.is_empty() {
        None
    } else {
        let rustc_token = rustc_lexer::first_token(text);
        Some(rustc_token_to_parsed_token(&rustc_token, text, TextUnit::from(0)))
    }
}

/// Describes the values of `SyntaxErrorKind::TokenizeError` enum variant.
/// It describes all the types of errors that may happen during the tokenization
/// of Rust source.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TokenizeError {
    /// Base prefix was provided, but there were no digits
    /// after it, e.g. `0x`.
    EmptyInt,
    /// Float exponent lacks digits e.g. `e+`, `E+`, `e-`, `E-`,
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

/// Mapper function that converts `rustc_lexer::Token` with some additional context
/// to `ParsedToken`
fn rustc_token_to_parsed_token(
    rustc_token: &rustc_lexer::Token,
    text: &str,
    token_start_offset: TextUnit,
) -> ParsedToken {
    // We drop some useful infromation here (see patterns with double dots `..`)
    // Storing that info in `SyntaxKind` is not possible due to its layout requirements of
    // being `u16` that come from `rowan::SyntaxKind` type and changes to `rowan::SyntaxKind`
    // would mean hell of a rewrite

    let token_range =
        TextRange::offset_len(token_start_offset, TextUnit::from_usize(rustc_token.len));

    let token_text = &text[token_range];

    let (syntax_kind, error) = {
        use rustc_lexer::TokenKind as TK;
        use TokenizeError as TE;

        match rustc_token.kind {
            TK::LineComment => ok(COMMENT),
            TK::BlockComment { terminated } => {
                ok_if(terminated, COMMENT, TE::UnterminatedBlockComment)
            }
            TK::Whitespace => ok(WHITESPACE),
            TK::Ident => ok(if token_text == "_" {
                UNDERSCORE
            } else {
                SyntaxKind::from_keyword(token_text).unwrap_or(IDENT)
            }),
            TK::RawIdent => ok(IDENT),
            TK::Literal { kind, .. } => match_literal_kind(&kind),
            TK::Lifetime { starts_with_number } => {
                ok_if(!starts_with_number, LIFETIME, TE::LifetimeStartsWithNumber)
            }
            TK::Semi => ok(SEMI),
            TK::Comma => ok(COMMA),
            TK::Dot => ok(DOT),
            TK::OpenParen => ok(L_PAREN),
            TK::CloseParen => ok(R_PAREN),
            TK::OpenBrace => ok(L_CURLY),
            TK::CloseBrace => ok(R_CURLY),
            TK::OpenBracket => ok(L_BRACK),
            TK::CloseBracket => ok(R_BRACK),
            TK::At => ok(AT),
            TK::Pound => ok(POUND),
            TK::Tilde => ok(TILDE),
            TK::Question => ok(QUESTION),
            TK::Colon => ok(COLON),
            TK::Dollar => ok(DOLLAR),
            TK::Eq => ok(EQ),
            TK::Not => ok(EXCL),
            TK::Lt => ok(L_ANGLE),
            TK::Gt => ok(R_ANGLE),
            TK::Minus => ok(MINUS),
            TK::And => ok(AMP),
            TK::Or => ok(PIPE),
            TK::Plus => ok(PLUS),
            TK::Star => ok(STAR),
            TK::Slash => ok(SLASH),
            TK::Caret => ok(CARET),
            TK::Percent => ok(PERCENT),
            TK::Unknown => ok(ERROR),
        }
    };

    return ParsedToken {
        token: Token { kind: syntax_kind, len: token_range.len() },
        error: error
            .map(|error| SyntaxError::new(SyntaxErrorKind::TokenizeError(error), token_range)),
    };

    type ParsedSyntaxKind = (SyntaxKind, Option<TokenizeError>);

    fn match_literal_kind(kind: &rustc_lexer::LiteralKind) -> ParsedSyntaxKind {
        use rustc_lexer::LiteralKind as LK;
        use TokenizeError as TE;

        match *kind {
            LK::Int { empty_int, .. } => ok_if(!empty_int, INT_NUMBER, TE::EmptyInt),
            LK::Float { empty_exponent, .. } => {
                ok_if(!empty_exponent, FLOAT_NUMBER, TE::EmptyExponent)
            }
            LK::Char { terminated } => ok_if(terminated, CHAR, TE::UnterminatedChar),
            LK::Byte { terminated } => ok_if(terminated, BYTE, TE::UnterminatedByte),
            LK::Str { terminated } => ok_if(terminated, STRING, TE::UnterminatedString),
            LK::ByteStr { terminated } => {
                ok_if(terminated, BYTE_STRING, TE::UnterminatedByteString)
            }

            LK::RawStr { started: true, terminated, .. } => {
                ok_if(terminated, RAW_STRING, TE::UnterminatedRawString)
            }
            LK::RawStr { started: false, .. } => err(RAW_STRING, TE::UnstartedRawString),

            LK::RawByteStr { started: true, terminated, .. } => {
                ok_if(terminated, RAW_BYTE_STRING, TE::UnterminatedRawByteString)
            }
            LK::RawByteStr { started: false, .. } => {
                err(RAW_BYTE_STRING, TE::UnstartedRawByteString)
            }
        }
    }
    const fn ok(syntax_kind: SyntaxKind) -> ParsedSyntaxKind {
        (syntax_kind, None)
    }
    const fn err(syntax_kind: SyntaxKind, error: TokenizeError) -> ParsedSyntaxKind {
        (syntax_kind, Some(error))
    }
    fn ok_if(cond: bool, syntax_kind: SyntaxKind, error: TokenizeError) -> ParsedSyntaxKind {
        if cond {
            ok(syntax_kind)
        } else {
            err(syntax_kind, error)
        }
    }
}
