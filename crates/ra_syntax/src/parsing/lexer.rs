//! Lexer analyzes raw input string and produces lexemes (tokens).

use std::iter::{FromIterator, IntoIterator};

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
impl Token {
    pub const fn new(kind: SyntaxKind, len: TextUnit) -> Self {
        Self { kind, len }
    }
}

#[derive(Debug)]
/// Represents the result of parsing one token.
pub struct ParsedToken {
    /// Parsed token.
    pub token: Token,
    /// If error is present then parsed token is malformed.
    pub error: Option<TokenizeError>,
}
impl ParsedToken {
    pub const fn new(token: Token, error: Option<TokenizeError>) -> Self {
        Self { token, error }
    }
}

#[derive(Debug, Default)]
/// Represents the result of parsing one token.
pub struct ParsedTokens {
    /// Parsed token.
    pub tokens: Vec<Token>,
    /// If error is present then parsed token is malformed.
    pub errors: Vec<TokenizeError>,
}

impl FromIterator<ParsedToken> for ParsedTokens {
    fn from_iter<I: IntoIterator<Item = ParsedToken>>(iter: I) -> Self {
        let res = Self::default();
        for entry in iter {
            res.tokens.push(entry.token);
            if let Some(error) = entry.error {
                res.errors.push(error);
            }
        }
        res
    }
}

/// Returns the first encountered token from the string.
/// If the string contains zero or two or more tokens returns `None`.
pub fn single_token(text: &str) -> Option<ParsedToken> {
    // TODO: test whether this condition indeed checks for a single token
    first_token(text).filter(|parsed| parsed.token.len.to_usize() == text.len())
}

/*
/// Returns `ParsedTokens` which are basically a pair `(Vec<Token>, Vec<TokenizeError>)`
/// This is just a shorthand for `tokenize(text).collect()`
pub fn tokenize_to_vec_with_errors(text: &str) -> ParsedTokens {
    tokenize(text).collect()
}

/// The simplest version of tokenize, it just retunst a ready-made `Vec<Token>`.
/// It discards all tokenization errors while parsing. If you need that infromation
/// consider using `tokenize()` or `tokenize_to_vec_with_errors()`.
pub fn tokenize_to_vec(text: &str) -> Vec<Token> {
    tokenize(text).map(|parsed_token| parsed_token.token).collect()
}
*/

/// Break a string up into its component tokens
/// This is the core function, all other `tokenize*()` functions are simply
/// handy shortcuts for this one.
pub fn tokenize(text: &str) -> impl Iterator<Item = ParsedToken> + '_ {
    let shebang = rustc_lexer::strip_shebang(text).map(|shebang_len| {
        text = &text[shebang_len..];
        ParsedToken::new(Token::new(SHEBANG, TextUnit::from_usize(shebang_len)), None)
    });

    // Notice that we eagerly evaluate shebang since it may change text slice
    // and we cannot simplify this into a single method call chain
    shebang.into_iter().chain(tokenize_without_shebang(text))
}

pub fn tokenize_without_shebang(text: &str) -> impl Iterator<Item = ParsedToken> + '_ {
    rustc_lexer::tokenize(text).map(|rustc_token| {
        let token_text = &text[..rustc_token.len];
        text = &text[rustc_token.len..];
        rustc_token_kind_to_parsed_token(&rustc_token.kind, token_text)
    })
}

#[derive(Debug)]
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

    /// Raw string lacks a quote after pound characters e.g. `r###`
    UnstartedRawString,
    /// Raw byte string lacks a quote after pound characters e.g. `br###`
    UnstartedRawByteString,

    /// Lifetime starts with a number e.g. `'4ever`
    LifetimeStartsWithNumber,
}

fn rustc_token_kind_to_parsed_token(
    rustc_token_kind: &rustc_lexer::TokenKind,
    token_text: &str,
) -> ParsedToken {
    use rustc_lexer::TokenKind as TK;
    use TokenizeError as TE;

    // We drop some useful infromation here (see patterns with double dots `..`)
    // Storing that info in `SyntaxKind` is not possible due to its layout requirements of
    // being `u16` that come from `rowan::SyntaxKind` type and changes to `rowan::SyntaxKind`
    // would mean hell of a rewrite.

    let (syntax_kind, error) = match *rustc_token_kind {
        TK::LineComment => ok(COMMENT),
        TK::BlockComment { terminated } => ok_if(terminated, COMMENT, TE::UnterminatedBlockComment),
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
    };

    return ParsedToken::new(
        Token::new(syntax_kind, TextUnit::from_usize(token_text.len())),
        error,
    );

    type ParsedSyntaxKind = (SyntaxKind, Option<TokenizeError>);

    const fn ok(syntax_kind: SyntaxKind) -> ParsedSyntaxKind {
        (syntax_kind, None)
    }
    const fn ok_if(cond: bool, syntax_kind: SyntaxKind, error: TokenizeError) -> ParsedSyntaxKind {
        if cond {
            ok(syntax_kind)
        } else {
            err(syntax_kind, error)
        }
    }
    const fn err(syntax_kind: SyntaxKind, error: TokenizeError) -> ParsedSyntaxKind {
        (syntax_kind, Some(error))
    }

    const fn match_literal_kind(kind: &rustc_lexer::LiteralKind) -> ParsedSyntaxKind {
        use rustc_lexer::LiteralKind as LK;
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
}

pub fn first_token(text: &str) -> Option<ParsedToken> {
    // Checking for emptyness because of `rustc_lexer::first_token()` invariant (see its body)
    if text.is_empty() {
        None
    } else {
        let rustc_token = rustc_lexer::first_token(text);
        Some(rustc_token_kind_to_parsed_token(&rustc_token.kind, &text[..rustc_token.len]))
    }
}

// TODO: think what to do with this ad hoc function
pub fn classify_literal(text: &str) -> Option<ParsedToken> {
    let t = rustc_lexer::first_token(text);
    if t.len != text.len() {
        return None;
    }
    let kind = match t.kind {
        rustc_lexer::TokenKind::Literal { kind, .. } => match_literal_kind(kind),
        _ => return None,
    };
    Some(ParsedToken::new(Token::new(kind, TextUnit::from_usize(t.len))))
}
