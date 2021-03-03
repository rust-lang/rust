//! Low-level Rust lexer.
//!
//! The idea with `librustc_lexer` is to make a reusable library,
//! by separating out pure lexing and rustc-specific concerns, like spans,
//! error reporting, and interning.  So, rustc_lexer operates directly on `&str`,
//! produces simple tokens which are a pair of type-tag and a bit of original text,
//! and does not report errors, instead storing them as flags on the token.
//!
//! Tokens produced by this lexer are not yet ready for parsing the Rust syntax.
//! For that see [`librustc_parse::lexer`], which converts this basic token stream
//! into wide tokens used by actual parser.
//!
//! The purpose of this crate is to convert raw sources into a labeled sequence
//! of well-known token types, so building an actual Rust token stream will
//! be easier.
//!
//! The main entity of this crate is the [`TokenKind`] enum which represents common
//! lexeme types.
//!
//! [`librustc_parse::lexer`]: ../rustc_parse/lexer/index.html
// We want to be able to build this crate with a stable compiler, so no
// `#![feature]` attributes should be added.

mod cursor;
mod literals;
pub mod unescape;

#[cfg(test)]
mod tests;

use self::TokenKind::*;
use crate::cursor::Cursor;
use crate::literals::{
    double_quoted_string, eat_literal_suffix, lifetime_or_char, number, raw_double_quoted_string,
    single_quoted_string, LiteralKind,
};

/// Parsed token.
/// It doesn't contain information about data that has been parsed,
/// only the type of the token and its size.
#[derive(Debug)]
pub struct Token {
    pub kind: TokenKind,
    pub len: usize,
}

impl Token {
    fn new(kind: TokenKind, len: usize) -> Token {
        Token { kind, len }
    }
}

/// Enum representing common lexeme types.
// perf note: Changing all `usize` to `u32` doesn't change performance. See #77629
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum TokenKind {
    // Multi-char tokens:
    /// "// comment"
    LineComment { doc_style: Option<DocStyle> },
    /// `/* block comment */`
    ///
    /// Block comments can be recursive, so the sequence like `/* /* */`
    /// will not be considered terminated and will result in a parsing error.
    BlockComment { doc_style: Option<DocStyle>, terminated: bool },
    /// Any whitespace characters sequence.
    Whitespace,
    /// "ident" or "continue"
    /// At this step keywords are also considered identifiers.
    Ident,
    /// "r#ident"
    RawIdent,
    /// "12_u8", "1.0e-40", "b"123"". See `LiteralKind` for more details.
    Literal { kind: LiteralKind, suffix_start: usize },
    /// "'a"
    Lifetime { starts_with_number: bool },

    // One-char tokens:
    /// ";"
    Semi,
    /// ","
    Comma,
    /// "."
    Dot,
    /// "("
    OpenParen,
    /// ")"
    CloseParen,
    /// "{"
    OpenBrace,
    /// "}"
    CloseBrace,
    /// "["
    OpenBracket,
    /// "]"
    CloseBracket,
    /// "@"
    At,
    /// "#"
    Pound,
    /// "~"
    Tilde,
    /// "?"
    Question,
    /// ":"
    Colon,
    /// "$"
    Dollar,
    /// "="
    Eq,
    /// "!"
    Bang,
    /// "<"
    Lt,
    /// ">"
    Gt,
    /// "-"
    Minus,
    /// "&"
    And,
    /// "|"
    Or,
    /// "+"
    Plus,
    /// "*"
    Star,
    /// "/"
    Slash,
    /// "^"
    Caret,
    /// "%"
    Percent,

    /// Unknown token, not expected by the lexer, e.g. "№"
    Unknown,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum DocStyle {
    Outer,
    Inner,
}

/// `rustc` allows files to have a shebang, e.g. "#!/usr/bin/rustrun",
/// but shebang isn't a part of rust syntax.
pub fn strip_shebang(input: &str) -> Option<usize> {
    // Shebang must start with `#!` literally, without any preceding whitespace.
    // For simplicity we consider any line starting with `#!` a shebang,
    // regardless of restrictions put on shebangs by specific platforms.
    if let Some(input_tail) = input.strip_prefix("#!") {
        // Ok, this is a shebang but if the next non-whitespace token is `[`,
        // then it may be valid Rust code, so consider it Rust code.
        let next_non_whitespace_token = tokenize(input_tail).map(|tok| tok.kind).find(|tok| {
            !matches!(
                tok,
                TokenKind::Whitespace
                    | TokenKind::LineComment { doc_style: None }
                    | TokenKind::BlockComment { doc_style: None, .. }
            )
        });
        if next_non_whitespace_token != Some(TokenKind::OpenBracket) {
            // No other choice than to consider this a shebang.
            return Some(2 + input_tail.lines().next().unwrap_or_default().len());
        }
    }
    None
}

/// Parses the first token from the provided input string.
pub fn first_token(input: &str) -> Token {
    debug_assert!(!input.is_empty());
    advance_token(&mut Cursor::new(input))
}

/// Creates an iterator that produces tokens from the input string.
pub fn tokenize(mut input: &str) -> impl Iterator<Item = Token> + '_ {
    std::iter::from_fn(move || {
        if input.is_empty() {
            return None;
        }
        let token = first_token(input);
        input = &input[token.len..];
        Some(token)
    })
}

/// True if `c` is considered a whitespace according to Rust language definition.
/// See [Rust language reference](https://doc.rust-lang.org/reference/whitespace.html)
/// for definitions of these classes.
pub fn is_whitespace(c: char) -> bool {
    // This is Pattern_White_Space.
    //
    // Note that this set is stable (ie, it doesn't change with different
    // Unicode versions), so it's ok to just hard-code the values.

    matches!(
        c,
        // Usual ASCII suspects
        '\u{0009}'   // \t
        | '\u{000A}' // \n
        | '\u{000B}' // vertical tab
        | '\u{000C}' // form feed
        | '\u{000D}' // \r
        | '\u{0020}' // space

        // NEXT LINE from latin1
        | '\u{0085}'

        // Bidi markers
        | '\u{200E}' // LEFT-TO-RIGHT MARK
        | '\u{200F}' // RIGHT-TO-LEFT MARK

        // Dedicated whitespace characters from Unicode
        | '\u{2028}' // LINE SEPARATOR
        | '\u{2029}' // PARAGRAPH SEPARATOR
    )
}

/// True if `c` is valid as a first character of an identifier.
/// See [Rust language reference](https://doc.rust-lang.org/reference/identifiers.html) for
/// a formal definition of valid identifier name.
pub fn is_id_start(c: char) -> bool {
    // This is XID_Start OR '_' (which formally is not a XID_Start).
    // We also add fast-path for ascii idents
    ('a'..='z').contains(&c)
        || ('A'..='Z').contains(&c)
        || c == '_'
        || (c > '\x7f' && unicode_xid::UnicodeXID::is_xid_start(c))
}

/// True if `c` is valid as a non-first character of an identifier.
/// See [Rust language reference](https://doc.rust-lang.org/reference/identifiers.html) for
/// a formal definition of valid identifier name.
pub fn is_id_continue(c: char) -> bool {
    // This is exactly XID_Continue.
    // We also add fast-path for ascii idents
    ('a'..='z').contains(&c)
        || ('A'..='Z').contains(&c)
        || ('0'..='9').contains(&c)
        || c == '_'
        || (c > '\x7f' && unicode_xid::UnicodeXID::is_xid_continue(c))
}

/// The passed string is lexically an identifier.
pub fn is_ident(string: &str) -> bool {
    let mut chars = string.chars();
    if let Some(start) = chars.next() {
        is_id_start(start) && chars.all(is_id_continue)
    } else {
        false
    }
}

/// Parses a token from the input string.
fn advance_token(cursor: &mut Cursor) -> Token {
    let first_char = cursor.bump().unwrap();
    let token_kind = match first_char {
        // Slash, comment or block comment.
        '/' => match cursor.peek() {
            '/' => line_comment(cursor),
            '*' => block_comment(cursor),
            _ => Slash,
        },

        // Whitespace sequence.
        c if is_whitespace(c) => whitespace(cursor),

        // Raw identifier, raw string literal or identifier.
        'r' => match (cursor.peek(), cursor.peek_second()) {
            ('#', c1) if is_id_start(c1) => raw_ident(cursor),
            ('#', _) | ('"', _) => {
                let (n_hashes, err) = raw_double_quoted_string(cursor, 1);
                let suffix_start = cursor.len_consumed();
                if err.is_none() {
                    eat_literal_suffix(cursor);
                }
                let kind = LiteralKind::RawStr { n_hashes, err };
                Literal { kind, suffix_start }
            }
            _ => ident(cursor),
        },

        // Byte literal, byte string literal, raw byte string literal or identifier.
        'b' => match (cursor.peek(), cursor.peek_second()) {
            ('\'', _) => {
                cursor.bump();
                let terminated = single_quoted_string(cursor);
                let suffix_start = cursor.len_consumed();
                if terminated {
                    eat_literal_suffix(cursor);
                }
                let kind = LiteralKind::Byte { terminated };
                Literal { kind, suffix_start }
            }
            ('"', _) => {
                cursor.bump();
                let terminated = double_quoted_string(cursor);
                let suffix_start = cursor.len_consumed();
                if terminated {
                    eat_literal_suffix(cursor);
                }
                let kind = LiteralKind::ByteStr { terminated };
                Literal { kind, suffix_start }
            }
            ('r', '"') | ('r', '#') => {
                cursor.bump();
                let (n_hashes, err) = raw_double_quoted_string(cursor, 2);
                let suffix_start = cursor.len_consumed();
                if err.is_none() {
                    eat_literal_suffix(cursor);
                }
                let kind = LiteralKind::RawByteStr { n_hashes, err };
                Literal { kind, suffix_start }
            }
            _ => ident(cursor),
        },

        // Identifier (this should be checked after other variant that can
        // start as identifier).
        c if is_id_start(c) => ident(cursor),

        // Numeric literal.
        c @ '0'..='9' => {
            let literal_kind = number(cursor, c);
            let suffix_start = cursor.len_consumed();
            eat_literal_suffix(cursor);
            TokenKind::Literal { kind: literal_kind, suffix_start }
        }

        // One-symbol tokens.
        ';' => Semi,
        ',' => Comma,
        '.' => Dot,
        '(' => OpenParen,
        ')' => CloseParen,
        '{' => OpenBrace,
        '}' => CloseBrace,
        '[' => OpenBracket,
        ']' => CloseBracket,
        '@' => At,
        '#' => Pound,
        '~' => Tilde,
        '?' => Question,
        ':' => Colon,
        '$' => Dollar,
        '=' => Eq,
        '!' => Bang,
        '<' => Lt,
        '>' => Gt,
        '-' => Minus,
        '&' => And,
        '|' => Or,
        '+' => Plus,
        '*' => Star,
        '^' => Caret,
        '%' => Percent,

        // Lifetime or character literal.
        '\'' => lifetime_or_char(cursor),

        // String literal.
        '"' => {
            let terminated = double_quoted_string(cursor);
            let suffix_start = cursor.len_consumed();
            if terminated {
                eat_literal_suffix(cursor);
            }
            let kind = LiteralKind::Str { terminated };
            Literal { kind, suffix_start }
        }
        _ => Unknown,
    };
    Token::new(token_kind, cursor.len_consumed())
}

fn line_comment(cursor: &mut Cursor) -> TokenKind {
    debug_assert!(cursor.prev() == '/' && cursor.peek() == '/');
    cursor.bump();

    let doc_style = match cursor.peek() {
        // `//!` is an inner line doc comment.
        '!' => Some(DocStyle::Inner),
        // `////` (more than 3 slashes) is not considered a doc comment.
        '/' if cursor.peek_second() != '/' => Some(DocStyle::Outer),
        _ => None,
    };

    cursor.bump_while(|c| c != '\n');
    LineComment { doc_style }
}

fn block_comment(cursor: &mut Cursor) -> TokenKind {
    debug_assert!(cursor.prev() == '/' && cursor.peek() == '*');
    cursor.bump();

    let doc_style = match cursor.peek() {
        // `/*!` is an inner block doc comment.
        '!' => Some(DocStyle::Inner),
        // `/***` (more than 2 stars) is not considered a doc comment.
        // `/**/` is not considered a doc comment.
        '*' if !matches!(cursor.peek_second(), '*' | '/') => Some(DocStyle::Outer),
        _ => None,
    };

    let mut depth = 1usize;
    while let Some(c) = cursor.bump() {
        match c {
            '/' if cursor.peek() == '*' => {
                cursor.bump();
                depth += 1;
            }
            '*' if cursor.peek() == '/' => {
                cursor.bump();
                depth -= 1;
                if depth == 0 {
                    // This block comment is closed, so for a construction like "/* */ */"
                    // there will be a successfully parsed block comment "/* */"
                    // and " */" will be processed separately.
                    break;
                }
            }
            _ => (),
        }
    }

    BlockComment { doc_style, terminated: depth == 0 }
}

fn whitespace(cursor: &mut Cursor) -> TokenKind {
    debug_assert!(is_whitespace(cursor.prev()));
    cursor.bump_while(is_whitespace);
    Whitespace
}

fn raw_ident(cursor: &mut Cursor) -> TokenKind {
    debug_assert!(cursor.prev() == 'r' && cursor.peek() == '#' && is_id_start(cursor.peek_second()));
    // Eat "#" symbol.
    cursor.bump();
    // Eat the identifier part of RawIdent.
    eat_identifier(cursor);
    RawIdent
}

fn ident(cursor: &mut Cursor) -> TokenKind {
    debug_assert!(is_id_start(cursor.prev()));
    // Start is already eaten, eat the rest of identifier.
    cursor.bump_while(is_id_continue);
    Ident
}

/// Eats one identifier.
pub(crate) fn eat_identifier(cursor: &mut Cursor) {
    if !is_id_start(cursor.peek()) {
        return;
    }
    cursor.bump();

    cursor.bump_while(is_id_continue);
}
