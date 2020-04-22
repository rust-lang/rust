//! Low-level Rust lexer.
//!
//! Tokens produced by this lexer are not yet ready for parsing the Rust syntax,
//! for that see `librustc_parse::lexer`, which converts this basic token stream
//! into wide tokens used by actual parser.
//!
//! The purpose of this crate is to convert raw sources into a labeled sequence
//! of well-known token types, so building an actual Rust token stream will
//! be easier.
//!
//! Main entity of this crate is [`TokenKind`] enum which represents common
//! lexeme types.

// We want to be able to build this crate with a stable compiler, so no
// `#![feature]` attributes should be added.

mod cursor;
pub mod unescape;

#[cfg(test)]
mod tests;

use self::LiteralKind::*;
use self::TokenKind::*;
use crate::cursor::{Cursor, EOF_CHAR};
use std::convert::TryInto;

/// Parsed token.
/// It doesn't contain information about data that has been parsed,
/// only the type of the token and its size.
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
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum TokenKind {
    // Multi-char tokens:
    /// "// comment"
    LineComment,
    /// "/* block comment */"
    /// Block comments can be recursive, so the sequence like "/* /* */"
    /// will not be considered terminated and will result in a parsing error.
    BlockComment { terminated: bool },
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
    Not,
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
pub enum LiteralKind {
    /// "12_u8", "0o100", "0b120i99"
    Int { base: Base, empty_int: bool },
    /// "12.34f32", "0b100.100"
    Float { base: Base, empty_exponent: bool },
    /// "'a'", "'\\'", "'''", "';"
    Char { terminated: bool },
    /// "b'a'", "b'\\'", "b'''", "b';"
    Byte { terminated: bool },
    /// ""abc"", ""abc"
    Str { terminated: bool },
    /// "b"abc"", "b"abc"
    ByteStr { terminated: bool },
    /// "r"abc"", "r#"abc"#", "r####"ab"###"c"####", "r#"a"
    RawStr(UnvalidatedRawStr),
    /// "br"abc"", "br#"abc"#", "br####"ab"###"c"####", "br#"a"
    RawByteStr(UnvalidatedRawStr),
}

/// Represents something that looks like a raw string, but may have some
/// problems. Use `.validate()` to convert it into something
/// usable.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct UnvalidatedRawStr {
    /// The prefix (`r###"`) is valid
    valid_start: bool,

    /// The postfix (`"###`) is valid
    valid_end: bool,

    /// The number of leading `#`
    n_start_hashes: usize,
    /// The number of trailing `#`. `n_end_hashes` <= `n_start_hashes`
    n_end_hashes: usize,
    /// The offset starting at `r` or `br` where the user may have intended to end the string.
    /// Currently, it is the longest sequence of pattern `"#+"`.
    possible_terminator_offset: Option<usize>,
}

/// Error produced validating a raw string. Represents cases like:
/// - `r##~"abcde"##`: `LexRawStrError::InvalidStarter`
/// - `r###"abcde"##`: `LexRawStrError::NoTerminator { expected: 3, found: 2, possible_terminator_offset: Some(11)`
/// - Too many `#`s (>65536): `TooManyDelimiters`
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum LexRawStrError {
    /// Non `#` characters exist between `r` and `"` eg. `r#~"..`
    InvalidStarter,
    /// The string was never terminated. `possible_terminator_offset` is the number of characters after `r` or `br` where they
    /// may have intended to terminate it.
    NoTerminator { expected: usize, found: usize, possible_terminator_offset: Option<usize> },
    /// More than 65536 `#`s exist.
    TooManyDelimiters,
}

/// Raw String that contains a valid prefix (`#+"`) and postfix (`"#+`) where
/// there are a matching number of `#` characters in both. Note that this will
/// not consume extra trailing `#` characters: `r###"abcde"####` is lexed as a
/// `ValidatedRawString { n_hashes: 3 }` followed by a `#` token.
#[derive(Debug, Eq, PartialEq, Copy, Clone)]
pub struct ValidatedRawStr {
    n_hashes: u16,
}

impl ValidatedRawStr {
    pub fn num_hashes(&self) -> u16 {
        self.n_hashes
    }
}

impl UnvalidatedRawStr {
    pub fn validate(self) -> Result<ValidatedRawStr, LexRawStrError> {
        if !self.valid_start {
            return Err(LexRawStrError::InvalidStarter);
        }

        // Only up to 65535 `#`s are allowed in raw strings
        let n_start_safe: u16 =
            self.n_start_hashes.try_into().map_err(|_| LexRawStrError::TooManyDelimiters)?;

        if self.n_start_hashes > self.n_end_hashes || !self.valid_end {
            Err(LexRawStrError::NoTerminator {
                expected: self.n_start_hashes,
                found: self.n_end_hashes,
                possible_terminator_offset: self.possible_terminator_offset,
            })
        } else {
            // Since the lexer should never produce a literal with n_end > n_start, if n_start <= n_end,
            // they must be equal.
            debug_assert_eq!(self.n_start_hashes, self.n_end_hashes);
            Ok(ValidatedRawStr { n_hashes: n_start_safe })
        }
    }
}

/// Base of numeric literal encoding according to its prefix.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Base {
    /// Literal starts with "0b".
    Binary,
    /// Literal starts with "0o".
    Octal,
    /// Literal starts with "0x".
    Hexadecimal,
    /// Literal doesn't contain a prefix.
    Decimal,
}

/// `rustc` allows files to have a shebang, e.g. "#!/usr/bin/rustrun",
/// but shebang isn't a part of rust syntax, so this function
/// skips the line if it starts with a shebang ("#!").
/// Line won't be skipped if it represents a valid Rust syntax
/// (e.g. "#![deny(missing_docs)]").
pub fn strip_shebang(input: &str) -> Option<usize> {
    debug_assert!(!input.is_empty());
    let s: &str = &remove_whitespace(input);
    if !s.starts_with("#!") || s.starts_with("#![") {
        return None;
    }
    Some(input.find('\n').unwrap_or(input.len()))
}

fn remove_whitespace(s: &str) -> String {
    s.chars().filter(|c| !c.is_whitespace()).collect()
}

/// Parses the first token from the provided input string.
pub fn first_token(input: &str) -> Token {
    debug_assert!(!input.is_empty());
    Cursor::new(input).advance_token()
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

    match c {
        // Usual ASCII suspects
        | '\u{0009}' // \t
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
        => true,
        _ => false,
    }
}

/// True if `c` is valid as a first character of an identifier.
/// See [Rust language reference](https://doc.rust-lang.org/reference/identifiers.html) for
/// a formal definition of valid identifier name.
pub fn is_id_start(c: char) -> bool {
    // This is XID_Start OR '_' (which formally is not a XID_Start).
    // We also add fast-path for ascii idents
    ('a' <= c && c <= 'z')
        || ('A' <= c && c <= 'Z')
        || c == '_'
        || (c > '\x7f' && unicode_xid::UnicodeXID::is_xid_start(c))
}

/// True if `c` is valid as a non-first character of an identifier.
/// See [Rust language reference](https://doc.rust-lang.org/reference/identifiers.html) for
/// a formal definition of valid identifier name.
pub fn is_id_continue(c: char) -> bool {
    // This is exactly XID_Continue.
    // We also add fast-path for ascii idents
    ('a' <= c && c <= 'z')
        || ('A' <= c && c <= 'Z')
        || ('0' <= c && c <= '9')
        || c == '_'
        || (c > '\x7f' && unicode_xid::UnicodeXID::is_xid_continue(c))
}

impl Cursor<'_> {
    /// Parses a token from the input string.
    fn advance_token(&mut self) -> Token {
        let first_char = self.bump().unwrap();
        let token_kind = match first_char {
            // Slash, comment or block comment.
            '/' => match self.first() {
                '/' => self.line_comment(),
                '*' => self.block_comment(),
                _ => Slash,
            },

            // Whitespace sequence.
            c if is_whitespace(c) => self.whitespace(),

            // Raw identifier, raw string literal or identifier.
            'r' => match (self.first(), self.second()) {
                ('#', c1) if is_id_start(c1) => self.raw_ident(),
                ('#', _) | ('"', _) => {
                    let raw_str_i = self.raw_double_quoted_string(1);
                    let suffix_start = self.len_consumed();
                    if raw_str_i.n_end_hashes == raw_str_i.n_start_hashes {
                        self.eat_literal_suffix();
                    }
                    let kind = RawStr(raw_str_i);
                    Literal { kind, suffix_start }
                }
                _ => self.ident(),
            },

            // Byte literal, byte string literal, raw byte string literal or identifier.
            'b' => match (self.first(), self.second()) {
                ('\'', _) => {
                    self.bump();
                    let terminated = self.single_quoted_string();
                    let suffix_start = self.len_consumed();
                    if terminated {
                        self.eat_literal_suffix();
                    }
                    let kind = Byte { terminated };
                    Literal { kind, suffix_start }
                }
                ('"', _) => {
                    self.bump();
                    let terminated = self.double_quoted_string();
                    let suffix_start = self.len_consumed();
                    if terminated {
                        self.eat_literal_suffix();
                    }
                    let kind = ByteStr { terminated };
                    Literal { kind, suffix_start }
                }
                ('r', '"') | ('r', '#') => {
                    self.bump();
                    let raw_str_i = self.raw_double_quoted_string(2);
                    let suffix_start = self.len_consumed();
                    let terminated = raw_str_i.n_start_hashes == raw_str_i.n_end_hashes;
                    if terminated {
                        self.eat_literal_suffix();
                    }

                    let kind = RawByteStr(raw_str_i);
                    Literal { kind, suffix_start }
                }
                _ => self.ident(),
            },

            // Identifier (this should be checked after other variant that can
            // start as identifier).
            c if is_id_start(c) => self.ident(),

            // Numeric literal.
            c @ '0'..='9' => {
                let literal_kind = self.number(c);
                let suffix_start = self.len_consumed();
                self.eat_literal_suffix();
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
            '!' => Not,
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
            '\'' => self.lifetime_or_char(),

            // String literal.
            '"' => {
                let terminated = self.double_quoted_string();
                let suffix_start = self.len_consumed();
                if terminated {
                    self.eat_literal_suffix();
                }
                let kind = Str { terminated };
                Literal { kind, suffix_start }
            }
            _ => Unknown,
        };
        Token::new(token_kind, self.len_consumed())
    }

    fn line_comment(&mut self) -> TokenKind {
        debug_assert!(self.prev() == '/' && self.first() == '/');
        self.bump();
        self.eat_while(|c| c != '\n');
        LineComment
    }

    fn block_comment(&mut self) -> TokenKind {
        debug_assert!(self.prev() == '/' && self.first() == '*');
        self.bump();
        let mut depth = 1usize;
        while let Some(c) = self.bump() {
            match c {
                '/' if self.first() == '*' => {
                    self.bump();
                    depth += 1;
                }
                '*' if self.first() == '/' => {
                    self.bump();
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

        BlockComment { terminated: depth == 0 }
    }

    fn whitespace(&mut self) -> TokenKind {
        debug_assert!(is_whitespace(self.prev()));
        self.eat_while(is_whitespace);
        Whitespace
    }

    fn raw_ident(&mut self) -> TokenKind {
        debug_assert!(self.prev() == 'r' && self.first() == '#' && is_id_start(self.second()));
        // Eat "#" symbol.
        self.bump();
        // Eat the identifier part of RawIdent.
        self.eat_identifier();
        RawIdent
    }

    fn ident(&mut self) -> TokenKind {
        debug_assert!(is_id_start(self.prev()));
        // Start is already eaten, eat the rest of identifier.
        self.eat_while(is_id_continue);
        Ident
    }

    fn number(&mut self, first_digit: char) -> LiteralKind {
        debug_assert!('0' <= self.prev() && self.prev() <= '9');
        let mut base = Base::Decimal;
        if first_digit == '0' {
            // Attempt to parse encoding base.
            let has_digits = match self.first() {
                'b' => {
                    base = Base::Binary;
                    self.bump();
                    self.eat_decimal_digits()
                }
                'o' => {
                    base = Base::Octal;
                    self.bump();
                    self.eat_decimal_digits()
                }
                'x' => {
                    base = Base::Hexadecimal;
                    self.bump();
                    self.eat_hexadecimal_digits()
                }
                // Not a base prefix.
                '0'..='9' | '_' | '.' | 'e' | 'E' => {
                    self.eat_decimal_digits();
                    true
                }
                // Just a 0.
                _ => return Int { base, empty_int: false },
            };
            // Base prefix was provided, but there were no digits
            // after it, e.g. "0x".
            if !has_digits {
                return Int { base, empty_int: true };
            }
        } else {
            // No base prefix, parse number in the usual way.
            self.eat_decimal_digits();
        };

        match self.first() {
            // Don't be greedy if this is actually an
            // integer literal followed by field/method access or a range pattern
            // (`0..2` and `12.foo()`)
            '.' if self.second() != '.' && !is_id_start(self.second()) => {
                // might have stuff after the ., and if it does, it needs to start
                // with a number
                self.bump();
                let mut empty_exponent = false;
                if self.first().is_digit(10) {
                    self.eat_decimal_digits();
                    match self.first() {
                        'e' | 'E' => {
                            self.bump();
                            empty_exponent = !self.eat_float_exponent();
                        }
                        _ => (),
                    }
                }
                Float { base, empty_exponent }
            }
            'e' | 'E' => {
                self.bump();
                let empty_exponent = !self.eat_float_exponent();
                Float { base, empty_exponent }
            }
            _ => Int { base, empty_int: false },
        }
    }

    fn lifetime_or_char(&mut self) -> TokenKind {
        debug_assert!(self.prev() == '\'');

        let can_be_a_lifetime = if self.second() == '\'' {
            // It's surely not a lifetime.
            false
        } else {
            // If the first symbol is valid for identifier, it can be a lifetime.
            // Also check if it's a number for a better error reporting (so '0 will
            // be reported as invalid lifetime and not as unterminated char literal).
            is_id_start(self.first()) || self.first().is_digit(10)
        };

        if !can_be_a_lifetime {
            let terminated = self.single_quoted_string();
            let suffix_start = self.len_consumed();
            if terminated {
                self.eat_literal_suffix();
            }
            let kind = Char { terminated };
            return Literal { kind, suffix_start };
        }

        // Either a lifetime or a character literal with
        // length greater than 1.

        let starts_with_number = self.first().is_digit(10);

        // Skip the literal contents.
        // First symbol can be a number (which isn't a valid identifier start),
        // so skip it without any checks.
        self.bump();
        self.eat_while(is_id_continue);

        // Check if after skipping literal contents we've met a closing
        // single quote (which means that user attempted to create a
        // string with single quotes).
        if self.first() == '\'' {
            self.bump();
            let kind = Char { terminated: true };
            Literal { kind, suffix_start: self.len_consumed() }
        } else {
            Lifetime { starts_with_number }
        }
    }

    fn single_quoted_string(&mut self) -> bool {
        debug_assert!(self.prev() == '\'');
        // Check if it's a one-symbol literal.
        if self.second() == '\'' && self.first() != '\\' {
            self.bump();
            self.bump();
            return true;
        }

        // Literal has more than one symbol.

        // Parse until either quotes are terminated or error is detected.
        loop {
            match self.first() {
                // Quotes are terminated, finish parsing.
                '\'' => {
                    self.bump();
                    return true;
                }
                // Probably beginning of the comment, which we don't want to include
                // to the error report.
                '/' => break,
                // Newline without following '\'' means unclosed quote, stop parsing.
                '\n' if self.second() != '\'' => break,
                // End of file, stop parsing.
                EOF_CHAR if self.is_eof() => break,
                // Escaped slash is considered one character, so bump twice.
                '\\' => {
                    self.bump();
                    self.bump();
                }
                // Skip the character.
                _ => {
                    self.bump();
                }
            }
        }
        // String was not terminated.
        false
    }

    /// Eats double-quoted string and returns true
    /// if string is terminated.
    fn double_quoted_string(&mut self) -> bool {
        debug_assert!(self.prev() == '"');
        while let Some(c) = self.bump() {
            match c {
                '"' => {
                    return true;
                }
                '\\' if self.first() == '\\' || self.first() == '"' => {
                    // Bump again to skip escaped character.
                    self.bump();
                }
                _ => (),
            }
        }
        // End of file reached.
        false
    }

    /// Eats the double-quoted string and returns an `UnvalidatedRawStr`.
    fn raw_double_quoted_string(&mut self, prefix_len: usize) -> UnvalidatedRawStr {
        debug_assert!(self.prev() == 'r');
        let mut valid_start: bool = false;
        let start_pos = self.len_consumed();
        let (mut possible_terminator_offset, mut max_hashes) = (None, 0);

        // Count opening '#' symbols.
        let n_start_hashes = self.eat_while(|c| c == '#');

        // Check that string is started.
        match self.bump() {
            Some('"') => valid_start = true,
            _ => {
                return UnvalidatedRawStr {
                    valid_start,
                    valid_end: false,
                    n_start_hashes,
                    n_end_hashes: 0,
                    possible_terminator_offset,
                };
            }
        }

        // Skip the string contents and on each '#' character met, check if this is
        // a raw string termination.
        loop {
            self.eat_while(|c| c != '"');

            if self.is_eof() {
                return UnvalidatedRawStr {
                    valid_start,
                    valid_end: false,
                    n_start_hashes,
                    n_end_hashes: max_hashes,
                    possible_terminator_offset,
                };
            }

            // Eat closing double quote.
            self.bump();

            // Check that amount of closing '#' symbols
            // is equal to the amount of opening ones.
            let mut hashes_left = n_start_hashes;
            let is_closing_hash = |c| {
                if c == '#' && hashes_left != 0 {
                    hashes_left -= 1;
                    true
                } else {
                    false
                }
            };
            let n_end_hashes = self.eat_while(is_closing_hash);

            if n_end_hashes == n_start_hashes {
                return UnvalidatedRawStr {
                    valid_start,
                    valid_end: true,
                    n_start_hashes,
                    n_end_hashes,
                    possible_terminator_offset: None,
                };
            } else if n_end_hashes > max_hashes {
                // Keep track of possible terminators to give a hint about where there might be
                // a missing terminator
                possible_terminator_offset =
                    Some(self.len_consumed() - start_pos - n_end_hashes + prefix_len);
                max_hashes = n_end_hashes;
            }
        }
    }

    fn eat_decimal_digits(&mut self) -> bool {
        let mut has_digits = false;
        loop {
            match self.first() {
                '_' => {
                    self.bump();
                }
                '0'..='9' => {
                    has_digits = true;
                    self.bump();
                }
                _ => break,
            }
        }
        has_digits
    }

    fn eat_hexadecimal_digits(&mut self) -> bool {
        let mut has_digits = false;
        loop {
            match self.first() {
                '_' => {
                    self.bump();
                }
                '0'..='9' | 'a'..='f' | 'A'..='F' => {
                    has_digits = true;
                    self.bump();
                }
                _ => break,
            }
        }
        has_digits
    }

    /// Eats the float exponent. Returns true if at least one digit was met,
    /// and returns false otherwise.
    fn eat_float_exponent(&mut self) -> bool {
        debug_assert!(self.prev() == 'e' || self.prev() == 'E');
        if self.first() == '-' || self.first() == '+' {
            self.bump();
        }
        self.eat_decimal_digits()
    }

    // Eats the suffix of the literal, e.g. "_u8".
    fn eat_literal_suffix(&mut self) {
        self.eat_identifier();
    }

    // Eats the identifier.
    fn eat_identifier(&mut self) {
        if !is_id_start(self.first()) {
            return;
        }
        self.bump();

        self.eat_while(is_id_continue);
    }

    /// Eats symbols while predicate returns true or until the end of file is reached.
    /// Returns amount of eaten symbols.
    fn eat_while<F>(&mut self, mut predicate: F) -> usize
    where
        F: FnMut(char) -> bool,
    {
        let mut eaten: usize = 0;
        while predicate(self.first()) && !self.is_eof() {
            eaten += 1;
            self.bump();
        }

        eaten
    }
}
