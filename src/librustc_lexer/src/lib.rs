//! Low-level Rust lexer.
//!
//! Tokens produced by this lexer are not yet ready for parsing the Rust syntax,
//! for that see `libsyntax::parse::lexer`, which converts this basic token stream
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

use crate::cursor::{Cursor, EOF_CHAR};

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

/// Enum represening common lexeme types.
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
use self::TokenKind::*;

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
    RawStr { n_hashes: usize, started: bool, terminated: bool },
    /// "br"abc"", "br#"abc"#", "br####"ab"###"c"####", "br#"a"
    RawByteStr { n_hashes: usize, started: bool, terminated: bool },
}
use self::LiteralKind::*;

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
    if !input.starts_with("#!") || input.starts_with("#![") {
        return None;
    }
    Some(input.find('\n').unwrap_or(input.len()))
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
            '/' => match self.nth_char(0) {
                '/' => self.line_comment(),
                '*' => self.block_comment(),
                _ => Slash,
            },

            // Whitespace sequence.
            c if is_whitespace(c) => self.whitespace(),

            // Raw string literal or identifier.
            'r' => match (self.nth_char(0), self.nth_char(1)) {
                ('#', c1) if is_id_start(c1) => self.raw_ident(),
                ('#', _) | ('"', _) => {
                    let (n_hashes, started, terminated) = self.raw_double_quoted_string();
                    let suffix_start = self.len_consumed();
                    if terminated {
                        self.eat_literal_suffix();
                    }
                    let kind = RawStr { n_hashes, started, terminated };
                    Literal { kind, suffix_start }
                }
                _ => self.ident(),
            },

            // Byte literal, byte string literal, raw byte string literal or identifier.
            'b' => match (self.nth_char(0), self.nth_char(1)) {
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
                    let (n_hashes, started, terminated) = self.raw_double_quoted_string();
                    let suffix_start = self.len_consumed();
                    if terminated {
                        self.eat_literal_suffix();
                    }
                    let kind = RawByteStr { n_hashes, started, terminated };
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
        debug_assert!(self.prev() == '/' && self.nth_char(0) == '/');
        self.bump();
        loop {
            match self.nth_char(0) {
                '\n' => break,
                EOF_CHAR if self.is_eof() => break,
                _ => {
                    self.bump();
                }
            }
        }
        LineComment
    }

    fn block_comment(&mut self) -> TokenKind {
        debug_assert!(self.prev() == '/' && self.nth_char(0) == '*');
        self.bump();
        let mut depth = 1usize;
        while let Some(c) = self.bump() {
            match c {
                '/' if self.nth_char(0) == '*' => {
                    self.bump();
                    depth += 1;
                }
                '*' if self.nth_char(0) == '/' => {
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
        while is_whitespace(self.nth_char(0)) {
            self.bump();
        }
        Whitespace
    }

    fn raw_ident(&mut self) -> TokenKind {
        debug_assert!(
            self.prev() == 'r'
                && self.nth_char(0) == '#'
                && is_id_start(self.nth_char(1))
        );
        self.bump();
        self.bump();
        while is_id_continue(self.nth_char(0)) {
            self.bump();
        }
        RawIdent
    }

    fn ident(&mut self) -> TokenKind {
        debug_assert!(is_id_start(self.prev()));
        while is_id_continue(self.nth_char(0)) {
            self.bump();
        }
        Ident
    }

    fn number(&mut self, first_digit: char) -> LiteralKind {
        debug_assert!('0' <= self.prev() && self.prev() <= '9');
        let mut base = Base::Decimal;
        if first_digit == '0' {
            // Attempt to parse encoding base.
            let has_digits = match self.nth_char(0) {
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

        match self.nth_char(0) {
            // Don't be greedy if this is actually an
            // integer literal followed by field/method access or a range pattern
            // (`0..2` and `12.foo()`)
            '.' if self.nth_char(1) != '.'
                && !is_id_start(self.nth_char(1)) =>
            {
                // might have stuff after the ., and if it does, it needs to start
                // with a number
                self.bump();
                let mut empty_exponent = false;
                if self.nth_char(0).is_digit(10) {
                    self.eat_decimal_digits();
                    match self.nth_char(0) {
                        'e' | 'E' => {
                            self.bump();
                            empty_exponent = self.float_exponent().is_err()
                        }
                        _ => (),
                    }
                }
                Float { base, empty_exponent }
            }
            'e' | 'E' => {
                self.bump();
                let empty_exponent = self.float_exponent().is_err();
                Float { base, empty_exponent }
            }
            _ => Int { base, empty_int: false },
        }
    }

    fn lifetime_or_char(&mut self) -> TokenKind {
        debug_assert!(self.prev() == '\'');
        let mut starts_with_number = false;

        // Check if the first symbol after '\'' is a valid identifier
        // character or a number (not a digit followed by '\'').
        if (is_id_start(self.nth_char(0))
            || self.nth_char(0).is_digit(10) && {
                starts_with_number = true;
                true
            })
            && self.nth_char(1) != '\''
        {
            self.bump();

            // Skip the identifier.
            while is_id_continue(self.nth_char(0)) {
                self.bump();
            }

            return if self.nth_char(0) == '\'' {
                self.bump();
                let kind = Char { terminated: true };
                Literal { kind, suffix_start: self.len_consumed() }
            } else {
                Lifetime { starts_with_number }
            };
        }

        // This is not a lifetime (checked above), parse a char literal.
        let terminated = self.single_quoted_string();
        let suffix_start = self.len_consumed();
        if terminated {
            self.eat_literal_suffix();
        }
        let kind = Char { terminated };
        return Literal { kind, suffix_start };
    }

    fn single_quoted_string(&mut self) -> bool {
        debug_assert!(self.prev() == '\'');
        // Parse `'''` as a single char literal.
        if self.nth_char(0) == '\'' && self.nth_char(1) == '\'' {
            self.bump();
        }
        // Parse until either quotes are terminated or error is detected.
        let mut first = true;
        loop {
            match self.nth_char(0) {
                // Probably beginning of the comment, which we don't want to include
                // to the error report.
                '/' if !first => break,
                // Newline without following '\'' means unclosed quote, stop parsing.
                '\n' if self.nth_char(1) != '\'' => break,
                // End of file, stop parsing.
                EOF_CHAR if self.is_eof() => break,
                // Quotes are terminated, finish parsing.
                '\'' => {
                    self.bump();
                    return true;
                }
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
            first = false;
        }
        false
    }

    /// Eats double-quoted string and returns true
    /// if string is terminated.
    fn double_quoted_string(&mut self) -> bool {
        debug_assert!(self.prev() == '"');
        loop {
            match self.nth_char(0) {
                '"' => {
                    self.bump();
                    return true;
                }
                EOF_CHAR if self.is_eof() => return false,
                '\\' if self.nth_char(1) == '\\' || self.nth_char(1) == '"' => {
                    self.bump();
                }
                _ => (),
            }
            self.bump();
        }
    }

    /// Eats the double-quoted string and returns a tuple of
    /// (amount of the '#' symbols, raw string started, raw string terminated)
    fn raw_double_quoted_string(&mut self) -> (usize, bool, bool) {
        debug_assert!(self.prev() == 'r');
        // Count opening '#' symbols.
        let n_hashes = {
            let mut acc: usize = 0;
            loop {
                match self.bump() {
                    Some('#') => acc += 1,
                    Some('"') => break acc,
                    None | Some(_) => return (acc, false, false),
                }
            }
        };

        // Skip the string itself and check that amount of closing '#'
        // symbols is equal to the amount of opening ones.
        loop {
            match self.bump() {
                Some('"') => {
                    let mut acc = n_hashes;
                    while self.nth_char(0) == '#' && acc > 0 {
                        self.bump();
                        acc -= 1;
                    }
                    if acc == 0 {
                        return (n_hashes, true, true);
                    }
                }
                Some(_) => (),
                None => return (n_hashes, true, false),
            }
        }
    }

    fn eat_decimal_digits(&mut self) -> bool {
        let mut has_digits = false;
        loop {
            match self.nth_char(0) {
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
            match self.nth_char(0) {
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

    fn float_exponent(&mut self) -> Result<(), ()> {
        debug_assert!(self.prev() == 'e' || self.prev() == 'E');
        if self.nth_char(0) == '-' || self.nth_char(0) == '+' {
            self.bump();
        }
        if self.eat_decimal_digits() { Ok(()) } else { Err(()) }
    }

    // Eats the suffix if it's an identifier.
    fn eat_literal_suffix(&mut self) {
        if !is_id_start(self.nth_char(0)) {
            return;
        }
        self.bump();

        while is_id_continue(self.nth_char(0)) {
            self.bump();
        }
    }
}
