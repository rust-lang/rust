// We want to be able to build this crate with a stable compiler, so no
// `#![feature]` attributes should be added.

mod cursor;
pub mod unescape;

use crate::cursor::{Cursor, EOF_CHAR};

pub struct Token {
    pub kind: TokenKind,
    pub len: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum TokenKind {
    LineComment,
    BlockComment { terminated: bool },
    Whitespace,
    Ident,
    RawIdent,
    Literal { kind: LiteralKind, suffix_start: usize },
    Lifetime { starts_with_number: bool },
    Semi,
    Comma,
    Dot,
    OpenParen,
    CloseParen,
    OpenBrace,
    CloseBrace,
    OpenBracket,
    CloseBracket,
    At,
    Pound,
    Tilde,
    Question,
    Colon,
    Dollar,
    Eq,
    Not,
    Lt,
    Gt,
    Minus,
    And,
    Or,
    Plus,
    Star,
    Slash,
    Caret,
    Percent,
    Unknown,
}
use self::TokenKind::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum LiteralKind {
    Int { base: Base, empty_int: bool },
    Float { base: Base, empty_exponent: bool },
    Char { terminated: bool },
    Byte { terminated: bool },
    Str { terminated: bool },
    ByteStr { terminated: bool },
    RawStr { n_hashes: usize, started: bool, terminated: bool },
    RawByteStr { n_hashes: usize, started: bool, terminated: bool },
}
use self::LiteralKind::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Base {
    Binary,
    Octal,
    Hexadecimal,
    Decimal,
}

impl Token {
    fn new(kind: TokenKind, len: usize) -> Token {
        Token { kind, len }
    }
}

pub fn strip_shebang(input: &str) -> Option<usize> {
    debug_assert!(!input.is_empty());
    if !input.starts_with("#!") || input.starts_with("#![") {
        return None;
    }
    Some(input.find('\n').unwrap_or(input.len()))
}

pub fn first_token(input: &str) -> Token {
    debug_assert!(!input.is_empty());
    Cursor::new(input).advance_token()
}

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

// See [UAX #31](http://unicode.org/reports/tr31) for definitions of these
// classes.

/// True if `c` is considered a whitespace according to Rust language definition.
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
pub fn is_id_start(c: char) -> bool {
    // This is XID_Start OR '_' (which formally is not a XID_Start).
    // We also add fast-path for ascii idents
    ('a' <= c && c <= 'z')
        || ('A' <= c && c <= 'Z')
        || c == '_'
        || (c > '\x7f' && unicode_xid::UnicodeXID::is_xid_start(c))
}

/// True if `c` is valid as a non-first character of an identifier.
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
    fn advance_token(&mut self) -> Token {
        let first_char = self.bump().unwrap();
        let token_kind = match first_char {
            '/' => match self.nth_char(0) {
                '/' => self.line_comment(),
                '*' => self.block_comment(),
                _ => Slash,
            },
            c if is_whitespace(c) => self.whitespace(),
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
            c if is_id_start(c) => self.ident(),
            c @ '0'..='9' => {
                let literal_kind = self.number(c);
                let suffix_start = self.len_consumed();
                self.eat_literal_suffix();
                TokenKind::Literal { kind: literal_kind, suffix_start }
            }
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
            '\'' => self.lifetime_or_char(),
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
                '0'..='9' | '_' | '.' | 'e' | 'E' => {
                    self.eat_decimal_digits();
                    true
                }
                // just a 0
                _ => return Int { base, empty_int: false },
            };
            if !has_digits {
                return Int { base, empty_int: true };
            }
        } else {
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
        if (is_id_start(self.nth_char(0))
            || self.nth_char(0).is_digit(10) && {
                starts_with_number = true;
                true
            })
            && self.nth_char(1) != '\''
        {
            self.bump();
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
        // parse `'''` as a single char literal
        if self.nth_char(0) == '\'' && self.nth_char(1) == '\'' {
            self.bump();
        }
        let mut first = true;
        loop {
            match self.nth_char(0) {
                '/' if !first => break,
                '\n' if self.nth_char(1) != '\'' => break,
                EOF_CHAR if self.is_eof() => break,
                '\'' => {
                    self.bump();
                    return true;
                }
                '\\' => {
                    self.bump();
                    self.bump();
                }
                _ => {
                    self.bump();
                }
            }
            first = false;
        }
        false
    }

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

    fn raw_double_quoted_string(&mut self) -> (usize, bool, bool) {
        debug_assert!(self.prev() == 'r');
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
