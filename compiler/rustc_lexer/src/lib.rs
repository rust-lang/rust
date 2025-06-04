//! Low-level Rust lexer.
//!
//! The idea with `rustc_lexer` is to make a reusable library,
//! by separating out pure lexing and rustc-specific concerns, like spans,
//! error reporting, and interning. So, rustc_lexer operates directly on `&str`,
//! produces simple tokens which are a pair of type-tag and a bit of original text,
//! and does not report errors, instead storing them as flags on the token.
//!
//! Tokens produced by this lexer are not yet ready for parsing the Rust syntax.
//! For that see [`rustc_parse::lexer`], which converts this basic token stream
//! into wide tokens used by actual parser.
//!
//! The purpose of this crate is to convert raw sources into a labeled sequence
//! of well-known token types, so building an actual Rust token stream will
//! be easier.
//!
//! The main entity of this crate is the [`TokenKind`] enum which represents common
//! lexeme types.
//!
//! [`rustc_parse::lexer`]: ../rustc_parse/lexer/index.html

// tidy-alphabetical-start
// We want to be able to build this crate with a stable compiler,
// so no `#![feature]` attributes should be added.
#![deny(unstable_features)]
// tidy-alphabetical-end

mod cursor;

#[cfg(test)]
mod tests;

use unicode_properties::UnicodeEmoji;
pub use unicode_xid::UNICODE_VERSION as UNICODE_XID_VERSION;

use self::LiteralKind::*;
use self::TokenKind::*;
use crate::cursor::EOF_CHAR;
pub use crate::cursor::{Cursor, FrontmatterAllowed};

/// Parsed token.
/// It doesn't contain information about data that has been parsed,
/// only the type of the token and its size.
#[derive(Debug)]
pub struct Token {
    pub kind: TokenKind,
    pub len: u32,
}

impl Token {
    fn new(kind: TokenKind, len: u32) -> Token {
        Token { kind, len }
    }
}

/// Enum representing common lexeme types.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TokenKind {
    /// A line comment, e.g. `// comment`.
    LineComment {
        doc_style: Option<DocStyle>,
    },

    /// A block comment, e.g. `/* block comment */`.
    ///
    /// Block comments can be recursive, so a sequence like `/* /* */`
    /// will not be considered terminated and will result in a parsing error.
    BlockComment {
        doc_style: Option<DocStyle>,
        terminated: bool,
    },

    /// Any whitespace character sequence.
    Whitespace,

    Frontmatter {
        has_invalid_preceding_whitespace: bool,
        invalid_infostring: bool,
    },

    /// An identifier or keyword, e.g. `ident` or `continue`.
    Ident,

    /// An identifier that is invalid because it contains emoji.
    InvalidIdent,

    /// A raw identifier, e.g. "r#ident".
    RawIdent,

    /// An unknown literal prefix, like `foo#`, `foo'`, `foo"`. Excludes
    /// literal prefixes that contain emoji, which are considered "invalid".
    ///
    /// Note that only the
    /// prefix (`foo`) is included in the token, not the separator (which is
    /// lexed as its own distinct token). In Rust 2021 and later, reserved
    /// prefixes are reported as errors; in earlier editions, they result in a
    /// (allowed by default) lint, and are treated as regular identifier
    /// tokens.
    UnknownPrefix,

    /// An unknown prefix in a lifetime, like `'foo#`.
    ///
    /// Like `UnknownPrefix`, only the `'` and prefix are included in the token
    /// and not the separator.
    UnknownPrefixLifetime,

    /// A raw lifetime, e.g. `'r#foo`. In edition < 2021 it will be split into
    /// several tokens: `'r` and `#` and `foo`.
    RawLifetime,

    /// Guarded string literal prefix: `#"` or `##`.
    ///
    /// Used for reserving "guarded strings" (RFC 3598) in edition 2024.
    /// Split into the component tokens on older editions.
    GuardedStrPrefix,

    /// Literals, e.g. `12u8`, `1.0e-40`, `b"123"`. Note that `_` is an invalid
    /// suffix, but may be present here on string and float literals. Users of
    /// this type will need to check for and reject that case.
    ///
    /// See [LiteralKind] for more details.
    Literal {
        kind: LiteralKind,
        suffix_start: u32,
    },

    /// A lifetime, e.g. `'a`.
    Lifetime {
        starts_with_number: bool,
    },

    /// `;`
    Semi,
    /// `,`
    Comma,
    /// `.`
    Dot,
    /// `(`
    OpenParen,
    /// `)`
    CloseParen,
    /// `{`
    OpenBrace,
    /// `}`
    CloseBrace,
    /// `[`
    OpenBracket,
    /// `]`
    CloseBracket,
    /// `@`
    At,
    /// `#`
    Pound,
    /// `~`
    Tilde,
    /// `?`
    Question,
    /// `:`
    Colon,
    /// `$`
    Dollar,
    /// `=`
    Eq,
    /// `!`
    Bang,
    /// `<`
    Lt,
    /// `>`
    Gt,
    /// `-`
    Minus,
    /// `&`
    And,
    /// `|`
    Or,
    /// `+`
    Plus,
    /// `*`
    Star,
    /// `/`
    Slash,
    /// `^`
    Caret,
    /// `%`
    Percent,

    /// Unknown token, not expected by the lexer, e.g. "№"
    Unknown,

    /// End of input.
    Eof,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DocStyle {
    Outer,
    Inner,
}

/// Enum representing the literal types supported by the lexer.
///
/// Note that the suffix is *not* considered when deciding the `LiteralKind` in
/// this type. This means that float literals like `1f32` are classified by this
/// type as `Int`. (Compare against `rustc_ast::token::LitKind` and
/// `rustc_ast::ast::LitKind`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum LiteralKind {
    /// `12_u8`, `0o100`, `0b120i99`, `1f32`.
    Int { base: Base, empty_int: bool },
    /// `12.34f32`, `1e3`, but not `1f32`.
    Float { base: Base, empty_exponent: bool },
    /// `'a'`, `'\\'`, `'''`, `';`
    Char { terminated: bool },
    /// `b'a'`, `b'\\'`, `b'''`, `b';`
    Byte { terminated: bool },
    /// `"abc"`, `"abc`
    Str { terminated: bool },
    /// `b"abc"`, `b"abc`
    ByteStr { terminated: bool },
    /// `c"abc"`, `c"abc`
    CStr { terminated: bool },
    /// `r"abc"`, `r#"abc"#`, `r####"ab"###"c"####`, `r#"a`. `None` indicates
    /// an invalid literal.
    RawStr { n_hashes: Option<u8> },
    /// `br"abc"`, `br#"abc"#`, `br####"ab"###"c"####`, `br#"a`. `None`
    /// indicates an invalid literal.
    RawByteStr { n_hashes: Option<u8> },
    /// `cr"abc"`, "cr#"abc"#", `cr#"a`. `None` indicates an invalid literal.
    RawCStr { n_hashes: Option<u8> },
}

/// `#"abc"#`, `##"a"` (fewer closing), or even `#"a` (unterminated).
///
/// Can capture fewer closing hashes than starting hashes,
/// for more efficient lexing and better backwards diagnostics.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct GuardedStr {
    pub n_hashes: u32,
    pub terminated: bool,
    pub token_len: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum RawStrError {
    /// Non `#` characters exist between `r` and `"`, e.g. `r##~"abcde"##`
    InvalidStarter { bad_char: char },
    /// The string was not terminated, e.g. `r###"abcde"##`.
    /// `possible_terminator_offset` is the number of characters after `r` or
    /// `br` where they may have intended to terminate it.
    NoTerminator { expected: u32, found: u32, possible_terminator_offset: Option<u32> },
    /// More than 255 `#`s exist.
    TooManyDelimiters { found: u32 },
}

/// Base of numeric literal encoding according to its prefix.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Base {
    /// Literal starts with "0b".
    Binary = 2,
    /// Literal starts with "0o".
    Octal = 8,
    /// Literal doesn't contain a prefix.
    Decimal = 10,
    /// Literal starts with "0x".
    Hexadecimal = 16,
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

/// Validates a raw string literal. Used for getting more information about a
/// problem with a `RawStr`/`RawByteStr` with a `None` field.
#[inline]
pub fn validate_raw_str(input: &str, prefix_len: u32) -> Result<(), RawStrError> {
    debug_assert!(!input.is_empty());
    let mut cursor = Cursor::new(input, FrontmatterAllowed::No);
    // Move past the leading `r` or `br`.
    for _ in 0..prefix_len {
        cursor.bump().unwrap();
    }
    cursor.raw_double_quoted_string(prefix_len).map(|_| ())
}

/// Creates an iterator that produces tokens from the input string.
pub fn tokenize(input: &str) -> impl Iterator<Item = Token> {
    let mut cursor = Cursor::new(input, FrontmatterAllowed::No);
    std::iter::from_fn(move || {
        let token = cursor.advance_token();
        if token.kind != TokenKind::Eof { Some(token) } else { None }
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
    c == '_' || unicode_xid::UnicodeXID::is_xid_start(c)
}

/// True if `c` is valid as a non-first character of an identifier.
/// See [Rust language reference](https://doc.rust-lang.org/reference/identifiers.html) for
/// a formal definition of valid identifier name.
pub fn is_id_continue(c: char) -> bool {
    unicode_xid::UnicodeXID::is_xid_continue(c)
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

impl Cursor<'_> {
    /// Parses a token from the input string.
    pub fn advance_token(&mut self) -> Token {
        let first_char = match self.bump() {
            Some(c) => c,
            None => return Token::new(TokenKind::Eof, 0),
        };

        let token_kind = match first_char {
            c if matches!(self.frontmatter_allowed, FrontmatterAllowed::Yes)
                && is_whitespace(c) =>
            {
                let mut last = first_char;
                while is_whitespace(self.first()) {
                    let Some(c) = self.bump() else {
                        break;
                    };
                    last = c;
                }
                // invalid frontmatter opening as whitespace preceding it isn't newline.
                // combine the whitespace and the frontmatter to a single token as we shall
                // error later.
                if last != '\n' && self.as_str().starts_with("---") {
                    self.bump();
                    self.frontmatter(true)
                } else {
                    Whitespace
                }
            }
            '-' if matches!(self.frontmatter_allowed, FrontmatterAllowed::Yes)
                && self.as_str().starts_with("--") =>
            {
                // happy path
                self.frontmatter(false)
            }
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
                    let res = self.raw_double_quoted_string(1);
                    let suffix_start = self.pos_within_token();
                    if res.is_ok() {
                        self.eat_literal_suffix();
                    }
                    let kind = RawStr { n_hashes: res.ok() };
                    Literal { kind, suffix_start }
                }
                _ => self.ident_or_unknown_prefix(),
            },

            // Byte literal, byte string literal, raw byte string literal or identifier.
            'b' => self.c_or_byte_string(
                |terminated| ByteStr { terminated },
                |n_hashes| RawByteStr { n_hashes },
                Some(|terminated| Byte { terminated }),
            ),

            // c-string literal, raw c-string literal or identifier.
            'c' => self.c_or_byte_string(
                |terminated| CStr { terminated },
                |n_hashes| RawCStr { n_hashes },
                None,
            ),

            // Identifier (this should be checked after other variant that can
            // start as identifier).
            c if is_id_start(c) => self.ident_or_unknown_prefix(),

            // Numeric literal.
            c @ '0'..='9' => {
                let literal_kind = self.number(c);
                let suffix_start = self.pos_within_token();
                self.eat_literal_suffix();
                TokenKind::Literal { kind: literal_kind, suffix_start }
            }

            // Guarded string literal prefix: `#"` or `##`
            '#' if matches!(self.first(), '"' | '#') => {
                self.bump();
                TokenKind::GuardedStrPrefix
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
            '\'' => self.lifetime_or_char(),

            // String literal.
            '"' => {
                let terminated = self.double_quoted_string();
                let suffix_start = self.pos_within_token();
                if terminated {
                    self.eat_literal_suffix();
                }
                let kind = Str { terminated };
                Literal { kind, suffix_start }
            }
            // Identifier starting with an emoji. Only lexed for graceful error recovery.
            c if !c.is_ascii() && c.is_emoji_char() => self.invalid_ident(),
            _ => Unknown,
        };
        if matches!(self.frontmatter_allowed, FrontmatterAllowed::Yes)
            && !matches!(token_kind, Whitespace)
        {
            // stop allowing frontmatters after first non-whitespace token
            self.frontmatter_allowed = FrontmatterAllowed::No;
        }
        let res = Token::new(token_kind, self.pos_within_token());
        self.reset_pos_within_token();
        res
    }

    /// Given that one `-` was eaten, eat the rest of the frontmatter.
    fn frontmatter(&mut self, has_invalid_preceding_whitespace: bool) -> TokenKind {
        debug_assert_eq!('-', self.prev());

        let pos = self.pos_within_token();
        self.eat_while(|c| c == '-');

        // one `-` is eaten by the caller.
        let length_opening = self.pos_within_token() - pos + 1;

        // must be ensured by the caller
        debug_assert!(length_opening >= 3);

        // whitespace between the opening and the infostring.
        self.eat_while(|ch| ch != '\n' && is_whitespace(ch));

        // copied from `eat_identifier`, but allows `.` in infostring to allow something like
        // `---Cargo.toml` as a valid opener
        if is_id_start(self.first()) {
            self.bump();
            self.eat_while(|c| is_id_continue(c) || c == '.');
        }

        self.eat_while(|ch| ch != '\n' && is_whitespace(ch));
        let invalid_infostring = self.first() != '\n';

        let mut s = self.as_str();
        let mut found = false;
        let mut size = 0;
        while let Some(closing) = s.find(&"-".repeat(length_opening as usize)) {
            let preceding_chars_start = s[..closing].rfind("\n").map_or(0, |i| i + 1);
            if s[preceding_chars_start..closing].chars().all(is_whitespace) {
                // candidate found
                self.bump_bytes(size + closing);
                // in case like
                // ---cargo
                // --- blahblah
                // or
                // ---cargo
                // ----
                // combine those stuff into this frontmatter token such that it gets detected later.
                self.eat_until(b'\n');
                found = true;
                break;
            } else {
                s = &s[closing + length_opening as usize..];
                size += closing + length_opening as usize;
            }
        }

        if !found {
            // recovery strategy: a closing statement might have precending whitespace/newline
            // but not have enough dashes to properly close. In this case, we eat until there,
            // and report a mismatch in the parser.
            let mut rest = self.as_str();
            // We can look for a shorter closing (starting with four dashes but closing with three)
            // and other indications that Rust has started and the infostring has ended.
            let mut potential_closing = rest
                .find("\n---")
                // n.b. only in the case where there are dashes, we move the index to the line where
                // the dashes start as we eat to include that line. For other cases those are Rust code
                // and not included in the frontmatter.
                .map(|x| x + 1)
                .or_else(|| rest.find("\nuse "))
                .or_else(|| rest.find("\n//!"))
                .or_else(|| rest.find("\n#!["));

            if potential_closing.is_none() {
                // a less fortunate recovery if all else fails which finds any dashes preceded by whitespace
                // on a standalone line. Might be wrong.
                while let Some(closing) = rest.find("---") {
                    let preceding_chars_start = rest[..closing].rfind("\n").map_or(0, |i| i + 1);
                    if rest[preceding_chars_start..closing].chars().all(is_whitespace) {
                        // candidate found
                        potential_closing = Some(closing);
                        break;
                    } else {
                        rest = &rest[closing + 3..];
                    }
                }
            }

            if let Some(potential_closing) = potential_closing {
                // bump to the potential closing, and eat everything on that line.
                self.bump_bytes(potential_closing);
                self.eat_until(b'\n');
            } else {
                // eat everything. this will get reported as an unclosed frontmatter.
                self.eat_while(|_| true);
            }
        }

        Frontmatter { has_invalid_preceding_whitespace, invalid_infostring }
    }

    fn line_comment(&mut self) -> TokenKind {
        debug_assert!(self.prev() == '/' && self.first() == '/');
        self.bump();

        let doc_style = match self.first() {
            // `//!` is an inner line doc comment.
            '!' => Some(DocStyle::Inner),
            // `////` (more than 3 slashes) is not considered a doc comment.
            '/' if self.second() != '/' => Some(DocStyle::Outer),
            _ => None,
        };

        self.eat_until(b'\n');
        LineComment { doc_style }
    }

    fn block_comment(&mut self) -> TokenKind {
        debug_assert!(self.prev() == '/' && self.first() == '*');
        self.bump();

        let doc_style = match self.first() {
            // `/*!` is an inner block doc comment.
            '!' => Some(DocStyle::Inner),
            // `/***` (more than 2 stars) is not considered a doc comment.
            // `/**/` is not considered a doc comment.
            '*' if !matches!(self.second(), '*' | '/') => Some(DocStyle::Outer),
            _ => None,
        };

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

        BlockComment { doc_style, terminated: depth == 0 }
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

    fn ident_or_unknown_prefix(&mut self) -> TokenKind {
        debug_assert!(is_id_start(self.prev()));
        // Start is already eaten, eat the rest of identifier.
        self.eat_while(is_id_continue);
        // Known prefixes must have been handled earlier. So if
        // we see a prefix here, it is definitely an unknown prefix.
        match self.first() {
            '#' | '"' | '\'' => UnknownPrefix,
            c if !c.is_ascii() && c.is_emoji_char() => self.invalid_ident(),
            _ => Ident,
        }
    }

    fn invalid_ident(&mut self) -> TokenKind {
        // Start is already eaten, eat the rest of identifier.
        self.eat_while(|c| {
            const ZERO_WIDTH_JOINER: char = '\u{200d}';
            is_id_continue(c) || (!c.is_ascii() && c.is_emoji_char()) || c == ZERO_WIDTH_JOINER
        });
        // An invalid identifier followed by '#' or '"' or '\'' could be
        // interpreted as an invalid literal prefix. We don't bother doing that
        // because the treatment of invalid identifiers and invalid prefixes
        // would be the same.
        InvalidIdent
    }

    fn c_or_byte_string(
        &mut self,
        mk_kind: fn(bool) -> LiteralKind,
        mk_kind_raw: fn(Option<u8>) -> LiteralKind,
        single_quoted: Option<fn(bool) -> LiteralKind>,
    ) -> TokenKind {
        match (self.first(), self.second(), single_quoted) {
            ('\'', _, Some(single_quoted)) => {
                self.bump();
                let terminated = self.single_quoted_string();
                let suffix_start = self.pos_within_token();
                if terminated {
                    self.eat_literal_suffix();
                }
                let kind = single_quoted(terminated);
                Literal { kind, suffix_start }
            }
            ('"', _, _) => {
                self.bump();
                let terminated = self.double_quoted_string();
                let suffix_start = self.pos_within_token();
                if terminated {
                    self.eat_literal_suffix();
                }
                let kind = mk_kind(terminated);
                Literal { kind, suffix_start }
            }
            ('r', '"', _) | ('r', '#', _) => {
                self.bump();
                let res = self.raw_double_quoted_string(2);
                let suffix_start = self.pos_within_token();
                if res.is_ok() {
                    self.eat_literal_suffix();
                }
                let kind = mk_kind_raw(res.ok());
                Literal { kind, suffix_start }
            }
            _ => self.ident_or_unknown_prefix(),
        }
    }

    fn number(&mut self, first_digit: char) -> LiteralKind {
        debug_assert!('0' <= self.prev() && self.prev() <= '9');
        let mut base = Base::Decimal;
        if first_digit == '0' {
            // Attempt to parse encoding base.
            match self.first() {
                'b' => {
                    base = Base::Binary;
                    self.bump();
                    if !self.eat_decimal_digits() {
                        return Int { base, empty_int: true };
                    }
                }
                'o' => {
                    base = Base::Octal;
                    self.bump();
                    if !self.eat_decimal_digits() {
                        return Int { base, empty_int: true };
                    }
                }
                'x' => {
                    base = Base::Hexadecimal;
                    self.bump();
                    if !self.eat_hexadecimal_digits() {
                        return Int { base, empty_int: true };
                    }
                }
                // Not a base prefix; consume additional digits.
                '0'..='9' | '_' => {
                    self.eat_decimal_digits();
                }

                // Also not a base prefix; nothing more to do here.
                '.' | 'e' | 'E' => {}

                // Just a 0.
                _ => return Int { base, empty_int: false },
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
                if self.first().is_ascii_digit() {
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
            is_id_start(self.first()) || self.first().is_ascii_digit()
        };

        if !can_be_a_lifetime {
            let terminated = self.single_quoted_string();
            let suffix_start = self.pos_within_token();
            if terminated {
                self.eat_literal_suffix();
            }
            let kind = Char { terminated };
            return Literal { kind, suffix_start };
        }

        if self.first() == 'r' && self.second() == '#' && is_id_start(self.third()) {
            // Eat "r" and `#`, and identifier start characters.
            self.bump();
            self.bump();
            self.bump();
            self.eat_while(is_id_continue);
            return RawLifetime;
        }

        // Either a lifetime or a character literal with
        // length greater than 1.
        let starts_with_number = self.first().is_ascii_digit();

        // Skip the literal contents.
        // First symbol can be a number (which isn't a valid identifier start),
        // so skip it without any checks.
        self.bump();
        self.eat_while(is_id_continue);

        match self.first() {
            // Check if after skipping literal contents we've met a closing
            // single quote (which means that user attempted to create a
            // string with single quotes).
            '\'' => {
                self.bump();
                let kind = Char { terminated: true };
                Literal { kind, suffix_start: self.pos_within_token() }
            }
            '#' if !starts_with_number => UnknownPrefixLifetime,
            _ => Lifetime { starts_with_number },
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

    /// Attempt to lex for a guarded string literal.
    ///
    /// Used by `rustc_parse::lexer` to lex for guarded strings
    /// conditionally based on edition.
    ///
    /// Note: this will not reset the `Cursor` when a
    /// guarded string is not found. It is the caller's
    /// responsibility to do so.
    pub fn guarded_double_quoted_string(&mut self) -> Option<GuardedStr> {
        debug_assert!(self.prev() != '#');

        let mut n_start_hashes: u32 = 0;
        while self.first() == '#' {
            n_start_hashes += 1;
            self.bump();
        }

        if self.first() != '"' {
            return None;
        }
        self.bump();
        debug_assert!(self.prev() == '"');

        // Lex the string itself as a normal string literal
        // so we can recover that for older editions later.
        let terminated = self.double_quoted_string();
        if !terminated {
            let token_len = self.pos_within_token();
            self.reset_pos_within_token();

            return Some(GuardedStr { n_hashes: n_start_hashes, terminated: false, token_len });
        }

        // Consume closing '#' symbols.
        // Note that this will not consume extra trailing `#` characters:
        // `###"abcde"####` is lexed as a `GuardedStr { n_end_hashes: 3, .. }`
        // followed by a `#` token.
        let mut n_end_hashes = 0;
        while self.first() == '#' && n_end_hashes < n_start_hashes {
            n_end_hashes += 1;
            self.bump();
        }

        // Reserved syntax, always an error, so it doesn't matter if
        // `n_start_hashes != n_end_hashes`.

        self.eat_literal_suffix();

        let token_len = self.pos_within_token();
        self.reset_pos_within_token();

        Some(GuardedStr { n_hashes: n_start_hashes, terminated: true, token_len })
    }

    /// Eats the double-quoted string and returns `n_hashes` and an error if encountered.
    fn raw_double_quoted_string(&mut self, prefix_len: u32) -> Result<u8, RawStrError> {
        // Wrap the actual function to handle the error with too many hashes.
        // This way, it eats the whole raw string.
        let n_hashes = self.raw_string_unvalidated(prefix_len)?;
        // Only up to 255 `#`s are allowed in raw strings
        match u8::try_from(n_hashes) {
            Ok(num) => Ok(num),
            Err(_) => Err(RawStrError::TooManyDelimiters { found: n_hashes }),
        }
    }

    fn raw_string_unvalidated(&mut self, prefix_len: u32) -> Result<u32, RawStrError> {
        debug_assert!(self.prev() == 'r');
        let start_pos = self.pos_within_token();
        let mut possible_terminator_offset = None;
        let mut max_hashes = 0;

        // Count opening '#' symbols.
        let mut eaten = 0;
        while self.first() == '#' {
            eaten += 1;
            self.bump();
        }
        let n_start_hashes = eaten;

        // Check that string is started.
        match self.bump() {
            Some('"') => (),
            c => {
                let c = c.unwrap_or(EOF_CHAR);
                return Err(RawStrError::InvalidStarter { bad_char: c });
            }
        }

        // Skip the string contents and on each '#' character met, check if this is
        // a raw string termination.
        loop {
            self.eat_until(b'"');

            if self.is_eof() {
                return Err(RawStrError::NoTerminator {
                    expected: n_start_hashes,
                    found: max_hashes,
                    possible_terminator_offset,
                });
            }

            // Eat closing double quote.
            self.bump();

            // Check that amount of closing '#' symbols
            // is equal to the amount of opening ones.
            // Note that this will not consume extra trailing `#` characters:
            // `r###"abcde"####` is lexed as a `RawStr { n_hashes: 3 }`
            // followed by a `#` token.
            let mut n_end_hashes = 0;
            while self.first() == '#' && n_end_hashes < n_start_hashes {
                n_end_hashes += 1;
                self.bump();
            }

            if n_end_hashes == n_start_hashes {
                return Ok(n_start_hashes);
            } else if n_end_hashes > max_hashes {
                // Keep track of possible terminators to give a hint about
                // where there might be a missing terminator
                possible_terminator_offset =
                    Some(self.pos_within_token() - start_pos - n_end_hashes + prefix_len);
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

    // Eats the suffix of the literal, e.g. "u8".
    fn eat_literal_suffix(&mut self) {
        self.eat_identifier();
    }

    // Eats the identifier. Note: succeeds on `_`, which isn't a valid
    // identifier.
    fn eat_identifier(&mut self) {
        if !is_id_start(self.first()) {
            return;
        }
        self.bump();

        self.eat_while(is_id_continue);
    }
}
