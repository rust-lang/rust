use crate::cursor::{Cursor, EOF_CHAR};
use crate::{ident, is_id_continue, is_id_start, TokenKind};
use std::convert::TryFrom;

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
    RawStr { n_hashes: u16, err: Option<RawStrError> },
    /// "br"abc"", "br#"abc"#", "br####"ab"###"c"####", "br#"a"
    RawByteStr { n_hashes: u16, err: Option<RawStrError> },
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

/// Error produced validating a raw string. Represents cases like:
/// - `r##~"abcde"##`: `InvalidStarter`
/// - `r###"abcde"##`: `NoTerminator { expected: 3, found: 2, possible_terminator_offset: Some(11)`
/// - Too many `#`s (>65535): `TooManyDelimiters`
// perf note: It doesn't matter that this makes `Token` 36 bytes bigger. See #77629
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum RawStrError {
    /// Non `#` characters exist between `r` and `"` eg. `r#~"..`
    InvalidStarter { bad_char: char },
    /// The string was never terminated. `possible_terminator_offset` is the number of characters after `r` or `br` where they
    /// may have intended to terminate it.
    NoTerminator { expected: usize, found: usize, possible_terminator_offset: Option<usize> },
    /// More than 65535 `#`s exist.
    TooManyDelimiters { found: usize },
}

pub(crate) fn number(cursor: &mut Cursor<'_>, first_digit: char) -> LiteralKind {
    debug_assert!('0' <= cursor.prev() && cursor.prev() <= '9');
    let mut base = Base::Decimal;
    if first_digit == '0' {
        // Attempt to parse encoding base.
        let has_digits = match cursor.peek() {
            'b' => {
                base = Base::Binary;
                cursor.bump();
                eat_decimal_digits(cursor)
            }
            'o' => {
                base = Base::Octal;
                cursor.bump();
                eat_decimal_digits(cursor)
            }
            'x' => {
                base = Base::Hexadecimal;
                cursor.bump();
                eat_hexadecimal_digits(cursor)
            }
            // Not a base prefix.
            '0'..='9' | '_' | '.' | 'e' | 'E' => {
                eat_decimal_digits(cursor);
                true
            }
            // Just a 0.
            _ => return LiteralKind::Int { base, empty_int: false },
        };
        // Base prefix was provided, but there were no digits
        // after it, e.g. "0x".
        if !has_digits {
            return LiteralKind::Int { base, empty_int: true };
        }
    } else {
        // No base prefix, parse number in the usual way.
        eat_decimal_digits(cursor);
    };

    match cursor.peek() {
        // Don't be greedy if this is actually an
        // integer literal followed by field/method access or a range pattern
        // (`0..2` and `12.foo()`)
        '.' if cursor.peek_second() != '.' && !is_id_start(cursor.peek_second()) => {
            // might have stuff after the ., and if it does, it needs to start
            // with a number
            cursor.bump();
            let mut empty_exponent = false;
            if cursor.peek().is_digit(10) {
                eat_decimal_digits(cursor);
                match cursor.peek() {
                    'e' | 'E' => {
                        cursor.bump();
                        empty_exponent = !eat_float_exponent(cursor);
                    }
                    _ => (),
                }
            }
            LiteralKind::Float { base, empty_exponent }
        }
        'e' | 'E' => {
            cursor.bump();
            let empty_exponent = !eat_float_exponent(cursor);
            LiteralKind::Float { base, empty_exponent }
        }
        _ => LiteralKind::Int { base, empty_int: false },
    }
}

pub(crate) fn eat_decimal_digits(cursor: &mut Cursor<'_>) -> bool {
    let mut has_digits = false;
    loop {
        match cursor.peek() {
            '_' => {
                cursor.bump();
            }
            '0'..='9' => {
                has_digits = true;
                cursor.bump();
            }
            _ => break,
        }
    }
    has_digits
}

pub(crate) fn eat_hexadecimal_digits(cursor: &mut Cursor<'_>) -> bool {
    let mut has_digits = false;
    loop {
        match cursor.peek() {
            '_' => {
                cursor.bump();
            }
            '0'..='9' | 'a'..='f' | 'A'..='F' => {
                has_digits = true;
                cursor.bump();
            }
            _ => break,
        }
    }
    has_digits
}

/// Eats the float exponent. Returns true if at least one digit was met,
/// and returns false otherwise.
fn eat_float_exponent(cursor: &mut Cursor<'_>) -> bool {
    debug_assert!(cursor.prev() == 'e' || cursor.prev() == 'E');
    if cursor.peek() == '-' || cursor.peek() == '+' {
        cursor.bump();
    }
    eat_decimal_digits(cursor)
}

pub(crate) fn lifetime_or_char(cursor: &mut Cursor<'_>) -> TokenKind {
    debug_assert!(cursor.prev() == '\'');

    let can_be_a_lifetime = if cursor.peek_second() == '\'' {
        // It's surely not a lifetime.
        false
    } else {
        // If the first symbol is valid for identifier, it can be a lifetime.
        // Also check if it's a number for a better error reporting (so '0 will
        // be reported as invalid lifetime and not as unterminated char literal).
        is_id_start(cursor.peek()) || cursor.peek().is_digit(10)
    };

    if !can_be_a_lifetime {
        let terminated = single_quoted_string(cursor);
        let suffix_start = cursor.len_consumed();
        if terminated {
            eat_literal_suffix(cursor);
        }
        let kind = LiteralKind::Char { terminated };
        return TokenKind::Literal { kind, suffix_start };
    }

    // Either a lifetime or a character literal with
    // length greater than 1.

    let starts_with_number = cursor.peek().is_digit(10);

    // Skip the literal contents.
    // First symbol can be a number (which isn't a valid identifier start),
    // so skip it without any checks.
    cursor.bump();
    cursor.bump_while(is_id_continue);

    // Check if after skipping literal contents we've met a closing
    // single quote (which means that user attempted to create a
    // string with single quotes).
    if cursor.peek() == '\'' {
        cursor.bump();
        let kind = LiteralKind::Char { terminated: true };
        TokenKind::Literal { kind, suffix_start: cursor.len_consumed() }
    } else {
        TokenKind::Lifetime { starts_with_number }
    }
}

pub(crate) fn single_quoted_string(cursor: &mut Cursor<'_>) -> bool {
    debug_assert!(cursor.prev() == '\'');
    // Check if it's a one-symbol literal.
    if cursor.peek_second() == '\'' && cursor.peek() != '\\' {
        cursor.bump();
        cursor.bump();
        return true;
    }

    // Literal has more than one symbol.

    // Parse until either quotes are terminated or error is detected.
    loop {
        match cursor.peek() {
            // Quotes are terminated, finish parsing.
            '\'' => {
                cursor.bump();
                return true;
            }
            // Probably beginning of the comment, which we don't want to include
            // to the error report.
            '/' => break,
            // Newline without following '\'' means unclosed quote, stop parsing.
            '\n' if cursor.peek_second() != '\'' => break,
            // End of file, stop parsing.
            EOF_CHAR if cursor.is_eof() => break,
            // Escaped slash is considered one character, so bump twice.
            '\\' => {
                cursor.bump();
                cursor.bump();
            }
            // Skip the character.
            _ => {
                cursor.bump();
            }
        }
    }
    // String was not terminated.
    false
}

/// Eats double-quoted string and returns true
/// if string is terminated.
pub(crate) fn double_quoted_string(cursor: &mut Cursor<'_>) -> bool {
    debug_assert!(cursor.prev() == '"');
    while let Some(c) = cursor.bump() {
        match c {
            '"' => {
                return true;
            }
            '\\' if cursor.peek() == '\\' || cursor.peek() == '"' => {
                // Bump again to skip escaped character.
                cursor.bump();
            }
            _ => (),
        }
    }
    // End of file reached.
    false
}

/// Eats the double-quoted string and returns `n_hashes` and an error if encountered.
pub(crate) fn raw_double_quoted_string(
    cursor: &mut Cursor<'_>,
    prefix_len: usize,
) -> (u16, Option<RawStrError>) {
    // Wrap the actual function to handle the error with too many hashes.
    // This way, it eats the whole raw string.
    let (n_hashes, err) = raw_string_unvalidated(cursor, prefix_len);

    // Only up to 65535 `#`s are allowed in raw strings.
    match u16::try_from(n_hashes) {
        Ok(num) => (num, err),
        // We lie about the number of hashes here :P
        Err(_) => (0, Some(RawStrError::TooManyDelimiters { found: n_hashes })),
    }
}

fn raw_string_unvalidated(
    cursor: &mut Cursor<'_>,
    prefix_len: usize,
) -> (usize, Option<RawStrError>) {
    debug_assert!(cursor.prev() == 'r');
    let start_pos = cursor.len_consumed();
    let mut possible_terminator_offset = None;
    let mut max_hashes = 0;

    // Count opening '#' symbols.
    let mut eaten = 0;
    while cursor.peek() == '#' {
        eaten += 1;
        cursor.bump();
    }
    let n_start_hashes = eaten;

    // Check that string is started.
    match cursor.bump() {
        Some('"') => (),
        c => {
            let c = c.unwrap_or(EOF_CHAR);
            return (n_start_hashes, Some(RawStrError::InvalidStarter { bad_char: c }));
        }
    }

    // Skip the string contents and on each '#' character met, check if this is
    // a raw string termination.
    loop {
        cursor.bump_while(|c| c != '"');

        if cursor.is_eof() {
            return (
                n_start_hashes,
                Some(RawStrError::NoTerminator {
                    expected: n_start_hashes,
                    found: max_hashes,
                    possible_terminator_offset,
                }),
            );
        }

        // Eat closing double quote.
        cursor.bump();

        // Check that amount of closing '#' symbols
        // is equal to the amount of opening ones.
        // Note that this will not consume extra trailing `#` characters:
        // `r###"abcde"####` is lexed as a `RawStr { n_hashes: 3 }`
        // followed by a `#` token.
        let mut n_end_hashes = 0;
        while cursor.peek() == '#' && n_end_hashes < n_start_hashes {
            n_end_hashes += 1;
            cursor.bump();
        }

        if n_end_hashes == n_start_hashes {
            return (n_start_hashes, None);
        } else if n_end_hashes > max_hashes {
            // Keep track of possible terminators to give a hint about
            // where there might be a missing terminator
            possible_terminator_offset =
                Some(cursor.len_consumed() - start_pos - n_end_hashes + prefix_len);
            max_hashes = n_end_hashes;
        }
    }
}

/// Eats the suffix of a literal, e.g. "_u8".
pub(crate) fn eat_literal_suffix(cursor: &mut Cursor<'_>) {
    // Eats one identifier.
    if is_id_start(cursor.peek()) {
        cursor.bump();
        ident(cursor);
    }
}
