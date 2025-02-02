//! Utilities for validating string and char literals and turning them into
//! values they represent.

use std::iter::{Peekable, from_fn};
use std::ops::Range;
use std::str::{CharIndices, Chars};

use Mode::*;

#[cfg(test)]
mod tests;

/// Errors and warnings that can occur during string unescaping. They mostly
/// relate to malformed escape sequences, but there are a few that are about
/// other problems.
#[derive(Debug, PartialEq, Eq)]
pub enum EscapeError {
    /// Expected 1 char, but 0 were found.
    ZeroChars,
    /// Expected 1 char, but more than 1 were found.
    MoreThanOneChar,

    /// Escaped '\' character without continuation.
    LoneSlash,
    /// Invalid escape character (e.g. '\z').
    InvalidEscape,
    /// Raw '\r' encountered.
    BareCarriageReturn,
    /// Raw '\r' encountered in raw string.
    BareCarriageReturnInRawString,
    /// Unescaped character that was expected to be escaped (e.g. raw '\t').
    EscapeOnlyChar,

    /// Numeric character escape is too short (e.g. '\x1').
    TooShortHexEscape,
    /// Invalid character in numeric escape (e.g. '\xz')
    InvalidCharInHexEscape,
    /// Character code in numeric escape is non-ascii (e.g. '\xFF').
    OutOfRangeHexEscape,

    /// '\u' not followed by '{'.
    NoBraceInUnicodeEscape,
    /// Non-hexadecimal value in '\u{..}'.
    InvalidCharInUnicodeEscape,
    /// '\u{}'
    EmptyUnicodeEscape,
    /// No closing brace in '\u{..}', e.g. '\u{12'.
    UnclosedUnicodeEscape,
    /// '\u{_12}'
    LeadingUnderscoreUnicodeEscape,
    /// More than 6 characters in '\u{..}', e.g. '\u{10FFFF_FF}'
    OverlongUnicodeEscape,
    /// Invalid in-bound unicode character code, e.g. '\u{DFFF}'.
    LoneSurrogateUnicodeEscape,
    /// Out of bounds unicode character code, e.g. '\u{FFFFFF}'.
    OutOfRangeUnicodeEscape,

    /// Unicode escape code in byte literal.
    UnicodeEscapeInByte,
    /// Non-ascii character in byte literal, byte string literal, or raw byte string literal.
    NonAsciiCharInByte,

    // `\0` in a C string literal.
    NulInCStr,

    /// After a line ending with '\', the next line contains whitespace
    /// characters that are not skipped.
    UnskippedWhitespaceWarning,

    /// After a line ending with '\', multiple lines are skipped.
    MultipleSkippedLinesWarning,
}

impl EscapeError {
    /// Returns true for actual errors, as opposed to warnings.
    pub fn is_fatal(&self) -> bool {
        !matches!(
            self,
            EscapeError::UnskippedWhitespaceWarning | EscapeError::MultipleSkippedLinesWarning
        )
    }
}

/// Used for mixed utf8 string literals, i.e. those that allow both unicode
/// chars and high bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MixedUnit {
    /// Used for ASCII chars (written directly or via `\x00`..`\x7f` escapes)
    /// and Unicode chars (written directly or via `\u` escapes).
    ///
    /// For example, if '¥' appears in a string it is represented here as
    /// `MixedUnit::Char('¥')`, and it will be appended to the relevant byte
    /// string as the two-byte UTF-8 sequence `[0xc2, 0xa5]`
    Char(char),

    /// Used for high bytes (`\x80`..`\xff`).
    ///
    /// For example, if `\xa5` appears in a string it is represented here as
    /// `MixedUnit::HighByte(0xa5)`, and it will be appended to the relevant
    /// byte string as the single byte `0xa5`.
    HighByte(u8),
}

impl TryFrom<char> for MixedUnit {
    type Error = EscapeError;

    fn try_from(c: char) -> Result<Self, EscapeError> {
        match c {
            '\0' => Err(EscapeError::NulInCStr),
            _ => Ok(MixedUnit::Char(c)),
        }
    }
}

impl TryFrom<u8> for MixedUnit {
    type Error = EscapeError;

    fn try_from(byte: u8) -> Result<Self, EscapeError> {
        match byte {
            0 => Err(EscapeError::NulInCStr),
            _ if byte.is_ascii() => Ok(MixedUnit::Char(byte as char)),
            _ => Ok(MixedUnit::HighByte(byte)),
        }
    }
}

/// Takes the contents of a raw string literal (without quotes) and produces a
/// sequence of characters or errors, which are returned by invoking `callback`.
/// NOTE: Raw strings don't do any unescaping, but do produce errors on bare CR.
fn check_raw_str<F>(src: &str, callback: &mut F)
where
    F: FnMut(Range<usize>, Result<char, EscapeError>),
{
    src.char_indices().for_each(|(pos, c)| {
        callback(
            pos..pos + c.len_utf8(),
            if c == '\r' { Err(EscapeError::BareCarriageReturnInRawString) } else { Ok(c) },
        );
    });
}

/// Takes the contents of a raw byte string literal (without quotes) and produces a
/// sequence of characters or errors, which are returned by invoking `callback`.
/// NOTE: Raw strings don't do any unescaping, but do produce errors on bare CR.
fn check_raw_byte_str<F>(src: &str, callback: &mut F)
where
    F: FnMut(Range<usize>, Result<u8, EscapeError>),
{
    src.char_indices().for_each(|(pos, c)| {
        callback(
            pos..pos + c.len_utf8(),
            if c == '\r' {
                Err(EscapeError::BareCarriageReturnInRawString)
            } else {
                ascii_char_to_byte(c)
            },
        );
    });
}

/// Takes the contents of a raw C string literal (without quotes) and produces a
/// sequence of characters or errors, which are returned by invoking `callback`.
/// NOTE: Raw strings don't do any unescaping, but do produce errors on bare CR.
fn check_raw_cstr<F>(src: &str, callback: &mut F)
where
    F: FnMut(Range<usize>, Result<char, EscapeError>),
{
    src.char_indices().for_each(|(pos, c)| {
        callback(pos..pos + c.len_utf8(), match c {
            '\r' => Err(EscapeError::BareCarriageReturnInRawString),
            '\0' => Err(EscapeError::NulInCStr),
            _ => Ok(c),
        });
    });
}

/// Take the contents of a string literal (without quotes)
/// and produce a sequence of escaped characters or errors,
/// which are returned by invoking `callback`.
pub fn unescape_str<F>(src: &str, callback: &mut F)
where
    F: FnMut(Range<usize>, Result<char, EscapeError>),
{
    let mut chars = src.char_indices().peekable();
    while let Some((start, c)) = chars.next() {
        let res = match c {
            // skip whitespace for backslash newline, see [Rust language reference]
            // (https://doc.rust-lang.org/reference/tokens.html#string-literals).
            '\\' if chars.next_if(|&(_, c)| c == '\n').is_some() => {
                let mut callback_err = |range, err| callback(range, Err(err));
                skip_ascii_whitespace(&mut chars, start, &mut callback_err);
                continue;
            }
            '\\' => scan_escape_for_char(&mut from_fn(|| chars.next().map(|i| i.1))),
            '"' => Err(EscapeError::EscapeOnlyChar),
            '\r' => Err(EscapeError::BareCarriageReturn),
            c => Ok(c),
        };
        let end = chars.peek().map(|&(end, _)| end).unwrap_or(src.len());
        callback(start..end, res);
    }
}

/// Take the contents of a byte string literal (without quotes)
/// and produce a sequence of unescaped bytes or errors,
/// which are returned by invoking `callback`.
pub fn unescape_byte_str<F>(src: &str, callback: &mut F)
where
    F: FnMut(Range<usize>, Result<u8, EscapeError>),
{
    let mut chars = src.char_indices().peekable();
    while let Some((start, c)) = chars.next() {
        let res = match c {
            // skip whitespace for backslash newline, see [Rust language reference]
            // (https://doc.rust-lang.org/reference/tokens.html#string-literals).
            '\\' if chars.next_if(|&(_, c)| c == '\n').is_some() => {
                let mut callback_err = |range, err| callback(range, Err(err));
                skip_ascii_whitespace(&mut chars, start, &mut callback_err);
                continue;
            }
            '\\' => scan_escape_for_byte(&mut from_fn(|| chars.next().map(|i| i.1))),
            '"' => Err(EscapeError::EscapeOnlyChar),
            '\r' => Err(EscapeError::BareCarriageReturn),
            c => ascii_char_to_byte(c),
        };
        let end = chars.peek().map(|&(end, _)| end).unwrap_or(src.len());
        callback(start..end, res);
    }
}

/// Take the contents of a C string literal (without quotes)
/// and produce a sequence of unescaped characters or errors,
/// which are returned by invoking `callback`.
pub fn unescape_cstr<F>(src: &str, callback: &mut F)
where
    F: FnMut(Range<usize>, Result<MixedUnit, EscapeError>),
{
    let mut chars = src.char_indices().peekable();
    while let Some((start, c)) = chars.next() {
        let res = match c {
            // skip whitespace for backslash newline, see [Rust language reference]
            // (https://doc.rust-lang.org/reference/tokens.html#string-literals).
            '\\' if chars.next_if(|&(_, c)| c == '\n').is_some() => {
                let mut callback_err = |range, err| callback(range, Err(err));
                skip_ascii_whitespace(&mut chars, start, &mut callback_err);
                continue;
            }
            '\\' => scan_escape_for_cstr(&mut from_fn(|| chars.next().map(|i| i.1))),
            '"' => Err(EscapeError::EscapeOnlyChar),
            '\r' => Err(EscapeError::BareCarriageReturn),
            c => c.try_into(),
        };
        let end = chars.peek().map(|&(end, _)| end).unwrap_or(src.len());
        callback(start..end, res);
    }
}

/// Skip ASCII whitespace, except for the formfeed character
/// (see [this issue](https://github.com/rust-lang/rust/issues/136600)).
/// Warns on unescaped newline and following non-ASCII whitespace.
fn skip_ascii_whitespace<F>(chars: &mut Peekable<CharIndices<'_>>, start: usize, callback: &mut F)
where
    F: FnMut(Range<usize>, EscapeError),
{
    // the escaping slash and newline characters add 2 bytes
    let mut end = start + 2;
    let mut contains_nl = false;
    while let Some((_, c)) = chars.next_if(|(_, c)| c.is_ascii_whitespace() && *c != '\x0c') {
        end += 1;
        contains_nl = contains_nl || c == '\n';
    }

    if contains_nl {
        callback(start..end, EscapeError::MultipleSkippedLinesWarning);
    }
    if let Some((_, c)) = chars.peek() {
        if c.is_whitespace() {
            // for error reporting, include the character that was not skipped in the span
            callback(start..end + c.len_utf8(), EscapeError::UnskippedWhitespaceWarning);
        }
    }
}

/// Takes the contents of a char literal (without quotes),
/// and returns an unescaped char or an error.
pub fn unescape_char(src: &str) -> Result<char, EscapeError> {
    unescape_char_iter(&mut src.chars())
}

fn unescape_char_iter(chars: &mut Chars<'_>) -> Result<char, EscapeError> {
    let res = match chars.next().ok_or(EscapeError::ZeroChars)? {
        '\\' => scan_escape_for_char(chars),
        '\n' | '\t' | '\'' => Err(EscapeError::EscapeOnlyChar),
        '\r' => Err(EscapeError::BareCarriageReturn),
        c => Ok(c),
    }?;
    if chars.next().is_some() {
        return Err(EscapeError::MoreThanOneChar);
    }
    Ok(res)
}

/// Takes the contents of a byte literal (without quotes),
/// and returns an unescaped byte or an error.
pub fn unescape_byte(src: &str) -> Result<u8, EscapeError> {
    unescape_byte_iter(&mut src.chars())
}

fn unescape_byte_iter(chars: &mut Chars<'_>) -> Result<u8, EscapeError> {
    let res = match chars.next().ok_or(EscapeError::ZeroChars)? {
        '\\' => scan_escape_for_byte(chars),
        '\n' | '\t' | '\'' => Err(EscapeError::EscapeOnlyChar),
        '\r' => Err(EscapeError::BareCarriageReturn),
        c => ascii_char_to_byte(c),
    }?;
    if chars.next().is_some() {
        return Err(EscapeError::MoreThanOneChar);
    }
    Ok(res)
}

/// Scan an escape sequence for a char
fn scan_escape_for_char(chars: &mut impl Iterator<Item = char>) -> Result<char, EscapeError> {
    // Previous character was '\\', unescape what follows.
    let c = chars.next().ok_or(EscapeError::LoneSlash)?;
    nul_escape(c).map(char::from).or_else(|c| {
        simple_escape(c).map(char::from).or_else(|c| match c {
            'x' => {
                let byte = hex_escape(chars)?;
                if byte.is_ascii() {
                    Ok(byte as char)
                } else {
                    Err(EscapeError::OutOfRangeHexEscape)
                }
            }
            'u' => {
                let value = unicode_escape(chars)?;
                if value > char::MAX as u32 {
                    Err(EscapeError::OutOfRangeUnicodeEscape)
                } else {
                    char::from_u32(value).ok_or(EscapeError::LoneSurrogateUnicodeEscape)
                }
            }
            _ => Err(EscapeError::InvalidEscape),
        })
    })
}

/// Scan an escape sequence for a byte
fn scan_escape_for_byte(chars: &mut impl Iterator<Item = char>) -> Result<u8, EscapeError> {
    // Previous character was '\\', unescape what follows.
    let c = chars.next().ok_or(EscapeError::LoneSlash)?;
    nul_escape(c).or_else(|c| {
        simple_escape(c).or_else(|c| match c {
            'x' => hex_escape(chars),
            'u' => {
                let _ = unicode_escape(chars)?;
                Err(EscapeError::UnicodeEscapeInByte)
            }
            _ => Err(EscapeError::InvalidEscape),
        })
    })
}

fn scan_escape_for_cstr(chars: &mut impl Iterator<Item = char>) -> Result<MixedUnit, EscapeError> {
    // Previous character was '\\', unescape what follows.
    let c = chars.next().ok_or(EscapeError::LoneSlash)?;
    simple_escape(nul_escape_cstr(c)?).map(|b| MixedUnit::Char(b as char)).or_else(|c| match c {
        'x' => hex_escape(chars)?.try_into(),
        'u' => {
            let value = unicode_escape(chars)?;
            match value {
                0 => Err(EscapeError::NulInCStr),
                _ if value > char::MAX as u32 => Err(EscapeError::OutOfRangeUnicodeEscape),
                _ => char::from_u32(value)
                    .map(MixedUnit::Char)
                    .ok_or(EscapeError::LoneSurrogateUnicodeEscape),
            }
        }
        _ => Err(EscapeError::InvalidEscape),
    })
}

/// Parse a nul character without the leading backslash.
fn nul_escape(c: char) -> Result<u8, char> {
    if c == '0' { Ok(b'\0') } else { Err(c) }
}

/// Parse a nul character without the leading backslash for a C string.
fn nul_escape_cstr(c: char) -> Result<char, EscapeError> {
    if c == '0' { Err(EscapeError::NulInCStr) } else { Ok(c) }
}

/// Parse the character of an ASCII escape (except nul) without the leading backslash.
fn simple_escape(c: char) -> Result<u8, char> {
    // Previous character was '\\', unescape what follows.
    Ok(match c {
        '"' => b'"',
        'n' => b'\n',
        'r' => b'\r',
        't' => b'\t',
        '\\' => b'\\',
        '\'' => b'\'',
        _ => Err(c)?,
    })
}

/// Parse the two hexadecimal characters of a hexadecimal escape without the leading r"\x".
fn hex_escape(chars: &mut impl Iterator<Item = char>) -> Result<u8, EscapeError> {
    let hi = chars.next().ok_or(EscapeError::TooShortHexEscape)?;
    let hi = hi.to_digit(16).ok_or(EscapeError::InvalidCharInHexEscape)?;

    let lo = chars.next().ok_or(EscapeError::TooShortHexEscape)?;
    let lo = lo.to_digit(16).ok_or(EscapeError::InvalidCharInHexEscape)?;

    Ok((hi * 16 + lo) as u8)
}

/// Parse the braces with hexadecimal characters (and underscores) part of a unicode escape.
/// This r"{...}" normally comes after r"\u" and cannot start with an underscore.
fn unicode_escape(chars: &mut impl Iterator<Item = char>) -> Result<u32, EscapeError> {
    if chars.next() != Some('{') {
        return Err(EscapeError::NoBraceInUnicodeEscape);
    }

    // First character must be a hexadecimal digit.
    let mut n_digits = 1;
    let mut value: u32 = match chars.next().ok_or(EscapeError::UnclosedUnicodeEscape)? {
        '_' => return Err(EscapeError::LeadingUnderscoreUnicodeEscape),
        '}' => return Err(EscapeError::EmptyUnicodeEscape),
        c => c.to_digit(16).ok_or(EscapeError::InvalidCharInUnicodeEscape)?,
    };

    // First character is valid, now parse the rest of the number
    // and closing brace.
    loop {
        match chars.next() {
            None => return Err(EscapeError::UnclosedUnicodeEscape),
            Some('_') => continue,
            Some('}') => {
                // Incorrect syntax has higher priority for error reporting
                // than unallowed value for a literal.
                return if n_digits > 6 {
                    Err(EscapeError::OverlongUnicodeEscape)
                } else {
                    Ok(value)
                };
            }
            Some(c) => {
                let digit: u32 = c.to_digit(16).ok_or(EscapeError::InvalidCharInUnicodeEscape)?;
                n_digits += 1;
                if n_digits > 6 {
                    // Stop updating value since we're sure that it's incorrect already.
                    continue;
                }
                value = value * 16 + digit;
            }
        };
    }
}

/// Takes the contents of a unicode-only (non-mixed-utf8) literal (without quotes)
/// and produces a sequence of unescaped characters or errors,
/// which are returned by invoking `callback`.
///
/// For `Char` and `Byte` modes, the callback will be called exactly once.
pub fn unescape_unicode<F>(src: &str, mode: Mode, callback: &mut F)
where
    F: FnMut(Range<usize>, Result<char, EscapeError>),
{
    let mut byte_callback =
        |range, res: Result<u8, EscapeError>| callback(range, res.map(char::from));
    match mode {
        Char => {
            let mut chars = src.chars();
            let res = unescape_char_iter(&mut chars);
            callback(0..(src.len() - chars.as_str().len()), res);
        }
        Byte => {
            let mut chars = src.chars();
            let res = unescape_byte_iter(&mut chars).map(char::from);
            callback(0..(src.len() - chars.as_str().len()), res);
        }
        Str => unescape_str(src, callback),
        ByteStr => unescape_byte_str(src, &mut byte_callback),
        RawStr => check_raw_str(src, callback),
        RawByteStr => check_raw_byte_str(src, &mut byte_callback),
        RawCStr => check_raw_cstr(src, callback),
        CStr => unreachable!(),
    }
}

/// What kind of literal do we parse.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Mode {
    Char,

    Byte,

    Str,
    RawStr,

    ByteStr,
    RawByteStr,

    CStr,
    RawCStr,
}

impl Mode {
    pub fn in_double_quotes(self) -> bool {
        match self {
            Str | RawStr | ByteStr | RawByteStr | CStr | RawCStr => true,
            Char | Byte => false,
        }
    }

    pub fn prefix_noraw(self) -> &'static str {
        match self {
            Char | Str | RawStr => "",
            Byte | ByteStr | RawByteStr => "b",
            CStr | RawCStr => "c",
        }
    }
}

fn ascii_char_to_byte(c: char) -> Result<u8, EscapeError> {
    // do NOT do: c.try_into().ok_or(EscapeError::NonAsciiCharInByte)
    if c.is_ascii() { Ok(c as u8) } else { Err(EscapeError::NonAsciiCharInByte) }
}
