//! Utilities for validating string and char literals and turning them into
//! values they represent.

use std::ops::Range;
use std::str::Chars;

#[cfg(test)]
mod tests;

/// Errors and warnings that can occur during string unescaping.
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

/// Takes a contents of a literal (without quotes) and produces a
/// sequence of escaped characters or errors.
/// Values are returned through invoking of the provided callback.
pub fn unescape_literal<F>(src: &str, mode: Mode, callback: &mut F)
where
    F: FnMut(Range<usize>, Result<char, EscapeError>),
{
    match mode {
        Mode::Char | Mode::Byte => {
            let mut chars = src.chars();
            let res = unescape_char_or_byte(&mut chars, mode == Mode::Byte);
            callback(0..(src.len() - chars.as_str().len()), res);
        }
        Mode::Str | Mode::ByteStr => unescape_str_common(src, mode, callback),

        Mode::RawStr | Mode::RawByteStr => {
            unescape_raw_str_or_raw_byte_str(src, mode == Mode::RawByteStr, callback)
        }
        Mode::CStr | Mode::RawCStr => unreachable!(),
    }
}

/// A unit within CStr. Must not be a nul character.
pub enum CStrUnit {
    Byte(u8),
    Char(char),
}

impl From<u8> for CStrUnit {
    fn from(value: u8) -> Self {
        CStrUnit::Byte(value)
    }
}

impl From<char> for CStrUnit {
    fn from(value: char) -> Self {
        CStrUnit::Char(value)
    }
}

pub fn unescape_c_string<F>(src: &str, mode: Mode, callback: &mut F)
where
    F: FnMut(Range<usize>, Result<CStrUnit, EscapeError>),
{
    if mode == Mode::RawCStr {
        unescape_raw_str_or_raw_byte_str(
            src,
            mode.characters_should_be_ascii(),
            &mut |r, result| callback(r, result.map(CStrUnit::Char)),
        );
    } else {
        unescape_str_common(src, mode, callback);
    }
}

/// Takes a contents of a char literal (without quotes), and returns an
/// unescaped char or an error.
pub fn unescape_char(src: &str) -> Result<char, EscapeError> {
    unescape_char_or_byte(&mut src.chars(), false)
}

/// Takes a contents of a byte literal (without quotes), and returns an
/// unescaped byte or an error.
pub fn unescape_byte(src: &str) -> Result<u8, EscapeError> {
    unescape_char_or_byte(&mut src.chars(), true).map(byte_from_char)
}

/// What kind of literal do we parse.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Mode {
    Char,
    Str,
    Byte,
    ByteStr,
    RawStr,
    RawByteStr,
    CStr,
    RawCStr,
}

impl Mode {
    pub fn in_double_quotes(self) -> bool {
        match self {
            Mode::Str
            | Mode::ByteStr
            | Mode::RawStr
            | Mode::RawByteStr
            | Mode::CStr
            | Mode::RawCStr => true,
            Mode::Char | Mode::Byte => false,
        }
    }

    /// Non-byte literals should have `\xXX` escapes that are within the ASCII range.
    pub fn ascii_escapes_should_be_ascii(self) -> bool {
        match self {
            Mode::Char | Mode::Str | Mode::RawStr => true,
            Mode::Byte | Mode::ByteStr | Mode::RawByteStr | Mode::CStr | Mode::RawCStr => false,
        }
    }

    /// Whether characters within the literal must be within the ASCII range
    pub fn characters_should_be_ascii(self) -> bool {
        match self {
            Mode::Byte | Mode::ByteStr | Mode::RawByteStr => true,
            Mode::Char | Mode::Str | Mode::RawStr | Mode::CStr | Mode::RawCStr => false,
        }
    }

    /// Byte literals do not allow unicode escape.
    pub fn is_unicode_escape_disallowed(self) -> bool {
        match self {
            Mode::Byte | Mode::ByteStr | Mode::RawByteStr => true,
            Mode::Char | Mode::Str | Mode::RawStr | Mode::CStr | Mode::RawCStr => false,
        }
    }

    pub fn prefix_noraw(self) -> &'static str {
        match self {
            Mode::Byte | Mode::ByteStr | Mode::RawByteStr => "b",
            Mode::CStr | Mode::RawCStr => "c",
            Mode::Char | Mode::Str | Mode::RawStr => "",
        }
    }
}

fn scan_escape<T: From<u8> + From<char>>(
    chars: &mut Chars<'_>,
    mode: Mode,
) -> Result<T, EscapeError> {
    // Previous character was '\\', unescape what follows.
    let res = match chars.next().ok_or(EscapeError::LoneSlash)? {
        '"' => b'"',
        'n' => b'\n',
        'r' => b'\r',
        't' => b'\t',
        '\\' => b'\\',
        '\'' => b'\'',
        '0' => b'\0',

        'x' => {
            // Parse hexadecimal character code.

            let hi = chars.next().ok_or(EscapeError::TooShortHexEscape)?;
            let hi = hi.to_digit(16).ok_or(EscapeError::InvalidCharInHexEscape)?;

            let lo = chars.next().ok_or(EscapeError::TooShortHexEscape)?;
            let lo = lo.to_digit(16).ok_or(EscapeError::InvalidCharInHexEscape)?;

            let value = hi * 16 + lo;

            if mode.ascii_escapes_should_be_ascii() && !is_ascii(value) {
                return Err(EscapeError::OutOfRangeHexEscape);
            }

            value as u8
        }

        'u' => return scan_unicode(chars, mode.is_unicode_escape_disallowed()).map(Into::into),
        _ => return Err(EscapeError::InvalidEscape),
    };
    Ok(res.into())
}

fn scan_unicode(
    chars: &mut Chars<'_>,
    is_unicode_escape_disallowed: bool,
) -> Result<char, EscapeError> {
    // We've parsed '\u', now we have to parse '{..}'.

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
                if n_digits > 6 {
                    return Err(EscapeError::OverlongUnicodeEscape);
                }

                // Incorrect syntax has higher priority for error reporting
                // than unallowed value for a literal.
                if is_unicode_escape_disallowed {
                    return Err(EscapeError::UnicodeEscapeInByte);
                }

                break std::char::from_u32(value).ok_or_else(|| {
                    if value > 0x10FFFF {
                        EscapeError::OutOfRangeUnicodeEscape
                    } else {
                        EscapeError::LoneSurrogateUnicodeEscape
                    }
                });
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

#[inline]
fn ascii_check(c: char, characters_should_be_ascii: bool) -> Result<char, EscapeError> {
    if characters_should_be_ascii && !c.is_ascii() {
        // Byte literal can't be a non-ascii character.
        Err(EscapeError::NonAsciiCharInByte)
    } else {
        Ok(c)
    }
}

fn unescape_char_or_byte(chars: &mut Chars<'_>, is_byte: bool) -> Result<char, EscapeError> {
    let c = chars.next().ok_or(EscapeError::ZeroChars)?;
    let res = match c {
        '\\' => scan_escape(chars, if is_byte { Mode::Byte } else { Mode::Char }),
        '\n' | '\t' | '\'' => Err(EscapeError::EscapeOnlyChar),
        '\r' => Err(EscapeError::BareCarriageReturn),
        _ => ascii_check(c, is_byte),
    }?;
    if chars.next().is_some() {
        return Err(EscapeError::MoreThanOneChar);
    }
    Ok(res)
}

/// Takes a contents of a string literal (without quotes) and produces a
/// sequence of escaped characters or errors.
fn unescape_str_common<F, T: From<u8> + From<char>>(src: &str, mode: Mode, callback: &mut F)
where
    F: FnMut(Range<usize>, Result<T, EscapeError>),
{
    let mut chars = src.chars();

    // The `start` and `end` computation here is complicated because
    // `skip_ascii_whitespace` makes us to skip over chars without counting
    // them in the range computation.
    while let Some(c) = chars.next() {
        let start = src.len() - chars.as_str().len() - c.len_utf8();
        let res = match c {
            '\\' => {
                match chars.clone().next() {
                    Some('\n') => {
                        // Rust language specification requires us to skip whitespaces
                        // if unescaped '\' character is followed by '\n'.
                        // For details see [Rust language reference]
                        // (https://doc.rust-lang.org/reference/tokens.html#string-literals).
                        skip_ascii_whitespace(&mut chars, start, &mut |range, err| {
                            callback(range, Err(err))
                        });
                        continue;
                    }
                    _ => scan_escape::<T>(&mut chars, mode),
                }
            }
            '\n' => Ok(b'\n'.into()),
            '\t' => Ok(b'\t'.into()),
            '"' => Err(EscapeError::EscapeOnlyChar),
            '\r' => Err(EscapeError::BareCarriageReturn),
            _ => ascii_check(c, mode.characters_should_be_ascii()).map(Into::into),
        };
        let end = src.len() - chars.as_str().len();
        callback(start..end, res.map(Into::into));
    }
}

fn skip_ascii_whitespace<F>(chars: &mut Chars<'_>, start: usize, callback: &mut F)
where
    F: FnMut(Range<usize>, EscapeError),
{
    let tail = chars.as_str();
    let first_non_space = tail
        .bytes()
        .position(|b| b != b' ' && b != b'\t' && b != b'\n' && b != b'\r')
        .unwrap_or(tail.len());
    if tail[1..first_non_space].contains('\n') {
        // The +1 accounts for the escaping slash.
        let end = start + first_non_space + 1;
        callback(start..end, EscapeError::MultipleSkippedLinesWarning);
    }
    let tail = &tail[first_non_space..];
    if let Some(c) = tail.chars().next() {
        if c.is_whitespace() {
            // For error reporting, we would like the span to contain the character that was not
            // skipped. The +1 is necessary to account for the leading \ that started the escape.
            let end = start + first_non_space + c.len_utf8() + 1;
            callback(start..end, EscapeError::UnskippedWhitespaceWarning);
        }
    }
    *chars = tail.chars();
}

/// Takes a contents of a string literal (without quotes) and produces a
/// sequence of characters or errors.
/// NOTE: Raw strings do not perform any explicit character escaping, here we
/// only produce errors on bare CR.
fn unescape_raw_str_or_raw_byte_str<F>(src: &str, is_byte: bool, callback: &mut F)
where
    F: FnMut(Range<usize>, Result<char, EscapeError>),
{
    let mut chars = src.chars();

    // The `start` and `end` computation here matches the one in
    // `unescape_str_or_byte_str` for consistency, even though this function
    // doesn't have to worry about skipping any chars.
    while let Some(c) = chars.next() {
        let start = src.len() - chars.as_str().len() - c.len_utf8();
        let res = match c {
            '\r' => Err(EscapeError::BareCarriageReturnInRawString),
            _ => ascii_check(c, is_byte),
        };
        let end = src.len() - chars.as_str().len();
        callback(start..end, res);
    }
}

#[inline]
pub fn byte_from_char(c: char) -> u8 {
    let res = c as u32;
    debug_assert!(res <= u8::MAX as u32, "guaranteed because of Mode::ByteStr");
    res as u8
}

fn is_ascii(x: u32) -> bool {
    x <= 0x7F
}
