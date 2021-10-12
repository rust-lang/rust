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
    /// Non-ascii character in byte literal.
    NonAsciiCharInByte,
    /// Non-ascii character in byte string literal.
    NonAsciiCharInByteString,

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
pub fn unescape_literal<F>(literal_text: &str, mode: Mode, callback: &mut F)
where
    F: FnMut(Range<usize>, Result<char, EscapeError>),
{
    match mode {
        Mode::Char | Mode::Byte => {
            let mut chars = literal_text.chars();
            let result = unescape_char_or_byte(&mut chars, mode);
            // The Chars iterator moved forward.
            callback(0..(literal_text.len() - chars.as_str().len()), result);
        }
        Mode::Str | Mode::ByteStr => unescape_str_or_byte_str(literal_text, mode, callback),
        // NOTE: Raw strings do not perform any explicit character escaping, here we
        // only translate CRLF to LF and produce errors on bare CR.
        Mode::RawStr | Mode::RawByteStr => {
            unescape_raw_str_or_byte_str(literal_text, mode, callback)
        }
    }
}

/// Takes a contents of a byte, byte string or raw byte string (without quotes)
/// and produces a sequence of bytes or errors.
/// Values are returned through invoking of the provided callback.
pub fn unescape_byte_literal<F>(literal_text: &str, mode: Mode, callback: &mut F)
where
    F: FnMut(Range<usize>, Result<u8, EscapeError>),
{
    assert!(mode.is_bytes());
    unescape_literal(literal_text, mode, &mut |range, result| {
        callback(range, result.map(byte_from_char));
    })
}

/// Takes a contents of a char literal (without quotes), and returns an
/// unescaped char or an error
pub fn unescape_char(literal_text: &str) -> Result<char, (usize, EscapeError)> {
    let mut chars = literal_text.chars();
    unescape_char_or_byte(&mut chars, Mode::Char)
        .map_err(|err| (literal_text.len() - chars.as_str().len(), err))
}

/// Takes a contents of a byte literal (without quotes), and returns an
/// unescaped byte or an error.
pub fn unescape_byte(literal_text: &str) -> Result<u8, (usize, EscapeError)> {
    let mut chars = literal_text.chars();
    unescape_char_or_byte(&mut chars, Mode::Byte)
        .map(byte_from_char)
        .map_err(|err| (literal_text.len() - chars.as_str().len(), err))
}

/// What kind of literal do we parse.
#[derive(Debug, Clone, Copy)]
pub enum Mode {
    Char,
    Str,
    Byte,
    ByteStr,
    RawStr,
    RawByteStr,
}

impl Mode {
    pub fn in_single_quotes(self) -> bool {
        match self {
            Mode::Char | Mode::Byte => true,
            Mode::Str | Mode::ByteStr | Mode::RawStr | Mode::RawByteStr => false,
        }
    }

    pub fn in_double_quotes(self) -> bool {
        !self.in_single_quotes()
    }

    pub fn is_bytes(self) -> bool {
        match self {
            Mode::Byte | Mode::ByteStr | Mode::RawByteStr => true,
            Mode::Char | Mode::Str | Mode::RawStr => false,
        }
    }
}

fn scan_escape(first_char: char, chars: &mut Chars<'_>, mode: Mode) -> Result<char, EscapeError> {
    if first_char != '\\' {
        // Previous character was not a slash, and we don't expect it to be
        // an escape-only character.
        return match first_char {
            '\t' | '\n' => Err(EscapeError::EscapeOnlyChar),
            '\r' => Err(EscapeError::BareCarriageReturn),
            '\'' if mode.in_single_quotes() => Err(EscapeError::EscapeOnlyChar),
            '"' if mode.in_double_quotes() => Err(EscapeError::EscapeOnlyChar),
            _ => {
                if mode.is_bytes() && !first_char.is_ascii() {
                    // Byte literal can't be a non-ascii character.
                    return Err(EscapeError::NonAsciiCharInByte);
                }
                Ok(first_char)
            }
        };
    }

    // Previous character is '\\', try to unescape it.

    let second_char = chars.next().ok_or(EscapeError::LoneSlash)?;

    let res = match second_char {
        '"' => '"',
        'n' => '\n',
        'r' => '\r',
        't' => '\t',
        '\\' => '\\',
        '\'' => '\'',
        '0' => '\0',

        'x' => {
            // Parse hexadecimal character code.

            let hi = chars.next().ok_or(EscapeError::TooShortHexEscape)?;
            let hi = hi.to_digit(16).ok_or(EscapeError::InvalidCharInHexEscape)?;

            let lo = chars.next().ok_or(EscapeError::TooShortHexEscape)?;
            let lo = lo.to_digit(16).ok_or(EscapeError::InvalidCharInHexEscape)?;

            let value = hi * 16 + lo;

            // For a byte literal verify that it is within ASCII range.
            if !mode.is_bytes() && !is_ascii(value) {
                return Err(EscapeError::OutOfRangeHexEscape);
            }
            let value = value as u8;

            value as char
        }

        'u' => {
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
                        if mode.is_bytes() {
                            return Err(EscapeError::UnicodeEscapeInByte);
                        }

                        break std::char::from_u32(value).ok_or_else(|| {
                            if value > 0x10FFFF {
                                EscapeError::OutOfRangeUnicodeEscape
                            } else {
                                EscapeError::LoneSurrogateUnicodeEscape
                            }
                        })?;
                    }
                    Some(c) => {
                        let digit =
                            c.to_digit(16).ok_or(EscapeError::InvalidCharInUnicodeEscape)?;
                        n_digits += 1;
                        if n_digits > 6 {
                            // Stop updating value since we're sure that it's is incorrect already.
                            continue;
                        }
                        let digit = digit as u32;
                        value = value * 16 + digit;
                    }
                };
            }
        }
        _ => return Err(EscapeError::InvalidEscape),
    };
    Ok(res)
}

fn unescape_char_or_byte(chars: &mut Chars<'_>, mode: Mode) -> Result<char, EscapeError> {
    let first_char = chars.next().ok_or(EscapeError::ZeroChars)?;
    let res = scan_escape(first_char, chars, mode)?;
    if chars.next().is_some() {
        return Err(EscapeError::MoreThanOneChar);
    }
    Ok(res)
}

/// Takes a contents of a string literal (without quotes) and produces a
/// sequence of escaped characters or errors.
fn unescape_str_or_byte_str<F>(src: &str, mode: Mode, callback: &mut F)
where
    F: FnMut(Range<usize>, Result<char, EscapeError>),
{
    assert!(mode.in_double_quotes());
    let initial_len = src.len();
    let mut chars = src.chars();
    while let Some(first_char) = chars.next() {
        let start = initial_len - chars.as_str().len() - first_char.len_utf8();

        let unescaped_char = match first_char {
            '\\' => {
                let second_char = chars.clone().next();
                match second_char {
                    Some('\n') => {
                        // Rust language specification requires us to skip whitespaces
                        // if unescaped '\' character is followed by '\n'.
                        // For details see [Rust language reference]
                        // (https://doc.rust-lang.org/reference/tokens.html#string-literals).
                        skip_ascii_whitespace(&mut chars, start, callback);
                        continue;
                    }
                    _ => scan_escape(first_char, &mut chars, mode),
                }
            }
            '\n' => Ok('\n'),
            '\t' => Ok('\t'),
            _ => scan_escape(first_char, &mut chars, mode),
        };
        let end = initial_len - chars.as_str().len();
        callback(start..end, unescaped_char);
    }

    fn skip_ascii_whitespace<F>(chars: &mut Chars<'_>, start: usize, callback: &mut F)
    where
        F: FnMut(Range<usize>, Result<char, EscapeError>),
    {
        let tail = chars.as_str();
        let first_non_space = tail
            .bytes()
            .position(|b| b != b' ' && b != b'\t' && b != b'\n' && b != b'\r')
            .unwrap_or(tail.len());
        if tail[1..first_non_space].contains('\n') {
            // The +1 accounts for the escaping slash.
            let end = start + first_non_space + 1;
            callback(start..end, Err(EscapeError::MultipleSkippedLinesWarning));
        }
        let tail = &tail[first_non_space..];
        if let Some(c) = tail.chars().next() {
            // For error reporting, we would like the span to contain the character that was not
            // skipped.  The +1 is necessary to account for the leading \ that started the escape.
            let end = start + first_non_space + c.len_utf8() + 1;
            if c.is_whitespace() {
                callback(start..end, Err(EscapeError::UnskippedWhitespaceWarning));
            }
        }
        *chars = tail.chars();
    }
}

/// Takes a contents of a string literal (without quotes) and produces a
/// sequence of characters or errors.
/// NOTE: Raw strings do not perform any explicit character escaping, here we
/// only translate CRLF to LF and produce errors on bare CR.
fn unescape_raw_str_or_byte_str<F>(literal_text: &str, mode: Mode, callback: &mut F)
where
    F: FnMut(Range<usize>, Result<char, EscapeError>),
{
    assert!(mode.in_double_quotes());
    let initial_len = literal_text.len();

    let mut chars = literal_text.chars();
    while let Some(curr) = chars.next() {
        let start = initial_len - chars.as_str().len() - curr.len_utf8();

        let result = match curr {
            '\r' => Err(EscapeError::BareCarriageReturnInRawString),
            c if mode.is_bytes() && !c.is_ascii() => Err(EscapeError::NonAsciiCharInByteString),
            c => Ok(c),
        };
        let end = initial_len - chars.as_str().len();

        callback(start..end, result);
    }
}

fn byte_from_char(c: char) -> u8 {
    let res = c as u32;
    assert!(res <= u8::MAX as u32, "guaranteed because of Mode::ByteStr");
    res as u8
}

fn is_ascii(x: u32) -> bool {
    x <= 0x7F
}
