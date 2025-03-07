//! Utilities for validating string and char literals and turning them into
//! values they represent.

use std::num::NonZero;
use std::ops::Range;
use std::str::Chars;

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
    /// Used for ASCII chars (written directly or via `\x01`..`\x7f` escapes)
    /// and Unicode chars (written directly or via `\u` escapes).
    ///
    /// For example, if '¥' appears in a string it is represented here as
    /// `MixedUnit::Char('¥')`, and it will be appended to the relevant byte
    /// string as the two-byte UTF-8 sequence `[0xc2, 0xa5]`
    Char(NonZero<char>),

    /// Used for high bytes (`\x80`..`\xff`).
    ///
    /// For example, if `\xa5` appears in a string it is represented here as
    /// `MixedUnit::HighByte(0xa5)`, and it will be appended to the relevant
    /// byte string as the single byte `0xa5`.
    HighByte(NonZero<u8>),
}

impl From<NonZero<char>> for MixedUnit {
    fn from(c: NonZero<char>) -> Self {
        MixedUnit::Char(c)
    }
}

impl From<NonZero<u8>> for MixedUnit {
    fn from(byte: NonZero<u8>) -> Self {
        if byte.get().is_ascii() {
            MixedUnit::Char(NonZero::new(byte.get() as char).unwrap())
        } else {
            MixedUnit::HighByte(byte)
        }
    }
}
impl TryFrom<char> for MixedUnit {
    type Error = EscapeError;

    fn try_from(c: char) -> Result<Self, EscapeError> {
        NonZero::new(c).map(MixedUnit::Char).ok_or(EscapeError::NulInCStr)
    }
}

impl TryFrom<u8> for MixedUnit {
    type Error = EscapeError;

    fn try_from(byte: u8) -> Result<Self, EscapeError> {
        NonZero::<u8>::new(byte).map(From::from).ok_or(EscapeError::NulInCStr)
    }
}

macro_rules! check {
    ($string_ty:literal
     ($check:ident: $char2unit:expr => $unit:ty)) => {
        #[doc = concat!("Take the contents of a raw ", stringify!($string_ty),
                        " literal (without quotes) and produce a sequence of results of ",
                        stringify!($unit_ty), " or error (returned via `callback`).",
                        "\nNB: Raw strings don't do any unescaping, but do produce errors on bare CR.")]
        pub fn $check(src: &str, callback: &mut impl FnMut(Range<usize>, Result<$unit, EscapeError>))
        {
            src.char_indices().for_each(|(pos, c)| {
                callback(
                    pos..pos + c.len_utf8(),
                    if c == '\r' { Err(EscapeError::BareCarriageReturnInRawString) } else { $char2unit(c) },
                );
            });
        }
    };
}

check!("string" (check_raw_str: Ok => char));
check!("byte string" (check_raw_byte_str: ascii_char_to_byte => u8));
check!("C string" (check_raw_cstr: |c| NonZero::<char>::new(c).ok_or(EscapeError::NulInCStr) => NonZero<char>));

macro_rules! unescape {
    ($string_ty:literal
     ($unescape:ident: $char2unit:expr => $unit:ty)
     $scan_escape:ident) => {
        #[doc = concat!("Take the contents of a ", stringify!($string_ty),
                        " literal (without quotes) and produce a sequence of results of escaped ",
                        stringify!($unit_ty), " or error (returned via `callback`).")]
        pub fn $unescape(src: &str, callback: &mut impl FnMut(Range<usize>, Result<$unit, EscapeError>))
        {
            let mut chars = src.chars();
            while let Some(c) = chars.next() {
                let start = src.len() - chars.as_str().len() - c.len_utf8();
                let res = match c {
                    '\\' => {
                        if let Some(b'\n') = chars.as_str().as_bytes().first() {
                            let _ = chars.next();
                            // skip whitespace for backslash newline, see [Rust language reference]
                            // (https://doc.rust-lang.org/reference/tokens.html#string-literals).
                            let mut callback_err = |range, err| callback(range, Err(err));
                            skip_ascii_whitespace(&mut chars, start, &mut callback_err);
                            continue;
                        } else {
                            $scan_escape(&mut chars)
                        }
                    }
                    '"' => Err(EscapeError::EscapeOnlyChar),
                    '\r' => Err(EscapeError::BareCarriageReturn),
                    c => $char2unit(c),
                };
                let end = src.len() - chars.as_str().len();
                callback(start..end, res);
            }
        }
    };
}

unescape!("string" (unescape_str: Ok => char) scan_escape_str);
unescape!("byte string" (unescape_byte_str: ascii_char_to_byte => u8) scan_escape_byte_str);
unescape!("C string" (unescape_cstr: TryFrom::try_from => MixedUnit) scan_escape_c_str);

/// Skip ASCII whitespace, except for the formfeed character
/// (see [this issue](https://github.com/rust-lang/rust/issues/136600)).
/// Warns on unescaped newline and following non-ASCII whitespace.
fn skip_ascii_whitespace<F>(chars: &mut Chars<'_>, start: usize, callback: &mut F)
where
    F: FnMut(Range<usize>, EscapeError),
{
    let rest = chars.as_str();
    let first_non_space = rest
        .bytes()
        .position(|b| b != b' ' && b != b'\t' && b != b'\n' && b != b'\r')
        .unwrap_or(rest.len());
    let (space, rest) = rest.split_at(first_non_space);
    // backslash newline adds 2 bytes
    let end = start + 2 + first_non_space;
    if space.contains('\n') {
        callback(start..end, EscapeError::MultipleSkippedLinesWarning);
    }
    *chars = rest.chars();
    if let Some(c) = chars.clone().next() {
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

/// Takes the contents of a byte literal (without quotes),
/// and returns an unescaped byte or an error.
pub fn unescape_byte(src: &str) -> Result<u8, EscapeError> {
    unescape_byte_iter(&mut src.chars())
}

macro_rules! unescape_iter {
    (($unescape:ident: $char2unit:expr => $unit:ty) $scan_escape:ident) => {
        fn $unescape(chars: &mut Chars<'_>) -> Result<$unit, EscapeError> {
            let res = match chars.next().ok_or(EscapeError::ZeroChars)? {
                '\\' => $scan_escape(chars),
                '\n' | '\t' | '\'' => Err(EscapeError::EscapeOnlyChar),
                '\r' => Err(EscapeError::BareCarriageReturn),
                c => $char2unit(c),
            }?;
            if chars.next().is_some() {
                return Err(EscapeError::MoreThanOneChar);
            }
            Ok(res)
        }
    };
}

unescape_iter!((unescape_char_iter: Ok => char) scan_escape_str);
unescape_iter!((unescape_byte_iter: ascii_char_to_byte => u8) scan_escape_byte_str);

macro_rules! scan_escape {
    ($scan:ident: $zero_result:expr, $from_hex:expr, $from_unicode:expr => $unit:ty) => {
        fn $scan(chars: &mut Chars<'_>) -> Result<$unit, EscapeError> {
            // Previous character was '\\', unescape what follows.
            let c = chars.next().ok_or(EscapeError::LoneSlash)?;
            if c == '0' {
                $zero_result
            } else {
                simple_escape(c).map(|b| b.get().try_into().unwrap()).or_else(|c| match c {
                    'x' => $from_hex(hex_escape(chars)?),
                    'u' => $from_unicode({
                        let value = unicode_escape(chars)?;
                        if value > char::MAX as u32 {
                            Err(EscapeError::OutOfRangeUnicodeEscape)
                        } else {
                            char::from_u32(value).ok_or(EscapeError::LoneSurrogateUnicodeEscape)
                        }
                    }),
                    _ => Err(EscapeError::InvalidEscape),
                })
            }
        }
    };
}

scan_escape!(scan_escape_str: Ok('\0'), char_from_byte, |id| id => char);
scan_escape!(scan_escape_byte_str: Ok(b'\0'), Ok, |_| Err(EscapeError::UnicodeEscapeInByte) => u8);
scan_escape!(scan_escape_c_str: Err(EscapeError::NulInCStr), TryInto::try_into, |r: Result<char, _>| r?.try_into() => MixedUnit);

fn char_from_byte(b: u8) -> Result<char, EscapeError> {
    if b.is_ascii() { Ok(b as char) } else { Err(EscapeError::OutOfRangeHexEscape) }
}

/// Parse the character of an ASCII escape (except nul) without the leading backslash.
fn simple_escape(c: char) -> Result<NonZero<u8>, char> {
    // Previous character was '\\', unescape what follows.
    Ok(NonZero::new(match c {
        '"' => b'"',
        'n' => b'\n',
        'r' => b'\r',
        't' => b'\t',
        '\\' => b'\\',
        '\'' => b'\'',
        _ => Err(c)?,
    })
    .unwrap())
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
    let mut value: u32 = match chars.next().ok_or(EscapeError::UnclosedUnicodeEscape)? {
        '_' => return Err(EscapeError::LeadingUnderscoreUnicodeEscape),
        '}' => return Err(EscapeError::EmptyUnicodeEscape),
        c => c.to_digit(16).ok_or(EscapeError::InvalidCharInUnicodeEscape)?,
    };

    // First character is valid, now parse the rest of the number
    // and closing brace.
    let mut n_digits = 1;
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
        RawCStr => check_raw_cstr(src, &mut |r, res: Result<NonZero<char>, EscapeError>| {
            callback(r, res.map(|c| c.get()))
        }),
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
