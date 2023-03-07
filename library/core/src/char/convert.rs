//! Character conversions.

use crate::char::TryFromCharError;
use crate::convert::TryFrom;
use crate::error::Error;
use crate::fmt;
use crate::mem::transmute;
use crate::str::FromStr;

/// Converts a `u32` to a `char`. See [`char::from_u32`].
#[must_use]
#[inline]
pub(super) const fn from_u32(i: u32) -> Option<char> {
    // FIXME: once Result::ok is const fn, use it here
    match char_try_from_u32(i) {
        Ok(c) => Some(c),
        Err(_) => None,
    }
}

/// Converts a `u32` to a `char`, ignoring validity. See [`char::from_u32_unchecked`].
#[inline]
#[must_use]
pub(super) const unsafe fn from_u32_unchecked(i: u32) -> char {
    // SAFETY: the caller must guarantee that `i` is a valid char value.
    if cfg!(debug_assertions) { char::from_u32(i).unwrap() } else { unsafe { transmute(i) } }
}

#[stable(feature = "char_convert", since = "1.13.0")]
#[rustc_const_unstable(feature = "const_convert", issue = "88674")]
impl const From<char> for u32 {
    /// Converts a [`char`] into a [`u32`].
    ///
    /// # Examples
    ///
    /// ```
    /// use std::mem;
    ///
    /// let c = 'c';
    /// let u = u32::from(c);
    /// assert!(4 == mem::size_of_val(&u))
    /// ```
    #[inline]
    fn from(c: char) -> Self {
        c as u32
    }
}

#[stable(feature = "more_char_conversions", since = "1.51.0")]
#[rustc_const_unstable(feature = "const_convert", issue = "88674")]
impl const From<char> for u64 {
    /// Converts a [`char`] into a [`u64`].
    ///
    /// # Examples
    ///
    /// ```
    /// use std::mem;
    ///
    /// let c = '👤';
    /// let u = u64::from(c);
    /// assert!(8 == mem::size_of_val(&u))
    /// ```
    #[inline]
    fn from(c: char) -> Self {
        // The char is casted to the value of the code point, then zero-extended to 64 bit.
        // See [https://doc.rust-lang.org/reference/expressions/operator-expr.html#semantics]
        c as u64
    }
}

#[stable(feature = "more_char_conversions", since = "1.51.0")]
#[rustc_const_unstable(feature = "const_convert", issue = "88674")]
impl const From<char> for u128 {
    /// Converts a [`char`] into a [`u128`].
    ///
    /// # Examples
    ///
    /// ```
    /// use std::mem;
    ///
    /// let c = '⚙';
    /// let u = u128::from(c);
    /// assert!(16 == mem::size_of_val(&u))
    /// ```
    #[inline]
    fn from(c: char) -> Self {
        // The char is casted to the value of the code point, then zero-extended to 128 bit.
        // See [https://doc.rust-lang.org/reference/expressions/operator-expr.html#semantics]
        c as u128
    }
}

/// Map `char` with code point in U+0000..=U+00FF to byte in 0x00..=0xFF with same value, failing
/// if the code point is greater than U+00FF.
///
/// See [`impl From<u8> for char`](char#impl-From<u8>-for-char) for details on the encoding.
#[stable(feature = "u8_from_char", since = "1.59.0")]
impl TryFrom<char> for u8 {
    type Error = TryFromCharError;

    #[inline]
    fn try_from(c: char) -> Result<u8, Self::Error> {
        u8::try_from(u32::from(c)).map_err(|_| TryFromCharError(()))
    }
}

/// Maps a byte in 0x00..=0xFF to a `char` whose code point has the same value, in U+0000..=U+00FF.
///
/// Unicode is designed such that this effectively decodes bytes
/// with the character encoding that IANA calls ISO-8859-1.
/// This encoding is compatible with ASCII.
///
/// Note that this is different from ISO/IEC 8859-1 a.k.a. ISO 8859-1 (with one less hyphen),
/// which leaves some "blanks", byte values that are not assigned to any character.
/// ISO-8859-1 (the IANA one) assigns them to the C0 and C1 control codes.
///
/// Note that this is *also* different from Windows-1252 a.k.a. code page 1252,
/// which is a superset ISO/IEC 8859-1 that assigns some (not all!) blanks
/// to punctuation and various Latin characters.
///
/// To confuse things further, [on the Web](https://encoding.spec.whatwg.org/)
/// `ascii`, `iso-8859-1`, and `windows-1252` are all aliases
/// for a superset of Windows-1252 that fills the remaining blanks with corresponding
/// C0 and C1 control codes.
#[stable(feature = "char_convert", since = "1.13.0")]
#[rustc_const_unstable(feature = "const_convert", issue = "88674")]
impl const From<u8> for char {
    /// Converts a [`u8`] into a [`char`].
    ///
    /// # Examples
    ///
    /// ```
    /// use std::mem;
    ///
    /// let u = 32 as u8;
    /// let c = char::from(u);
    /// assert!(4 == mem::size_of_val(&c))
    /// ```
    #[inline]
    fn from(i: u8) -> Self {
        i as char
    }
}

/// An error which can be returned when parsing a char.
///
/// This `struct` is created when using the [`char::from_str`] method.
#[stable(feature = "char_from_str", since = "1.20.0")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParseCharError {
    kind: CharErrorKind,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum CharErrorKind {
    EmptyString,
    TooManyChars,
}

#[stable(feature = "char_from_str", since = "1.20.0")]
impl Error for ParseCharError {
    #[allow(deprecated)]
    fn description(&self) -> &str {
        match self.kind {
            CharErrorKind::EmptyString => "cannot parse char from empty string",
            CharErrorKind::TooManyChars => "too many characters in string",
        }
    }
}

#[stable(feature = "char_from_str", since = "1.20.0")]
impl fmt::Display for ParseCharError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        #[allow(deprecated)]
        self.description().fmt(f)
    }
}

#[stable(feature = "char_from_str", since = "1.20.0")]
impl FromStr for char {
    type Err = ParseCharError;

    #[inline]
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut chars = s.chars();
        match (chars.next(), chars.next()) {
            (None, _) => Err(ParseCharError { kind: CharErrorKind::EmptyString }),
            (Some(c), None) => Ok(c),
            _ => Err(ParseCharError { kind: CharErrorKind::TooManyChars }),
        }
    }
}

#[inline]
const fn char_try_from_u32(i: u32) -> Result<char, CharTryFromError> {
    // This is an optimized version of the check
    // (i > MAX as u32) || (i >= 0xD800 && i <= 0xDFFF),
    // which can also be written as
    // i >= 0x110000 || (i >= 0xD800 && i < 0xE000).
    //
    // The XOR with 0xD800 permutes the ranges such that 0xD800..0xE000 is
    // mapped to 0x0000..0x0800, while keeping all the high bits outside 0xFFFF the same.
    // In particular, numbers >= 0x110000 stay in this range.
    //
    // Subtracting 0x800 causes 0x0000..0x0800 to wrap, meaning that a single
    // unsigned comparison against 0x110000 - 0x800 will detect both the wrapped
    // surrogate range as well as the numbers originally larger than 0x110000.
    //
    if (i ^ 0xD800).wrapping_sub(0x800) >= 0x110000 - 0x800 {
        Err(CharTryFromError(()))
    } else {
        // SAFETY: checked that it's a legal unicode value
        Ok(unsafe { transmute(i) })
    }
}

#[stable(feature = "try_from", since = "1.34.0")]
impl TryFrom<u32> for char {
    type Error = CharTryFromError;

    #[inline]
    fn try_from(i: u32) -> Result<Self, Self::Error> {
        char_try_from_u32(i)
    }
}

/// The error type returned when a conversion from [`prim@u32`] to [`prim@char`] fails.
///
/// This `struct` is created by the [`char::try_from<u32>`](char#impl-TryFrom<u32>-for-char) method.
/// See its documentation for more.
#[stable(feature = "try_from", since = "1.34.0")]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct CharTryFromError(());

#[stable(feature = "try_from", since = "1.34.0")]
impl fmt::Display for CharTryFromError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        "converted integer out of range for `char`".fmt(f)
    }
}

/// Converts a digit in the given radix to a `char`. See [`char::from_digit`].
#[inline]
#[must_use]
pub(super) const fn from_digit(num: u32, radix: u32) -> Option<char> {
    if radix > 36 {
        panic!("from_digit: radix is too high (maximum 36)");
    }
    if num < radix {
        let num = num as u8;
        if num < 10 { Some((b'0' + num) as char) } else { Some((b'a' + num - 10) as char) }
    } else {
        None
    }
}
