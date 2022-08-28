//! Implementation of UCS-2 encoding. It is similar to UTF-16, but without surrogate pairs. Used by
//! UEFI.
#![allow(dead_code)]

use crate::fmt;
use crate::iter::FusedIterator;
use crate::num::NonZeroU16;
use crate::str;

/// A struct to represent UCS-2 character
#[unstable(feature = "ucs2", issue = "none")]
#[repr(transparent)]
#[derive(PartialEq, Eq, PartialOrd, Ord, Copy, Clone)]
pub struct Ucs2Char {
    value: NonZeroU16,
}

/// Format the code point as `U+` followed by four to six hexadecimal digits.
/// Example: `U+1F4A9`
#[unstable(feature = "ucs2", issue = "none")]
impl fmt::Debug for Ucs2Char {
    #[inline]
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "U+{:04X}", self.value)
    }
}

impl Ucs2Char {
    #[unstable(feature = "ucs2", issue = "none")]
    pub const REPLACEMENT_CHARACTER: Ucs2Char =
        Ucs2Char { value: NonZeroU16::new(0xfffdu16).unwrap() };

    pub(crate) const CR: Self = Ucs2Char { value: NonZeroU16::new(0x000du16).unwrap() };
    pub(crate) const LF: Self = Ucs2Char { value: NonZeroU16::new(0x000au16).unwrap() };

    fn new(value: NonZeroU16) -> Self {
        Self { value }
    }

    #[unstable(feature = "ucs2", issue = "none")]
    pub fn from_u16(c: u16) -> Option<Self> {
        match c {
            // UEFI does not support surrogate characters.
            0xd800..=0xdfff => None,
            _ => Some(Self::new(NonZeroU16::new(c)?)),
        }
    }

    #[unstable(feature = "ucs2", issue = "none")]
    pub fn from_char(ch: char) -> Option<Self> {
        Ucs2Char::from_u16(u16::try_from(ch as u32).ok()?)
    }

    /// Get the number of bytes that will be needed to represent this character in UTF-8. It can be
    /// 1, 2 or 3
    #[unstable(feature = "ucs2", issue = "none")]
    pub fn len_utf8(&self) -> usize {
        char::from(*self).len_utf8()
    }
}

#[unstable(feature = "ucs2", issue = "none")]
impl From<Ucs2Char> for u16 {
    #[inline]
    fn from(c: Ucs2Char) -> Self {
        u16::from(c.value)
    }
}

#[unstable(feature = "ucs2", issue = "none")]
impl From<Ucs2Char> for char {
    fn from(c: Ucs2Char) -> Self {
        unsafe { char::from_u32_unchecked(c.value.get() as u32) }
    }
}

#[unstable(feature = "ucs2", issue = "none")]
pub struct EncodeUcs2<'a> {
    utf8_buf: str::Chars<'a>,
}

impl<'a> EncodeUcs2<'a> {
    #[unstable(feature = "ucs2", issue = "none")]
    #[inline]
    pub fn from_str(s: &'a str) -> Self {
        Self { utf8_buf: s.chars() }
    }

    // Returns error if slice of bytes is not valid UTF-8
    #[unstable(feature = "ucs2", issue = "none")]
    #[inline]
    pub fn from_bytes(s: &'a [u8]) -> Result<Self, str::Utf8Error> {
        Ok(Self::from_str(str::from_utf8(s)?))
    }

    #[unstable(feature = "ucs2", issue = "none")]
    #[inline]
    pub unsafe fn from_bytes_unchecked(s: &'a [u8]) -> Self {
        Self::from_str(str::from_utf8_unchecked(s))
    }
}

#[unstable(feature = "ucs2", issue = "none")]
pub struct EncodeError(char);

#[unstable(feature = "ucs2", issue = "none")]
impl fmt::Debug for EncodeError {
    #[inline]
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "Char {} cannot be converted to UCS-2", self.0)
    }
}

#[unstable(feature = "ucs2", issue = "none")]
impl<'a> Iterator for EncodeUcs2<'a> {
    type Item = Result<u16, EncodeError>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(ch) = self.utf8_buf.next() {
            match Ucs2Char::from_char(ch) {
                Some(x) => Some(Ok(u16::from(x))),
                None => Some(Err(EncodeError(ch))),
            }
        } else {
            None
        }
    }
}

#[unstable(feature = "ucs2", issue = "none")]
impl<'a> FusedIterator for EncodeUcs2<'a> {}
