//! Implementating of UCS-2 encoding. This is a lesser known subset of UTF-16 without surrogate
//! pairs. It is used in UEFI
#![allow(dead_code)]

use crate::fmt;
use crate::iter::FusedIterator;
use crate::marker::PhantomData;
use crate::num::NonZeroU16;
use crate::ptr::NonNull;
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
    #[inline]
    pub fn from_u16(c: u16) -> Option<Self> {
        match NonZeroU16::try_from(c) {
            Ok(x) => Some(Ucs2Char::new(x)),
            Err(_) => None,
        }
    }

    #[unstable(feature = "ucs2", issue = "none")]
    pub fn from_char(ch: char) -> Option<Self> {
        let mut buf = [0u8; 4];
        ch.encode_utf8(&mut buf);

        match ch.len_utf8() {
            1 => Ucs2Char::from_u16(u16::from(buf[0])),
            2 => {
                let a = u16::from(buf[0] & 0b0001_1111);
                let b = u16::from(buf[1] & 0b0011_1111);
                Ucs2Char::from_u16(a << 6 | b)
            }
            3 => {
                let a = u16::from(buf[0] & 0b0000_1111);
                let b = u16::from(buf[1] & 0b0011_1111);
                let c = u16::from(buf[2] & 0b0011_1111);
                Ucs2Char::from_u16(a << 12 | b << 6 | c)
            }
            4 => None,
            _ => unreachable!(),
        }
    }

    /// Get the number of bytes that will be needed to represent this character in UTF-8. It can be
    /// 1, 2 or 3
    #[unstable(feature = "ucs2", issue = "none")]
    #[inline]
    pub fn len_utf8(&self) -> usize {
        match u16::from(self.value) {
            0b0000_0000_0000_0000..0b0000_0000_0111_1111 => 1,
            0b0000_0000_0111_1111..0b0000_0111_1111_1111 => 2,
            _ => 3,
        }
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
        let val = u16::from(c);
        match val {
            0b0000_0000_0000_0000..0b0000_0000_0111_1111 => {
                // 1-byte
                unsafe { char::from_u32_unchecked(u32::from(val)) }
            }
            0b0000_0000_0111_1111..0b0000_0111_1111_1111 => {
                // 2-byte
                let high = ((val & 0b0000_0111_1100_0000) << 2) | 0b1100_0000_0000_0000;
                let low = (val & 0b0000_0000_0011_1111) | 0b0000_0000_1000_0000;

                unsafe { char::from_u32_unchecked(u32::from(high | low)) }
            }
            _ => {
                // 3-byte
                let ch = u32::from(val);
                let high = ((ch & 0b0000_0000_0000_0000_1111_0000_0000_0000) << 4)
                    | 0b0000_0000_1110_0000_0000_0000_0000_0000;
                let mid = ((ch & 0b0000_0000_0000_0000_0000_1111_1100_0000) << 2)
                    | 0b0000_0000_0000_0000_1000_0000_0000_0000;
                let low = (ch & 0b0000_0000_0000_0000_0000_0000_0011_1111)
                    | 0b0000_0000_0000_0000_0000_0000_1000_0000;

                unsafe { char::from_u32_unchecked(u32::from(high | mid | low)) }
            }
        }
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
impl<'a> Iterator for EncodeUcs2<'a> {
    type Item = Ucs2Char;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(ch) = self.utf8_buf.next() {
            match Ucs2Char::from_char(ch) {
                Some(x) => Some(x),
                None => Some(Ucs2Char::REPLACEMENT_CHARACTER),
            }
        } else {
            None
        }
    }
}

#[unstable(feature = "ucs2", issue = "none")]
impl<'a> FusedIterator for EncodeUcs2<'a> {}

/// A safe iterator over a LPWSTR
/// (aka a pointer to a series of UCS-2 code units terminated by a NULL).
#[unstable(feature = "ucs2", issue = "none")]
pub struct Ucs2Units<'a> {
    // The pointer must never be null...
    lpwstr: NonNull<u16>,
    // ...and the memory it points to must be valid for this lifetime.
    lifetime: PhantomData<&'a [u16]>,
}

impl Ucs2Units<'_> {
    /// Create the iterator. Returns `None` if `lpwstr` is null.
    ///
    /// SAFETY: `lpwstr` must point to a null-terminated wide string that lives
    /// at least as long as the lifetime of this struct.
    pub unsafe fn new(lpwstr: *const u16) -> Option<Self> {
        Some(Self { lpwstr: NonNull::new(lpwstr as _)?, lifetime: PhantomData })
    }

    pub fn peek(&self) -> Option<NonZeroU16> {
        // SAFETY: It's always safe to read the current item because we don't
        // ever move out of the array's bounds.
        unsafe { NonZeroU16::new(*self.lpwstr.as_ptr()) }
    }

    /// Advance the iterator while `predicate` returns true.
    /// Returns the number of items it advanced by.
    pub fn advance_while<P: FnMut(NonZeroU16) -> bool>(&mut self, mut predicate: P) -> usize {
        let mut counter = 0;
        while let Some(w) = self.peek() {
            if !predicate(w) {
                break;
            }
            counter += 1;
            self.next();
        }
        counter
    }
}

impl Iterator for Ucs2Units<'_> {
    // This can never return zero as that marks the end of the string.
    type Item = NonZeroU16;
    fn next(&mut self) -> Option<NonZeroU16> {
        // SAFETY: If NULL is reached we immediately return.
        // Therefore it's safe to advance the pointer after that.
        unsafe {
            let next = self.peek()?;
            self.lpwstr = NonNull::new_unchecked(self.lpwstr.as_ptr().add(1));
            Some(next)
        }
    }
}
