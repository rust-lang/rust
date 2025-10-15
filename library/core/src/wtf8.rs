//! Implementation of [the WTF-8 encoding](https://simonsapin.github.io/wtf-8/).
//!
//! This library uses Rust’s type system to maintain
//! [well-formedness](https://simonsapin.github.io/wtf-8/#well-formed),
//! like the `String` and `&str` types do for UTF-8.
//!
//! Since [WTF-8 must not be used
//! for interchange](https://simonsapin.github.io/wtf-8/#intended-audience),
//! this library deliberately does not provide access to the underlying bytes
//! of WTF-8 strings,
//! nor can it decode WTF-8 from arbitrary bytes.
//! WTF-8 strings can be obtained from UTF-8, UTF-16, or code points.
#![unstable(
    feature = "wtf8_internals",
    issue = "none",
    reason = "this is internal code for representing OsStr on some platforms and not a public API"
)]
// rustdoc bug: doc(hidden) on the module won't stop types in the module from showing up in trait
// implementations, so, we'll have to add more doc(hidden)s anyway
#![doc(hidden)]

use crate::char::{MAX_LEN_UTF16, encode_utf16_raw};
use crate::clone::CloneToUninit;
use crate::fmt::{self, Write};
use crate::hash::{Hash, Hasher};
use crate::iter::FusedIterator;
use crate::num::niche_types::CodePointInner;
use crate::str::next_code_point;
use crate::{ops, slice, str};

/// A Unicode code point: from U+0000 to U+10FFFF.
///
/// Compares with the `char` type,
/// which represents a Unicode scalar value:
/// a code point that is not a surrogate (U+D800 to U+DFFF).
#[derive(Eq, PartialEq, Ord, PartialOrd, Clone, Copy)]
#[doc(hidden)]
pub struct CodePoint(CodePointInner);

/// Format the code point as `U+` followed by four to six hexadecimal digits.
/// Example: `U+1F4A9`
impl fmt::Debug for CodePoint {
    #[inline]
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "U+{:04X}", self.0.as_inner())
    }
}

impl CodePoint {
    /// Unsafely creates a new `CodePoint` without checking the value.
    ///
    /// Only use when `value` is known to be less than or equal to 0x10FFFF.
    #[inline]
    pub unsafe fn from_u32_unchecked(value: u32) -> CodePoint {
        // SAFETY: Guaranteed by caller.
        CodePoint(unsafe { CodePointInner::new_unchecked(value) })
    }

    /// Creates a new `CodePoint` if the value is a valid code point.
    ///
    /// Returns `None` if `value` is above 0x10FFFF.
    #[inline]
    pub fn from_u32(value: u32) -> Option<CodePoint> {
        Some(CodePoint(CodePointInner::new(value)?))
    }

    /// Creates a new `CodePoint` from a `char`.
    ///
    /// Since all Unicode scalar values are code points, this always succeeds.
    #[inline]
    pub fn from_char(value: char) -> CodePoint {
        // SAFETY: All char are valid for this type.
        unsafe { CodePoint::from_u32_unchecked(value as u32) }
    }

    /// Returns the numeric value of the code point.
    #[inline]
    pub fn to_u32(&self) -> u32 {
        self.0.as_inner()
    }

    /// Returns the numeric value of the code point if it is a leading surrogate.
    #[inline]
    pub fn to_lead_surrogate(&self) -> Option<u16> {
        match self.to_u32() {
            lead @ 0xD800..=0xDBFF => Some(lead as u16),
            _ => None,
        }
    }

    /// Returns the numeric value of the code point if it is a trailing surrogate.
    #[inline]
    pub fn to_trail_surrogate(&self) -> Option<u16> {
        match self.to_u32() {
            trail @ 0xDC00..=0xDFFF => Some(trail as u16),
            _ => None,
        }
    }

    /// Optionally returns a Unicode scalar value for the code point.
    ///
    /// Returns `None` if the code point is a surrogate (from U+D800 to U+DFFF).
    #[inline]
    pub fn to_char(&self) -> Option<char> {
        match self.to_u32() {
            0xD800..=0xDFFF => None,
            // SAFETY: We explicitly check that the char is valid.
            valid => Some(unsafe { char::from_u32_unchecked(valid) }),
        }
    }

    /// Returns a Unicode scalar value for the code point.
    ///
    /// Returns `'\u{FFFD}'` (the replacement character “�”)
    /// if the code point is a surrogate (from U+D800 to U+DFFF).
    #[inline]
    pub fn to_char_lossy(&self) -> char {
        self.to_char().unwrap_or(char::REPLACEMENT_CHARACTER)
    }
}

/// A borrowed slice of well-formed WTF-8 data.
///
/// Similar to `&str`, but can additionally contain surrogate code points
/// if they’re not in a surrogate pair.
#[derive(Eq, Ord, PartialEq, PartialOrd)]
#[repr(transparent)]
#[rustc_has_incoherent_inherent_impls]
#[doc(hidden)]
pub struct Wtf8 {
    bytes: [u8],
}

impl AsRef<[u8]> for Wtf8 {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        &self.bytes
    }
}

/// Formats the string in double quotes, with characters escaped according to
/// [`char::escape_debug`] and unpaired surrogates represented as `\u{xxxx}`,
/// where each `x` is a hexadecimal digit.
impl fmt::Debug for Wtf8 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fn write_str_escaped(f: &mut fmt::Formatter<'_>, s: &str) -> fmt::Result {
            use crate::fmt::Write;
            for c in s.chars().flat_map(|c| c.escape_debug()) {
                f.write_char(c)?
            }
            Ok(())
        }

        formatter.write_str("\"")?;
        let mut pos = 0;
        while let Some((surrogate_pos, surrogate)) = self.next_surrogate(pos) {
            // SAFETY: next_surrogate provides an index for a range of valid UTF-8 bytes.
            write_str_escaped(formatter, unsafe {
                str::from_utf8_unchecked(&self.bytes[pos..surrogate_pos])
            })?;
            write!(formatter, "\\u{{{:x}}}", surrogate)?;
            pos = surrogate_pos + 3;
        }

        // SAFETY: after next_surrogate returns None, the remainder is valid UTF-8.
        write_str_escaped(formatter, unsafe { str::from_utf8_unchecked(&self.bytes[pos..]) })?;
        formatter.write_str("\"")
    }
}

/// Formats the string with unpaired surrogates substituted with the replacement
/// character, U+FFFD.
impl fmt::Display for Wtf8 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let wtf8_bytes = &self.bytes;
        let mut pos = 0;
        loop {
            match self.next_surrogate(pos) {
                Some((surrogate_pos, _)) => {
                    // SAFETY: next_surrogate provides an index for a range of valid UTF-8 bytes.
                    formatter.write_str(unsafe {
                        str::from_utf8_unchecked(&wtf8_bytes[pos..surrogate_pos])
                    })?;
                    formatter.write_char(char::REPLACEMENT_CHARACTER)?;
                    pos = surrogate_pos + 3;
                }
                None => {
                    // SAFETY: after next_surrogate returns None, the remainder is valid UTF-8.
                    let s = unsafe { str::from_utf8_unchecked(&wtf8_bytes[pos..]) };
                    if pos == 0 { return s.fmt(formatter) } else { return formatter.write_str(s) }
                }
            }
        }
    }
}

impl Wtf8 {
    /// Creates a WTF-8 slice from a UTF-8 `&str` slice.
    #[inline]
    pub fn from_str(value: &str) -> &Wtf8 {
        // SAFETY: Since WTF-8 is a superset of UTF-8, this always is valid.
        unsafe { Wtf8::from_bytes_unchecked(value.as_bytes()) }
    }

    /// Creates a WTF-8 slice from a WTF-8 byte slice.
    ///
    /// Since the byte slice is not checked for valid WTF-8, this functions is
    /// marked unsafe.
    #[inline]
    pub unsafe fn from_bytes_unchecked(value: &[u8]) -> &Wtf8 {
        // SAFETY: start with &[u8], end with fancy &[u8]
        unsafe { &*(value as *const [u8] as *const Wtf8) }
    }

    /// Creates a mutable WTF-8 slice from a mutable WTF-8 byte slice.
    ///
    /// Since the byte slice is not checked for valid WTF-8, this functions is
    /// marked unsafe.
    #[inline]
    pub unsafe fn from_mut_bytes_unchecked(value: &mut [u8]) -> &mut Wtf8 {
        // SAFETY: start with &mut [u8], end with fancy &mut [u8]
        unsafe { &mut *(value as *mut [u8] as *mut Wtf8) }
    }

    /// Returns the length, in WTF-8 bytes.
    #[inline]
    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    /// Returns the code point at `position` if it is in the ASCII range,
    /// or `b'\xFF'` otherwise.
    ///
    /// # Panics
    ///
    /// Panics if `position` is beyond the end of the string.
    #[inline]
    pub fn ascii_byte_at(&self, position: usize) -> u8 {
        match self.bytes[position] {
            ascii_byte @ 0x00..=0x7F => ascii_byte,
            _ => 0xFF,
        }
    }

    /// Returns an iterator for the string’s code points.
    #[inline]
    pub fn code_points(&self) -> Wtf8CodePoints<'_> {
        Wtf8CodePoints { bytes: self.bytes.iter() }
    }

    /// Access raw bytes of WTF-8 data
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Tries to convert the string to UTF-8 and return a `&str` slice.
    ///
    /// Returns `None` if the string contains surrogates.
    ///
    /// This does not copy the data.
    #[inline]
    pub fn as_str(&self) -> Result<&str, str::Utf8Error> {
        str::from_utf8(&self.bytes)
    }

    /// Converts the WTF-8 string to potentially ill-formed UTF-16
    /// and return an iterator of 16-bit code units.
    ///
    /// This is lossless:
    /// calling `Wtf8Buf::from_ill_formed_utf16` on the resulting code units
    /// would always return the original WTF-8 string.
    #[inline]
    pub fn encode_wide(&self) -> EncodeWide<'_> {
        EncodeWide { code_points: self.code_points(), extra: 0 }
    }

    #[inline]
    pub fn next_surrogate(&self, mut pos: usize) -> Option<(usize, u16)> {
        let mut iter = self.bytes[pos..].iter();
        loop {
            let b = *iter.next()?;
            if b < 0x80 {
                pos += 1;
            } else if b < 0xE0 {
                iter.next();
                pos += 2;
            } else if b == 0xED {
                match (iter.next(), iter.next()) {
                    (Some(&b2), Some(&b3)) if b2 >= 0xA0 => {
                        return Some((pos, decode_surrogate(b2, b3)));
                    }
                    _ => pos += 3,
                }
            } else if b < 0xF0 {
                iter.next();
                iter.next();
                pos += 3;
            } else {
                iter.next();
                iter.next();
                iter.next();
                pos += 4;
            }
        }
    }

    #[inline]
    pub fn final_lead_surrogate(&self) -> Option<u16> {
        match self.bytes {
            [.., 0xED, b2 @ 0xA0..=0xAF, b3] => Some(decode_surrogate(b2, b3)),
            _ => None,
        }
    }

    #[inline]
    pub fn initial_trail_surrogate(&self) -> Option<u16> {
        match self.bytes {
            [0xED, b2 @ 0xB0..=0xBF, b3, ..] => Some(decode_surrogate(b2, b3)),
            _ => None,
        }
    }

    #[inline]
    pub fn make_ascii_lowercase(&mut self) {
        self.bytes.make_ascii_lowercase()
    }

    #[inline]
    pub fn make_ascii_uppercase(&mut self) {
        self.bytes.make_ascii_uppercase()
    }

    #[inline]
    pub fn is_ascii(&self) -> bool {
        self.bytes.is_ascii()
    }

    #[inline]
    pub fn eq_ignore_ascii_case(&self, other: &Self) -> bool {
        self.bytes.eq_ignore_ascii_case(&other.bytes)
    }
}

/// Returns a slice of the given string for the byte range \[`begin`..`end`).
///
/// # Panics
///
/// Panics when `begin` and `end` do not point to code point boundaries,
/// or point beyond the end of the string.
impl ops::Index<ops::Range<usize>> for Wtf8 {
    type Output = Wtf8;

    #[inline]
    fn index(&self, range: ops::Range<usize>) -> &Wtf8 {
        if range.start <= range.end
            && self.is_code_point_boundary(range.start)
            && self.is_code_point_boundary(range.end)
        {
            // SAFETY: is_code_point_boundary checks that the index is valid
            unsafe { slice_unchecked(self, range.start, range.end) }
        } else {
            slice_error_fail(self, range.start, range.end)
        }
    }
}

/// Returns a slice of the given string from byte `begin` to its end.
///
/// # Panics
///
/// Panics when `begin` is not at a code point boundary,
/// or is beyond the end of the string.
impl ops::Index<ops::RangeFrom<usize>> for Wtf8 {
    type Output = Wtf8;

    #[inline]
    fn index(&self, range: ops::RangeFrom<usize>) -> &Wtf8 {
        if self.is_code_point_boundary(range.start) {
            // SAFETY: is_code_point_boundary checks that the index is valid
            unsafe { slice_unchecked(self, range.start, self.len()) }
        } else {
            slice_error_fail(self, range.start, self.len())
        }
    }
}

/// Returns a slice of the given string from its beginning to byte `end`.
///
/// # Panics
///
/// Panics when `end` is not at a code point boundary,
/// or is beyond the end of the string.
impl ops::Index<ops::RangeTo<usize>> for Wtf8 {
    type Output = Wtf8;

    #[inline]
    fn index(&self, range: ops::RangeTo<usize>) -> &Wtf8 {
        if self.is_code_point_boundary(range.end) {
            // SAFETY: is_code_point_boundary checks that the index is valid
            unsafe { slice_unchecked(self, 0, range.end) }
        } else {
            slice_error_fail(self, 0, range.end)
        }
    }
}

impl ops::Index<ops::RangeFull> for Wtf8 {
    type Output = Wtf8;

    #[inline]
    fn index(&self, _range: ops::RangeFull) -> &Wtf8 {
        self
    }
}

#[inline]
fn decode_surrogate(second_byte: u8, third_byte: u8) -> u16 {
    // The first byte is assumed to be 0xED
    0xD800 | (second_byte as u16 & 0x3F) << 6 | third_byte as u16 & 0x3F
}

impl Wtf8 {
    /// Copied from str::is_char_boundary
    #[inline]
    pub fn is_code_point_boundary(&self, index: usize) -> bool {
        if index == 0 {
            return true;
        }
        match self.bytes.get(index) {
            None => index == self.len(),
            Some(&b) => (b as i8) >= -0x40,
        }
    }

    /// Verify that `index` is at the edge of either a valid UTF-8 codepoint
    /// (i.e. a codepoint that's not a surrogate) or of the whole string.
    ///
    /// These are the cases currently permitted by `OsStr::self_encoded_bytes`.
    /// Splitting between surrogates is valid as far as WTF-8 is concerned, but
    /// we do not permit it in the public API because WTF-8 is considered an
    /// implementation detail.
    #[track_caller]
    #[inline]
    pub fn check_utf8_boundary(&self, index: usize) {
        if index == 0 {
            return;
        }
        match self.bytes.get(index) {
            Some(0xED) => (), // Might be a surrogate
            Some(&b) if (b as i8) >= -0x40 => return,
            Some(_) => panic!("byte index {index} is not a codepoint boundary"),
            None if index == self.len() => return,
            None => panic!("byte index {index} is out of bounds"),
        }
        if self.bytes[index + 1] >= 0xA0 {
            // There's a surrogate after index. Now check before index.
            if index >= 3 && self.bytes[index - 3] == 0xED && self.bytes[index - 2] >= 0xA0 {
                panic!("byte index {index} lies between surrogate codepoints");
            }
        }
    }
}

/// Copied from core::str::raw::slice_unchecked
#[inline]
unsafe fn slice_unchecked(s: &Wtf8, begin: usize, end: usize) -> &Wtf8 {
    // SAFETY: memory layout of a &[u8] and &Wtf8 are the same
    unsafe {
        let len = end - begin;
        let start = s.as_bytes().as_ptr().add(begin);
        Wtf8::from_bytes_unchecked(slice::from_raw_parts(start, len))
    }
}

/// Copied from core::str::raw::slice_error_fail
#[inline(never)]
fn slice_error_fail(s: &Wtf8, begin: usize, end: usize) -> ! {
    assert!(begin <= end);
    panic!("index {begin} and/or {end} in `{s:?}` do not lie on character boundary");
}

/// Iterator for the code points of a WTF-8 string.
///
/// Created with the method `.code_points()`.
#[derive(Clone)]
#[doc(hidden)]
pub struct Wtf8CodePoints<'a> {
    bytes: slice::Iter<'a, u8>,
}

impl Iterator for Wtf8CodePoints<'_> {
    type Item = CodePoint;

    #[inline]
    fn next(&mut self) -> Option<CodePoint> {
        // SAFETY: `self.bytes` has been created from a WTF-8 string
        unsafe { next_code_point(&mut self.bytes).map(|c| CodePoint::from_u32_unchecked(c)) }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.bytes.len();
        (len.saturating_add(3) / 4, Some(len))
    }
}

impl fmt::Debug for Wtf8CodePoints<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Wtf8CodePoints")
            // SAFETY: We always leave the string in a valid state after each iteration.
            .field(&unsafe { Wtf8::from_bytes_unchecked(self.bytes.as_slice()) })
            .finish()
    }
}

/// Generates a wide character sequence for potentially ill-formed UTF-16.
#[stable(feature = "rust1", since = "1.0.0")]
#[derive(Clone)]
#[doc(hidden)]
pub struct EncodeWide<'a> {
    code_points: Wtf8CodePoints<'a>,
    extra: u16,
}

// Copied from libunicode/u_str.rs
#[stable(feature = "rust1", since = "1.0.0")]
impl Iterator for EncodeWide<'_> {
    type Item = u16;

    #[inline]
    fn next(&mut self) -> Option<u16> {
        if self.extra != 0 {
            let tmp = self.extra;
            self.extra = 0;
            return Some(tmp);
        }

        let mut buf = [0; MAX_LEN_UTF16];
        self.code_points.next().map(|code_point| {
            let n = encode_utf16_raw(code_point.to_u32(), &mut buf).len();
            if n == 2 {
                self.extra = buf[1];
            }
            buf[0]
        })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (low, high) = self.code_points.size_hint();
        let ext = (self.extra != 0) as usize;
        // every code point gets either one u16 or two u16,
        // so this iterator is between 1 or 2 times as
        // long as the underlying iterator.
        (low + ext, high.and_then(|n| n.checked_mul(2)).and_then(|n| n.checked_add(ext)))
    }
}

impl fmt::Debug for EncodeWide<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EncodeWide").finish_non_exhaustive()
    }
}

#[stable(feature = "encode_wide_fused_iterator", since = "1.62.0")]
impl FusedIterator for EncodeWide<'_> {}

impl Hash for CodePoint {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state)
    }
}

impl Hash for Wtf8 {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write(&self.bytes);
        0xfeu8.hash(state)
    }
}

#[unstable(feature = "clone_to_uninit", issue = "126799")]
unsafe impl CloneToUninit for Wtf8 {
    #[inline]
    #[cfg_attr(debug_assertions, track_caller)]
    unsafe fn clone_to_uninit(&self, dst: *mut u8) {
        // SAFETY: we're just a transparent wrapper around [u8]
        unsafe { self.bytes.clone_to_uninit(dst) }
    }
}
