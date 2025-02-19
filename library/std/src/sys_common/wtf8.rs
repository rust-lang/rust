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

// this module is imported from @SimonSapin's repo and has tons of dead code on
// unix (it's mostly used on windows), so don't worry about dead code here.
#![allow(dead_code)]

#[cfg(test)]
mod tests;

use core::char::{MAX_LEN_UTF8, MAX_LEN_UTF16, encode_utf8_raw, encode_utf16_raw};
use core::clone::CloneToUninit;
use core::str::next_code_point;

use crate::borrow::Cow;
use crate::collections::TryReserveError;
use crate::hash::{Hash, Hasher};
use crate::iter::FusedIterator;
use crate::rc::Rc;
use crate::sync::Arc;
use crate::sys_common::AsInner;
use crate::{fmt, mem, ops, slice, str};

const UTF8_REPLACEMENT_CHARACTER: &str = "\u{FFFD}";

/// A Unicode code point: from U+0000 to U+10FFFF.
///
/// Compares with the `char` type,
/// which represents a Unicode scalar value:
/// a code point that is not a surrogate (U+D800 to U+DFFF).
#[derive(Eq, PartialEq, Ord, PartialOrd, Clone, Copy)]
pub struct CodePoint {
    value: u32,
}

/// Format the code point as `U+` followed by four to six hexadecimal digits.
/// Example: `U+1F4A9`
impl fmt::Debug for CodePoint {
    #[inline]
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "U+{:04X}", self.value)
    }
}

impl CodePoint {
    /// Unsafely creates a new `CodePoint` without checking the value.
    ///
    /// Only use when `value` is known to be less than or equal to 0x10FFFF.
    #[inline]
    pub unsafe fn from_u32_unchecked(value: u32) -> CodePoint {
        CodePoint { value }
    }

    /// Creates a new `CodePoint` if the value is a valid code point.
    ///
    /// Returns `None` if `value` is above 0x10FFFF.
    #[inline]
    pub fn from_u32(value: u32) -> Option<CodePoint> {
        match value {
            0..=0x10FFFF => Some(CodePoint { value }),
            _ => None,
        }
    }

    /// Creates a new `CodePoint` from a `char`.
    ///
    /// Since all Unicode scalar values are code points, this always succeeds.
    #[inline]
    pub fn from_char(value: char) -> CodePoint {
        CodePoint { value: value as u32 }
    }

    /// Returns the numeric value of the code point.
    #[inline]
    pub fn to_u32(&self) -> u32 {
        self.value
    }

    /// Returns the numeric value of the code point if it is a leading surrogate.
    #[inline]
    pub fn to_lead_surrogate(&self) -> Option<u16> {
        match self.value {
            lead @ 0xD800..=0xDBFF => Some(lead as u16),
            _ => None,
        }
    }

    /// Returns the numeric value of the code point if it is a trailing surrogate.
    #[inline]
    pub fn to_trail_surrogate(&self) -> Option<u16> {
        match self.value {
            trail @ 0xDC00..=0xDFFF => Some(trail as u16),
            _ => None,
        }
    }

    /// Optionally returns a Unicode scalar value for the code point.
    ///
    /// Returns `None` if the code point is a surrogate (from U+D800 to U+DFFF).
    #[inline]
    pub fn to_char(&self) -> Option<char> {
        match self.value {
            0xD800..=0xDFFF => None,
            _ => Some(unsafe { char::from_u32_unchecked(self.value) }),
        }
    }

    /// Returns a Unicode scalar value for the code point.
    ///
    /// Returns `'\u{FFFD}'` (the replacement character “�”)
    /// if the code point is a surrogate (from U+D800 to U+DFFF).
    #[inline]
    pub fn to_char_lossy(&self) -> char {
        self.to_char().unwrap_or('\u{FFFD}')
    }
}

/// An owned, growable string of well-formed WTF-8 data.
///
/// Similar to `String`, but can additionally contain surrogate code points
/// if they’re not in a surrogate pair.
#[derive(Eq, PartialEq, Ord, PartialOrd, Clone)]
pub struct Wtf8Buf {
    bytes: Vec<u8>,

    /// Do we know that `bytes` holds a valid UTF-8 encoding? We can easily
    /// know this if we're constructed from a `String` or `&str`.
    ///
    /// It is possible for `bytes` to have valid UTF-8 without this being
    /// set, such as when we're concatenating `&Wtf8`'s and surrogates become
    /// paired, as we don't bother to rescan the entire string.
    is_known_utf8: bool,
}

impl ops::Deref for Wtf8Buf {
    type Target = Wtf8;

    fn deref(&self) -> &Wtf8 {
        self.as_slice()
    }
}

impl ops::DerefMut for Wtf8Buf {
    fn deref_mut(&mut self) -> &mut Wtf8 {
        self.as_mut_slice()
    }
}

/// Formats the string in double quotes, with characters escaped according to
/// [`char::escape_debug`] and unpaired surrogates represented as `\u{xxxx}`,
/// where each `x` is a hexadecimal digit.
///
/// For example, the code units [U+0061, U+D800, U+000A] are formatted as
/// `"a\u{D800}\n"`.
impl fmt::Debug for Wtf8Buf {
    #[inline]
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&**self, formatter)
    }
}

impl Wtf8Buf {
    /// Creates a new, empty WTF-8 string.
    #[inline]
    pub fn new() -> Wtf8Buf {
        Wtf8Buf { bytes: Vec::new(), is_known_utf8: true }
    }

    /// Creates a new, empty WTF-8 string with pre-allocated capacity for `capacity` bytes.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Wtf8Buf {
        Wtf8Buf { bytes: Vec::with_capacity(capacity), is_known_utf8: true }
    }

    /// Creates a WTF-8 string from a WTF-8 byte vec.
    ///
    /// Since the byte vec is not checked for valid WTF-8, this function is
    /// marked unsafe.
    #[inline]
    pub unsafe fn from_bytes_unchecked(value: Vec<u8>) -> Wtf8Buf {
        Wtf8Buf { bytes: value, is_known_utf8: false }
    }

    /// Creates a WTF-8 string from a UTF-8 `String`.
    ///
    /// This takes ownership of the `String` and does not copy.
    ///
    /// Since WTF-8 is a superset of UTF-8, this always succeeds.
    #[inline]
    pub fn from_string(string: String) -> Wtf8Buf {
        Wtf8Buf { bytes: string.into_bytes(), is_known_utf8: true }
    }

    /// Creates a WTF-8 string from a UTF-8 `&str` slice.
    ///
    /// This copies the content of the slice.
    ///
    /// Since WTF-8 is a superset of UTF-8, this always succeeds.
    #[inline]
    pub fn from_str(s: &str) -> Wtf8Buf {
        Wtf8Buf { bytes: s.as_bytes().to_vec(), is_known_utf8: true }
    }

    pub fn clear(&mut self) {
        self.bytes.clear();
        self.is_known_utf8 = true;
    }

    /// Creates a WTF-8 string from a potentially ill-formed UTF-16 slice of 16-bit code units.
    ///
    /// This is lossless: calling `.encode_wide()` on the resulting string
    /// will always return the original code units.
    pub fn from_wide(v: &[u16]) -> Wtf8Buf {
        let mut string = Wtf8Buf::with_capacity(v.len());
        for item in char::decode_utf16(v.iter().cloned()) {
            match item {
                Ok(ch) => string.push_char(ch),
                Err(surrogate) => {
                    let surrogate = surrogate.unpaired_surrogate();
                    // Surrogates are known to be in the code point range.
                    let code_point = unsafe { CodePoint::from_u32_unchecked(surrogate as u32) };
                    // The string will now contain an unpaired surrogate.
                    string.is_known_utf8 = false;
                    // Skip the WTF-8 concatenation check,
                    // surrogate pairs are already decoded by decode_utf16
                    string.push_code_point_unchecked(code_point);
                }
            }
        }
        string
    }

    /// Appends the given `char` to the end of this string.
    /// This does **not** include the WTF-8 concatenation check or `is_known_utf8` check.
    /// Copied from String::push.
    fn push_code_point_unchecked(&mut self, code_point: CodePoint) {
        let mut bytes = [0; MAX_LEN_UTF8];
        let bytes = encode_utf8_raw(code_point.value, &mut bytes);
        self.bytes.extend_from_slice(bytes)
    }

    #[inline]
    pub fn as_slice(&self) -> &Wtf8 {
        unsafe { Wtf8::from_bytes_unchecked(&self.bytes) }
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut Wtf8 {
        // Safety: `Wtf8` doesn't expose any way to mutate the bytes that would
        // cause them to change from well-formed UTF-8 to ill-formed UTF-8,
        // which would break the assumptions of the `is_known_utf8` field.
        unsafe { Wtf8::from_mut_bytes_unchecked(&mut self.bytes) }
    }

    /// Reserves capacity for at least `additional` more bytes to be inserted
    /// in the given `Wtf8Buf`.
    /// The collection may reserve more space to avoid frequent reallocations.
    ///
    /// # Panics
    ///
    /// Panics if the new capacity exceeds `isize::MAX` bytes.
    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        self.bytes.reserve(additional)
    }

    /// Tries to reserve capacity for at least `additional` more bytes to be
    /// inserted in the given `Wtf8Buf`. The `Wtf8Buf` may reserve more space to
    /// avoid frequent reallocations. After calling `try_reserve`, capacity will
    /// be greater than or equal to `self.len() + additional`. Does nothing if
    /// capacity is already sufficient. This method preserves the contents even
    /// if an error occurs.
    ///
    /// # Errors
    ///
    /// If the capacity overflows, or the allocator reports a failure, then an error
    /// is returned.
    #[inline]
    pub fn try_reserve(&mut self, additional: usize) -> Result<(), TryReserveError> {
        self.bytes.try_reserve(additional)
    }

    #[inline]
    pub fn reserve_exact(&mut self, additional: usize) {
        self.bytes.reserve_exact(additional)
    }

    /// Tries to reserve the minimum capacity for exactly `additional` more
    /// bytes to be inserted in the given `Wtf8Buf`. After calling
    /// `try_reserve_exact`, capacity will be greater than or equal to
    /// `self.len() + additional` if it returns `Ok(())`.
    /// Does nothing if the capacity is already sufficient.
    ///
    /// Note that the allocator may give the `Wtf8Buf` more space than it
    /// requests. Therefore, capacity can not be relied upon to be precisely
    /// minimal. Prefer [`try_reserve`] if future insertions are expected.
    ///
    /// [`try_reserve`]: Wtf8Buf::try_reserve
    ///
    /// # Errors
    ///
    /// If the capacity overflows, or the allocator reports a failure, then an error
    /// is returned.
    #[inline]
    pub fn try_reserve_exact(&mut self, additional: usize) -> Result<(), TryReserveError> {
        self.bytes.try_reserve_exact(additional)
    }

    #[inline]
    pub fn shrink_to_fit(&mut self) {
        self.bytes.shrink_to_fit()
    }

    #[inline]
    pub fn shrink_to(&mut self, min_capacity: usize) {
        self.bytes.shrink_to(min_capacity)
    }

    #[inline]
    pub fn leak<'a>(self) -> &'a mut Wtf8 {
        unsafe { Wtf8::from_mut_bytes_unchecked(self.bytes.leak()) }
    }

    /// Returns the number of bytes that this string buffer can hold without reallocating.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.bytes.capacity()
    }

    /// Append a UTF-8 slice at the end of the string.
    #[inline]
    pub fn push_str(&mut self, other: &str) {
        self.bytes.extend_from_slice(other.as_bytes())
    }

    /// Append a WTF-8 slice at the end of the string.
    ///
    /// This replaces newly paired surrogates at the boundary
    /// with a supplementary code point,
    /// like concatenating ill-formed UTF-16 strings effectively would.
    #[inline]
    pub fn push_wtf8(&mut self, other: &Wtf8) {
        match ((&*self).final_lead_surrogate(), other.initial_trail_surrogate()) {
            // Replace newly paired surrogates by a supplementary code point.
            (Some(lead), Some(trail)) => {
                let len_without_lead_surrogate = self.len() - 3;
                self.bytes.truncate(len_without_lead_surrogate);
                let other_without_trail_surrogate = &other.bytes[3..];
                // 4 bytes for the supplementary code point
                self.bytes.reserve(4 + other_without_trail_surrogate.len());
                self.push_char(decode_surrogate_pair(lead, trail));
                self.bytes.extend_from_slice(other_without_trail_surrogate);
            }
            _ => {
                // If we'll be pushing a string containing a surrogate, we may
                // no longer have UTF-8.
                if other.next_surrogate(0).is_some() {
                    self.is_known_utf8 = false;
                }

                self.bytes.extend_from_slice(&other.bytes);
            }
        }
    }

    /// Append a Unicode scalar value at the end of the string.
    #[inline]
    pub fn push_char(&mut self, c: char) {
        self.push_code_point_unchecked(CodePoint::from_char(c))
    }

    /// Append a code point at the end of the string.
    ///
    /// This replaces newly paired surrogates at the boundary
    /// with a supplementary code point,
    /// like concatenating ill-formed UTF-16 strings effectively would.
    #[inline]
    pub fn push(&mut self, code_point: CodePoint) {
        if let Some(trail) = code_point.to_trail_surrogate() {
            if let Some(lead) = (&*self).final_lead_surrogate() {
                let len_without_lead_surrogate = self.len() - 3;
                self.bytes.truncate(len_without_lead_surrogate);
                self.push_char(decode_surrogate_pair(lead, trail));
                return;
            }

            // We're pushing a trailing surrogate.
            self.is_known_utf8 = false;
        } else if code_point.to_lead_surrogate().is_some() {
            // We're pushing a leading surrogate.
            self.is_known_utf8 = false;
        }

        // No newly paired surrogates at the boundary.
        self.push_code_point_unchecked(code_point)
    }

    /// Shortens a string to the specified length.
    ///
    /// # Panics
    ///
    /// Panics if `new_len` > current length,
    /// or if `new_len` is not a code point boundary.
    #[inline]
    pub fn truncate(&mut self, new_len: usize) {
        assert!(is_code_point_boundary(self, new_len));
        self.bytes.truncate(new_len)
    }

    /// Consumes the WTF-8 string and tries to convert it to a vec of bytes.
    #[inline]
    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }

    /// Consumes the WTF-8 string and tries to convert it to UTF-8.
    ///
    /// This does not copy the data.
    ///
    /// If the contents are not well-formed UTF-8
    /// (that is, if the string contains surrogates),
    /// the original WTF-8 string is returned instead.
    pub fn into_string(self) -> Result<String, Wtf8Buf> {
        if self.is_known_utf8 || self.next_surrogate(0).is_none() {
            Ok(unsafe { String::from_utf8_unchecked(self.bytes) })
        } else {
            Err(self)
        }
    }

    /// Consumes the WTF-8 string and converts it lossily to UTF-8.
    ///
    /// This does not copy the data (but may overwrite parts of it in place).
    ///
    /// Surrogates are replaced with `"\u{FFFD}"` (the replacement character “�”)
    pub fn into_string_lossy(mut self) -> String {
        if !self.is_known_utf8 {
            let mut pos = 0;
            while let Some((surrogate_pos, _)) = self.next_surrogate(pos) {
                pos = surrogate_pos + 3;
                // Surrogates and the replacement character are all 3 bytes, so
                // they can substituted in-place.
                self.bytes[surrogate_pos..pos]
                    .copy_from_slice(UTF8_REPLACEMENT_CHARACTER.as_bytes());
            }
        }
        unsafe { String::from_utf8_unchecked(self.bytes) }
    }

    /// Converts this `Wtf8Buf` into a boxed `Wtf8`.
    #[inline]
    pub fn into_box(self) -> Box<Wtf8> {
        // SAFETY: relies on `Wtf8` being `repr(transparent)`.
        unsafe { mem::transmute(self.bytes.into_boxed_slice()) }
    }

    /// Converts a `Box<Wtf8>` into a `Wtf8Buf`.
    pub fn from_box(boxed: Box<Wtf8>) -> Wtf8Buf {
        let bytes: Box<[u8]> = unsafe { mem::transmute(boxed) };
        Wtf8Buf { bytes: bytes.into_vec(), is_known_utf8: false }
    }

    /// Provides plumbing to core `Vec::extend_from_slice`.
    /// More well behaving alternative to allowing outer types
    /// full mutable access to the core `Vec`.
    #[inline]
    pub(crate) fn extend_from_slice(&mut self, other: &[u8]) {
        self.bytes.extend_from_slice(other);
        self.is_known_utf8 = false;
    }
}

/// Creates a new WTF-8 string from an iterator of code points.
///
/// This replaces surrogate code point pairs with supplementary code points,
/// like concatenating ill-formed UTF-16 strings effectively would.
impl FromIterator<CodePoint> for Wtf8Buf {
    fn from_iter<T: IntoIterator<Item = CodePoint>>(iter: T) -> Wtf8Buf {
        let mut string = Wtf8Buf::new();
        string.extend(iter);
        string
    }
}

/// Append code points from an iterator to the string.
///
/// This replaces surrogate code point pairs with supplementary code points,
/// like concatenating ill-formed UTF-16 strings effectively would.
impl Extend<CodePoint> for Wtf8Buf {
    fn extend<T: IntoIterator<Item = CodePoint>>(&mut self, iter: T) {
        let iterator = iter.into_iter();
        let (low, _high) = iterator.size_hint();
        // Lower bound of one byte per code point (ASCII only)
        self.bytes.reserve(low);
        iterator.for_each(move |code_point| self.push(code_point));
    }

    #[inline]
    fn extend_one(&mut self, code_point: CodePoint) {
        self.push(code_point);
    }

    #[inline]
    fn extend_reserve(&mut self, additional: usize) {
        // Lower bound of one byte per code point (ASCII only)
        self.bytes.reserve(additional);
    }
}

/// A borrowed slice of well-formed WTF-8 data.
///
/// Similar to `&str`, but can additionally contain surrogate code points
/// if they’re not in a surrogate pair.
#[derive(Eq, Ord, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct Wtf8 {
    bytes: [u8],
}

impl AsInner<[u8]> for Wtf8 {
    #[inline]
    fn as_inner(&self) -> &[u8] {
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
            write_str_escaped(formatter, unsafe {
                str::from_utf8_unchecked(&self.bytes[pos..surrogate_pos])
            })?;
            write!(formatter, "\\u{{{:x}}}", surrogate)?;
            pos = surrogate_pos + 3;
        }
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
                    formatter.write_str(unsafe {
                        str::from_utf8_unchecked(&wtf8_bytes[pos..surrogate_pos])
                    })?;
                    formatter.write_str(UTF8_REPLACEMENT_CHARACTER)?;
                    pos = surrogate_pos + 3;
                }
                None => {
                    let s = unsafe { str::from_utf8_unchecked(&wtf8_bytes[pos..]) };
                    if pos == 0 { return s.fmt(formatter) } else { return formatter.write_str(s) }
                }
            }
        }
    }
}

impl Wtf8 {
    /// Creates a WTF-8 slice from a UTF-8 `&str` slice.
    ///
    /// Since WTF-8 is a superset of UTF-8, this always succeeds.
    #[inline]
    pub fn from_str(value: &str) -> &Wtf8 {
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
    unsafe fn from_mut_bytes_unchecked(value: &mut [u8]) -> &mut Wtf8 {
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

    /// Creates an owned `Wtf8Buf` from a borrowed `Wtf8`.
    pub fn to_owned(&self) -> Wtf8Buf {
        Wtf8Buf { bytes: self.bytes.to_vec(), is_known_utf8: false }
    }

    /// Lossily converts the string to UTF-8.
    /// Returns a UTF-8 `&str` slice if the contents are well-formed in UTF-8.
    ///
    /// Surrogates are replaced with `"\u{FFFD}"` (the replacement character “�”).
    ///
    /// This only copies the data if necessary (if it contains any surrogate).
    pub fn to_string_lossy(&self) -> Cow<'_, str> {
        let Some((surrogate_pos, _)) = self.next_surrogate(0) else {
            return Cow::Borrowed(unsafe { str::from_utf8_unchecked(&self.bytes) });
        };
        let wtf8_bytes = &self.bytes;
        let mut utf8_bytes = Vec::with_capacity(self.len());
        utf8_bytes.extend_from_slice(&wtf8_bytes[..surrogate_pos]);
        utf8_bytes.extend_from_slice(UTF8_REPLACEMENT_CHARACTER.as_bytes());
        let mut pos = surrogate_pos + 3;
        loop {
            match self.next_surrogate(pos) {
                Some((surrogate_pos, _)) => {
                    utf8_bytes.extend_from_slice(&wtf8_bytes[pos..surrogate_pos]);
                    utf8_bytes.extend_from_slice(UTF8_REPLACEMENT_CHARACTER.as_bytes());
                    pos = surrogate_pos + 3;
                }
                None => {
                    utf8_bytes.extend_from_slice(&wtf8_bytes[pos..]);
                    return Cow::Owned(unsafe { String::from_utf8_unchecked(utf8_bytes) });
                }
            }
        }
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
    fn next_surrogate(&self, mut pos: usize) -> Option<(usize, u16)> {
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
    fn final_lead_surrogate(&self) -> Option<u16> {
        match self.bytes {
            [.., 0xED, b2 @ 0xA0..=0xAF, b3] => Some(decode_surrogate(b2, b3)),
            _ => None,
        }
    }

    #[inline]
    fn initial_trail_surrogate(&self) -> Option<u16> {
        match self.bytes {
            [0xED, b2 @ 0xB0..=0xBF, b3, ..] => Some(decode_surrogate(b2, b3)),
            _ => None,
        }
    }

    pub fn clone_into(&self, buf: &mut Wtf8Buf) {
        buf.is_known_utf8 = false;
        self.bytes.clone_into(&mut buf.bytes);
    }

    /// Boxes this `Wtf8`.
    #[inline]
    pub fn into_box(&self) -> Box<Wtf8> {
        let boxed: Box<[u8]> = self.bytes.into();
        unsafe { mem::transmute(boxed) }
    }

    /// Creates a boxed, empty `Wtf8`.
    pub fn empty_box() -> Box<Wtf8> {
        let boxed: Box<[u8]> = Default::default();
        unsafe { mem::transmute(boxed) }
    }

    #[inline]
    pub fn into_arc(&self) -> Arc<Wtf8> {
        let arc: Arc<[u8]> = Arc::from(&self.bytes);
        unsafe { Arc::from_raw(Arc::into_raw(arc) as *const Wtf8) }
    }

    #[inline]
    pub fn into_rc(&self) -> Rc<Wtf8> {
        let rc: Rc<[u8]> = Rc::from(&self.bytes);
        unsafe { Rc::from_raw(Rc::into_raw(rc) as *const Wtf8) }
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
    pub fn to_ascii_lowercase(&self) -> Wtf8Buf {
        Wtf8Buf { bytes: self.bytes.to_ascii_lowercase(), is_known_utf8: false }
    }

    #[inline]
    pub fn to_ascii_uppercase(&self) -> Wtf8Buf {
        Wtf8Buf { bytes: self.bytes.to_ascii_uppercase(), is_known_utf8: false }
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
        // is_code_point_boundary checks that the index is in [0, .len()]
        if range.start <= range.end
            && is_code_point_boundary(self, range.start)
            && is_code_point_boundary(self, range.end)
        {
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
        // is_code_point_boundary checks that the index is in [0, .len()]
        if is_code_point_boundary(self, range.start) {
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
        // is_code_point_boundary checks that the index is in [0, .len()]
        if is_code_point_boundary(self, range.end) {
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

#[inline]
fn decode_surrogate_pair(lead: u16, trail: u16) -> char {
    let code_point = 0x10000 + ((((lead - 0xD800) as u32) << 10) | (trail - 0xDC00) as u32);
    unsafe { char::from_u32_unchecked(code_point) }
}

/// Copied from str::is_char_boundary
#[inline]
pub fn is_code_point_boundary(slice: &Wtf8, index: usize) -> bool {
    if index == 0 {
        return true;
    }
    match slice.bytes.get(index) {
        None => index == slice.len(),
        Some(&b) => (b as i8) >= -0x40,
    }
}

/// Verify that `index` is at the edge of either a valid UTF-8 codepoint
/// (i.e. a codepoint that's not a surrogate) or of the whole string.
///
/// These are the cases currently permitted by `OsStr::slice_encoded_bytes`.
/// Splitting between surrogates is valid as far as WTF-8 is concerned, but
/// we do not permit it in the public API because WTF-8 is considered an
/// implementation detail.
#[track_caller]
#[inline]
pub fn check_utf8_boundary(slice: &Wtf8, index: usize) {
    if index == 0 {
        return;
    }
    match slice.bytes.get(index) {
        Some(0xED) => (), // Might be a surrogate
        Some(&b) if (b as i8) >= -0x40 => return,
        Some(_) => panic!("byte index {index} is not a codepoint boundary"),
        None if index == slice.len() => return,
        None => panic!("byte index {index} is out of bounds"),
    }
    if slice.bytes[index + 1] >= 0xA0 {
        // There's a surrogate after index. Now check before index.
        if index >= 3 && slice.bytes[index - 3] == 0xED && slice.bytes[index - 2] >= 0xA0 {
            panic!("byte index {index} lies between surrogate codepoints");
        }
    }
}

/// Copied from core::str::raw::slice_unchecked
#[inline]
pub unsafe fn slice_unchecked(s: &Wtf8, begin: usize, end: usize) -> &Wtf8 {
    // SAFETY: memory layout of a &[u8] and &Wtf8 are the same
    unsafe {
        let len = end - begin;
        let start = s.as_bytes().as_ptr().add(begin);
        Wtf8::from_bytes_unchecked(slice::from_raw_parts(start, len))
    }
}

/// Copied from core::str::raw::slice_error_fail
#[inline(never)]
pub fn slice_error_fail(s: &Wtf8, begin: usize, end: usize) -> ! {
    assert!(begin <= end);
    panic!("index {begin} and/or {end} in `{s:?}` do not lie on character boundary");
}

/// Iterator for the code points of a WTF-8 string.
///
/// Created with the method `.code_points()`.
#[derive(Clone)]
pub struct Wtf8CodePoints<'a> {
    bytes: slice::Iter<'a, u8>,
}

impl Iterator for Wtf8CodePoints<'_> {
    type Item = CodePoint;

    #[inline]
    fn next(&mut self) -> Option<CodePoint> {
        // SAFETY: `self.bytes` has been created from a WTF-8 string
        unsafe { next_code_point(&mut self.bytes).map(|c| CodePoint { value: c }) }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.bytes.len();
        (len.saturating_add(3) / 4, Some(len))
    }
}

/// Generates a wide character sequence for potentially ill-formed UTF-16.
#[stable(feature = "rust1", since = "1.0.0")]
#[derive(Clone)]
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
            let n = encode_utf16_raw(code_point.value, &mut buf).len();
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

#[stable(feature = "encode_wide_fused_iterator", since = "1.62.0")]
impl FusedIterator for EncodeWide<'_> {}

impl Hash for CodePoint {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.value.hash(state)
    }
}

impl Hash for Wtf8Buf {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write(&self.bytes);
        0xfeu8.hash(state)
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
