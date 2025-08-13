//! Heap-allocated counterpart to core `wtf8` module.
#![unstable(
    feature = "wtf8_internals",
    issue = "none",
    reason = "this is internal code for representing OsStr on some platforms and not a public API"
)]
// rustdoc bug: doc(hidden) on the module won't stop types in the module from showing up in trait
// implementations, so, we'll have to add more doc(hidden)s anyway
#![doc(hidden)]

// Note: This module is also included in the alloctests crate using #[path] to
// run the tests. See the comment there for an explanation why this is the case.

#[cfg(test)]
mod tests;

use core::char::{MAX_LEN_UTF8, encode_utf8_raw};
use core::hash::{Hash, Hasher};
pub use core::wtf8::{CodePoint, Wtf8};
#[cfg(not(test))]
pub use core::wtf8::{EncodeWide, Wtf8CodePoints};
use core::{fmt, mem, ops, str};

use crate::borrow::{Cow, ToOwned};
use crate::boxed::Box;
use crate::collections::TryReserveError;
#[cfg(not(test))]
use crate::rc::Rc;
use crate::string::String;
#[cfg(all(not(test), target_has_atomic = "ptr"))]
use crate::sync::Arc;
use crate::vec::Vec;

/// An owned, growable string of well-formed WTF-8 data.
///
/// Similar to `String`, but can additionally contain surrogate code points
/// if they’re not in a surrogate pair.
#[derive(Eq, PartialEq, Ord, PartialOrd, Clone)]
#[doc(hidden)]
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

/// Formats the string with unpaired surrogates substituted with the replacement
/// character, U+FFFD.
impl fmt::Display for Wtf8Buf {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(s) = self.as_known_utf8() {
            fmt::Display::fmt(s, formatter)
        } else {
            fmt::Display::fmt(&**self, formatter)
        }
    }
}

#[cfg_attr(test, allow(dead_code))]
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
    pub const fn from_string(string: String) -> Wtf8Buf {
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
                    unsafe {
                        string.push_code_point_unchecked(code_point);
                    }
                }
            }
        }
        string
    }

    /// Appends the given `char` to the end of this string.
    /// This does **not** include the WTF-8 concatenation check or `is_known_utf8` check.
    /// Copied from String::push.
    unsafe fn push_code_point_unchecked(&mut self, code_point: CodePoint) {
        let mut bytes = [0; MAX_LEN_UTF8];
        let bytes = encode_utf8_raw(code_point.to_u32(), &mut bytes);
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

    /// Converts the string to UTF-8 without validation, if it was created from
    /// valid UTF-8.
    #[inline]
    fn as_known_utf8(&self) -> Option<&str> {
        if self.is_known_utf8 {
            // SAFETY: The buffer is known to be valid UTF-8.
            Some(unsafe { str::from_utf8_unchecked(self.as_bytes()) })
        } else {
            None
        }
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
                let other_without_trail_surrogate = &other.as_bytes()[3..];
                // 4 bytes for the supplementary code point
                self.bytes.reserve(4 + other_without_trail_surrogate.len());
                self.push_char(decode_surrogate_pair(lead, trail));
                self.bytes.extend_from_slice(other_without_trail_surrogate);
            }
            _ => {
                // If we'll be pushing a string containing a surrogate, we may
                // no longer have UTF-8.
                if self.is_known_utf8 && other.next_surrogate(0).is_some() {
                    self.is_known_utf8 = false;
                }

                self.bytes.extend_from_slice(other.as_bytes());
            }
        }
    }

    /// Append a Unicode scalar value at the end of the string.
    #[inline]
    pub fn push_char(&mut self, c: char) {
        // SAFETY: It's always safe to push a char.
        unsafe { self.push_code_point_unchecked(CodePoint::from_char(c)) }
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
        unsafe { self.push_code_point_unchecked(code_point) }
    }

    /// Shortens a string to the specified length.
    ///
    /// # Panics
    ///
    /// Panics if `new_len` > current length,
    /// or if `new_len` is not a code point boundary.
    #[inline]
    pub fn truncate(&mut self, new_len: usize) {
        assert!(self.is_code_point_boundary(new_len));
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
                self.bytes[surrogate_pos..pos].copy_from_slice("\u{FFFD}".as_bytes());
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
    pub unsafe fn extend_from_slice_unchecked(&mut self, other: &[u8]) {
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

/// Creates an owned `Wtf8Buf` from a borrowed `Wtf8`.
pub(super) fn to_owned(slice: &Wtf8) -> Wtf8Buf {
    Wtf8Buf { bytes: slice.as_bytes().to_vec(), is_known_utf8: false }
}

/// Lossily converts the string to UTF-8.
/// Returns a UTF-8 `&str` slice if the contents are well-formed in UTF-8.
///
/// Surrogates are replaced with `"\u{FFFD}"` (the replacement character “�”).
///
/// This only copies the data if necessary (if it contains any surrogate).
pub(super) fn to_string_lossy(slice: &Wtf8) -> Cow<'_, str> {
    let Some((surrogate_pos, _)) = slice.next_surrogate(0) else {
        return Cow::Borrowed(unsafe { str::from_utf8_unchecked(slice.as_bytes()) });
    };
    let wtf8_bytes = slice.as_bytes();
    let mut utf8_bytes = Vec::with_capacity(slice.len());
    utf8_bytes.extend_from_slice(&wtf8_bytes[..surrogate_pos]);
    utf8_bytes.extend_from_slice("\u{FFFD}".as_bytes());
    let mut pos = surrogate_pos + 3;
    loop {
        match slice.next_surrogate(pos) {
            Some((surrogate_pos, _)) => {
                utf8_bytes.extend_from_slice(&wtf8_bytes[pos..surrogate_pos]);
                utf8_bytes.extend_from_slice("\u{FFFD}".as_bytes());
                pos = surrogate_pos + 3;
            }
            None => {
                utf8_bytes.extend_from_slice(&wtf8_bytes[pos..]);
                return Cow::Owned(unsafe { String::from_utf8_unchecked(utf8_bytes) });
            }
        }
    }
}

#[inline]
pub(super) fn clone_into(slice: &Wtf8, buf: &mut Wtf8Buf) {
    buf.is_known_utf8 = false;
    slice.as_bytes().clone_into(&mut buf.bytes);
}

#[cfg(not(test))]
impl Wtf8 {
    #[rustc_allow_incoherent_impl]
    pub fn to_owned(&self) -> Wtf8Buf {
        to_owned(self)
    }

    #[rustc_allow_incoherent_impl]
    pub fn clone_into(&self, buf: &mut Wtf8Buf) {
        clone_into(self, buf)
    }

    #[rustc_allow_incoherent_impl]
    pub fn to_string_lossy(&self) -> Cow<'_, str> {
        to_string_lossy(self)
    }

    #[rustc_allow_incoherent_impl]
    pub fn into_box(&self) -> Box<Wtf8> {
        let boxed: Box<[u8]> = self.as_bytes().into();
        unsafe { mem::transmute(boxed) }
    }

    #[rustc_allow_incoherent_impl]
    pub fn empty_box() -> Box<Wtf8> {
        let boxed: Box<[u8]> = Default::default();
        unsafe { mem::transmute(boxed) }
    }

    #[cfg(target_has_atomic = "ptr")]
    #[rustc_allow_incoherent_impl]
    pub fn into_arc(&self) -> Arc<Wtf8> {
        let arc: Arc<[u8]> = Arc::from(self.as_bytes());
        unsafe { Arc::from_raw(Arc::into_raw(arc) as *const Wtf8) }
    }

    #[rustc_allow_incoherent_impl]
    pub fn into_rc(&self) -> Rc<Wtf8> {
        let rc: Rc<[u8]> = Rc::from(self.as_bytes());
        unsafe { Rc::from_raw(Rc::into_raw(rc) as *const Wtf8) }
    }

    #[inline]
    #[rustc_allow_incoherent_impl]
    pub fn to_ascii_lowercase(&self) -> Wtf8Buf {
        Wtf8Buf { bytes: self.as_bytes().to_ascii_lowercase(), is_known_utf8: false }
    }

    #[inline]
    #[rustc_allow_incoherent_impl]
    pub fn to_ascii_uppercase(&self) -> Wtf8Buf {
        Wtf8Buf { bytes: self.as_bytes().to_ascii_uppercase(), is_known_utf8: false }
    }
}

#[inline]
fn decode_surrogate_pair(lead: u16, trail: u16) -> char {
    let code_point = 0x10000 + ((((lead - 0xD800) as u32) << 10) | (trail - 0xDC00) as u32);
    unsafe { char::from_u32_unchecked(code_point) }
}

impl Hash for Wtf8Buf {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write(&self.bytes);
        0xfeu8.hash(state)
    }
}
