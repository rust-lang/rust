//! The `ByteStr` and `ByteString` types and trait implementations.

// This could be more fine-grained.
#![cfg(not(no_global_oom_handling))]

use core::borrow::{Borrow, BorrowMut};
#[unstable(feature = "bstr", issue = "134915")]
pub use core::bstr::ByteStr;
use core::bstr::{impl_partial_eq, impl_partial_eq_n, impl_partial_eq_ord};
use core::cmp::Ordering;
use core::ops::{
    Deref, DerefMut, DerefPure, Index, IndexMut, Range, RangeFrom, RangeFull, RangeInclusive,
    RangeTo, RangeToInclusive,
};
use core::str::FromStr;
use core::{fmt, hash};

use crate::borrow::{Cow, ToOwned};
use crate::boxed::Box;
use crate::collections::TryReserveError;
#[cfg(not(no_rc))]
use crate::rc::Rc;
use crate::string::String;
#[cfg(all(not(no_rc), not(no_sync), target_has_atomic = "ptr"))]
use crate::sync::Arc;
use crate::vec::Vec;

/// A wrapper for `Vec<u8>` representing a human-readable string that's conventionally, but not
/// always, UTF-8.
///
/// Unlike `String`, this type permits non-UTF-8 contents, making it suitable for user input,
/// non-native filenames (as `Path` only supports native filenames), and other applications that
/// need to round-trip whatever data the user provides.
///
/// A `ByteString` owns its contents and can grow and shrink, like a `Vec` or `String`. For a
/// borrowed byte string, see [`ByteStr`](../../std/bstr/struct.ByteStr.html).
///
/// `ByteString` implements `Deref` to `&Vec<u8>`, so all methods available on `&Vec<u8>` are
/// available on `ByteString`. Similarly, `ByteString` implements `DerefMut` to `&mut Vec<u8>`,
/// so you can modify a `ByteString` using any method available on `&mut Vec<u8>`.
///
/// The `Debug` and `Display` implementations for `ByteString` are the same as those for `ByteStr`,
/// showing invalid UTF-8 as hex escapes or the Unicode replacement character, respectively.
#[unstable(feature = "bstr", issue = "134915")]
#[repr(transparent)]
#[derive(Clone)]
#[doc(alias = "BString")]
pub struct ByteString(pub(crate) Vec<u8>);

impl ByteString {
    /// Creates an empty `ByteString`.
    #[unstable(feature = "bstr", issue = "134915")]
    #[inline]
    #[rustc_const_unstable(feature = "bstr", issue = "134915")]
    pub const fn new() -> ByteString {
        ByteString(Vec::new())
    }

    /// Converts to a [`ByteStr`] slice.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(bstr)]
    /// use std::bstr::ByteStr;
    ///
    /// let byte_str = ByteStr::new("foo");
    /// let byte_string = byte_str.to_byte_string();
    /// assert_eq!(byte_string.as_byte_str(), byte_str);
    /// ```
    #[unstable(feature = "bstr", issue = "134915")]
    #[inline]
    pub fn as_byte_str(&self) -> &ByteStr {
        ByteStr::new(&self.0)
    }

    /// Converts to a mutable [`ByteStr`] slice.
    #[unstable(feature = "bstr", issue = "134915")]
    #[inline]
    pub fn as_mut_byte_str(&mut self) -> &mut ByteStr {
        ByteStr::new_mut(&mut self.0)
    }

    /// Returns a reference to the underlying vector for this `ByteString`.
    #[unstable(feature = "bstr", issue = "134915")]
    #[inline]
    pub fn as_vec(&self) -> &Vec<u8> {
        &self.0
    }

    /// Returns a mutable reference to the underlying vector for this `ByteString`.
    #[unstable(feature = "bstr", issue = "134915")]
    #[inline]
    pub fn as_mut_vec(&mut self) -> &mut Vec<u8> {
        &mut self.0
    }

    /// Converts to a vector of bytes.
    #[unstable(feature = "bstr", issue = "134915")]
    #[inline]
    pub fn into_bytes(self) -> Vec<u8> {
        self.0
    }

    /// Converts to a boxed slice of bytes.
    #[unstable(feature = "bstr", issue = "134915")]
    #[inline]
    pub fn into_boxed_bytes(self) -> Box<[u8]> {
        self.into_bytes().into_boxed_slice()
    }

    /// Converts to a boxed byte string.
    #[unstable(feature = "bstr", issue = "134915")]
    #[inline]
    pub fn into_boxed_byte_str(self) -> Box<ByteStr> {
        self.into_bytes().into_boxed_slice().into_boxed_byte_str()
    }

    /// Converts a `ByteString` into a [`String`] if it contains valid Unicode data.
    ///
    /// On failure, ownership of the original `ByteString` is returned.
    #[unstable(feature = "bstr", issue = "134915")]
    #[inline]
    pub fn into_string(self) -> Result<String, ByteString> {
        String::from_utf8(self.0).map_err(|e| ByteString(e.into_bytes()))
    }

    /// Extends the byte string with the given <code>&[ByteStr]</code> slice.
    #[unstable(feature = "bstr", issue = "134915")]
    #[inline]
    pub fn push<S: AsRef<ByteStr>>(&mut self, s: S) {
        self.0.extend_from_slice(s.as_ref().as_bytes())
    }

    /// Pushes a single byte onto the byte string.
    #[unstable(feature = "bstr", issue = "134915")]
    #[inline]
    pub fn push_byte(&mut self, b: u8) {
        self.0.push(b)
    }

    /// Creates a new [`ByteString`] with at least the given capacity.
    #[unstable(feature = "bstr", issue = "134915")]
    #[inline]
    pub fn with_capacity(capacity: usize) -> ByteString {
        ByteString(Vec::with_capacity(capacity))
    }

    /// Truncates the byte string to zero length.
    #[unstable(feature = "bstr", issue = "134915")]
    #[inline]
    pub fn clear(&mut self) {
        self.0.clear()
    }

    /// Returns the number of bytes that can be pushed to this `ByteString` without reallocating.
    #[unstable(feature = "bstr", issue = "134915")]
    #[inline]
    pub fn capacity(&self) -> usize {
        self.0.capacity()
    }

    /// Reserves capacity for at least `additional` more capacity to be inserted in the given
    /// `ByteString`. Does nothing if the capacity is already sufficient.
    ///
    /// The collection may reserve more space to speculatively avoid frequent reallocations.
    #[unstable(feature = "bstr", issue = "134915")]
    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        self.0.reserve(additional)
    }

    /// Tries to reserve capacity for at least `additional` more bytes
    /// in the given `ByteString`. The string may reserve more space to speculatively avoid
    /// frequent reallocations. After calling `try_reserve`, capacity will be
    /// greater than or equal to `self.len() + additional` if it returns `Ok(())`.
    /// Does nothing if capacity is already sufficient. This method preserves
    /// the contents even if an error occurs.
    ///
    /// # Errors
    ///
    /// If the capacity overflows, or the allocator reports a failure, then an error
    /// is returned.
    #[unstable(feature = "bstr", issue = "134915")]
    #[inline]
    pub fn try_reserve(&mut self, additional: usize) -> Result<(), TryReserveError> {
        self.0.try_reserve(additional)
    }

    /// Reserves the minimum capacity for at least `additional` more bytes to
    /// be inserted in the given `ByteString`. Does nothing if the capacity is
    /// already sufficient.
    ///
    /// Note that the allocator may give the collection more space than it
    /// requests. Therefore, capacity can not be relied upon to be precisely
    /// minimal. Prefer [`reserve`] if future insertions are expected.
    ///
    /// [`reserve`]: ByteString::reserve
    #[unstable(feature = "bstr", issue = "134915")]
    #[inline]
    pub fn reserve_exact(&mut self, additional: usize) {
        self.0.reserve_exact(additional)
    }

    /// Tries to reserve the minimum capacity for at least `additional`
    /// more bytes in the given `ByteString`. After calling
    /// `try_reserve_exact`, capacity will be greater than or equal to
    /// `self.len() + additional` if it returns `Ok(())`.
    /// Does nothing if the capacity is already sufficient.
    ///
    /// Note that the allocator may give the `ByteString` more space than it
    /// requests. Therefore, capacity can not be relied upon to be precisely
    /// minimal. Prefer [`try_reserve`] if future insertions are expected.
    ///
    /// [`try_reserve`]: ByteString::try_reserve
    ///
    /// # Errors
    ///
    /// If the capacity overflows, or the allocator reports a failure, then an error
    /// is returned.
    #[unstable(feature = "bstr", issue = "134915")]
    #[inline]
    pub fn try_reserve_exact(&mut self, additional: usize) -> Result<(), TryReserveError> {
        self.0.try_reserve_exact(additional)
    }

    /// Shrinks the capacity of the `ByteString` to match its length.
    #[unstable(feature = "bstr", issue = "134915")]
    #[inline]
    pub fn shrink_to_fit(&mut self) {
        self.0.shrink_to_fit()
    }

    /// Shrinks the capacity of the [`ByteString`] with a lower bound.
    ///
    /// The capacity will remain at least as large as both the length and the supplied value.
    ///
    /// If the current capacity is less than the lower limit, this is a no-op.
    #[unstable(feature = "bstr", issue = "134915")]
    #[inline]
    pub fn shrink_to(&mut self, min_capacity: usize) {
        self.0.shrink_to(min_capacity)
    }

    /// Consumes and leaks the `ByteString`, returning a mutable reference to the contents,
    /// `&'a mut ByteStr`.
    ///
    /// The caller has free choice over the returned lifetime, including `’static`. Indeed, this function is ideally used for data that lives for the remainder of the program’s life, as dropping the returned reference will cause a memory leak.
    ///
    /// It does not reallocate or shrink the `ByteString`, so the leaked allocation may include
    /// unused capacity that is not part of the returned slice. If you want discard excess capacity,
    /// call [`into_boxed_byte_str`], and then [`Box::leak`] instead. However, keep in mind that
    /// trimming the capacity may result in a reallocation and copy.
    ///
    /// [`into_boxed_byte_str`]: ByteString::into_boxed_byte_str
    #[unstable(feature = "bstr", issue = "134915")]
    #[inline]
    pub fn leak<'a>(self) -> &'a mut ByteStr {
        ByteStr::new_mut(self.0.leak())
    }

    /// Truncate the [`ByteString`] to the specified length.
    #[unstable(feature = "bstr", issue = "134915")]
    #[inline]
    pub fn truncate(&mut self, len: usize) {
        self.0.truncate(len)
    }
}

impl ByteStr {
    /// Converts a `ByteStr` to a <code>[Cow]<[str]></code>.
    ///
    /// Any non-UTF-8 sequences are replaced with
    /// [U+FFFD REPLACEMENT CHARACTER][char::REPLACEMENT_CHARACTER].
    ///
    /// # Examples
    ///
    /// Calling `to_string_lossy` on a `ByteStr` with invalid unicode:
    ///
    /// ```
    /// #![feature(bstr)]
    /// use std::bstr::ByteStr;
    ///
    /// let bstr = ByteStr::new(b"hello, \x80\x80!");
    /// assert_eq!(bstr.to_string_lossy(), "hello, ��!");
    /// ```
    #[unstable(feature = "bstr", issue = "134915")]
    #[inline]
    #[rustc_allow_incoherent_impl]
    pub fn to_string_lossy(&self) -> Cow<'_, str> {
        String::from_utf8_lossy(self.as_bytes())
    }

    /// Converts a `ByteStr` to an owned [`ByteString`].
    #[unstable(feature = "bstr", issue = "134915")]
    #[inline]
    #[rustc_allow_incoherent_impl]
    pub fn to_byte_string(&self) -> ByteString {
        ByteString(self.as_bytes().to_vec())
    }

    /// Returns an owned [`ByteString`] containing a copy of this string where each byte
    /// is mapped to its ASCII upper case equivalent.
    ///
    /// ASCII letters 'a' to 'z' are mapped to 'A' to 'Z',
    /// but non-ASCII letters are unchanged.
    ///
    /// To uppercase the value in-place, use [`make_ascii_uppercase`].
    ///
    /// [`make_ascii_uppercase`]: ByteStr::make_ascii_uppercase
    #[unstable(feature = "bstr", issue = "134915")]
    #[must_use = "this returns the uppercase bytes as a new ByteString, \
                  without modifying the original"]
    #[inline]
    #[rustc_allow_incoherent_impl]
    pub fn to_ascii_uppercase(&self) -> ByteString {
        let mut me = self.to_byte_string();
        me.make_ascii_uppercase();
        me
    }

    /// Returns an owned [`ByteString`] containing a copy of this string where each byte
    /// is mapped to its ASCII lower case equivalent.
    ///
    /// ASCII letters 'A' to 'Z' are mapped to 'a' to 'z',
    /// but non-ASCII letters are unchanged.
    ///
    /// To lowercase the value in-place, use [`make_ascii_lowercase`].
    ///
    /// [`make_ascii_lowercase`]: ByteStr::make_ascii_lowercase
    #[unstable(feature = "bstr", issue = "134915")]
    #[must_use = "this returns the uppercase bytes as a new ByteString, \
                  without modifying the original"]
    #[inline]
    #[rustc_allow_incoherent_impl]
    pub fn to_ascii_lowercase(&self) -> ByteString {
        let mut me = self.to_byte_string();
        me.make_ascii_lowercase();
        me
    }

    /// Converts a <code>[Box]<[ByteStr]></code> into a <code>[Box]<\[u8\]></code> without
    /// copying or allocating.
    #[unstable(feature = "bstr", issue = "134915")]
    #[inline]
    #[rustc_allow_incoherent_impl]
    pub fn into_boxed_bytes(self: Box<Self>) -> Box<[u8]> {
        // SAFETY: `ByteStr` is a transparent wrapper around `[u8]`.
        unsafe { Box::from_raw(Box::into_raw(self) as _) }
    }

    /// Converts a <code>[Box]<[ByteStr]></code> to an owned [`ByteString`].
    #[unstable(feature = "bstr", issue = "134915")]
    #[inline]
    #[rustc_allow_incoherent_impl]
    pub fn into_byte_string(self: Box<Self>) -> ByteString {
        ByteString(self.into_boxed_bytes().into_vec())
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl Deref for ByteString {
    type Target = ByteStr;

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.as_byte_str()
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl DerefMut for ByteString {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_byte_str()
    }
}

#[unstable(feature = "deref_pure_trait", issue = "87121")]
unsafe impl DerefPure for ByteString {}

#[unstable(feature = "bstr", issue = "134915")]
impl fmt::Debug for ByteString {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self.as_byte_str(), f)
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl fmt::Display for ByteString {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self.as_byte_str(), f)
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl AsRef<[u8]> for ByteString {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        self.as_bytes()
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl AsRef<ByteStr> for ByteString {
    #[inline]
    fn as_ref(&self) -> &ByteStr {
        self.as_byte_str()
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl AsMut<[u8]> for ByteString {
    #[inline]
    fn as_mut(&mut self) -> &mut [u8] {
        self.as_bytes_mut()
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl AsMut<ByteStr> for ByteString {
    #[inline]
    fn as_mut(&mut self) -> &mut ByteStr {
        self.as_mut_byte_str()
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl Borrow<[u8]> for ByteString {
    #[inline]
    fn borrow(&self) -> &[u8] {
        self.as_bytes()
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl Borrow<ByteStr> for ByteString {
    #[inline]
    fn borrow(&self) -> &ByteStr {
        self.as_byte_str()
    }
}

// `impl Borrow<ByteStr> for Vec<u8>` omitted to avoid inference failures
// `impl Borrow<ByteStr> for String` omitted to avoid inference failures

#[unstable(feature = "bstr", issue = "134915")]
impl BorrowMut<[u8]> for ByteString {
    #[inline]
    fn borrow_mut(&mut self) -> &mut [u8] {
        self.as_bytes_mut()
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl BorrowMut<ByteStr> for ByteString {
    #[inline]
    fn borrow_mut(&mut self) -> &mut ByteStr {
        self.as_mut_byte_str()
    }
}

// `impl BorrowMut<ByteStr> for Vec<u8>` omitted to avoid inference failures

#[unstable(feature = "bstr", issue = "134915")]
impl Default for ByteString {
    fn default() -> Self {
        ByteString::new()
    }
}

// Omitted due to inference failures
//
// #[unstable(feature = "bstr", issue = "134915")]
// impl<'a, const N: usize> From<&'a [u8; N]> for ByteString {
//     #[inline]
//     fn from(s: &'a [u8; N]) -> Self {
//         ByteString(s.as_slice().to_vec())
//     }
// }
//
// #[unstable(feature = "bstr", issue = "134915")]
// impl<const N: usize> From<[u8; N]> for ByteString {
//     #[inline]
//     fn from(s: [u8; N]) -> Self {
//         ByteString(s.as_slice().to_vec())
//     }
// }
//
// #[unstable(feature = "bstr", issue = "134915")]
// impl<'a> From<&'a [u8]> for ByteString {
//     #[inline]
//     fn from(s: &'a [u8]) -> Self {
//         ByteString(s.to_vec())
//     }
// }
//
// #[unstable(feature = "bstr", issue = "134915")]
// impl From<Vec<u8>> for ByteString {
//     #[inline]
//     fn from(s: Vec<u8>) -> Self {
//         ByteString(s)
//     }
// }

#[unstable(feature = "bstr", issue = "134915")]
impl From<ByteString> for Vec<u8> {
    #[inline]
    fn from(s: ByteString) -> Self {
        s.0
    }
}

// Omitted due to inference failures
//
// #[unstable(feature = "bstr", issue = "134915")]
// impl<'a> From<&'a str> for ByteString {
//     #[inline]
//     fn from(s: &'a str) -> Self {
//         ByteString(s.as_bytes().to_vec())
//     }
// }
//
// #[unstable(feature = "bstr", issue = "134915")]
// impl From<String> for ByteString {
//     #[inline]
//     fn from(s: String) -> Self {
//         ByteString(s.into_bytes())
//     }
// }

#[unstable(feature = "bstr", issue = "134915")]
impl<'a> From<&'a ByteStr> for ByteString {
    #[inline]
    fn from(s: &'a ByteStr) -> Self {
        s.to_byte_string()
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl<'a> From<ByteString> for Cow<'a, ByteStr> {
    #[inline]
    fn from(s: ByteString) -> Self {
        Cow::Owned(s)
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl<'a> From<&'a ByteString> for Cow<'a, ByteStr> {
    #[inline]
    fn from(s: &'a ByteString) -> Self {
        Cow::Borrowed(s.as_byte_str())
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl FromIterator<char> for ByteString {
    #[inline]
    fn from_iter<T: IntoIterator<Item = char>>(iter: T) -> Self {
        ByteString(iter.into_iter().collect::<String>().into_bytes())
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl FromIterator<u8> for ByteString {
    #[inline]
    fn from_iter<T: IntoIterator<Item = u8>>(iter: T) -> Self {
        ByteString(iter.into_iter().collect())
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl<'a> FromIterator<&'a str> for ByteString {
    #[inline]
    fn from_iter<T: IntoIterator<Item = &'a str>>(iter: T) -> Self {
        ByteString(iter.into_iter().collect::<String>().into_bytes())
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl<'a> FromIterator<&'a [u8]> for ByteString {
    #[inline]
    fn from_iter<T: IntoIterator<Item = &'a [u8]>>(iter: T) -> Self {
        let mut buf = ByteString::new();
        for b in iter {
            buf.push(ByteStr::new(b));
        }
        buf
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl<'a> FromIterator<&'a ByteStr> for ByteString {
    #[inline]
    fn from_iter<T: IntoIterator<Item = &'a ByteStr>>(iter: T) -> Self {
        let mut buf = ByteString::new();
        for b in iter {
            buf.push(b);
        }
        buf
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl FromIterator<ByteString> for ByteString {
    #[inline]
    fn from_iter<T: IntoIterator<Item = ByteString>>(iter: T) -> Self {
        let mut buf = Vec::new();
        for mut b in iter {
            buf.append(&mut b.0);
        }
        ByteString(buf)
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl FromStr for ByteString {
    type Err = core::convert::Infallible;

    #[inline]
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(ByteString(s.as_bytes().to_vec()))
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl Index<usize> for ByteString {
    type Output = u8;

    #[inline]
    fn index(&self, idx: usize) -> &u8 {
        &self.0[idx]
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl Index<RangeFull> for ByteString {
    type Output = ByteStr;

    #[inline]
    fn index(&self, _: RangeFull) -> &ByteStr {
        self.as_byte_str()
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl Index<Range<usize>> for ByteString {
    type Output = ByteStr;

    #[inline]
    fn index(&self, r: Range<usize>) -> &ByteStr {
        ByteStr::new(&self.0[r])
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl Index<RangeInclusive<usize>> for ByteString {
    type Output = ByteStr;

    #[inline]
    fn index(&self, r: RangeInclusive<usize>) -> &ByteStr {
        ByteStr::new(&self.0[r])
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl Index<RangeFrom<usize>> for ByteString {
    type Output = ByteStr;

    #[inline]
    fn index(&self, r: RangeFrom<usize>) -> &ByteStr {
        ByteStr::new(&self.0[r])
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl Index<RangeTo<usize>> for ByteString {
    type Output = ByteStr;

    #[inline]
    fn index(&self, r: RangeTo<usize>) -> &ByteStr {
        ByteStr::new(&self.0[r])
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl Index<RangeToInclusive<usize>> for ByteString {
    type Output = ByteStr;

    #[inline]
    fn index(&self, r: RangeToInclusive<usize>) -> &ByteStr {
        ByteStr::new(&self.0[r])
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl IndexMut<usize> for ByteString {
    #[inline]
    fn index_mut(&mut self, idx: usize) -> &mut u8 {
        &mut self.0[idx]
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl IndexMut<RangeFull> for ByteString {
    #[inline]
    fn index_mut(&mut self, _: RangeFull) -> &mut ByteStr {
        self.as_mut_byte_str()
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl IndexMut<Range<usize>> for ByteString {
    #[inline]
    fn index_mut(&mut self, r: Range<usize>) -> &mut ByteStr {
        ByteStr::new_mut(&mut self.0[r])
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl IndexMut<RangeInclusive<usize>> for ByteString {
    #[inline]
    fn index_mut(&mut self, r: RangeInclusive<usize>) -> &mut ByteStr {
        ByteStr::new_mut(&mut self.0[r])
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl IndexMut<RangeFrom<usize>> for ByteString {
    #[inline]
    fn index_mut(&mut self, r: RangeFrom<usize>) -> &mut ByteStr {
        ByteStr::new_mut(&mut self.0[r])
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl IndexMut<RangeTo<usize>> for ByteString {
    #[inline]
    fn index_mut(&mut self, r: RangeTo<usize>) -> &mut ByteStr {
        ByteStr::new_mut(&mut self.0[r])
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl IndexMut<RangeToInclusive<usize>> for ByteString {
    #[inline]
    fn index_mut(&mut self, r: RangeToInclusive<usize>) -> &mut ByteStr {
        ByteStr::new_mut(&mut self.0[r])
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl hash::Hash for ByteString {
    #[inline]
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl Eq for ByteString {}

#[unstable(feature = "bstr", issue = "134915")]
impl PartialEq for ByteString {
    #[inline]
    fn eq(&self, other: &ByteString) -> bool {
        self.0 == other.0
    }
}

macro_rules! impl_partial_eq_ord_cow {
    ($lhs:ty, $rhs:ty) => {
        #[allow(unused_lifetimes)]
        #[unstable(feature = "bstr", issue = "134915")]
        impl<'a> PartialEq<$rhs> for $lhs {
            #[inline]
            fn eq(&self, other: &$rhs) -> bool {
                let other: &[u8] = (&**other).as_ref();
                PartialEq::eq(self.as_bytes(), other)
            }
        }

        #[allow(unused_lifetimes)]
        #[unstable(feature = "bstr", issue = "134915")]
        impl<'a> PartialEq<$lhs> for $rhs {
            #[inline]
            fn eq(&self, other: &$lhs) -> bool {
                let this: &[u8] = (&**self).as_ref();
                PartialEq::eq(this, other.as_bytes())
            }
        }

        #[allow(unused_lifetimes)]
        #[unstable(feature = "bstr", issue = "134915")]
        impl<'a> PartialOrd<$rhs> for $lhs {
            #[inline]
            fn partial_cmp(&self, other: &$rhs) -> Option<Ordering> {
                let other: &[u8] = (&**other).as_ref();
                PartialOrd::partial_cmp(self.as_bytes(), other)
            }
        }

        #[allow(unused_lifetimes)]
        #[unstable(feature = "bstr", issue = "134915")]
        impl<'a> PartialOrd<$lhs> for $rhs {
            #[inline]
            fn partial_cmp(&self, other: &$lhs) -> Option<Ordering> {
                let this: &[u8] = (&**self).as_ref();
                PartialOrd::partial_cmp(this, other.as_bytes())
            }
        }
    };
}

// PartialOrd with `Vec<u8>` omitted to avoid inference failures
impl_partial_eq!(ByteString, Vec<u8>);
// PartialOrd with `[u8]` omitted to avoid inference failures
impl_partial_eq!(ByteString, [u8]);
// PartialOrd with `&[u8]` omitted to avoid inference failures
impl_partial_eq!(ByteString, &[u8]);
// PartialOrd with `String` omitted to avoid inference failures
impl_partial_eq!(ByteString, String);
// PartialOrd with `str` omitted to avoid inference failures
impl_partial_eq!(ByteString, str);
// PartialOrd with `&str` omitted to avoid inference failures
impl_partial_eq!(ByteString, &str);
impl_partial_eq_ord!(ByteString, ByteStr);
impl_partial_eq_ord!(ByteString, &ByteStr);
// PartialOrd with `[u8; N]` omitted to avoid inference failures
impl_partial_eq_n!(ByteString, [u8; N]);
// PartialOrd with `&[u8; N]` omitted to avoid inference failures
impl_partial_eq_n!(ByteString, &[u8; N]);
impl_partial_eq_ord_cow!(ByteString, Cow<'_, ByteStr>);
impl_partial_eq_ord_cow!(ByteString, Cow<'_, str>);
impl_partial_eq_ord_cow!(ByteString, Cow<'_, [u8]>);

#[unstable(feature = "bstr", issue = "134915")]
impl Ord for ByteString {
    #[inline]
    fn cmp(&self, other: &ByteString) -> Ordering {
        Ord::cmp(&self.0, &other.0)
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl PartialOrd for ByteString {
    #[inline]
    fn partial_cmp(&self, other: &ByteString) -> Option<Ordering> {
        PartialOrd::partial_cmp(&self.0, &other.0)
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl ToOwned for ByteStr {
    type Owned = ByteString;

    #[inline]
    fn to_owned(&self) -> ByteString {
        self.to_byte_string()
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl TryFrom<ByteString> for String {
    type Error = crate::string::FromUtf8Error;

    #[inline]
    fn try_from(s: ByteString) -> Result<Self, Self::Error> {
        String::from_utf8(s.0)
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl<'a> TryFrom<&'a ByteString> for &'a str {
    type Error = crate::str::Utf8Error;

    #[inline]
    fn try_from(s: &'a ByteString) -> Result<Self, Self::Error> {
        crate::str::from_utf8(s.0.as_slice())
    }
}

// Additional impls for `ByteStr` that require types from `alloc`:

#[unstable(feature = "bstr", issue = "134915")]
impl Clone for Box<ByteStr> {
    #[inline]
    fn clone(&self) -> Self {
        Self::from(Box::<[u8]>::from(self.as_bytes()))
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl<'a> From<&'a ByteStr> for Cow<'a, ByteStr> {
    #[inline]
    fn from(s: &'a ByteStr) -> Self {
        Cow::Borrowed(s)
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl From<Box<[u8]>> for Box<ByteStr> {
    #[inline]
    fn from(s: Box<[u8]>) -> Box<ByteStr> {
        // SAFETY: `ByteStr` is a transparent wrapper around `[u8]`.
        unsafe { Box::from_raw(Box::into_raw(s) as _) }
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl From<Box<ByteStr>> for Box<[u8]> {
    #[inline]
    fn from(s: Box<ByteStr>) -> Box<[u8]> {
        // SAFETY: `ByteStr` is a transparent wrapper around `[u8]`.
        unsafe { Box::from_raw(Box::into_raw(s) as _) }
    }
}

#[unstable(feature = "bstr", issue = "134915")]
#[cfg(not(no_rc))]
impl From<Rc<[u8]>> for Rc<ByteStr> {
    #[inline]
    fn from(s: Rc<[u8]>) -> Rc<ByteStr> {
        // SAFETY: `ByteStr` is a transparent wrapper around `[u8]`.
        unsafe { Rc::from_raw(Rc::into_raw(s) as _) }
    }
}

#[unstable(feature = "bstr", issue = "134915")]
#[cfg(not(no_rc))]
impl From<Rc<ByteStr>> for Rc<[u8]> {
    #[inline]
    fn from(s: Rc<ByteStr>) -> Rc<[u8]> {
        // SAFETY: `ByteStr` is a transparent wrapper around `[u8]`.
        unsafe { Rc::from_raw(Rc::into_raw(s) as _) }
    }
}

#[unstable(feature = "bstr", issue = "134915")]
#[cfg(all(not(no_rc), not(no_sync), target_has_atomic = "ptr"))]
impl From<Arc<[u8]>> for Arc<ByteStr> {
    #[inline]
    fn from(s: Arc<[u8]>) -> Arc<ByteStr> {
        // SAFETY: `ByteStr` is a transparent wrapper around `[u8]`.
        unsafe { Arc::from_raw(Arc::into_raw(s) as _) }
    }
}

#[unstable(feature = "bstr", issue = "134915")]
#[cfg(all(not(no_rc), not(no_sync), target_has_atomic = "ptr"))]
impl From<Arc<ByteStr>> for Arc<[u8]> {
    #[inline]
    fn from(s: Arc<ByteStr>) -> Arc<[u8]> {
        // SAFETY: `ByteStr` is a transparent wrapper around `[u8]`.
        unsafe { Arc::from_raw(Arc::into_raw(s) as _) }
    }
}

// PartialOrd with `Vec<u8>` omitted to avoid inference failures
impl_partial_eq!(ByteStr, Vec<u8>);
// PartialOrd with `String` omitted to avoid inference failures
impl_partial_eq!(ByteStr, String);
impl_partial_eq_ord_cow!(&'a ByteStr, Cow<'a, ByteStr>);
impl_partial_eq_ord_cow!(&'a ByteStr, Cow<'a, str>);
impl_partial_eq_ord_cow!(&'a ByteStr, Cow<'a, [u8]>);

#[unstable(feature = "bstr", issue = "134915")]
impl<'a> TryFrom<&'a ByteStr> for String {
    type Error = core::str::Utf8Error;

    #[inline]
    fn try_from(s: &'a ByteStr) -> Result<Self, Self::Error> {
        Ok(core::str::from_utf8(s.as_bytes())?.into())
    }
}
