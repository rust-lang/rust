//! The `ByteStr` and `ByteString` types and trait implementations.

// This could be more fine-grained.
#![cfg(not(no_global_oom_handling))]

use core::borrow::{Borrow, BorrowMut};
#[unstable(feature = "byte_str", issue = "134915")]
pub use core::byte_str::ByteStr;
use core::byte_str::{impl_partial_eq, impl_partial_eq_n, impl_partial_eq_ord};
use core::cmp::Ordering;
use core::ops::{
    Deref, DerefMut, DerefPure, Index, IndexMut, Range, RangeFrom, RangeFull, RangeInclusive,
    RangeTo, RangeToInclusive,
};
use core::str::{FromStr, Utf8Error};
use core::{fmt, hash};

use crate::alloc::{Allocator, Global};
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
/// borrowed byte string, see [`ByteStr`](../../std/byte_str/struct.ByteStr.html).
///
/// `ByteString` implements `Deref` to `&Vec<u8>`, so all methods available on `&Vec<u8>` are
/// available on `ByteString`. Similarly, `ByteString` implements `DerefMut` to `&mut Vec<u8>`,
/// so you can modify a `ByteString` using any method available on `&mut Vec<u8>`.
///
/// The `Debug` and `Display` implementations for `ByteString` are the same as those for `ByteStr`,
/// showing invalid UTF-8 as hex escapes or the Unicode replacement character, respectively.
#[unstable(feature = "byte_str", issue = "134915")]
#[repr(transparent)]
#[doc(alias = "BString")]
#[derive(Clone)]
pub struct ByteString<
    #[unstable(feature = "byte_str_allocator", issue = "134915")] A: Allocator = Global,
>(pub(crate) Vec<u8, A>);

impl ByteString<Global> {
    /// Creates an empty `ByteString`.
    #[unstable(feature = "byte_str", issue = "134915")]
    #[inline]
    #[rustc_const_unstable(feature = "byte_str", issue = "134915")]
    pub const fn new() -> ByteString {
        ByteString(Vec::new())
    }

    /// Creates a new [`ByteString`] with at least the given capacity.
    #[unstable(feature = "byte_str", issue = "134915")]
    #[inline]
    pub fn with_capacity(capacity: usize) -> ByteString {
        ByteString(Vec::with_capacity(capacity))
    }

    /// Converts a `ByteString` into a [`String`] if it contains valid Unicode data.
    ///
    /// On failure, ownership of the original `ByteString` is returned.
    #[unstable(feature = "byte_str", issue = "134915")]
    #[inline]
    pub fn into_string(self) -> Result<String, ByteString> {
        String::from_utf8(self.0).map_err(|e| ByteString(e.into_bytes()))
    }

    /// Try to get a `String` representation of the `&ByteString`, if it is
    /// valid UTF-8.
    ///
    /// This method is named `to_string()` because we want `ByteString` to
    /// implement `Display`, but the `ToString` trait has a blanket
    /// implementation for types that implement `Display`, and the trait version
    /// will use the Unicode replacement character rather than returning a
    /// `Result` and allowing for the possibility of the content not being UTF-8.
    #[unstable(feature = "byte_str_to_string", issue = "134915")]
    #[rustc_allow_incoherent_impl]
    pub fn to_string(&self) -> Result<String, Utf8Error> {
        // Avoid allocating a copy of the contents for invalid UTF-8
        if let Err(e) = str::from_utf8(&self.0) {
            return Err(e);
        }
        // SAFETY: we just checked that the contents are valid UTF-8
        Ok(unsafe { String::from_utf8_unchecked(self.0.clone()) })
    }
}

impl<A: Allocator> ByteString<A> {
    /// Creates an empty `ByteString` with a particular allocator.
    #[unstable(feature = "byte_str", issue = "134915")]
    #[inline]
    #[rustc_const_unstable(feature = "byte_str", issue = "134915")]
    pub const fn new_in(alloc: A) -> ByteString<A> {
        ByteString(Vec::new_in(alloc))
    }

    /// Creates a new [`ByteString`] with at least the given capacity and the given allocator.
    #[unstable(feature = "byte_str", issue = "134915")]
    #[inline]
    pub fn with_capacity_in(capacity: usize, alloc: A) -> ByteString<A> {
        ByteString(Vec::with_capacity_in(capacity, alloc))
    }

    /// Converts to a [`ByteStr`] slice.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(byte_str)]
    /// use std::byte_str::ByteStr;
    ///
    /// let byte_str = ByteStr::new("foo");
    /// let byte_string = byte_str.to_byte_string();
    /// assert_eq!(byte_string.as_byte_str(), byte_str);
    /// ```
    #[unstable(feature = "byte_str", issue = "134915")]
    #[inline]
    pub fn as_byte_str(&self) -> &ByteStr {
        ByteStr::new(&self.0)
    }

    /// Converts to a mutable [`ByteStr`] slice.
    #[unstable(feature = "byte_str", issue = "134915")]
    #[inline]
    pub fn as_mut_byte_str(&mut self) -> &mut ByteStr {
        ByteStr::new_mut(&mut self.0)
    }

    /// Returns a reference to the underlying vector for this `ByteString`.
    #[unstable(feature = "byte_str", issue = "134915")]
    #[inline]
    pub fn as_vec(&self) -> &Vec<u8, A> {
        &self.0
    }

    /// Returns a mutable reference to the underlying vector for this `ByteString`.
    #[unstable(feature = "byte_str", issue = "134915")]
    #[inline]
    pub fn as_mut_vec(&mut self) -> &mut Vec<u8, A> {
        &mut self.0
    }

    /// Converts to a vector of bytes.
    #[unstable(feature = "byte_str", issue = "134915")]
    #[inline]
    pub fn into_bytes(self) -> Vec<u8, A> {
        self.0
    }

    /// Converts to a boxed slice of bytes.
    #[unstable(feature = "byte_str", issue = "134915")]
    #[inline]
    pub fn into_boxed_bytes(self) -> Box<[u8], A> {
        self.into_bytes().into_boxed_slice()
    }

    /// Converts to a boxed byte string.
    #[unstable(feature = "byte_str", issue = "134915")]
    #[inline]
    pub fn into_boxed_byte_str(self) -> Box<ByteStr, A> {
        self.into_bytes().into_boxed_slice().into_boxed_byte_str()
    }

    /// Extends the byte string with the given <code>&[ByteStr]</code> slice.
    #[unstable(feature = "byte_str", issue = "134915")]
    #[inline]
    pub fn push<S: AsRef<ByteStr>>(&mut self, s: S) {
        self.0.extend_from_slice(s.as_ref().as_bytes())
    }

    /// Pushes a single byte onto the byte string.
    #[unstable(feature = "byte_str", issue = "134915")]
    #[inline]
    pub fn push_byte(&mut self, b: u8) {
        self.0.push(b)
    }

    /// Truncates the byte string to zero length.
    #[unstable(feature = "byte_str", issue = "134915")]
    #[inline]
    pub fn clear(&mut self) {
        self.0.clear()
    }

    /// Returns the number of bytes that can be pushed to this `ByteString` without reallocating.
    #[unstable(feature = "byte_str", issue = "134915")]
    #[inline]
    pub fn capacity(&self) -> usize {
        self.0.capacity()
    }

    /// Reserves capacity for at least `additional` more capacity to be inserted in the given
    /// `ByteString`. Does nothing if the capacity is already sufficient.
    ///
    /// The collection may reserve more space to speculatively avoid frequent reallocations.
    #[unstable(feature = "byte_str", issue = "134915")]
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
    #[unstable(feature = "byte_str", issue = "134915")]
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
    #[unstable(feature = "byte_str", issue = "134915")]
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
    #[unstable(feature = "byte_str", issue = "134915")]
    #[inline]
    pub fn try_reserve_exact(&mut self, additional: usize) -> Result<(), TryReserveError> {
        self.0.try_reserve_exact(additional)
    }

    /// Shrinks the capacity of the `ByteString` to match its length.
    #[unstable(feature = "byte_str", issue = "134915")]
    #[inline]
    pub fn shrink_to_fit(&mut self) {
        self.0.shrink_to_fit()
    }

    /// Shrinks the capacity of the [`ByteString`] with a lower bound.
    ///
    /// The capacity will remain at least as large as both the length and the supplied value.
    ///
    /// If the current capacity is less than the lower limit, this is a no-op.
    #[unstable(feature = "byte_str", issue = "134915")]
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
    #[unstable(feature = "byte_str", issue = "134915")]
    #[inline]
    pub fn leak<'a>(self) -> &'a mut ByteStr
    where
        A: 'a,
    {
        ByteStr::new_mut(self.0.leak())
    }

    /// Truncate the [`ByteString`] to the specified length.
    #[unstable(feature = "byte_str", issue = "134915")]
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
    /// #![feature(byte_str)]
    /// use std::byte_str::ByteStr;
    ///
    /// let byte_str = ByteStr::new(b"hello, \x80\x80!");
    /// assert_eq!(byte_str.to_string_lossy(), "hello, ��!");
    /// ```
    #[unstable(feature = "byte_str", issue = "134915")]
    #[inline]
    #[rustc_allow_incoherent_impl]
    pub fn to_string_lossy(&self) -> Cow<'_, str> {
        String::from_utf8_lossy(self.as_bytes())
    }

    /// Converts a `ByteStr` to an owned [`Vec`] of bytes.
    #[unstable(feature = "byte_str", issue = "134915")]
    #[inline]
    #[rustc_allow_incoherent_impl]
    pub fn to_bytes(&self) -> Vec<u8> {
        self.as_bytes().to_vec()
    }

    /// Converts a `ByteStr` to an owned [`ByteString`].
    #[unstable(feature = "byte_str", issue = "134915")]
    #[inline]
    #[rustc_allow_incoherent_impl]
    pub fn to_byte_string(&self) -> ByteString {
        ByteString(self.to_bytes())
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
    #[unstable(feature = "byte_str", issue = "134915")]
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
    #[unstable(feature = "byte_str", issue = "134915")]
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
    #[unstable(feature = "byte_str", issue = "134915")]
    #[inline]
    #[rustc_allow_incoherent_impl]
    pub fn into_boxed_bytes<A: Allocator>(self: Box<Self, A>) -> Box<[u8], A> {
        let (raw, alloc) = Box::into_raw_with_allocator(self);
        // SAFETY: `ByteStr` is a transparent wrapper around `[u8]`.
        unsafe { Box::from_raw_in(raw as *mut [u8], alloc) }
    }

    /// Converts a <code>[Box]<[ByteStr]></code> to an owned [`ByteString`].
    #[unstable(feature = "byte_str", issue = "134915")]
    #[inline]
    #[rustc_allow_incoherent_impl]
    pub fn into_byte_string<A: Allocator>(self: Box<Self, A>) -> ByteString<A> {
        ByteString(self.into_boxed_bytes().into_vec())
    }
}

#[unstable(feature = "byte_str", issue = "134915")]
impl Deref for ByteString {
    type Target = ByteStr;

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.as_byte_str()
    }
}

#[unstable(feature = "byte_str", issue = "134915")]
impl DerefMut for ByteString {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_byte_str()
    }
}

#[unstable(feature = "deref_pure_trait", issue = "87121")]
unsafe impl DerefPure for ByteString {}

#[unstable(feature = "byte_str", issue = "134915")]
impl fmt::Debug for ByteString {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self.as_byte_str(), f)
    }
}

#[unstable(feature = "byte_str_to_string", issue = "134915")]
impl fmt::Display for ByteString {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self.as_byte_str(), f)
    }
}

#[unstable(feature = "byte_str", issue = "134915")]
impl AsRef<[u8]> for ByteString {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        self.as_bytes()
    }
}

#[unstable(feature = "byte_str", issue = "134915")]
impl AsRef<ByteStr> for ByteString {
    #[inline]
    fn as_ref(&self) -> &ByteStr {
        self.as_byte_str()
    }
}

#[unstable(feature = "byte_str", issue = "134915")]
impl AsMut<[u8]> for ByteString {
    #[inline]
    fn as_mut(&mut self) -> &mut [u8] {
        self.as_bytes_mut()
    }
}

#[unstable(feature = "byte_str", issue = "134915")]
impl AsMut<ByteStr> for ByteString {
    #[inline]
    fn as_mut(&mut self) -> &mut ByteStr {
        self.as_mut_byte_str()
    }
}

#[unstable(feature = "byte_str", issue = "134915")]
impl Borrow<[u8]> for ByteString {
    #[inline]
    fn borrow(&self) -> &[u8] {
        self.as_bytes()
    }
}

#[unstable(feature = "byte_str", issue = "134915")]
impl Borrow<ByteStr> for ByteString {
    #[inline]
    fn borrow(&self) -> &ByteStr {
        self.as_byte_str()
    }
}

// `impl Borrow<ByteStr> for Vec<u8>` omitted to avoid inference failures
// `impl Borrow<ByteStr> for String` omitted to avoid inference failures

#[unstable(feature = "byte_str", issue = "134915")]
impl BorrowMut<[u8]> for ByteString {
    #[inline]
    fn borrow_mut(&mut self) -> &mut [u8] {
        self.as_bytes_mut()
    }
}

#[unstable(feature = "byte_str", issue = "134915")]
impl BorrowMut<ByteStr> for ByteString {
    #[inline]
    fn borrow_mut(&mut self) -> &mut ByteStr {
        self.as_mut_byte_str()
    }
}

// `impl BorrowMut<ByteStr> for Vec<u8>` omitted to avoid inference failures

// `impl<'a, const N: usize> From<&'a [u8; N]> for ByteString` omitted to avoid inference failures
// `impl<const N: usize> From<[u8; N]> for ByteString` omitted to avoid inference failures
// `impl<'a> From<&'a [u8]> for ByteString` omitted to avoid inference failures
// `impl<A: Allocator> From<Vec<u8, A>> for ByteString<A>` omitted to avoid inference failures

#[unstable(feature = "byte_str", issue = "134915")]
impl<A: Allocator> From<ByteString<A>> for Vec<u8, A> {
    #[inline]
    fn from(s: ByteString<A>) -> Vec<u8, A> {
        s.0
    }
}

// `impl<'a> From<&'a str> for ByteString` omitted to avoid inference failures
// `impl From<String> for ByteString` omitted to avoid inference failures

#[unstable(feature = "byte_str", issue = "134915")]
impl<'a> From<&'a ByteStr> for ByteString {
    #[inline]
    fn from(s: &'a ByteStr) -> Self {
        s.to_byte_string()
    }
}

#[unstable(feature = "byte_str", issue = "134915")]
impl<'a> From<ByteString> for Cow<'a, ByteStr> {
    #[inline]
    fn from(s: ByteString) -> Self {
        Cow::Owned(s)
    }
}

#[unstable(feature = "byte_str", issue = "134915")]
impl<'a> From<&'a ByteString> for Cow<'a, ByteStr> {
    #[inline]
    fn from(s: &'a ByteString) -> Self {
        Cow::Borrowed(s.as_byte_str())
    }
}

#[unstable(feature = "byte_str", issue = "134915")]
impl FromIterator<char> for ByteString {
    #[inline]
    fn from_iter<I: IntoIterator<Item = char>>(iter: I) -> Self {
        ByteString(iter.into_iter().collect::<String>().into_bytes())
    }
}

#[unstable(feature = "byte_str", issue = "134915")]
impl FromIterator<u8> for ByteString {
    #[inline]
    fn from_iter<I: IntoIterator<Item = u8>>(iter: I) -> Self {
        ByteString(iter.into_iter().collect())
    }
}

#[unstable(feature = "byte_str", issue = "134915")]
impl<'a> FromIterator<&'a str> for ByteString {
    #[inline]
    fn from_iter<I: IntoIterator<Item = &'a str>>(iter: I) -> Self {
        ByteString(iter.into_iter().collect::<String>().into_bytes())
    }
}

#[unstable(feature = "byte_str", issue = "134915")]
impl<'a> FromIterator<&'a [u8]> for ByteString {
    #[inline]
    fn from_iter<I: IntoIterator<Item = &'a [u8]>>(iter: I) -> Self {
        let mut buf = ByteString::new();
        for b in iter {
            buf.push(ByteStr::new(b));
        }
        buf
    }
}

#[unstable(feature = "byte_str", issue = "134915")]
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

#[unstable(feature = "byte_str", issue = "134915")]
impl FromIterator<ByteString> for ByteString {
    #[inline]
    fn from_iter<I: IntoIterator<Item = ByteString>>(iter: I) -> Self {
        let mut buf = Vec::new();
        for mut b in iter {
            buf.append(&mut b.0);
        }
        ByteString(buf)
    }
}

#[unstable(feature = "byte_str", issue = "134915")]
impl FromStr for ByteString {
    type Err = !;

    #[inline]
    fn from_str(s: &str) -> Result<Self, !> {
        Ok(ByteString(s.as_bytes().to_vec()))
    }
}

#[unstable(feature = "byte_str", issue = "134915")]
impl<A: Allocator> Index<usize> for ByteString<A> {
    type Output = u8;

    #[inline]
    fn index(&self, idx: usize) -> &u8 {
        &self.0[idx]
    }
}

#[unstable(feature = "byte_str", issue = "134915")]
impl<A: Allocator> Index<RangeFull> for ByteString<A> {
    type Output = ByteStr;

    #[inline]
    fn index(&self, _: RangeFull) -> &ByteStr {
        self.as_byte_str()
    }
}

#[unstable(feature = "byte_str", issue = "134915")]
impl<A: Allocator> Index<Range<usize>> for ByteString<A> {
    type Output = ByteStr;

    #[inline]
    fn index(&self, r: Range<usize>) -> &ByteStr {
        ByteStr::new(&self.0[r])
    }
}

#[unstable(feature = "byte_str", issue = "134915")]
impl<A: Allocator> Index<RangeInclusive<usize>> for ByteString<A> {
    type Output = ByteStr;

    #[inline]
    fn index(&self, r: RangeInclusive<usize>) -> &ByteStr {
        ByteStr::new(&self.0[r])
    }
}

#[unstable(feature = "byte_str", issue = "134915")]
impl<A: Allocator> Index<RangeFrom<usize>> for ByteString<A> {
    type Output = ByteStr;

    #[inline]
    fn index(&self, r: RangeFrom<usize>) -> &ByteStr {
        ByteStr::new(&self.0[r])
    }
}

#[unstable(feature = "byte_str", issue = "134915")]
impl<A: Allocator> Index<RangeTo<usize>> for ByteString<A> {
    type Output = ByteStr;

    #[inline]
    fn index(&self, r: RangeTo<usize>) -> &ByteStr {
        ByteStr::new(&self.0[r])
    }
}

#[unstable(feature = "byte_str", issue = "134915")]
impl<A: Allocator> Index<RangeToInclusive<usize>> for ByteString<A> {
    type Output = ByteStr;

    #[inline]
    fn index(&self, r: RangeToInclusive<usize>) -> &ByteStr {
        ByteStr::new(&self.0[r])
    }
}

#[unstable(feature = "byte_str", issue = "134915")]
impl<A: Allocator> IndexMut<usize> for ByteString<A> {
    #[inline]
    fn index_mut(&mut self, idx: usize) -> &mut u8 {
        &mut self.0[idx]
    }
}

#[unstable(feature = "byte_str", issue = "134915")]
impl<A: Allocator> IndexMut<RangeFull> for ByteString<A> {
    #[inline]
    fn index_mut(&mut self, _: RangeFull) -> &mut ByteStr {
        self.as_mut_byte_str()
    }
}

#[unstable(feature = "byte_str", issue = "134915")]
impl<A: Allocator> IndexMut<Range<usize>> for ByteString<A> {
    #[inline]
    fn index_mut(&mut self, r: Range<usize>) -> &mut ByteStr {
        ByteStr::new_mut(&mut self.0[r])
    }
}

#[unstable(feature = "byte_str", issue = "134915")]
impl<A: Allocator> IndexMut<RangeInclusive<usize>> for ByteString<A> {
    #[inline]
    fn index_mut(&mut self, r: RangeInclusive<usize>) -> &mut ByteStr {
        ByteStr::new_mut(&mut self.0[r])
    }
}

#[unstable(feature = "byte_str", issue = "134915")]
impl<A: Allocator> IndexMut<RangeFrom<usize>> for ByteString<A> {
    #[inline]
    fn index_mut(&mut self, r: RangeFrom<usize>) -> &mut ByteStr {
        ByteStr::new_mut(&mut self.0[r])
    }
}

#[unstable(feature = "byte_str", issue = "134915")]
impl<A: Allocator> IndexMut<RangeTo<usize>> for ByteString<A> {
    #[inline]
    fn index_mut(&mut self, r: RangeTo<usize>) -> &mut ByteStr {
        ByteStr::new_mut(&mut self.0[r])
    }
}

#[unstable(feature = "byte_str", issue = "134915")]
impl<A: Allocator> IndexMut<RangeToInclusive<usize>> for ByteString<A> {
    #[inline]
    fn index_mut(&mut self, r: RangeToInclusive<usize>) -> &mut ByteStr {
        ByteStr::new_mut(&mut self.0[r])
    }
}

#[unstable(feature = "byte_str", issue = "134915")]
impl Default for ByteString {
    fn default() -> ByteString {
        ByteString::new()
    }
}

#[unstable(feature = "byte_str", issue = "134915")]
impl<A: Allocator> hash::Hash for ByteString<A> {
    #[inline]
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

#[unstable(feature = "byte_str", issue = "134915")]
impl<A: Allocator> Eq for ByteString<A> {}

#[unstable(feature = "byte_str", issue = "134915")]
impl<A1: Allocator, A2: Allocator> PartialEq<ByteString<A2>> for ByteString<A1> {
    #[inline]
    fn eq(&self, other: &ByteString<A2>) -> bool {
        self.0 == other.0
    }
}

macro_rules! impl_partial_eq_ord_cow {
    ($lhs:ty, $rhs:ty) => {
        #[unstable(feature = "byte_str", issue = "134915")]
        impl PartialEq<$rhs> for $lhs {
            #[inline]
            fn eq(&self, other: &$rhs) -> bool {
                let other: &[u8] = (&**other).as_ref();
                PartialEq::eq(self.as_bytes(), other)
            }
        }

        #[unstable(feature = "byte_str", issue = "134915")]
        impl PartialEq<$lhs> for $rhs {
            #[inline]
            fn eq(&self, other: &$lhs) -> bool {
                let this: &[u8] = (&**self).as_ref();
                PartialEq::eq(this, other.as_bytes())
            }
        }

        #[unstable(feature = "byte_str", issue = "134915")]
        impl PartialOrd<$rhs> for $lhs {
            #[inline]
            fn partial_cmp(&self, other: &$rhs) -> Option<Ordering> {
                let other: &[u8] = (&**other).as_ref();
                PartialOrd::partial_cmp(self.as_bytes(), other)
            }
        }

        #[unstable(feature = "byte_str", issue = "134915")]
        impl PartialOrd<$lhs> for $rhs {
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

#[unstable(feature = "byte_str", issue = "134915")]
impl<A: Allocator> Ord for ByteString<A> {
    #[inline]
    fn cmp(&self, other: &ByteString<A>) -> Ordering {
        Ord::cmp(&self.0, &other.0)
    }
}

#[unstable(feature = "byte_str", issue = "134915")]
impl<A1: Allocator, A2: Allocator> PartialOrd<ByteString<A2>> for ByteString<A1> {
    #[inline]
    fn partial_cmp(&self, other: &ByteString<A2>) -> Option<Ordering> {
        PartialOrd::partial_cmp(&self.0, &other.0)
    }
}

#[unstable(feature = "byte_str", issue = "134915")]
impl ToOwned for ByteStr {
    type Owned = ByteString;

    #[inline]
    fn to_owned(&self) -> ByteString {
        self.to_byte_string()
    }
}

#[unstable(feature = "byte_str", issue = "134915")]
impl TryFrom<ByteString> for String {
    type Error = crate::string::FromUtf8Error;

    #[inline]
    fn try_from(s: ByteString) -> Result<Self, Self::Error> {
        String::from_utf8(s.0)
    }
}

#[unstable(feature = "byte_str", issue = "134915")]
impl<'a> TryFrom<&'a ByteString> for &'a str {
    type Error = crate::str::Utf8Error;

    #[inline]
    fn try_from(s: &'a ByteString) -> Result<Self, Self::Error> {
        crate::str::from_utf8(s.0.as_slice())
    }
}

// Additional impls for `ByteStr` that require types from `alloc`:

#[unstable(feature = "byte_str", issue = "134915")]
impl<A: Allocator + Clone> Clone for Box<ByteStr, A> {
    #[inline]
    fn clone(&self) -> Self {
        Box::clone_from_ref_in(&**self, Self::allocator(self).clone())
    }
}

#[unstable(feature = "byte_str", issue = "134915")]
impl<'a> From<&'a ByteStr> for Cow<'a, ByteStr> {
    #[inline]
    fn from(s: &'a ByteStr) -> Self {
        Cow::Borrowed(s)
    }
}

#[unstable(feature = "byte_str", issue = "134915")]
impl<A: Allocator> From<Box<[u8], A>> for Box<ByteStr, A> {
    #[inline]
    fn from(s: Box<[u8], A>) -> Box<ByteStr, A> {
        let (raw, alloc) = Box::into_raw_with_allocator(s);
        // SAFETY: `ByteStr` is a transparent wrapper around `[u8]`.
        unsafe { Box::from_raw_in(raw as *mut ByteStr, alloc) }
    }
}

#[unstable(feature = "byte_str", issue = "134915")]
impl<A: Allocator> From<Box<ByteStr, A>> for Box<[u8], A> {
    #[inline]
    fn from(s: Box<ByteStr, A>) -> Box<[u8], A> {
        let (raw, alloc) = Box::into_raw_with_allocator(s);
        // SAFETY: `ByteStr` is a transparent wrapper around `[u8]`.
        unsafe { Box::from_raw_in(raw as *mut [u8], alloc) }
    }
}

#[unstable(feature = "byte_str", issue = "134915")]
#[cfg(not(no_rc))]
impl From<Rc<[u8]>> for Rc<ByteStr> {
    #[inline]
    fn from(s: Rc<[u8]>) -> Rc<ByteStr> {
        // SAFETY: `ByteStr` is a transparent wrapper around `[u8]`.
        unsafe { Rc::from_raw(Rc::into_raw(s) as _) }
    }
}

#[unstable(feature = "byte_str", issue = "134915")]
#[cfg(not(no_rc))]
impl From<Rc<ByteStr>> for Rc<[u8]> {
    #[inline]
    fn from(s: Rc<ByteStr>) -> Rc<[u8]> {
        // SAFETY: `ByteStr` is a transparent wrapper around `[u8]`.
        unsafe { Rc::from_raw(Rc::into_raw(s) as _) }
    }
}

#[unstable(feature = "byte_str", issue = "134915")]
#[cfg(all(not(no_rc), not(no_sync), target_has_atomic = "ptr"))]
impl From<Arc<[u8]>> for Arc<ByteStr> {
    #[inline]
    fn from(s: Arc<[u8]>) -> Arc<ByteStr> {
        // SAFETY: `ByteStr` is a transparent wrapper around `[u8]`.
        unsafe { Arc::from_raw(Arc::into_raw(s) as _) }
    }
}

#[unstable(feature = "byte_str", issue = "134915")]
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
impl_partial_eq_ord_cow!(&ByteStr, Cow<'_, ByteStr>);
impl_partial_eq_ord_cow!(&ByteStr, Cow<'_, str>);
impl_partial_eq_ord_cow!(&ByteStr, Cow<'_, [u8]>);

#[unstable(feature = "byte_str", issue = "134915")]
impl<'a> TryFrom<&'a ByteStr> for String {
    type Error = core::str::Utf8Error;

    #[inline]
    fn try_from(s: &'a ByteStr) -> Result<Self, Self::Error> {
        Ok(core::str::from_utf8(s.as_bytes())?.into())
    }
}

impl ByteStr {
    /// Try to get a `String` representation of the `&ByteStr`, if it is valid
    /// UTF-8.
    ///
    /// This method is named `to_string()` because we want `ByteStr` to
    /// implement `Display`, but the `ToString` trait has a blanket
    /// implementation for types that implement `Display`, and the trait version
    /// will use the Unicode replacement character rather than returning a
    /// `Result` and allowing for the possibility of the content not being UTF-8.
    #[unstable(feature = "byte_str_to_string", issue = "134915")]
    #[rustc_allow_incoherent_impl]
    pub fn to_string(&self) -> Result<String, Utf8Error> {
        // Avoid allocating a copy of the contents for invalid UTF-8
        if let Err(e) = str::from_utf8(self.as_bytes()) {
            return Err(e);
        }
        // SAFETY: we just checked that the contents are valid UTF-8
        Ok(unsafe { String::from_utf8_unchecked(self.to_bytes()) })
    }
}
