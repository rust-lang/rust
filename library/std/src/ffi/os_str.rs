#[cfg(test)]
mod tests;

use crate::borrow::{Borrow, Cow};
use crate::cmp;
use crate::fmt;
use crate::hash::{Hash, Hasher};
use crate::iter::{Extend, FromIterator};
use crate::ops;
use crate::rc::Rc;
use crate::str::FromStr;
use crate::sync::Arc;

use crate::sys::os_str::{Buf, Slice};
use crate::sys_common::{AsInner, FromInner, IntoInner};

/// A type that can represent owned, mutable platform-native strings, but is
/// cheaply inter-convertible with Rust strings.
///
/// The need for this type arises from the fact that:
///
/// * On Unix systems, strings are often arbitrary sequences of non-zero
///   bytes, in many cases interpreted as UTF-8.
///
/// * On Windows, strings are often arbitrary sequences of non-zero 16-bit
///   values, interpreted as UTF-16 when it is valid to do so.
///
/// * In Rust, strings are always valid UTF-8, which may contain zeros.
///
/// `OsString` and [`OsStr`] bridge this gap by simultaneously representing Rust
/// and platform-native string values, and in particular allowing a Rust string
/// to be converted into an "OS" string with no cost if possible. A consequence
/// of this is that `OsString` instances are *not* `NUL` terminated; in order
/// to pass to e.g., Unix system call, you should create a [`CStr`].
///
/// `OsString` is to <code>&[OsStr]</code> as [`String`] is to <code>&[str]</code>: the former
/// in each pair are owned strings; the latter are borrowed
/// references.
///
/// Note, `OsString` and [`OsStr`] internally do not necessarily hold strings in
/// the form native to the platform; While on Unix, strings are stored as a
/// sequence of 8-bit values, on Windows, where strings are 16-bit value based
/// as just discussed, strings are also actually stored as a sequence of 8-bit
/// values, encoded in a less-strict variant of UTF-8. This is useful to
/// understand when handling capacity and length values.
///
/// # Creating an `OsString`
///
/// **From a Rust string**: `OsString` implements
/// <code>[From]<[String]></code>, so you can use <code>my_string.[into]\()</code> to
/// create an `OsString` from a normal Rust string.
///
/// **From slices:** Just like you can start with an empty Rust
/// [`String`] and then [`String::push_str`] some <code>&[str]</code>
/// sub-string slices into it, you can create an empty `OsString` with
/// the [`OsString::new`] method and then push string slices into it with the
/// [`OsString::push`] method.
///
/// # Extracting a borrowed reference to the whole OS string
///
/// You can use the [`OsString::as_os_str`] method to get an <code>&[OsStr]</code> from
/// an `OsString`; this is effectively a borrowed reference to the
/// whole string.
///
/// # Conversions
///
/// See the [module's toplevel documentation about conversions][conversions] for a discussion on
/// the traits which `OsString` implements for [conversions] from/to native representations.
///
/// [`CStr`]: crate::ffi::CStr
/// [conversions]: super#conversions
/// [into]: Into::into
#[cfg_attr(not(test), rustc_diagnostic_item = "OsString")]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct OsString {
    inner: Buf,
}

/// Allows extension traits within `std`.
#[unstable(feature = "sealed", issue = "none")]
impl crate::sealed::Sealed for OsString {}

/// Borrowed reference to an OS string (see [`OsString`]).
///
/// This type represents a borrowed reference to a string in the operating system's preferred
/// representation.
///
/// `&OsStr` is to [`OsString`] as <code>&[str]</code> is to [`String`]: the
/// former in each pair are borrowed references; the latter are owned strings.
///
/// See the [module's toplevel documentation about conversions][conversions] for a discussion on
/// the traits which `OsStr` implements for [conversions] from/to native representations.
///
/// [conversions]: super#conversions
#[cfg_attr(not(test), rustc_diagnostic_item = "OsStr")]
#[stable(feature = "rust1", since = "1.0.0")]
// FIXME:
// `OsStr::from_inner` current implementation relies
// on `OsStr` being layout-compatible with `Slice`.
// When attribute privacy is implemented, `OsStr` should be annotated as `#[repr(transparent)]`.
// Anyway, `OsStr` representation and layout are considered implementation details, are
// not documented and must not be relied upon.
pub struct OsStr {
    inner: Slice,
}

/// Allows extension traits within `std`.
#[unstable(feature = "sealed", issue = "none")]
impl crate::sealed::Sealed for OsStr {}

impl OsString {
    /// Constructs a new empty `OsString`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ffi::OsString;
    ///
    /// let os_string = OsString::new();
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn new() -> OsString {
        OsString { inner: Buf::from_string(String::new()) }
    }

    /// Converts to an [`OsStr`] slice.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ffi::{OsString, OsStr};
    ///
    /// let os_string = OsString::from("foo");
    /// let os_str = OsStr::new("foo");
    /// assert_eq!(os_string.as_os_str(), os_str);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn as_os_str(&self) -> &OsStr {
        self
    }

    /// Converts the `OsString` into a [`String`] if it contains valid Unicode data.
    ///
    /// On failure, ownership of the original `OsString` is returned.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ffi::OsString;
    ///
    /// let os_string = OsString::from("foo");
    /// let string = os_string.into_string();
    /// assert_eq!(string, Ok(String::from("foo")));
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn into_string(self) -> Result<String, OsString> {
        self.inner.into_string().map_err(|buf| OsString { inner: buf })
    }

    /// Extends the string with the given <code>&[OsStr]</code> slice.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ffi::OsString;
    ///
    /// let mut os_string = OsString::from("foo");
    /// os_string.push("bar");
    /// assert_eq!(&os_string, "foobar");
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn push<T: AsRef<OsStr>>(&mut self, s: T) {
        self.inner.push_slice(&s.as_ref().inner)
    }

    /// Creates a new `OsString` with the given capacity.
    ///
    /// The string will be able to hold exactly `capacity` length units of other
    /// OS strings without reallocating. If `capacity` is 0, the string will not
    /// allocate.
    ///
    /// See main `OsString` documentation information about encoding.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ffi::OsString;
    ///
    /// let mut os_string = OsString::with_capacity(10);
    /// let capacity = os_string.capacity();
    ///
    /// // This push is done without reallocating
    /// os_string.push("foo");
    ///
    /// assert_eq!(capacity, os_string.capacity());
    /// ```
    #[stable(feature = "osstring_simple_functions", since = "1.9.0")]
    #[inline]
    pub fn with_capacity(capacity: usize) -> OsString {
        OsString { inner: Buf::with_capacity(capacity) }
    }

    /// Truncates the `OsString` to zero length.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ffi::OsString;
    ///
    /// let mut os_string = OsString::from("foo");
    /// assert_eq!(&os_string, "foo");
    ///
    /// os_string.clear();
    /// assert_eq!(&os_string, "");
    /// ```
    #[stable(feature = "osstring_simple_functions", since = "1.9.0")]
    #[inline]
    pub fn clear(&mut self) {
        self.inner.clear()
    }

    /// Returns the capacity this `OsString` can hold without reallocating.
    ///
    /// See `OsString` introduction for information about encoding.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ffi::OsString;
    ///
    /// let os_string = OsString::with_capacity(10);
    /// assert!(os_string.capacity() >= 10);
    /// ```
    #[stable(feature = "osstring_simple_functions", since = "1.9.0")]
    #[inline]
    pub fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    /// Reserves capacity for at least `additional` more capacity to be inserted
    /// in the given `OsString`.
    ///
    /// The collection may reserve more space to avoid frequent reallocations.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ffi::OsString;
    ///
    /// let mut s = OsString::new();
    /// s.reserve(10);
    /// assert!(s.capacity() >= 10);
    /// ```
    #[stable(feature = "osstring_simple_functions", since = "1.9.0")]
    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        self.inner.reserve(additional)
    }

    /// Reserves the minimum capacity for exactly `additional` more capacity to
    /// be inserted in the given `OsString`. Does nothing if the capacity is
    /// already sufficient.
    ///
    /// Note that the allocator may give the collection more space than it
    /// requests. Therefore, capacity can not be relied upon to be precisely
    /// minimal. Prefer [`reserve`] if future insertions are expected.
    ///
    /// [`reserve`]: OsString::reserve
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ffi::OsString;
    ///
    /// let mut s = OsString::new();
    /// s.reserve_exact(10);
    /// assert!(s.capacity() >= 10);
    /// ```
    #[stable(feature = "osstring_simple_functions", since = "1.9.0")]
    #[inline]
    pub fn reserve_exact(&mut self, additional: usize) {
        self.inner.reserve_exact(additional)
    }

    /// Shrinks the capacity of the `OsString` to match its length.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ffi::OsString;
    ///
    /// let mut s = OsString::from("foo");
    ///
    /// s.reserve(100);
    /// assert!(s.capacity() >= 100);
    ///
    /// s.shrink_to_fit();
    /// assert_eq!(3, s.capacity());
    /// ```
    #[stable(feature = "osstring_shrink_to_fit", since = "1.19.0")]
    #[inline]
    pub fn shrink_to_fit(&mut self) {
        self.inner.shrink_to_fit()
    }

    /// Shrinks the capacity of the `OsString` with a lower bound.
    ///
    /// The capacity will remain at least as large as both the length
    /// and the supplied value.
    ///
    /// If the current capacity is less than the lower limit, this is a no-op.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ffi::OsString;
    ///
    /// let mut s = OsString::from("foo");
    ///
    /// s.reserve(100);
    /// assert!(s.capacity() >= 100);
    ///
    /// s.shrink_to(10);
    /// assert!(s.capacity() >= 10);
    /// s.shrink_to(0);
    /// assert!(s.capacity() >= 3);
    /// ```
    #[inline]
    #[stable(feature = "shrink_to", since = "1.56.0")]
    pub fn shrink_to(&mut self, min_capacity: usize) {
        self.inner.shrink_to(min_capacity)
    }

    /// Converts this `OsString` into a boxed [`OsStr`].
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ffi::{OsString, OsStr};
    ///
    /// let s = OsString::from("hello");
    ///
    /// let b: Box<OsStr> = s.into_boxed_os_str();
    /// ```
    #[stable(feature = "into_boxed_os_str", since = "1.20.0")]
    pub fn into_boxed_os_str(self) -> Box<OsStr> {
        let rw = Box::into_raw(self.inner.into_box()) as *mut OsStr;
        unsafe { Box::from_raw(rw) }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl From<String> for OsString {
    /// Converts a [`String`] into an [`OsString`].
    ///
    /// This conversion does not allocate or copy memory.
    #[inline]
    fn from(s: String) -> OsString {
        OsString { inner: Buf::from_string(s) }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized + AsRef<OsStr>> From<&T> for OsString {
    fn from(s: &T) -> OsString {
        s.as_ref().to_os_string()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl ops::Index<ops::RangeFull> for OsString {
    type Output = OsStr;

    #[inline]
    fn index(&self, _index: ops::RangeFull) -> &OsStr {
        OsStr::from_inner(self.inner.as_slice())
    }
}

#[stable(feature = "mut_osstr", since = "1.44.0")]
impl ops::IndexMut<ops::RangeFull> for OsString {
    #[inline]
    fn index_mut(&mut self, _index: ops::RangeFull) -> &mut OsStr {
        OsStr::from_inner_mut(self.inner.as_mut_slice())
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl ops::Deref for OsString {
    type Target = OsStr;

    #[inline]
    fn deref(&self) -> &OsStr {
        &self[..]
    }
}

#[stable(feature = "mut_osstr", since = "1.44.0")]
impl ops::DerefMut for OsString {
    #[inline]
    fn deref_mut(&mut self) -> &mut OsStr {
        &mut self[..]
    }
}

#[stable(feature = "osstring_default", since = "1.9.0")]
impl Default for OsString {
    /// Constructs an empty `OsString`.
    #[inline]
    fn default() -> OsString {
        OsString::new()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Clone for OsString {
    #[inline]
    fn clone(&self) -> Self {
        OsString { inner: self.inner.clone() }
    }

    #[inline]
    fn clone_from(&mut self, source: &Self) {
        self.inner.clone_from(&source.inner)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl fmt::Debug for OsString {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&**self, formatter)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl PartialEq for OsString {
    #[inline]
    fn eq(&self, other: &OsString) -> bool {
        &**self == &**other
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl PartialEq<str> for OsString {
    #[inline]
    fn eq(&self, other: &str) -> bool {
        &**self == other
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl PartialEq<OsString> for str {
    #[inline]
    fn eq(&self, other: &OsString) -> bool {
        &**other == self
    }
}

#[stable(feature = "os_str_str_ref_eq", since = "1.29.0")]
impl PartialEq<&str> for OsString {
    #[inline]
    fn eq(&self, other: &&str) -> bool {
        **self == **other
    }
}

#[stable(feature = "os_str_str_ref_eq", since = "1.29.0")]
impl<'a> PartialEq<OsString> for &'a str {
    #[inline]
    fn eq(&self, other: &OsString) -> bool {
        **other == **self
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Eq for OsString {}

#[stable(feature = "rust1", since = "1.0.0")]
impl PartialOrd for OsString {
    #[inline]
    fn partial_cmp(&self, other: &OsString) -> Option<cmp::Ordering> {
        (&**self).partial_cmp(&**other)
    }
    #[inline]
    fn lt(&self, other: &OsString) -> bool {
        &**self < &**other
    }
    #[inline]
    fn le(&self, other: &OsString) -> bool {
        &**self <= &**other
    }
    #[inline]
    fn gt(&self, other: &OsString) -> bool {
        &**self > &**other
    }
    #[inline]
    fn ge(&self, other: &OsString) -> bool {
        &**self >= &**other
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl PartialOrd<str> for OsString {
    #[inline]
    fn partial_cmp(&self, other: &str) -> Option<cmp::Ordering> {
        (&**self).partial_cmp(other)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Ord for OsString {
    #[inline]
    fn cmp(&self, other: &OsString) -> cmp::Ordering {
        (&**self).cmp(&**other)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Hash for OsString {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        (&**self).hash(state)
    }
}

impl OsStr {
    /// Coerces into an `OsStr` slice.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ffi::OsStr;
    ///
    /// let os_str = OsStr::new("foo");
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn new<S: AsRef<OsStr> + ?Sized>(s: &S) -> &OsStr {
        s.as_ref()
    }

    #[inline]
    fn from_inner(inner: &Slice) -> &OsStr {
        // SAFETY: OsStr is just a wrapper of Slice,
        // therefore converting &Slice to &OsStr is safe.
        unsafe { &*(inner as *const Slice as *const OsStr) }
    }

    #[inline]
    fn from_inner_mut(inner: &mut Slice) -> &mut OsStr {
        // SAFETY: OsStr is just a wrapper of Slice,
        // therefore converting &mut Slice to &mut OsStr is safe.
        // Any method that mutates OsStr must be careful not to
        // break platform-specific encoding, in particular Wtf8 on Windows.
        unsafe { &mut *(inner as *mut Slice as *mut OsStr) }
    }

    /// Yields a <code>&[str]</code> slice if the `OsStr` is valid Unicode.
    ///
    /// This conversion may entail doing a check for UTF-8 validity.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ffi::OsStr;
    ///
    /// let os_str = OsStr::new("foo");
    /// assert_eq!(os_str.to_str(), Some("foo"));
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn to_str(&self) -> Option<&str> {
        self.inner.to_str()
    }

    /// Converts an `OsStr` to a <code>[Cow]<[str]></code>.
    ///
    /// Any non-Unicode sequences are replaced with
    /// [`U+FFFD REPLACEMENT CHARACTER`][U+FFFD].
    ///
    /// [U+FFFD]: crate::char::REPLACEMENT_CHARACTER
    ///
    /// # Examples
    ///
    /// Calling `to_string_lossy` on an `OsStr` with invalid unicode:
    ///
    /// ```
    /// // Note, due to differences in how Unix and Windows represent strings,
    /// // we are forced to complicate this example, setting up example `OsStr`s
    /// // with different source data and via different platform extensions.
    /// // Understand that in reality you could end up with such example invalid
    /// // sequences simply through collecting user command line arguments, for
    /// // example.
    ///
    /// #[cfg(unix)] {
    ///     use std::ffi::OsStr;
    ///     use std::os::unix::ffi::OsStrExt;
    ///
    ///     // Here, the values 0x66 and 0x6f correspond to 'f' and 'o'
    ///     // respectively. The value 0x80 is a lone continuation byte, invalid
    ///     // in a UTF-8 sequence.
    ///     let source = [0x66, 0x6f, 0x80, 0x6f];
    ///     let os_str = OsStr::from_bytes(&source[..]);
    ///
    ///     assert_eq!(os_str.to_string_lossy(), "fo�o");
    /// }
    /// #[cfg(windows)] {
    ///     use std::ffi::OsString;
    ///     use std::os::windows::prelude::*;
    ///
    ///     // Here the values 0x0066 and 0x006f correspond to 'f' and 'o'
    ///     // respectively. The value 0xD800 is a lone surrogate half, invalid
    ///     // in a UTF-16 sequence.
    ///     let source = [0x0066, 0x006f, 0xD800, 0x006f];
    ///     let os_string = OsString::from_wide(&source[..]);
    ///     let os_str = os_string.as_os_str();
    ///
    ///     assert_eq!(os_str.to_string_lossy(), "fo�o");
    /// }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn to_string_lossy(&self) -> Cow<'_, str> {
        self.inner.to_string_lossy()
    }

    /// Copies the slice into an owned [`OsString`].
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ffi::{OsStr, OsString};
    ///
    /// let os_str = OsStr::new("foo");
    /// let os_string = os_str.to_os_string();
    /// assert_eq!(os_string, OsString::from("foo"));
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn to_os_string(&self) -> OsString {
        OsString { inner: self.inner.to_owned() }
    }

    /// Checks whether the `OsStr` is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ffi::OsStr;
    ///
    /// let os_str = OsStr::new("");
    /// assert!(os_str.is_empty());
    ///
    /// let os_str = OsStr::new("foo");
    /// assert!(!os_str.is_empty());
    /// ```
    #[stable(feature = "osstring_simple_functions", since = "1.9.0")]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.inner.inner.is_empty()
    }

    /// Returns the length of this `OsStr`.
    ///
    /// Note that this does **not** return the number of bytes in the string in
    /// OS string form.
    ///
    /// The length returned is that of the underlying storage used by `OsStr`.
    /// As discussed in the [`OsString`] introduction, [`OsString`] and `OsStr`
    /// store strings in a form best suited for cheap inter-conversion between
    /// native-platform and Rust string forms, which may differ significantly
    /// from both of them, including in storage size and encoding.
    ///
    /// This number is simply useful for passing to other methods, like
    /// [`OsString::with_capacity`] to avoid reallocations.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ffi::OsStr;
    ///
    /// let os_str = OsStr::new("");
    /// assert_eq!(os_str.len(), 0);
    ///
    /// let os_str = OsStr::new("foo");
    /// assert_eq!(os_str.len(), 3);
    /// ```
    #[stable(feature = "osstring_simple_functions", since = "1.9.0")]
    #[inline]
    pub fn len(&self) -> usize {
        self.inner.inner.len()
    }

    /// Converts a <code>[Box]<[OsStr]></code> into an [`OsString`] without copying or allocating.
    #[stable(feature = "into_boxed_os_str", since = "1.20.0")]
    pub fn into_os_string(self: Box<OsStr>) -> OsString {
        let boxed = unsafe { Box::from_raw(Box::into_raw(self) as *mut Slice) };
        OsString { inner: Buf::from_box(boxed) }
    }

    /// Gets the underlying byte representation.
    ///
    /// Note: it is *crucial* that this API is not externally public, to avoid
    /// revealing the internal, platform-specific encodings.
    #[inline]
    pub(crate) fn bytes(&self) -> &[u8] {
        unsafe { &*(&self.inner as *const _ as *const [u8]) }
    }

    /// Converts this string to its ASCII lower case equivalent in-place.
    ///
    /// ASCII letters 'A' to 'Z' are mapped to 'a' to 'z',
    /// but non-ASCII letters are unchanged.
    ///
    /// To return a new lowercased value without modifying the existing one, use
    /// [`OsStr::to_ascii_lowercase`].
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ffi::OsString;
    ///
    /// let mut s = OsString::from("GRÜßE, JÜRGEN ❤");
    ///
    /// s.make_ascii_lowercase();
    ///
    /// assert_eq!("grÜße, jÜrgen ❤", s);
    /// ```
    #[stable(feature = "osstring_ascii", since = "1.53.0")]
    #[inline]
    pub fn make_ascii_lowercase(&mut self) {
        self.inner.make_ascii_lowercase()
    }

    /// Converts this string to its ASCII upper case equivalent in-place.
    ///
    /// ASCII letters 'a' to 'z' are mapped to 'A' to 'Z',
    /// but non-ASCII letters are unchanged.
    ///
    /// To return a new uppercased value without modifying the existing one, use
    /// [`OsStr::to_ascii_uppercase`].
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ffi::OsString;
    ///
    /// let mut s = OsString::from("Grüße, Jürgen ❤");
    ///
    /// s.make_ascii_uppercase();
    ///
    /// assert_eq!("GRüßE, JüRGEN ❤", s);
    /// ```
    #[stable(feature = "osstring_ascii", since = "1.53.0")]
    #[inline]
    pub fn make_ascii_uppercase(&mut self) {
        self.inner.make_ascii_uppercase()
    }

    /// Returns a copy of this string where each character is mapped to its
    /// ASCII lower case equivalent.
    ///
    /// ASCII letters 'A' to 'Z' are mapped to 'a' to 'z',
    /// but non-ASCII letters are unchanged.
    ///
    /// To lowercase the value in-place, use [`OsStr::make_ascii_lowercase`].
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ffi::OsString;
    /// let s = OsString::from("Grüße, Jürgen ❤");
    ///
    /// assert_eq!("grüße, jürgen ❤", s.to_ascii_lowercase());
    /// ```
    #[stable(feature = "osstring_ascii", since = "1.53.0")]
    pub fn to_ascii_lowercase(&self) -> OsString {
        OsString::from_inner(self.inner.to_ascii_lowercase())
    }

    /// Returns a copy of this string where each character is mapped to its
    /// ASCII upper case equivalent.
    ///
    /// ASCII letters 'a' to 'z' are mapped to 'A' to 'Z',
    /// but non-ASCII letters are unchanged.
    ///
    /// To uppercase the value in-place, use [`OsStr::make_ascii_uppercase`].
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ffi::OsString;
    /// let s = OsString::from("Grüße, Jürgen ❤");
    ///
    /// assert_eq!("GRüßE, JüRGEN ❤", s.to_ascii_uppercase());
    /// ```
    #[stable(feature = "osstring_ascii", since = "1.53.0")]
    pub fn to_ascii_uppercase(&self) -> OsString {
        OsString::from_inner(self.inner.to_ascii_uppercase())
    }

    /// Checks if all characters in this string are within the ASCII range.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ffi::OsString;
    ///
    /// let ascii = OsString::from("hello!\n");
    /// let non_ascii = OsString::from("Grüße, Jürgen ❤");
    ///
    /// assert!(ascii.is_ascii());
    /// assert!(!non_ascii.is_ascii());
    /// ```
    #[stable(feature = "osstring_ascii", since = "1.53.0")]
    #[inline]
    pub fn is_ascii(&self) -> bool {
        self.inner.is_ascii()
    }

    /// Checks that two strings are an ASCII case-insensitive match.
    ///
    /// Same as `to_ascii_lowercase(a) == to_ascii_lowercase(b)`,
    /// but without allocating and copying temporaries.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ffi::OsString;
    ///
    /// assert!(OsString::from("Ferris").eq_ignore_ascii_case("FERRIS"));
    /// assert!(OsString::from("Ferrös").eq_ignore_ascii_case("FERRöS"));
    /// assert!(!OsString::from("Ferrös").eq_ignore_ascii_case("FERRÖS"));
    /// ```
    #[stable(feature = "osstring_ascii", since = "1.53.0")]
    pub fn eq_ignore_ascii_case<S: AsRef<OsStr>>(&self, other: S) -> bool {
        self.inner.eq_ignore_ascii_case(&other.as_ref().inner)
    }
}

#[stable(feature = "box_from_os_str", since = "1.17.0")]
impl From<&OsStr> for Box<OsStr> {
    #[inline]
    fn from(s: &OsStr) -> Box<OsStr> {
        let rw = Box::into_raw(s.inner.into_box()) as *mut OsStr;
        unsafe { Box::from_raw(rw) }
    }
}

#[stable(feature = "box_from_cow", since = "1.45.0")]
impl From<Cow<'_, OsStr>> for Box<OsStr> {
    #[inline]
    fn from(cow: Cow<'_, OsStr>) -> Box<OsStr> {
        match cow {
            Cow::Borrowed(s) => Box::from(s),
            Cow::Owned(s) => Box::from(s),
        }
    }
}

#[stable(feature = "os_string_from_box", since = "1.18.0")]
impl From<Box<OsStr>> for OsString {
    /// Converts a <code>[Box]<[OsStr]></code> into an [`OsString`] without copying or
    /// allocating.
    #[inline]
    fn from(boxed: Box<OsStr>) -> OsString {
        boxed.into_os_string()
    }
}

#[stable(feature = "box_from_os_string", since = "1.20.0")]
impl From<OsString> for Box<OsStr> {
    /// Converts an [`OsString`] into a <code>[Box]<[OsStr]></code> without copying or allocating.
    #[inline]
    fn from(s: OsString) -> Box<OsStr> {
        s.into_boxed_os_str()
    }
}

#[stable(feature = "more_box_slice_clone", since = "1.29.0")]
impl Clone for Box<OsStr> {
    #[inline]
    fn clone(&self) -> Self {
        self.to_os_string().into_boxed_os_str()
    }
}

#[stable(feature = "shared_from_slice2", since = "1.24.0")]
impl From<OsString> for Arc<OsStr> {
    /// Converts an [`OsString`] into an <code>[Arc]<[OsStr]></code> without copying or allocating.
    #[inline]
    fn from(s: OsString) -> Arc<OsStr> {
        let arc = s.inner.into_arc();
        unsafe { Arc::from_raw(Arc::into_raw(arc) as *const OsStr) }
    }
}

#[stable(feature = "shared_from_slice2", since = "1.24.0")]
impl From<&OsStr> for Arc<OsStr> {
    #[inline]
    fn from(s: &OsStr) -> Arc<OsStr> {
        let arc = s.inner.into_arc();
        unsafe { Arc::from_raw(Arc::into_raw(arc) as *const OsStr) }
    }
}

#[stable(feature = "shared_from_slice2", since = "1.24.0")]
impl From<OsString> for Rc<OsStr> {
    /// Converts an [`OsString`] into an <code>[Rc]<[OsStr]></code> without copying or allocating.
    #[inline]
    fn from(s: OsString) -> Rc<OsStr> {
        let rc = s.inner.into_rc();
        unsafe { Rc::from_raw(Rc::into_raw(rc) as *const OsStr) }
    }
}

#[stable(feature = "shared_from_slice2", since = "1.24.0")]
impl From<&OsStr> for Rc<OsStr> {
    #[inline]
    fn from(s: &OsStr) -> Rc<OsStr> {
        let rc = s.inner.into_rc();
        unsafe { Rc::from_raw(Rc::into_raw(rc) as *const OsStr) }
    }
}

#[stable(feature = "cow_from_osstr", since = "1.28.0")]
impl<'a> From<OsString> for Cow<'a, OsStr> {
    #[inline]
    fn from(s: OsString) -> Cow<'a, OsStr> {
        Cow::Owned(s)
    }
}

#[stable(feature = "cow_from_osstr", since = "1.28.0")]
impl<'a> From<&'a OsStr> for Cow<'a, OsStr> {
    #[inline]
    fn from(s: &'a OsStr) -> Cow<'a, OsStr> {
        Cow::Borrowed(s)
    }
}

#[stable(feature = "cow_from_osstr", since = "1.28.0")]
impl<'a> From<&'a OsString> for Cow<'a, OsStr> {
    #[inline]
    fn from(s: &'a OsString) -> Cow<'a, OsStr> {
        Cow::Borrowed(s.as_os_str())
    }
}

#[stable(feature = "osstring_from_cow_osstr", since = "1.28.0")]
impl<'a> From<Cow<'a, OsStr>> for OsString {
    #[inline]
    fn from(s: Cow<'a, OsStr>) -> Self {
        s.into_owned()
    }
}

#[stable(feature = "box_default_extra", since = "1.17.0")]
impl Default for Box<OsStr> {
    #[inline]
    fn default() -> Box<OsStr> {
        let rw = Box::into_raw(Slice::empty_box()) as *mut OsStr;
        unsafe { Box::from_raw(rw) }
    }
}

#[stable(feature = "osstring_default", since = "1.9.0")]
impl Default for &OsStr {
    /// Creates an empty `OsStr`.
    #[inline]
    fn default() -> Self {
        OsStr::new("")
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl PartialEq for OsStr {
    #[inline]
    fn eq(&self, other: &OsStr) -> bool {
        self.bytes().eq(other.bytes())
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl PartialEq<str> for OsStr {
    #[inline]
    fn eq(&self, other: &str) -> bool {
        *self == *OsStr::new(other)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl PartialEq<OsStr> for str {
    #[inline]
    fn eq(&self, other: &OsStr) -> bool {
        *other == *OsStr::new(self)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Eq for OsStr {}

#[stable(feature = "rust1", since = "1.0.0")]
impl PartialOrd for OsStr {
    #[inline]
    fn partial_cmp(&self, other: &OsStr) -> Option<cmp::Ordering> {
        self.bytes().partial_cmp(other.bytes())
    }
    #[inline]
    fn lt(&self, other: &OsStr) -> bool {
        self.bytes().lt(other.bytes())
    }
    #[inline]
    fn le(&self, other: &OsStr) -> bool {
        self.bytes().le(other.bytes())
    }
    #[inline]
    fn gt(&self, other: &OsStr) -> bool {
        self.bytes().gt(other.bytes())
    }
    #[inline]
    fn ge(&self, other: &OsStr) -> bool {
        self.bytes().ge(other.bytes())
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl PartialOrd<str> for OsStr {
    #[inline]
    fn partial_cmp(&self, other: &str) -> Option<cmp::Ordering> {
        self.partial_cmp(OsStr::new(other))
    }
}

// FIXME (#19470): cannot provide PartialOrd<OsStr> for str until we
// have more flexible coherence rules.

#[stable(feature = "rust1", since = "1.0.0")]
impl Ord for OsStr {
    #[inline]
    fn cmp(&self, other: &OsStr) -> cmp::Ordering {
        self.bytes().cmp(other.bytes())
    }
}

macro_rules! impl_cmp {
    ($lhs:ty, $rhs: ty) => {
        #[stable(feature = "cmp_os_str", since = "1.8.0")]
        impl<'a, 'b> PartialEq<$rhs> for $lhs {
            #[inline]
            fn eq(&self, other: &$rhs) -> bool {
                <OsStr as PartialEq>::eq(self, other)
            }
        }

        #[stable(feature = "cmp_os_str", since = "1.8.0")]
        impl<'a, 'b> PartialEq<$lhs> for $rhs {
            #[inline]
            fn eq(&self, other: &$lhs) -> bool {
                <OsStr as PartialEq>::eq(self, other)
            }
        }

        #[stable(feature = "cmp_os_str", since = "1.8.0")]
        impl<'a, 'b> PartialOrd<$rhs> for $lhs {
            #[inline]
            fn partial_cmp(&self, other: &$rhs) -> Option<cmp::Ordering> {
                <OsStr as PartialOrd>::partial_cmp(self, other)
            }
        }

        #[stable(feature = "cmp_os_str", since = "1.8.0")]
        impl<'a, 'b> PartialOrd<$lhs> for $rhs {
            #[inline]
            fn partial_cmp(&self, other: &$lhs) -> Option<cmp::Ordering> {
                <OsStr as PartialOrd>::partial_cmp(self, other)
            }
        }
    };
}

impl_cmp!(OsString, OsStr);
impl_cmp!(OsString, &'a OsStr);
impl_cmp!(Cow<'a, OsStr>, OsStr);
impl_cmp!(Cow<'a, OsStr>, &'b OsStr);
impl_cmp!(Cow<'a, OsStr>, OsString);

#[stable(feature = "rust1", since = "1.0.0")]
impl Hash for OsStr {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.bytes().hash(state)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl fmt::Debug for OsStr {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.inner, formatter)
    }
}

impl OsStr {
    pub(crate) fn display(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.inner, formatter)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Borrow<OsStr> for OsString {
    #[inline]
    fn borrow(&self) -> &OsStr {
        &self[..]
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl ToOwned for OsStr {
    type Owned = OsString;
    #[inline]
    fn to_owned(&self) -> OsString {
        self.to_os_string()
    }
    #[inline]
    fn clone_into(&self, target: &mut OsString) {
        self.inner.clone_into(&mut target.inner)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl AsRef<OsStr> for OsStr {
    #[inline]
    fn as_ref(&self) -> &OsStr {
        self
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl AsRef<OsStr> for OsString {
    #[inline]
    fn as_ref(&self) -> &OsStr {
        self
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl AsRef<OsStr> for str {
    #[inline]
    fn as_ref(&self) -> &OsStr {
        OsStr::from_inner(Slice::from_str(self))
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl AsRef<OsStr> for String {
    #[inline]
    fn as_ref(&self) -> &OsStr {
        (&**self).as_ref()
    }
}

impl FromInner<Buf> for OsString {
    #[inline]
    fn from_inner(buf: Buf) -> OsString {
        OsString { inner: buf }
    }
}

impl IntoInner<Buf> for OsString {
    #[inline]
    fn into_inner(self) -> Buf {
        self.inner
    }
}

impl AsInner<Slice> for OsStr {
    #[inline]
    fn as_inner(&self) -> &Slice {
        &self.inner
    }
}

#[stable(feature = "osstring_from_str", since = "1.45.0")]
impl FromStr for OsString {
    type Err = core::convert::Infallible;

    #[inline]
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(OsString::from(s))
    }
}

#[stable(feature = "osstring_extend", since = "1.52.0")]
impl Extend<OsString> for OsString {
    #[inline]
    fn extend<T: IntoIterator<Item = OsString>>(&mut self, iter: T) {
        for s in iter {
            self.push(&s);
        }
    }
}

#[stable(feature = "osstring_extend", since = "1.52.0")]
impl<'a> Extend<&'a OsStr> for OsString {
    #[inline]
    fn extend<T: IntoIterator<Item = &'a OsStr>>(&mut self, iter: T) {
        for s in iter {
            self.push(s);
        }
    }
}

#[stable(feature = "osstring_extend", since = "1.52.0")]
impl<'a> Extend<Cow<'a, OsStr>> for OsString {
    #[inline]
    fn extend<T: IntoIterator<Item = Cow<'a, OsStr>>>(&mut self, iter: T) {
        for s in iter {
            self.push(&s);
        }
    }
}

#[stable(feature = "osstring_extend", since = "1.52.0")]
impl FromIterator<OsString> for OsString {
    #[inline]
    fn from_iter<I: IntoIterator<Item = OsString>>(iter: I) -> Self {
        let mut iterator = iter.into_iter();

        // Because we're iterating over `OsString`s, we can avoid at least
        // one allocation by getting the first string from the iterator
        // and appending to it all the subsequent strings.
        match iterator.next() {
            None => OsString::new(),
            Some(mut buf) => {
                buf.extend(iterator);
                buf
            }
        }
    }
}

#[stable(feature = "osstring_extend", since = "1.52.0")]
impl<'a> FromIterator<&'a OsStr> for OsString {
    #[inline]
    fn from_iter<I: IntoIterator<Item = &'a OsStr>>(iter: I) -> Self {
        let mut buf = Self::new();
        for s in iter {
            buf.push(s);
        }
        buf
    }
}

#[stable(feature = "osstring_extend", since = "1.52.0")]
impl<'a> FromIterator<Cow<'a, OsStr>> for OsString {
    #[inline]
    fn from_iter<I: IntoIterator<Item = Cow<'a, OsStr>>>(iter: I) -> Self {
        let mut iterator = iter.into_iter();

        // Because we're iterating over `OsString`s, we can avoid at least
        // one allocation by getting the first owned string from the iterator
        // and appending to it all the subsequent strings.
        match iterator.next() {
            None => OsString::new(),
            Some(Cow::Owned(mut buf)) => {
                buf.extend(iterator);
                buf
            }
            Some(Cow::Borrowed(buf)) => {
                let mut buf = OsString::from(buf);
                buf.extend(iterator);
                buf
            }
        }
    }
}
