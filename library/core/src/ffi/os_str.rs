//! [`OsStr`] abd their related types.

use crate::clone::CloneToUninit;
use crate::hash::{Hash, Hasher};
use crate::ops::{self, Range};
use crate::ptr::addr_of_mut;
use crate::{cmp, fmt, slice};

mod private {
    /// This trait being unreachable from outside the crate
    /// prevents outside implementations of our extension traits.
    /// This allows adding more trait methods in the future.
    #[unstable(feature = "sealed", issue = "none")]
    pub trait Sealed {}
}

#[cfg(any(target_os = "windows", target_os = "uefi"))]
mod wtf8;

#[cfg(any(target_os = "windows", target_os = "uefi"))]
#[unstable(
    feature = "os_str_internals",
    reason = "internal details of the implementation of os str",
    issue = "none"
)]
#[doc(hidden)]
pub use wtf8::Slice;

#[cfg(not(any(target_os = "windows", target_os = "uefi")))]
mod bytes;

#[cfg(not(any(target_os = "windows", target_os = "uefi")))]
#[unstable(
    feature = "os_str_internals",
    reason = "internal details of the implementation of os str",
    issue = "none"
)]
#[doc(hidden)]
pub use bytes::Slice;

#[cfg(any(target_os = "windows", target_os = "uefi"))]
#[stable(feature = "rust1", since = "1.0.0")]
pub mod os_str_ext_windows;

#[cfg(not(any(target_os = "windows", target_os = "uefi")))]
#[stable(feature = "rust1", since = "1.0.0")]
pub mod os_str_ext_unix;

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
// `OsStr::from_inner` current implementation relies
// on `OsStr` being layout-compatible with `Slice`.
// However, `OsStr` layout is considered an implementation detail and must not be relied upon.
#[repr(transparent)]
#[rustc_has_incoherent_inherent_impls]
pub struct OsStr {
    inner: Slice,
}

/// Allows extension traits within `std`.
#[unstable(feature = "sealed", issue = "none")]
impl private::Sealed for OsStr {}

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

    /// Converts a slice of bytes to an OS string slice without checking that the string contains
    /// valid `OsStr`-encoded data.
    ///
    /// The byte encoding is an unspecified, platform-specific, self-synchronizing superset of UTF-8.
    /// By being a self-synchronizing superset of UTF-8, this encoding is also a superset of 7-bit
    /// ASCII.
    ///
    /// See the [module's toplevel documentation about conversions][conversions] for safe,
    /// cross-platform [conversions] from/to native representations.
    ///
    /// # Safety
    ///
    /// As the encoding is unspecified, callers must pass in bytes that originated as a mixture of
    /// validated UTF-8 and bytes from [`OsStr::as_encoded_bytes`] from within the same Rust version
    /// built for the same target platform.  For example, reconstructing an `OsStr` from bytes sent
    /// over the network or stored in a file will likely violate these safety rules.
    ///
    /// Due to the encoding being self-synchronizing, the bytes from [`OsStr::as_encoded_bytes`] can be
    /// split either immediately before or immediately after any valid non-empty UTF-8 substring.
    ///
    /// # Example
    ///
    /// ```
    /// use std::ffi::OsStr;
    ///
    /// let os_str = OsStr::new("Mary had a little lamb");
    /// let bytes = os_str.as_encoded_bytes();
    /// let words = bytes.split(|b| *b == b' ');
    /// let words: Vec<&OsStr> = words.map(|word| {
    ///     // SAFETY:
    ///     // - Each `word` only contains content that originated from `OsStr::as_encoded_bytes`
    ///     // - Only split with ASCII whitespace which is a non-empty UTF-8 substring
    ///     unsafe { OsStr::from_encoded_bytes_unchecked(word) }
    /// }).collect();
    /// ```
    ///
    /// [conversions]: super#conversions
    #[inline]
    #[stable(feature = "os_str_bytes", since = "1.74.0")]
    pub unsafe fn from_encoded_bytes_unchecked(bytes: &[u8]) -> &Self {
        // SAFETY: unsafe fn
        Self::from_inner(unsafe { Slice::from_encoded_bytes_unchecked(bytes) })
    }

    /// Create immutable [`OsStr`] from [`Slice`]
    #[unstable(
        feature = "os_str_internals",
        reason = "internal details of the implementation of os str",
        issue = "none"
    )]
    #[inline]
    #[doc(hidden)]
    pub fn from_inner(inner: &Slice) -> &OsStr {
        // SAFETY: OsStr is just a wrapper of Slice,
        // therefore converting &Slice to &OsStr is safe.
        unsafe { &*(inner as *const Slice as *const OsStr) }
    }

    /// Create mutable [`OsStr`] from [`Slice`]
    #[unstable(
        feature = "os_str_internals",
        reason = "internal details of the implementation of os str",
        issue = "none"
    )]
    #[inline]
    #[doc(hidden)]
    pub fn from_inner_mut(inner: &mut Slice) -> &mut OsStr {
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
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[inline]
    pub fn to_str(&self) -> Option<&str> {
        self.inner.to_str().ok()
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
    #[must_use]
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
    /// See the main `OsString` documentation information about encoding and capacity units.
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
    #[must_use]
    #[inline]
    pub fn len(&self) -> usize {
        self.inner.inner.len()
    }

    /// Converts an OS string slice to a byte slice.  To convert the byte slice back into an OS
    /// string slice, use the [`OsStr::from_encoded_bytes_unchecked`] function.
    ///
    /// The byte encoding is an unspecified, platform-specific, self-synchronizing superset of UTF-8.
    /// By being a self-synchronizing superset of UTF-8, this encoding is also a superset of 7-bit
    /// ASCII.
    ///
    /// Note: As the encoding is unspecified, any sub-slice of bytes that is not valid UTF-8 should
    /// be treated as opaque and only comparable within the same Rust version built for the same
    /// target platform.  For example, sending the slice over the network or storing it in a file
    /// will likely result in incompatible byte slices.  See [`OsString`] for more encoding details
    /// and [`std::ffi`] for platform-specific, specified conversions.
    ///
    /// [`std::ffi`]: crate::ffi
    #[inline]
    #[stable(feature = "os_str_bytes", since = "1.74.0")]
    pub fn as_encoded_bytes(&self) -> &[u8] {
        self.inner.as_encoded_bytes()
    }

    /// Takes a substring based on a range that corresponds to the return value of
    /// [`OsStr::as_encoded_bytes`].
    ///
    /// The range's start and end must lie on valid `OsStr` boundaries.
    /// A valid `OsStr` boundary is one of:
    /// - The start of the string
    /// - The end of the string
    /// - Immediately before a valid non-empty UTF-8 substring
    /// - Immediately after a valid non-empty UTF-8 substring
    ///
    /// # Panics
    ///
    /// Panics if `range` does not lie on valid `OsStr` boundaries or if it
    /// exceeds the end of the string.
    ///
    /// # Example
    ///
    /// ```
    /// #![feature(os_str_slice)]
    ///
    /// use std::ffi::OsStr;
    ///
    /// let os_str = OsStr::new("foo=bar");
    /// let bytes = os_str.as_encoded_bytes();
    /// if let Some(index) = bytes.iter().position(|b| *b == b'=') {
    ///     let key = os_str.slice_encoded_bytes(..index);
    ///     let value = os_str.slice_encoded_bytes(index + 1..);
    ///     assert_eq!(key, "foo");
    ///     assert_eq!(value, "bar");
    /// }
    /// ```
    #[unstable(feature = "os_str_slice", issue = "118485")]
    pub fn slice_encoded_bytes<R: ops::RangeBounds<usize>>(&self, range: R) -> &Self {
        let encoded_bytes = self.as_encoded_bytes();
        let Range { start, end } = slice::range(range, ..encoded_bytes.len());

        // `check_public_boundary` should panic if the index does not lie on an
        // `OsStr` boundary as described above. It's possible to do this in an
        // encoding-agnostic way, but details of the internal encoding might
        // permit a more efficient implementation.
        self.inner.check_public_boundary(start);
        self.inner.check_public_boundary(end);

        // SAFETY: `slice::range` ensures that `start` and `end` are valid
        let slice = unsafe { encoded_bytes.get_unchecked(start..end) };

        // SAFETY: `slice` comes from `self` and we validated the boundaries
        unsafe { Self::from_encoded_bytes_unchecked(slice) }
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
    #[must_use]
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

    /// Returns an object that implements [`Display`] for safely printing an
    /// [`OsStr`] that may contain non-Unicode data. This may perform lossy
    /// conversion, depending on the platform.  If you would like an
    /// implementation which escapes the [`OsStr`] please use [`Debug`]
    /// instead.
    ///
    /// [`Display`]: fmt::Display
    /// [`Debug`]: fmt::Debug
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(os_str_display)]
    /// use std::ffi::OsStr;
    ///
    /// let s = OsStr::new("Hello, world!");
    /// println!("{}", s.display());
    /// ```
    #[unstable(feature = "os_str_display", issue = "120048")]
    #[must_use = "this does not display the `OsStr`; \
                  it returns an object that can be displayed"]
    #[inline]
    pub fn display(&self) -> Display<'_> {
        Display { os_str: self }
    }

    /// Get inner slice
    #[unstable(
        feature = "os_str_internals",
        reason = "internal details of the implementation of os str",
        issue = "none"
    )]
    #[inline]
    #[doc(hidden)]
    pub fn as_inner(&self) -> &Slice {
        &self.inner
    }
}

#[unstable(feature = "clone_to_uninit", issue = "126799")]
unsafe impl CloneToUninit for OsStr {
    #[inline]
    #[cfg_attr(debug_assertions, track_caller)]
    unsafe fn clone_to_uninit(&self, dst: *mut Self) {
        // SAFETY: we're just a wrapper around a platform-specific Slice
        unsafe { self.inner.clone_to_uninit(addr_of_mut!((*dst).inner)) }
    }
}

#[stable(feature = "str_tryfrom_osstr_impl", since = "1.72.0")]
impl<'a> TryFrom<&'a OsStr> for &'a str {
    type Error = crate::str::Utf8Error;

    /// Tries to convert an `&OsStr` to a `&str`.
    ///
    /// ```
    /// use std::ffi::OsStr;
    ///
    /// let os_str = OsStr::new("foo");
    /// let as_str = <&str>::try_from(os_str).unwrap();
    /// assert_eq!(as_str, "foo");
    /// ```
    fn try_from(value: &'a OsStr) -> Result<Self, Self::Error> {
        value.inner.to_str()
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
        self.as_encoded_bytes().eq(other.as_encoded_bytes())
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
        self.as_encoded_bytes().partial_cmp(other.as_encoded_bytes())
    }
    #[inline]
    fn lt(&self, other: &OsStr) -> bool {
        self.as_encoded_bytes().lt(other.as_encoded_bytes())
    }
    #[inline]
    fn le(&self, other: &OsStr) -> bool {
        self.as_encoded_bytes().le(other.as_encoded_bytes())
    }
    #[inline]
    fn gt(&self, other: &OsStr) -> bool {
        self.as_encoded_bytes().gt(other.as_encoded_bytes())
    }
    #[inline]
    fn ge(&self, other: &OsStr) -> bool {
        self.as_encoded_bytes().ge(other.as_encoded_bytes())
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
        self.as_encoded_bytes().cmp(other.as_encoded_bytes())
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Hash for OsStr {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_encoded_bytes().hash(state)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl fmt::Debug for OsStr {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.inner, formatter)
    }
}

/// Helper struct for safely printing an [`OsStr`] with [`format!`] and `{}`.
///
/// An [`OsStr`] might contain non-Unicode data. This `struct` implements the
/// [`Display`] trait in a way that mitigates that. It is created by the
/// [`display`](OsStr::display) method on [`OsStr`]. This may perform lossy
/// conversion, depending on the platform. If you would like an implementation
/// which escapes the [`OsStr`] please use [`Debug`] instead.
///
/// # Examples
///
/// ```
/// #![feature(os_str_display)]
/// use std::ffi::OsStr;
///
/// let s = OsStr::new("Hello, world!");
/// println!("{}", s.display());
/// ```
///
/// [`Display`]: fmt::Display
/// [`format!`]: crate::format
#[unstable(feature = "os_str_display", issue = "120048")]
pub struct Display<'a> {
    os_str: &'a OsStr,
}

#[unstable(feature = "os_str_display", issue = "120048")]
impl fmt::Debug for Display<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.os_str, f)
    }
}

#[unstable(feature = "os_str_display", issue = "120048")]
impl fmt::Display for Display<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.os_str.inner, f)
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
impl AsRef<OsStr> for str {
    #[inline]
    fn as_ref(&self) -> &OsStr {
        OsStr::from_inner(Slice::from_str(self))
    }
}

#[unstable(
    feature = "os_str_internals",
    reason = "internal details of the implementation of os str",
    issue = "none"
)]
#[doc(hidden)]
impl AsRef<Slice> for OsStr {
    #[inline]
    fn as_ref(&self) -> &Slice {
        &self.inner
    }
}
