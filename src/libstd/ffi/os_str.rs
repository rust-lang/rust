use crate::borrow::{Borrow, Cow};
use crate::fmt;
use crate::ops;
use crate::cmp;
use crate::hash::{Hash, Hasher};
use crate::rc::Rc;
use crate::sync::Arc;
use crate::needle::{
    ext, Hay, Haystack, Needle, Span, Searcher, ReverseSearcher,
    Consumer, ReverseConsumer, DoubleEndedConsumer,
};

use crate::sys::os_str::{Buf, InnerSearcher, Slice};
use crate::sys_common::{AsInner, IntoInner, FromInner};

use core::slice::needles::{TwoWaySearcher, SliceSearcher, NaiveSearcher};

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
/// `OsString` is to [`&OsStr`] as [`String`] is to [`&str`]: the former
/// in each pair are owned strings; the latter are borrowed
/// references.
///
/// Note, `OsString` and `OsStr` internally do not necessarily hold strings in
/// the form native to the platform; While on Unix, strings are stored as a
/// sequence of 8-bit values, on Windows, where strings are 16-bit value based
/// as just discussed, strings are also actually stored as a sequence of 8-bit
/// values, encoded in a less-strict variant of UTF-8. This is useful to
/// understand when handling capacity and length values.
///
/// # Creating an `OsString`
///
/// **From a Rust string**: `OsString` implements
/// [`From`]`<`[`String`]`>`, so you can use `my_string.from` to
/// create an `OsString` from a normal Rust string.
///
/// **From slices:** Just like you can start with an empty Rust
/// [`String`] and then [`push_str`][String.push_str] `&str`
/// sub-string slices into it, you can create an empty `OsString` with
/// the [`new`] method and then push string slices into it with the
/// [`push`] method.
///
/// # Extracting a borrowed reference to the whole OS string
///
/// You can use the [`as_os_str`] method to get an `&`[`OsStr`] from
/// an `OsString`; this is effectively a borrowed reference to the
/// whole string.
///
/// # Conversions
///
/// See the [module's toplevel documentation about conversions][conversions] for a discussion on
/// the traits which `OsString` implements for [conversions] from/to native representations.
///
/// [`OsStr`]: struct.OsStr.html
/// [`&OsStr`]: struct.OsStr.html
/// [`CStr`]: struct.CStr.html
/// [`From`]: ../convert/trait.From.html
/// [`String`]: ../string/struct.String.html
/// [`&str`]: ../primitive.str.html
/// [`u8`]: ../primitive.u8.html
/// [`u16`]: ../primitive.u16.html
/// [String.push_str]: ../string/struct.String.html#method.push_str
/// [`new`]: #method.new
/// [`push`]: #method.push
/// [`as_os_str`]: #method.as_os_str
/// [conversions]: index.html#conversions
#[derive(Clone)]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct OsString {
    inner: Buf
}

/// Borrowed reference to an OS string (see [`OsString`]).
///
/// This type represents a borrowed reference to a string in the operating system's preferred
/// representation.
///
/// `&OsStr` is to [`OsString`] as [`&str`] is to [`String`]: the former in each pair are borrowed
/// references; the latter are owned strings.
///
/// See the [module's toplevel documentation about conversions][conversions] for a discussion on
/// the traits which `OsStr` implements for [conversions] from/to native representations.
///
/// [`OsString`]: struct.OsString.html
/// [`&str`]: ../primitive.str.html
/// [`String`]: ../string/struct.String.html
/// [conversions]: index.html#conversions
#[stable(feature = "rust1", since = "1.0.0")]
pub struct OsStr {
    inner: Slice
}

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
    pub fn new() -> OsString {
        OsString { inner: Buf::from_string(String::new()) }
    }

    /// Converts to an [`OsStr`] slice.
    ///
    /// [`OsStr`]: struct.OsStr.html
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
    pub fn as_os_str(&self) -> &OsStr {
        self
    }

    /// Converts the `OsString` into a [`String`] if it contains valid Unicode data.
    ///
    /// On failure, ownership of the original `OsString` is returned.
    ///
    /// [`String`]: ../../std/string/struct.String.html
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
    pub fn into_string(self) -> Result<String, OsString> {
        self.inner.into_string().map_err(|buf| OsString { inner: buf} )
    }

    /// Extends the string with the given [`&OsStr`] slice.
    ///
    /// [`&OsStr`]: struct.OsStr.html
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
    pub fn with_capacity(capacity: usize) -> OsString {
        OsString {
            inner: Buf::with_capacity(capacity)
        }
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
    /// let mut os_string = OsString::with_capacity(10);
    /// assert!(os_string.capacity() >= 10);
    /// ```
    #[stable(feature = "osstring_simple_functions", since = "1.9.0")]
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
    pub fn reserve(&mut self, additional: usize) {
        self.inner.reserve(additional)
    }

    /// Reserves the minimum capacity for exactly `additional` more capacity to
    /// be inserted in the given `OsString`. Does nothing if the capacity is
    /// already sufficient.
    ///
    /// Note that the allocator may give the collection more space than it
    /// requests. Therefore, capacity can not be relied upon to be precisely
    /// minimal. Prefer reserve if future insertions are expected.
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
    pub fn shrink_to_fit(&mut self) {
        self.inner.shrink_to_fit()
    }

    /// Shrinks the capacity of the `OsString` with a lower bound.
    ///
    /// The capacity will remain at least as large as both the length
    /// and the supplied value.
    ///
    /// Panics if the current capacity is smaller than the supplied
    /// minimum capacity.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(shrink_to)]
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
    #[unstable(feature = "shrink_to", reason = "new API", issue="56431")]
    pub fn shrink_to(&mut self, min_capacity: usize) {
        self.inner.shrink_to(min_capacity)
    }

    /// Converts this `OsString` into a boxed [`OsStr`].
    ///
    /// [`OsStr`]: struct.OsStr.html
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
    /// Converts a [`String`] into a [`OsString`].
    ///
    /// The conversion copies the data, and includes an allocation on the heap.
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

#[stable(feature = "os_str_slice", since = "1.36.0")]
impl ops::Index<ops::Range<usize>> for OsString {
    type Output = OsStr;

    #[inline]
    fn index(&self, index: ops::Range<usize>) -> &OsStr {
        OsStr::from_inner(&self.inner.as_slice()[index])
    }
}

#[stable(feature = "os_str_slice", since = "1.36.0")]
impl ops::Index<ops::RangeFrom<usize>> for OsString {
    type Output = OsStr;

    #[inline]
    fn index(&self, index: ops::RangeFrom<usize>) -> &OsStr {
        OsStr::from_inner(&self.inner.as_slice()[index])
    }
}

#[stable(feature = "os_str_slice", since = "1.36.0")]
impl ops::Index<ops::RangeTo<usize>> for OsString {
    type Output = OsStr;

    #[inline]
    fn index(&self, index: ops::RangeTo<usize>) -> &OsStr {
        OsStr::from_inner(&self.inner.as_slice()[index])
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

#[stable(feature = "osstring_default", since = "1.9.0")]
impl Default for OsString {
    /// Constructs an empty `OsString`.
    #[inline]
    fn default() -> OsString {
        OsString::new()
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
    fn eq(&self, other: &OsString) -> bool {
        &**self == &**other
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl PartialEq<str> for OsString {
    fn eq(&self, other: &str) -> bool {
        &**self == other
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl PartialEq<OsString> for str {
    fn eq(&self, other: &OsString) -> bool {
        &**other == self
    }
}

#[stable(feature = "os_str_str_ref_eq", since = "1.29.0")]
impl PartialEq<&str> for OsString {
    fn eq(&self, other: &&str) -> bool {
        **self == **other
    }
}

#[stable(feature = "os_str_str_ref_eq", since = "1.29.0")]
impl<'a> PartialEq<OsString> for &'a str {
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
    fn lt(&self, other: &OsString) -> bool { &**self < &**other }
    #[inline]
    fn le(&self, other: &OsString) -> bool { &**self <= &**other }
    #[inline]
    fn gt(&self, other: &OsString) -> bool { &**self > &**other }
    #[inline]
    fn ge(&self, other: &OsString) -> bool { &**self >= &**other }
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
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn new<S: AsRef<OsStr> + ?Sized>(s: &S) -> &OsStr {
        s.as_ref()
    }

    fn from_inner(inner: &Slice) -> &OsStr {
        unsafe { &*(inner as *const Slice as *const OsStr) }
    }

    /// Yields a [`&str`] slice if the `OsStr` is valid Unicode.
    ///
    /// This conversion may entail doing a check for UTF-8 validity.
    ///
    /// [`&str`]: ../../std/primitive.str.html
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
    pub fn to_str(&self) -> Option<&str> {
        self.inner.to_str()
    }

    /// Converts an `OsStr` to a [`Cow`]`<`[`str`]`>`.
    ///
    /// Any non-Unicode sequences are replaced with
    /// [`U+FFFD REPLACEMENT CHARACTER`][U+FFFD].
    ///
    /// [`Cow`]: ../../std/borrow/enum.Cow.html
    /// [`str`]: ../../std/primitive.str.html
    /// [U+FFFD]: ../../std/char/constant.REPLACEMENT_CHARACTER.html
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
    /// #[cfg(any(unix, target_os = "redox"))] {
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
    pub fn to_string_lossy(&self) -> Cow<'_, str> {
        self.inner.to_string_lossy()
    }

    /// Copies the slice into an owned [`OsString`].
    ///
    /// [`OsString`]: struct.OsString.html
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
    pub fn is_empty(&self) -> bool {
        self.inner.inner.is_empty()
    }

    /// Returns the length of this `OsStr`.
    ///
    /// Note that this does **not** return the number of bytes in the string in
    /// OS string form.
    ///
    /// The length returned is that of the underlying storage used by `OsStr`;
    /// As discussed in the [`OsString`] introduction, [`OsString`] and `OsStr`
    /// store strings in a form best suited for cheap inter-conversion between
    /// native-platform and Rust string forms, which may differ significantly
    /// from both of them, including in storage size and encoding.
    ///
    /// This number is simply useful for passing to other methods, like
    /// [`OsString::with_capacity`] to avoid reallocations.
    ///
    /// [`OsString`]: struct.OsString.html
    /// [`OsString::with_capacity`]: struct.OsString.html#method.with_capacity
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
    pub fn len(&self) -> usize {
        self.inner.inner.len()
    }

    /// Converts a [`Box`]`<OsStr>` into an [`OsString`] without copying or allocating.
    ///
    /// [`Box`]: ../boxed/struct.Box.html
    /// [`OsString`]: struct.OsString.html
    #[stable(feature = "into_boxed_os_str", since = "1.20.0")]
    pub fn into_os_string(self: Box<OsStr>) -> OsString {
        let boxed = unsafe { Box::from_raw(Box::into_raw(self) as *mut Slice) };
        OsString { inner: Buf::from_box(boxed) }
    }

    /// Returns `true` if the given needle matches a prefix of this `OsStr`.
    #[unstable(feature = "os_str_needle_methods", issue = "56345")]
    #[inline]
    pub fn starts_with<'a, P>(&'a self, needle: P) -> bool
    where
        P: Needle<&'a OsStr>,
        P::Searcher: Searcher<OsStr>, // FIXME: RFC 2089
        P::Consumer: Consumer<OsStr>, // FIXME: RFC 2089
    {
        ext::starts_with(self, needle)
    }

    /// Returns `true` if the given needle matches a suffix of this `OsStr`.
    #[unstable(feature = "os_str_needle_methods", issue = "56345")]
    #[inline]
    pub fn ends_with<'a, P>(&'a self, needle: P) -> bool
    where
        P: Needle<&'a OsStr>,
        P::Searcher: Searcher<OsStr>, // FIXME: RFC 2089
        P::Consumer: ReverseConsumer<OsStr>,
    {
        ext::ends_with(self, needle)
    }

    /// Returns `true` if the given needle matches a sub-slice of this `OsStr`.
    #[unstable(feature = "os_str_needle_methods", issue = "56345")]
    #[inline]
    pub fn contains<'a, P>(&'a self, needle: P) -> bool
    where
        P: Needle<&'a OsStr>,
        P::Searcher: Searcher<OsStr>, // FIXME: RFC 2089
        P::Consumer: Consumer<OsStr>, // FIXME: RFC 2089
    {
        ext::contains(self, needle)
    }

    /// Returns the start index of first slice of this `OsStr` that matches the
    /// needle.
    #[unstable(feature = "os_str_needle_methods", issue = "56345")]
    #[inline]
    pub fn find<'a, P>(&'a self, needle: P) -> Option<usize>
    where
        P: Needle<&'a OsStr>,
        P::Searcher: Searcher<OsStr>, // FIXME: RFC 2089
        P::Consumer: Consumer<OsStr>, // FIXME: RFC 2089
    {
        ext::find(self, needle)
    }

    /// Returns the start index of last slice of this `OsStr` that matches the
    /// needle.
    #[unstable(feature = "os_str_needle_methods", issue = "56345")]
    #[inline]
    pub fn rfind<'a, P>(&'a self, needle: P) -> Option<usize>
    where
        P: Needle<&'a OsStr>,
        P::Searcher: ReverseSearcher<OsStr>,
        P::Consumer: Consumer<OsStr>, // FIXME: RFC 2089
    {
        ext::rfind(self, needle)
    }

    /// Returns the index range of first slice of this `OsStr` that matches the
    /// needle.
    #[unstable(feature = "os_str_needle_methods", issue = "56345")]
    #[inline]
    pub fn find_range<'a, P>(&'a self, needle: P) -> Option<ops::Range<usize>>
    where
        P: Needle<&'a OsStr>,
        P::Searcher: Searcher<OsStr>, // FIXME: RFC 2089
        P::Consumer: Consumer<OsStr>, // FIXME: RFC 2089
    {
        ext::find_range(self, needle)
    }

    /// Returns the start index of last slice of this `OsStr` that matches the
    /// needle.
    #[unstable(feature = "os_str_needle_methods", issue = "56345")]
    #[inline]
    pub fn rfind_range<'a, P>(&'a self, needle: P) -> Option<ops::Range<usize>>
    where
        P: Needle<&'a OsStr>,
        P::Searcher: ReverseSearcher<OsStr>,
        P::Consumer: Consumer<OsStr>, // FIXME: RFC 2089
    {
        ext::rfind_range(self, needle)
    }

    /// Returns an `OsStr` slice with all prefixes that match the needle
    /// repeatedly removed.
    #[unstable(feature = "os_str_needle_methods", issue = "56345")]
    #[inline]
    pub fn trim_start_matches<'a, P>(&'a self, needle: P) -> &'a OsStr
    where
        P: Needle<&'a OsStr>,
        P::Searcher: Searcher<OsStr>, // FIXME: RFC 2089
        P::Consumer: Consumer<OsStr>, // FIXME: RFC 2089
    {
        ext::trim_start(self, needle)
    }

    /// Returns an `OsStr` slice with all suffixes that match the needle
    /// repeatedly removed.
    #[unstable(feature = "os_str_needle_methods", issue = "56345")]
    #[inline]
    pub fn trim_end_matches<'a, P>(&'a self, needle: P) -> &'a OsStr
    where
        P: Needle<&'a OsStr>,
        P::Searcher: Searcher<OsStr>, // FIXME: RFC 2089
        P::Consumer: ReverseConsumer<OsStr>,
    {
        ext::trim_end(self, needle)
    }

    /// Returns an `OsStr` slice with all prefixes and suffixes that match the
    /// needle repeatedly removed.
    #[unstable(feature = "os_str_needle_methods", issue = "56345")]
    #[inline]
    pub fn trim_matches<'a, P>(&'a self, needle: P) -> &'a OsStr
    where
        P: Needle<&'a OsStr>,
        P::Searcher: Searcher<OsStr>, // FIXME: RFC 2089
        P::Consumer: DoubleEndedConsumer<OsStr>,
    {
        ext::trim(self, needle)
    }

    /// An iterator over the disjoint matches of the needle within the given
    /// `OsStr`.
    #[unstable(feature = "os_str_needle_methods", issue = "56345")]
    #[inline]
    pub fn matches<'a, P>(&'a self, needle: P) -> ext::Matches<&'a OsStr, P::Searcher>
    where
        P: Needle<&'a OsStr>,
        P::Searcher: Searcher<OsStr>, // FIXME: RFC 2089
        P::Consumer: Consumer<OsStr>, // FIXME: RFC 2089
    {
        ext::matches(self, needle)
    }

    /// An iterator over the disjoint matches of the needle within the given
    /// `OsStr`, yielded in reverse order.
    #[unstable(feature = "os_str_needle_methods", issue = "56345")]
    #[inline]
    pub fn rmatches<'a, P>(&'a self, needle: P) -> ext::RMatches<&'a OsStr, P::Searcher>
    where
        P: Needle<&'a OsStr>,
        P::Searcher: ReverseSearcher<OsStr>,
        P::Consumer: Consumer<OsStr>, // FIXME: RFC 2089
    {
        ext::rmatches(self, needle)
    }

    /// An iterator over the disjoint matches of a needle within this `OsStr`
    /// as well as the index that the match starts at.
    #[unstable(feature = "os_str_needle_methods", issue = "56345")]
    #[inline]
    pub fn match_indices<'a, P>(&'a self, needle: P) -> ext::MatchIndices<&'a OsStr, P::Searcher>
    where
        P: Needle<&'a OsStr>,
        P::Searcher: Searcher<OsStr>, // FIXME: RFC 2089
        P::Consumer: Consumer<OsStr>, // FIXME: RFC 2089
    {
        ext::match_indices(self, needle)
    }

    /// An iterator over the disjoint matches of a needle within this `OsStr`,
    /// yielded in reverse order along with the index of the match.
    #[unstable(feature = "os_str_needle_methods", issue = "56345")]
    #[inline]
    pub fn rmatch_indices<'a, P>(&'a self, needle: P) -> ext::RMatchIndices<&'a OsStr, P::Searcher>
    where
        P: Needle<&'a OsStr>,
        P::Searcher: ReverseSearcher<OsStr>,
        P::Consumer: Consumer<OsStr>, // FIXME: RFC 2089
    {
        ext::rmatch_indices(self, needle)
    }

    /// An iterator over the disjoint matches of a needle within this `OsStr`
    /// as well as the index ranges of each match.
    #[unstable(feature = "os_str_needle_methods", issue = "56345")]
    #[inline]
    pub fn match_ranges<'a, P>(&'a self, needle: P) -> ext::MatchRanges<&'a OsStr, P::Searcher>
    where
        P: Needle<&'a OsStr>,
        P::Searcher: Searcher<OsStr>, // FIXME: RFC 2089
        P::Consumer: Consumer<OsStr>, // FIXME: RFC 2089
    {
        ext::match_ranges(self, needle)
    }

    /// An iterator over the disjoint matches of a needle within this `OsStr`,
    /// yielded in reverse order along with the index ranges of each match.
    #[unstable(feature = "os_str_needle_methods", issue = "56345")]
    #[inline]
    pub fn rmatch_ranges<'a, P>(&'a self, needle: P) -> ext::RMatchRanges<&'a OsStr, P::Searcher>
    where
        P: Needle<&'a OsStr>,
        P::Searcher: ReverseSearcher<OsStr>,
        P::Consumer: Consumer<OsStr>, // FIXME: RFC 2089
    {
        ext::rmatch_ranges(self, needle)
    }

    /// An iterator over slices of this `OsStr`, separated by parts matched by
    /// the needle.
    #[unstable(feature = "os_str_needle_methods", issue = "56345")]
    #[inline]
    pub fn split<'a, P>(&'a self, needle: P) -> ext::Split<&'a OsStr, P::Searcher>
    where
        P: Needle<&'a OsStr>,
        P::Searcher: Searcher<OsStr>, // FIXME: RFC 2089
        P::Consumer: Consumer<OsStr>, // FIXME: RFC 2089
    {
        ext::split(self, needle)
    }

    /// An iterator over slices of this `OsStr`, separated by parts matched by
    /// the needle and yielded in reverse order.
    #[unstable(feature = "os_str_needle_methods", issue = "56345")]
    #[inline]
    pub fn rsplit<'a, P>(&'a self, needle: P) -> ext::RSplit<&'a OsStr, P::Searcher>
    where
        P: Needle<&'a OsStr>,
        P::Searcher: ReverseSearcher<OsStr>,
        P::Consumer: Consumer<OsStr>, // FIXME: RFC 2089
    {
        ext::rsplit(self, needle)
    }

    /// An iterator over slices of this `OsStr`, separated by parts matched by
    /// the needle.
    ///
    /// Equivalent to [`split`](#method.split), except that the trailing slice
    /// is skipped if empty.
    #[unstable(feature = "os_str_needle_methods", issue = "56345")]
    #[inline]
    pub fn split_terminator<'a, P>(&'a self, needle: P)
        -> ext::SplitTerminator<&'a OsStr, P::Searcher>
    where
        P: Needle<&'a OsStr>,
        P::Searcher: Searcher<OsStr>, // FIXME: RFC 2089
        P::Consumer: Consumer<OsStr>, // FIXME: RFC 2089
    {
        ext::split_terminator(self, needle)
    }

    /// An iterator over slices of this `OsStr`, separated by parts matched by
    /// the needle and yielded in reverse order.
    ///
    /// Equivalent to [`rsplit`](#method.rsplit), except that the trailing slice
    /// is skipped if empty.
    #[unstable(feature = "os_str_needle_methods", issue = "56345")]
    #[inline]
    pub fn rsplit_terminator<'a, P>(&'a self, needle: P)
        -> ext::RSplitTerminator<&'a OsStr, P::Searcher>
    where
        P: Needle<&'a OsStr>,
        P::Searcher: ReverseSearcher<OsStr>,
        P::Consumer: Consumer<OsStr>, // FIXME: RFC 2089
    {
        ext::rsplit_terminator(self, needle)
    }

    /// An iterator over slices of the given `OsStr`, separated by a needle,
    /// restricted to returning at most `n` items.
    #[unstable(feature = "os_str_needle_methods", issue = "56345")]
    #[inline]
    pub fn splitn<'a, P>(&'a self, n: usize, needle: P) -> ext::SplitN<&'a OsStr, P::Searcher>
    where
        P: Needle<&'a OsStr>,
        P::Searcher: Searcher<OsStr>, // FIXME: RFC 2089
        P::Consumer: Consumer<OsStr>, // FIXME: RFC 2089
    {
        ext::splitn(self, n, needle)
    }

    /// An iterator over slices of the given `OsStr`, separated by a needle,
    /// starting from the end of the `OsStr`, restricted to returning at most
    /// `n` items.
    #[unstable(feature = "os_str_needle_methods", issue = "56345")]
    #[inline]
    pub fn rsplitn<'a, P>(&'a self, n: usize, needle: P) -> ext::RSplitN<&'a OsStr, P::Searcher>
    where
        P: Needle<&'a OsStr>,
        P::Searcher: ReverseSearcher<OsStr>,
        P::Consumer: Consumer<OsStr>, // FIXME: RFC 2089
    {
        ext::rsplitn(self, n, needle)
    }

    /// Replaces all matches of a needle with another `OsStr`.
    #[unstable(feature = "os_str_needle_methods", issue = "56345")]
    #[inline]
    pub fn replace<'s: 'a, 'a, P>(&'s self, from: P, to: &'a OsStr) -> OsString
    where
        P: Needle<&'a OsStr>,
        P::Searcher: Searcher<OsStr>, // FIXME: RFC 2089
        P::Consumer: Consumer<OsStr>, // FIXME: RFC 2089
    {
        let mut result = OsString::with_capacity(self.len());
        ext::replace_with(self, from, |_| to, |s| result.push(s));
        result
    }

    /// Replaces first N matches of a needle with another `OsStr`.
    #[unstable(feature = "os_str_needle_methods", issue = "56345")]
    #[inline]
    pub fn replacen<'s: 'a, 'a, P>(&'s self, from: P, to: &'a OsStr, count: usize) -> OsString
    where
        P: Needle<&'a OsStr>,
        P::Searcher: Searcher<OsStr>, // FIXME: RFC 2089
        P::Consumer: Consumer<OsStr>, // FIXME: RFC 2089
    {
        let mut result = OsString::with_capacity(self.len());
        ext::replacen_with(self, from, |_| to, count, |s| result.push(s));
        result
    }
}

#[stable(feature = "box_from_os_str", since = "1.17.0")]
impl From<&OsStr> for Box<OsStr> {
    fn from(s: &OsStr) -> Box<OsStr> {
        let rw = Box::into_raw(s.inner.into_box()) as *mut OsStr;
        unsafe { Box::from_raw(rw) }
    }
}

#[stable(feature = "os_string_from_box", since = "1.18.0")]
impl From<Box<OsStr>> for OsString {
    /// Converts a `Box<OsStr>` into a `OsString` without copying or allocating.
    ///
    /// [`Box`]: ../boxed/struct.Box.html
    /// [`OsString`]: ../ffi/struct.OsString.html
    fn from(boxed: Box<OsStr>) -> OsString {
        boxed.into_os_string()
    }
}

#[stable(feature = "box_from_os_string", since = "1.20.0")]
impl From<OsString> for Box<OsStr> {
    /// Converts a [`OsString`] into a [`Box`]`<OsStr>` without copying or allocating.
    ///
    /// [`Box`]: ../boxed/struct.Box.html
    /// [`OsString`]: ../ffi/struct.OsString.html
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
    /// Converts a [`OsString`] into a [`Arc`]`<OsStr>` without copying or allocating.
    ///
    /// [`Arc`]: ../sync/struct.Arc.html
    /// [`OsString`]: ../ffi/struct.OsString.html
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
    /// Converts a [`OsString`] into a [`Rc`]`<OsStr>` without copying or allocating.
    ///
    /// [`Rc`]: ../rc/struct.Rc.html
    /// [`OsString`]: ../ffi/struct.OsString.html
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
    fn eq(&self, other: &OsStr) -> bool {
        self.inner == other.inner
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl PartialEq<str> for OsStr {
    fn eq(&self, other: &str) -> bool {
        *self == *OsStr::new(other)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl PartialEq<OsStr> for str {
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
        self.inner.partial_cmp(&other.inner)
    }
    #[inline]
    fn lt(&self, other: &OsStr) -> bool { self.inner < other.inner }
    #[inline]
    fn le(&self, other: &OsStr) -> bool { self.inner <= other.inner }
    #[inline]
    fn gt(&self, other: &OsStr) -> bool { self.inner > other.inner }
    #[inline]
    fn ge(&self, other: &OsStr) -> bool { self.inner >= other.inner }
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
    fn cmp(&self, other: &OsStr) -> cmp::Ordering { self.inner.cmp(&other.inner) }
}

macro_rules! impl_cmp {
    ($lhs:ty, $rhs: ty) => {
        #[stable(feature = "cmp_os_str", since = "1.8.0")]
        impl<'a, 'b> PartialEq<$rhs> for $lhs {
            #[inline]
            fn eq(&self, other: &$rhs) -> bool { <OsStr as PartialEq>::eq(self, other) }
        }

        #[stable(feature = "cmp_os_str", since = "1.8.0")]
        impl<'a, 'b> PartialEq<$lhs> for $rhs {
            #[inline]
            fn eq(&self, other: &$lhs) -> bool { <OsStr as PartialEq>::eq(self, other) }
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
    }
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
        self.inner.hash(state)
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
    fn borrow(&self) -> &OsStr { &self[..] }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl ToOwned for OsStr {
    type Owned = OsString;
    fn to_owned(&self) -> OsString {
        self.to_os_string()
    }
    fn clone_into(&self, target: &mut OsString) {
        target.clear();
        target.push(self);
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl AsRef<OsStr> for OsStr {
    fn as_ref(&self) -> &OsStr {
        self
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl AsRef<OsStr> for OsString {
    fn as_ref(&self) -> &OsStr {
        self
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl AsRef<OsStr> for str {
    fn as_ref(&self) -> &OsStr {
        OsStr::from_inner(Slice::from_str(self))
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl AsRef<OsStr> for String {
    fn as_ref(&self) -> &OsStr {
        (&**self).as_ref()
    }
}

impl FromInner<Buf> for OsString {
    fn from_inner(buf: Buf) -> OsString {
        OsString { inner: buf }
    }
}

impl IntoInner<Buf> for OsString {
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

#[stable(feature = "os_str_slice", since = "1.36.0")]
impl ops::Index<ops::RangeFull> for OsStr {
    type Output = OsStr;

    #[inline]
    fn index(&self, _: ops::RangeFull) -> &OsStr {
        self
    }
}

#[stable(feature = "os_str_slice", since = "1.36.0")]
impl ops::Index<ops::Range<usize>> for OsStr {
    type Output = OsStr;

    #[inline]
    fn index(&self, index: ops::Range<usize>) -> &OsStr {
        OsStr::from_inner(&self.inner[index])
    }
}

#[stable(feature = "os_str_slice", since = "1.36.0")]
impl ops::Index<ops::RangeFrom<usize>> for OsStr {
    type Output = OsStr;

    #[inline]
    fn index(&self, index: ops::RangeFrom<usize>) -> &OsStr {
        OsStr::from_inner(&self.inner[index])
    }
}

#[stable(feature = "os_str_slice", since = "1.36.0")]
impl ops::Index<ops::RangeTo<usize>> for OsStr {
    type Output = OsStr;

    #[inline]
    fn index(&self, index: ops::RangeTo<usize>) -> &OsStr {
        OsStr::from_inner(&self.inner[index])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sys_common::{AsInner, IntoInner};

    use crate::rc::Rc;
    use crate::sync::Arc;

    #[test]
    fn test_os_string_with_capacity() {
        let os_string = OsString::with_capacity(0);
        assert_eq!(0, os_string.inner.into_inner().capacity());

        let os_string = OsString::with_capacity(10);
        assert_eq!(10, os_string.inner.into_inner().capacity());

        let mut os_string = OsString::with_capacity(0);
        os_string.push("abc");
        assert!(os_string.inner.into_inner().capacity() >= 3);
    }

    #[test]
    fn test_os_string_clear() {
        let mut os_string = OsString::from("abc");
        assert_eq!(3, os_string.inner.as_inner().len());

        os_string.clear();
        assert_eq!(&os_string, "");
        assert_eq!(0, os_string.inner.as_inner().len());
    }

    #[test]
    fn test_os_string_capacity() {
        let os_string = OsString::with_capacity(0);
        assert_eq!(0, os_string.capacity());

        let os_string = OsString::with_capacity(10);
        assert_eq!(10, os_string.capacity());

        let mut os_string = OsString::with_capacity(0);
        os_string.push("abc");
        assert!(os_string.capacity() >= 3);
    }

    #[test]
    fn test_os_string_reserve() {
        let mut os_string = OsString::new();
        assert_eq!(os_string.capacity(), 0);

        os_string.reserve(2);
        assert!(os_string.capacity() >= 2);

        for _ in 0..16 {
            os_string.push("a");
        }

        assert!(os_string.capacity() >= 16);
        os_string.reserve(16);
        assert!(os_string.capacity() >= 32);

        os_string.push("a");

        os_string.reserve(16);
        assert!(os_string.capacity() >= 33)
    }

    #[test]
    fn test_os_string_reserve_exact() {
        let mut os_string = OsString::new();
        assert_eq!(os_string.capacity(), 0);

        os_string.reserve_exact(2);
        assert!(os_string.capacity() >= 2);

        for _ in 0..16 {
            os_string.push("a");
        }

        assert!(os_string.capacity() >= 16);
        os_string.reserve_exact(16);
        assert!(os_string.capacity() >= 32);

        os_string.push("a");

        os_string.reserve_exact(16);
        assert!(os_string.capacity() >= 33)
    }

    #[test]
    fn test_os_string_default() {
        let os_string: OsString = Default::default();
        assert_eq!("", &os_string);
    }

    #[test]
    fn test_os_str_is_empty() {
        let mut os_string = OsString::new();
        assert!(os_string.is_empty());

        os_string.push("abc");
        assert!(!os_string.is_empty());

        os_string.clear();
        assert!(os_string.is_empty());
    }

    #[test]
    fn test_os_str_len() {
        let mut os_string = OsString::new();
        assert_eq!(0, os_string.len());

        os_string.push("abc");
        assert_eq!(3, os_string.len());

        os_string.clear();
        assert_eq!(0, os_string.len());
    }

    #[test]
    fn test_os_str_default() {
        let os_str: &OsStr = Default::default();
        assert_eq!("", os_str);
    }

    #[test]
    fn into_boxed() {
        let orig = "Hello, world!";
        let os_str = OsStr::new(orig);
        let boxed: Box<OsStr> = Box::from(os_str);
        let os_string = os_str.to_owned().into_boxed_os_str().into_os_string();
        assert_eq!(os_str, &*boxed);
        assert_eq!(&*boxed, &*os_string);
        assert_eq!(&*os_string, os_str);
    }

    #[test]
    fn boxed_default() {
        let boxed = <Box<OsStr>>::default();
        assert!(boxed.is_empty());
    }

    #[test]
    fn test_os_str_clone_into() {
        let mut os_string = OsString::with_capacity(123);
        os_string.push("hello");
        let os_str = OsStr::new("bonjour");
        os_str.clone_into(&mut os_string);
        assert_eq!(os_str, os_string);
        assert!(os_string.capacity() >= 123);
    }

    #[test]
    fn into_rc() {
        let orig = "Hello, world!";
        let os_str = OsStr::new(orig);
        let rc: Rc<OsStr> = Rc::from(os_str);
        let arc: Arc<OsStr> = Arc::from(os_str);

        assert_eq!(&*rc, os_str);
        assert_eq!(&*arc, os_str);

        let rc2: Rc<OsStr> = Rc::from(os_str.to_owned());
        let arc2: Arc<OsStr> = Arc::from(os_str.to_owned());

        assert_eq!(&*rc2, os_str);
        assert_eq!(&*arc2, os_str);
    }

    #[test]
    fn slice_with_utf8_boundary() {
        let os_str = OsStr::new("Hello🌍🌎🌏");
        assert_eq!(os_str.len(), 17);

        assert_eq!(os_str, &os_str[..]);
        assert_eq!(os_str, &os_str[..17]);
        assert_eq!(os_str, &os_str[0..]);
        assert_eq!(os_str, &os_str[0..17]);

        assert_eq!(OsStr::new("Hello"), &os_str[..5]);
        assert_eq!(OsStr::new("🌎🌏"), &os_str[9..]);
        assert_eq!(OsStr::new("lo🌍"), &os_str[3..9]);

        let os_string = os_str.to_owned();
        assert_eq!(os_str, &os_string[..]);
        assert_eq!(os_str, &os_string[..17]);
        assert_eq!(os_str, &os_string[0..]);
        assert_eq!(os_str, &os_string[0..17]);

        assert_eq!(OsStr::new("Hello"), &os_string[..5]);
        assert_eq!(OsStr::new("🌎🌏"), &os_string[9..]);
        assert_eq!(OsStr::new("lo🌍"), &os_string[3..9]);
    }

    #[test]
    #[cfg(any(unix, target_os = "redox", target_arch = "wasm32"))]
    fn slice_with_non_utf8_boundary_unix() {
        #[cfg(unix)]
        use crate::os::unix::ffi::OsStrExt;
        #[cfg(target_os = "redox")]
        use crate::os::redox::ffi::OsStrExt;

        let os_str = OsStr::new("Hello🌍🌎🌏");
        assert_eq!(OsStr::from_bytes(b"Hello\xf0"), &os_str[..6]);
        assert_eq!(OsStr::from_bytes(b"\x9f\x8c\x8e\xf0\x9f\x8c\x8f"), &os_str[10..]);
        assert_eq!(OsStr::from_bytes(b"\x8d\xf0\x9f\x8c\x8e"), &os_str[8..13]);
    }

    #[test]
    #[cfg(windows)]
    fn slice_with_non_utf8_boundary_windows() {
        use crate::os::windows::ffi::OsStringExt;

        let os_str = OsStr::new("Hello🌍🌎🌏");
        assert_eq!(OsString::from_wide(&[0x48, 0x65, 0x6C, 0x6C, 0x6F, 0xD83C]), &os_str[..7]);
        assert_eq!(OsString::from_wide(&[0xDF0E, 0xD83C, 0xDF0F]), &os_str[11..]);
        assert_eq!(OsString::from_wide(&[0xDF0D, 0xD83C]), &os_str[7..11]);
    }
}

#[unstable(feature = "needle", issue = "56345")]
unsafe impl Hay for OsStr {
    type Index = usize;

    #[inline]
    fn empty<'a>() -> &'a Self {
        Self::new("")
    }

    #[inline]
    fn start_index(&self) -> usize {
        0
    }

    #[inline]
    fn end_index(&self) -> usize {
        self.len()
    }

    #[inline]
    unsafe fn slice_unchecked(&self, range: ops::Range<usize>) -> &Self {
        &self[range]
    }

    #[inline]
    unsafe fn next_index(&self, index: usize) -> usize {
        self.inner.next_index(index)
    }

    #[inline]
    unsafe fn prev_index(&self, index: usize) -> usize {
        self.inner.prev_index(index)
    }
}

// use a macro here since the type of `hay.inner.inner` is platform dependent
// and we don't want to expose that type.
macro_rules! span_as_inner {
    ($span:expr) => {{
        let (hay, range) = $span.into_parts();
        unsafe { Span::from_parts(&hay.inner.inner, range) }
    }}
}

fn span_as_inner_bytes(span: Span<&OsStr>) -> Span<&[u8]> {
    let (hay, range) = span.into_parts();
    unsafe { Span::from_parts(hay.inner.as_bytes_for_searcher(), range) }
}

#[unstable(feature = "needle", issue = "56345")]
unsafe impl<'p> Searcher<OsStr> for TwoWaySearcher<'p, u8> {
    #[inline]
    fn search(&mut self, span: Span<&OsStr>) -> Option<ops::Range<usize>> {
        self.search(span_as_inner_bytes(span))
    }
}

#[unstable(feature = "needle", issue = "56345")]
unsafe impl<'p> ReverseSearcher<OsStr> for TwoWaySearcher<'p, u8> {
    #[inline]
    fn rsearch(&mut self, span: Span<&OsStr>) -> Option<ops::Range<usize>> {
        self.rsearch(span_as_inner_bytes(span))
    }
}

#[unstable(feature = "needle", issue = "56345")]
unsafe impl<'p> Consumer<OsStr> for NaiveSearcher<'p, u8> {
    #[inline]
    fn consume(&mut self, span: Span<&OsStr>) -> Option<usize> {
        self.consume(span_as_inner_bytes(span))
    }

    #[inline]
    fn trim_start(&mut self, hay: &OsStr) -> usize {
        self.trim_start(hay.inner.as_bytes_for_searcher())
    }
}

#[unstable(feature = "needle", issue = "56345")]
unsafe impl<'p> ReverseConsumer<OsStr> for NaiveSearcher<'p, u8> {
    #[inline]
    fn rconsume(&mut self, span: Span<&OsStr>) -> Option<usize> {
        self.rconsume(span_as_inner_bytes(span))
    }

    #[inline]
    fn trim_end(&mut self, hay: &OsStr) -> usize {
        self.trim_end(hay.inner.as_bytes_for_searcher())
    }
}

#[unstable(feature = "needle", issue = "56345")]
#[derive(Debug)]
pub struct OsStrSearcher<S>(InnerSearcher<S>);

#[unstable(feature = "needle", issue = "56345")]
unsafe impl<'p> Searcher<OsStr> for OsStrSearcher<SliceSearcher<'p, u8>> {
    #[inline]
    fn search(&mut self, span: Span<&OsStr>) -> Option<ops::Range<usize>> {
        self.0.search(span_as_inner!(span))
    }
}

#[unstable(feature = "needle", issue = "56345")]
unsafe impl<'p> ReverseSearcher<OsStr> for OsStrSearcher<SliceSearcher<'p, u8>> {
    #[inline]
    fn rsearch(&mut self, span: Span<&OsStr>) -> Option<ops::Range<usize>> {
        self.0.rsearch(span_as_inner!(span))
    }
}

#[unstable(feature = "needle", issue = "56345")]
unsafe impl<'p> Consumer<OsStr> for OsStrSearcher<NaiveSearcher<'p, u8>> {
    #[inline]
    fn consume(&mut self, span: Span<&OsStr>) -> Option<usize> {
        self.0.consume(span_as_inner!(span))
    }

    #[inline]
    fn trim_start(&mut self, hay: &OsStr) -> usize {
        self.0.trim_start(&hay.inner.inner)
    }
}

#[unstable(feature = "needle", issue = "56345")]
unsafe impl<'p> ReverseConsumer<OsStr> for OsStrSearcher<NaiveSearcher<'p, u8>> {
    #[inline]
    fn rconsume(&mut self, span: Span<&OsStr>) -> Option<usize> {
        self.0.rconsume(span_as_inner!(span))
    }

    #[inline]
    fn trim_end(&mut self, hay: &OsStr) -> usize {
        self.0.trim_end(&hay.inner.inner)
    }
}

#[unstable(feature = "needle", issue = "56345")]
impl<'p, H: Haystack<Target = OsStr>> Needle<H> for &'p OsStr {
    type Searcher = OsStrSearcher<SliceSearcher<'p, u8>>;
    type Consumer = OsStrSearcher<NaiveSearcher<'p, u8>>;

    fn into_searcher(self) -> Self::Searcher {
        OsStrSearcher(self.inner.into_searcher())
    }

    fn into_consumer(self) -> Self::Consumer {
        OsStrSearcher(self.inner.into_consumer())
    }
}

// FIXME cannot impl `Needle<(_: Haystack<Target = OsStr>)>` due to RFC 1672 being postponed.
// (need to wait for chalk)
#[unstable(feature = "needle", issue = "56345")]
impl<'h, 'p> Needle<&'h OsStr> for &'p str {
    type Searcher = SliceSearcher<'p, u8>;
    type Consumer = NaiveSearcher<'p, u8>;

    fn into_searcher(self) -> Self::Searcher {
        SliceSearcher::new(self.as_bytes())
    }

    fn into_consumer(self) -> Self::Consumer {
        NaiveSearcher::new(self.as_bytes())
    }
}

#[cfg(test)]
mod needle_tests {
    use super::*;

    #[cfg(windows)]
    use crate::os::windows::ffi::OsStringExt;
    #[cfg(unix)]
    use crate::os::unix::ffi::OsStrExt;

    #[test]
    #[cfg(any(unix, target_os = "redox", target_arch = "wasm32"))]
    fn test_trim() {
        assert_eq!(
            OsStr::from_bytes(b"\xaa\xbb\xaa\xcc\xaa\xbb\xaa")
                .trim_start_matches(OsStr::from_bytes(b"\xaa")),
            OsStr::from_bytes(b"\xbb\xaa\xcc\xaa\xbb\xaa"),
        );
        assert_eq!(
            OsStr::from_bytes(b"\xaa\xbb\xaa\xcc\xaa\xbb\xaa")
                .trim_end_matches(OsStr::from_bytes(b"\xaa")),
            OsStr::from_bytes(b"\xaa\xbb\xaa\xcc\xaa\xbb"),
        );
    }

    #[test]
    #[cfg(windows)]
    fn test_trim_start_low_surrogate() {
        let pat = OsString::from_wide(&[0xdc00]);
        let a = &OsStr::new("\u{10000}aaa")[2..];
        assert_eq!(a.trim_start_matches(&*pat), OsStr::new("aaa"));

        let b = OsString::from_wide(&[0xd800, 0xdc00, 0xdc00, 0x62, 0x62, 0x62]);
        assert_eq!(b[2..].trim_start_matches(&*pat), OsStr::new("bbb"));

        let c = OsString::from_wide(&[0xdc00, 0xdc00, 0x63, 0x63, 0x63]);
        assert_eq!(c.trim_start_matches(&*pat), OsStr::new("ccc"));

        let d = &OsStr::new("\u{ffc00}ddd")[2..];
        assert_eq!(d.trim_start_matches(&*pat), OsStr::new("ddd"));

        let e = OsStr::new("㰀eee");
        assert_eq!(e.trim_start_matches(&*pat), e);
    }

    #[test]
    #[cfg(windows)]
    fn test_trim_start_high_surrogate() {
        let pat = OsString::from_wide(&[0xd800]);
        let a = OsStr::new("\u{10000}");
        assert_eq!(a.trim_start_matches(&*pat), &*OsString::from_wide(&[0xdc00]));

        let b = OsString::from_wide(&[0xd800, 0x62, 0x62, 0x62]);
        assert_eq!(b.trim_start_matches(&*pat), OsStr::new("bbb"));

        let c = OsString::from_wide(&[0xd800, 0xd800, 0xdc00, 0x63, 0x63, 0x63]);
        assert_eq!(c.trim_start_matches(&*pat), &c[5..]);
    }

    #[test]
    #[cfg(windows)]
    fn test_trim_end_high_surrogate() {
        let pat = OsString::from_wide(&[0xd800]);
        let a = OsStr::new("aaa\u{10000}");
        assert_eq!(a[..a.len()-2].trim_end_matches(&*pat), OsStr::new("aaa"));

        let b = OsString::from_wide(&[0x62, 0x62, 0x62, 0xd800, 0xd800, 0xdc00]);
        assert_eq!(b[..b.len()-2].trim_end_matches(&*pat), OsStr::new("bbb"));

        let c = OsString::from_wide(&[0x63, 0x63, 0x63, 0xd800, 0xd800]);
        assert_eq!(c.trim_end_matches(&*pat), OsStr::new("ccc"));

        let d = OsStr::new("ddd\u{103ff}");
        assert_eq!(d[..d.len()-2].trim_end_matches(&*pat), OsStr::new("ddd"));

        let e = OsStr::new("eee\u{11000}");
        let e = &e[..e.len()-2];
        assert_eq!(e.trim_end_matches(&*pat), e);

        let f = OsString::from_wide(&[0x66, 0x66, 0x66, 0xdc00]);
        assert_eq!(f.trim_end_matches(&*pat), &*f);
    }


    #[test]
    #[cfg(windows)]
    fn test_trim_end_low_surrogate() {
        let pat = OsString::from_wide(&[0xdc00]);
        let a = OsStr::new("\u{10000}");
        assert_eq!(a.trim_end_matches(&*pat), &*OsString::from_wide(&[0xd800]));

        let b = OsString::from_wide(&[0x62, 0x62, 0x62, 0xdc00]);
        assert_eq!(b.trim_end_matches(&*pat), OsStr::new("bbb"));

        let c = OsString::from_wide(&[0x63, 0x63, 0x63, 0xdbff, 0xdc00, 0xdc00]);
        assert_eq!(c.trim_end_matches(&*pat), &c[..c.len()-5]);
    }

    #[test]
    #[cfg(windows)]
    fn test_match_string_with_surrogates() {
        let haystack = &OsStr::new("\u{10000}a\u{10000}a\u{10000}\u{10000}")[2..16];
        // 0..3 = U+DC00
        // 3..4 = 'a'
        // 4..6 = U+D800
        // 6..8 = U+DC00
        // 8..9 = 'a'
        // 9..11 = U+D800
        // 11..13 = U+DC00
        // 13..16 = U+D800

        let pat = "a";
        let matched_pat = OsStr::new(pat);
        assert_eq!(haystack.match_ranges(pat).collect::<Vec<_>>(), vec![
            (3..4, matched_pat),
            (8..9, matched_pat),
        ]);
        assert_eq!(haystack.rmatch_ranges(pat).collect::<Vec<_>>(), vec![
            (8..9, matched_pat),
            (3..4, matched_pat),
        ]);

        let pat = OsString::from_wide(&[0xdc00, 0x61]);
        assert_eq!(haystack.match_ranges(&*pat).collect::<Vec<_>>(), vec![
            (0..4, &*pat),
            (6..9, &*pat),
        ]);
        assert_eq!(haystack.rmatch_ranges(&*pat).collect::<Vec<_>>(), vec![
            (6..9, &*pat),
            (0..4, &*pat),
        ]);

        let pat = OsString::from_wide(&[0x61, 0xd800]);
        assert_eq!(haystack.match_ranges(&*pat).collect::<Vec<_>>(), vec![
            (3..6, &*pat),
            (8..11, &*pat),
        ]);
        assert_eq!(haystack.rmatch_ranges(&*pat).collect::<Vec<_>>(), vec![
            (8..11, &*pat),
            (3..6, &*pat),
        ]);

        let pat = "\u{10000}";
        let matched_pat = OsStr::new(pat);
        assert_eq!(haystack.match_ranges(pat).collect::<Vec<_>>(), vec![
            (4..8, matched_pat),
            (9..13, matched_pat),
        ]);
        assert_eq!(haystack.rmatch_ranges(pat).collect::<Vec<_>>(), vec![
            (9..13, matched_pat),
            (4..8, matched_pat),
        ]);

        let pat = OsString::from_wide(&[0xd800]);
        assert_eq!(haystack.match_ranges(&*pat).collect::<Vec<_>>(), vec![
            (4..6, &*pat),
            (9..11, &*pat),
            (13..16, &*pat),
        ]);
        assert_eq!(haystack.rmatch_ranges(&*pat).collect::<Vec<_>>(), vec![
            (13..16, &*pat),
            (9..11, &*pat),
            (4..6, &*pat),
        ]);

        let pat = OsString::from_wide(&[0xdc00]);
        assert_eq!(haystack.match_ranges(&*pat).collect::<Vec<_>>(), vec![
            (0..3, &*pat),
            (6..8, &*pat),
            (11..13, &*pat),
        ]);
        assert_eq!(haystack.rmatch_ranges(&*pat).collect::<Vec<_>>(), vec![
            (11..13, &*pat),
            (6..8, &*pat),
            (0..3, &*pat),
        ]);

        let pat = OsString::from_wide(&[0xdc00, 0xd800]);
        assert_eq!(haystack.match_ranges(&*pat).collect::<Vec<_>>(), vec![
            (11..16, &*pat),
        ]);
        assert_eq!(haystack.rmatch_ranges(&*pat).collect::<Vec<_>>(), vec![
            (11..16, &*pat),
        ]);
    }
}

