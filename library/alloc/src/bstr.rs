//! The `ByteStr` and `ByteString` types and trait implementations.

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
use crate::rc::Rc;
use crate::string::String;
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
///
/// # Examples
///
/// You can create a new `ByteString` from a `Vec<u8>` directly, or via a `From` impl from various
/// string types:
///
/// ```
/// # #![feature(bstr)]
/// # use std::bstr::ByteString;
/// let s1 = ByteString(vec![b'H', b'e', b'l', b'l', b'o']);
/// let s2 = ByteString::from("Hello");
/// let s3 = ByteString::from(b"Hello");
/// assert_eq!(s1, s2);
/// assert_eq!(s2, s3);
/// ```
#[unstable(feature = "bstr", issue = "134915")]
#[repr(transparent)]
#[derive(Clone)]
pub struct ByteString(pub Vec<u8>);

impl ByteString {
    #[inline]
    pub(crate) fn as_bytes(&self) -> &[u8] {
        &self.0
    }

    #[inline]
    pub(crate) fn as_bytestr(&self) -> &ByteStr {
        ByteStr::new(&self.0)
    }

    #[inline]
    pub(crate) fn as_mut_bytestr(&mut self) -> &mut ByteStr {
        ByteStr::from_bytes_mut(&mut self.0)
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl Deref for ByteString {
    type Target = Vec<u8>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl DerefMut for ByteString {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[unstable(feature = "deref_pure_trait", issue = "87121")]
unsafe impl DerefPure for ByteString {}

#[unstable(feature = "bstr", issue = "134915")]
impl fmt::Debug for ByteString {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self.as_bytestr(), f)
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl fmt::Display for ByteString {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self.as_bytestr(), f)
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl AsRef<[u8]> for ByteString {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl AsRef<ByteStr> for ByteString {
    #[inline]
    fn as_ref(&self) -> &ByteStr {
        self.as_bytestr()
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl AsMut<[u8]> for ByteString {
    #[inline]
    fn as_mut(&mut self) -> &mut [u8] {
        &mut self.0
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl AsMut<ByteStr> for ByteString {
    #[inline]
    fn as_mut(&mut self) -> &mut ByteStr {
        self.as_mut_bytestr()
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl Borrow<[u8]> for ByteString {
    #[inline]
    fn borrow(&self) -> &[u8] {
        &self.0
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl Borrow<ByteStr> for ByteString {
    #[inline]
    fn borrow(&self) -> &ByteStr {
        self.as_bytestr()
    }
}

// `impl Borrow<ByteStr> for Vec<u8>` omitted to avoid inference failures
// `impl Borrow<ByteStr> for String` omitted to avoid inference failures

#[unstable(feature = "bstr", issue = "134915")]
impl BorrowMut<[u8]> for ByteString {
    #[inline]
    fn borrow_mut(&mut self) -> &mut [u8] {
        &mut self.0
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl BorrowMut<ByteStr> for ByteString {
    #[inline]
    fn borrow_mut(&mut self) -> &mut ByteStr {
        self.as_mut_bytestr()
    }
}

// `impl BorrowMut<ByteStr> for Vec<u8>` omitted to avoid inference failures

#[unstable(feature = "bstr", issue = "134915")]
impl Default for ByteString {
    fn default() -> Self {
        ByteString(Vec::new())
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl<'a, const N: usize> From<&'a [u8; N]> for ByteString {
    #[inline]
    fn from(s: &'a [u8; N]) -> Self {
        ByteString(s.as_slice().to_vec())
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl<const N: usize> From<[u8; N]> for ByteString {
    #[inline]
    fn from(s: [u8; N]) -> Self {
        ByteString(s.as_slice().to_vec())
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl<'a> From<&'a [u8]> for ByteString {
    #[inline]
    fn from(s: &'a [u8]) -> Self {
        ByteString(s.to_vec())
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl From<Vec<u8>> for ByteString {
    #[inline]
    fn from(s: Vec<u8>) -> Self {
        ByteString(s)
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl From<ByteString> for Vec<u8> {
    #[inline]
    fn from(s: ByteString) -> Self {
        s.0
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl<'a> From<&'a str> for ByteString {
    #[inline]
    fn from(s: &'a str) -> Self {
        ByteString(s.as_bytes().to_vec())
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl From<String> for ByteString {
    #[inline]
    fn from(s: String) -> Self {
        ByteString(s.into_bytes())
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl<'a> From<&'a ByteStr> for ByteString {
    #[inline]
    fn from(s: &'a ByteStr) -> Self {
        ByteString(s.0.to_vec())
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
        Cow::Borrowed(s.as_bytestr())
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
        let mut buf = Vec::new();
        for b in iter {
            buf.extend_from_slice(b);
        }
        ByteString(buf)
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl<'a> FromIterator<&'a ByteStr> for ByteString {
    #[inline]
    fn from_iter<T: IntoIterator<Item = &'a ByteStr>>(iter: T) -> Self {
        let mut buf = Vec::new();
        for b in iter {
            buf.extend_from_slice(&b.0);
        }
        ByteString(buf)
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
        self.as_bytestr()
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl Index<Range<usize>> for ByteString {
    type Output = ByteStr;

    #[inline]
    fn index(&self, r: Range<usize>) -> &ByteStr {
        ByteStr::from_bytes(&self.0[r])
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl Index<RangeInclusive<usize>> for ByteString {
    type Output = ByteStr;

    #[inline]
    fn index(&self, r: RangeInclusive<usize>) -> &ByteStr {
        ByteStr::from_bytes(&self.0[r])
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl Index<RangeFrom<usize>> for ByteString {
    type Output = ByteStr;

    #[inline]
    fn index(&self, r: RangeFrom<usize>) -> &ByteStr {
        ByteStr::from_bytes(&self.0[r])
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl Index<RangeTo<usize>> for ByteString {
    type Output = ByteStr;

    #[inline]
    fn index(&self, r: RangeTo<usize>) -> &ByteStr {
        ByteStr::from_bytes(&self.0[r])
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl Index<RangeToInclusive<usize>> for ByteString {
    type Output = ByteStr;

    #[inline]
    fn index(&self, r: RangeToInclusive<usize>) -> &ByteStr {
        ByteStr::from_bytes(&self.0[r])
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
        self.as_mut_bytestr()
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl IndexMut<Range<usize>> for ByteString {
    #[inline]
    fn index_mut(&mut self, r: Range<usize>) -> &mut ByteStr {
        ByteStr::from_bytes_mut(&mut self.0[r])
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl IndexMut<RangeInclusive<usize>> for ByteString {
    #[inline]
    fn index_mut(&mut self, r: RangeInclusive<usize>) -> &mut ByteStr {
        ByteStr::from_bytes_mut(&mut self.0[r])
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl IndexMut<RangeFrom<usize>> for ByteString {
    #[inline]
    fn index_mut(&mut self, r: RangeFrom<usize>) -> &mut ByteStr {
        ByteStr::from_bytes_mut(&mut self.0[r])
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl IndexMut<RangeTo<usize>> for ByteString {
    #[inline]
    fn index_mut(&mut self, r: RangeTo<usize>) -> &mut ByteStr {
        ByteStr::from_bytes_mut(&mut self.0[r])
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl IndexMut<RangeToInclusive<usize>> for ByteString {
    #[inline]
    fn index_mut(&mut self, r: RangeToInclusive<usize>) -> &mut ByteStr {
        ByteStr::from_bytes_mut(&mut self.0[r])
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
        ByteString(self.0.to_vec())
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
        Self::from(Box::<[u8]>::from(&self.0))
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
impl From<Rc<[u8]>> for Rc<ByteStr> {
    #[inline]
    fn from(s: Rc<[u8]>) -> Rc<ByteStr> {
        // SAFETY: `ByteStr` is a transparent wrapper around `[u8]`.
        unsafe { Rc::from_raw(Rc::into_raw(s) as _) }
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl From<Rc<ByteStr>> for Rc<[u8]> {
    #[inline]
    fn from(s: Rc<ByteStr>) -> Rc<[u8]> {
        // SAFETY: `ByteStr` is a transparent wrapper around `[u8]`.
        unsafe { Rc::from_raw(Rc::into_raw(s) as _) }
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl From<Arc<[u8]>> for Arc<ByteStr> {
    #[inline]
    fn from(s: Arc<[u8]>) -> Arc<ByteStr> {
        // SAFETY: `ByteStr` is a transparent wrapper around `[u8]`.
        unsafe { Arc::from_raw(Arc::into_raw(s) as _) }
    }
}

#[unstable(feature = "bstr", issue = "134915")]
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
        Ok(core::str::from_utf8(&s.0)?.into())
    }
}
