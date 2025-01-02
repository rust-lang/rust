//! The `ByteStr` type and trait implementations.

use crate::borrow::{Borrow, BorrowMut};
use crate::cmp::Ordering;
use crate::ops::{
    Deref, DerefMut, DerefPure, Index, IndexMut, Range, RangeFrom, RangeFull, RangeInclusive,
    RangeTo, RangeToInclusive,
};
use crate::{fmt, hash};

/// A wrapper for `&[u8]` representing a human-readable string that's conventionally, but not
/// always, UTF-8.
///
/// Unlike `&str`, this type permits non-UTF-8 contents, making it suitable for user input,
/// non-native filenames (as `Path` only supports native filenames), and other applications that
/// need to round-trip whatever data the user provides.
///
/// For an owned, growable byte string buffer, use
/// [`ByteString`](../../std/bstr/struct.ByteString.html).
///
/// `ByteStr` implements `Deref` to `[u8]`, so all methods available on `[u8]` are available on
/// `ByteStr`.
///
/// # Representation
///
/// A `&ByteStr` has the same representation as a `&str`. That is, a `&ByteStr` is a wide pointer
/// which includes a pointer to some bytes and a length.
///
/// # Trait implementations
///
/// The `ByteStr` type has a number of trait implementations, and in particular, defines equality
/// and comparisons between `&ByteStr`, `&str`, and `&[u8]`, for convenience.
///
/// The `Debug` implementation for `ByteStr` shows its bytes as a normal string, with invalid UTF-8
/// presented as hex escape sequences.
///
/// The `Display` implementation behaves as if the `ByteStr` were first lossily converted to a
/// `str`, with invalid UTF-8 presented as the Unicode replacement character: �
///
#[unstable(feature = "bstr", issue = "134915")]
#[repr(transparent)]
pub struct ByteStr(pub [u8]);

impl ByteStr {
    /// Creates a `ByteStr` slice from anything that can be converted to a byte slice.
    ///
    /// This is a zero-cost conversion.
    ///
    /// # Example
    ///
    /// You can create a `ByteStr` from a byte array, a byte slice or a string slice:
    ///
    /// ```
    /// # #![feature(bstr)]
    /// # use std::bstr::ByteStr;
    /// let a = ByteStr::new(b"abc");
    /// let b = ByteStr::new(&b"abc"[..]);
    /// let c = ByteStr::new("abc");
    ///
    /// assert_eq!(a, b);
    /// assert_eq!(a, c);
    /// ```
    #[inline]
    #[unstable(feature = "bstr", issue = "134915")]
    pub fn new<B: ?Sized + AsRef<[u8]>>(bytes: &B) -> &Self {
        ByteStr::from_bytes(bytes.as_ref())
    }

    #[doc(hidden)]
    #[unstable(feature = "bstr_internals", issue = "none")]
    #[inline]
    pub fn from_bytes(slice: &[u8]) -> &Self {
        // SAFETY: `ByteStr` is a transparent wrapper around `[u8]`, so we can turn a reference to
        // the wrapped type into a reference to the wrapper type.
        unsafe { &*(slice as *const [u8] as *const Self) }
    }

    #[doc(hidden)]
    #[unstable(feature = "bstr_internals", issue = "none")]
    #[inline]
    pub fn from_bytes_mut(slice: &mut [u8]) -> &mut Self {
        // SAFETY: `ByteStr` is a transparent wrapper around `[u8]`, so we can turn a reference to
        // the wrapped type into a reference to the wrapper type.
        unsafe { &mut *(slice as *mut [u8] as *mut Self) }
    }

    #[doc(hidden)]
    #[unstable(feature = "bstr_internals", issue = "none")]
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl Deref for ByteStr {
    type Target = [u8];

    #[inline]
    fn deref(&self) -> &[u8] {
        &self.0
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl DerefMut for ByteStr {
    #[inline]
    fn deref_mut(&mut self) -> &mut [u8] {
        &mut self.0
    }
}

#[unstable(feature = "deref_pure_trait", issue = "87121")]
unsafe impl DerefPure for ByteStr {}

#[unstable(feature = "bstr", issue = "134915")]
impl fmt::Debug for ByteStr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "\"")?;
        for chunk in self.utf8_chunks() {
            for c in chunk.valid().chars() {
                match c {
                    '\0' => write!(f, "\\0")?,
                    '\x01'..='\x7f' => write!(f, "{}", (c as u8).escape_ascii())?,
                    _ => write!(f, "{}", c.escape_debug())?,
                }
            }
            write!(f, "{}", chunk.invalid().escape_ascii())?;
        }
        write!(f, "\"")?;
        Ok(())
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl fmt::Display for ByteStr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fn fmt_nopad(this: &ByteStr, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            for chunk in this.utf8_chunks() {
                f.write_str(chunk.valid())?;
                if !chunk.invalid().is_empty() {
                    f.write_str("\u{FFFD}")?;
                }
            }
            Ok(())
        }

        let Some(align) = f.align() else {
            return fmt_nopad(self, f);
        };
        let nchars: usize = self
            .utf8_chunks()
            .map(|chunk| chunk.valid().len() + if chunk.invalid().is_empty() { 0 } else { 1 })
            .sum();
        let padding = f.width().unwrap_or(0).saturating_sub(nchars);
        let fill = f.fill();
        let (lpad, rpad) = match align {
            fmt::Alignment::Left => (0, padding),
            fmt::Alignment::Right => (padding, 0),
            fmt::Alignment::Center => {
                let half = padding / 2;
                (half, half + padding % 2)
            }
        };
        for _ in 0..lpad {
            write!(f, "{fill}")?;
        }
        fmt_nopad(self, f)?;
        for _ in 0..rpad {
            write!(f, "{fill}")?;
        }

        Ok(())
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl AsRef<[u8]> for ByteStr {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl AsRef<ByteStr> for ByteStr {
    #[inline]
    fn as_ref(&self) -> &ByteStr {
        self
    }
}

// `impl AsRef<ByteStr> for [u8]` omitted to avoid widespread inference failures

#[unstable(feature = "bstr", issue = "134915")]
impl AsRef<ByteStr> for str {
    #[inline]
    fn as_ref(&self) -> &ByteStr {
        ByteStr::new(self)
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl AsMut<[u8]> for ByteStr {
    #[inline]
    fn as_mut(&mut self) -> &mut [u8] {
        &mut self.0
    }
}

// `impl AsMut<ByteStr> for [u8]` omitted to avoid widespread inference failures

// `impl Borrow<ByteStr> for [u8]` omitted to avoid widespread inference failures

// `impl Borrow<ByteStr> for str` omitted to avoid widespread inference failures

#[unstable(feature = "bstr", issue = "134915")]
impl Borrow<[u8]> for ByteStr {
    #[inline]
    fn borrow(&self) -> &[u8] {
        &self.0
    }
}

// `impl BorrowMut<ByteStr> for [u8]` omitted to avoid widespread inference failures

#[unstable(feature = "bstr", issue = "134915")]
impl BorrowMut<[u8]> for ByteStr {
    #[inline]
    fn borrow_mut(&mut self) -> &mut [u8] {
        &mut self.0
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl<'a> Default for &'a ByteStr {
    fn default() -> Self {
        ByteStr::from_bytes(b"")
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl<'a> Default for &'a mut ByteStr {
    fn default() -> Self {
        ByteStr::from_bytes_mut(&mut [])
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl<'a, const N: usize> From<&'a [u8; N]> for &'a ByteStr {
    #[inline]
    fn from(s: &'a [u8; N]) -> Self {
        ByteStr::from_bytes(s)
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl<'a> From<&'a [u8]> for &'a ByteStr {
    #[inline]
    fn from(s: &'a [u8]) -> Self {
        ByteStr::from_bytes(s)
    }
}

// Omitted due to slice-from-array-issue-113238:
//
// #[unstable(feature = "bstr", issue = "134915")]
// impl<'a> From<&'a ByteStr> for &'a [u8] {
//     #[inline]
//     fn from(s: &'a ByteStr) -> Self {
//         &s.0
//     }
// }
//
// #[unstable(feature = "bstr", issue = "134915")]
// impl<'a> From<&'a mut ByteStr> for &'a mut [u8] {
//     #[inline]
//     fn from(s: &'a mut ByteStr) -> Self {
//         &mut s.0
//     }
// }

#[unstable(feature = "bstr", issue = "134915")]
impl<'a> From<&'a str> for &'a ByteStr {
    #[inline]
    fn from(s: &'a str) -> Self {
        ByteStr::from_bytes(s.as_bytes())
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl hash::Hash for ByteStr {
    #[inline]
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl Index<usize> for ByteStr {
    type Output = u8;

    #[inline]
    fn index(&self, idx: usize) -> &u8 {
        &self.0[idx]
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl Index<RangeFull> for ByteStr {
    type Output = ByteStr;

    #[inline]
    fn index(&self, _: RangeFull) -> &ByteStr {
        self
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl Index<Range<usize>> for ByteStr {
    type Output = ByteStr;

    #[inline]
    fn index(&self, r: Range<usize>) -> &ByteStr {
        ByteStr::from_bytes(&self.0[r])
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl Index<RangeInclusive<usize>> for ByteStr {
    type Output = ByteStr;

    #[inline]
    fn index(&self, r: RangeInclusive<usize>) -> &ByteStr {
        ByteStr::from_bytes(&self.0[r])
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl Index<RangeFrom<usize>> for ByteStr {
    type Output = ByteStr;

    #[inline]
    fn index(&self, r: RangeFrom<usize>) -> &ByteStr {
        ByteStr::from_bytes(&self.0[r])
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl Index<RangeTo<usize>> for ByteStr {
    type Output = ByteStr;

    #[inline]
    fn index(&self, r: RangeTo<usize>) -> &ByteStr {
        ByteStr::from_bytes(&self.0[r])
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl Index<RangeToInclusive<usize>> for ByteStr {
    type Output = ByteStr;

    #[inline]
    fn index(&self, r: RangeToInclusive<usize>) -> &ByteStr {
        ByteStr::from_bytes(&self.0[r])
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl IndexMut<usize> for ByteStr {
    #[inline]
    fn index_mut(&mut self, idx: usize) -> &mut u8 {
        &mut self.0[idx]
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl IndexMut<RangeFull> for ByteStr {
    #[inline]
    fn index_mut(&mut self, _: RangeFull) -> &mut ByteStr {
        self
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl IndexMut<Range<usize>> for ByteStr {
    #[inline]
    fn index_mut(&mut self, r: Range<usize>) -> &mut ByteStr {
        ByteStr::from_bytes_mut(&mut self.0[r])
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl IndexMut<RangeInclusive<usize>> for ByteStr {
    #[inline]
    fn index_mut(&mut self, r: RangeInclusive<usize>) -> &mut ByteStr {
        ByteStr::from_bytes_mut(&mut self.0[r])
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl IndexMut<RangeFrom<usize>> for ByteStr {
    #[inline]
    fn index_mut(&mut self, r: RangeFrom<usize>) -> &mut ByteStr {
        ByteStr::from_bytes_mut(&mut self.0[r])
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl IndexMut<RangeTo<usize>> for ByteStr {
    #[inline]
    fn index_mut(&mut self, r: RangeTo<usize>) -> &mut ByteStr {
        ByteStr::from_bytes_mut(&mut self.0[r])
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl IndexMut<RangeToInclusive<usize>> for ByteStr {
    #[inline]
    fn index_mut(&mut self, r: RangeToInclusive<usize>) -> &mut ByteStr {
        ByteStr::from_bytes_mut(&mut self.0[r])
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl Eq for ByteStr {}

#[unstable(feature = "bstr", issue = "134915")]
impl PartialEq<ByteStr> for ByteStr {
    #[inline]
    fn eq(&self, other: &ByteStr) -> bool {
        &self.0 == &other.0
    }
}

#[doc(hidden)]
#[macro_export]
#[unstable(feature = "bstr_internals", issue = "none")]
macro_rules! impl_partial_eq {
    ($lhs:ty, $rhs:ty) => {
        #[allow(unused_lifetimes)]
        impl<'a> PartialEq<$rhs> for $lhs {
            #[inline]
            fn eq(&self, other: &$rhs) -> bool {
                let other: &[u8] = other.as_ref();
                PartialEq::eq(self.as_bytes(), other)
            }
        }

        #[allow(unused_lifetimes)]
        impl<'a> PartialEq<$lhs> for $rhs {
            #[inline]
            fn eq(&self, other: &$lhs) -> bool {
                let this: &[u8] = self.as_ref();
                PartialEq::eq(this, other.as_bytes())
            }
        }
    };
}

#[doc(hidden)]
#[unstable(feature = "bstr_internals", issue = "none")]
pub use impl_partial_eq;

#[doc(hidden)]
#[macro_export]
#[unstable(feature = "bstr_internals", issue = "none")]
macro_rules! impl_partial_eq_ord {
    ($lhs:ty, $rhs:ty) => {
        $crate::bstr::impl_partial_eq!($lhs, $rhs);

        #[allow(unused_lifetimes)]
        #[unstable(feature = "bstr", issue = "134915")]
        impl<'a> PartialOrd<$rhs> for $lhs {
            #[inline]
            fn partial_cmp(&self, other: &$rhs) -> Option<Ordering> {
                let other: &[u8] = other.as_ref();
                PartialOrd::partial_cmp(self.as_bytes(), other)
            }
        }

        #[allow(unused_lifetimes)]
        #[unstable(feature = "bstr", issue = "134915")]
        impl<'a> PartialOrd<$lhs> for $rhs {
            #[inline]
            fn partial_cmp(&self, other: &$lhs) -> Option<Ordering> {
                let this: &[u8] = self.as_ref();
                PartialOrd::partial_cmp(this, other.as_bytes())
            }
        }
    };
}

#[doc(hidden)]
#[unstable(feature = "bstr_internals", issue = "none")]
pub use impl_partial_eq_ord;

#[doc(hidden)]
#[macro_export]
#[unstable(feature = "bstr_internals", issue = "none")]
macro_rules! impl_partial_eq_n {
    ($lhs:ty, $rhs:ty) => {
        #[allow(unused_lifetimes)]
        #[unstable(feature = "bstr", issue = "134915")]
        impl<const N: usize> PartialEq<$rhs> for $lhs {
            #[inline]
            fn eq(&self, other: &$rhs) -> bool {
                let other: &[u8] = other.as_ref();
                PartialEq::eq(self.as_bytes(), other)
            }
        }

        #[allow(unused_lifetimes)]
        #[unstable(feature = "bstr", issue = "134915")]
        impl<const N: usize> PartialEq<$lhs> for $rhs {
            #[inline]
            fn eq(&self, other: &$lhs) -> bool {
                let this: &[u8] = self.as_ref();
                PartialEq::eq(this, other.as_bytes())
            }
        }
    };
}

#[doc(hidden)]
#[unstable(feature = "bstr_internals", issue = "none")]
pub use impl_partial_eq_n;

// PartialOrd with `[u8]` omitted to avoid inference failures
impl_partial_eq!(ByteStr, [u8]);
// PartialOrd with `&[u8]` omitted to avoid inference failures
impl_partial_eq!(ByteStr, &[u8]);
// PartialOrd with `str` omitted to avoid inference failures
impl_partial_eq!(ByteStr, str);
// PartialOrd with `&str` omitted to avoid inference failures
impl_partial_eq!(ByteStr, &str);
// PartialOrd with `[u8; N]` omitted to avoid inference failures
impl_partial_eq_n!(ByteStr, [u8; N]);
// PartialOrd with `[u8; N]` omitted to avoid inference failures
impl_partial_eq_n!(ByteStr, &[u8; N]);

#[unstable(feature = "bstr", issue = "134915")]
impl Ord for ByteStr {
    #[inline]
    fn cmp(&self, other: &ByteStr) -> Ordering {
        Ord::cmp(&self.0, &other.0)
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl PartialOrd for ByteStr {
    #[inline]
    fn partial_cmp(&self, other: &ByteStr) -> Option<Ordering> {
        PartialOrd::partial_cmp(&self.0, &other.0)
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl<'a> TryFrom<&'a ByteStr> for &'a str {
    type Error = crate::str::Utf8Error;

    #[inline]
    fn try_from(s: &'a ByteStr) -> Result<Self, Self::Error> {
        crate::str::from_utf8(&s.0)
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl<'a> TryFrom<&'a mut ByteStr> for &'a mut str {
    type Error = crate::str::Utf8Error;

    #[inline]
    fn try_from(s: &'a mut ByteStr) -> Result<Self, Self::Error> {
        crate::str::from_utf8_mut(&mut s.0)
    }
}
