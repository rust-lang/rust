//! The `ByteStr` type and trait implementations.

mod traits;

#[unstable(feature = "bstr_internals", issue = "none")]
pub use traits::{impl_partial_eq, impl_partial_eq_n, impl_partial_eq_ord};

use crate::borrow::{Borrow, BorrowMut};
use crate::marker::Destruct;
use crate::{fmt, str};

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
/// `str`, with invalid UTF-8 presented as the Unicode replacement character (�).
#[unstable(feature = "bstr", issue = "134915")]
#[repr(transparent)]
#[doc(alias = "BStr")]
#[rustc_has_incoherent_inherent_impls]
pub struct ByteStr(pub(crate) [u8]);

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
    /// #![feature(bstr)]
    /// use std::bstr::ByteStr;
    ///
    /// let a = ByteStr::new(b"abc");
    /// let b = ByteStr::new(&b"abc"[..]);
    /// let c = ByteStr::new("abc");
    ///
    /// assert_eq!(a, b);
    /// assert_eq!(a, c);
    /// ```
    #[inline]
    #[unstable(feature = "bstr", issue = "134915")]
    #[rustc_const_unstable(feature = "const_convert", issue = "143773")]
    pub const fn new<B: ?Sized + [const] AsRef<[u8]>>(bytes: &B) -> &Self {
        ByteStr::from_bytes(bytes.as_ref())
    }

    /// Creates a mutable `ByteStr` slice from anything that can be converted to a mutable byte slice.
    ///
    /// This is a zero-cost conversion.
    ///
    /// # Example
    ///
    /// Unlike `str`, the raw bytes of a `ByteStr` can be safely mutated at any time, since the
    /// result is not guaranteed to be valid UTF-8.
    ///
    /// ```
    /// #![feature(bstr)]
    /// use std::bstr::ByteStr;
    ///
    /// let mut buf = "😀".to_string().into_bytes().into_byte_string();
    /// assert_eq!(format!("{buf}"), "😀");
    ///
    /// let s = ByteStr::new_mut(&mut buf);
    /// s.as_bytes_mut().reverse();
    /// s[1] = b':';
    /// s[2] = b'(';
    ///
    /// assert_eq!(format!("{buf}"), "�:(�");
    /// ```
    #[inline]
    #[unstable(feature = "bstr", issue = "134915")]
    #[rustc_const_unstable(feature = "const_convert", issue = "143773")]
    pub const fn new_mut<B: ?Sized + [const] AsMut<[u8]>>(bytes: &mut B) -> &mut Self {
        ByteStr::from_bytes_mut(bytes.as_mut())
    }

    #[inline]
    pub(crate) const fn from_bytes(slice: &[u8]) -> &Self {
        // SAFETY: `ByteStr` is a transparent wrapper around `[u8]`, so we can turn a reference to
        // the wrapped type into a reference to the wrapper type.
        unsafe { &*(slice as *const [u8] as *const Self) }
    }

    #[inline]
    pub(crate) const fn from_bytes_mut(slice: &mut [u8]) -> &mut Self {
        // SAFETY: `ByteStr` is a transparent wrapper around `[u8]`, so we can turn a reference to
        // the wrapped type into a reference to the wrapper type.
        unsafe { &mut *(slice as *mut [u8] as *mut Self) }
    }

    /// Converts a `ByteStr` slice to a byte slice. To convert the byte slice back into a byte string
    /// slice, use [`ByteStr::new`].
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(bstr)]
    /// use std::bstr::ByteStr;
    ///
    /// let s = ByteStr::new("bors");
    /// let bytes = s.as_bytes();
    /// assert_eq!(b"bors", bytes);
    /// ```
    #[unstable(feature = "bstr", issue = "134915")]
    #[inline]
    #[rustc_const_unstable(feature = "bstr", issue = "134915")]
    pub const fn as_bytes(&self) -> &[u8] {
        &self.0
    }

    /// Converts a mutable `ByteStr` slice to a mutable byte slice.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// #![feature(bstr)]
    /// use std::bstr::ByteStr;
    ///
    /// let mut s = ByteStr::new("Hello").to_byte_string();
    /// let bytes = s.as_bytes_mut();
    /// assert_eq!(b"Hello", bytes);
    /// ```
    ///
    /// Mutability:
    ///
    /// ```
    /// #![feature(bstr)]
    /// use std::bstr::ByteStr;
    ///
    /// let mut s = ByteStr::new("🗻∈🌏").to_byte_string();
    /// let bytes = s.as_bytes_mut();
    ///
    /// bytes[0] = 0xF0;
    /// bytes[1] = 0x9F;
    /// bytes[2] = 0x8D;
    /// bytes[3] = 0x94;
    ///
    /// assert_eq!("🍔∈🌏", s);
    /// ```
    #[unstable(feature = "bstr", issue = "134915")]
    #[inline]
    #[rustc_const_unstable(feature = "bstr", issue = "134915")]
    pub const fn as_bytes_mut(&mut self) -> &mut [u8] {
        &mut self.0
    }

    /// Yields a <code>&[prim@str]</code> slice if the `ByteStr` is valid unicode.
    ///
    /// This conversion may entail a check for UTF-8 validity.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(bstr)]
    /// use std::bstr::ByteStr;
    ///
    /// let bstr = ByteStr::new("foo");
    /// assert_eq!(bstr.to_str(), Some("foo"));
    /// ```
    #[unstable(feature = "bstr", issue = "134915")]
    #[inline]
    #[rustc_const_unstable(feature = "bstr", issue = "134915")]
    pub const fn to_str(&self) -> Option<&str> {
        str::from_utf8(&self.0).ok()
    }

    /// Checks whether the `ByteStr` is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(bstr)]
    /// use std::bstr::ByteStr;
    ///
    /// let byte_str = ByteStr::new("");
    /// assert!(byte_str.is_empty());
    ///
    /// let byte_str = ByteStr::new("foo");
    /// assert!(!byte_str.is_empty());
    /// ```
    #[unstable(feature = "bstr", issue = "134915")]
    #[inline]
    #[rustc_const_unstable(feature = "bstr", issue = "134915")]
    pub const fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Returns the length of this `ByteStr` in bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(bstr)]
    /// use std::bstr::ByteStr;
    ///
    /// let byte_str = ByteStr::new("");
    /// assert_eq!(byte_str.len(), 0);
    ///
    /// let byte_str = ByteStr::new("foo");
    /// assert_eq!(byte_str.len(), 3);
    /// ```
    #[unstable(feature = "bstr", issue = "134915")]
    #[inline]
    #[rustc_const_unstable(feature = "bstr", issue = "134915")]
    pub const fn len(&self) -> usize {
        self.0.len()
    }

    /// Checks if all bytes in this byte string are within the ASCII range.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(bstr)]
    /// use std::bstr::ByteStr;
    ///
    /// let ascii = ByteStr::new("hello!\n");
    /// let non_ascii = ByteStr::new("Grüße, Jürgen ❤");
    ///
    /// assert!(ascii.is_ascii());
    /// assert!(!non_ascii.is_ascii());
    /// ```
    #[unstable(feature = "bstr", issue = "134915")]
    #[inline]
    #[rustc_const_unstable(feature = "bstr", issue = "134915")]
    pub const fn is_ascii(&self) -> bool {
        self.0.is_ascii()
    }

    /// Converts this byte string to its ASCII lower case equivalent in-place.
    ///
    /// ASCII letters ‘A’ to ‘Z’ are mapped to ‘a’ to ‘z’, but non-ASCII letters are unchanged.
    /// To return a new lowercased value without modifying the existing one, use
    /// `ByteStr::to_ascii_lowercase`.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(bstr)]
    /// use std::bstr::ByteStr;
    ///
    /// let mut s = ByteStr::new("GRÜßE, JÜRGEN ❤").to_byte_string();
    ///
    /// s.make_ascii_lowercase();
    ///
    /// assert_eq!("grÜße, jÜrgen ❤", s);
    /// ```
    #[unstable(feature = "bstr", issue = "134915")]
    #[inline]
    #[rustc_const_unstable(feature = "bstr", issue = "134915")]
    pub const fn make_ascii_lowercase(&mut self) {
        self.0.make_ascii_lowercase()
    }

    /// Converts this byte string to its ASCII upper case equivalent in-place.
    ///
    /// ASCII letters ‘A’ to ‘Z’ are mapped to ‘a’ to ‘z’, but non-ASCII letters are unchanged.
    /// To return a new uppercased value without modifying the existing one, use
    /// `ByteStr::to_ascii_uppercase`.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(bstr)]
    /// use std::bstr::ByteStr;
    ///
    /// let mut s = ByteStr::new("Grüße, Jürgen ❤").to_byte_string();
    ///
    /// s.make_ascii_uppercase();
    ///
    /// assert_eq!("GRüßE, JüRGEN ❤", s);
    /// ```
    #[unstable(feature = "bstr", issue = "134915")]
    #[inline]
    #[rustc_const_unstable(feature = "bstr", issue = "134915")]
    pub const fn make_ascii_uppercase(&mut self) {
        self.0.make_ascii_uppercase()
    }

    /// Checks if two byte strings are an ASCII case-insensitive match.
    ///
    /// Same as `to_ascii_lowercase(a) == to_ascii_lowercase(b)`, but without allocating and copying
    /// temporaries.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(bstr)]
    /// use std::bstr::ByteStr;
    ///
    /// assert!(ByteStr::new("Ferris").eq_ignore_ascii_case("FERRIS"));
    /// assert!(ByteStr::new("Ferrös").eq_ignore_ascii_case("FERRöS"));
    /// assert!(!ByteStr::new("Ferrös").eq_ignore_ascii_case("FERRÖS"));
    /// ```
    #[unstable(feature = "bstr", issue = "134915")]
    #[inline]
    #[rustc_const_unstable(feature = "bstr", issue = "134915")]
    pub const fn eq_ignore_ascii_case<S: [const] AsRef<ByteStr> + [const] Destruct>(
        &self,
        other: S,
    ) -> bool {
        let other: &ByteStr = other.as_ref();
        self.0.eq_ignore_ascii_case(&other.0)
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl fmt::Debug for ByteStr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "\"")?;
        for chunk in self.0.utf8_chunks() {
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
            for chunk in this.0.utf8_chunks() {
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
            .0
            .utf8_chunks()
            .map(|chunk| {
                chunk.valid().chars().count() + if chunk.invalid().is_empty() { 0 } else { 1 }
            })
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
#[rustc_const_unstable(feature = "const_convert", issue = "143773")]
impl const AsRef<[u8]> for ByteStr {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

#[unstable(feature = "bstr", issue = "134915")]
#[rustc_const_unstable(feature = "const_convert", issue = "143773")]
impl const AsRef<ByteStr> for ByteStr {
    #[inline]
    fn as_ref(&self) -> &ByteStr {
        self
    }
}

// `impl AsRef<ByteStr> for [u8]` omitted to avoid widespread inference failures

#[unstable(feature = "bstr", issue = "134915")]
#[rustc_const_unstable(feature = "const_convert", issue = "143773")]
impl const AsRef<ByteStr> for str {
    #[inline]
    fn as_ref(&self) -> &ByteStr {
        ByteStr::new(self)
    }
}

#[unstable(feature = "bstr", issue = "134915")]
#[rustc_const_unstable(feature = "const_convert", issue = "143773")]
impl const AsMut<[u8]> for ByteStr {
    #[inline]
    fn as_mut(&mut self) -> &mut [u8] {
        &mut self.0
    }
}

// `impl AsMut<ByteStr> for [u8]` omitted to avoid widespread inference failures

// `impl Borrow<ByteStr> for [u8]` omitted to avoid widespread inference failures

// `impl Borrow<ByteStr> for str` omitted to avoid widespread inference failures

#[unstable(feature = "bstr", issue = "134915")]
#[rustc_const_unstable(feature = "const_convert", issue = "143773")]
impl const Borrow<[u8]> for ByteStr {
    #[inline]
    fn borrow(&self) -> &[u8] {
        &self.0
    }
}

// `impl BorrowMut<ByteStr> for [u8]` omitted to avoid widespread inference failures

#[unstable(feature = "bstr", issue = "134915")]
#[rustc_const_unstable(feature = "const_convert", issue = "143773")]
impl const BorrowMut<[u8]> for ByteStr {
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

// Omitted due to inference failures
//
// #[unstable(feature = "bstr", issue = "134915")]
// impl<'a, const N: usize> From<&'a [u8; N]> for &'a ByteStr {
//     #[inline]
//     fn from(s: &'a [u8; N]) -> Self {
//         ByteStr::from_bytes(s)
//     }
// }
//
// #[unstable(feature = "bstr", issue = "134915")]
// impl<'a> From<&'a [u8]> for &'a ByteStr {
//     #[inline]
//     fn from(s: &'a [u8]) -> Self {
//         ByteStr::from_bytes(s)
//     }
// }

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

// Omitted due to inference failures
//
// #[unstable(feature = "bstr", issue = "134915")]
// impl<'a> From<&'a str> for &'a ByteStr {
//     #[inline]
//     fn from(s: &'a str) -> Self {
//         ByteStr::from_bytes(s.as_bytes())
//     }
// }

#[unstable(feature = "bstr", issue = "134915")]
#[rustc_const_unstable(feature = "const_convert", issue = "143773")]
impl<'a> const TryFrom<&'a ByteStr> for &'a str {
    type Error = str::Utf8Error;

    #[inline]
    fn try_from(s: &'a ByteStr) -> Result<Self, Self::Error> {
        str::from_utf8(&s.0)
    }
}

#[unstable(feature = "bstr", issue = "134915")]
#[rustc_const_unstable(feature = "const_convert", issue = "143773")]
impl<'a> const TryFrom<&'a mut ByteStr> for &'a mut str {
    type Error = str::Utf8Error;

    #[inline]
    fn try_from(s: &'a mut ByteStr) -> Result<Self, Self::Error> {
        str::from_utf8_mut(&mut s.0)
    }
}
