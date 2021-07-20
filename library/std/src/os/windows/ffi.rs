//! Windows-specific extensions to the primitives in the `std::ffi` module.
//!
//! # Overview
//!
//! For historical reasons, the Windows API uses a form of potentially
//! ill-formed UTF-16 encoding for strings. Specifically, the 16-bit
//! code units in Windows strings may contain [isolated surrogate code
//! points which are not paired together][ill-formed-utf-16]. The
//! Unicode standard requires that surrogate code points (those in the
//! range U+D800 to U+DFFF) always be *paired*, because in the UTF-16
//! encoding a *surrogate code unit pair* is used to encode a single
//! character. For compatibility with code that does not enforce
//! these pairings, Windows does not enforce them, either.
//!
//! While it is not always possible to convert such a string losslessly into
//! a valid UTF-16 string (or even UTF-8), it is often desirable to be
//! able to round-trip such a string from and to Windows APIs
//! losslessly. For example, some Rust code may be "bridging" some
//! Windows APIs together, just passing `WCHAR` strings among those
//! APIs without ever really looking into the strings.
//!
//! If Rust code *does* need to look into those strings, it can
//! convert them to valid UTF-8, possibly lossily, by substituting
//! invalid sequences with [`U+FFFD REPLACEMENT CHARACTER`][U+FFFD], as is
//! conventionally done in other Rust APIs that deal with string
//! encodings.
//!
//! # `OsStringExt` and `OsStrExt`
//!
//! [`OsString`] is the Rust wrapper for owned strings in the
//! preferred representation of the operating system. On Windows,
//! this struct gets augmented with an implementation of the
//! [`OsStringExt`] trait, which has a [`OsStringExt::from_wide`] method. This
//! lets you create an [`OsString`] from a `&[u16]` slice; presumably
//! you get such a slice out of a `WCHAR` Windows API.
//!
//! Similarly, [`OsStr`] is the Rust wrapper for borrowed strings from
//! preferred representation of the operating system. On Windows, the
//! [`OsStrExt`] trait provides the [`OsStrExt::encode_wide`] method, which
//! outputs an [`EncodeWide`] iterator. You can [`collect`] this
//! iterator, for example, to obtain a `Vec<u16>`; you can later get a
//! pointer to this vector's contents and feed it to Windows APIs.
//!
//! These traits, along with [`OsString`] and [`OsStr`], work in
//! conjunction so that it is possible to **round-trip** strings from
//! Windows and back, with no loss of data, even if the strings are
//! ill-formed UTF-16.
//!
//! [ill-formed-utf-16]: https://simonsapin.github.io/wtf-8/#ill-formed-utf-16
//! [`collect`]: crate::iter::Iterator::collect
//! [U+FFFD]: crate::char::REPLACEMENT_CHARACTER

#![stable(feature = "rust1", since = "1.0.0")]

use crate::cmp::Ordering;
use crate::ffi::{OsStr, OsString};
use crate::io;
use crate::sealed::Sealed;
use crate::sys::os_str::Buf;
use crate::sys_common::wtf8::Wtf8Buf;
use crate::sys_common::{AsInner, FromInner};

#[stable(feature = "rust1", since = "1.0.0")]
pub use crate::sys_common::wtf8::EncodeWide;

/// Windows-specific extensions to [`OsString`].
///
/// This trait is sealed: it cannot be implemented outside the standard library.
/// This is so that future additional methods are not breaking changes.
#[stable(feature = "rust1", since = "1.0.0")]
pub trait OsStringExt: Sealed {
    /// Creates an `OsString` from a potentially ill-formed UTF-16 slice of
    /// 16-bit code units.
    ///
    /// This is lossless: calling [`OsStrExt::encode_wide`] on the resulting string
    /// will always return the original code units.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ffi::OsString;
    /// use std::os::windows::prelude::*;
    ///
    /// // UTF-16 encoding for "Unicode".
    /// let source = [0x0055, 0x006E, 0x0069, 0x0063, 0x006F, 0x0064, 0x0065];
    ///
    /// let string = OsString::from_wide(&source[..]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn from_wide(wide: &[u16]) -> Self;
}

#[stable(feature = "rust1", since = "1.0.0")]
impl OsStringExt for OsString {
    fn from_wide(wide: &[u16]) -> OsString {
        FromInner::from_inner(Buf { inner: Wtf8Buf::from_wide(wide) })
    }
}

/// Windows-specific extensions to [`OsStr`].
///
/// This trait is sealed: it cannot be implemented outside the standard library.
/// This is so that future additional methods are not breaking changes.
#[stable(feature = "rust1", since = "1.0.0")]
pub trait OsStrExt: Sealed {
    /// Re-encodes an `OsStr` as a wide character sequence, i.e., potentially
    /// ill-formed UTF-16.
    ///
    /// This is lossless: calling [`OsStringExt::from_wide`] and then
    /// `encode_wide` on the result will yield the original code units.
    /// Note that the encoding does not add a final null terminator.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ffi::OsString;
    /// use std::os::windows::prelude::*;
    ///
    /// // UTF-16 encoding for "Unicode".
    /// let source = [0x0055, 0x006E, 0x0069, 0x0063, 0x006F, 0x0064, 0x0065];
    ///
    /// let string = OsString::from_wide(&source[..]);
    ///
    /// let result: Vec<u16> = string.encode_wide().collect();
    /// assert_eq!(&source[..], &result[..]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn encode_wide(&self) -> EncodeWide<'_>;

    /// Compares two `OsStr` by using the Windows system implementation.
    /// This performs a case-insensitive comparison of UTF-16 code units using the system case mappings.
    /// The comparison is locale-independent, but exact results may depend on the Windows version,
    /// file system, and system settings.
    ///
    /// This is the correct way to compare various strings on Windows:
    /// environment variable keys, registry keys and resource handle names are all case-insensitive.
    /// Note that this does not include file names or paths; those can be case-sensitive depending on
    /// the system, file system or directory settings.
    ///
    /// Note that this operation requires encoding both strings to UTF-16 and potentially performing system calls.
    /// This operation is thus more computationally expensive than a normal comparison using [`Ord`].
    ///
    /// # Errors
    ///
    /// This function will return an error in the following situations, but is not limited to just these cases:
    ///  - If the string contains any null characters.
    ///
    /// # Examples
    /// ```
    /// #![feature(windows_case_insensitive)]
    ///
    /// use std::ffi::OsString;
    /// use std::os::windows::prelude::*;
    ///
    /// let list = [ OsString::from("A"), OsString::from("Z"), OsString::from("a") ];
    ///
    /// let mut sorted = list.clone();
    /// sorted.sort();
    ///
    /// let mut sorted_with_system_cmp = list.clone();
    /// sorted_with_system_cmp.sort_by(|a, b| a.system_cmp(b).unwrap());
    ///
    /// assert_eq!(sorted, list); // unchanged, since `Z` < `a`
    /// assert_eq!(sorted_with_system_cmp, [ OsString::from("A"), OsString::from("a"), OsString::from("Z") ]);
    /// ```
    #[unstable(feature = "windows_case_insensitive", issue = "86007")]
    fn system_cmp(&self, other: &Self) -> io::Result<Ordering>;

    /// Checks two `OsStr` for equality by using the Windows system implementation.
    /// This performs a case-insensitive comparison of UTF-16 code units using the system case mappings.
    /// The comparison is locale-independent, but exact results may depend on the Windows version,
    /// file system, and system settings.
    ///
    /// This is the correct way to compare various strings on Windows:
    /// environment variable keys, registry keys and resource handle names are all case-insensitive.
    /// Note that this does not include file names or paths; those can be case-sensitive depending on
    /// the system, file system or directory settings.
    ///
    /// Note that this operation requires encoding both strings to UTF-16 and potentially performing system calls.
    /// This operation is thus more computationally expensive than a normal comparison using [`Eq`].
    ///
    /// # Errors
    ///
    /// This function will return an error in the following situations, but is not limited to just these cases:
    ///  - If the string contains any null characters.
    ///
    /// # Examples
    /// ```
    /// #![feature(windows_case_insensitive)]
    ///
    /// use std::ffi::OsString;
    /// use std::os::windows::prelude::*;
    ///
    /// let a = OsString::from("Path");
    /// let b = OsString::from("PATH");
    ///
    /// assert!(a.eq(&b) == false);
    /// assert!(a.system_eq(&b).unwrap() == true);
    #[unstable(feature = "windows_case_insensitive", issue = "86007")]
    fn system_eq(&self, other: &Self) -> io::Result<bool>;
}

#[stable(feature = "rust1", since = "1.0.0")]
impl OsStrExt for OsStr {
    fn encode_wide(&self) -> EncodeWide<'_> {
        self.as_inner().inner.encode_wide()
    }

    fn system_cmp(&self, other: &Self) -> io::Result<Ordering> {
        crate::sys::compare_case_insensitive(self, other)
    }

    fn system_eq(&self, other: &Self) -> io::Result<bool> {
        if self.len() == other.len() {
            Ok(crate::sys::compare_case_insensitive(self, other)? == Ordering::Equal)
        } else {
            // The system implementation performs an "ordinal" check, so directly comparing every
            // code unit in the same position in the two strings. As a consequence, two strings
            // with different lengths can never be equal, even if they contain characters that
            // change length when changing case according to Unicode.
            Ok(false)
        }
    }
}
