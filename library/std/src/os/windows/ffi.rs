//! Windows-specific extensions to primitives in the [`std::ffi`] module.
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
//! [`OsStringExt`] trait, which has an [`OsStringExt::from_wide`] method. This
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
//! [`std::ffi`]: crate::ffi

#![stable(feature = "rust1", since = "1.0.0")]

use crate::ffi::{OsStr, OsString};
#[cfg(not(target_os = "uefi"))]
use crate::path::NativePath;
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
}

#[stable(feature = "rust1", since = "1.0.0")]
impl OsStrExt for OsStr {
    #[inline]
    fn encode_wide(&self) -> EncodeWide<'_> {
        self.as_inner().inner.encode_wide()
    }
}

/// On Windows `NativePath` wraps a wide string for use in filesystem function calls.
/// These strings are `&[u16]` slices.
///
/// # Wide strings
///
/// Filesystem paths in Windows are encoded as UTF-16 strings.
/// However, because the kernel does not verify validity this may contain invalid UTF-16.
/// Therefore we use the term "wide string" for potentially invalid UTF-16.
#[cfg(windows)]
#[unstable(feature = "fs_native_path", issue = "108979")]
pub trait NativePathExt: Sealed {
    /// Wrap a Windows wide string as a `NativePath`.
    /// The `wide` string must be null terminated and must not otherwise contain nulls.
    fn from_wide(wide: &[u16]) -> &NativePath;
    /// Wrap a Windows wide string as a `NativePath` without checking for null termination or internal nulls.
    unsafe fn from_wide_unchecked(wide: &[u16]) -> &NativePath;
    /// Unwrap the `NativePath` to return the inner wide string.
    fn into_wide(&self) -> &[u16];
}
#[cfg(windows)]
#[unstable(feature = "fs_native_path", issue = "108979")]
impl NativePathExt for NativePath {
    fn from_wide(wide: &[u16]) -> &NativePath {
        assert_eq!(crate::sys::unrolled_find_u16s(0, wide), Some(wide.len().saturating_sub(1)));
        unsafe { Self::from_wide_unchecked(wide) }
    }
    unsafe fn from_wide_unchecked(wide: &[u16]) -> &NativePath {
        &*(wide as *const [u16] as *const Self)
    }
    fn into_wide(&self) -> &[u16] {
        unsafe { &*(self as *const Self as *const [u16]) }
    }
}
