//! [`OsStringExt`] for windows.

use super::{private, Buf, OsString};
use crate::ffi::wtf8::Wtf8Buf;

/// Windows-specific extensions to [`OsString`].
///
/// This trait is sealed: it cannot be implemented outside the standard library.
/// This is so that future additional methods are not breaking changes.
#[stable(feature = "rust1", since = "1.0.0")]
pub trait OsStringExt: private::Sealed {
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
        From::from(Buf { inner: Wtf8Buf::from_wide(wide) })
    }
}
