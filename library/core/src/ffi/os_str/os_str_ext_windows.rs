//! [`OsStrExt`] for windows

use super::{private, OsStr};
#[stable(feature = "rust1", since = "1.0.0")]
pub use crate::ffi::wtf8::EncodeWide;

/// Windows-specific extensions to [`OsStr`].
///
/// This trait is sealed: it cannot be implemented outside the standard library.
/// This is so that future additional methods are not breaking changes.
#[stable(feature = "rust1", since = "1.0.0")]
pub trait OsStrExt: private::Sealed {
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
        self.inner.inner.encode_wide()
    }
}
