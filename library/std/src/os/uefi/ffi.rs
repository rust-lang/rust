use crate::ffi::{OsStr, OsString};
use crate::sealed::Sealed;
use crate::sys_common::ucs2;

#[unstable(feature = "uefi_std", issue = "100499")]
pub use ucs2::EncodeUcs2;

#[unstable(feature = "uefi_std", issue = "100499")]
pub trait OsStrExt: Sealed {
    /// Re-encodes an `OsStr` as a wide character sequence, i.e., UCS-2
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ffi::OsString;
    /// use std::os::uefi::ffi::*;
    ///
    /// // UTF-2 encoding for "Unicode".
    /// let source = [0x0055, 0x006E, 0x0069, 0x0063, 0x006F, 0x0064, 0x0065];
    ///
    /// let string = OsString::from_wide(&source[..]);
    ///
    /// let result: Vec<u16> = string.encode_wide().collect();
    /// assert_eq!(&source[..], &result[..]);
    /// ```
    fn encode_wide<'a>(&'a self) -> EncodeUcs2<'a>;
}

impl OsStrExt for OsStr {
    fn encode_wide<'a>(&'a self) -> EncodeUcs2<'a> {
        // SAFETY: Calling unwrap on `self.to_str` is safe since the underlying OsStr is UTF-8
        // encoded
        ucs2::EncodeUcs2::from_str(self.to_str().unwrap())
    }
}

#[unstable(feature = "uefi_std", issue = "100499")]
pub trait OsStringExt: Sealed
where
    Self: Sized,
{
    /// Creates an `OsString` from a UCS-2 slice of 16-bit code units.
    /// # Examples
    ///
    /// ```
    /// use std::ffi::OsString;
    /// use std::os::uefi::ffi::*;
    ///
    /// // UTF-16 encoding for "Unicode".
    /// let source = [0x0055, 0x006E, 0x0069, 0x0063, 0x006F, 0x0064, 0x0065];
    ///
    /// let string = OsString::from_wide(&source[..]);
    /// ```
    fn from_wide(ucs: &[u16]) -> Self;
}

impl OsStringExt for OsString {
    fn from_wide(ucs: &[u16]) -> Self {
        // Min capacity(in case of all ASCII) is `ucs.len()`
        let mut buf = String::with_capacity(ucs.len());

        for i in ucs {
            let c = match ucs2::Ucs2Char::from_u16(*i) {
                None => char::REPLACEMENT_CHARACTER,
                Some(x) => char::from(x),
            };
            buf.push(c);
        }

        Self::from(buf)
    }
}
