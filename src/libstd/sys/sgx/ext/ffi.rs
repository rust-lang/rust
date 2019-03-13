//! SGX-specific extension to the primitives in the `std::ffi` module

#![unstable(feature = "sgx_platform", issue = "56975")]

use crate::ffi::{OsStr, OsString};
use crate::mem;
use crate::sys::os_str::Buf;
use crate::sys_common::{FromInner, IntoInner, AsInner};

/// SGX-specific extensions to [`OsString`].
///
/// [`OsString`]: ../../../../std/ffi/struct.OsString.html
#[unstable(feature = "sgx_platform", issue = "56975")]
pub trait OsStringExt {
    /// Creates an [`OsString`] from a byte vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ffi::OsString;
    /// use std::os::unix::ffi::OsStringExt;
    ///
    /// let bytes = b"foo".to_vec();
    /// let os_string = OsString::from_vec(bytes);
    /// assert_eq!(os_string.to_str(), Some("foo"));
    /// ```
    ///
    /// [`OsString`]: ../../../ffi/struct.OsString.html
    #[unstable(feature = "sgx_platform", issue = "56975")]
    fn from_vec(vec: Vec<u8>) -> Self;

    /// Yields the underlying byte vector of this [`OsString`].
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ffi::OsString;
    /// use std::os::unix::ffi::OsStringExt;
    ///
    /// let mut os_string = OsString::new();
    /// os_string.push("foo");
    /// let bytes = os_string.into_vec();
    /// assert_eq!(bytes, b"foo");
    /// ```
    ///
    /// [`OsString`]: ../../../ffi/struct.OsString.html
    #[unstable(feature = "sgx_platform", issue = "56975")]
    fn into_vec(self) -> Vec<u8>;
}

#[unstable(feature = "sgx_platform", issue = "56975")]
impl OsStringExt for OsString {
    fn from_vec(vec: Vec<u8>) -> OsString {
        FromInner::from_inner(Buf { inner: vec })
    }
    fn into_vec(self) -> Vec<u8> {
        self.into_inner().inner
    }
}

/// SGX-specific extensions to [`OsStr`].
///
/// [`OsStr`]: ../../../../std/ffi/struct.OsStr.html
#[unstable(feature = "sgx_platform", issue = "56975")]
pub trait OsStrExt {
    #[unstable(feature = "sgx_platform", issue = "56975")]
    /// Creates an [`OsStr`] from a byte slice.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ffi::OsStr;
    /// use std::os::unix::ffi::OsStrExt;
    ///
    /// let bytes = b"foo";
    /// let os_str = OsStr::from_bytes(bytes);
    /// assert_eq!(os_str.to_str(), Some("foo"));
    /// ```
    ///
    /// [`OsStr`]: ../../../ffi/struct.OsStr.html
    fn from_bytes(slice: &[u8]) -> &Self;

    /// Gets the underlying byte view of the [`OsStr`] slice.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ffi::OsStr;
    /// use std::os::unix::ffi::OsStrExt;
    ///
    /// let mut os_str = OsStr::new("foo");
    /// let bytes = os_str.as_bytes();
    /// assert_eq!(bytes, b"foo");
    /// ```
    ///
    /// [`OsStr`]: ../../../ffi/struct.OsStr.html
    #[unstable(feature = "sgx_platform", issue = "56975")]
    fn as_bytes(&self) -> &[u8];
}

#[unstable(feature = "sgx_platform", issue = "56975")]
impl OsStrExt for OsStr {
    fn from_bytes(slice: &[u8]) -> &OsStr {
        unsafe { mem::transmute(slice) }
    }
    fn as_bytes(&self) -> &[u8] {
        &self.as_inner().inner
    }
}
