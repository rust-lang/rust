// Note: This file is publicly exported as various modules to reduce
// duplication. Currently they are:
// - `std::os::unix::ffi`
// - `std::os::hermit::ffi`
// - `std::os::fortanix_sgx::ffi`
// - `std::os::solid::ffi`
// - `std::os::wasi::ffi`
// - `std::os::xous::ffi`

#![doc = cfg_select! {
    target_family = "unix" => "Unix-specific",
    target_os = "hermit" => "HermitCore-specific",
    all(target_vendor = "fortanix", target_env = "sgx") => "SGX-specific",
    target_os = "solid_asp3" => "SOLID-specific",
    target_os = "wasi" => "WASI-specific",
    target_os = "xous" => "Xous-specific",
}]
//! extensions to primitives in the [`std::ffi`] module.
//!
//! # Examples
//!
//! ```
//! use std::ffi::OsString;
#![doc = cfg_select! {
    target_family = "unix" => "use std::os::unix::ffi::OsStringExt;",
    target_os = "hermit" => "use std::os::hermit::ffi::OsStringExt;",
    all(target_vendor = "fortanix", target_env = "sgx") => "use std::os::fortanix_sgx::ffi::OsStringExt;",
    target_os = "solid_asp3" => "use std::os::solid::ffi::OsStringExt;",
    target_os = "wasi" => "use std::os::wasi::ffi::OsStringExt;",
    target_os = "xous" => "use std::os::xous::ffi::OsStringExt;",
}]
//!
//! let bytes = b"foo".to_vec();
//!
//! // OsStringExt::from_vec
//! let os_string = OsString::from_vec(bytes);
//! assert_eq!(os_string.to_str(), Some("foo"));
//!
//! // OsStringExt::into_vec
//! let bytes = os_string.into_vec();
//! assert_eq!(bytes, b"foo");
//! ```
//!
//! ```
//! use std::ffi::OsStr;
#![doc = cfg_select! {
    target_family = "unix" => "use std::os::unix::ffi::OsStrExt;",
    target_os = "hermit" => "use std::os::hermit::ffi::OsStrExt;",
    all(target_vendor = "fortanix", target_env = "sgx") => "use std::os::fortanix_sgx::ffi::OsStrExt;",
    target_os = "solid_asp3" => "use std::os::solid::ffi::OsStrExt;",
    target_os = "wasi" => "use std::os::wasi::ffi::OsStrExt;",
    target_os = "xous" => "use std::os::xous::ffi::OsStrExt;",
}]
//!
//! let bytes = b"foo";
//!
//! // OsStrExt::from_bytes
//! let os_str = OsStr::from_bytes(bytes);
//! assert_eq!(os_str.to_str(), Some("foo"));
//!
//! // OsStrExt::as_bytes
//! let bytes = os_str.as_bytes();
//! assert_eq!(bytes, b"foo");
//! ```
//!
//! [`std::ffi`]: crate::ffi

#![cfg_attr(
    not(all(target_vendor = "fortanix", target_env = "sgx")),
    stable(feature = "rust1", since = "1.0.0")
)]
#![cfg_attr(
    all(target_vendor = "fortanix", target_env = "sgx"),
    unstable(feature = "sgx_platform", issue = "56975")
)]

use crate::ffi::{OsStr, OsString};
use crate::mem;
use crate::sealed::Sealed;
use crate::sys::os_str::Buf;
use crate::sys_common::{AsInner, FromInner, IntoInner};

/// Platform-specific extensions to [`OsString`].
///
/// This trait is sealed: it cannot be implemented outside the standard library.
/// This is so that future additional methods are not breaking changes.
#[stable(feature = "rust1", since = "1.0.0")]
pub trait OsStringExt: Sealed {
    /// Creates an [`OsString`] from a byte vector.
    ///
    /// See the module documentation for an example.
    #[stable(feature = "rust1", since = "1.0.0")]
    fn from_vec(vec: Vec<u8>) -> Self;

    /// Yields the underlying byte vector of this [`OsString`].
    ///
    /// See the module documentation for an example.
    #[stable(feature = "rust1", since = "1.0.0")]
    fn into_vec(self) -> Vec<u8>;
}

#[stable(feature = "rust1", since = "1.0.0")]
impl OsStringExt for OsString {
    #[inline]
    fn from_vec(vec: Vec<u8>) -> OsString {
        FromInner::from_inner(Buf { inner: vec })
    }
    #[inline]
    fn into_vec(self) -> Vec<u8> {
        self.into_inner().inner
    }
}

/// Platform-specific extensions to [`OsStr`].
///
/// This trait is sealed: it cannot be implemented outside the standard library.
/// This is so that future additional methods are not breaking changes.
#[stable(feature = "rust1", since = "1.0.0")]
pub trait OsStrExt: Sealed {
    #[stable(feature = "rust1", since = "1.0.0")]
    /// Creates an [`OsStr`] from a byte slice.
    ///
    /// See the module documentation for an example.
    fn from_bytes(slice: &[u8]) -> &Self;

    /// Gets the underlying byte view of the [`OsStr`] slice.
    ///
    /// See the module documentation for an example.
    #[stable(feature = "rust1", since = "1.0.0")]
    fn as_bytes(&self) -> &[u8];
}

#[stable(feature = "rust1", since = "1.0.0")]
impl OsStrExt for OsStr {
    #[inline]
    fn from_bytes(slice: &[u8]) -> &OsStr {
        unsafe { mem::transmute(slice) }
    }
    #[inline]
    fn as_bytes(&self) -> &[u8] {
        &self.as_inner().inner
    }
}
