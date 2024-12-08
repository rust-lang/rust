//! SGX-specific extension to the primitives in the `std::ffi` module
//!
//! # Examples
//!
//! ```
//! use std::ffi::OsString;
//! use std::os::fortanix_sgx::ffi::OsStringExt;
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
//! use std::os::fortanix_sgx::ffi::OsStrExt;
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

#![unstable(feature = "sgx_platform", issue = "56975")]

#[path = "../unix/ffi/os_str.rs"]
mod os_str;

#[unstable(feature = "sgx_platform", issue = "56975")]
pub use self::os_str::{OsStrExt, OsStringExt};
