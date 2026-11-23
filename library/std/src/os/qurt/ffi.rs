//! QuRT-specific extension to the primitives in the [`std::ffi`] module.
//!
//! [`std::ffi`]: crate::ffi

#![stable(feature = "raw_ext", since = "1.1.0")]

#[path = "../unix/ffi/os_str.rs"]
mod os_str;

#[stable(feature = "raw_ext", since = "1.1.0")]
pub use self::os_str::{OsStrExt, OsStringExt};
