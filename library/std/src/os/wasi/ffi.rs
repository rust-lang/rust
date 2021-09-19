//! WASI-specific extensions to primitives in the [`std::ffi`] module
//!
//! [`std::ffi`]: crate::ffi

#![stable(feature = "rust1", since = "1.0.0")]

#[path = "../unix/ffi/os_str.rs"]
mod os_str;

#[stable(feature = "rust1", since = "1.0.0")]
pub use self::os_str::{OsStrExt, OsStringExt};
