//! Platform-specific extensions to `std` for Windows.
//!
//! Provides access to platform-level information for Windows, and exposes
//! Windows-specific idioms that would otherwise be inappropriate as part
//! the core `std` library. These extensions allow developers to use
//! `std` types and idioms with Windows in a way that the normal
//! platform-agnostic idioms would not normally support.
//!
//! # Examples
//!
//! ```no_run
//! use std::fs::File;
//! use std::os::windows::prelude::*;
//!
//! fn main() -> std::io::Result<()> {
//!     let f = File::create("foo.txt")?;
//!     let handle = f.as_raw_handle();
//!
//!     // use handle with native windows bindings
//!
//!     Ok(())
//! }
//! ```

#![stable(feature = "rust1", since = "1.0.0")]
#![doc(cfg(windows))]
#![deny(unsafe_op_in_unsafe_fn)]

pub mod ffi;
pub mod fs;
pub mod io;
pub mod process;
pub mod raw;
pub mod thread;

/// A prelude for conveniently writing platform-specific code.
///
/// Includes all extension traits, and some important type definitions.
#[stable(feature = "rust1", since = "1.0.0")]
pub mod prelude {
    #[doc(no_inline)]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub use super::ffi::{OsStrExt, OsStringExt};
    #[doc(no_inline)]
    #[stable(feature = "file_offset", since = "1.15.0")]
    pub use super::fs::FileExt;
    #[doc(no_inline)]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub use super::fs::{MetadataExt, OpenOptionsExt};
    #[doc(no_inline)]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub use super::io::{
        AsHandle, AsSocket, BorrowedHandle, BorrowedSocket, FromRawHandle, FromRawSocket,
        HandleOrInvalid, IntoRawHandle, IntoRawSocket, OwnedHandle, OwnedSocket,
    };
    #[doc(no_inline)]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub use super::io::{AsRawHandle, AsRawSocket, RawHandle, RawSocket};
}
