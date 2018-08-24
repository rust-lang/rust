//! Experimental extensions to `std` for Unix platforms.
//!
//! For now, this module is limited to extracting file descriptors,
//! but its functionality will grow over time.
//!
//! # Examples
//!
//! ```no_run
//! use std::fs::File;
//! use std::os::unix::prelude::*;
//!
//! fn main() {
//!     let f = File::create("foo.txt").unwrap();
//!     let fd = f.as_raw_fd();
//!
//!     // use fd with native unix bindings
//! }
//! ```

#![stable(feature = "rust1", since = "1.0.0")]
#![doc(cfg(target_os = "redox"))]

pub mod ffi;
pub mod fs;
pub mod io;
pub mod net;
pub mod process;
pub mod thread;

/// A prelude for conveniently writing platform-specific code.
///
/// Includes all extension traits, and some important type definitions.
#[stable(feature = "rust1", since = "1.0.0")]
pub mod prelude {
    #[doc(no_inline)] #[stable(feature = "rust1", since = "1.0.0")]
    pub use super::io::{RawFd, AsRawFd, FromRawFd, IntoRawFd};
    #[doc(no_inline)] #[stable(feature = "rust1", since = "1.0.0")]
    pub use super::ffi::{OsStrExt, OsStringExt};
    #[doc(no_inline)] #[stable(feature = "rust1", since = "1.0.0")]
    pub use super::fs::{FileTypeExt, PermissionsExt, OpenOptionsExt, MetadataExt};
    #[doc(no_inline)] #[stable(feature = "rust1", since = "1.0.0")]
    pub use super::thread::JoinHandleExt;
    #[doc(no_inline)] #[stable(feature = "rust1", since = "1.0.0")]
    pub use super::process::{CommandExt, ExitStatusExt};
}
