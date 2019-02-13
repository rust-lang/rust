//! Platform-specific extensions to `std` for Unix platforms.
//!
//! Provides access to platform-level information on Unix platforms, and
//! exposes Unix-specific functions that would otherwise be inappropriate as
//! part of the core `std` library.
//!
//! It exposes more ways to deal with platform-specific strings (`OsStr`,
//! `OsString`), allows to set permissions more granularly, extract low-level
//! file descriptors from files and sockets, and has platform-specific helpers
//! for spawning processes.
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
#![doc(cfg(unix))]
#![allow(missing_docs)]

pub mod io;
pub mod ffi;
pub mod fs;
pub mod process;
pub mod raw;
pub mod thread;
pub mod net;

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
    pub use super::fs::{PermissionsExt, OpenOptionsExt, MetadataExt, FileTypeExt};
    #[doc(no_inline)] #[stable(feature = "rust1", since = "1.0.0")]
    pub use super::fs::DirEntryExt;
    #[doc(no_inline)] #[stable(feature = "file_offset", since = "1.15.0")]
    pub use super::fs::FileExt;
    #[doc(no_inline)] #[stable(feature = "rust1", since = "1.0.0")]
    pub use super::thread::JoinHandleExt;
    #[doc(no_inline)] #[stable(feature = "rust1", since = "1.0.0")]
    pub use super::process::{CommandExt, ExitStatusExt};
}
