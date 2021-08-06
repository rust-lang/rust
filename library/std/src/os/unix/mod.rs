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
//! fn main() -> std::io::Result<()> {
//!     let f = File::create("foo.txt")?;
//!     let fd = f.as_raw_fd();
//!
//!     // use fd with native unix bindings
//!
//!     Ok(())
//! }
//! ```

#![stable(feature = "rust1", since = "1.0.0")]
#![doc(cfg(unix))]

// Use linux as the default platform when documenting on other platforms like Windows
#[cfg(doc)]
use crate::os::linux as platform;

#[cfg(not(doc))]
mod platform {
    #[cfg(target_os = "android")]
    pub use crate::os::android::*;
    #[cfg(target_os = "dragonfly")]
    pub use crate::os::dragonfly::*;
    #[cfg(target_os = "emscripten")]
    pub use crate::os::emscripten::*;
    #[cfg(target_os = "freebsd")]
    pub use crate::os::freebsd::*;
    #[cfg(target_os = "fuchsia")]
    pub use crate::os::fuchsia::*;
    #[cfg(target_os = "haiku")]
    pub use crate::os::haiku::*;
    #[cfg(target_os = "illumos")]
    pub use crate::os::illumos::*;
    #[cfg(target_os = "ios")]
    pub use crate::os::ios::*;
    #[cfg(any(target_os = "linux", target_os = "l4re"))]
    pub use crate::os::linux::*;
    #[cfg(target_os = "macos")]
    pub use crate::os::macos::*;
    #[cfg(target_os = "netbsd")]
    pub use crate::os::netbsd::*;
    #[cfg(target_os = "openbsd")]
    pub use crate::os::openbsd::*;
    #[cfg(target_os = "redox")]
    pub use crate::os::redox::*;
    #[cfg(target_os = "solaris")]
    pub use crate::os::solaris::*;
    #[cfg(target_os = "vxworks")]
    pub use crate::os::vxworks::*;
}

pub mod ffi;
pub mod fs;
pub mod io;
pub mod net;
pub mod process;
pub mod raw;
pub mod thread;

#[unstable(feature = "peer_credentials_unix_socket", issue = "42839", reason = "unstable")]
#[cfg(any(
    target_os = "android",
    target_os = "linux",
    target_os = "dragonfly",
    target_os = "freebsd",
    target_os = "ios",
    target_os = "macos",
    target_os = "netbsd",
    target_os = "openbsd"
))]
pub mod ucred;

/// A prelude for conveniently writing platform-specific code.
///
/// Includes all extension traits, and some important type definitions.
#[stable(feature = "rust1", since = "1.0.0")]
pub mod prelude {
    #[doc(no_inline)]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub use super::ffi::{OsStrExt, OsStringExt};
    #[doc(no_inline)]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub use super::fs::DirEntryExt;
    #[doc(no_inline)]
    #[stable(feature = "file_offset", since = "1.15.0")]
    pub use super::fs::FileExt;
    #[doc(no_inline)]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub use super::fs::{FileTypeExt, MetadataExt, OpenOptionsExt, PermissionsExt};
    #[doc(no_inline)]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub use super::io::{AsRawFd, FromRawFd, IntoRawFd, RawFd};
    #[doc(no_inline)]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub use super::process::{CommandExt, ExitStatusExt};
    #[doc(no_inline)]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub use super::thread::JoinHandleExt;
}
