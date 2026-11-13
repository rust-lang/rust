//! Platform-specific extensions to `std` for the Web platform.

#![unstable(feature = "webp2", issue = "none")]

#[path = "../unix/ffi/os_str.rs"]
mod os_str;

pub mod ffi {
    pub use super::os_str::{OsStrExt, OsStringExt};
}

pub mod io {
    pub use crate::os::fd::*;
}

/// A prelude for conveniently writing platform-specific code.
pub mod prelude {
    #[doc(no_inline)]
    pub use super::ffi::{OsStrExt, OsStringExt};
    #[doc(no_inline)]
    pub use super::io::{AsFd, AsRawFd, BorrowedFd, FromRawFd, IntoRawFd, OwnedFd, RawFd};
}
