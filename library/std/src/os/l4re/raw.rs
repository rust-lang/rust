//! L4Re-specific raw type definitions.

#![stable(feature = "raw_ext", since = "1.1.0")]
#![deprecated(
    since = "1.8.0",
    note = "these type aliases are no longer supported by \
            the standard library, the `libc` crate on \
            crates.io should be used instead for the correct \
            definitions"
)]
#![allow(deprecated)]

#[stable(feature = "raw_ext", since = "1.1.0")]
pub type dev_t = libc::dev_t;
#[stable(feature = "raw_ext", since = "1.1.0")]
pub type mode_t = libc::mode_t;

#[stable(feature = "pthread_t", since = "1.8.0")]
pub type pthread_t = libc::pthread_t;

#[doc(inline)]
#[stable(feature = "raw_ext", since = "1.1.0")]
pub use libc::{blkcnt_t, blksize_t, ino_t, nlink_t, off_t, stat, time_t};
