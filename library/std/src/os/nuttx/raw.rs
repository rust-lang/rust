//! NuttX raw type definitions

#![stable(feature = "raw_ext", since = "1.1.0")]
#![deprecated(
    since = "1.8.0",
    note = "these type aliases are no longer supported by \
            the standard library, the `libc` crate on \
            crates.io should be used instead for the correct \
            definitions"
)]
#![allow(deprecated)]

#[stable(feature = "pthread_t", since = "1.8.0")]
pub type pthread_t = libc::pthread_t;

#[stable(feature = "raw_ext", since = "1.1.0")]
pub type blkcnt_t = libc::blkcnt_t;

#[stable(feature = "raw_ext", since = "1.1.0")]
pub type blksize_t = libc::blksize_t;
#[stable(feature = "raw_ext", since = "1.1.0")]
pub type dev_t = libc::dev_t;
#[stable(feature = "raw_ext", since = "1.1.0")]
pub type ino_t = libc::ino_t;
#[stable(feature = "raw_ext", since = "1.1.0")]
pub type mode_t = libc::mode_t;
#[stable(feature = "raw_ext", since = "1.1.0")]
pub type nlink_t = libc::nlink_t;
#[stable(feature = "raw_ext", since = "1.1.0")]
pub type off_t = libc::off_t;

#[stable(feature = "raw_ext", since = "1.1.0")]
pub type time_t = libc::time_t;
